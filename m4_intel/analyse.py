"""M4 entry point: documents -> verified, tagged signals -> coverage-aware verdict.

Implements the shared contract ``analyse(vendor, docs, reports) -> (signals, verdict)``.
The run is deterministic and makes no network or LLM calls. Specificity comes from the
saved classifier when a usable local artifact exists and otherwise from the rule score.
Quote verification proves only that an excerpt exists verbatim in a document; it does not
prove the vendor's claim is true.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from m4_intel.adjudicate import LLM_TECHNICAL_SPECIFICITY
from m4_intel.candidates import extract_document_candidates
from m4_intel.specificity import DEFAULT_WEIGHTS_PATH, score_signal, score_specificity_rule
from m4_intel.train_specificity import DEFAULT_MODEL_DIR, score_specificity_with_method
from m4_intel.verdict import generate_verdict
from m4_intel.verify import verify_candidate
from radar.contracts.models import Signal, VendorVerdict

EXAMPLE_VENDOR_ID = "V-000"
DEFAULT_PLANNED_SOURCES = ("TRUST", "DOCS", "WEB")
DISAGREEMENT_THRESHOLD = 0.4          # plan 4.5: |classifier - rule| above this is ambiguous
AMBIGUOUS_BAND = (0.3, 0.6)           # plan M4: specificity inside this band is ambiguous
_UNPLANNED_STATUSES = {"NOT_APPLICABLE", "SKIPPED_BY_PLAN"}


@dataclass
class AnalysisResult:
    signals: list[Signal]
    verdict: VendorVerdict
    # signal_id -> reasons it should go to an adjudicator/analyst review. Never auto-resolved.
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    llm_proposed: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_documents: dict[str, list[str]] = field(default_factory=dict)


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _vendor_id(vendor: object) -> str:
    candidate = vendor if isinstance(vendor, str) else (_get(vendor, "vendor_id") or _get(vendor, "Vendor ID"))
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("vendor must provide a non-empty vendor_id")
    if candidate.strip() == EXAMPLE_VENDOR_ID:
        raise ValueError("V-000 is the worked example and must never be processed")
    return candidate.strip()


def _as_mapping(document: object) -> dict[str, Any]:
    if isinstance(document, Mapping):
        return dict(document)
    if is_dataclass(document) and not isinstance(document, type):
        return asdict(document)
    raise TypeError("documents must be Document dataclasses or mappings")


def _planned_sources(planned: Sequence[object] | None, docs: Sequence[Mapping[str, Any]],
                     reports: Sequence[object]) -> list[object]:
    """Planned families: explicit plan, else reports not marked N/A or skipped, else none (coverage 0.0)."""
    if planned is not None:
        return list(planned)
    from_reports: list[object] = []
    for report in reports:
        source_type, status = _get(report, "source_type"), _get(report, "status")
        if isinstance(source_type, str) and str(status).upper() not in _UNPLANNED_STATUSES \
                and source_type not in from_reports:
            from_reports.append(source_type)
    if from_reports:
        return from_reports
    # No plan and no reports: the planned families are unknown, so coverage is 0.0 and the verdict can
    # never be "No". Deriving the plan from the documents themselves would make coverage 1.0 by construction.
    return []


def _manual_class(document: Mapping[str, Any]) -> str | None:
    if document.get("source_type") != "MANUAL":
        return None
    meta = document.get("meta")
    value = meta.get("source_class") if isinstance(meta, Mapping) else None
    # Unknown manual provenance gets the lower (third-party) weight, never the higher one.
    return value if value in {"primary", "third_party"} else "third_party"


def _host(url: object) -> str:
    parsed = urlparse(url) if isinstance(url, str) else None
    host = (parsed.hostname or "").lower().rstrip(".") if parsed else ""
    return host[4:] if host.startswith("www.") else host


def vendor_website_host(vendor: object) -> str:
    site = _get(vendor, "Public Website") or _get(vendor, "website") or ""
    if not isinstance(site, str) or not site.strip():
        return ""
    site = site.strip()
    return _host(site if "://" in site else "https://" + site)


def check_document_integrity(document: Mapping[str, Any], website_host: str = "") -> list[str]:
    """Return reasons a document must be excluded; empty means basic integrity passed."""
    problems: list[str] = []
    url = document.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname:
        problems.append("source URL is not a valid http(s) URL")
    if document.get("source_type") != "MANUAL":
        text = document.get("text")
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None
        if expected is None or document.get("content_hash") != expected:
            problems.append("content_hash does not match the document text")
        host = _host(url)
        if website_host and host and not (host == website_host or host.endswith("." + website_host)):
            problems.append(f"host {host!r} is not the vendor's website host {website_host!r}")
    return problems


def analyse_detailed(vendor: object, docs: Sequence[object], reports: Sequence[object] = (),
                     planned_sources: Sequence[object] | None = None, *,
                     model_dir: str | Path = DEFAULT_MODEL_DIR, use_model: bool = True,
                     weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
                     strict_integrity: bool = False,
                     adjudicator: Callable[[str], dict[str, Any] | None] | None = None) -> AnalysisResult:
    vendor_id = _vendor_id(vendor)
    documents = [d for d in (_as_mapping(item) for item in docs) if d.get("vendor_id") == vendor_id]
    rejected_documents: dict[str, list[str]] = {}
    if strict_integrity:
        website_host = vendor_website_host(vendor)
        accepted: list[dict[str, Any]] = []
        for document in documents:
            problems = check_document_integrity(document, website_host)
            if problems:
                rejected_documents[str(document.get("doc_id"))] = problems
            else:
                accepted.append(document)
        documents = accepted
    vendor_reports = [r for r in reports if _get(r, "vendor_id") in (None, vendor_id)]

    signals: list[Signal] = []
    ambiguous: dict[str, list[str]] = {}
    llm_proposed: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for document in documents:
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        for candidate in extract_document_candidates(document):
            signal = verify_candidate(document, candidate)
            if signal is None or signal.signal_id in seen:
                continue
            seen.add(signal.signal_id)
            rule = score_specificity_rule(signal.excerpt)
            specificity, method = rule, "RULE"
            if use_model:
                specificity, method = score_specificity_with_method(signal.excerpt, model_dir)
            score_signal(signal, _manual_class(document), weights_path,
                         specificity=specificity, method=method)
            reasons: list[str] = []
            if method == "ML" and abs(specificity - rule) > DISAGREEMENT_THRESHOLD:
                reasons.append(f"classifier and rule scores differ by more than {DISAGREEMENT_THRESHOLD}")
            if AMBIGUOUS_BAND[0] <= specificity <= AMBIGUOUS_BAND[1]:
                reasons.append("specificity falls in the ambiguous band")
            if reasons:
                ambiguous[signal.signal_id] = reasons
                if adjudicator is not None:
                    proposal = adjudicator(signal.excerpt)
                    if proposal is not None:
                        llm_proposed[signal.signal_id] = proposal
                        if proposal.get("is_technical_evidence") and specificity < LLM_TECHNICAL_SPECIFICITY:
                            score_signal(signal, _manual_class(document), weights_path,
                                         specificity=LLM_TECHNICAL_SPECIFICITY, method="LLM")
                        for provider in proposal.get("providers", []):
                            if provider not in signal.providers:
                                signal.providers.append(provider)
            signals.append(signal)

    verdict = generate_verdict(vendor_id, _planned_sources(planned_sources, documents, vendor_reports),
                               documents, vendor_reports, signals, weights_path)
    if rejected_documents:
        verdict.reasons.append(
            f"{len(rejected_documents)} document(s) failed integrity or attribution checks and were excluded: "
            + "; ".join(f"{doc_id}: {', '.join(problems)}"
                         for doc_id, problems in sorted(rejected_documents.items())))
    return AnalysisResult(signals=signals, verdict=verdict, ambiguous=ambiguous,
                          llm_proposed=llm_proposed, rejected_documents=rejected_documents)


def analyse(vendor: object, docs: Sequence[object], reports: Sequence[object] = (),
            planned_sources: Sequence[object] | None = None, **kwargs: Any) -> tuple[list[Signal], VendorVerdict]:
    """Shared-contract entry point. See ``analyse_detailed`` for ambiguity flags."""
    result = analyse_detailed(vendor, docs, reports, planned_sources, **kwargs)
    return result.signals, result.verdict
