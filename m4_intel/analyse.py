"""M4 entry point: documents -> verified, tagged signals -> coverage-aware verdict.

Implements the shared contract ``analyse(vendor, docs, reports) -> (signals, verdict)``.
The run is deterministic and makes no network or LLM calls. Specificity comes from the
saved classifier when a usable local artifact exists and otherwise from the rule score.
Quote verification proves only that an excerpt exists verbatim in a document; it does not
prove the vendor's claim is true.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from m4_intel.candidates import extract_document_candidates
from m4_intel.specificity import DEFAULT_WEIGHTS_PATH, score_signal, score_specificity_rule
from m4_intel.train_specificity import DEFAULT_MODEL_DIR, score_specificity_with_method
from m4_intel.verdict import generate_verdict
from m4_intel.verify import verify_candidate
from radar.contracts.models import Signal, VendorVerdict

EXAMPLE_VENDOR_ID = "V-000"
DISAGREEMENT_THRESHOLD = 0.4          # plan 4.5: |classifier - rule| above this is ambiguous
AMBIGUOUS_BAND = (0.3, 0.6)           # plan M4: specificity inside this band is ambiguous
_UNPLANNED_STATUSES = {"NOT_APPLICABLE", "SKIPPED_BY_PLAN"}


@dataclass
class AnalysisResult:
    signals: list[Signal]
    verdict: VendorVerdict
    # signal_id -> reasons it should go to an adjudicator/analyst review. Never auto-resolved.
    ambiguous: dict[str, list[str]] = field(default_factory=dict)


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
    """Planned families: explicit plan, else reports not marked N/A or skipped, else docs."""
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
    return list(dict.fromkeys(d["source_type"] for d in docs if isinstance(d.get("source_type"), str)))


def _manual_class(document: Mapping[str, Any]) -> str | None:
    if document.get("source_type") != "MANUAL":
        return None
    meta = document.get("meta")
    value = meta.get("source_class") if isinstance(meta, Mapping) else None
    # Unknown manual provenance gets the lower (third-party) weight, never the higher one.
    return value if value in {"primary", "third_party"} else "third_party"


def analyse_detailed(vendor: object, docs: Sequence[object], reports: Sequence[object] = (),
                     planned_sources: Sequence[object] | None = None, *,
                     model_dir: str | Path = DEFAULT_MODEL_DIR, use_model: bool = True,
                     weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> AnalysisResult:
    vendor_id = _vendor_id(vendor)
    documents = [d for d in (_as_mapping(item) for item in docs) if d.get("vendor_id") == vendor_id]
    vendor_reports = [r for r in reports if _get(r, "vendor_id") in (None, vendor_id)]

    signals: list[Signal] = []
    ambiguous: dict[str, list[str]] = {}
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
            signals.append(signal)

    verdict = generate_verdict(vendor_id, _planned_sources(planned_sources, documents, vendor_reports),
                               documents, vendor_reports, signals, weights_path)
    return AnalysisResult(signals=signals, verdict=verdict, ambiguous=ambiguous)


def analyse(vendor: object, docs: Sequence[object], reports: Sequence[object] = (),
            planned_sources: Sequence[object] | None = None, **kwargs: Any) -> tuple[list[Signal], VendorVerdict]:
    """Shared-contract entry point. See ``analyse_detailed`` for ambiguity flags."""
    result = analyse_detailed(vendor, docs, reports, planned_sources, **kwargs)
    return result.signals, result.verdict
