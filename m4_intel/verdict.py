"""Coverage-aware vendor AI-usage verdicts from verified evidence signals.

This module implements the project's draft rules only. It does not infer that
AI is absent from missing evidence and does not perform model/LLM analysis.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from m4_intel.specificity import DEFAULT_WEIGHTS_PATH, load_source_weights
from radar.contracts.models import VendorVerdict


_USABLE_REPORT_STATUSES = {"OK"}
_TAG_RANK = {"NONE": 0, "MARKETING_ONLY": 1, "PROBABLE": 2, "STRONG": 3, "CONFIRMED": 4}
_ALLOWED_VERDICTS = {"Yes", "No", "Inconclusive"}
POSSIBLE_AI_SPECIFICITY = 0.5


def _possible_ai(signal: object) -> bool:
    specificity = _field(signal, "specificity", 0.0)
    numeric = specificity if isinstance(specificity, (int, float)) and not isinstance(specificity, bool) else 0.0
    providers = _field(signal, "providers", [])
    named = isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)) and bool(providers)
    return numeric >= POSSIBLE_AI_SPECIFICITY or named


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_class(value: object) -> str | None:
    meta = _field(value, "meta", {})
    if isinstance(meta, Mapping):
        candidate = meta.get("source_class")
        return candidate if candidate in {"primary", "third_party"} else None
    return None


def _normalize_planned_source(item: object) -> tuple[str, str | None] | None:
    if isinstance(item, str) and item.strip():
        return item.strip(), None
    if isinstance(item, Mapping):
        source_type = item.get("source_type")
        source_class = item.get("source_class")
        if _nonempty_string(source_type) and (source_class is None or source_class in {"primary", "third_party"}):
            return source_type.strip(), source_class
    return None


def _usable_document(document: object) -> bool:
    return (
        _nonempty_string(_field(document, "source_type"))
        and _nonempty_string(_field(document, "text"))
        and _nonempty_string(_field(document, "doc_id"))
    )


def calculate_coverage(
    planned_source_types: Sequence[object],
    documents: Sequence[object],
    reports: Sequence[object] | None = None,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
) -> float:
    """Return weighted source-family coverage in [0, 1].

    Each planned source family contributes once, regardless of document count.
    A family is covered only when at least one usable document of that source
    type exists. Planned items may be source-type strings or mappings with
    ``source_type`` and optional ``source_class``. For a bare MANUAL family,
    the primary weight is used as the planned family weight; callers can pass
    an explicit source_class mapping when the plan is specifically third-party.
    Reports are accepted for interface compatibility and diagnostics but a
    status alone never substitutes for a usable document.
    """
    if not isinstance(planned_source_types, Sequence) or isinstance(planned_source_types, (str, bytes)):
        raise ValueError("planned_source_types must be a sequence")
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise ValueError("documents must be a sequence")
    if reports is not None and (not isinstance(reports, Sequence) or isinstance(reports, (str, bytes))):
        raise ValueError("reports must be a sequence or None")

    config = load_source_weights(weights_path)
    weights = config["source_weights"]
    planned: dict[str, float] = {}
    for item in planned_source_types:
        normalized = _normalize_planned_source(item)
        if normalized is None:
            raise ValueError(f"Malformed planned source entry: {item!r}")
        source_type, source_class = normalized
        if source_type not in weights:
            raise ValueError(f"Unknown planned source type: {source_type!r}")
        configured = weights[source_type]
        if isinstance(configured, Mapping):
            selected_class = source_class or "primary"
            if selected_class not in configured:
                raise ValueError(f"No configured weight for {source_type!r} class {selected_class!r}")
            weight = float(configured[selected_class])
        else:
            if source_class is not None:
                raise ValueError(f"source_class is not valid for {source_type!r}")
            weight = float(configured)
        # Duplicate planned source families count only once.
        planned.setdefault(source_type, weight)

    denominator = sum(planned.values())
    if denominator == 0:
        return 0.0

    usable_families: set[str] = set()
    for document in documents:
        if not _usable_document(document):
            continue
        source_type = _field(document, "source_type")
        if source_type in planned:
            usable_families.add(source_type)

    numerator = sum(planned[source_type] for source_type in usable_families)
    return round(min(1.0, max(0.0, numerator / denominator)), 10)


def _valid_signal(signal: object) -> bool:
    return (
        _field(signal, "verified") is True
        and _nonempty_string(_field(signal, "signal_id"))
        and _nonempty_string(_field(signal, "source_type"))
        and _field(signal, "tag") in _TAG_RANK
    )


def _strong_evidence(signals: list[object]) -> bool:
    if any(_field(signal, "tag") == "CONFIRMED" for signal in signals):
        return True
    strong = [signal for signal in signals if _field(signal, "tag") == "STRONG"]
    probable_or_better = [signal for signal in signals if _TAG_RANK.get(_field(signal, "tag"), 0) >= 2]

    # A strong signal plus independent probable-or-better evidence from another source family.
    for left in strong:
        for right in probable_or_better:
            if left is not right and _field(left, "source_type") != _field(right, "source_type"):
                return True

    # Two strong signals from distinct source families.
    for index, left in enumerate(strong):
        for right in strong[index + 1:]:
            if _field(left, "source_type") != _field(right, "source_type"):
                return True
    return False


def _unique_values(signals: list[object], field_name: str) -> list[str]:
    values: list[str] = []
    for signal in signals:
        raw = _field(signal, field_name, [])
        if isinstance(raw, str):
            # Signal.category is a string in the shared contract; specificity.py
            # may serialize multiple mapped categories as a comma-separated list.
            candidates = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, str)):
            candidates = list(raw)
        else:
            candidates = []
        for value in candidates:
            if _nonempty_string(value) and value not in values:
                values.append(value)
    return values


def _strength(signals: list[object]) -> str:
    if not signals:
        return "None"
    strongest = max((_TAG_RANK.get(_field(signal, "tag"), 0) for signal in signals), default=0)
    return {0: "None", 1: "Marketing-only", 2: "Probable", 3: "Strong", 4: "Confirmed"}[strongest]


def generate_verdict(
    vendor_id: str,
    planned_source_types: Sequence[object],
    documents: Sequence[object],
    reports: Sequence[object],
    signals: Sequence[object],
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
) -> VendorVerdict:
    """Build a VendorVerdict from source coverage and verified Signals.

    Invalid/unverified signals are excluded. Malformed collections raise a
    ValueError, while malformed individual signals are ignored independently.
    """
    if not _nonempty_string(vendor_id):
        raise ValueError("vendor_id must be a non-empty string")
    for name, value in (("planned_source_types", planned_source_types), ("documents", documents),
                        ("reports", reports), ("signals", signals)):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a sequence")

    coverage = calculate_coverage(planned_source_types, documents, reports, weights_path)
    verified: list[object] = []
    seen_ids: set[str] = set()
    for signal in signals:
        if not _valid_signal(signal):
            continue
        # Do not let duplicate copies of one signal count as independent evidence.
        signal_id = _field(signal, "signal_id")
        if signal_id in seen_ids:
            continue
        seen_ids.add(signal_id)
        # A signal for a different vendor must not influence this vendor's verdict.
        if _field(signal, "vendor_id") != vendor_id:
            continue
        verified.append(signal)

    if _strong_evidence(verified):
        verdict = "Yes"
        reasons = ["Verified evidence met the configured draft threshold for AI-use signals."]
    elif (coverage >= 0.7 and not any(_TAG_RANK.get(_field(signal, "tag"), 0) >= 2 for signal in verified)
          and not any(_possible_ai(signal) for signal in verified)):
        verdict = "No"
        reasons = ["Coverage met the 0.7 threshold and no verified signal exceeded MARKETING_ONLY; this is a coverage-qualified result, not proof from missing evidence."]
    else:
        verdict = "Inconclusive"
        reasons = ["Available verified evidence did not meet the draft threshold for Yes or the coverage requirement for No."]

    if verdict != "No" and any(_possible_ai(signal) for signal in verified) and not _strong_evidence(verified):
        reasons.append("At least one verified signal indicates possible AI use (provider named or specificity at or above "
                       f"{POSSIBLE_AI_SPECIFICITY}); a No verdict is not allowed.")

    if coverage < 0.7:
        reasons.append(f"Coverage was {coverage:.3f}, below the 0.700 threshold required for No; missing evidence is not treated as evidence of absence.")
    if not verified:
        reasons.append("No valid verified signals were available for this vendor.")

    failed_sources: list[str] = []
    for report in reports:
        source_type = _field(report, "source_type")
        status = _field(report, "status")
        if _nonempty_string(source_type) and _nonempty_string(status) and status.upper() not in _USABLE_REPORT_STATUSES:
            label = f"{source_type}:{status}"
            if label not in failed_sources:
                failed_sources.append(label)
    if failed_sources:
        reasons.append("Source reports include non-OK outcomes: " + ", ".join(failed_sources) + ".")

    return VendorVerdict(
        vendor_id=vendor_id,
        ai_usage=verdict,
        strength=_strength(verified),
        coverage=coverage,
        signal_ids=[_field(signal, "signal_id") for signal in verified],
        providers=_unique_values(verified, "providers"),
        use_cases=_unique_values(verified, "category"),
        reasons=reasons,
    )


# Friendly aliases for callers that use the contract's original analyse wording.
def analyse_vendor(vendor_id: str, docs: Sequence[object], reports: Sequence[object],
                   signals: Sequence[object], planned_source_types: Sequence[object],
                   weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> VendorVerdict:
    """Compatibility wrapper with documents/reports/signals first."""
    return generate_verdict(vendor_id, planned_source_types, docs, reports, signals, weights_path)
