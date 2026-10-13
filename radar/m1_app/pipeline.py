"""M1 orchestration for the AI Exposure Radar modules."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radar.contracts.models import (
    CriticalityResult,
    DepthPlan,
    Document,
    RiskAssessment,
    Signal,
    SourceReport,
    VendorProfile,
    VendorVerdict,
)
from m5_risk.criticality import assess_criticality
from m5_risk.planner import plan_depth
from m5_risk.risk import assess_risk
from m4_intel.analyse import analyse_detailed, DEFAULT_PLANNED_SOURCES
from radar.m3_sources.collect import collect_sources


# M2 has its own import root and separate contract classes.
_M2_ROOT = Path(__file__).resolve().parents[2] / "m2"
if str(_M2_ROOT) not in sys.path:
    sys.path.insert(0, str(_M2_ROOT))

from m2_web.collector import collect_web
from contracts.models import (  # noqa: E402
    DepthPlan as M2DepthPlan,
    VendorProfile as M2VendorProfile,
)


M2_SOURCE_TYPES = {"TRUST", "DOCS", "WEB"}
M3_SOURCE_TYPES = {"CAREERS", "REPOS", "FILINGS", "MANUAL"}
SUPPORTED_SOURCE_TYPES = M2_SOURCE_TYPES | M3_SOURCE_TYPES


@dataclass
class VendorRunResult:
    vendor_id: str
    criticality: CriticalityResult
    depth_plan: DepthPlan
    documents: list[Document]
    reports: list[SourceReport]
    verdict: VendorVerdict
    risk: RiskAssessment
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    rejected_documents: dict[str, list[str]] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    llm_proposed: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str = ""


def _m2_vendor(vendor: VendorProfile) -> M2VendorProfile:
    """Convert the shared profile into M2's equivalent contract."""
    return M2VendorProfile(
        vendor_id=vendor.vendor_id,
        name=vendor.name,
        description=vendor.description,
        service=vendor.service,
        category=vendor.category,
        website=vendor.website,
        process=vendor.process,
        dependency=vendor.dependency,
        data_class=vendor.data_class,
        data_volume=vendor.data_volume,
    )


def _m2_plan(vendor: VendorProfile, plan: DepthPlan) -> M2DepthPlan:
    sources = [source for source in plan.sources if source in M2_SOURCE_TYPES]
    return M2DepthPlan(
        vendor_id=vendor.vendor_id,
        tier=plan.tier,
        sources=sources,
        max_pages=plan.max_pages,
        max_depth=plan.max_depth,
        stop_rule=plan.stop_rule,
    )


def _as_shared_document(item: object) -> Document:
    if isinstance(item, Document):
        return item
    data = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
    return Document(**data)


def _as_shared_report(item: object) -> SourceReport:
    if isinstance(item, SourceReport):
        return item
    data = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
    return SourceReport(**data)


def run_vendor(
    vendor: VendorProfile,
    *,
    mode: str = "cache",
    cache_dir: str | Path | None = None,
    use_model: bool = True,
    strict_integrity: bool = False,
) -> VendorRunResult:
    """Run M5 planning, M2/M3 collection, M4 analysis and M5 risk for one vendor.

    Cache mode is the safe default. M2's fetcher uses RADAR_MODE, while M3
    accepts its mode explicitly. Live mode can make network requests.
    """
    if mode not in {"cache", "live"}:
        raise ValueError("mode must be 'cache' or 'live'")
    if vendor.vendor_id == "V-000":
        raise ValueError("V-000 is an example row and must not be analysed")

    criticality = assess_criticality(vendor)
    depth = plan_depth(criticality, vendor)

    documents: list[Document] = []
    reports: list[SourceReport] = []

    # M2 relies on an environment variable. Restore it after collection so
    # the selected mode does not leak into unrelated operations.
    previous_mode = os.environ.get("RADAR_MODE")
    os.environ["RADAR_MODE"] = mode
    try:
        if any(source in M2_SOURCE_TYPES for source in depth.sources):
            m2_docs, m2_reports = collect_web(
                _m2_vendor(vendor),
                _m2_plan(vendor, depth),
            )
            documents.extend(_as_shared_document(item) for item in m2_docs)
            reports.extend(_as_shared_report(item) for item in m2_reports)
    finally:
        if previous_mode is None:
            os.environ.pop("RADAR_MODE", None)
        else:
            os.environ["RADAR_MODE"] = previous_mode

    if any(source in M3_SOURCE_TYPES for source in depth.sources):
        m3_docs, m3_reports = collect_sources(
            vendor,
            depth,
            mode=mode,
            cache_dir=cache_dir,
        )
        documents.extend(m3_docs)
        reports.extend(m3_reports)

    planned_sources = [
        source for source in depth.sources
        if source in SUPPORTED_SOURCE_TYPES
    ]
    # M4's coverage model is based on TRUST, DOCS and WEB by default. Include
    # M3 families when they are planned, so missing sources reduce coverage.
    planned_sources = list(dict.fromkeys(planned_sources or DEFAULT_PLANNED_SOURCES))

    analysis = analyse_detailed(
        vendor,
        documents,
        reports,
        planned_sources,
        use_model=use_model,
        strict_integrity=strict_integrity,
    )

    risk = assess_risk(vendor, criticality, analysis.verdict)

    return VendorRunResult(
        vendor_id=vendor.vendor_id,
        criticality=criticality,
        depth_plan=depth,
        documents=documents,
        reports=reports,
        verdict=analysis.verdict,
        risk=risk,
        ambiguous=analysis.ambiguous,
        rejected_documents=analysis.rejected_documents,
        signals=analysis.signals,
        llm_proposed=analysis.llm_proposed,
    )


def result_to_dict(result: VendorRunResult) -> dict[str, Any]:
    """Return a JSON-friendly summary without embedding full document text."""
    return {
        "vendor_id": result.vendor_id,
        "criticality": asdict(result.criticality),
        "depth_plan": asdict(result.depth_plan),
        "documents_collected": len(result.documents),
        "source_reports": [asdict(report) for report in result.reports],
        "verdict": asdict(result.verdict),
        "risk": asdict(result.risk),
        "ambiguous_count": len(result.ambiguous),
        "rejected_document_count": len(result.rejected_documents),
        "error": result.error,
    }
