"""Backend entry point for the frontend: one call per vendor, M2 -> M3 -> M4 -> M5.

    result = run_vendor(vendor, mode="cache")      # or mode="live"
    result.verdict, result.risk, result.signals, result.notes

M2 (web) is optional: if it cannot run, the vendor is still analysed with M3
evidence and the problem is recorded in ``notes`` and as ERROR SourceReports.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from radar.contracts.models import (
    CriticalityResult, DepthPlan, Document, RiskAssessment, Signal,
    SourceReport, VendorProfile, VendorVerdict,
)
from radar.m3_sources.collect import collect_sources
from m4_intel.analyse import analyse
from m5_risk.criticality import assess_criticality
from m5_risk.planner import plan_depth
from m5_risk.risk import assess_risk

log = logging.getLogger(__name__)
DEFAULT_VENDORS = Path(__file__).resolve().parent / "fixtures" / "vendors.json"
_M2_TYPES = ("TRUST", "DOCS", "WEB")


@dataclass
class VendorResult:
    vendor: VendorProfile
    criticality: CriticalityResult
    plan: DepthPlan
    documents: list[Document]
    reports: list[SourceReport]
    signals: list[Signal]
    verdict: VendorVerdict
    risk: RiskAssessment
    notes: list[str] = field(default_factory=list)


def load_vendors(path: Union[str, Path] = DEFAULT_VENDORS) -> list[VendorProfile]:
    with open(path, "r", encoding="utf-8") as fh:
        return [VendorProfile(**row) for row in json.load(fh)]


def _collect_m2(vendor: VendorProfile, plan: DepthPlan, notes: list[str]
                ) -> tuple[list[Document], list[SourceReport]]:
    planned = [s for s in dict.fromkeys(plan.sources) if s in _M2_TYPES]
    if not planned:
        return [], []
    try:
        from radar.m2_web.collector import collect_web  # imported lazily: needs trafilatura
        m2_plan = DepthPlan(vendor_id=plan.vendor_id, tier=plan.tier, sources=planned,
                            max_pages=plan.max_pages, max_depth=plan.max_depth,
                            stop_rule=plan.stop_rule)
        return collect_web(vendor, m2_plan)
    except Exception as exc:  # noqa: BLE001 - M2 must never take the pipeline down
        msg = f"M2 web collection failed: {type(exc).__name__}: {exc}"
        log.warning("%s: %s", vendor.vendor_id, msg)
        notes.append(msg)
        return [], [SourceReport(vendor_id=vendor.vendor_id, source_type=s, status="ERROR",
                                 pages_fetched=0, note=msg) for s in planned]


def run_vendor(vendor: VendorProfile, *, mode: Optional[str] = None, use_web: bool = True,
               cache_dir: Optional[Union[str, Path]] = None, **m4_kwargs: Any) -> VendorResult:
    notes: list[str] = []
    crit = assess_criticality(vendor)
    plan = plan_depth(crit, vendor)
    docs_web, reps_web = _collect_m2(vendor, plan, notes) if use_web else ([], [])
    docs_src, reps_src = collect_sources(vendor, plan, mode=mode, cache_dir=cache_dir)
    docs, reports = docs_web + docs_src, reps_web + reps_src
    signals, verdict = analyse(vendor, docs, reports, planned_sources=plan.sources, **m4_kwargs)
    risk = assess_risk(vendor, crit, verdict)
    return VendorResult(vendor, crit, plan, docs, reports, signals, verdict, risk, notes)


def run_all(vendors: Sequence[VendorProfile], **kwargs: Any) -> list[VendorResult]:
    return [run_vendor(v, **kwargs) for v in vendors]
