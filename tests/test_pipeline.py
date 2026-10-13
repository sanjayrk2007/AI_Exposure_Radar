"""End-to-end contract test for radar.pipeline (offline: fixture cache, no web)."""
from pathlib import Path

import pytest

from radar import pipeline
from radar.contracts.models import RiskAssessment, VendorVerdict

CACHE = Path(__file__).resolve().parent.parent / "radar" / "fixtures" / "m3" / "cache"
VENDORS = pipeline.load_vendors()


def test_six_vendors_loaded():
    assert [v.vendor_id for v in VENDORS] == [f"V-00{i}" for i in range(1, 7)]


@pytest.mark.parametrize("vendor", VENDORS, ids=lambda v: v.vendor_id)
def test_run_vendor_end_to_end(vendor):
    r = pipeline.run_vendor(vendor, mode="cache", use_web=False, cache_dir=CACHE)
    assert isinstance(r.verdict, VendorVerdict) and r.verdict.vendor_id == vendor.vendor_id
    assert isinstance(r.risk, RiskAssessment) and r.risk.risk_class
    assert r.criticality.tier and r.plan.sources
    for s in r.signals:  # no unverified excerpt may leave M4
        assert s.excerpt


def test_deterministic():
    a = pipeline.run_vendor(VENDORS[1], mode="cache", use_web=False, cache_dir=CACHE)
    b = pipeline.run_vendor(VENDORS[1], mode="cache", use_web=False, cache_dir=CACHE)
    assert (a.verdict, a.risk.risk_class) == (b.verdict, b.risk.risk_class)


def test_m2_failure_degrades_gracefully(monkeypatch):
    import radar.m2_web.collector as m2

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(m2, "collect_web", boom)
    r = pipeline.run_vendor(VENDORS[1], mode="cache", cache_dir=CACHE)
    assert r.notes and "M2 web collection failed" in r.notes[0]
    assert any(rep.source_type in ("TRUST", "DOCS", "WEB") and rep.status == "ERROR" for rep in r.reports)
    assert r.verdict.vendor_id == "V-002"
