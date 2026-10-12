import json
import threading
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from radar.contracts.models import DepthPlan, Document, SourceReport, VendorProfile
from radar.m3_sources import add_manual_evidence, collect_sources, store
from radar.m3_sources.common import M3_SOURCE_TYPES, STATUSES, make_report
from radar.tests.m3.fakes import FakeFetcher, plan, resp, vendor
from radar.tests.m3.test_careers import CAREERS, GH_FEED, HOME, gh_job, gh_routes

FIX = Path(__file__).resolve().parents[2] / "fixtures"
ORDER = ["CAREERS", "REPOS", "FILINGS", "MANUAL"]


def types(reports):
    return [r.source_type for r in reports]


def by(reports):
    return {r.source_type: r for r in reports}


def live_fetcher():
    return FakeFetcher(gh_routes([gh_job(1, "Machine Learning Engineer", "We use machine learning for fraud scoring.")]))


def test_always_four_reports_in_fixed_order_with_allowed_values():
    docs, reps = collect_sources(vendor(), plan(), mode="live", fetcher=live_fetcher())
    assert types(reps) == ORDER and all(isinstance(r, SourceReport) and r.status in STATUSES for r in reps)
    assert all(r.vendor_id == "V-002" for r in reps) and all(isinstance(d, Document) for d in docs)


def test_live_run_statuses_for_each_collector():
    docs, reps = collect_sources(vendor(), plan(), mode="live", fetcher=live_fetcher())
    r = by(reps)
    assert r["CAREERS"].status == "OK" and len(docs) == 1
    assert r["REPOS"].status == "NOT_APPLICABLE"                      # no GitHub org -> 404s everywhere
    assert r["FILINGS"].status == "ERROR" and "User-Agent" in r["FILINGS"].note   # no RADAR_SEC_USER_AGENT
    assert r["MANUAL"].status == "EMPTY"


def test_skipped_by_plan_never_touches_network():
    f = FakeFetcher()
    low = plan(tier="Low", sources=["TRUST", "DOCS"], max_pages=6, max_depth=1)
    docs, reps = collect_sources(vendor(), low, mode="live", fetcher=f)
    assert [r.status for r in reps] == ["SKIPPED_BY_PLAN"] * 4 and docs == [] and f.calls == []
    med = plan(tier="Medium", sources=["TRUST", "CAREERS"])
    _, reps = collect_sources(vendor(), med, mode="cache")
    r = by(reps)
    assert r["REPOS"].status == r["FILINGS"].status == r["MANUAL"].status == "SKIPPED_BY_PLAN"
    assert r["CAREERS"].status == "ERROR" and "cache miss" in r["CAREERS"].note


def test_cache_miss_is_error_and_never_uses_network():
    f = FakeFetcher()
    docs, reps = collect_sources(vendor(), plan(), mode="cache", fetcher=f)
    assert f.calls == [] and docs == []
    assert [r.status for r in reps[:3]] == ["ERROR"] * 3 and "RADAR_MODE=live" in reps[0].note


def test_live_then_cache_round_trip_is_identical_and_repeatable():
    live_docs, live_reps = collect_sources(vendor(), plan(), mode="live", fetcher=live_fetcher())
    f = FakeFetcher()
    d1, r1 = collect_sources(vendor(), plan(), mode="cache", fetcher=f)
    d2, r2 = collect_sources(vendor(), plan(), mode="cache", fetcher=f)
    assert f.calls == []
    assert [asdict(d) for d in d1] == [asdict(d) for d in live_docs] == [asdict(d) for d in d2]
    assert [asdict(r) for r in r1] == [asdict(r) for r in r2]                       # identical on re-run
    assert [(r.source_type, r.status, r.pages_fetched) for r in r1] == [(r.source_type, r.status, r.pages_fetched) for r in live_reps]
    assert all(r.note.endswith("[from cache]") for r in r1[:3])


def test_mode_comes_from_environment(monkeypatch):
    monkeypatch.setenv("RADAR_MODE", "live")
    f = live_fetcher()
    collect_sources(vendor(), plan(), fetcher=f)
    assert f.calls
    monkeypatch.setenv("RADAR_MODE", "cache")
    f2 = FakeFetcher()
    collect_sources(vendor(), plan(), fetcher=f2)
    assert f2.calls == []
    monkeypatch.setenv("RADAR_MODE", "bogus")
    with pytest.raises(ValueError):
        collect_sources(vendor(), plan())


def test_live_write_preserves_other_modules_cache_entries(tmp_path):
    m2_doc = Document("V-002-WEB-1", "V-002", "WEB", "https://x.example/", "2026-10-09T00:00:00Z", 200, "h", "t", "txt", "s", {})
    m2_rep = SourceReport("V-002", "WEB", "OK", 3, "m2 note")
    store.write_source_results("V-002", "WEB", [m2_doc], m2_rep)
    collect_sources(vendor(), plan(), mode="live", fetcher=live_fetcher())
    assert [d.doc_id for d in store.read_documents("V-002", ["WEB"])] == ["V-002-WEB-1"]
    assert store.read_reports("V-002", ["WEB"])[0].note == "m2 note"
    assert {r.source_type for r in store.read_reports("V-002")} == {"WEB", "CAREERS", "REPOS", "FILINGS", "MANUAL"} - {"MANUAL"}


def test_wrapped_json_shape_from_other_module_is_preserved():
    path = store.documents_path("V-002")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"documents": [asdict(Document("V-002-WEB-1", "V-002", "WEB", "u", "t", 200, "h", "t", "x", "s", {}))]}))
    collect_sources(vendor(), plan(), mode="live", fetcher=live_fetcher())
    raw = json.loads(path.read_text())
    assert set(raw) == {"documents"} and {d["source_type"] for d in raw["documents"]} == {"WEB", "CAREERS"}


def test_collector_exception_is_isolated_to_that_source():
    class Exploding(FakeFetcher):
        def fetch(self, url, **kw):
            if "api.github.com" in url:
                raise RuntimeError("kaboom")
            return super().fetch(url, **kw)
    docs, reps = collect_sources(vendor(), plan(), mode="live", fetcher=Exploding(gh_routes([gh_job(1, "ML Engineer", "machine learning")])))
    r = by(reps)
    assert r["CAREERS"].status == "OK" and r["REPOS"].status == "ERROR" and "RuntimeError" in r["REPOS"].note
    assert types(reps) == ORDER


@pytest.mark.parametrize("vid", ["V-000", "bad"])
def test_invalid_or_example_vendor_rejected(vid):
    with pytest.raises(ValueError):
        collect_sources(vendor(vid=vid), plan(vid=vid), mode="cache")


def test_plan_vendor_mismatch_rejected():
    with pytest.raises(ValueError):
        collect_sources(vendor(), plan(vid="V-003"), mode="cache")


def test_manual_evidence_flows_through_regardless_of_plan():
    _, reps = collect_sources(vendor(), plan(sources=["CAREERS"]), mode="cache")
    assert by(reps)["MANUAL"].status == "SKIPPED_BY_PLAN"
    d = add_manual_evidence("V-002", "https://x.example/p", "Our assistant drafts replies.", "primary", "2026-10-08", "")
    docs, reps = collect_sources(vendor(), plan(sources=["CAREERS"]), mode="cache")
    m = by(reps)["MANUAL"]
    assert m.status == "OK" and "not in the depth plan" in m.note and [x.doc_id for x in docs if x.source_type == "MANUAL"] == [d.doc_id]
    _, reps = collect_sources(vendor(), plan(), mode="cache")
    assert by(reps)["MANUAL"].status == "OK" and "1 primary" in by(reps)["MANUAL"].note


def test_cache_inconsistency_detected():
    store.write_source_results("V-002", "CAREERS", [], make_report("V-002", "CAREERS", "OK", 3, "claimed ok"))
    _, reps = collect_sources(vendor(), plan(), mode="cache")
    assert by(reps)["CAREERS"].status == "ERROR" and "inconsistent" in by(reps)["CAREERS"].note


def test_concurrent_writers_do_not_lose_updates():
    def job(stype):
        for i in range(10):
            d = Document(f"V-002-{stype}-{i}", "V-002", stype, "u", "t", 200, "h", "t", "x", "s", {})
            store.write_source_results("V-002", stype, [d], make_report("V-002", stype, "OK", 1, "n"))
    ts = [threading.Thread(target=job, args=(s,)) for s in ("CAREERS", "REPOS", "FILINGS")]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert {d.source_type for d in store.read_documents("V-002")} == {"CAREERS", "REPOS", "FILINGS"}
    assert len(store.read_reports("V-002")) == 3


# ---- synthetic fixture cache, all six vendors -----------------------------------------------
def fixture_run():
    vendors = {v["vendor_id"]: VendorProfile(**v) for v in json.loads((FIX / "vendors.json").read_text())}
    plans = json.loads((FIX / "m3" / "depth_plans.json").read_text())
    tiers = {"V-001": "High", "V-002": "Critical", "V-003": "High", "V-004": "Medium", "V-005": "High", "V-006": "Critical"}
    out = {}
    for vid, tier in tiers.items():
        out[vid] = collect_sources(vendors[vid], DepthPlan(vendor_id=vid, **plans[tier]), mode="cache", cache_dir=FIX / "m3" / "cache")
    return out


def test_fixture_cache_covers_every_status_for_all_six_vendors():
    res = fixture_run()
    statuses = {(vid, r.source_type): r.status for vid, (_, reps) in res.items() for r in reps}
    assert {s for s in statuses.values()} == set(STATUSES)                          # every enum value is exercised
    assert statuses[("V-002", "CAREERS")] == statuses[("V-002", "REPOS")] == statuses[("V-002", "FILINGS")] == "OK"
    assert statuses[("V-003", "CAREERS")] == "ERROR" and statuses[("V-006", "CAREERS")] == "BLOCKED"
    assert statuses[("V-004", "REPOS")] == "SKIPPED_BY_PLAN" and statuses[("V-001", "FILINGS")] == "NOT_APPLICABLE"
    assert statuses[("V-001", "CAREERS")] == "EMPTY" and statuses[("V-003", "MANUAL")] == "OK"
    assert all(types(reps) == ORDER for _, reps in res.values())


def test_fixture_documents_are_contract_valid_and_clearly_synthetic():
    seen = set()
    for vid, (docs, _) in fixture_run().items():
        for d in docs:
            assert [f.name for f in fields(d)] == [f.name for f in fields(Document)]
            assert d.vendor_id == vid and d.source_type in M3_SOURCE_TYPES and d.doc_id not in seen
            seen.add(d.doc_id)
            assert isinstance(d.http_status, int) and d.fetched_at.endswith("Z") and d.text.strip() and d.content_hash
            assert d.meta["synthetic"] is True and d.title.startswith("[SYNTHETIC FIXTURE]")
            assert ".invalid" in d.url
            assert (FIX / "m3" / "cache" / d.snapshot_path).is_file()
    assert len(seen) >= 8


def test_fixture_manual_documents_have_source_class():
    for _, (docs, _) in fixture_run().items():
        for d in docs:
            if d.source_type == "MANUAL":
                assert d.meta["source_class"] in ("primary", "third_party")
