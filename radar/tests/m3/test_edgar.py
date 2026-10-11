import json

from radar.m3_sources.edgar import collect_filings
from radar.m3_sources.textutil import html_to_text
from radar.tests.m3.fakes import FakeFetcher, plan, resp, vendor

UA = "Team Test tester@example.edu"
CIK = 123456
SUB = f"https://data.sec.gov/submissions/CIK{CIK:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
FILING = f"https://www.sec.gov/Archives/edgar/data/{CIK}/000012345626000003/acme-3.htm"   # latest 10-K in sub()
FILLER = " ".join(f"Filler sentence number {i} about operations." for i in range(8))
HTML = f"""<html><body><div style="display:none"><ix:header>hidden machine learning boilerplate</ix:header></div>
<p>Item 1. Business</p>
<p>We serve clients in the U.S. and abroad. Our clients rely on us. We use machine learning to detect fraud across payment
flows. The models run daily. Results are reviewed by staff. Oversight is provided by committees.</p>
<p>{FILLER}</p>
<p>Item 1A. Risk Factors</p>
<p>Competition is strong. Generative AI may change expectations. We may not adopt it. Costs could rise. Rules are evolving.</p>
</body></html>"""


def sub(forms=("10-K/A", "10-Q", "10-K", "10-K"), dates=("2026-03-01", "2026-05-01", "2026-02-20", "2025-02-20"), name="ACME TEST VENDOR INC"):
    n = len(forms)
    return {"name": name, "filings": {"recent": {
        "form": list(forms), "filingDate": list(dates), "reportDate": ["2025-12-31"] * n,
        "accessionNumber": [f"0000123456-26-00000{i}" for i in range(1, n + 1)],
        "primaryDocument": [f"acme-{i}.htm" for i in range(1, n + 1)]}}}


def conf(cfg):
    c = json.loads(json.dumps(cfg))
    c["edgar"]["known_filers"] = {"V-002": {"cik": CIK, "name_hints": ["acme"]}}
    return c


def run(routes, cfg, ua=UA, vend=None):
    f = FakeFetcher(routes)
    docs, rep = collect_filings(vend or vendor(), plan(), f, conf(cfg), __import__("radar.m3_sources.common", fromlist=["x"]).AiTermMatcher(["machine learning", "generative AI", "AI"]), user_agent=ua)
    return docs, rep, f


GOOD = {SUB: resp(sub()), FILING: resp(HTML)}


def test_missing_user_agent_makes_no_request_and_errors(cfg):
    for ua in (None, "", "NoContactInfo"):
        docs, rep, f = run(GOOD, cfg, ua=ua)
        assert rep.status == "ERROR" and "User-Agent" in rep.note and f.calls == [] and docs == []


def test_known_filer_produces_exact_context_chunks(cfg):
    docs, rep, f = run(GOOD, cfg)
    assert rep.status == "OK" and rep.source_type == "FILINGS" and len(docs) == 2 and rep.pages_fetched == 2
    full = html_to_text(HTML)
    for d in docs:
        assert d.text in full and d.source_type == "FILINGS" and d.url == FILING     # exact substring, filing URL
        assert d.meta["char_start"] == full.index(d.text) and d.content_hash
        assert "not proof" in d.meta["evidence_caveat"] and d.meta["synthetic"] is False
    a, b = sorted(docs, key=lambda d: d.meta["char_start"])
    assert "machine learning" in a.text and a.meta["item"].startswith("Item 1. Business")
    assert "Generative AI" in b.text and b.meta["item"].startswith("Item 1A")
    assert "payment flows." in a.text and a.text.startswith("We serve clients in the U.S. and abroad.")             # 2 sentences before; U.S. not split
    assert "Results are reviewed by staff." in a.text and "Oversight is provided" not in a.text   # exactly 2 sentences after
    assert "hidden machine learning" not in full and a.meta["form"] == "10-K" and a.meta["filing_date"] == "2026-02-20T00:00:00Z"
    assert all(c["headers"]["User-Agent"] == UA and c["kind"] == "api" and c["min_interval"] == 0.3 for c in f.calls)


def test_latest_10k_chosen_and_amendments_ignored(cfg):
    docs, rep, f = run(GOOD, cfg)
    assert f.urls() == [SUB, FILING]
    assert docs[0].meta["accession"] == "0000123456-26-000003"        # the 2026-02-20 10-K, not the 10-K/A or the 2025 one


def test_filing_without_ai_terms_is_empty(cfg):
    docs, rep, _ = run({SUB: resp(sub()), FILING: resp("<p>Item 1. Business</p><p>We process payments.</p>")}, cfg)
    assert docs == [] and rep.status == "EMPTY" and "none of the AI terms" in rep.note


def test_no_annual_report_is_empty(cfg):
    docs, rep, _ = run({SUB: resp(sub(("10-Q",), ("2026-05-01",)))}, cfg)
    assert rep.status == "EMPTY" and "no annual report" in rep.note


def test_entity_mismatch_is_error_not_used(cfg):
    docs, rep, f = run({SUB: resp(sub(name="SOMEONE ELSE CORP"))}, cfg)
    assert rep.status == "ERROR" and "entity mismatch" in rep.note and FILING not in f.urls()


def test_non_filer_is_not_applicable(cfg):
    other = vendor(vid="V-004", name="Private Co, Inc.")
    docs, rep, f = run({TICKERS: resp({"0": {"cik_str": 1, "ticker": "X", "title": "Other Corp"}})}, cfg, vend=other)
    assert rep.status == "NOT_APPLICABLE" and docs == [] and f.urls() == [TICKERS]


def test_unknown_vendor_resolved_by_exact_name_in_company_list(cfg):
    other = vendor(vid="V-004", name="Acme Test Vendor, Inc.")
    r = {TICKERS: resp({"0": {"cik_str": CIK, "ticker": "ACME", "title": "ACME TEST VENDOR INC"}}), SUB: resp(sub()), FILING: resp(HTML)}
    docs, rep, _ = run(r, cfg, vend=other)
    assert rep.status == "OK" and "exact name match" in docs[0].meta["filer_resolution"] and rep.pages_fetched == 3


def test_sec_denial_is_blocked_and_failures_are_errors(cfg):
    assert run({SUB: resp("", 403)}, cfg)[1].status == "BLOCKED"
    assert run({SUB: resp("", 429)}, cfg)[1].status == "BLOCKED"
    assert run({SUB: resp("", 500)}, cfg)[1].status == "ERROR"
    assert run({SUB: resp("not json")}, cfg)[1].status == "ERROR"
    assert run({SUB: resp(sub()), FILING: resp("", 404)}, cfg)[1].status == "ERROR"


def test_unsupported_primary_document_type(cfg):
    s = sub(("10-K",), ("2026-02-20",)); s["filings"]["recent"]["primaryDocument"] = ["report.pdf"]
    docs, rep, _ = run({SUB: resp(s)}, cfg)
    assert rep.status == "ERROR" and "not supported" in rep.note
