
from contracts.models import DepthPlan, VendorProfile
from m2_web import collector
from m2_web.collector import collect_web


def make_vendor(website="https://example.com"):
    return VendorProfile(
        vendor_id="example",
        name="Example",
        website=website,
    )


def test_collector_returns_documents_and_reports(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(vendor_id="example")

    monkeypatch.setattr(
        collector,
        "discover_urls",
        lambda website, max_pages, max_depth: [
            ("https://example.com/security", 1, "TRUST")
        ],
    )
    monkeypatch.setattr(
        collector,
        "fetch_url",
        lambda url: {
            "url": url,
            "status": 200,
            "html": "<html><head><title>Security</title></head>"
                    "<body><main>We protect customer data.</main></body></html>",
            "content_hash": "testhash",
            "fetched_at": "2026-10-12T00:00:00+00:00",
            "snapshot_path": "cache/test.json",
            "content_type": "text/html",
            "error": "",
            "blocked": False,
            "from_cache": False,
        },
    )

    documents, reports = collect_web(vendor, plan)

    assert len(documents) == 1
    assert documents[0].vendor_id == "example"
    assert documents[0].source_type == "TRUST"
    assert "customer data" in documents[0].text
    assert documents[0].content_hash == "testhash"
    assert documents[0].fetched_at
    assert any(
        report.source_type == "TRUST" and report.status == "OK"
        for report in reports
    )


def test_collector_missing_website_reports_error():
    vendor = make_vendor("")
    plan = DepthPlan(vendor_id="example")

    documents, reports = collect_web(vendor, plan)

    assert documents == []
    assert reports
    assert all(report.status == "ERROR" for report in reports)


def test_collector_blocked_page_reports_blocked(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(vendor_id="example")

    monkeypatch.setattr(
        collector,
        "discover_urls",
        lambda website, max_pages, max_depth: [
            ("https://example.com/security", 1, "TRUST")
        ],
    )
    monkeypatch.setattr(
        collector,
        "fetch_url",
        lambda url: {
            "url": url,
            "status": 0,
            "html": "",
            "blocked": True,
            "error": "Disallowed by robots policy",
        },
    )

    documents, reports = collect_web(vendor, plan)

    assert documents == []
    trust_report = next(
        report for report in reports if report.source_type == "TRUST"
    )
    assert trust_report.status == "BLOCKED"


def test_collector_handles_discovery_failure(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(vendor_id="example")

    def fail_discovery(*args, **kwargs):
        raise RuntimeError("Test discovery failure")

    monkeypatch.setattr(collector, "discover_urls", fail_discovery)

    documents, reports = collect_web(vendor, plan)

    assert documents == []
    assert reports
    assert all(report.status == "ERROR" for report in reports)


def test_collector_skips_duplicate_urls(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(vendor_id="example")
    calls = []

    monkeypatch.setattr(
        collector,
        "discover_urls",
        lambda website, max_pages, max_depth: [
            ("https://example.com/security", 1, "TRUST"),
            ("https://example.com/security", 1, "TRUST"),
        ],
    )

    def fake_fetch(url):
        calls.append(url)
        return {
            "url": url,
            "status": 200,
            "html": "<html><body><main>Security information.</main></body></html>",
            "content_hash": "testhash",
            "fetched_at": "2026-10-12T00:00:00+00:00",
            "snapshot_path": "cache/test.json",
            "content_type": "text/html",
            "blocked": False,
            "from_cache": False,
        }

    monkeypatch.setattr(collector, "fetch_url", fake_fetch)

    documents, reports = collect_web(vendor, plan)

    assert len(documents) == 1
    assert len(calls) == 1


def test_collector_collects_pdf_document(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(vendor_id="example")
    pdf_bytes = b"%PDF-1.4 mock pdf bytes"

    monkeypatch.setattr(
        collector,
        "discover_urls",
        lambda website, max_pages, max_depth: [
            ("https://example.com/security.pdf", 1, "TRUST")
        ],
    )

    monkeypatch.setattr(
        collector,
        "fetch_url",
        lambda url: {
            "url": url,
            "status": 200,
            "html": "",
            "pdf_bytes": pdf_bytes,
            "content_hash": "pdfhash",
            "fetched_at": "2026-10-12T00:00:00+00:00",
            "snapshot_path": "cache/security.pdf.json",
            "content_type": "application/pdf",
            "error": "",
            "blocked": False,
            "from_cache": False,
        },
    )

    def fake_parse(url, html="", pdf_bytes=None):
        assert pdf_bytes == b"%PDF-1.4 mock pdf bytes"
        return "security.pdf", "Security report protects customer data."

    monkeypatch.setattr(collector, "parse_page", fake_parse)

    documents, reports = collect_web(vendor, plan)

    assert len(documents) == 1
    assert documents[0].source_type == "TRUST"
    assert documents[0].content_hash == "pdfhash"
    assert documents[0].meta["content_type"] == "application/pdf"
    assert "customer data" in documents[0].text

    trust_report = next(
        report for report in reports if report.source_type == "TRUST"
    )
    assert trust_report.status == "OK"

def test_collector_respects_page_budget(monkeypatch):
    vendor = make_vendor()
    plan = DepthPlan(
        vendor_id="example",
        sources=["TRUST", "DOCS", "WEB"],
        max_pages=1,
        max_depth=2,
    )

    fetched_urls = []

    monkeypatch.setattr(
        collector,
        "discover_urls",
        lambda website, max_pages, max_depth: [
            ("https://example.com/security", 1, "TRUST"),
            ("https://example.com/docs", 1, "DOCS"),
        ],
    )

    def fake_fetch(url):
        fetched_urls.append(url)
        return {
            "url": url,
            "status": 200,
            "html": "<html><body><main>Useful content</main></body></html>",
            "content_hash": "testhash",
            "fetched_at": "2026-10-12T00:00:00+00:00",
            "snapshot_path": "cache/test.json",
            "content_type": "text/html",
            "blocked": False,
            "from_cache": False,
        }

    monkeypatch.setattr(collector, "fetch_url", fake_fetch)

    documents, reports = collect_web(vendor, plan)

    assert len(fetched_urls) <= 1
    assert len(documents) <= 1
