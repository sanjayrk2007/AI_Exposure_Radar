"""Regression: bare-domain vendor websites ("www.example.com") must not be dropped."""
from radar.m2_web import discover


def test_bare_domain_gets_https(monkeypatch):
    fetched = []

    def fake_fetch(url, *a, **k):
        fetched.append(url)
        return {"status": 404, "html": ""}

    monkeypatch.setattr(discover, "fetch_url", fake_fetch)
    urls = discover.discover_urls("www.example.com", max_pages=3, max_depth=1)
    assert fetched and fetched[0] == "https://www.example.com/"
    assert urls == [("https://www.example.com/", 0, "WEB")]


def test_explicit_scheme_and_blank_unchanged(monkeypatch):
    monkeypatch.setattr(discover, "fetch_url", lambda *a, **k: {"status": 404, "html": ""})
    assert discover.discover_urls("http://example.com")[0][0] == "http://example.com/"
    assert discover.discover_urls("   ") == []
    assert discover.discover_urls("ftp://example.com") == []
