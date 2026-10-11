import requests
from requests.adapters import BaseAdapter
from requests.models import Response

from radar.m3_sources.fetching import FetchResult, M2FetcherAdapter, PoliteFetcher


class StubAdapter(BaseAdapter):
    """Serves canned responses per URL; records every request. No network."""
    def __init__(self, routes):
        super().__init__()
        self.routes, self.requests = routes, []

    def send(self, request, **kw):
        self.requests.append(request)
        spec = self.routes.get(request.url)
        if isinstance(spec, list):                      # sequence of responses (retry tests)
            spec = spec.pop(0) if len(spec) > 1 else spec[0]
        if spec is None:
            spec = (404, b"nf", {})
        if isinstance(spec, Exception):
            raise spec
        status, body, headers = spec
        r = Response()
        r.status_code, r._content, r.url = status, body, request.url
        r._content_consumed = True
        r.headers.update({"Content-Type": "text/html", **headers})
        r.encoding = "utf-8"
        return r

    def close(self): pass


def make(routes, tmp_path, **kw):
    s = requests.Session()
    ad = StubAdapter(routes)
    s.mount("https://", ad)
    slept = []
    t = [0.0]
    def sleep(x): slept.append(x); t[0] += x
    f = PoliteFetcher(cache_dir=tmp_path, session=s, clock=lambda: t[0], sleep=sleep, retries=2,
                      api_hosts=("api.example",), **kw)
    return f, ad, slept


def test_get_only_ua_hash_snapshot_and_no_cookies(tmp_path):
    f, ad, _ = make({"https://a.example/robots.txt": (404, b"", {}),
                     "https://a.example/p": (200, b"<html>hi</html>", {"Set-Cookie": "sid=1"})}, tmp_path)
    r = f.fetch("https://a.example/p", vendor_id="V-002")
    assert r.outcome() == "ok" and r.text == "<html>hi</html>"
    assert len(r.content_hash) == 64 and r.snapshot_path.startswith("V-002/snapshots/")
    assert (tmp_path / r.snapshot_path).read_bytes() == b"<html>hi</html>"
    assert r.fetched_at.endswith("Z")
    assert all(q.method == "GET" for q in ad.requests)
    assert all(q.headers["User-Agent"].startswith("RadarBot") for q in ad.requests)
    assert len(f._session.cookies) == 0


def test_robots_disallow_blocks_without_fetching_page(tmp_path):
    f, ad, _ = make({"https://a.example/robots.txt": (200, b"User-agent: *\nDisallow: /careers", {})}, tmp_path)
    r = f.fetch("https://a.example/careers", vendor_id="V-002")
    assert r.outcome() == "blocked" and r.blocked_reason == "robots"
    assert [q.url for q in ad.requests] == ["https://a.example/robots.txt"]


def test_robots_unreachable_5xx_is_conservative_but_4xx_is_open(tmp_path):
    f, _, _ = make({"https://a.example/robots.txt": (503, b"", {})}, tmp_path)
    assert f.fetch("https://a.example/x", vendor_id="V-002").outcome() == "blocked"
    f, _, _ = make({"https://b.example/robots.txt": (403, b"", {}), "https://b.example/x": (200, b"ok", {})}, tmp_path)
    assert f.fetch("https://b.example/x", vendor_id="V-002").outcome() == "ok"


def test_api_hosts_skip_robots_pages_do_not(tmp_path):
    f, ad, _ = make({"https://api.example/v1": (200, b"{}", {})}, tmp_path)
    assert f.fetch("https://api.example/v1", vendor_id="V-002", kind="api").outcome() == "ok"
    assert not any(q.url.endswith("robots.txt") for q in ad.requests)


def test_per_host_rate_limit_at_least_1_5s(tmp_path):
    f, _, slept = make({"https://a.example/robots.txt": (404, b"", {}), "https://a.example/1": (200, b"1", {}),
                        "https://a.example/2": (200, b"2", {})}, tmp_path)
    f.fetch("https://a.example/1", vendor_id="V-002")
    f.fetch("https://a.example/2", vendor_id="V-002")
    assert slept and all(s >= 1.5 - 1e-9 for s in slept)


def test_api_host_may_use_shorter_interval_but_pages_cannot(tmp_path):
    f, _, slept = make({"https://api.example/1": (200, b"1", {}), "https://api.example/2": (200, b"2", {}),
                        "https://a.example/robots.txt": (404, b"", {}), "https://a.example/1": (200, b"1", {}),
                        "https://a.example/2": (200, b"2", {})}, tmp_path)
    f.fetch("https://api.example/1", vendor_id="V-002", kind="api", min_interval=0.3)
    f.fetch("https://api.example/2", vendor_id="V-002", kind="api", min_interval=0.3)
    assert slept == [0.3]
    slept.clear()
    f.fetch("https://a.example/1", vendor_id="V-002", min_interval=0.01)     # attempt to be faster: ignored
    f.fetch("https://a.example/2", vendor_id="V-002", min_interval=0.01)
    assert max(slept) >= 1.5 - 1e-9


def test_retry_then_success_and_final_429_is_blocked(tmp_path):
    f, ad, slept = make({"https://a.example/robots.txt": (404, b"", {}),
                         "https://a.example/p": [(503, b"", {"Retry-After": "2"}), (200, b"ok", {})]}, tmp_path)
    assert f.fetch("https://a.example/p", vendor_id="V-002").outcome() == "ok"
    assert 2.0 in slept
    f, _, _ = make({"https://a.example/robots.txt": (404, b"", {}), "https://a.example/q": (429, b"", {})}, tmp_path)
    r = f.fetch("https://a.example/q", vendor_id="V-002")
    assert r.outcome() == "blocked" and r.blocked_reason == "rate_limited"


def test_statuses_and_network_errors(tmp_path):
    f, _, _ = make({"https://a.example/robots.txt": (404, b"", {}), "https://a.example/gone": (404, b"", {}),
                    "https://a.example/deny": (403, b"", {}), "https://a.example/boom": requests.ConnectionError("down")},
                   tmp_path)
    assert f.fetch("https://a.example/gone", vendor_id="V-002").outcome() == "missing"
    assert f.fetch("https://a.example/deny", vendor_id="V-002").outcome() == "blocked"
    r = f.fetch("https://a.example/boom", vendor_id="V-002")
    assert r.outcome() == "error" and "ConnectionError" in r.error and r.status == 0
    assert f.fetch("ftp://a.example/x", vendor_id="V-002").outcome() == "error"


def test_size_cap_flags_truncation(tmp_path):
    f, _, _ = make({"https://a.example/robots.txt": (404, b"", {}), "https://a.example/big": (200, b"x" * 500, {})},
                   tmp_path, max_bytes=100)
    r = f.fetch("https://a.example/big", vendor_id="V-002")
    assert r.truncated and r.outcome() == "error" and r.snapshot_path == ""


def test_login_redirect_is_blocked(tmp_path):
    class Redirecting(StubAdapter):
        def send(self, request, **kw):
            r = super().send(request, **kw)
            if request.url.endswith("/jobs"):
                r.url = "https://a.example/login?next=/jobs"
            return r
    s = requests.Session(); s.mount("https://", Redirecting({"https://a.example/robots.txt": (404, b"", {}),
                                                              "https://a.example/jobs": (200, b"login form", {})}))
    f = PoliteFetcher(cache_dir=tmp_path, session=s, clock=lambda: 0.0, sleep=lambda x: None)
    r = f.fetch("https://a.example/jobs", vendor_id="V-002")
    assert r.outcome() == "blocked" and r.blocked_reason == "login_required"


def test_m2_adapter_maps_attributes_and_passes_supported_kwargs():
    class Res:  # M2-style result object
        http_status, text, content_hash, snapshot_path, fetched_at = 200, "<p>x</p>", "abc", "V-002/snapshots/a.html", "2026-10-09T00:00:00Z"
        url = "https://a.example/p"
        headers = {"X-Foo": "1"}
    class M2:
        def fetch(self, url, vendor_id): self.seen = (url, vendor_id); return Res()
    inner = M2()
    r = M2FetcherAdapter(inner).fetch("https://a.example/p", vendor_id="V-002", kind="api", headers={"a": "b"})
    assert inner.seen == ("https://a.example/p", "V-002")      # unsupported kwargs dropped
    assert isinstance(r, FetchResult) and r.outcome() == "ok" and r.content_hash == "abc" and r.headers == {"x-foo": "1"}


def test_m2_adapter_never_raises():
    class M2:
        def fetch(self, url): raise RuntimeError("boom")
    r = M2FetcherAdapter(M2()).fetch("https://a.example/p", vendor_id="V-002")
    assert r.outcome() == "error" and "RuntimeError" in r.error
