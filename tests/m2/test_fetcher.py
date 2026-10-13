
import hashlib
import json

from radar.m2_web import fetcher


def test_cache_mode_reads_saved_response(tmp_path, monkeypatch):
    url = "https://example.com/security"
    cache_file = tmp_path / (
        hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
    )

    cached = {
        "url": url,
        "status": 200,
        "html": "<html><title>Security</title></html>",
        "content_hash": "testhash",
        "fetched_at": "2026-10-12T00:00:00+00:00",
        "snapshot_path": str(cache_file),
        "content_type": "text/html",
        "error": "",
        "blocked": False,
        "from_cache": True,
    }

    cache_file.write_text(json.dumps(cached), encoding="utf-8")
    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("RADAR_MODE", "cache")

    result = fetcher.fetch_url(url)

    assert result["status"] == 200
    assert result["html"] == cached["html"]
    assert result["content_hash"] == "testhash"


def test_cache_mode_missing_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("RADAR_MODE", "cache")

    result = fetcher.fetch_url("https://example.com/missing")

    assert result["status"] == 0
    assert result["from_cache"] is True
    assert result["error"] == "Not cached"


def test_cache_mode_invalid_entry(tmp_path, monkeypatch):
    url = "https://example.com/security"
    cache_file = tmp_path / (
        hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
    )
    cache_file.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("RADAR_MODE", "cache")

    result = fetcher.fetch_url(url)

    assert result["status"] == 0
    assert result["error"] == "Invalid cache entry"


def test_pdf_response_can_be_cached(tmp_path, monkeypatch):
    url = "https://example.com/report.pdf"

    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.4 test"
        text = content.decode("latin-1")
        headers = {"Content-Type": "application/pdf"}

    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetcher, "_robots_allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_last_request", {})
    monkeypatch.setattr(fetcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        fetcher._session,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setenv("RADAR_MODE", "live")

    result = fetcher.fetch_url(url)

    assert result["status"] == 200
    assert result["content_type"] == "application/pdf"
    assert result["content_hash"] == hashlib.sha256(
        FakeResponse.content
    ).hexdigest()

    cache_file = tmp_path / (
        hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
    )
    assert cache_file.exists()


def test_retry_after_temporary_server_error(monkeypatch, tmp_path):
    responses = iter([500, 200])
    sleeps = []

    class FakeResponse:
        content = b"<html><body>Security</body></html>"
        text = content.decode()
        headers = {"Content-Type": "text/html"}

        def __init__(self, status):
            self.status_code = status

    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetcher, "_robots_allowed", lambda url: True)
    monkeypatch.setattr(fetcher, "_last_request", {})
    monkeypatch.setattr(
        fetcher.time, "sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setattr(
        fetcher._session,
        "get",
        lambda *args, **kwargs: FakeResponse(next(responses)),
    )
    monkeypatch.setenv("RADAR_MODE", "live")

    result = fetcher.fetch_url("https://example.com/security")

    assert result["status"] == 200
    assert sleeps


def test_robots_404_allows_crawling(monkeypatch):
    class FakeResponse:
        status_code = 404
        text = ""

    monkeypatch.setattr(fetcher, "_robots_cache", {})
    monkeypatch.setattr(
        fetcher._session,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert fetcher._robots_allowed("https://example.com/security") is True


def test_robots_disallow_rule_blocks_page(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "User-agent: *\nDisallow: /private"

    monkeypatch.setattr(fetcher, "_robots_cache", {})
    monkeypatch.setattr(
        fetcher._session,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert fetcher._robots_allowed("https://example.com/private") is False


def test_cache_mode_does_not_make_network_request(
    tmp_path, monkeypatch
):
    url = "https://example.com/security"
    cache_file = tmp_path / (
        hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
    )

    cached = {
        "url": url,
        "status": 200,
        "html": "<html><body>Cached security page</body></html>",
        "content_hash": "cachedhash",
        "fetched_at": "2026-10-12T00:00:00+00:00",
        "snapshot_path": str(cache_file),
        "content_type": "text/html",
        "error": "",
        "blocked": False,
        "from_cache": True,
    }

    cache_file.write_text(json.dumps(cached), encoding="utf-8")

    def fail_network(*args, **kwargs):
        raise AssertionError("Network request should not occur in cache mode")

    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetcher._session, "get", fail_network)
    monkeypatch.setenv("RADAR_MODE", "cache")

    result = fetcher.fetch_url(url)

    assert result["status"] == 200
    assert result["html"] == cached["html"]
    assert result["from_cache"] is True



def test_rate_limit_waits_for_same_host(monkeypatch, tmp_path):
    url = "https://example.com/security"
    host = "https://example.com"
    waits = []
    clock = {"now": 100.5}

    monkeypatch.setattr(fetcher, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetcher, "_last_request", {host: 100.0})
    monkeypatch.setattr(fetcher, "_robots_allowed", lambda url: True)
    monkeypatch.setattr(
        fetcher.time,
        "monotonic",
        lambda: clock["now"],
    )

    def fake_sleep(seconds):
        waits.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(fetcher.time, "sleep", fake_sleep)

    class FakeResponse:
        status_code = 200
        content = b"<html><body>Security page</body></html>"
        text = content.decode()
        headers = {"Content-Type": "text/html"}

    monkeypatch.setattr(
        fetcher._session,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setenv("RADAR_MODE", "live")

    fetcher.fetch_url(url)

    assert waits
    assert abs(waits[0] - 1.0) < 0.0001
