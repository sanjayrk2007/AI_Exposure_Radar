
from pathlib import Path
from radar.collector import _extract_page

import pytest

from radar.collector import (
    collect_vendor_pages,
    normalize_start_url,
)


class FakeResponse:
    def __init__(self, url, html, status=200, content_type="text/html"):
        self.url = url
        self.text = html
        self.status_code = status
        self.headers = {"Content-Type": content_type}


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        response = self.responses.get(url)
        if response is None:
            raise AssertionError(f"Unexpected request: {url}")
        return response

    def close(self):
        pass


def profile(website="example.com", vendor_id="V-001"):
    return {
        "Vendor ID": vendor_id,
        "Vendor Name": "Test Vendor",
        "Public Website": website,
    }


def test_normalize_website():
    assert normalize_start_url("www.example.com") == "https://www.example.com/"


def test_rejects_invalid_website():
    with pytest.raises(ValueError):
        normalize_start_url("ftp://example.com")


def test_collects_homepage_and_relevant_same_host_page(tmp_path):
    home = "https://example.com/"
    ai_page = "https://example.com/ai-platform"
    session = FakeSession({
        home: FakeResponse(
            home,
            '<html><title>Home</title><body>'
            '<p>Vendor homepage.</p>'
            '<a href="/ai-platform">AI platform</a>'
            '<a href="https://other.example/page">External AI</a>'
            '</body></html>',
        ),
        ai_page: FakeResponse(
            ai_page,
            '<html><title>AI Platform</title>'
            '<body><p>Current AI platform details.</p></body></html>',
        ),
    })

    docs, events = collect_vendor_pages(
        profile(), tmp_path, max_pages=2,
        delay_seconds=0, session=session,
    )

    assert len(docs) == 2
    assert all(doc.vendor_id == "V-001" for doc in docs)
    assert all(doc.source_type == "WEB" for doc in docs)
    assert docs[0].title == "Home"
    assert Path(docs[0].snapshot_path).exists()
    assert len({doc.content_hash for doc in docs}) == 2
    assert not any("other.example" in url for url in session.requested)
    assert sum(event["status"] == "FETCHED" for event in events) == 2


def test_rejects_worked_example(tmp_path):
    with pytest.raises(ValueError, match="V-000"):
        collect_vendor_pages(
            profile(vendor_id="V-000"),
            tmp_path,
            delay_seconds=0,
            session=FakeSession({}),
        )


def test_http_error_is_logged_without_document(tmp_path):
    home = "https://example.com/"
    session = FakeSession({
        home: FakeResponse(home, "Unavailable", status=503),
    })

    docs, events = collect_vendor_pages(
        profile(), tmp_path, delay_seconds=0, session=session,
    )

    assert docs == []
    assert events[0]["status"] == "HTTP_ERROR"
    assert events[0]["http_status"] == 503

def test_extract_page_separates_content_blocks_and_removes_navigation():
    html = """
    <html>
      <head><title>Example AI Platform</title></head>
      <body>
        <nav><a href="/ai">AI navigation item</a></nav>
        <main>
          <h1>AI Platform</h1>
          <p>Our platform uses machine learning.</p>
          <p>It classifies incoming documents.</p>
        </main>
        <footer>Generic footer text</footer>
      </body>
    </html>
    """

    title, text, links = _extract_page(
        html, "https://example.com/", "example.com"
    )

    assert title == "Example AI Platform"
    assert "Our platform uses machine learning." in text
    assert "It classifies incoming documents." in text
    assert "\n" in text
    assert "Generic footer text" not in text
    assert "AI navigation item" not in text
    # Link discovery is retained for the collector.
    assert any(url == "https://example.com/ai" for url, _ in links)


def test_extract_page_fallback_handles_generic_html():
    html = """
    <html><body>
      <div>Generative AI supports document search.</div>
    </body></html>
    """

    _, text, _ = _extract_page(
        html, "https://example.com/", "example.com"
    )

    assert "Generative AI supports document search." in text


def test_extract_page_preserves_relevant_text_in_generic_containers():
    html = """
    <html><body>
      <main>
        <h1>BNY</h1>
        <p>We provide financial services.</p>
        <div>Cloud-native and AI enabled platform for data and operations.</div>
        <div>Connect people and platforms.</div>
      </main>
      <nav>Navigation noise</nav>
      <footer>Footer noise</footer>
    </body></html>
    """

    _, text, _ = _extract_page(
        html, "https://example.com/", "example.com"
    )

    assert "We provide financial services." in text
    assert "Cloud-native and AI enabled platform for data and operations." in text
    assert "Connect people and platforms." in text
    assert "Navigation noise" not in text
    assert "Footer noise" not in text
