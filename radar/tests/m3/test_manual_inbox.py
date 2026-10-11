import dataclasses

import pytest

from radar.contracts.models import Document
from radar.m3_sources import add_manual_evidence
from radar.m3_sources.manual_inbox import list_manual_evidence, remove_manual_evidence
from radar.m3_sources.store import read_documents

TEXT = 'The post says agents "draft borrower correspondence and summarise call history".'


def add(**kw):
    args = dict(vendor_id="V-003", url="https://example.invalid/post/1", text=TEXT,
                source_class="primary", captured_at="2026-10-08", screenshot_path="")
    args.update(kw)
    return add_manual_evidence(**args)


def test_creates_contract_conformant_manual_document():
    d = add()
    assert isinstance(d, Document)
    assert {f.name for f in dataclasses.fields(d)} == {"doc_id", "vendor_id", "source_type", "url", "fetched_at",
        "http_status", "content_hash", "title", "text", "snapshot_path", "meta"}
    assert d.source_type == "MANUAL" and d.vendor_id == "V-003"
    assert d.text == TEXT                         # stored verbatim so quotes verify
    assert d.fetched_at == "2026-10-08T00:00:00Z" # ISO 8601 UTC
    assert d.meta["source_class"] == "primary"
    assert d.meta["synthetic"] is False and d.meta["http_status_nominal"] is True
    assert len(d.content_hash) == 64


def test_positional_call_matches_plan_signature():
    d = add_manual_evidence("V-003", "https://example.invalid/a", "Some exact excerpt text.", "third_party",
                            "2026-10-01T10:30:00+02:00", "")
    assert d.meta["source_class"] == "third_party"
    assert d.fetched_at == "2026-10-01T08:30:00Z"


@pytest.mark.parametrize("raw", ["third-party", "Third Party", "THIRD_PARTY"])
def test_source_class_aliases(raw):
    assert add(source_class=raw).meta["source_class"] == "third_party"


@pytest.mark.parametrize("bad", ["secondary", "", None])
def test_bad_source_class_rejected(bad):
    with pytest.raises(ValueError):
        add(source_class=bad)


@pytest.mark.parametrize("vid", ["V-000", "V-12", "x", "", "../V-001"])
def test_bad_or_example_vendor_rejected(vid):
    with pytest.raises(ValueError):
        add(vendor_id=vid)


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_text_rejected(text):
    with pytest.raises(ValueError):
        add(text=text)


@pytest.mark.parametrize("url", ["", "not a url", "ftp://x.example/a", "https://nodots"])
def test_bad_url_rejected(url):
    with pytest.raises(ValueError):
        add(url=url)


def test_url_without_scheme_as_in_v000_example():
    d = add(url="linkedin.com/posts/example-exec-post")
    assert d.url == "https://linkedin.com/posts/example-exec-post"
    assert d.meta["url_as_entered"] == "linkedin.com/posts/example-exec-post"  # never fetched


@pytest.mark.parametrize("raw,iso", [("09-03-2026", "2026-03-09T00:00:00Z"), ("2026-03-09", "2026-03-09T00:00:00Z"),
                                      ("2026-03-09T10:00:00Z", "2026-03-09T10:00:00Z")])
def test_captured_at_formats(raw, iso):
    assert add(captured_at=raw).fetched_at == iso


def test_invalid_date_rejected_and_empty_means_now():
    with pytest.raises(ValueError):
        add(captured_at="yesterday")
    assert add(captured_at="").fetched_at.endswith("Z")


def test_screenshot_hashed_when_present_and_flagged_when_missing(tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"png-bytes")
    d = add(screenshot_path=str(shot))
    assert d.meta["screenshot_found"] is True and len(d.meta["screenshot_sha256"]) == 64
    assert d.snapshot_path == str(shot)
    d2 = add(text="Another excerpt here.", screenshot_path=str(tmp_path / "missing.png"))
    assert d2.meta["screenshot_found"] is False


def test_persisted_idempotent_listed_and_removable():
    a = add()
    b = add()                                    # identical evidence -> same id, no duplicate
    assert a.doc_id == b.doc_id
    assert len(read_documents("V-003", ["MANUAL"])) == 1
    c = add(text="A second, different excerpt.")
    assert [d.doc_id for d in list_manual_evidence("V-003")] and len(list_manual_evidence("V-003")) == 2
    assert remove_manual_evidence("V-003", c.doc_id) is True
    assert remove_manual_evidence("V-003", c.doc_id) is False
    assert len(list_manual_evidence("V-003")) == 1


def test_crlf_normalised_but_content_preserved():
    d = add(text="line one.\r\nline two.  ")
    assert d.text == "line one.\nline two."
