"""Manual evidence inbox (plan 5/M3, MUST).

M1 builds the form; M3 owns :func:`add_manual_evidence` and its tests. The
analyst pastes the exact excerpt, the URL it came from, whether it is a
``primary`` (the vendor's own statement, e.g. an executive post) or
``third_party`` source, the capture date and, optionally, the path of a
screenshot. Nothing is fetched: executive posts (LinkedIn etc.) are never
scraped, they arrive here (plan section 9, "Ethics and rules").
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from radar.contracts.models import Document
from radar.m3_sources import store
from radar.m3_sources.common import (
    CAVEAT_MANUAL, COLLECTOR_VERSION, MANUAL_SOURCE_CLASSES, Clock, cache_root,
    iso_utc, make_doc_id, sha256_hex, to_iso_utc, utc_now, validate_vendor_id,
)

log = logging.getLogger(__name__)

# Document.http_status is an int in the shared contract, but nothing is fetched
# for manual evidence. 200 is a NOMINAL value meaning "analyst captured the page
# successfully"; the true situation is stated in meta["http_status_nominal"].
MANUAL_NOMINAL_HTTP_STATUS = 200
MAX_TEXT_CHARS = 200_000

_CLASS_ALIASES = {
    "primary": "primary",
    "third_party": "third_party",
    "third-party": "third_party",
    "third party": "third_party",
    "thirdparty": "third_party",
}


def normalize_source_class(value: str) -> str:
    key = str(value or "").strip().lower()
    if key not in _CLASS_ALIASES:
        raise ValueError(
            f"source_class must be one of {MANUAL_SOURCE_CLASSES}, got {value!r}")
    return _CLASS_ALIASES[key]


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw or re.search(r"\s", raw):
        raise ValueError("url is required and must not contain whitespace")
    if "://" not in raw:
        raw = "https://" + raw           # e.g. "linkedin.com/posts/..." as typed in V-000
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or "." not in parsed.hostname:
        raise ValueError(f"url is not a valid http(s) address: {url!r}")
    return raw


def add_manual_evidence(vendor_id: str, url: str, text: str, source_class: str,
                        captured_at: str, screenshot_path: str = "", *,
                        cache_dir: Optional[Union[str, Path]] = None,
                        clock: Optional[Clock] = None) -> Document:
    """Create (and persist) a MANUAL :class:`Document`.

    * ``text`` is stored exactly as pasted (only line endings are normalised and
      outer whitespace trimmed) so the quote can later be verified verbatim.
    * ``captured_at`` is the date/time the analyst captured the page: ISO 8601,
      or day-first ``DD-MM-YYYY`` (as in the V-000 example). Empty -> now.
      Stored as ISO 8601 UTC in ``Document.fetched_at``.
    * ``screenshot_path`` is optional. When it points to an existing file its
      SHA-256 is recorded; a path that does not exist is kept but flagged
      (``meta["screenshot_found"] = False``) rather than silently trusted.
    * Re-submitting identical evidence is idempotent (same ``doc_id``).

    Raises ``ValueError`` for an invalid vendor id (including the V-000 worked
    example), empty text, unknown ``source_class`` or unparsable date/URL.
    """
    validate_vendor_id(vendor_id)
    source_class = normalize_source_class(source_class)
    clean_url = _normalize_url(url)
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        raise ValueError("text is required: paste the exact excerpt from the source")
    if len(body) > MAX_TEXT_CHARS:
        raise ValueError(f"text too long ({len(body)} > {MAX_TEXT_CHARS} characters)")

    now = (clock or utc_now)()
    captured_iso = to_iso_utc(captured_at) if str(captured_at or "").strip() else iso_utc(now)

    shot = str(screenshot_path or "").strip()
    meta: dict = {
        "source_class": source_class,
        "captured_at": captured_iso,
        "url_as_entered": str(url).strip(),
        "screenshot_path": shot,
        "retrieval_method": "analyst_manual_capture",
        "http_status_nominal": True,
        "submitted_at": iso_utc(now),
        "synthetic": False,
        "evidence_caveat": CAVEAT_MANUAL,
        "collector": COLLECTOR_VERSION,
    }
    if shot:
        shot_file = Path(shot)
        if shot_file.is_file():
            meta["screenshot_found"] = True
            meta["screenshot_sha256"] = sha256_hex(shot_file.read_bytes())
        else:
            meta["screenshot_found"] = False
            log.warning("manual evidence screenshot not found: %s", shot)

    content_hash = sha256_hex(body)
    doc_id = make_doc_id(vendor_id, "MANUAL", clean_url, content_hash)
    root = Path(cache_dir) if cache_dir else cache_root()

    snapshot_rel = Path(vendor_id) / "snapshots" / f"manual-{doc_id}.txt"
    snap = root / snapshot_rel
    snap.parent.mkdir(parents=True, exist_ok=True)
    tmp = snap.with_suffix(".txt.tmp")
    tmp.write_text(body + "\n", encoding="utf-8")
    os.replace(tmp, snap)
    meta["text_snapshot_path"] = snapshot_rel.as_posix()

    host = urlparse(clean_url).hostname or clean_url
    doc = Document(
        doc_id=doc_id, vendor_id=vendor_id, source_type="MANUAL", url=clean_url,
        fetched_at=captured_iso, http_status=MANUAL_NOMINAL_HTTP_STATUS,
        content_hash=content_hash,
        title=f"Manual evidence ({source_class.replace('_', ' ')}): {host}",
        text=body,
        snapshot_path=shot or snapshot_rel.as_posix(),
        meta=meta,
    )
    store.upsert_document(doc, cache_dir=root)
    return doc


def list_manual_evidence(vendor_id: str, cache_dir: Optional[Union[str, Path]] = None) -> list[Document]:
    """All persisted MANUAL documents of a vendor, in a stable order."""
    validate_vendor_id(vendor_id)
    docs = store.read_documents(vendor_id, ["MANUAL"], cache_dir=cache_dir)
    return sorted(docs, key=lambda d: (d.fetched_at, d.doc_id))


def remove_manual_evidence(vendor_id: str, doc_id: str,
                           cache_dir: Optional[Union[str, Path]] = None) -> bool:
    """Delete a mistaken manual entry. Only MANUAL documents can be removed."""
    return store.remove_document(vendor_id, doc_id, "MANUAL", cache_dir=cache_dir)
