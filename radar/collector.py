
"""Polite, bounded collection of public vendor webpages.

This module stores evidence locally. Collection failures are not negative
evidence of AI usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from radar.contracts.models import Document
from radar.input_inspection import read_vendor_profiles

USER_AGENT = (
    "AIExposureRadarCaseStudy/1.0 "
    "(public-page research; respectful low-volume collection)"
)
RELEVANT_LINK_TERMS = (
    "ai", "artificial intelligence", "machine learning", "technology",
    "innovation", "product", "platform", "solution", "automation",
    "data", "security", "research", "documentation",
)
SKIPPED_EXTENSIONS = (
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)


def normalize_start_url(value: str) -> str:
    """Normalize a vendor website field to an HTTPS homepage URL."""
    value = str(value or "").strip()
    if not value:
        raise ValueError("Vendor website is empty")
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) websites are supported")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid website URL")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Non-standard website ports are not supported")
    # Start at the hostname root, not at an arbitrary path in the workbook.
    return urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))


def _canonical_url(url: str) -> str | None:
    """Return a clean HTTP(S) URL, or None for unsupported links."""
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    if parsed.path.lower().endswith(SKIPPED_EXTENSIONS):
        return None
    # Drop fragments; retain query parameters because they may identify pages.
    return urlunparse((
        parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""
    ))


def _same_hostname(url: str, hostname: str) -> bool:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower().rstrip(".") == hostname.lower().rstrip(".")


def _extract_page(html: str, page_url: str, hostname: str):
    soup = BeautifulSoup(html, "html.parser")

    # Discover links before removing navigation/footer text.
    links: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = _canonical_url(urljoin(page_url, anchor["href"]))
        label = anchor.get_text(" ", strip=True)
        if href and _same_hostname(href, hostname):
            links.append((href, label))

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # Remove non-content elements from the evidence text.
    for node in soup([
        "script", "style", "noscript", "svg", "template",
        "nav", "footer",
    ]):
        node.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body or soup

    block_tags = {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "blockquote", "td", "th",
    }
    blocks = []

    # Extract semantic blocks without nested duplicates.
    for element in content_root.find_all(list(block_tags)):
        if element.find_parent(list(block_tags)) is not None:
            continue

        block_text = re.sub(
            r"\s+", " ", element.get_text(" ", strip=True)
        ).strip()

        if block_text:
            blocks.append(block_text)

    # Capture meaningful text in generic containers that semantic
    # extraction may have missed. Skip containers holding semantic
    # blocks, because their text is already represented above.
    if blocks:
        for element in content_root.find_all(["div", "section"]):
            if element.find(list(block_tags)) is not None:
                continue

            block_text = re.sub(
                r"\s+", " ", element.get_text(" ", strip=True)
            ).strip()

            if block_text and block_text not in blocks:
                blocks.append(block_text)

    # Preserve the existing fallback for pages without semantic blocks.
    if not blocks:
        fallback = re.sub(
            r"\s+", " ", content_root.get_text(" ", strip=True)
        ).strip()
        if fallback:
            blocks.append(fallback)

    return title, "\n".join(blocks), links


def _document_id(vendor_id: str, url: str) -> str:
    suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{vendor_id}-WEB-{suffix}"


def collect_vendor_pages(
    profile: dict,
    output_dir: str | Path,
    *,
    max_pages: int = 4,
    delay_seconds: float = 1.0,
    timeout_seconds: float = 15.0,
    session: requests.Session | None = None,
) -> tuple[list[Document], list[dict]]:
    """Collect homepage plus a few relevant same-host pages for one vendor."""
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    vendor_id = str(profile.get("Vendor ID") or "").strip()
    vendor_name = str(profile.get("Vendor Name") or "").strip()
    if not vendor_id or vendor_id == "V-000":
        raise ValueError("A real vendor ID is required; V-000 is excluded")

    website = profile.get("Public Website")
    start_url = normalize_start_url(str(website or ""))
    hostname = urlparse(start_url).hostname
    assert hostname is not None

    root = Path(output_dir)
    snapshot_dir = root / "documents" / vendor_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    own_session = session is None
    client = session or requests.Session()
    client.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    })

    documents: list[Document] = []
    events: list[dict] = []
    queue = [start_url]
    queued = {start_url}
    visited: set[str] = set()
    last_request_at: float | None = None

    try:
        while queue and len(documents) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if last_request_at is not None:
                wait = delay_seconds - (time.monotonic() - last_request_at)
                if wait > 0:
                    time.sleep(wait)

            try:
                response = client.get(
                    url, timeout=timeout_seconds, allow_redirects=True
                )
                last_request_at = time.monotonic()
            except requests.RequestException as exc:
                events.append({
                    "vendor_id": vendor_id, "url": url,
                    "status": "REQUEST_ERROR", "note": str(exc),
                })
                continue

            final_url = _canonical_url(response.url)
            if not final_url or not _same_hostname(final_url, hostname):
                events.append({
                    "vendor_id": vendor_id, "url": url,
                    "status": "REDIRECT_OUTSIDE_HOST",
                    "note": "Redirect left the supplied hostname; page not stored",
                })
                continue

            content_type = response.headers.get("Content-Type", "").lower()
            if response.status_code != 200:
                events.append({
                    "vendor_id": vendor_id, "url": final_url,
                    "status": "HTTP_ERROR", "http_status": response.status_code,
                    "note": "Non-200 response; page not stored",
                })
                continue
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                events.append({
                    "vendor_id": vendor_id, "url": final_url,
                    "status": "SKIPPED_CONTENT_TYPE",
                    "http_status": response.status_code,
                    "note": content_type or "Missing Content-Type",
                })
                continue

            title, text, links = _extract_page(
                response.text, final_url, hostname
            )
            if not text:
                events.append({
                    "vendor_id": vendor_id, "url": final_url,
                    "status": "EMPTY_TEXT", "http_status": response.status_code,
                    "note": "HTML returned no extractable text",
                })
                continue

            doc_id = _document_id(vendor_id, final_url)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            snapshot_path = snapshot_dir / f"{doc_id}.txt"
            snapshot_path.write_text(text + "\n", encoding="utf-8")

            document = Document(
                doc_id=doc_id,
                vendor_id=vendor_id,
                source_type="WEB",
                url=final_url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                http_status=response.status_code,
                content_hash=content_hash,
                title=title,
                text=text,
                snapshot_path=str(snapshot_path),
                meta={
                    "vendor_name": vendor_name,
                    "hostname": hostname,
                    "content_type": content_type,
                    "collector": "AIExposureRadarCaseStudy/1.0",
                },
            )
            documents.append(document)
            events.append({
                "vendor_id": vendor_id, "url": final_url,
                "doc_id": doc_id, "status": "FETCHED",
                "http_status": response.status_code,
                "characters": len(text),
            })

            if len(documents) + len(visited) <= max_pages * 3:
                candidates = []
                for link_url, label in links:
                    if link_url in queued or link_url in visited:
                        continue
                    relevance_text = f"{label} {urlparse(link_url).path}".lower()
                    if any(term in relevance_text for term in RELEVANT_LINK_TERMS):
                        candidates.append(link_url)
                for candidate in candidates:
                    if candidate not in queued:
                        queue.append(candidate)
                        queued.add(candidate)
    finally:
        if own_session:
            client.close()

    return documents, events


def collect_workbook(
    workbook_path: str | Path,
    output_dir: str | Path = "cache",
    *,
    max_pages: int = 4,
    delay_seconds: float = 1.0,
) -> dict:
    """Collect each real vendor, writing a JSONL manifest and event log."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    profiles = read_vendor_profiles(workbook_path)
    all_documents: list[Document] = []
    all_events: list[dict] = []

    for profile in profiles:
        vendor_id = str(profile.get("Vendor ID") or "").strip()
        if vendor_id == "V-000":
            continue
        try:
            docs, events = collect_vendor_pages(
                profile, root,
                max_pages=max_pages,
                delay_seconds=delay_seconds,
            )
            all_documents.extend(docs)
            all_events.extend(events)
            if not docs and not events:
                all_events.append({
                    "vendor_id": vendor_id,
                    "status": "NO_PAGES_COLLECTED",
                    "note": "No pages collected; this is not evidence of no AI usage",
                })
        except (ValueError, requests.RequestException) as exc:
            all_events.append({
                "vendor_id": vendor_id, "status": "COLLECTION_ERROR",
                "note": str(exc),
            })

    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for document in all_documents:
            record = {
                "doc_id": document.doc_id,
                "vendor_id": document.vendor_id,
                "source_type": document.source_type,
                "url": document.url,
                "fetched_at": document.fetched_at,
                "http_status": document.http_status,
                "content_hash": document.content_hash,
                "title": document.title,
                "text": document.text,
                "snapshot_path": document.snapshot_path,
                "meta": document.meta,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    event_log = root / "collection_events.jsonl"
    with event_log.open("w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary = {
        "vendors_in_input": len(profiles),
        "documents_saved": len(all_documents),
        "events_logged": len(all_events),
        "manifest": str(manifest),
        "event_log": str(event_log),
        "note": "Fetch failures and missing pages are not evidence of no AI usage.",
    }
    (root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a small set of public vendor webpages locally."
    )
    parser.add_argument("--workbook", required=True, help="Path to vendor input XLSX")
    parser.add_argument("--output-dir", default="cache", help="Local output directory")
    parser.add_argument("--max-pages", type=int, default=4, help="Pages per vendor")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()

    summary = collect_workbook(
        args.workbook,
        args.output_dir,
        max_pages=args.max_pages,
        delay_seconds=args.delay,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
