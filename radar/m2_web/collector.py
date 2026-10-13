
import hashlib
import uuid
from urllib.parse import urlparse

from radar.contracts.models import Document, SourceReport
from radar.m2_web.discover import discover_urls
from radar.m2_web.fetcher import fetch_url
from radar.m2_web.parse import parse_page


def collect_web(vendor, plan):
    documents = []
    sources = list(dict.fromkeys(plan.sources))

    if not sources:
        return documents, [
            SourceReport(
                vendor_id=vendor.vendor_id,
                source_type="WEB",
                status="SKIPPED_BY_PLAN",
                pages_fetched=0,
                note="No sources were planned",
            )
        ]

    if not vendor.website.strip():
        return documents, [
            SourceReport(
                vendor_id=vendor.vendor_id,
                source_type=source,
                status="ERROR",
                pages_fetched=0,
                note="Vendor website is missing",
            )
            for source in sources
        ]

    counts = {source: 0 for source in sources}
    blocked = {source: 0 for source in sources}
    errors = {source: 0 for source in sources}
    unreadable = {source: 0 for source in sources}
    seen_urls = set()

    try:
        urls = discover_urls(
            vendor.website,
            max_pages=max(1, int(plan.max_pages)),
            max_depth=max(0, int(plan.max_depth)),
        )
    except Exception as exc:
        return documents, [
            SourceReport(
                vendor_id=vendor.vendor_id,
                source_type=source,
                status="ERROR",
                pages_fetched=0,
                note=f"Discovery failed: {exc}",
            )
            for source in sources
        ]

    page_budget = max(1, int(plan.max_pages))

    for url, depth, discovered_type in urls[:page_budget]:
        if url in seen_urls:
            continue
        seen_urls.add(url)

        if discovered_type in counts:
            source_type = discovered_type
        elif "WEB" in counts:
            source_type = "WEB"
        else:
            source_type = sources[0]

        try:
            result = fetch_url(url)
        except Exception:
            errors[source_type] += 1
            continue

        if result.get("blocked"):
            blocked[source_type] += 1
            continue

        if result.get("status") != 200:
            errors[source_type] += 1
            continue

        html = result.get("html", "")
        pdf_bytes = result.get("pdf_bytes")

        if not html and not pdf_bytes:
            errors[source_type] += 1
            continue

        try:
            title, text = parse_page(
                url,
                html=html,
                pdf_bytes=pdf_bytes,
            )
        except Exception:
            errors[source_type] += 1
            continue

        if not text.strip():
            unreadable[source_type] += 1
            continue

        digest = result.get("content_hash")
        if not digest:
            content = pdf_bytes if pdf_bytes else html.encode("utf-8")
            digest = hashlib.sha256(content).hexdigest()

        documents.append(
            Document(
                doc_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{vendor.vendor_id}:{url}:{digest}",
                    )
                ),
                vendor_id=vendor.vendor_id,
                source_type=source_type,
                url=url,
                fetched_at=result.get("fetched_at", ""),
                http_status=result["status"],
                content_hash=digest,
                title=title,
                text=text,
                snapshot_path=result.get("snapshot_path", ""),
                meta={
                    "depth": depth,
                    "content_type": result.get("content_type", ""),
                    "from_cache": result.get("from_cache", False),
                },
            )
        )
        counts[source_type] += 1

    reports = []

    for source in sources:
        if counts[source]:
            status = "OK"
            note = "Readable pages collected"
        elif blocked[source]:
            status = "BLOCKED"
            note = "Pages were disallowed by robots policy"
        elif errors[source]:
            status = "ERROR"
            note = "Pages could not be fetched or parsed"
        elif unreadable[source]:
            status = "EMPTY"
            note = "Pages fetched, but no readable text was extracted"
        else:
            status = "EMPTY"
            note = "No readable pages found for this source"

        reports.append(
            SourceReport(
                vendor_id=vendor.vendor_id,
                source_type=source,
                status=status,
                pages_fetched=counts[source],
                note=note,
            )
        )

    return documents, reports
