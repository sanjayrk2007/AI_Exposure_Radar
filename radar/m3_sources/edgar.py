"""FILINGS collector (plan 5/M3 Day 3-4, SHOULD).

For vendors that file with the SEC: fetch the latest annual report (10-K, or
20-F/40-F for foreign issuers) and return keyword-in-context chunks (two
sentences either side of AI terms) as FILINGS documents carrying the filing URL.
Non-filers are ``NOT_APPLICABLE``.

SEC access rules (https://www.sec.gov/os/accessing-edgar-data, checked when this
was written): every request must declare a User-Agent with contact details
(here: environment variable ``RADAR_SEC_USER_AGENT``, e.g.
``"Team Kestrel jane.doe@example.edu"``) and stay below 10 requests/second; this
collector uses ~3/s and makes only 2-3 requests per vendor. Without the variable
the collector makes NO request and reports ERROR.

Statements in filings are corporate disclosures, often generic risk-factor
language: each Document carries a caveat so it is not read as proof that AI is
used in a specific service for Meridian.
"""
from __future__ import annotations

import bisect
import json
import re
from typing import Any, Optional

from radar.contracts.models import DepthPlan, Document, SourceReport, VendorProfile
from radar.m3_sources.common import (
    BLOCKED, CAVEAT_FILINGS, COLLECTOR_VERSION, EMPTY, ERROR, NOT_APPLICABLE, OK,
    AiTermMatcher, make_doc_id, make_report, normalize_company_name, sha256_hex, to_iso_utc,
)
from radar.m3_sources.fetching import FetchResult, Fetcher
from radar.m3_sources.textutil import html_to_text, keyword_in_context, reflow_plain_text

SOURCE = "FILINGS"
_ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[ab]?)\s*[.:\u2014\u2013\-]?\s*(.{0,80})$")


def _fail(res: Optional[FetchResult], what: str, vid: str, pages: int) -> SourceReport:
    if res is not None and res.outcome() == "blocked":
        return make_report(vid, SOURCE, BLOCKED, pages, f"{what}: blocked by SEC ({res.reason})")
    why = res.reason if res is not None else "no response"
    return make_report(vid, SOURCE, ERROR, pages, f"{what} failed: {why}")


def _latest_annual(submissions: dict[str, Any], forms: list[str]) -> Optional[dict[str, str]]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    n = len(recent.get("form", []))
    best: Optional[dict[str, str]] = None
    for i in range(n):
        form = str(recent["form"][i])
        if form not in forms:       # amendments (10-K/A) are deliberately not used
            continue
        row = {
            "form": form,
            "accession": str(recent["accessionNumber"][i]),
            "primary": str(recent["primaryDocument"][i]),
            "filed": str(recent["filingDate"][i]),
            "report_date": str((recent.get("reportDate") or [""] * n)[i] or ""),
        }
        if best is None or row["filed"] > best["filed"]:
            best = row
    return best


def _resolve_by_name(vendor: VendorProfile, tickers: Any, suffixes: list[str]) -> Optional[tuple[int, str]]:
    wanted = normalize_company_name(vendor.name, suffixes)
    if not wanted or not isinstance(tickers, dict):
        return None
    ciks: dict[int, str] = {}
    for row in tickers.values():
        if isinstance(row, dict) and normalize_company_name(str(row.get("title", "")), suffixes) == wanted:
            ciks[int(row["cik_str"])] = str(row.get("title"))
    if len(ciks) == 1:
        cik, title = next(iter(ciks.items()))
        return cik, title
    return None


def collect_filings(vendor: VendorProfile, plan: DepthPlan, fetcher: Fetcher, cfg: dict[str, Any],
                    matcher: AiTermMatcher, *, user_agent: Optional[str] = None
                    ) -> tuple[list[Document], SourceReport]:
    vid = vendor.vendor_id
    ecfg = cfg["edgar"]
    suffixes = ecfg.get("legal_suffixes", [])
    if not user_agent or "@" not in user_agent:
        return [], make_report(
            vid, SOURCE, ERROR, 0,
            f"SEC fair-access policy requires a declared User-Agent with contact email: set "
            f"{ecfg.get('user_agent_env', 'RADAR_SEC_USER_AGENT')} (e.g. 'Team Name name@example.edu'); no SEC request was made")
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    interval = float(ecfg.get("min_interval_seconds", 0.3))
    pages = 0

    # 1. who is the filer? --------------------------------------------------------
    known = (ecfg.get("known_filers") or {}).get(vid)
    hints: list[str]
    if known:
        cik, hints, how = int(known["cik"]), [h.lower() for h in known.get("name_hints", [])], "configured filer (verified against EDGAR entity name)"
    else:
        res = fetcher.fetch(ecfg["tickers_url"], vendor_id=vid, kind="api", headers=headers, min_interval=interval)
        if res.outcome() != "ok":
            return [], _fail(res, "SEC company list", vid, pages)
        pages += 1
        try:
            match = _resolve_by_name(vendor, json.loads(res.text), suffixes)
        except (json.JSONDecodeError, KeyError, ValueError):
            return [], make_report(vid, SOURCE, ERROR, pages, "SEC company list could not be parsed")
        if match is None:
            return [], make_report(
                vid, SOURCE, NOT_APPLICABLE, pages,
                "no SEC registrant with this exact company name in EDGAR's company list; treated as a non-filer "
                "(absence from the list is not proof the vendor never files)")
        cik, title = match
        hints, how = [normalize_company_name(title, suffixes)], f"exact name match with EDGAR company '{title}'"

    # 2. submissions -> latest annual report --------------------------------------------
    res = fetcher.fetch(ecfg["submissions_url"].format(cik10=f"{cik:010d}"), vendor_id=vid, kind="api",
                        headers=headers, min_interval=interval)
    if res.outcome() != "ok":
        return [], _fail(res, f"SEC submissions for CIK {cik}", vid, pages)
    pages += 1
    try:
        sub = json.loads(res.text)
    except json.JSONDecodeError:
        return [], make_report(vid, SOURCE, ERROR, pages, f"SEC submissions for CIK {cik} were not valid JSON")
    entity = str(sub.get("name") or "")
    if not any(h and h in entity.lower() for h in hints):
        return [], make_report(
            vid, SOURCE, ERROR, pages,
            f"entity mismatch: CIK {cik} is '{entity}', which does not match expected name hints {hints}; not used")
    filing = _latest_annual(sub, list(ecfg.get("annual_forms", ["10-K"])))
    if filing is None:
        return [], make_report(vid, SOURCE, EMPTY, pages,
                               f"{entity} (CIK {cik}) has no annual report ({', '.join(ecfg.get('annual_forms', []))}) in recent filings")

    # 3. the filing itself -----------------------------------------------------------------
    ext = filing["primary"].rsplit(".", 1)[-1].lower()
    if ext not in ("htm", "html", "xhtml", "txt"):
        return [], make_report(vid, SOURCE, ERROR, pages,
                               f"primary document type '.{ext}' of {filing['accession']} is not supported; not downloaded")
    url = ecfg["archive_url"].format(cik=cik, accession_nodash=filing["accession"].replace("-", ""),
                                     primary_document=filing["primary"])
    res = fetcher.fetch(url, vendor_id=vid, kind="api", headers=headers, min_interval=interval,
                        max_bytes=int(ecfg.get("max_filing_bytes", 40_000_000)))
    if res.outcome() != "ok":
        return [], _fail(res, f"{filing['form']} {filing['accession']}", vid, pages)
    pages += 1
    text = html_to_text(res.text) if ext in ("htm", "html", "xhtml") else reflow_plain_text(res.text)

    # 4. keyword-in-context chunks ---------------------------------------------------------
    chunks = keyword_in_context(text, matcher, window=int(ecfg.get("context_sentences", 2)),
                                max_chunks=int(ecfg.get("max_chunks", 30)),
                                max_chars=int(ecfg.get("max_chunk_chars", 3000)))
    heads = [(m.start(), f"Item {m.group(1).upper()}. {m.group(2).strip()}".strip()) for m in _ITEM_RE.finditer(text)]
    head_pos = [p for p, _ in heads]
    try:
        filed_iso = to_iso_utc(filing["filed"])
    except ValueError:
        filed_iso = filing["filed"]
    docs: list[Document] = []
    for i, c in enumerate(chunks, start=1):
        idx = bisect.bisect_right(head_pos, c.start) - 1
        item = heads[idx][1] if idx >= 0 else ""
        chunk_text = text[c.start:c.end]
        docs.append(Document(
            doc_id=make_doc_id(vid, SOURCE, filing["accession"], str(c.start)),
            vendor_id=vid, source_type=SOURCE, url=url, fetched_at=res.fetched_at,
            http_status=res.status, content_hash=res.content_hash,
            title=f"{entity} {filing['form']} filed {filing['filed']} - excerpt {i} of {len(chunks)}",
            text=chunk_text, snapshot_path=res.snapshot_path,
            meta={
                "entity_name": entity, "cik": cik, "form": filing["form"], "accession": filing["accession"],
                "filing_date": filed_iso, "report_date": filing["report_date"], "primary_document": filing["primary"],
                "char_start": c.start, "char_end": c.end, "filing_text_sha256": sha256_hex(text),
                "chunk_sha256": sha256_hex(chunk_text), "matched_terms": list(c.terms), "item": item,
                "context_sentences": int(ecfg.get("context_sentences", 2)), "filer_resolution": how,
                "evidence_class": "regulatory_disclosure", "evidence_caveat": CAVEAT_FILINGS,
                "synthetic": False, "collector": COLLECTOR_VERSION,
            }))
    base = f"{entity} (CIK {cik}) {filing['form']} filed {filing['filed']}"
    if docs:
        return docs, make_report(vid, SOURCE, OK, pages,
                                 f"{base}; {len(docs)} keyword-in-context excerpts around AI terms; filer: {how}")
    return docs, make_report(vid, SOURCE, EMPTY, pages, f"{base} fetched; it contains none of the AI terms searched")
