"""CAREERS collector (plan 5/M3 Day 2, MUST).

Flow
----
1. Fetch the vendor homepage and look for a careers page: homepage links first,
   then common paths (``/careers`` ...). A missing careers page is a finding
   (``EMPTY``), not a failure.
2. Detect the applicant tracking system (ATS) from URL patterns. For Greenhouse,
   Lever and Ashby the vendor's own *public job-board JSON feed* is used (the
   same documented endpoint the careers page widget calls). Other ATS products
   (Workday, iCIMS, Taleo ...) are detected and reported, but their unofficial
   or JavaScript-only endpoints are not scraped.
3. If no feed is available, individual posting pages linked from the careers
   page are fetched and parsed (schema.org ``JobPosting`` JSON-LD first, then
   the page's main content).
4. Technical / data roles are kept; one ``CAREERS`` Document per posting.

All HTTP goes through the shared :class:`~radar.m3_sources.fetching.Fetcher`
(robots.txt, rate limits, snapshots, hashes). Login-gated hosts such as
LinkedIn are never requested.

A posting that mentions AI shows *capability and intent*; each Document carries
that caveat in ``meta`` so it is never mistaken for deployment evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote, urlparse

from radar.contracts.models import DepthPlan, Document, SourceReport, VendorProfile
from radar.m3_sources.common import (
    BLOCKED, CAVEAT_CAREERS, COLLECTOR_VERSION, EMPTY, ERROR, OK, AiTermMatcher, Clock,
    NoNewTermsTracker, PageBudget, host_in, host_of, make_doc_id, make_report,
    normalize_website, same_site, sha256_hex, to_iso_utc, utc_now,
)
from radar.m3_sources.fetching import FetchResult, Fetcher
from radar.m3_sources.textutil import (
    Link, extract_embedded_urls, extract_jobposting_jsonld, extract_links, html_title,
    html_to_text, main_content_text, normalize_ws, unescape_html_fragment,
)

SOURCE = "CAREERS"

# --------------------------------------------------------------------------- #
# ATS detection
# --------------------------------------------------------------------------- #
_ATS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("greenhouse", re.compile(r"https?://(?:boards|job-boards)(?P<eu>\.eu)?\.greenhouse\.io/embed/job_board(?:/js)?\?(?:[^#\s\"']*&)?for=(?P<token>[\w\-]+)", re.I)),
    ("greenhouse", re.compile(r"https?://boards-api\.greenhouse\.io/v1/boards/(?P<token>[\w\-]+)", re.I)),
    ("greenhouse", re.compile(r"https?://(?:boards|job-boards)(?P<eu>\.eu)?\.greenhouse\.io/(?!embed)(?P<token>[\w\-]+)", re.I)),
    ("lever", re.compile(r"https?://jobs(?P<eu>\.eu)?\.lever\.co/(?P<token>[\w\-.]+)", re.I)),
    ("ashby", re.compile(r"https?://jobs\.ashbyhq\.com/(?P<token>[\w\-.%]+)", re.I)),
    ("workable", re.compile(r"https?://apply\.workable\.com/(?P<token>[\w\-]+)", re.I)),
    ("smartrecruiters", re.compile(r"https?://(?:careers|jobs)\.smartrecruiters\.com/(?P<token>[\w\-]+)", re.I)),
    ("workday", re.compile(r"https?://(?P<token>[\w\-]+\.wd\d+\.myworkdayjobs\.com)", re.I)),
    ("icims", re.compile(r"https?://(?P<token>[\w\-]+\.icims\.com)", re.I)),
    ("taleo", re.compile(r"https?://(?P<token>[\w\-]+\.taleo\.net)", re.I)),
    ("oracle_recruiting", re.compile(r"https?://(?P<token>[\w\-]+\.oraclecloud\.com)/hcmUI/CandidateExperience", re.I)),
    ("successfactors", re.compile(r"https?://(?P<token>[\w\-.]+\.successfactors\.(?:com|eu))", re.I)),
    ("jobvite", re.compile(r"https?://jobs\.jobvite\.com/(?P<token>[\w\-]+)", re.I)),
]
_BAD_TOKENS = {"embed", "jobs", "api", "v1", "assets", "static", "js", "css"}


@dataclass(frozen=True)
class AtsRef:
    ats: str
    token: str
    eu: bool = False


def detect_ats(urls: list[str]) -> list[AtsRef]:
    """Distinct ATS references found in ``urls`` in order of first appearance."""
    found: dict[tuple[str, str], AtsRef] = {}
    for url in urls:
        for ats, rx in _ATS_RULES:
            m = rx.search(url)
            if not m:
                continue
            token = m.group("token")
            if token.lower() in _BAD_TOKENS:
                continue
            eu = bool(m.groupdict().get("eu"))
            found.setdefault((ats, token.lower()), AtsRef(ats, token, eu))
            break
    return list(found.values())


def feed_url(ref: AtsRef) -> Optional[str]:
    if ref.ats == "greenhouse" and not ref.eu:
        return f"https://boards-api.greenhouse.io/v1/boards/{ref.token}/jobs?content=true"
    if ref.ats == "lever":
        host = "api.eu.lever.co" if ref.eu else "api.lever.co"
        return f"https://{host}/v0/postings/{ref.token}?mode=json"
    if ref.ats == "ashby":
        return f"https://api.ashbyhq.com/posting-api/job-board/{ref.token}"
    return None


# --------------------------------------------------------------------------- #
# Postings
# --------------------------------------------------------------------------- #
@dataclass
class Posting:
    title: str
    url: str
    text: str
    ats: str = ""
    job_id: str = ""
    department: str = ""
    location: str = ""
    posted_at: str = ""


def _safe_iso(value: Any) -> str:
    try:
        return to_iso_utc(value) if value not in (None, "") else ""
    except (ValueError, OverflowError, OSError):
        return ""


def parse_greenhouse(raw: str) -> list[Posting]:
    data = json.loads(raw)
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("unexpected Greenhouse payload")
    out: list[Posting] = []
    for j in jobs:
        if not isinstance(j, dict) or not j.get("title"):
            continue
        depts = j.get("departments") or []
        loc = j.get("location") or {}
        out.append(Posting(
            title=normalize_ws(str(j["title"])), url=str(j.get("absolute_url") or ""),
            text=unescape_html_fragment(str(j.get("content") or "")), ats="greenhouse",
            job_id=str(j.get("id", "")),
            department=str(depts[0].get("name", "")) if depts and isinstance(depts[0], dict) else "",
            location=str(loc.get("name", "")) if isinstance(loc, dict) else "",
            posted_at=_safe_iso(j.get("first_published") or j.get("updated_at")),
        ))
    return out


def parse_lever(raw: str) -> list[Posting]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("unexpected Lever payload")
    out: list[Posting] = []
    for j in data:
        if not isinstance(j, dict) or not j.get("text"):
            continue
        cats = j.get("categories") or {}
        parts = [str(j.get("descriptionPlain") or html_to_text(str(j.get("description") or "")))]
        for block in j.get("lists") or []:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
                parts.append(html_to_text(str(block.get("content") or "")))
        parts.append(str(j.get("additionalPlain") or html_to_text(str(j.get("additional") or ""))))
        out.append(Posting(
            title=normalize_ws(str(j["text"])), url=str(j.get("hostedUrl") or ""),
            text=normalize_ws("\n".join(p for p in parts if p)), ats="lever",
            job_id=str(j.get("id", "")),
            department=str(cats.get("department") or cats.get("team") or ""),
            location=str(cats.get("location") or ""),
            posted_at=_safe_iso(j.get("createdAt")),
        ))
    return out


def parse_ashby(raw: str) -> list[Posting]:
    data = json.loads(raw)
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("unexpected Ashby payload")
    out: list[Posting] = []
    for j in jobs:
        if not isinstance(j, dict) or not j.get("title") or j.get("isListed") is False:
            continue
        text = str(j.get("descriptionPlain") or "") or html_to_text(str(j.get("descriptionHtml") or ""))
        out.append(Posting(
            title=normalize_ws(str(j["title"])), url=str(j.get("jobUrl") or j.get("applyUrl") or ""),
            text=normalize_ws(text), ats="ashby", job_id=str(j.get("id", "")),
            department=str(j.get("department") or j.get("team") or ""),
            location=str(j.get("location") or ""),
            posted_at=_safe_iso(j.get("publishedAt")),
        ))
    return out


_FEED_PARSERS = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby}


def parse_posting_page(html: str, url: str) -> Optional[Posting]:
    """Posting from a single page: schema.org JobPosting JSON-LD, else main content."""
    ld = extract_jobposting_jsonld(html)
    if ld:
        title = normalize_ws(str(ld.get("title") or ""))
        desc = unescape_html_fragment(str(ld.get("description") or ""))
        loc = ld.get("jobLocation")
        loc = loc[0] if isinstance(loc, list) and loc else loc
        addr = (loc or {}).get("address", {}) if isinstance(loc, dict) else {}
        location = ", ".join(str(addr.get(k)) for k in ("addressLocality", "addressRegion", "addressCountry")
                             if isinstance(addr, dict) and addr.get(k))
        ident = ld.get("identifier")
        job_id = str(ident.get("value", "")) if isinstance(ident, dict) else str(ident or "")
        if title and desc:
            return Posting(title=title, url=url, text=desc, ats="jsonld", job_id=job_id,
                           department=str(ld.get("occupationalCategory") or ""), location=location,
                           posted_at=_safe_iso(ld.get("datePosted")))
    text = main_content_text(html)
    title = ""
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)
    if m:
        title = normalize_ws(html_to_text(m.group(0)))
    title = title or html_title(html)
    if not title or not text:
        return None
    return Posting(title=title, url=url, text=text, ats="html")


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
def is_technical_role(title: str, department: str, ccfg: dict[str, Any], matcher: AiTermMatcher) -> bool:
    """Keep technical and data roles (plan 5/M3). Titles that mention an AI term
    are always kept; clear non-technical titles are dropped."""
    t = f" {title.lower()} "
    ai_in_title = matcher.has_match(title)
    if not ai_in_title and any(k.lower() in t for k in ccfg.get("exclude_title_keywords", [])):
        return False
    if ai_in_title:
        return True
    if any(k.lower() in t for k in ccfg.get("technical_title_keywords", [])):
        return True
    dept = (department or "").lower()
    return any(k.lower() in dept for k in ccfg.get("technical_department_keywords", []))


_SKIP_SCHEMES = ("mailto:", "tel:", "javascript:")
_LISTING_WORDS = {"search", "all", "results", "openings", "list", "teams", "locations",
                  "categories", "page", "apply", "benefits", "culture", "life", "students",
                  "internships", "faq", "talent-community", "alerts"}
_POSTING_PATH = re.compile(r"/(?:jobs?|careers?|positions?|openings?|opportunit(?:y|ies)|vacanc(?:y|ies)|requisitions?|postings?|roles?)/([^/?#]+)", re.I)
_POSTING_QUERY = re.compile(r"[?&](?:gh_jid|jobid|job_id|jid|reqid|requisitionid)=", re.I)


def looks_like_posting(url: str) -> bool:
    if _POSTING_QUERY.search(url):
        return True
    m = _POSTING_PATH.search(urlparse(url).path)
    if not m:
        return False
    seg = m.group(1).lower()
    return seg not in _LISTING_WORDS and not seg.startswith("page")


def looks_like_listing(url: str) -> bool:
    p = urlparse(url)
    return bool(re.search(r"/(?:jobs?|openings?|positions?|search|opportunities|vacancies)/?$", p.path, re.I)
                or re.search(r"[?&]page=\d+", p.query, re.I))


def rank_careers_links(links: list[Link], site_host: str, ccfg: dict[str, Any]) -> list[str]:
    """Homepage links that look like the careers page, best first."""
    kws = [k.lower() for k in ccfg.get("homepage_link_keywords", [])]
    skip = ccfg.get("skip_hosts", [])
    scored: list[tuple[int, int, str]] = []
    for i, link in enumerate(links):
        if host_in(link.url, skip) or link.url.lower().startswith(_SKIP_SCHEMES):
            continue
        ats_hit = bool(detect_ats([link.url]))
        if not (same_site(link.url, site_host) or ats_hit):
            continue
        text, path = link.text.lower(), urlparse(link.url).path.lower()
        score = 0
        score += 3 if any(k == text.strip() for k in kws) else 0
        score += 2 if any(k in text for k in kws) else 0
        score += 2 if any(k.replace(" ", "-") in path or k.replace(" ", "") in path for k in kws) else 0
        score += 2 if ats_hit else 0
        if score:
            scored.append((-score, i, link.url))
    scored.sort()
    out: list[str] = []
    for _, _, url in scored:
        if url not in out:
            out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
@dataclass
class _Outcome:
    phase: str
    url: str
    outcome: str
    reason: str


@dataclass
class _Run:
    vendor: VendorProfile
    plan: DepthPlan
    fetcher: Fetcher
    ccfg: dict[str, Any]
    budget: PageBudget
    outcomes: list[_Outcome] = field(default_factory=list)

    def get(self, url: str, phase: str, kind: str = "page") -> Optional[FetchResult]:
        """Fetch one URL (counts a page when successful). Returns None when the
        page budget is exhausted or the host is login-gated/skipped."""
        if host_in(url, self.ccfg.get("skip_hosts", [])):
            return None
        if not self.budget.can_spend(1):
            return None
        res = self.fetcher.fetch(url, vendor_id=self.vendor.vendor_id, kind=kind)
        oc = res.outcome()
        self.outcomes.append(_Outcome(phase, url, oc, res.reason))
        if oc == "ok":
            self.budget.spend(1)
        return res


def collect_careers(vendor: VendorProfile, plan: DepthPlan, fetcher: Fetcher, cfg: dict[str, Any],
                    matcher: AiTermMatcher, *, clock: Optional[Clock] = None
                    ) -> tuple[list[Document], SourceReport]:
    vid = vendor.vendor_id
    ccfg = cfg["careers"]
    site = normalize_website(vendor.website)
    if site is None:
        return [], make_report(vid, SOURCE, ERROR, 0,
                               f"vendor website {vendor.website!r} could not be parsed; careers not collected")
    site_host = host_of(site)
    run = _Run(vendor, plan, fetcher, ccfg, PageBudget(plan.max_pages))
    notes: list[str] = []

    # 1. homepage ------------------------------------------------------------
    home = run.get(site + "/", "home") if plan.max_depth >= 0 else None
    home_ok = home is not None and home.outcome() == "ok"
    home_links = extract_links(home.text, home.final_url or site) if home_ok else []
    embedded = extract_embedded_urls(home.text, home.final_url or site) if home_ok else []

    # 2. careers page discovery ---------------------------------------------
    candidates = rank_careers_links(home_links, site_host, ccfg)
    candidates += [site + p for p in ccfg.get("common_paths", [])]
    seen_c: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        key = c.rstrip("/").lower()
        if key not in seen_c:
            seen_c.add(key)
            ordered.append(c)

    careers: Optional[FetchResult] = None
    careers_url = ""
    ats_urls: list[str] = list(embedded)
    probes = 0
    if plan.max_depth >= 1:
        for url in ordered:
            if probes >= int(ccfg.get("max_candidate_probes", 8)):
                break
            refs_here = detect_ats([url])
            if refs_here and feed_url(refs_here[0]) and refs_here[0].ats in ccfg.get("ats_feeds_enabled", []):
                careers_url = url          # the ATS board itself: use its feed, no HTML fetch needed
                ats_urls.append(url)
                break
            probes += 1
            res = run.get(url, "probe")
            if res is not None and res.outcome() == "ok":
                careers, careers_url = res, res.final_url or url
                ats_urls.append(careers_url)
                ats_urls.extend(extract_embedded_urls(res.text, careers_url))
                break

    # 3. ATS detection + public feed ----------------------------------------
    refs = detect_ats(ats_urls)
    ats_labels = sorted({r.ats for r in refs})
    postings: list[Posting] = []
    feed_doc_meta: dict[str, Any] = {}
    feed_used = ""
    feed_depth = 1 if (careers is None and careers_url) else 2
    for ref in refs:
        furl = feed_url(ref)
        if not furl or ref.ats not in ccfg.get("ats_feeds_enabled", []):
            continue
        if plan.max_depth < feed_depth:
            notes.append(f"{ref.ats} feed not read: max_depth {plan.max_depth} < {feed_depth}")
            continue
        res = run.get(furl, "feed", kind="api")
        if res is None or res.outcome() != "ok":
            notes.append(f"{ref.ats} feed for board '{ref.token}' unavailable ({res.reason if res else 'budget'})")
            continue
        try:
            parsed = _FEED_PARSERS[ref.ats](res.text)
        except (ValueError, json.JSONDecodeError) as exc:
            notes.append(f"{ref.ats} feed for board '{ref.token}' not understood ({exc})")
            continue
        postings = parsed
        feed_used = ref.ats
        feed_doc_meta = {"feed_url": furl, "board": ref.token, "fetched_at": res.fetched_at,
                         "http_status": res.status, "content_hash": res.content_hash,
                         "snapshot_path": res.snapshot_path}
        break

    # 4. HTML fallback: posting pages linked from the careers page ------------
    page_postings: dict[str, tuple[Posting, FetchResult]] = {}
    html_listing_note = ""
    if not feed_used and careers is not None and plan.max_depth >= 2:
        tracker = NoNewTermsTracker(cfg["common"].get("stop_after_pages_without_new_terms", 3))
        links = extract_links(careers.text, careers_url)
        listing = [l.url for l in links if looks_like_listing(l.url) and same_site(l.url, site_host)
                   and l.url != careers_url][: int(ccfg.get("max_listing_pages", 3))]
        for lurl in listing:
            lres = run.get(lurl, "listing")
            if lres is not None and lres.outcome() == "ok":
                links.extend(extract_links(lres.text, lres.final_url or lurl))
        cands: list[Link] = []
        seen_u: set[str] = set()
        for l in links:
            if l.url in seen_u or host_in(l.url, ccfg.get("skip_hosts", [])) or not looks_like_posting(l.url):
                continue
            if not (same_site(l.url, site_host) or detect_ats([l.url])):
                continue
            if l.text and not (matcher.has_match(l.text) or is_technical_role(l.text, "", ccfg, matcher)):
                continue                    # clearly non-technical link text: do not spend a page on it
            seen_u.add(l.url)
            cands.append(l)
        cands.sort(key=lambda l: (0 if matcher.has_match(l.text) else 1, l.text.lower(), l.url))
        max_docs = min(int(ccfg.get("max_postings", 40)), plan.max_pages)
        for l in cands:
            if len(page_postings) >= max_docs or not run.budget.can_spend(1):
                break
            res = run.get(l.url, "posting")
            if res is None or res.outcome() != "ok":
                continue
            p = parse_posting_page(res.text, res.final_url or l.url)
            if p is None or len(p.text) < int(ccfg.get("min_posting_text_chars", 120)):
                continue
            page_postings[p.url] = (p, res)
            if tracker.observe(matcher.terms_in(p.title + "\n" + p.text)):
                html_listing_note = f"stopped after {tracker.patience} consecutive postings with no new AI terms"
                break
        postings = [p for p, _ in page_postings.values()]

    # 5. filter, rank, cap ----------------------------------------------------
    total = len(postings)
    unique: dict[str, Posting] = {}
    for p in postings:
        if is_technical_role(p.title, p.department, ccfg, matcher):
            unique.setdefault(p.url or f"{p.ats}:{p.job_id}:{p.title}", p)
    ranked: list[tuple[Posting, list[str]]] = [
        (p, matcher.terms_in(p.title + "\n" + p.text)) for p in unique.values()]
    # AI-mentioning postings first (most distinct terms first), then the rest; stable and deterministic.
    ranked.sort(key=lambda pt: (0 if pt[1] else 1, -len(pt[1]), pt[0].title.lower(), pt[0].url))
    n_tech = len(ranked)
    with_ai = sum(1 for _, terms in ranked if terms)
    max_docs = min(int(ccfg.get("max_postings", 40)), plan.max_pages)
    selected: list[tuple[Posting, list[str]]] = []
    stop_note = html_listing_note
    if feed_used:
        tracker = NoNewTermsTracker(cfg["common"].get("stop_after_pages_without_new_terms", 3))
        for p, terms in ranked:
            if len(selected) >= max_docs:
                stop_note = f"capped at {max_docs} postings (page budget / max_postings)"
                break
            selected.append((p, terms))
            if tracker.observe(terms):
                stop_note = f"stopped after {tracker.patience} consecutive postings with no new AI terms"
                break
    else:
        selected = ranked[:max_docs]

    # 6. documents --------------------------------------------------------------
    docs: list[Document] = []
    for p, terms in selected:
        if feed_used:
            fetched_at, status = feed_doc_meta["fetched_at"], feed_doc_meta["http_status"]
            chash, snap = feed_doc_meta["content_hash"], feed_doc_meta["snapshot_path"]
        else:
            res = page_postings[p.url][1]
            fetched_at, status, chash, snap = res.fetched_at, res.status, res.content_hash, res.snapshot_path
        text = f"{p.title}\n{p.text}".strip()
        meta = {
            "ats": p.ats, "job_id": p.job_id, "department": p.department, "location": p.location,
            "posted_at": p.posted_at, "ai_terms": terms,
            "is_technical_role": True, "evidence_class": "capability_and_intent",
            "evidence_caveat": CAVEAT_CAREERS, "careers_page": careers_url,
            "posting_text_sha256": sha256_hex(text), "synthetic": False,
            "collector": COLLECTOR_VERSION,
        }
        if feed_used:
            meta.update({"feed_url": feed_doc_meta["feed_url"], "board": feed_doc_meta["board"]})
        docs.append(Document(
            doc_id=make_doc_id(vendor.vendor_id, SOURCE, p.url or f"{p.ats}:{p.job_id}", p.job_id, p.title),
            vendor_id=vendor.vendor_id, source_type=SOURCE, url=p.url or careers_url,
            fetched_at=fetched_at, http_status=int(status), content_hash=chash,
            title=p.title, text=text, snapshot_path=snap, meta=meta,
        ))
    docs.sort(key=lambda d: d.doc_id)

    # 7. report -------------------------------------------------------------------
    pages = run.budget.used
    where = f"careers page {careers_url}" if careers_url else "no careers page located"
    ats_txt = f"; ATS detected: {', '.join(ats_labels)}" if ats_labels else ""
    if docs:
        how = (f"public {feed_used} job-board feed" if feed_used else "individual posting pages")
        note = (f"{where}{ats_txt}; read via {how}; {total} postings seen, {n_tech} technical/data roles "
                f"({with_ai} mention AI terms), {len(docs)} documents kept")
        if stop_note:
            note += f"; {stop_note}"
        if notes:
            note += "; " + "; ".join(notes)
        return docs, make_report(vid, SOURCE, OK, pages, note)

    status, reason = _empty_status(run.outcomes, careers_url, bool(feed_used))
    unsupported = [a for a in ats_labels if a not in ccfg.get("ats_feeds_enabled", [])]
    note = f"{reason}{ats_txt}"
    if unsupported and not feed_used:
        note += (f" (no public feed implemented for {', '.join(unsupported)}; "
                 "JavaScript-only listings are not scraped)")
    if feed_used or total:
        note += f"; {total} postings seen, none technical/data roles kept"
    if notes:
        note += "; " + "; ".join(notes)
    return docs, make_report(vid, SOURCE, status, pages, note)


def _empty_status(outcomes: list[_Outcome], careers_url: str, feed_ok: bool) -> tuple[str, str]:
    """Pick EMPTY / BLOCKED / ERROR when no document was produced."""
    content = [o for o in outcomes if o.phase != "home"]
    blocked = [o for o in content if o.outcome == "blocked"]
    errors = [o for o in content if o.outcome == "error"]
    reached = bool(careers_url) or feed_ok
    if not reached:
        probes = [o for o in outcomes if o.phase in ("probe", "home")]
        if any(o.outcome == "blocked" for o in probes):
            who = next(o for o in probes if o.outcome == "blocked")
            return BLOCKED, f"careers page not readable: blocked ({who.reason}) at {who.url}"
        if any(o.outcome == "error" for o in probes):
            who = next(o for o in probes if o.outcome == "error")
            return ERROR, f"careers page not reachable: {who.reason} at {who.url}"
        return EMPTY, "no careers page found (homepage links and common paths returned nothing); a missing page is a finding, not a failure"
    if blocked and not any(o.outcome == "ok" and o.phase in ("feed", "posting") for o in outcomes):
        who = blocked[0]
        return BLOCKED, f"{who.phase} blocked ({who.reason}) at {who.url}"
    if errors and not any(o.outcome == "ok" and o.phase in ("feed", "posting") for o in outcomes):
        who = errors[0]
        return ERROR, f"{who.phase} fetch failed ({who.reason}) at {who.url}"
    return EMPTY, f"careers page found at {careers_url or 'ATS board'} but no postings could be collected"
