"""``collect_sources`` - the M3 entry point (plan 4.3 signature).

    collect_sources(vendor: VendorProfile, plan: DepthPlan)
        -> tuple[list[Document], list[SourceReport]]

Always returns exactly four SourceReports, in this order: CAREERS, REPOS,
FILINGS, MANUAL. Semantics:

* ``SKIPPED_BY_PLAN``  the DepthPlan does not list the source type.
* ``NOT_APPLICABLE``   the source cannot apply to this vendor (non-filer; no
                       GitHub organisation clearly matching the vendor).
* ``OK`` / ``EMPTY``   collected with / without usable documents.
* ``BLOCKED``          robots.txt, login/403, or API rate limit stopped us.
* ``ERROR``            technical failure, bad configuration or cache miss.

RADAR_MODE=live  collects from the network and writes cache/<vendor_id>/.
RADAR_MODE=cache (default) NEVER touches the network: it replays the cache.
Plan-based skipping is evaluated in both modes. MANUAL documents are analyst
input and are read from the cache in both modes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from radar.contracts.models import DepthPlan, Document, SourceReport, VendorProfile
from radar.m3_sources import careers, edgar, github, store
from radar.m3_sources.common import (
    EMPTY, ERROR, M3_SOURCE_TYPES, NOT_APPLICABLE, OK, SKIPPED_BY_PLAN, build_matcher, cache_root,
    get_mode, load_m3_config, make_report, planned_source_types, validate_vendor_id,
)
from radar.m3_sources.fetching import Fetcher, load_shared_fetcher

log = logging.getLogger(__name__)
_LIVE_TYPES = ("CAREERS", "REPOS", "FILINGS")


def collect_sources(vendor: VendorProfile, plan: DepthPlan, *, mode: Optional[str] = None,
                    fetcher: Optional[Fetcher] = None, cache_dir: Optional[Union[str, Path]] = None,
                    config_dir: Optional[Union[str, Path]] = None, github_token: Optional[str] = None,
                    sec_user_agent: Optional[str] = None
                    ) -> tuple[list[Document], list[SourceReport]]:
    """Collect CAREERS, REPOS, FILINGS and MANUAL evidence for one vendor.

    The keyword-only arguments are optional conveniences for tests and tooling;
    callers following the plan simply use ``collect_sources(vendor, plan)``.
    """
    vid = validate_vendor_id(vendor.vendor_id)
    if plan.vendor_id != vid:
        raise ValueError(f"DepthPlan is for {plan.vendor_id!r}, vendor is {vid!r}")
    run_mode = get_mode(mode)
    root = Path(cache_dir) if cache_dir else cache_root()
    planned = planned_source_types(plan)

    docs: list[Document] = []
    reports: list[SourceReport] = []
    live_ctx: Optional[dict[str, Any]] = None

    for stype in M3_SOURCE_TYPES:
        if stype == "MANUAL":
            d, r = _collect_manual(vid, stype in planned, root)
        elif stype not in planned:
            d, r = [], make_report(vid, stype, SKIPPED_BY_PLAN, 0,
                                   f"{stype} is not in the depth plan for tier {plan.tier}")
        elif run_mode == "cache":
            d, r = _replay(vid, stype, root)
        else:
            try:
                if live_ctx is None:
                    live_ctx = _live_context(fetcher, root, config_dir)
                d, r = _run_live(stype, vendor, plan, live_ctx, github_token, sec_user_agent)
            except Exception as exc:  # noqa: BLE001 - one collector must never take the vendor down
                log.exception("M3 %s collector failed for %s", stype, vid)
                d, r = [], make_report(vid, stype, ERROR, 0, f"unexpected {type(exc).__name__}: {exc}")
            store.write_source_results(vid, stype, d, r, cache_dir=root)
        docs.extend(d)
        reports.append(r)
    return docs, reports


# --------------------------------------------------------------------------- #
def _live_context(fetcher: Optional[Fetcher], root: Path, config_dir: Optional[Union[str, Path]]) -> dict[str, Any]:
    cfg = load_m3_config(config_dir)
    matcher, lexicon = build_matcher(cfg, config_dir)
    fx = fetcher or load_shared_fetcher(cfg["fetch"], cache_dir=root)
    return {"cfg": cfg, "matcher": matcher, "lexicon": lexicon, "fetcher": fx}


def _run_live(stype: str, vendor: VendorProfile, plan: DepthPlan, ctx: dict[str, Any],
              github_token: Optional[str], sec_ua: Optional[str]) -> tuple[list[Document], SourceReport]:
    cfg, fx = ctx["cfg"], ctx["fetcher"]
    if stype == "CAREERS":
        return careers.collect_careers(vendor, plan, fx, cfg, ctx["matcher"])
    if stype == "REPOS":
        token = github_token if github_token is not None else os.environ.get("GITHUB_TOKEN") or None
        return github.collect_repos(vendor, plan, fx, cfg, token=token)
    ua = sec_ua if sec_ua is not None else os.environ.get(cfg["edgar"].get("user_agent_env", "RADAR_SEC_USER_AGENT"))
    return edgar.collect_filings(vendor, plan, fx, cfg, ctx["matcher"], user_agent=ua)


def _replay(vid: str, stype: str, root: Path) -> tuple[list[Document], SourceReport]:
    """Cache mode: return exactly what a previous live run stored; no network."""
    docs = sorted(store.read_documents(vid, [stype], cache_dir=root), key=lambda d: d.doc_id)
    reps = store.read_reports(vid, [stype], cache_dir=root)
    if reps:
        rep = reps[-1]
        if rep.status == OK and not docs:
            return [], make_report(vid, stype, ERROR, rep.pages_fetched,
                                   "cache inconsistent: report says OK but no documents are cached; re-run in live mode")
        return docs, make_report(vid, stype, rep.status, rep.pages_fetched, f"{rep.note} [from cache]")
    if docs:
        return docs, make_report(vid, stype, OK, len(docs),
                                 f"{len(docs)} cached documents (no cached report; reconstructed) [from cache]")
    return [], make_report(vid, stype, ERROR, 0,
                           f"cache miss: nothing cached for {vid}/{stype}; run once with RADAR_MODE=live to populate cache/")


def _collect_manual(vid: str, planned: bool, root: Path) -> tuple[list[Document], SourceReport]:
    docs = sorted(store.read_documents(vid, ["MANUAL"], cache_dir=root), key=lambda d: (d.fetched_at, d.doc_id))
    if docs:
        n_primary = sum(1 for d in docs if d.meta.get("source_class") == "primary")
        note = f"{len(docs)} analyst-submitted items ({n_primary} primary, {len(docs) - n_primary} third party)"
        if not planned:
            note += "; included although MANUAL is not in the depth plan, because an analyst submitted it"
        return docs, make_report(vid, "MANUAL", OK, len(docs), note)
    if not planned:
        return [], make_report(vid, "MANUAL", SKIPPED_BY_PLAN, 0, "MANUAL is not in the depth plan and nothing was submitted")
    return [], make_report(vid, "MANUAL", EMPTY, 0, "no analyst evidence submitted for this vendor yet")
