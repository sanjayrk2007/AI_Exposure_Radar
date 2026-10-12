"""Stand-alone runner for M3 (development / cache freezing; not part of the app).

    # offline demo against the synthetic fixture cache
    RADAR_MODE=cache RADAR_CACHE_DIR=radar/fixtures/m3/cache python -m radar.m3_sources.cli --all

    # freeze the real cache (needs network; for SEC set RADAR_SEC_USER_AGENT)
    RADAR_MODE=live RADAR_SEC_USER_AGENT="Team Name you@example.edu" python -m radar.m3_sources.cli --vendor V-002

Vendors come from fixtures/vendors.json. Tiers default to the plan's DRAFT
sanity-check tiers (appendix A.4: not conclusions); override with --tier. The
real DepthPlan comes from M5's planner in the application.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar.contracts.models import DepthPlan, VendorProfile
from radar.m3_sources.collect import collect_sources
from radar.m3_sources.common import package_root

_DRAFT_TIERS = {"V-001": "High", "V-002": "Critical", "V-003": "High",
                "V-004": "Medium", "V-005": "High", "V-006": "Critical"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", action="append", help="vendor id (repeatable)")
    ap.add_argument("--all", action="store_true", help="all vendors in fixtures/vendors.json")
    ap.add_argument("--tier", help="override tier (Critical/High/Medium/Low)")
    ap.add_argument("--mode", choices=["cache", "live"], help="overrides RADAR_MODE")
    args = ap.parse_args(argv)

    fx = package_root() / "fixtures"
    vendors = {v["vendor_id"]: VendorProfile(**v) for v in json.loads((fx / "vendors.json").read_text("utf-8"))}
    plans = json.loads((fx / "m3" / "depth_plans.json").read_text("utf-8"))
    ids = list(vendors) if args.all else (args.vendor or [])
    if not ids:
        ap.error("give --vendor ID (repeatable) or --all")
    for vid in ids:
        tier = args.tier or _DRAFT_TIERS[vid]
        plan = DepthPlan(vendor_id=vid, **plans[tier])
        docs, reports = collect_sources(vendors[vid], plan, mode=args.mode)
        print(f"\n== {vid} {vendors[vid].name} (tier {tier}): {len(docs)} documents")
        for r in reports:
            print(f"  {r.source_type:<8} {r.status:<16} pages={r.pages_fetched:<3} {r.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
