import os
from pathlib import Path
import yaml
try:
    from radar.contracts.models import CriticalityResult, DepthPlan, VendorProfile
except ImportError:
    from contracts.models import CriticalityResult, DepthPlan, VendorProfile

def _load_depth_plans():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "depth_plans.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plan_depth(c: CriticalityResult, vendor: VendorProfile) -> DepthPlan:
    plans = _load_depth_plans()
    tier_plan = plans.get(c.tier, plans["Low"])

    return DepthPlan(
        vendor_id=vendor.vendor_id,
        tier=c.tier,
        sources=tier_plan["sources"],
        max_pages=tier_plan["max_pages"],
        max_depth=tier_plan["max_depth"],
        stop_rule=tier_plan["stop_rule"]
    )