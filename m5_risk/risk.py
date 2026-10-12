import os
from pathlib import Path
import yaml
try:
    from radar.contracts.models import VendorProfile, CriticalityResult, VendorVerdict, RiskAssessment
except ImportError:
    from contracts.models import VendorProfile, CriticalityResult, VendorVerdict, RiskAssessment
from m5_risk.render import render_risk_rationale, render_recommended_action

def _load_risk_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "risk_matrix.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def assess_risk(vendor: VendorProfile, c: CriticalityResult, v: VendorVerdict) -> RiskAssessment:
    config = _load_risk_config()
    matrix = config["base_matrix"]
    bands_order = config["bands_order"]

    tier_row = matrix.get(c.tier, matrix["Low"])
    
    strength_key = (v.strength or "").upper()
    if "MARKETING" in strength_key:
        lookup_key = "MARKETING_ONLY"
    elif strength_key in ["CONFIRMED", "STRONG", "PROBABLE"]:
        lookup_key = strength_key
    elif v.ai_usage == "Inconclusive":
        lookup_key = "INCONCLUSIVE"
    else:
        lookup_key = "NONE"

    base_risk = tier_row.get(lookup_key, "Low")
    idx = bands_order.index(base_risk) if base_risk in bands_order else 1

    aggravators = []
    s_score = c.scores.get("S", 0)

    # Aggravator A1: AI confirmed/strong, provider not named, and data sensitivity S >= 2
    if lookup_key in ["CONFIRMED", "STRONG"] and (not v.providers or "none" in [str(p).lower() for p in v.providers]) and s_score >= 2:
        aggravators.append("A1 (undisclosed AI provider with high data sensitivity)")
        idx = min(len(bands_order) - 1, idx + 1)

    # Aggravator A2: AI used for automated decisions or agent-style actions affecting customers
    proc_corpus = f"{vendor.process} {' '.join(v.use_cases)}".lower()
    if any(k in proc_corpus for k in ["decision", "automated", "agent", "scoring", "without human handoff"]):
        aggravators.append("A2 (automated decisions or agentic workflows)")
        idx = min(len(bands_order) - 1, idx + 1)

    final_risk = bands_order[idx]

    # Stash contextual strings into scores dict for render template
    c.scores["data_class"] = vendor.data_class
    c.scores["dep"] = vendor.dependency

    rationale = render_risk_rationale(c, v, final_risk, aggravators)
    action = render_recommended_action(final_risk)

    return RiskAssessment(
        vendor_id=vendor.vendor_id,
        risk_class=final_risk,
        rationale=rationale,
        recommended_action=action
    )