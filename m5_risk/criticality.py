import os
from pathlib import Path
import yaml
from contracts.models import VendorProfile, CriticalityResult

def _load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "rubric.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_templates():
    tmpl_path = Path(__file__).resolve().parent.parent / "config" / "templates.yaml"
    with open(tmpl_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def score_dimension(text: str, keywords_dict: dict) -> int:
    text_lower = (text or "").lower()
    for score in [3, 2, 1]:
        kws = keywords_dict.get(score, [])
        if any(kw in text_lower for kw in kws):
            return score
    return 0

def assess_criticality(vendor: VendorProfile) -> CriticalityResult:
    config = _load_config()
    templates = _load_templates()
    kw = config["criteria_keywords"]

    # 1. Dependency (D)
    d = score_dimension(vendor.dependency, kw["dependency"])

    # 2. Data Sensitivity (S)
    s = score_dimension(vendor.data_class, kw["data_sensitivity"])

    # 3. Payment Flow (P)
    p_text = f"{vendor.service} {vendor.process}"
    p = score_dimension(p_text, kw["payment_flow"])

    # 4. Regulatory Exposure (R)
    r_text = f"{vendor.category} {vendor.description} {vendor.service} {vendor.process}"
    r = score_dimension(r_text, kw["regulatory"])

    # 5. Volume (V)
    v = score_dimension(vendor.data_volume, kw["volume"])

    # Weighted calculation
    weights = config["weights"]
    weighted_score = (
        d * weights["D"] +
        s * weights["S"] +
        p * weights["P"] +
        r * weights["R"] +
        v * weights["V"]
    )

    # Escalation floors (F1, F2, F3)
    floors_applied = []
    text_corpus = f"{vendor.description} {vendor.data_class}".lower()
    if any(k in text_corpus for k in ["privileged", "credential", "job schedule"]):
        floors_applied.append("F1")
    if d == 3:
        floors_applied.append("F2")
    if s == 3 and v == 3:
        floors_applied.append("F3")

    # Base thresholds
    th = config["thresholds"]
    if weighted_score >= th["critical"]:
        tier = "Critical"
    elif weighted_score >= th["high"]:
        tier = "High"
    elif weighted_score >= th["medium"]:
        tier = "Medium"
    else:
        tier = "Low"

    # Escalation floor overrides
    if floors_applied and tier in ["Medium", "Low"]:
        tier = "High"
    if "F2" in floors_applied and weighted_score >= th["critical"]:
        tier = "Critical"

    # Render rationale
    floors_text = "".join(templates["criticality"]["floors"].get(fl, "") for fl in floors_applied)
    rationale = templates["criticality"]["rationale"].format(
        score=weighted_score,
        d=d, s=s, p=p, r=r, v=v,
        floors_text=floors_text,
        dep=vendor.dependency,
        process=vendor.process,
        data_class=vendor.data_class
    )

    return CriticalityResult(
        vendor_id=vendor.vendor_id,
        tier=tier,
        scores={"D": d, "S": s, "P": p, "R": r, "V": v, "weighted": weighted_score},
        floors_applied=floors_applied,
        rationale=rationale
    )