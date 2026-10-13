import os
from pathlib import Path
import yaml
try:
    from radar.contracts.models import CriticalityResult, VendorVerdict
except ImportError:
    from contracts.models import CriticalityResult, VendorVerdict

def _load_templates():
    tmpl_path = Path(__file__).resolve().parent.parent / "config" / "templates.yaml"
    with open(tmpl_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_risk_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "risk_matrix.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def render_risk_rationale(c: CriticalityResult, v: VendorVerdict, risk_class: str, aggravators: list) -> str:
    tmpl = _load_templates()
    aggr_str = f" Aggravators applied: {', '.join(aggravators)}." if aggravators else ""
    return tmpl["risk"]["rationale"].format(
        tier=c.tier,
        data_class=c.scores.get("data_class", "vendor data"),
        dep=c.scores.get("dep", "operational systems"),
        strength=v.strength or "NONE",
        coverage=v.coverage or 0.0,
        ai_usage=v.ai_usage or "Inconclusive",
        risk_class=risk_class,
        aggravators_text=aggr_str
    )

def render_recommended_action(risk_class: str) -> str:
    tmpl = _load_templates()
    risk_cfg = _load_risk_config()
    timeframes = risk_cfg["timeframes"]
    tf = timeframes.get(risk_class, "30")
    return tmpl["action"]["recommended"].format(timeframe=tf)

def render_how_using(service: str, process: str, use_cases: list) -> str:
    """Render readable, cautious AI-use context without exposing detector labels."""
    descriptions = {
        "fraud detection": "fraud detection",
        "transaction scoring": "transaction scoring",
        "document classification": "document classification",
        "chatbot": "chatbot functionality",
        "virtual assistant": "virtual-assistant functionality",
        "summarisation": "summarisation",
        "summarization": "summarisation",
        "auto drafting": "automated drafting",
        "forecasting": "forecasting",
    }

    readable = []
    for item in use_cases or []:
        key = " ".join(
            str(item).strip().lower().replace("_", " ").replace("-", " ").split()
        )
        description = descriptions.get(key)
        if description and description not in readable:
            readable.append(description)

    if not readable:
        return (
            "The collected public evidence does not establish a specific AI "
            "application for this vendor. Further confirmation is required."
        )

    service_text = (
        service.strip() if service and service.strip()
        else "the vendor's services"
    )
    process_text = (
        process.strip() if process and process.strip()
        else "the relevant business process"
    )
    applications = ", ".join(readable)

    return (
        f"Public evidence identifies AI-related indicators associated with "
        f"{applications} in the context of {service_text}, which supports "
        f"{process_text}. This does not by itself confirm production deployment."
    )

def render_depth_applied(c: CriticalityResult, pages_fetched: int, sources_consulted: str, status_notes: str, coverage: float, plan_dict: dict) -> str:
    tmpl = _load_templates()
    return tmpl["depth"]["applied"].format(
        tier=c.tier,
        max_pages=plan_dict.get("max_pages", 25),
        max_depth=plan_dict.get("max_depth", 2),
        sources=", ".join(plan_dict.get("sources", [])),
        sources_consulted=sources_consulted,
        pages_fetched=pages_fetched,
        coverage=coverage,
        status_notes=status_notes,
        stop_rule=plan_dict.get("stop_rule", "Budget reached")
    )