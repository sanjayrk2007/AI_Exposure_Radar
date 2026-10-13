"""Automated integration, contract compliance, and regression test suite for Module M5.

Validates:
1. Criticality scoring, dimension weights, and escalation floors (F1, F2, F3) across V-001 to V-006.
2. Tier-driven depth planning (page budgets, crawl depths, source families).
3. 2D Risk Matrix lookup, coverage-aware handling, and aggravators (A1, A2).
4. Narrative render templating without formatting exceptions or missing keys.
5. Strict contract conformance with radar.contracts.models dataclasses.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from radar.contracts.models import (
    CriticalityResult,
    DepthPlan,
    RiskAssessment,
    VendorProfile,
    VendorVerdict,
)
from m5_risk.criticality import assess_criticality
from m5_risk.planner import plan_depth
from m5_risk.risk import assess_risk
from m5_risk.render import (
    render_depth_applied,
    render_how_using,
    render_recommended_action,
    render_risk_rationale,
)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "radar" / "fixtures"


@pytest.fixture
def vendor_fixtures(fixtures_dir: Path) -> dict[str, VendorProfile]:
    path = fixtures_dir / "vendors.json"
    assert path.is_file(), f"Missing fixtures file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {v["vendor_id"]: VendorProfile(**v) for v in data}


# ---------------------------------------------------------------------------
# 1. Criticality Scoring & Floor Rules Verification (Appendix A.4)
# ---------------------------------------------------------------------------

def test_v001_criticality_floor_f1(vendor_fixtures: dict[str, VendorProfile]):
    """V-001 (AutomWorx): Base score 1.10 (Medium), escalated to High via Floor F1."""
    vendor = vendor_fixtures["V-001"]
    res = assess_criticality(vendor)

    assert isinstance(res, CriticalityResult)
    assert res.vendor_id == "V-001"
    assert res.scores["D"] == 2
    assert res.scores["S"] == 1
    assert res.scores["P"] == 1
    assert res.scores["R"] == 1
    assert res.scores["V"] == 0
    assert pytest.approx(res.scores["weighted"], rel=1e-3) == 1.10
    assert "F1" in res.floors_applied
    assert res.tier == "High"
    assert "Floor F1" in res.rationale


def test_v002_criticality_core_banking(vendor_fixtures: dict[str, VendorProfile]):
    """V-002 (Fiserv): Core platform, D=3, S=3, P=3, R=3, V=3 -> 3.00 (Critical)."""
    vendor = vendor_fixtures["V-002"]
    res = assess_criticality(vendor)

    assert res.vendor_id == "V-002"
    assert res.scores["D"] == 3
    assert res.scores["S"] == 3
    assert res.scores["P"] == 3
    assert res.scores["R"] == 3
    assert res.scores["V"] == 3
    assert pytest.approx(res.scores["weighted"], rel=1e-3) == 3.00
    assert "F2" in res.floors_applied
    assert "F3" in res.floors_applied
    assert res.tier == "Critical"


def test_v003_criticality_floor_f3(vendor_fixtures: dict[str, VendorProfile]):
    """V-003 (FSSI): Population-scale PII (S=3, V=3) -> High tier via Floor F3."""
    vendor = vendor_fixtures["V-003"]
    res = assess_criticality(vendor)

    assert res.vendor_id == "V-003"
    assert res.scores["D"] == 1
    assert res.scores["S"] == 3
    assert res.scores["P"] == 1
    assert res.scores["R"] == 2
    assert res.scores["V"] == 3
    assert pytest.approx(res.scores["weighted"], rel=1e-3) == 1.95
    assert "F3" in res.floors_applied
    assert res.tier == "High"


def test_v004_criticality_medium(vendor_fixtures: dict[str, VendorProfile]):
    """V-004 (Terrapin): Wealth analytics, moderate dependency -> Medium tier."""
    vendor = vendor_fixtures["V-004"]
    res = assess_criticality(vendor)

    assert res.vendor_id == "V-004"
    assert res.scores["D"] == 1
    assert res.scores["S"] == 2
    assert res.scores["P"] == 0
    assert res.scores["R"] == 1
    assert res.scores["V"] == 1
    assert pytest.approx(res.scores["weighted"], rel=1e-3) == 1.05
    assert res.floors_applied == []
    assert res.tier == "Medium"


def test_v005_criticality_high(vendor_fixtures: dict[str, VendorProfile]):
    """V-005 (BNY): RTP transmission services -> High tier."""
    vendor = vendor_fixtures["V-005"]
    res = assess_criticality(vendor)

    assert res.vendor_id == "V-005"
    assert res.scores["D"] == 2
    assert res.scores["S"] == 2
    assert res.scores["P"] == 3
    assert res.scores["V"] == 3
    assert res.scores["weighted"] >= 1.8
    assert res.tier == "High"


def test_v006_criticality_floor_f2_critical(vendor_fixtures: dict[str, VendorProfile]):
    """V-006 (The Clearing House): ACH/RTP clearing, D=3 -> Critical tier."""
    vendor = vendor_fixtures["V-006"]
    res = assess_criticality(vendor)

    assert res.vendor_id == "V-006"
    assert res.scores["D"] == 3
    assert res.scores["S"] == 2
    assert res.scores["P"] == 3
    assert res.scores["R"] == 3
    assert res.scores["V"] == 3
    assert pytest.approx(res.scores["weighted"], rel=1e-3) == 2.75
    assert "F2" in res.floors_applied
    assert res.tier == "Critical"


# ---------------------------------------------------------------------------
# 2. Depth Planning by Tier (Appendix A.3)
# ---------------------------------------------------------------------------

def test_depth_planning_spec(vendor_fixtures: dict[str, VendorProfile]):
    """Validate page budgets, crawl depths, and source lists per tier."""
    # Critical plan
    v_crit = vendor_fixtures["V-002"]
    crit_res = assess_criticality(v_crit)
    plan_crit = plan_depth(crit_res, v_crit)
    assert isinstance(plan_crit, DepthPlan)
    assert plan_crit.tier == "Critical"
    assert plan_crit.max_pages == 40
    assert plan_crit.max_depth == 3
    assert set(plan_crit.sources) == {
        "TRUST", "DOCS", "WEB", "CAREERS", "REPOS", "FILINGS", "MANUAL", "FINGERPRINT"
    }

    # High plan
    v_high = vendor_fixtures["V-001"]
    high_res = assess_criticality(v_high)
    plan_high = plan_depth(high_res, v_high)
    assert plan_high.tier == "High"
    assert plan_high.max_pages == 25
    assert plan_high.max_depth == 2
    assert set(plan_high.sources) == {
        "TRUST", "DOCS", "WEB", "CAREERS", "REPOS", "FILINGS"
    }

    # Medium plan
    v_med = vendor_fixtures["V-004"]
    med_res = assess_criticality(v_med)
    plan_med = plan_depth(med_res, v_med)
    assert plan_med.tier == "Medium"
    assert plan_med.max_pages == 12
    assert plan_med.max_depth == 2
    assert set(plan_med.sources) == {"TRUST", "DOCS", "WEB", "CAREERS"}

    # Low plan (synthetic test)
    low_profile = VendorProfile(
        vendor_id="V-999", name="Low Risk Vendor", description="Catering supplies",
        service="Food", category="Facilities", website="www.low.test", process="Cafeteria",
        dependency="Low", data_class="None", data_volume="Not applicable"
    )
    low_res = assess_criticality(low_profile)
    assert low_res.tier == "Low"
    plan_low = plan_depth(low_res, low_profile)
    assert plan_low.tier == "Low"
    assert plan_low.max_pages == 6
    assert plan_low.max_depth == 1
    assert set(plan_low.sources) == {"TRUST", "DOCS"}


# ---------------------------------------------------------------------------
# 3. AI Security Risk Matrix & Aggravator Rules (Appendix B)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tier,strength,ai_usage,expected_base_risk",
    [
        ("Critical", "CONFIRMED", "Yes", "Critical"),
        ("Critical", "STRONG", "Yes", "Critical"),
        ("Critical", "PROBABLE", "Yes", "High"),
        ("Critical", "MARKETING_ONLY", "No", "Medium"),
        ("Critical", "NONE", "Inconclusive", "Medium"),
        ("Critical", "NONE", "No", "None identified"),
        ("High", "CONFIRMED", "Yes", "High"),
        ("High", "PROBABLE", "Yes", "Medium"),
        ("High", "MARKETING_ONLY", "No", "Low"),
        ("High", "NONE", "Inconclusive", "Low"),
        ("High", "NONE", "No", "None identified"),
        ("Medium", "CONFIRMED", "Yes", "Medium"),
        ("Medium", "PROBABLE", "Yes", "Low"),
        ("Medium", "MARKETING_ONLY", "No", "Low"),
        ("Low", "CONFIRMED", "Yes", "Low"),
        ("Low", "NONE", "No", "None identified"),
    ],
)
def test_base_risk_matrix_lookup(
    vendor_fixtures: dict[str, VendorProfile],
    tier: str,
    strength: str,
    ai_usage: str,
    expected_base_risk: str,
):
    """Test 2D risk matrix lookup across tiers and evidence strengths without aggravators."""
    vendor = vendor_fixtures["V-004"]  # Base vendor without automated decision process
    crit = CriticalityResult(
        vendor_id=vendor.vendor_id,
        tier=tier,
        scores={"D": 1, "S": 1, "P": 0, "R": 0, "V": 0},  # S < 2 prevents A1
        floors_applied=[],
        rationale="Base test",
    )
    verdict = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage=ai_usage,
        strength=strength,
        coverage=0.9,
        signal_ids=["SIG-1"],
        providers=["OpenAI"],  # Disclosed provider prevents A1
        use_cases=["general analytics"],  # No automated decision keywords prevents A2
        reasons=[],
    )

    risk = assess_risk(vendor, crit, verdict)
    assert isinstance(risk, RiskAssessment)
    assert risk.risk_class == expected_base_risk


def test_aggravator_a1_undisclosed_provider(vendor_fixtures: dict[str, VendorProfile]):
    """Aggravator A1: Confirmed/Strong AI with undisclosed provider and S >= 2 raises risk by 1 band."""
    vendor = vendor_fixtures["V-003"]  # S=3
    crit = assess_criticality(vendor)  # High tier
    assert crit.scores["S"] >= 2

    # With disclosed provider -> Base risk = High
    v_disclosed = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage="Yes",
        strength="CONFIRMED",
        coverage=0.85,
        providers=["Google Cloud Vertex AI"],
        use_cases=["document classification"],
    )
    risk_disclosed = assess_risk(vendor, crit, v_disclosed)
    assert risk_disclosed.risk_class == "High"

    # With undisclosed provider -> Escalates from High to Critical via A1
    v_undisclosed = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage="Yes",
        strength="CONFIRMED",
        coverage=0.85,
        providers=[],  # Undisclosed!
        use_cases=["document classification"],
    )
    risk_undisclosed = assess_risk(vendor, crit, v_undisclosed)
    assert risk_undisclosed.risk_class == "Critical"
    assert "A1" in risk_undisclosed.rationale


def test_aggravator_a2_automated_decisions(vendor_fixtures: dict[str, VendorProfile]):
    """Aggravator A2: Automated customer-affecting decisions raise risk by 1 band."""
    vendor = vendor_fixtures["V-004"]  # Medium tier
    crit = assess_criticality(vendor)

    # Base Probable -> Low
    v_standard = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage="Yes",
        strength="PROBABLE",
        coverage=0.75,
        providers=["Databricks"],
        use_cases=["reporting"],
    )
    risk_standard = assess_risk(vendor, crit, v_standard)
    assert risk_standard.risk_class == "Low"

    # With automated decision keywords -> Escalates Low to Medium via A2
    v_agentic = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage="Yes",
        strength="PROBABLE",
        coverage=0.75,
        providers=["Databricks"],
        use_cases=["automated customer credit scoring without human handoff"],
    )
    risk_agentic = assess_risk(vendor, crit, v_agentic)
    assert risk_agentic.risk_class == "Medium"
    assert "A2" in risk_agentic.rationale


def test_aggravator_capping_at_critical(vendor_fixtures: dict[str, VendorProfile]):
    """Aggravators applied to an already Critical risk cannot exceed Critical."""
    vendor = vendor_fixtures["V-002"]  # Critical tier
    crit = assess_criticality(vendor)
    verdict = VendorVerdict(
        vendor_id=vendor.vendor_id,
        ai_usage="Yes",
        strength="CONFIRMED",
        coverage=1.0,
        providers=[],  # Triggers A1
        use_cases=["automated agentic decision execution"],  # Triggers A2
    )
    risk = assess_risk(vendor, crit, verdict)
    assert risk.risk_class == "Critical"


# ---------------------------------------------------------------------------
# 4. Template String Rendering Robustness
# ---------------------------------------------------------------------------

def test_template_rendering_integrity(vendor_fixtures: dict[str, VendorProfile]):
    """Verify all template renderers execute without KeyErrors or missing format tokens."""
    for vendor in vendor_fixtures.values():
        crit = assess_criticality(vendor)
        plan = plan_depth(crit, vendor)

        # 1. Depth applied narrative
        plan_dict = {
            "max_pages": plan.max_pages,
            "max_depth": plan.max_depth,
            "sources": plan.sources,
            "stop_rule": plan.stop_rule,
        }
        depth_text = render_depth_applied(
            crit,
            pages_fetched=14,
            sources_consulted="TRUST, DOCS, WEB",
            status_notes="All responsive",
            coverage=0.825,
            plan_dict=plan_dict,
        )
        assert isinstance(depth_text, str)
        assert f"Planned {crit.tier} depth" in depth_text
        assert "0.825" in depth_text

        # 2. Risk rationale narrative
        verdict = VendorVerdict(
            vendor_id=vendor.vendor_id,
            ai_usage="Yes",
            strength="CONFIRMED",
            coverage=0.825,
            providers=["OpenAI"],
            use_cases=["transaction scoring"],
        )
        risk = assess_risk(vendor, crit, verdict)
        assert isinstance(risk.rationale, str)
        assert crit.tier in risk.rationale
        assert risk.risk_class in risk.rationale

        # 3. Action narrative and SLA timeframe
        assert isinstance(risk.recommended_action, str)
        assert "business days" in risk.recommended_action

        # 4. Hedged use-case narrative
        how_text = render_how_using(vendor.service, vendor.process, verdict.use_cases)
        assert isinstance(how_text, str)
        assert "transaction scoring" in how_text
        assert "remain unverified" in how_text
        assert vendor.service not in how_text


def test_action_timeframes():
    """Verify SLA response timeframes match risk tiers (15/30/60/90 days)."""
    assert "15 business days" in render_recommended_action("Critical")
    assert "30 business days" in render_recommended_action("High")
    assert "60 business days" in render_recommended_action("Medium")
    assert "90 business days" in render_recommended_action("Low")
