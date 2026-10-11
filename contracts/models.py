from dataclasses import dataclass, field

@dataclass
class VendorProfile:
    vendor_id: str
    name: str
    description: str
    service: str
    category: str
    website: str
    process: str
    dependency: str
    data_class: str
    data_volume: str

@dataclass
class CriticalityResult:
    vendor_id: str
    tier: str
    scores: dict
    floors_applied: list
    rationale: str

@dataclass
class DepthPlan:
    vendor_id: str
    tier: str
    sources: list
    max_pages: int
    max_depth: int
    stop_rule: str

@dataclass
class VendorVerdict:
    vendor_id: str
    ai_usage: str
    strength: str
    coverage: float
    signal_ids: list
    providers: list
    use_cases: list
    reasons: list

@dataclass
class RiskAssessment:
    vendor_id: str
    risk_class: str
    rationale: str
    recommended_action: str