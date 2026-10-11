"""Shared data contracts used by the AI Exposure Radar modules.

The fields follow the contracts in AI_Exposure_Radar_Final_Plan.md. This file
was absent from the Phase 0 workspace, so the minimal Document and Signal
contracts are introduced here rather than changing an existing implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    scores: dict[str, Any] = field(default_factory=dict)
    floors_applied: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DepthPlan:
    vendor_id: str
    tier: str
    sources: list[Any] = field(default_factory=list)
    max_pages: int = 0
    max_depth: int = 0
    stop_rule: str = ""


@dataclass
class Document:
    doc_id: str
    vendor_id: str
    source_type: str
    url: str
    fetched_at: str
    http_status: int
    content_hash: str
    title: str
    text: str
    snapshot_path: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    signal_id: str
    vendor_id: str
    doc_id: str
    source_type: str
    url: str
    excerpt: str
    char_start: int
    char_end: int
    category: str
    providers: list[str]
    specificity: float
    weight: float
    score: float
    tag: str
    verified: bool
    method: str


@dataclass
class SourceReport:
    vendor_id: str
    source_type: str
    status: str
    pages_fetched: int = 0
    note: str = ""


@dataclass
class VendorVerdict:
    vendor_id: str
    ai_usage: str
    strength: str
    coverage: float
    signal_ids: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    vendor_id: str
    risk_class: str
    rationale: str
    recommended_action: str

