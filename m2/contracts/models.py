
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VendorProfile:
    vendor_id: str
    name: str
    description: str = ""
    service: str = ""
    category: str = ""
    website: str = ""
    process: str = ""
    dependency: str = ""
    data_class: str = ""
    data_volume: str = ""


@dataclass
class DepthPlan:
    vendor_id: str
    tier: str = "standard"
    sources: list[str] = field(
        default_factory=lambda: ["TRUST", "DOCS", "WEB"]
    )
    max_pages: int = 20
    max_depth: int = 2
    stop_rule: str = "page_budget"


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
class SourceReport:
    vendor_id: str
    source_type: str
    status: str
    pages_fetched: int
    note: str = ""
