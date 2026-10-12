"""Test doubles for M3: an in-memory Fetcher and synthetic vendor/plan factories.
All hosts are RFC 2606 reserved (.example/.invalid); no network is ever used."""
from __future__ import annotations

import json
from typing import Any, Callable, Optional, Union

from radar.contracts.models import DepthPlan, VendorProfile
from radar.m3_sources.fetching import FetchResult

SITE = "www.acme-vendor.example"


def vendor(vid: str = "V-002", name: str = "Acme Test Vendor, Inc.", website: str = SITE) -> VendorProfile:
    return VendorProfile(vendor_id=vid, name=name, description="d", service="s", category="c",
                         website=website, process="p", dependency="High", data_class="x", data_volume="y")


def plan(vid: str = "V-002", tier: str = "Critical", sources: Optional[list] = None,
         max_pages: int = 40, max_depth: int = 3) -> DepthPlan:
    if sources is None:
        sources = ["TRUST", "DOCS", "WEB", "CAREERS", "REPOS", "FILINGS", "MANUAL"]
    return DepthPlan(vendor_id=vid, tier=tier, sources=sources, max_pages=max_pages,
                     max_depth=max_depth, stop_rule="test")


def resp(text: Union[str, dict, list] = "", status: int = 200, ctype: str = "text/html",
         headers: Optional[dict] = None, blocked: Optional[str] = None, error: Optional[str] = None,
         truncated: bool = False) -> dict[str, Any]:
    if not isinstance(text, str):
        text, ctype = json.dumps(text), "application/json"
    return dict(text=text, status=status, ctype=ctype, headers=headers or {}, blocked=blocked,
                error=error, truncated=truncated)


class FakeFetcher:
    """Routes: exact URL -> resp(...) dict, or a callable(url, headers) -> resp dict.
    Unknown URLs return 404. Every call is recorded in ``calls``."""

    def __init__(self, routes: Optional[dict[str, Any]] = None) -> None:
        self.routes = dict(routes or {})
        self.calls: list[dict[str, Any]] = []

    def fetch(self, url: str, *, vendor_id: str, kind: str = "page", headers: Optional[dict] = None,
              min_interval: Optional[float] = None, max_bytes: Optional[int] = None) -> FetchResult:
        self.calls.append(dict(url=url, kind=kind, headers=headers or {}, min_interval=min_interval,
                               vendor_id=vendor_id, max_bytes=max_bytes))
        spec = self.routes.get(url)
        if callable(spec):
            spec = spec(url, headers or {})
        if spec is None:
            spec = resp("not found", 404)
        return FetchResult(
            url=url, final_url=spec.get("final_url", url), status=spec["status"], text=spec["text"],
            content=spec["text"].encode(), content_type=spec["ctype"], fetched_at="2026-10-10T08:00:00Z",
            content_hash="h" * 8 + str(abs(hash(url)) % 10**6), snapshot_path=f"{vendor_id}/snapshots/x.html",
            headers={k.lower(): str(v) for k, v in spec["headers"].items()},
            blocked_reason=spec["blocked"], error=spec["error"], truncated=spec["truncated"])

    def urls(self) -> list[str]:
        return [c["url"] for c in self.calls]
