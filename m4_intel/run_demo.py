"""Single command: vendor workbook -> public evidence -> verified signals -> verdicts -> new workbook.

    python -m m4_intel.run_demo --workbook INPUT.xlsx --output OUTPUT.xlsx [--no-llm]

Each vendor is processed independently; a failure for one vendor is recorded and the run continues.
Failed or incomplete collection lowers coverage and never produces a "No" verdict by itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from m4_intel.adjudicate import LocalLLM, make_adjudicator
from m4_intel.analyse import DEFAULT_PLANNED_SOURCES, analyse_detailed
from radar.collector import MIN_DELAY_SECONDS, build_source_reports, collect_vendor_pages
from radar.contracts.models import SourceReport, VendorVerdict
from radar.export import VendorResult, export_workbook
from radar.input_inspection import read_vendor_profiles

DEFAULT_CACHE_DIR = Path("cache") / "demo"


def _failed_result(vendor_id: str, message: str) -> VendorResult:
    verdict = VendorVerdict(
        vendor_id=vendor_id, ai_usage="Inconclusive", strength="None", coverage=0.0,
        reasons=[f"Processing failed: {message}. A failure is not evidence that the vendor does not use AI."],
    )
    return VendorResult(vendor_id=vendor_id, verdict=verdict, error=message)


def _load_cache(path: Path) -> tuple[list[dict[str, Any]], list[SourceReport]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documents"], [SourceReport(**report) for report in data["reports"]]


def _collect(profile: dict, cache_dir: Path, mode: str, max_pages: int, delay: float,
             ) -> tuple[list[dict[str, Any]], list[SourceReport]]:
    vendor_id = str(profile["Vendor ID"])
    cache_file = cache_dir / vendor_id / "collected.json"
    if mode == "cache":
        cached = _load_cache(cache_file)
        if cached is None:
            raise FileNotFoundError(f"No cached collection for {vendor_id}")
        return cached
    documents, events = collect_vendor_pages(
        profile, cache_dir, max_pages=max_pages, delay_seconds=delay, probe_paths=True,
    )
    reports = build_source_reports(vendor_id, documents, events, DEFAULT_PLANNED_SOURCES)
    payload = {
        "documents": [asdict(document) for document in documents],
        "reports": [asdict(report) for report in reports],
        "events": events,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload["documents"], reports


def run(workbook: str | Path, output: str | Path, *, cache_dir: str | Path = DEFAULT_CACHE_DIR,
        mode: str = "live", max_pages: int = 8, delay: float = MIN_DELAY_SECONDS,
        use_llm: bool = True, llm_url: str | None = None, llm_model: str | None = None,
        vendors: set[str] | None = None) -> dict[str, Any]:
    if mode not in {"live", "cache"}:
        raise ValueError("mode must be 'live' or 'cache'")
    if delay < MIN_DELAY_SECONDS:
        raise ValueError(f"delay must be at least {MIN_DELAY_SECONDS} seconds")
    cache_root = Path(cache_dir)
    llm = LocalLLM(llm_url, llm_model, cache_root / "llm") if use_llm else None
    adjudicator = make_adjudicator(llm) if llm else None

    results: dict[str, VendorResult] = {}
    lines: list[dict[str, Any]] = []
    for profile in read_vendor_profiles(workbook):
        vendor_id = str(profile.get("Vendor ID") or "").strip()
        if not vendor_id or vendor_id == "V-000" or (vendors and vendor_id not in vendors):
            continue
        try:
            documents, reports = _collect(profile, cache_root, mode, max_pages, delay)
            outcome = analyse_detailed(
                {"vendor_id": vendor_id, "Public Website": profile.get("Public Website")},
                documents, reports, list(DEFAULT_PLANNED_SOURCES),
                strict_integrity=True, adjudicator=adjudicator,
            )
            results[vendor_id] = VendorResult(
                vendor_id=vendor_id, verdict=outcome.verdict, signals=outcome.signals, reports=reports,
                documents=documents, ambiguous=outcome.ambiguous, llm_proposed=outcome.llm_proposed,
            )
            lines.append({
                "vendor_id": vendor_id, "ai_usage": outcome.verdict.ai_usage,
                "strength": outcome.verdict.strength, "coverage": round(outcome.verdict.coverage, 3),
                "documents": len(documents), "signals": len(outcome.signals),
                "ambiguous": len(outcome.ambiguous), "llm_proposals": len(outcome.llm_proposed),
                "rejected_documents": len(outcome.rejected_documents), "error": "",
            })
        except Exception as exc:
            results[vendor_id] = _failed_result(vendor_id, f"{type(exc).__name__}: {exc}")
            lines.append({"vendor_id": vendor_id, "ai_usage": "Inconclusive", "error": str(exc)})

    notes = [f"Collection mode: {mode}; LLM: " + (
        f"{llm.model} (local), calls={llm.stats['calls']}, cache_hits={llm.stats['cache_hits']}, "
        f"unavailable={llm.stats['unavailable']}, invalid={llm.stats['invalid']}" if llm else "disabled")]
    produced = export_workbook(workbook, output, results, notes)
    summary = {"output": str(produced), "vendors": lines, "llm": llm.stats if llm else None,
               "llm_available": llm.available if llm else None}
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", required=True, help="Original vendor workbook (never modified)")
    parser.add_argument("--output", required=True, help="New workbook to write; must differ from the input")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Private local cache directory")
    parser.add_argument("--mode", choices=["live", "cache"], default="live",
                        help="live fetches public pages; cache reuses saved collections")
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--delay", type=float, default=MIN_DELAY_SECONDS)
    parser.add_argument("--no-llm", action="store_true", help="Skip the local LLM; use rule/ML scores only")
    parser.add_argument("--llm-url", default=None, help="Local LLM server URL (loopback only)")
    parser.add_argument("--llm-model", default=None, help="Local model name")
    parser.add_argument("--vendors", default="", help="Comma-separated vendor IDs to run (default: all)")
    args = parser.parse_args(argv)

    selected = {item.strip() for item in args.vendors.split(",") if item.strip()} or None
    summary = run(args.workbook, args.output, cache_dir=args.cache_dir, mode=args.mode,
                  max_pages=args.max_pages, delay=args.delay, use_llm=not args.no_llm,
                  llm_url=args.llm_url, llm_model=args.llm_model, vendors=selected)
    for line in summary["vendors"]:
        print(json.dumps(line))
    print(json.dumps({k: v for k, v in summary.items() if k != "vendors"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
