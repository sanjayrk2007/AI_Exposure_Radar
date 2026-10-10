"""Safe integration runner for cached documents and extracted candidates.

This runner verifies quote/offset integrity and computes provisional rule scores.
It does NOT treat quote integrity as factual verification and therefore does not
issue final vendor AI-usage verdicts.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from m4_intel.specificity import score_signal
from m4_intel.verify import verify_candidate


ALLOWED_VENDOR_IDS = {f"V-{number:03d}" for number in range(1, 7)}
SIGNAL_FIELDS = [
    "vendor_id", "doc_id", "signal_id", "source_type", "url", "title",
    "excerpt", "char_start", "char_end", "category", "specificity", "weight",
    "score", "tag", "quote_integrity", "factual_review_status", "method",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _signal_dict(signal: object) -> dict[str, Any]:
    if is_dataclass(signal):
        return asdict(signal)
    if isinstance(signal, dict):
        return dict(signal)
    raise TypeError(f"Unsupported signal type: {type(signal).__name__}")


def run_pipeline(
    manifest_path: str | Path,
    candidates_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create provisional signal artifacts without issuing factual verdicts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signals_path = output / "candidate_signals.csv"
    summary_path = output / "pipeline_summary.json"
    if not overwrite and (signals_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Output already exists in {output}; choose a new directory or pass --overwrite."
        )

    documents = read_jsonl(manifest_path)
    candidates = read_jsonl(candidates_path)

    docs_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for doc in documents:
        vendor_id, doc_id = doc.get("vendor_id"), doc.get("doc_id")
        if vendor_id not in ALLOWED_VENDOR_IDS:
            continue
        if not isinstance(doc_id, str) or not doc.get("text"):
            continue
        key = (vendor_id, doc_id)
        if key in docs_by_key:
            raise ValueError(f"Duplicate manifest document identity: {key}")
        docs_by_key[key] = doc

    signal_rows: list[dict[str, Any]] = []
    rejected_missing_doc = 0
    rejected_invalid_quote = 0
    scored_counts: Counter[str] = Counter()
    vendor_counts: Counter[str] = Counter()
    seen_candidate_keys: set[tuple[str, str, int, int, str]] = set()

    for candidate in candidates:
        vendor_id = candidate.get("vendor_id")
        # V-000 and any unknown vendor are always excluded.
        if vendor_id not in ALLOWED_VENDOR_IDS:
            continue
        doc_id = candidate.get("doc_id")
        doc = docs_by_key.get((vendor_id, doc_id))
        if doc is None:
            rejected_missing_doc += 1
            continue

        start = candidate.get("char_start", candidate.get("start"))
        end = candidate.get("char_end", candidate.get("end"))
        key = (vendor_id, doc_id, start, end, candidate.get("text", ""))
        if key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(key)

        # Adapter is explicit: saved artifacts use char_start/char_end, while
        # verify_candidate expects start/end.
        adapted_candidate = {
            "text": candidate.get("text"),
            "start": start,
            "end": end,
            "categories": candidate.get("categories", []),
        }
        signal = verify_candidate(doc, adapted_candidate)
        if signal is None:
            rejected_invalid_quote += 1
            continue

        # Scoring is provisional ranking only. It is not a truth determination.
        score_signal(signal)
        row = _signal_dict(signal)
        row.update({
            "title": candidate.get("title", doc.get("title", "")),
            "quote_integrity": "PASS",
            "factual_review_status": "UNREVIEWED",
        })
        signal_rows.append({field: row.get(field, "") for field in SIGNAL_FIELDS})
        scored_counts[str(row.get("tag", "NONE"))] += 1
        vendor_counts[vendor_id] += 1

    with signals_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS)
        writer.writeheader()
        writer.writerows(signal_rows)

    manifest_vendor_counts = Counter(
        doc.get("vendor_id") for doc in documents
        if doc.get("vendor_id") in ALLOWED_VENDOR_IDS
    )
    all_vendors = []
    for vendor_id in sorted(ALLOWED_VENDOR_IDS):
        all_vendors.append({
            "vendor_id": vendor_id,
            "documents_in_manifest": manifest_vendor_counts[vendor_id],
            "quote_verified_candidates": vendor_counts[vendor_id],
            "final_ai_usage_verdict": "Inconclusive",
            "verdict_status": "PENDING_FACTUAL_REVIEW",
            "note": (
                "Quote/offset integrity and rule score are not factual verification. "
                "Do not infer absence of AI from missing pages or missing candidates."
            ),
        })

    summary = {
        "status": "PROVISIONAL_ONLY",
        "documents_loaded": len(documents),
        "candidate_rows_loaded": len(candidates),
        "quote_verified_candidates": len(signal_rows),
        "rejected_missing_document": rejected_missing_doc,
        "rejected_invalid_quote_or_offsets": rejected_invalid_quote,
        "tag_counts_provisional": dict(scored_counts),
        "vendor_summary": all_vendors,
        "outputs": {
            "candidate_signals_csv": str(signals_path),
            "pipeline_summary_json": str(summary_path),
        },
        "limitations": [
            "Signals are quote/offset verified only, not factually verified.",
            "No final Yes/No AI-usage verdicts are issued by this runner.",
            "Only source families represented in the supplied manifest are present.",
            "V-000 is excluded; original workbook is not opened or modified.",
            "Manual evidence review is required before final verdict generation.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to collected manifest.jsonl")
    parser.add_argument("--candidates", required=True, help="Path to extracted candidates JSONL")
    parser.add_argument("--output-dir", default="outputs/pilot_v2", help="New output directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace this runner's two output files")
    args = parser.parse_args()
    summary = run_pipeline(args.manifest, args.candidates, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({
        "status": summary["status"],
        "documents_loaded": summary["documents_loaded"],
        "candidate_rows_loaded": summary["candidate_rows_loaded"],
        "quote_verified_candidates": summary["quote_verified_candidates"],
        "rejected_missing_document": summary["rejected_missing_document"],
        "rejected_invalid_quote_or_offsets": summary["rejected_invalid_quote_or_offsets"],
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
