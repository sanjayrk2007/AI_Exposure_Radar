"""Append human-reviewed rows from the review queue to data/labels.csv, safely.

Guarantees:
* Append-only: existing lines of labels.csv are never rewritten, reordered or relabelled.
* All-or-nothing: if any reviewed row is invalid or conflicts with an existing label, nothing is written.
* Only rows a human has finished are merged: ``reviewer_label`` must be TECHNICAL/VAGUE/EXCLUDE (or 1/0)
  AND ``review_status`` must be REVIEWED. Rows still PENDING_MANUAL_REVIEW are skipped, never guessed.
* Synthetic rows (``SYN`` doc ids or a SYNTHETIC provenance) and V-000 are rejected.

This script never assigns labels itself.

Usage:
    python -m m4_intel.merge_reviewed_labels --queue data/real_label_review_queue.csv --labels data/labels.csv [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from m4_intel.train_specificity import CSV_COLUMNS, DEFAULT_LABELS, EXPECTED_VENDORS, SYNTHETIC_LABELS_PATH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "data" / "real_label_review_queue.csv"
REVIEWED_STATUS = "REVIEWED"
_CANONICAL = {"TECHNICAL": "TECHNICAL", "1": "TECHNICAL", "VAGUE": "VAGUE", "0": "VAGUE", "EXCLUDE": "EXCLUDE"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _read_existing(path: Path) -> tuple[list[str], dict[tuple[str, str, str], str], bool]:
    """Return (header, identity -> canonical label, ends_with_newline) for an existing labels file."""
    if not path.exists():
        return list(CSV_COLUMNS), {}, True
    raw = path.read_bytes()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if not set(CSV_COLUMNS).issubset(header):
            raise ValueError(f"{path} must contain columns: {', '.join(CSV_COLUMNS)}")
        known: dict[tuple[str, str, str], str] = {}
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if not label:
                continue
            canonical = _CANONICAL.get(label)
            if canonical is None:
                raise ValueError(f"Existing labels file has invalid label {label!r}; fix it by hand first")
            identity = ((row.get("vendor_id") or "").strip(), (row.get("doc_id") or "").strip(),
                        _norm(row.get("sentence") or ""))
            if identity in known and known[identity] != canonical:
                raise ValueError(
                    f"{path} already contains conflicting labels ({known[identity]} and {canonical}) for the same "
                    f"sentence {identity!r}; resolve the conflict by hand before merging")
            known[identity] = canonical
    return header, known, (not raw) or raw.endswith(b"\n")


def merge_reviewed_labels(queue_path: str | Path = DEFAULT_QUEUE, labels_path: str | Path = DEFAULT_LABELS,
                          dry_run: bool = False) -> dict[str, Any]:
    queue, labels = Path(queue_path), Path(labels_path)
    if labels.exists() and labels.resolve() == SYNTHETIC_LABELS_PATH.resolve():
        raise ValueError("data/synthetic_labels.csv is synthetic and can never be a real labels file")
    if not queue.exists():
        raise FileNotFoundError(queue)
    header, known, ends_with_newline = _read_existing(labels)

    summary: dict[str, Any] = {"merged": 0, "skipped_pending": 0, "skipped_already_present": 0,
                               "errors": [], "dry_run": dry_run}
    to_append: list[dict[str, str]] = []
    pending: dict[tuple[str, str, str], str] = {}

    with queue.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"vendor_id", "doc_id", "sentence", "reviewer_label", "review_status"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Queue CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            vendor = (row.get("vendor_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            raw_label = (row.get("reviewer_label") or "").strip().upper()
            status = (row.get("review_status") or "").strip().upper()
            where = f"{queue}:{line_number}"
            if not raw_label:
                summary["skipped_pending"] += 1
                continue
            if status != REVIEWED_STATUS:
                # A label is present but the reviewer has not marked the row finished: do not merge.
                summary["skipped_pending"] += 1
                continue
            if raw_label not in _CANONICAL:
                summary["errors"].append(f"{where}: invalid reviewer_label {raw_label!r}")
                continue
            if vendor == "V-000" or vendor not in EXPECTED_VENDORS:
                summary["errors"].append(f"{where}: vendor_id {vendor!r} is not one of V-001..V-006 (V-000 is prohibited)")
                continue
            if not doc_id or not sentence:
                summary["errors"].append(f"{where}: doc_id and sentence are required")
                continue
            if doc_id.upper().startswith("SYN") or "SYNTHETIC" in (row.get("provenance") or "").upper():
                summary["errors"].append(f"{where}: synthetic rows can never be merged into real labels")
                continue
            identity = (vendor, doc_id, _norm(sentence))
            label = _CANONICAL[raw_label]
            if identity in known:
                if known[identity] != label:
                    summary["errors"].append(
                        f"{where}: conflicts with the existing label {known[identity]} for the same sentence; "
                        "existing human reviews are never overwritten")
                else:
                    summary["skipped_already_present"] += 1
                continue
            if identity in pending:
                if pending[identity] != label:
                    summary["errors"].append(f"{where}: the queue labels the same sentence two different ways")
                else:
                    summary["skipped_already_present"] += 1
                continue
            pending[identity] = label
            to_append.append({"vendor_id": vendor, "doc_id": doc_id, "sentence": sentence, "label": label})

    if summary["errors"]:
        summary["merged"] = 0
        return summary
    summary["merged"] = len(to_append)
    if dry_run or not to_append:
        return summary

    labels.parent.mkdir(parents=True, exist_ok=True)
    new_file = not labels.exists() or labels.stat().st_size == 0
    with labels.open("a", newline="", encoding="utf-8") as handle:
        if new_file:
            csv.writer(handle, lineterminator="\n").writerow(header)
        elif not ends_with_newline:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n", extrasaction="ignore", restval="")
        writer.writerows(to_append)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = parser.parse_args(argv)
    summary = merge_reviewed_labels(args.queue, args.labels, args.dry_run)
    for error in summary["errors"]:
        print(f"ERROR {error}", file=sys.stderr)
    print(f"merged={summary['merged']} skipped_pending={summary['skipped_pending']} "
          f"skipped_already_present={summary['skipped_already_present']} errors={len(summary['errors'])}"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
