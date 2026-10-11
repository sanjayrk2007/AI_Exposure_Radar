"""Build a human-review queue from existing candidates without auto-labeling them.

The output is a review aid, not training data. Only after manual review should
rows be merged into data/labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "candidates_pilot_v2.jsonl"
DEFAULT_LABELS = ROOT / "data" / "labels.csv"
DEFAULT_MANIFEST = ROOT / "cache_pilot_v2" / "manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "real_label_review_queue.csv"
QUEUE_FIELDS = (
    "vendor_id", "doc_id", "source_type", "url", "title", "sentence",
    "char_start", "char_end", "context_before", "context_after",
    "reviewer_label", "review_notes", "review_status",
)
VALID_LABELS = {"TECHNICAL", "VAGUE", "EXCLUDE", "0", "1"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {source}:{line_no}")
            rows.append(item)
    return rows


def read_labels(path: str | Path) -> set[tuple[str, str, str]]:
    source = Path(path)
    if not source.exists():
        return set()
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"vendor_id", "doc_id", "sentence", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Labels CSV must contain {sorted(required)}")
        keys = set()
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if not label:
                continue
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label {label!r} in {source}; expected one of {sorted(VALID_LABELS)}")
            keys.add(((row.get("vendor_id") or "").strip(),
                      (row.get("doc_id") or "").strip(), norm(row.get("sentence") or "")))
        return keys


def read_manifest(path: str | Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    return {str(row.get("doc_id", "")): row for row in read_jsonl(path) if row.get("doc_id")}


def build_queue(candidates_path: str | Path = DEFAULT_CANDIDATES,
                labels_path: str | Path = DEFAULT_LABELS,
                manifest_path: str | Path | None = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    candidates = read_jsonl(candidates_path)
    labelled = read_labels(labels_path)
    manifest = read_manifest(manifest_path)
    queue: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        vendor = str(candidate.get("vendor_id", "")).strip()
        if vendor == "V-000":
            continue
        doc_id = str(candidate.get("doc_id", "")).strip()
        sentence = str(candidate.get("text", candidate.get("sentence", ""))).strip()
        if not vendor or not doc_id or not sentence:
            continue
        key = (vendor, doc_id, norm(sentence))
        if key in labelled or key in seen:
            continue
        seen.add(key)
        doc = manifest.get(doc_id, {})
        body = str(doc.get("text", ""))
        try:
            start, end = int(candidate.get("char_start", 0)), int(candidate.get("char_end", 0))
        except (TypeError, ValueError):
            start, end = 0, 0
        # Only show context if offsets point to this candidate text in the source.
        offsets_valid = 0 <= start <= end <= len(body) and norm(body[start:end]) == norm(sentence)
        context_before = body[max(0, start - 240):start] if offsets_valid else ""
        context_after = body[end:min(len(body), end + 240)] if offsets_valid else ""
        queue.append({
            "vendor_id": vendor,
            "doc_id": doc_id,
            "source_type": str(candidate.get("source_type", doc.get("source_type", ""))),
            "url": str(candidate.get("url", doc.get("url", ""))),
            "title": str(candidate.get("title", doc.get("title", ""))),
            "sentence": sentence,
            "char_start": str(start) if offsets_valid else "",
            "char_end": str(end) if offsets_valid else "",
            "context_before": context_before,
            "context_after": context_after,
            "reviewer_label": "",
            "review_notes": "",
            "review_status": "PENDING_MANUAL_REVIEW",
        })
    return queue


def write_queue(rows: list[dict[str, str]], output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = build_queue(args.candidates, args.labels, args.manifest)
    target = write_queue(rows, args.output)
    print(f"Wrote {target} with {len(rows)} unlabelled candidate rows.")
    print("All labels are blank and review status is pending; this script does not assign labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
