from __future__ import annotations

import json
from pathlib import Path

import pytest

from m4_intel.run_pipeline import run_pipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pipeline_verifies_offsets_but_does_not_issue_final_verdicts(tmp_path: Path) -> None:
    text = "The platform uses an AI model to classify customer documents."
    doc = {
        "vendor_id": "V-001", "doc_id": "D1", "source_type": "WEB",
        "url": "https://example.test/", "fetched_at": "2026-01-01T00:00:00Z",
        "http_status": 200, "content_hash": "hash", "title": "Example",
        "text": text, "snapshot_path": "snapshot.txt", "meta": {},
    }
    start = text.index("The platform")
    candidate = {
        "vendor_id": "V-001", "doc_id": "D1", "source_type": "WEB",
        "url": doc["url"], "title": "Example", "text": text[start:],
        "char_start": start, "char_end": len(text), "categories": ["techniques"],
    }
    manifest, candidates = tmp_path / "manifest.jsonl", tmp_path / "candidates.jsonl"
    _write_jsonl(manifest, [doc])
    _write_jsonl(candidates, [candidate])

    result = run_pipeline(manifest, candidates, tmp_path / "out")
    assert result["quote_verified_candidates"] == 1
    assert result["rejected_invalid_quote_or_offsets"] == 0
    assert result["vendor_summary"][0]["final_ai_usage_verdict"] == "Inconclusive"
    assert result["vendor_summary"][0]["verdict_status"] == "PENDING_FACTUAL_REVIEW"


def test_pipeline_rejects_bad_offsets_and_excludes_v000(tmp_path: Path) -> None:
    doc = {
        "vendor_id": "V-001", "doc_id": "D1", "source_type": "WEB",
        "url": "https://example.test/", "text": "A valid sentence about AI.",
    }
    candidates = [
        {"vendor_id": "V-001", "doc_id": "D1", "text": "wrong quote",
         "char_start": 0, "char_end": 4, "categories": ["techniques"]},
        {"vendor_id": "V-000", "doc_id": "D0", "text": "example",
         "char_start": 0, "char_end": 7, "categories": ["techniques"]},
    ]
    manifest, candidate_file = tmp_path / "manifest.jsonl", tmp_path / "candidates.jsonl"
    _write_jsonl(manifest, [doc])
    _write_jsonl(candidate_file, candidates)
    result = run_pipeline(manifest, candidate_file, tmp_path / "out")
    assert result["quote_verified_candidates"] == 0
    assert result["rejected_invalid_quote_or_offsets"] == 1
    assert all(v["vendor_id"] != "V-000" for v in result["vendor_summary"])


def test_pipeline_does_not_overwrite_existing_outputs_without_opt_in(tmp_path: Path) -> None:
    manifest, candidate_file = tmp_path / "manifest.jsonl", tmp_path / "candidates.jsonl"
    _write_jsonl(manifest, [])
    _write_jsonl(candidate_file, [])
    output = tmp_path / "out"
    run_pipeline(manifest, candidate_file, output)
    with pytest.raises(FileExistsError):
        run_pipeline(manifest, candidate_file, output)
