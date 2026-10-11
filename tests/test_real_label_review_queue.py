import csv
import json

import pytest

from m4_intel.build_real_label_review_queue import build_queue, write_queue
from m4_intel.merge_reviewed_labels import merge_reviewed_labels


def test_queue_skips_existing_labels_and_never_assigns_a_label(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("\n".join([
        json.dumps({"vendor_id":"V-001","doc_id":"D1","source_type":"WEB","url":"https://example.test","title":"Page","text":"Already reviewed","char_start":0,"char_end":16}),
        json.dumps({"vendor_id":"V-002","doc_id":"D2","source_type":"WEB","url":"https://example.test/2","title":"Page 2","text":"Needs review","char_start":0,"char_end":12}),
        json.dumps({"vendor_id":"V-002","doc_id":"D2","source_type":"WEB","url":"https://example.test/2","title":"Page 2","text":"Needs review","char_start":0,"char_end":12}),
    ]), encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\nV-001,D1,Already reviewed,VAGUE\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"doc_id":"D2","text":"Needs review is sentence."}) + "\n", encoding="utf-8")
    rows = build_queue(candidates, labels, manifest)
    assert len(rows) == 1
    assert rows[0]["sentence"] == "Needs review"
    assert rows[0]["reviewer_label"] == ""
    assert rows[0]["review_status"] == "PENDING_MANUAL_REVIEW"
    assert rows[0]["char_start"] == "0"
    output = write_queue(rows, tmp_path / "queue.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["reviewer_label"] == ""


def test_queue_without_manifest_keeps_candidate_but_does_not_fake_context(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"vendor_id":"V-003","doc_id":"D3","text":"AI may help","char_start":0,"char_end":11}) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\n", encoding="utf-8")
    rows = build_queue(candidates, labels, tmp_path / "missing.jsonl")
    assert len(rows) == 1
    assert rows[0]["context_before"] == ""
    assert rows[0]["context_after"] == ""
    assert rows[0]["char_start"] == ""


HIGH = "OpenAI is used through an API for fraud detection on customer data."
LOW = "Our team enjoys community events and local offices."


def _candidates(tmp_path, vendors):
    lines = []
    for vendor in vendors:
        for index, text in enumerate((f"{HIGH} Case {vendor}.", f"{LOW} Case {vendor}.", f"{HIGH} Second {vendor}.")):
            lines.append(json.dumps({"vendor_id": vendor, "doc_id": f"{vendor}-D{index}", "text": text, "char_start": 0}))
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_queue_reaches_underrepresented_vendors_first_and_mixes_rule_strata(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\nV-001,X1,Reviewed one,VAGUE\nV-001,X2,Reviewed two,VAGUE\n",
                      encoding="utf-8")
    candidates = _candidates(tmp_path, ["V-001", "V-002", "V-003"])
    rows = build_queue(candidates, labels, None)
    assert [r["vendor_id"] for r in rows[:3]] == ["V-002", "V-003", "V-001"]     # fewest reviewed labels first
    v2 = [r for r in rows if r["vendor_id"] == "V-002"]
    assert [r["queue_stratum"] for r in v2] == ["rule_high", "rule_low", "rule_high"]
    assert all(float(r["rule_score"]) >= 0.5 for r in v2 if r["queue_stratum"] == "rule_high")
    assert all(r["reviewer_label"] == "" and r["review_status"] == "PENDING_MANUAL_REVIEW" for r in rows)
    assert build_queue(candidates, labels, None) == rows                          # deterministic


def test_rebuild_never_overwrites_recorded_human_decisions(tmp_path):
    rows = build_queue(_candidates(tmp_path, ["V-002"]), tmp_path / "none.csv", None)
    target = write_queue(rows, tmp_path / "queue.csv")
    write_queue(rows, target)                                                      # all pending: safe to rebuild
    saved = list(csv.DictReader(target.open(newline="", encoding="utf-8")))
    saved[0]["reviewer_label"], saved[0]["review_status"] = "TECHNICAL", "REVIEWED"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="recorded review decision"):
        write_queue(rows, target)
    assert target.read_bytes() == before
    write_queue(rows, target, force=True)                                          # explicit override only
    assert target.read_bytes() != before


def test_ranking_columns_are_not_merged_into_labels(tmp_path):
    rows = build_queue(_candidates(tmp_path, ["V-002"]), tmp_path / "none.csv", None)
    queue = write_queue(rows, tmp_path / "queue.csv")
    saved = list(csv.DictReader(queue.open(newline="", encoding="utf-8")))
    saved[0]["reviewer_label"], saved[0]["review_status"] = "TECHNICAL", "REVIEWED"
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    labels = tmp_path / "labels.csv"
    assert merge_reviewed_labels(queue, labels)["merged"] == 1
    lines = labels.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "vendor_id,doc_id,sentence,label" and len(lines) == 2 and "rule_" not in labels.read_text()
