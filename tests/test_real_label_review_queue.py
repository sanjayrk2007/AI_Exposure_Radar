import csv
import json

from m4_intel.build_real_label_review_queue import build_queue, write_queue


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
