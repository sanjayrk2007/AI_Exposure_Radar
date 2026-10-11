"""Mock-only tests (invented sentences) for label-conflict detection and the append-only merge."""
import pytest

from m4_intel.merge_reviewed_labels import main, merge_reviewed_labels
from m4_intel.train_specificity import load_labels

QUEUE_HEADER = "vendor_id,doc_id,sentence,reviewer_label,review_notes,review_status\n"
LABELS_HEADER = "vendor_id,doc_id,sentence,label\n"


def _labels(tmp_path, body=""):
    path = tmp_path / "labels.csv"
    path.write_bytes((LABELS_HEADER + body).encode("utf-8"))
    return path


def _queue(tmp_path, body):
    path = tmp_path / "queue.csv"
    path.write_text(QUEUE_HEADER + body, encoding="utf-8")
    return path


def test_conflicting_duplicate_labels_raise_instead_of_dropping_a_review(tmp_path):
    path = _labels(tmp_path, "V-002,D1,Fraud model scores transactions.,TECHNICAL\n"
                             "V-002,D1,Fraud  model scores transactions.,VAGUE\n")
    with pytest.raises(ValueError, match="Conflicting labels"):
        load_labels(path)
    same = _labels(tmp_path, "V-002,D1,Fraud model scores transactions.,TECHNICAL\n"
                             "V-002,D1,Fraud model scores transactions.,1\n")
    assert len(load_labels(same)) == 1


def test_merge_is_append_only_and_leaves_existing_bytes_unchanged(tmp_path):
    existing = "V-001,D1,Already reviewed sentence.,VAGUE\n"
    labels = _labels(tmp_path, existing)
    before = labels.read_bytes()
    queue = _queue(tmp_path, "V-002,D2,Invented technical sentence.,TECHNICAL,,REVIEWED\n"
                             "V-003,D3,Invented vague sentence.,0,,REVIEWED\n"
                             "V-004,D4,Still pending sentence.,,,PENDING_MANUAL_REVIEW\n"
                             "V-004,D5,Label but not finished.,TECHNICAL,,PENDING_MANUAL_REVIEW\n"
                             "V-001,D1,Already reviewed sentence.,VAGUE,,REVIEWED\n")
    summary = merge_reviewed_labels(queue, labels)
    assert summary["merged"] == 2 and summary["skipped_pending"] == 2 and summary["skipped_already_present"] == 1
    after = labels.read_bytes()
    assert after.startswith(before) and len(after) > len(before)
    assert [r["label"] for r in load_labels(labels)] == ["0", "1", "0"]


def test_merge_aborts_without_writing_on_conflict_synthetic_or_example_vendor(tmp_path):
    labels = _labels(tmp_path, "V-001,D1,Already reviewed sentence.,VAGUE\n")
    before = labels.read_bytes()
    for bad_row in ("V-001,D1,Already reviewed sentence.,TECHNICAL,,REVIEWED\n",   # conflicts with existing review
                    "V-002,SYN-0001,Synthetic sentence.,TECHNICAL,,REVIEWED\n",    # synthetic
                    "V-000,D0,Example vendor sentence.,TECHNICAL,,REVIEWED\n"):    # worked example vendor
        queue = _queue(tmp_path, "V-005,D5,A fine new sentence.,VAGUE,,REVIEWED\n" + bad_row)
        summary = merge_reviewed_labels(queue, labels)
        assert summary["errors"] and summary["merged"] == 0
        assert labels.read_bytes() == before
        assert main(["--queue", str(queue), "--labels", str(labels)]) == 1


def test_merge_dry_run_writes_nothing_and_creates_missing_file_with_header(tmp_path):
    queue = _queue(tmp_path, "V-002,D2,Invented technical sentence.,TECHNICAL,,REVIEWED\n")
    target = tmp_path / "new" / "labels.csv"
    assert merge_reviewed_labels(queue, target, dry_run=True)["merged"] == 1 and not target.exists()
    assert merge_reviewed_labels(queue, target)["merged"] == 1
    assert target.read_text(encoding="utf-8").splitlines()[0] == "vendor_id,doc_id,sentence,label"


def test_merge_aborts_without_modifying_labels_when_existing_file_already_conflicts(tmp_path):
    labels = _labels(tmp_path, "V-002,D1,Fraud model scores transactions.,TECHNICAL\n"
                               "V-002,D1,Fraud  model scores transactions.,VAGUE\n")
    before = labels.read_bytes()
    queue = _queue(tmp_path, "V-005,D5,A fine new sentence.,VAGUE,,REVIEWED\n")
    with pytest.raises(ValueError, match="conflicting labels"):
        merge_reviewed_labels(queue, labels)
    assert labels.read_bytes() == before
