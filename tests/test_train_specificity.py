import csv
import json

import pytest

from m4_intel.train_specificity import (
    CSV_COLUMNS,
    create_label_template,
    load_labels,
    remove_near_duplicates,
    run_training,
    score_specificity,
)


def test_label_template_has_only_required_header(tmp_path):
    path = create_label_template(tmp_path / "labels.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(CSV_COLUMNS)
        assert list(csv.reader(handle)) == []


def test_labels_reject_example_vendor_and_invalid_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("vendor_id,doc_id,sentence,label\nV-000,D0,example sentence,TECHNICAL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V-000"):
        load_labels(path)
    path.write_text("vendor_id,doc_id,sentence,label\nV-001,D1,sentence,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid label"):
        load_labels(path)


def test_exclude_rows_and_exact_duplicates_not_used(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "vendor_id,doc_id,sentence,label\n"
        "V-001,D1,OpenAI powers current document classification,TECHNICAL\n"
        "V-001,D1,OpenAI powers current document classification,TECHNICAL\n"
        "V-001,D2,Generic AI leadership,EXCLUDE\n", encoding="utf-8")
    assert load_labels(path) == [{"vendor_id": "V-001", "doc_id": "D1",
                                  "sentence": "OpenAI powers current document classification", "label": "1"}]


def test_near_duplicate_removal_is_deterministic_before_folds():
    rows = [
        {"vendor_id": "V-001", "doc_id": "D1", "sentence": "Our AI-powered platform supports intelligent workflow automation for enterprise teams today.", "label": "0"},
        {"vendor_id": "V-002", "doc_id": "D2", "sentence": "Our AI powered platform supports intelligent workflow automation for enterprise teams today.", "label": "0"},
        {"vendor_id": "V-003", "doc_id": "D3", "sentence": "OpenAI classifies invoice documents in the production workflow.", "label": "1"},
    ]
    kept, removed = remove_near_duplicates(rows)
    assert len(kept) == 2
    assert removed == 1


def test_empty_dataset_records_no_metrics_and_fallback(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\n", encoding="utf-8")
    model_dir = tmp_path / "models"
    result = run_training(labels, model_dir)
    assert result["status"] == "awaiting_manual_labels"
    assert result["model_name"] == "rule_based_fallback"
    assert "macro_f1_mean_fold" not in result
    assert json.loads((model_dir / "metrics.json").read_text())["status"] == "awaiting_manual_labels"


def test_score_specificity_uses_rule_fallback_when_model_missing(tmp_path):
    sentence = "OpenAI uses an API for fraud detection in customer data."
    assert score_specificity(sentence, tmp_path) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        score_specificity(None, tmp_path)


def test_leave_one_vendor_out_evaluation_reports_each_vendor_and_missing_class():
    from m4_intel.train_specificity import _fold_evaluate

    rows = []
    for index, vendor in enumerate((f"V-{number:03d}" for number in range(1, 7))):
        # Each vendor has two reviewed examples, one per class; distinctive wording
        # is only a test fixture and is not written to the Optiv label dataset.
        rows.extend([
            {"vendor_id": vendor, "doc_id": f"{vendor}-T", "sentence": f"Vendor {index} currently deploys OpenAI for invoice classification", "label": "1"},
            {"vendor_id": vendor, "doc_id": f"{vendor}-V", "sentence": f"Vendor {index} is exploring future AI innovation", "label": "0"},
        ])
    result = _fold_evaluate(rows)
    assert list(result["per_vendor_folds"]) == [f"V-{number:03d}" for number in range(1, 7)]
    assert all(fold["n_test"] == 2 for fold in result["per_vendor_folds"].values())
    assert all("macro_f1" in fold["rule_based"] for fold in result["per_vendor_folds"].values())
    assert result["model_summary"]["tfidf_logistic_regression"]["folds_with_metrics"] == 6

def test_small_dataset_reports_insufficient_data_without_metrics(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "vendor_id,doc_id,sentence,label\n"
        "V-001,D1,Generic AI innovation for the future,VAGUE\n"
        "V-001,D2,Generic AI leadership for the future,VAGUE\n"
        "V-002,D3,Generic AI services for the future,VAGUE\n"
        "V-005,D4,OpenAI powers current invoice classification,TECHNICAL\n"
        "V-005,D5,OpenAI classifies documents in production,TECHNICAL\n",
        encoding="utf-8",
    )

    result = run_training(labels, tmp_path / "models")

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["model_name"] == "rule_based_fallback"
    assert result["per_vendor_folds"] == {}
    assert result["model_summary"] == {}
    assert not (tmp_path / "models" / "tfidf_logistic_regression.joblib").exists()
    assert json.loads(
        (tmp_path / "models" / "metrics.json").read_text()
    )["status"] == "INSUFFICIENT_DATA"