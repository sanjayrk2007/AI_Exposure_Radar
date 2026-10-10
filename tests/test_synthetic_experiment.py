import csv
import json

import pytest

from m4_intel.generate_synthetic_data import FIELDS, build_rows, write_dataset
from m4_intel.run_synthetic_experiment import evaluate_synthetic, load_synthetic, run


def test_generation_is_deterministic_and_has_explicit_provenance():
    assert build_rows(seed=7) == build_rows(seed=7)
    assert build_rows(seed=7) != build_rows(seed=8)
    assert all(row["provenance"] == "SYNTHETIC_GENERATED_NOT_VENDOR_EVIDENCE" for row in build_rows())


def test_generation_counts_and_labels_are_valid(tmp_path):
    path = write_dataset(tmp_path / "synthetic.csv", technical_count=20, vague_count=20, exclude_count=10)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    assert len(rows) == 50
    assert {row["label"] for row in rows} == {"TECHNICAL", "VAGUE", "EXCLUDE"}
    assert len({row["synthetic_id"] for row in rows}) == len(rows)


def test_binary_evaluation_excludes_exclude_rows_and_reports_confusion_matrix():
    rows = build_rows(technical_count=20, vague_count=20, exclude_count=5)
    result = evaluate_synthetic(rows, n_splits=5)
    assert result["dataset"]["rows_used_for_binary_evaluation"] == 40
    assert result["dataset"]["rows_excluded_from_binary_evaluation"] == 5
    assert result["dataset"]["synthetic_rows_are_not_real_evidence"] is True
    assert len(result["folds"]) == 5
    for model in result["overall_oof_metrics"].values():
        assert len(model["confusion_matrix"]) == 2
        assert all(len(row) == 2 for row in model["confusion_matrix"])
        assert "technical_precision" in model and "macro_f1" in model


def test_insufficient_class_count_fails_clearly():
    rows = build_rows(technical_count=1, vague_count=5, exclude_count=0)
    with pytest.raises(ValueError, match="Not enough examples"):
        evaluate_synthetic(rows, n_splits=5)


def test_loader_rejects_non_synthetic_provenance(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "synthetic_id,sentence,label,scenario,provenance\n"
        "S1,AI powers classification,TECHNICAL,current,REAL_VENDOR_EVIDENCE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic provenance"):
        load_synthetic(path)


def test_run_preserves_existing_real_metrics_and_writes_report(tmp_path, monkeypatch):
    synthetic = write_dataset(tmp_path / "synthetic.csv", technical_count=15, vague_count=15, exclude_count=3)
    out = tmp_path / "models"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"status": "INSUFFICIENT_DATA", "real_data": {"rows": 13}}), encoding="utf-8")
    # Exercise report/merge behavior while keeping the actual rule scorer intact.
    result = run(synthetic, out, n_splits=3)
    stored = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert stored["status"] == "INSUFFICIENT_DATA"
    assert stored["real_data"]["rows"] == 13
    assert "synthetic_experiment" in stored
    assert (out / "experiment_report.md").exists()
    assert "do not establish real-world vendor generalization" in (out / "experiment_report.md").read_text(encoding="utf-8")
    assert result["status"] == "synthetic_experiment_completed"


def test_tfidf_vectorizer_never_sees_held_out_fold_text(monkeypatch):
    import m4_intel.run_synthetic_experiment as experiment
    from sklearn.model_selection import StratifiedKFold

    rows = build_rows(seed=21, technical_count=10, vague_count=10, exclude_count=0)
    for index, row in enumerate(rows):
        row["sentence"] += f" sentinelonlyrow{index:03d}"
    captured_models = []
    original_factory = experiment.make_tfidf_model

    def capture_factory():
        model = original_factory()
        captured_models.append(model)
        return model

    monkeypatch.setattr(experiment, "make_tfidf_model", capture_factory)
    result = experiment.evaluate_synthetic(rows, seed=42, n_splits=5)
    usable = [row for row in rows if row["label"] in {"TECHNICAL", "VAGUE"}]
    y = [1 if row["label"] == "TECHNICAL" else 0 for row in usable]
    expected_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        [row["sentence"] for row in usable], y))
    assert len(captured_models) == len(result["folds"]) == 5
    for model, (_, test_indices) in zip(captured_models, expected_splits):
        vocabulary = model.named_steps["tfidf"].vocabulary_
        for index in test_indices:
            sentinel = f"sentinelonlyrow{index:03d}"
            assert sentinel not in vocabulary
