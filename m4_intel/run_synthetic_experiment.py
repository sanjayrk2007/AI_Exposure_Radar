"""Compare rule and TF-IDF baselines on a separate synthetic M4 dataset."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from m4_intel.generate_synthetic_data import DEFAULT_OUTPUT as DEFAULT_SYNTHETIC
from m4_intel.train_specificity import SEED, make_tfidf_model
from m4_intel.specificity import score_specificity_rule

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "m4_intel" / "models" / "specificity_v1"


def load_synthetic(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"synthetic_id", "sentence", "label", "scenario", "template_family", "provenance"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Synthetic CSV must contain columns: {', '.join(sorted(required))}")
        rows = [dict(row) for row in reader]
    if any(row["provenance"] != "SYNTHETIC_GENERATED_NOT_VENDOR_EVIDENCE" for row in rows):
        raise ValueError("Every row must be explicitly marked as synthetic provenance")
    invalid = sorted({row["label"] for row in rows} - {"TECHNICAL", "VAGUE", "EXCLUDE"})
    if invalid:
        raise ValueError(f"Invalid synthetic labels: {invalid}")
    return rows


def _metric_block(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels_present = set(y_true)
    return {
        "n": len(y_true),
        "class_counts_true": {"VAGUE": int(sum(y == 0 for y in y_true)),
                               "TECHNICAL": int(sum(y == 1 for y in y_true))},
        "vague_precision": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "vague_recall": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "vague_f1": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "technical_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "missing_classes": [name for label, name in ((0, "VAGUE"), (1, "TECHNICAL")) if label not in labels_present],
        "confusion_matrix_labels": ["VAGUE", "TECHNICAL"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def evaluate_synthetic(rows: list[dict[str, str]], seed: int = SEED, n_splits: int = 5) -> dict[str, Any]:
    # EXCLUDE is intentionally excluded from this binary task.
    usable = [row for row in rows if row["label"] in {"TECHNICAL", "VAGUE"}]
    labels = [1 if row["label"] == "TECHNICAL" else 0 for row in usable]
    counts = Counter(labels)
    if len(counts) != 2:
        raise ValueError("Synthetic binary evaluation requires both TECHNICAL and VAGUE examples")
    actual_splits = min(n_splits, counts[0], counts[1])
    if actual_splits < 2:
        raise ValueError("Not enough examples per class for stratified cross-validation")
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    fold_records: list[dict[str, Any]] = []
    predictions = {"rule_based": [], "tfidf_logistic_regression": []}
    truth: list[int] = []
    sentences = [row["sentence"] for row in usable]

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(sentences, labels), start=1):
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]
        if len(set(y_train)) != 2 or len(set(y_test)) != 2:
            raise RuntimeError(f"Fold {fold_index} is invalid: training and test folds must contain both classes")
        x_train = [sentences[i] for i in train_idx]
        x_test = [sentences[i] for i in test_idx]
        # Rule threshold is fixed in advance; no test-fold tuning.
        rule_pred = [int(score_specificity_rule(text) >= 0.5) for text in x_test]
        # The vectorizer is part of the pipeline and is fit only on this fold's training text.
        model = make_tfidf_model()
        model.fit(x_train, y_train)
        tfidf_pred = model.predict(x_test).astype(int).tolist()
        fold_records.append({
            "fold": fold_index,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_class_counts": {"VAGUE": y_train.count(0), "TECHNICAL": y_train.count(1)},
            "test_class_counts": {"VAGUE": y_test.count(0), "TECHNICAL": y_test.count(1)},
            "rule_based": _metric_block(y_test, rule_pred),
            "tfidf_logistic_regression": _metric_block(y_test, tfidf_pred),
        })
        truth.extend(y_test)
        predictions["rule_based"].extend(rule_pred)
        predictions["tfidf_logistic_regression"].extend(tfidf_pred)

    summaries = {name: _metric_block(truth, preds) for name, preds in predictions.items()}
    return {
        "status": "synthetic_experiment_completed",
        "provenance": "SYNTHETIC_ONLY_NOT_VENDOR_EVIDENCE",
        "seed": seed,
        "requested_folds": n_splits,
        "actual_folds": actual_splits,
        "dataset": {
            "total_rows": len(rows),
            "rows_used_for_binary_evaluation": len(usable),
            "rows_excluded_from_binary_evaluation": len(rows) - len(usable),
            "label_counts_all": dict(sorted(Counter(row["label"] for row in rows).items())),
            "label_counts_binary": {"VAGUE": counts[0], "TECHNICAL": counts[1]},
            "synthetic_rows_are_not_real_evidence": True,
        },
        "protocol": {
            "method": "stratified k-fold cross-validation on synthetic binary rows",
            "text_vectorizer_fit_within_each_training_fold": True,
            "test_fold_used_for_training_or_tuning": False,
            "exclude_label_used_for_binary_training": False,
            "rule_threshold": 0.5,
            "tfidf_configuration": {
                "ngram_range": [1, 2], "min_df": 1, "max_features": 30000,
                "sublinear_tf": True, "strip_accents": "unicode",
                "classifier": "LogisticRegression", "max_iter": 2000,
                "class_weight": "balanced", "random_state": seed,
            },
            "limitation": "Synthetic cross-validation measures performance on generated patterns only; it does not estimate generalization to real vendors.",
        },
        "environment": {"python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__},
        "overall_oof_metrics": summaries,
        "folds": fold_records,
    }


def evaluate_grouped_synthetic(rows: list[dict[str, str]], seed: int = SEED,
                               n_splits: int = 5) -> dict[str, Any]:
    """Evaluate on held-out template families; invalid folds are reported, not hidden."""
    usable = [row for row in rows if row["label"] in {"TECHNICAL", "VAGUE"}]
    labels = [1 if row["label"] == "TECHNICAL" else 0 for row in usable]
    groups = [row["template_family"] for row in usable]
    counts = Counter(labels)
    if len(counts) != 2:
        raise ValueError("Grouped synthetic evaluation requires both TECHNICAL and VAGUE examples")
    unique_groups = len(set(groups))
    actual_splits = min(n_splits, unique_groups)
    if actual_splits < 2:
        raise ValueError("Not enough template families for grouped evaluation")

    splitter = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    sentences = [row["sentence"] for row in usable]
    fold_records: list[dict[str, Any]] = []
    all_truth: list[int] = []
    all_predictions: dict[str, list[int]] = {"rule_based": [], "tfidf_logistic_regression": []}
    errors: dict[str, list[dict[str, Any]]] = {"rule_based": [], "tfidf_logistic_regression": []}
    evaluated_indices: set[int] = set()

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(sentences, labels, groups), start=1):
        train_classes, test_classes = set(labels[i] for i in train_idx), set(labels[i] for i in test_idx)
        train_groups = {groups[i] for i in train_idx}
        test_groups = {groups[i] for i in test_idx}
        overlap = sorted(train_groups & test_groups)
        record: dict[str, Any] = {
            "fold": fold_index, "n_train": len(train_idx), "n_test": len(test_idx),
            "train_template_families": sorted(train_groups), "test_template_families": sorted(test_groups),
            "template_family_overlap": overlap,
            "train_class_counts": {"VAGUE": sum(labels[i] == 0 for i in train_idx),
                                   "TECHNICAL": sum(labels[i] == 1 for i in train_idx)},
            "test_class_counts": {"VAGUE": sum(labels[i] == 0 for i in test_idx),
                                  "TECHNICAL": sum(labels[i] == 1 for i in test_idx)},
        }
        if overlap:
            record["status"] = "invalid_template_family_leakage"
            fold_records.append(record)
            continue
        if len(train_classes) < 2 or len(test_classes) < 2:
            record["status"] = "not_evaluable_missing_class"
            record["missing_train_classes"] = [n for i, n in ((0, "VAGUE"), (1, "TECHNICAL")) if i not in train_classes]
            record["missing_test_classes"] = [n for i, n in ((0, "VAGUE"), (1, "TECHNICAL")) if i not in test_classes]
            fold_records.append(record)
            continue

        x_train = [sentences[i] for i in train_idx]
        x_test = [sentences[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]
        rule_pred = [int(score_specificity_rule(text) >= 0.5) for text in x_test]
        model = make_tfidf_model()
        model.fit(x_train, y_train)
        tfidf_pred = model.predict(x_test).astype(int).tolist()
        record["status"] = "evaluated"
        record["rule_based"] = _metric_block(y_test, rule_pred)
        record["tfidf_logistic_regression"] = _metric_block(y_test, tfidf_pred)
        fold_records.append(record)
        evaluated_indices.update(test_idx)
        all_truth.extend(y_test)
        all_predictions["rule_based"].extend(rule_pred)
        all_predictions["tfidf_logistic_regression"].extend(tfidf_pred)
        for local_i, original_i in enumerate(test_idx):
            row = usable[original_i]
            truth, preds = labels[original_i], {
                "rule_based": rule_pred[local_i], "tfidf_logistic_regression": tfidf_pred[local_i]
            }
            for model_name, prediction in preds.items():
                if prediction != truth:
                    errors[model_name].append({
                        "synthetic_id": row["synthetic_id"], "template_family": row["template_family"],
                        "truth": "TECHNICAL" if truth else "VAGUE",
                        "prediction": "TECHNICAL" if prediction else "VAGUE",
                        "error_type": "false_negative" if truth == 1 else "false_positive",
                        "sentence": row["sentence"],
                    })

    valid_folds = [fold for fold in fold_records if fold.get("status") == "evaluated"]
    summary = {}
    if all_truth:
        for model_name, preds in all_predictions.items():
            summary[model_name] = _metric_block(all_truth, preds)
    return {
        "status": "completed" if valid_folds else "not_evaluable",
        "method": "StratifiedGroupKFold grouped by template_family",
        "requested_folds": n_splits, "actual_folds": actual_splits,
        "valid_folds": len(valid_folds), "invalid_folds": len(fold_records) - len(valid_folds),
        "usable_rows": len(usable), "evaluated_rows": len(evaluated_indices),
        "evaluation_coverage": round(len(evaluated_indices) / len(usable), 4) if usable else 0.0,
        "template_families": unique_groups,
        "template_family_overlap_detected": any(fold.get("template_family_overlap") for fold in fold_records),
        "metrics_on_evaluable_folds_only": summary,
        "folds": fold_records,
        "error_analysis_on_evaluable_folds_only": errors,
        "limitations": [
            "Template-held-out evaluation is still synthetic-only and does not estimate real-vendor generalization.",
            "Invalid folds missing either class are excluded from grouped aggregate metrics; evaluation coverage is reported.",
        ],
    }


def make_report(metrics: dict[str, Any], real_metrics: dict[str, Any] | None = None) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# M4 Synthetic ML Experiment Report", "",
        "> **Important:** all examples in this experiment are synthetic. These results do not establish real-world vendor generalization and must not be described as performance on Optiv vendor evidence.", "",
        "## Dataset", "",
        f"- Total synthetic rows: {dataset['total_rows']}",
        f"- Binary evaluation rows: {dataset['rows_used_for_binary_evaluation']}",
        f"- EXCLUDE rows omitted from binary evaluation: {dataset['rows_excluded_from_binary_evaluation']}",
        f"- Label counts: `{json.dumps(dataset['label_counts_all'], sort_keys=True)}`",
        f"- Random seed: {metrics['seed']}",
        f"- Evaluation: {metrics['actual_folds']}-fold stratified cross-validation", "",
        "## Out-of-fold metrics", "",
        "| Model | Vague P | Vague R | Vague F1 | Technical P | Technical R | Technical F1 | Macro-F1 | Confusion matrix (true rows: VAGUE, TECHNICAL; predicted columns: VAGUE, TECHNICAL) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, display in (("rule_based", "Rule-based baseline"), ("tfidf_logistic_regression", "TF-IDF + Logistic Regression")):
        item = metrics["overall_oof_metrics"][name]
        lines.append(f"| {display} | {item['vague_precision']:.4f} | {item['vague_recall']:.4f} | {item['vague_f1']:.4f} | {item['technical_precision']:.4f} | {item['technical_recall']:.4f} | {item['technical_f1']:.4f} | {item['macro_f1']:.4f} | `{item['confusion_matrix']}` |")
    lines += ["", "### Stratified fold details", "", "| Fold | Train rows | Test rows | Rule macro-F1 | Rule technical recall | TF-IDF macro-F1 | TF-IDF technical recall |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in metrics["folds"]:
        lines.append(f"| {fold['fold']} | {fold['n_train']} | {fold['n_test']} | {fold['rule_based']['macro_f1']:.4f} | {fold['rule_based']['technical_recall']:.4f} | {fold['tfidf_logistic_regression']['macro_f1']:.4f} | {fold['tfidf_logistic_regression']['technical_recall']:.4f} |")
    grouped = metrics.get("grouped_evaluation", {})
    protocol = metrics.get("protocol", {})
    tfidf_config = protocol.get("tfidf_configuration", {})
    lines += ["", "## Model configuration and reproducibility", "",
              f"- Python: {metrics.get('environment', {}).get('python_version', 'not recorded')}.",
              f"- scikit-learn: {metrics.get('environment', {}).get('scikit_learn_version', 'not recorded')}.",
              f"- TF-IDF configuration: `{json.dumps(tfidf_config, sort_keys=True)}`.",
              f"- Seed: {metrics.get('seed')}; stratified folds: {metrics.get('actual_folds')}.",
              "- Re-run with `python -m m4_intel.generate_synthetic_data` followed by `python -m m4_intel.run_synthetic_experiment`.", ""]
    lines += ["", "## Template-family-held-out evaluation", "",
              f"- Status: `{grouped.get('status', 'not_run')}`",
              f"- Valid folds: {grouped.get('valid_folds', 0)}; invalid folds: {grouped.get('invalid_folds', 0)}",
              f"- Evaluation coverage: {grouped.get('evaluation_coverage', 0.0):.1%}",
              f"- Template families: {grouped.get('template_families', 0)}",
              "- Entire template families are held out; folds missing a class are marked not evaluable and excluded from grouped aggregate metrics.", ""]
    if grouped.get("metrics_on_evaluable_folds_only"):
        lines += ["| Model | Technical P | Technical R | Technical F1 | Vague P | Vague R | Vague F1 | Macro-F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, display in (("rule_based", "Rule-based baseline"), ("tfidf_logistic_regression", "TF-IDF + Logistic Regression")):
            item = grouped["metrics_on_evaluable_folds_only"].get(name)
            if item:
                lines.append(f"| {display} | {item['technical_precision']:.4f} | {item['technical_recall']:.4f} | {item['technical_f1']:.4f} | {item['vague_precision']:.4f} | {item['vague_recall']:.4f} | {item['vague_f1']:.4f} | {item['macro_f1']:.4f} |")
    if grouped.get("folds"):
        lines += ["", "### Grouped fold status", "", "| Fold | Status | Train rows | Test rows | Test families | Overlap |", "|---:|---|---:|---:|---|---|"]
        for fold in grouped["folds"]:
            lines.append(f"| {fold['fold']} | {fold['status']} | {fold['n_train']} | {fold['n_test']} | {', '.join(fold['test_template_families'])} | {', '.join(fold['template_family_overlap']) or 'none'} |")
    lines += ["", "## Error analysis", ""]
    for model_name, model_errors in grouped.get("error_analysis_on_evaluable_folds_only", {}).items():
        lines.append(f"- {model_name}: {len(model_errors)} errors on evaluable held-out template families.")
        for error in model_errors[:10]:
            lines.append(f"  - `{error['error_type']}` ({error['template_family']}): {error['sentence']}")
    real_metrics = real_metrics or {}
    real_dataset = real_metrics.get("dataset", {}) if isinstance(real_metrics, dict) else {}
    real_status = real_metrics.get("status", "unknown") if isinstance(real_metrics, dict) else "unknown"
    lines += ["", "## Real-data status (separate from synthetic results)", "",
              f"- Existing real-data training status: `{real_status}`.",
              f"- Genuine reviewed rows after EXCLUDE/deduplication: {real_dataset.get('rows_after_exclude_and_dedupe', 'not available')}.",
              f"- Real class counts: `{json.dumps(real_dataset.get('rows_by_class', {}), sort_keys=True)}`.",
              f"- Real vendor counts: `{json.dumps(real_dataset.get('rows_by_vendor', {}), sort_keys=True)}`.",
              "- Synthetic rows were not merged into `data/labels.csv` and are not included in real-data metrics.",
              "- Real-data model performance is not claimed unless the genuine-data readiness gate and evaluation protocol support it.", "",
              "## Interpretation and limitations", "",
              "- These are out-of-fold metrics on generated examples, not on actual vendor evidence.",
              "- Stratified row-level CV can still benefit from shared templates; grouped evaluation tests transfer to unseen template families.",
              "- Grouped evaluation may have invalid folds when a test or training fold lacks a class; invalid folds and coverage are reported rather than hidden.",
              "- Synthetic sentences use controlled patterns and may be easier or more repetitive than real public disclosures.",
              "- `data/labels.csv` was not used or modified by this experiment.",
              "- Keep the production model on the rule-based fallback until a sufficiently broad, manually reviewed real dataset supports vendor-held-out evaluation.", ""]
    return "\n".join(lines)


def run(synthetic_path: str | Path = DEFAULT_SYNTHETIC,
        output_dir: str | Path = DEFAULT_MODEL_DIR, seed: int = SEED,
        n_splits: int = 5) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_synthetic(synthetic_path)
    metrics = evaluate_synthetic(rows, seed=seed, n_splits=n_splits)
    metrics["grouped_evaluation"] = evaluate_grouped_synthetic(rows, seed=seed, n_splits=n_splits)
    metrics["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["synthetic_data_path"] = str(Path(synthetic_path))
    metrics_path = output / "metrics.json"
    # Preserve existing real-data diagnostics and add a distinct synthetic section.
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = parsed
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["synthetic_experiment"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output / "experiment_report.md"
    report_path.write_text(make_report(metrics, existing), encoding="utf-8")
    grouped = metrics["grouped_evaluation"]
    print(json.dumps({"status": metrics["status"], "dataset": metrics["dataset"],
                      "overall_oof_metrics": metrics["overall_oof_metrics"],
                      "grouped_evaluation": {"status": grouped["status"], "valid_folds": grouped["valid_folds"],
                                             "invalid_folds": grouped["invalid_folds"],
                                             "evaluation_coverage": grouped["evaluation_coverage"],
                                             "metrics_on_evaluable_folds_only": grouped["metrics_on_evaluable_folds_only"]},
                      "metrics_path": str(metrics_path), "report_path": str(report_path)}, indent=2))
    return metrics


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    run(args.synthetic, args.output_dir, args.seed, args.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
