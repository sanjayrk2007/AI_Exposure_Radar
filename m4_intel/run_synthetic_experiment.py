"""Compare rule and TF-IDF baselines on a separate synthetic M4 dataset."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from m4_intel.generate_synthetic_data import DEFAULT_OUTPUT as DEFAULT_SYNTHETIC
from m4_intel.train_specificity import SEED, make_tfidf_model
from m4_intel.specificity import score_specificity_rule

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "m4_intel" / "models" / "specificity_v1"


def load_synthetic(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"synthetic_id", "sentence", "label", "scenario", "provenance"}
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
    return {
        "n": len(y_true),
        "class_counts_true": {"VAGUE": int(sum(y == 0 for y in y_true)),
                               "TECHNICAL": int(sum(y == 1 for y in y_true))},
        "technical_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
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
            "limitation": "Synthetic cross-validation measures performance on generated patterns only; it does not estimate generalization to real vendors.",
        },
        "overall_oof_metrics": summaries,
        "folds": fold_records,
    }


def make_report(metrics: dict[str, Any]) -> str:
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
        "| Model | Technical precision | Technical recall | Technical F1 | Macro-F1 | Confusion matrix (true rows: VAGUE, TECHNICAL; predicted columns: VAGUE, TECHNICAL) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, display in (("rule_based", "Rule-based baseline"), ("tfidf_logistic_regression", "TF-IDF + Logistic Regression")):
        item = metrics["overall_oof_metrics"][name]
        lines.append(f"| {display} | {item['technical_precision']:.4f} | {item['technical_recall']:.4f} | {item['technical_f1']:.4f} | {item['macro_f1']:.4f} | `{item['confusion_matrix']}` |")
    lines += ["", "## Fold details", "", "| Fold | Train rows | Test rows | Rule macro-F1 | TF-IDF macro-F1 |", "|---:|---:|---:|---:|---:|"]
    for fold in metrics["folds"]:
        lines.append(f"| {fold['fold']} | {fold['n_train']} | {fold['n_test']} | {fold['rule_based']['macro_f1']:.4f} | {fold['tfidf_logistic_regression']['macro_f1']:.4f} |")
    lines += ["", "## Interpretation and limitations", "",
              "- These are out-of-fold metrics on generated examples, not on actual vendor evidence.",
              "- Synthetic sentences use controlled patterns and may be easier or more repetitive than real public disclosures.",
              "- The model may learn template or vocabulary cues; high scores are not proof of robust AI-usage detection.",
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
    report_path.write_text(make_report(metrics), encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "dataset": metrics["dataset"],
                      "overall_oof_metrics": metrics["overall_oof_metrics"],
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
