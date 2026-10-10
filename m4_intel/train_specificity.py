"""Phase 5: vendor-held-out evaluation for sentence specificity classification.

This module never creates labels. Only reviewed rows in data/labels.csv are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from m4_intel.specificity import score_specificity_rule

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "labels.csv"
DEFAULT_MODEL_DIR = ROOT / "m4_intel" / "models" / "specificity_v1"
EXPECTED_VENDORS = tuple(f"V-{number:03d}" for number in range(1, 7))
CSV_COLUMNS = ("vendor_id", "doc_id", "sentence", "label")
SEED = 42
# Conservative pilot-data readiness checks, not guarantees of statistical validity.
MIN_LABELED_ROWS = 30
MIN_PER_CLASS = 5
MIN_VENDORS_PER_CLASS = 2


def create_label_template(path: str | Path = DEFAULT_LABELS) -> Path:
    """Create an empty, header-only label CSV; never fabricate training examples."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_COLUMNS)
    return target


def load_labels(path: str | Path = DEFAULT_LABELS) -> list[dict[str, str]]:
    """Load manually reviewed rows; EXCLUDE and exact duplicates are not trained."""
    source = Path(path)
    if not source.exists():
        create_label_template(source)
        return []
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(CSV_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{source} must contain columns: {', '.join(CSV_COLUMNS)}")
        rows: list[dict[str, str]] = []
        seen_ids: set[tuple[str, str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            vendor = (row.get("vendor_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if not any((vendor, doc_id, sentence, label)):
                continue
            if not all((vendor, doc_id, sentence, label)):
                raise ValueError(f"Incomplete label row at {source}:{line_number}")
            if vendor not in EXPECTED_VENDORS:
                raise ValueError(f"Unexpected vendor_id {vendor!r} at {source}:{line_number}; V-000 is prohibited")
            if label not in {"TECHNICAL", "VAGUE", "EXCLUDE", "1", "0"}:
                raise ValueError(f"Invalid label {label!r} at {source}:{line_number}")
            identity = (vendor, doc_id, re.sub(r"\s+", " ", sentence).casefold())
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            if label == "EXCLUDE":
                continue
            rows.append({"vendor_id": vendor, "doc_id": doc_id, "sentence": sentence,
                         "label": "1" if label in {"TECHNICAL", "1"} else "0"})
    return rows


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def remove_near_duplicates(rows: list[dict[str, str]], threshold: float = 0.92) -> tuple[list[dict[str, str]], int]:
    """Deterministically keep the first sentence in each near-duplicate cluster.

    Runs before vendor folds, as requested. It uses no learned/vectorizer state.
    Similarity uses normalized character sequence and token Jaccard similarity.
    """
    kept: list[dict[str, str]] = []
    normalized: list[str] = []
    dropped = 0
    for row in rows:
        candidate = _normalise(row["sentence"])
        if not candidate:
            dropped += 1
            continue
        duplicate = False
        candidate_tokens = set(candidate.split())
        for prior in normalized:
            if candidate == prior:
                duplicate = True
                break
            if min(len(candidate), len(prior)) >= 30:
                seq_ratio = SequenceMatcher(None, candidate, prior, autojunk=False).ratio()
                prior_tokens = set(prior.split())
                union = candidate_tokens | prior_tokens
                jaccard = len(candidate_tokens & prior_tokens) / len(union) if union else 1.0
                if seq_ratio >= threshold or jaccard >= threshold:
                    duplicate = True
                    break
        if duplicate:
            dropped += 1
        else:
            kept.append(row)
            normalized.append(candidate)
    return kept, dropped


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels_present = sorted(set(y_true))
    return {
        "n_test": len(y_true),
        "n_technical": int(sum(y_true)),
        "n_vague": int(len(y_true) - sum(y_true)),
        "missing_classes": ["TECHNICAL" if label == 1 else "VAGUE"
                            for label in (0, 1) if label not in labels_present],
        "technical_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "technical_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def make_tfidf_model() -> Pipeline:
    """Create a fresh pipeline so vectorizer vocabulary is fit per training fold."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def _fold_evaluate(rows: list[dict[str, str]]) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    predictions_by_model: dict[str, list[dict[str, int | str]]] = {"rule_based": [], "tfidf_logistic_regression": []}
    scores_by_model: dict[str, list[float]] = {"rule_based": [], "tfidf_logistic_regression": []}
    all_vendors = list(EXPECTED_VENDORS)

    for held_out in all_vendors:
        train = [row for row in rows if row["vendor_id"] != held_out]
        test = [row for row in rows if row["vendor_id"] == held_out]
        fold: dict[str, Any] = {"held_out_vendor": held_out, "n_train": len(train), "n_test": len(test),
                                "class_counts_test": {"TECHNICAL": sum(r["label"] == "1" for r in test),
                                                       "VAGUE": sum(r["label"] == "0" for r in test)}}
        if not test:
            fold["status"] = "not_evaluable_no_test_rows"
            folds[held_out] = fold
            continue
        y_test = [int(row["label"]) for row in test]
        rule_preds = [int(score_specificity_rule(row["sentence"]) >= 0.5) for row in test]
        rule_metrics = _metrics(y_test, rule_preds)
        fold["rule_based"] = rule_metrics
        scores_by_model["rule_based"].append(rule_metrics["macro_f1"])
        predictions_by_model["rule_based"].extend({"vendor_id": held_out, "y_true": y, "y_pred": p}
                                                    for y, p in zip(y_test, rule_preds))

        if len({int(row["label"]) for row in train}) < 2:
            fold["tfidf_logistic_regression"] = {"status": "not_evaluable_training_has_one_class"}
        else:
            model = make_tfidf_model()
            model.fit([r["sentence"] for r in train], [int(r["label"]) for r in train])
            pred = model.predict([r["sentence"] for r in test]).astype(int).tolist()
            metrics = _metrics(y_test, pred)
            fold["tfidf_logistic_regression"] = metrics
            scores_by_model["tfidf_logistic_regression"].append(metrics["macro_f1"])
            predictions_by_model["tfidf_logistic_regression"].extend({"vendor_id": held_out, "y_true": y, "y_pred": p}
                                                                        for y, p in zip(y_test, pred))
        fold["status"] = "evaluated"
        folds[held_out] = fold

    summary: dict[str, Any] = {}
    for model_name, records in predictions_by_model.items():
        if not records:
            summary[model_name] = {"status": "not_evaluable", "macro_f1_mean_fold": None}
            continue
        y_true = [int(item["y_true"]) for item in records]
        y_pred = [int(item["y_pred"]) for item in records]
        summary[model_name] = {
            "status": "evaluated",
            "n_predictions": len(records),
            "overall_metrics_descriptive_only": _metrics(y_true, y_pred),
            "macro_f1_mean_fold": round(float(np.mean(scores_by_model[model_name])), 4) if scores_by_model[model_name] else None,
            "folds_with_metrics": len(scores_by_model[model_name]),
        }
    return {"per_vendor_folds": folds, "model_summary": summary}


def run_training(labels_path: str | Path = DEFAULT_LABELS,
                 output_dir: str | Path = DEFAULT_MODEL_DIR,
                 metrics_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate baselines, choose conservatively, and save a CPU inference artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(metrics_path) if metrics_path else output / "metrics.json"
    raw_rows = load_labels(labels_path)
    rows, duplicates_removed = remove_near_duplicates(raw_rows)
    label_counts = Counter(row["label"] for row in rows)
    vendor_counts = Counter(row["vendor_id"] for row in rows)
    result: dict[str, Any] = {
        "model_name": "rule_based_fallback",
        "dataset": {"labels_path": str(Path(labels_path)), "rows_after_exclude_and_dedupe": len(rows),
                    "rows_by_class": {"TECHNICAL": label_counts.get("1", 0), "VAGUE": label_counts.get("0", 0)},
                    "rows_by_vendor": dict(sorted(vendor_counts.items())),
                    "excluded_and_duplicate_rows_not_trained": True,
                    "near_duplicate_rows_removed_before_splitting": duplicates_removed,
                    "public_dataset_used": False},
        "evaluation_protocol": {"method": "six leave-one-vendor-out folds", "vendors": list(EXPECTED_VENDORS),
                                "seed": SEED, "decision_threshold": 0.5,
                                "learned_preprocessing_fit_per_training_fold": True,
                                "held_out_vendor_labels_used_for_training_or_selection": False},
        "per_vendor_folds": {}, "model_summary": {},
        "selection_rule": "Select a fine-tuned encoder only if it exceeds the best simpler model by >=0.03 macro-F1 and is no worse than -0.10 macro-F1 on every fold; encoder not run unless separately justified.",
        "limitations": [],
    }
    if len(rows) == 0:
        result["status"] = "awaiting_manual_labels"
        result["limitations"].append("No reviewed cached-document sentences were present in labels.csv; no model was trained and no performance metrics are available.")
        result["limitations"].append("Collect sentences from actual cached vendor documents and manually label them. Do not fabricate rows to reach 300.")
        result["limitations"].append("The workspace currently has no cached vendor documents, so candidate collection cannot be performed from evidence here.")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    technical_vendors = {
        row["vendor_id"] for row in rows if row["label"] == "1"
    }
    vague_vendors = {
        row["vendor_id"] for row in rows if row["label"] == "0"
    }

    readiness_reasons = []

    if len(rows) < MIN_LABELED_ROWS:
        readiness_reasons.append(
            f"Need at least {MIN_LABELED_ROWS} usable labeled rows; found {len(rows)}."
        )

    for label_name, count in (
        ("TECHNICAL", label_counts.get("1", 0)),
        ("VAGUE", label_counts.get("0", 0)),
    ):
        if count < MIN_PER_CLASS:
            readiness_reasons.append(
                f"Need at least {MIN_PER_CLASS} {label_name} rows; found {count}."
            )

    if len(technical_vendors) < MIN_VENDORS_PER_CLASS:
        readiness_reasons.append(
            "TECHNICAL examples must cover at least "
            f"{MIN_VENDORS_PER_CLASS} vendors."
        )

    if len(vague_vendors) < MIN_VENDORS_PER_CLASS:
        readiness_reasons.append(
            "VAGUE examples must cover at least "
            f"{MIN_VENDORS_PER_CLASS} vendors."
        )

    result["dataset"]["readiness"] = {
        "ready": not readiness_reasons,
        "technical_vendors": sorted(technical_vendors),
        "vague_vendors": sorted(vague_vendors),
    }

    if readiness_reasons:
        result["status"] = "INSUFFICIENT_DATA"
        result["limitations"].extend(readiness_reasons)
        result["limitations"].append(
            "No baseline evaluation or learned-model training was performed."
        )
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    missing_vendors = [vendor for vendor in EXPECTED_VENDORS if vendor_counts.get(vendor, 0) == 0]
    result["dataset"]["vendors_missing_rows"] = missing_vendors
    evaluation = _fold_evaluate(rows)
    result.update(evaluation)
    rule_summary = evaluation["model_summary"].get("rule_based", {})
    tfidf_summary = evaluation["model_summary"].get("tfidf_logistic_regression", {})
    result["status"] = "evaluated_baselines"
    result["limitations"].append("Scores are only an estimate from the manually labelled Optiv case-study sample; six vendors is a very small domain-specific evaluation.")
    if missing_vendors:
        result["limitations"].append("One or more required vendor folds have no usable labelled rows: " + ", ".join(missing_vendors))
    if any(len(set(int(r["label"]) for r in rows if r["vendor_id"] != vendor)) < 2 for vendor in EXPECTED_VENDORS):
        result["limitations"].append("At least one training fold lacks both classes; TF-IDF metrics may be missing for that fold.")

    # A learned model is selected only when its mean fold macro-F1 beats rule baseline.
    # No threshold/model tuning is done against held-out labels.
    selected = "rule_based_fallback"
    rule_mean = rule_summary.get("macro_f1_mean_fold")
    tfidf_mean = tfidf_summary.get("macro_f1_mean_fold")
    if (tfidf_summary.get("status") == "evaluated" and rule_mean is not None and tfidf_mean is not None
            and rule_summary.get("folds_with_metrics") == len(EXPECTED_VENDORS)
            and tfidf_summary.get("folds_with_metrics") == len(EXPECTED_VENDORS)
            and tfidf_mean >= rule_mean):
        selected = "tfidf_logistic_regression"
        final_model = make_tfidf_model()
        final_model.fit([r["sentence"] for r in rows], [int(r["label"]) for r in rows])
        joblib.dump(final_model, output / "tfidf_logistic_regression.joblib")
    result["model_name"] = selected
    result["status"] = "evaluated_baselines_model_selected"
    result["limitations"].append("Sentence-embedding and fine-tuned encoder baselines were not run: Phase 5 has no reviewed labels yet / no evidence they are justified; do not use a transformer until a meaningful labelled sample exists.")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def score_specificity(sentence: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> float:
    """Return estimated P(TECHNICAL) in [0,1], or rule score if model is absent.

    The persisted TF-IDF pipeline is local-only and runs on CPU without internet.
    The rule fallback is a specificity heuristic, not a calibrated probability.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    model_path = Path(model_dir) / "tfidf_logistic_regression.joblib"
    if model_path.is_file():
        model = joblib.load(model_path)
        classes = list(model.named_steps["classifier"].classes_)
        if 1 in classes:
            positive_index = classes.index(1)
            return float(np.clip(model.predict_proba([sentence])[0][positive_index], 0.0, 1.0))
    return float(np.clip(score_specificity_rule(sentence), 0.0, 1.0))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_training(args.labels, args.output_dir, args.metrics)
    print(json.dumps({"status": result["status"], "model_name": result["model_name"],
                      "dataset": result["dataset"], "metrics_path": str(args.metrics or args.output_dir / "metrics.json")}, indent=2))
    return 0 if result["status"] != "awaiting_manual_labels" else 2


if __name__ == "__main__":
    sys.exit(main())
