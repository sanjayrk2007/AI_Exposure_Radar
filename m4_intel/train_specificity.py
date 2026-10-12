"""Phase 5: vendor-held-out evaluation for sentence specificity classification.

This module never creates labels. Only reviewed rows in data/labels.csv are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from functools import lru_cache
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except Exception:
    joblib = None
    np = None
    sklearn = None
    TfidfVectorizer = None
    LogisticRegression = None
    f1_score = precision_score = recall_score = None
    Pipeline = None
    HAS_SKLEARN = False

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
# A fold whose held-out vendor has only one class gives a degenerate macro-F1, so
# model selection uses only folds with both classes and requires several of them.
MIN_BOTH_CLASS_FOLDS = 3
CALIBRATION_BINS = 5
# Plan Appendix E: a more complex model must beat the simpler rung by a margin (3 macro-F1 points).
MIN_MACRO_F1_MARGIN = 0.03
# Evaluation can run on a small pilot sample, but a learned model is only *deployed* (artifact
# written, probability multiplied into evidence scores) once the plan's labelling target is met.
# Draft/config value taken from the plan's "at least 300 labelled sentences"; lower it deliberately.
MIN_ROWS_TO_DEPLOY = 300
MODEL_FILENAME = "tfidf_logistic_regression.joblib"
MANIFEST_FILENAME = "model_manifest.json"
SYNTHETIC_LABELS_PATH = ROOT / "data" / "synthetic_labels.csv"


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
    if source.exists() and source.resolve() == SYNTHETIC_LABELS_PATH.resolve():
        raise ValueError("data/synthetic_labels.csv is synthetic and can never be used as real labels")
    if not source.exists():
        create_label_template(source)
        return []
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(CSV_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{source} must contain columns: {', '.join(CSV_COLUMNS)}")
        rows: list[dict[str, str]] = []
        seen_ids: dict[tuple[str, str, str], tuple[str, int]] = {}
        for line_number, row in enumerate(reader, start=2):
            vendor = (row.get("vendor_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if not any((vendor, doc_id, sentence, label)):
                continue
            if not all((vendor, doc_id, sentence, label)):
                raise ValueError(f"Incomplete label row at {source}:{line_number}")
            if doc_id.upper().startswith("SYN") or "SYNTHETIC" in (row.get("provenance") or "").upper():
                raise ValueError(f"Synthetic row rejected at {source}:{line_number}; synthetic data must not enter real labels")
            if vendor not in EXPECTED_VENDORS:
                raise ValueError(f"Unexpected vendor_id {vendor!r} at {source}:{line_number}; V-000 is prohibited")
            if label not in {"TECHNICAL", "VAGUE", "EXCLUDE", "1", "0"}:
                raise ValueError(f"Invalid label {label!r} at {source}:{line_number}")
            identity = (vendor, doc_id, re.sub(r"\s+", " ", sentence).casefold())
            canonical = "TECHNICAL" if label in {"TECHNICAL", "1"} else "VAGUE" if label in {"VAGUE", "0"} else "EXCLUDE"
            if identity in seen_ids:
                first_label, first_line = seen_ids[identity]
                if first_label != canonical:
                    # Never silently drop one of two human decisions: the reviewers must resolve it.
                    raise ValueError(
                        f"Conflicting labels for the same sentence at {source}:{first_line} ({first_label}) "
                        f"and {source}:{line_number} ({canonical}); resolve the conflict by hand")
                continue
            seen_ids[identity] = (canonical, line_number)
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


def calibration_table(y_true: list[int], probabilities: list[float], bins: int = CALIBRATION_BINS) -> dict[str, Any]:
    """Descriptive reliability table for out-of-fold P(TECHNICAL); not a recalibration.

    The specificity probability is multiplied into the evidence score, so it should be
    inspected. Small samples make individual bins noisy and this must be reported as such.
    """
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    table: list[dict[str, Any]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities)
                   if low <= p < high or (index == bins - 1 and p == 1.0)]
        row: dict[str, Any] = {"bin": f"[{low:.1f}, {high:.1f}{']' if index == bins - 1 else ')'}", "n": len(members)}
        if members:
            row["mean_predicted"] = round(float(np.mean([probabilities[i] for i in members])), 4)
            row["observed_technical_fraction"] = round(float(np.mean([y_true[i] for i in members])), 4)
        table.append(row)
    brier = float(np.mean([(p - y) ** 2 for y, p in zip(y_true, probabilities)])) if y_true else None
    return {"bins": table, "brier_score": None if brier is None else round(brier, 4), "n": len(y_true),
            "note": "Descriptive only; bins with few rows are unreliable. No recalibration has been applied."}


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
    both_class_scores: dict[str, list[float]] = {"rule_based": [], "tfidf_logistic_regression": []}
    calibration_truth: list[int] = []
    calibration_probs: list[float] = []
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
        test_has_both_classes = len(set(y_test)) == 2
        if test_has_both_classes:
            both_class_scores["rule_based"].append(rule_metrics["macro_f1"])
        predictions_by_model["rule_based"].extend({"vendor_id": held_out, "y_true": y, "y_pred": p}
                                                    for y, p in zip(y_test, rule_preds))

        if len({int(row["label"]) for row in train}) < 2:
            fold["tfidf_logistic_regression"] = {"status": "not_evaluable_training_has_one_class"}
        else:
            model = make_tfidf_model()
            model.fit([r["sentence"] for r in train], [int(r["label"]) for r in train])
            pred = model.predict([r["sentence"] for r in test]).astype(int).tolist()
            positive_index = list(model.named_steps["classifier"].classes_).index(1)
            probabilities = model.predict_proba([r["sentence"] for r in test])[:, positive_index]
            calibration_truth.extend(y_test)
            calibration_probs.extend(float(value) for value in probabilities)
            metrics = _metrics(y_test, pred)
            fold["tfidf_logistic_regression"] = metrics
            scores_by_model["tfidf_logistic_regression"].append(metrics["macro_f1"])
            if test_has_both_classes:
                both_class_scores["tfidf_logistic_regression"].append(metrics["macro_f1"])
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
            "both_class_folds": len(both_class_scores[model_name]),
            "macro_f1_mean_both_class_folds": (round(float(np.mean(both_class_scores[model_name])), 4)
                                               if both_class_scores[model_name] else None),
        }
    if calibration_truth and "tfidf_logistic_regression" in summary:
        summary["tfidf_logistic_regression"]["calibration_out_of_fold"] = calibration_table(
            calibration_truth, calibration_probs)
    return {"per_vendor_folds": folds, "model_summary": summary}


def make_real_report(result: dict[str, Any]) -> str:
    """Markdown report built only from a run_training result. It contains no synthetic metrics."""
    dataset = result.get("dataset", {})
    lines = ["# M4 real-label evaluation report", "",
             f"> Generated from reviewed rows in `{dataset.get('labels_path', 'data/labels.csv')}` only. Synthetic data is never used here.", "",
             f"- Status: `{result.get('status')}`", f"- Selected model: `{result.get('model_name')}`",
             f"- Reviewed rows after EXCLUDE/dedupe: {dataset.get('rows_after_exclude_and_dedupe', 0)}",
             f"- Rows by class: `{json.dumps(dataset.get('rows_by_class', {}), sort_keys=True)}`",
             f"- Rows by vendor: `{json.dumps(dataset.get('rows_by_vendor', {}), sort_keys=True)}`",
             f"- Seed: {result.get('evaluation_protocol', {}).get('seed', SEED)}; scikit-learn {sklearn.__version__}", ""]
    summary = result.get("model_summary", {})
    if not summary:
        lines += ["No performance metrics are available for this run; see limitations.", ""]
    else:
        lines += ["## Leave-one-vendor-out summary", "",
                  "| Model | Folds evaluated | Folds with both classes | Mean macro-F1 (all) | Mean macro-F1 (both-class folds) |",
                  "|---|---:|---:|---:|---:|"]
        for name, item in summary.items():
            lines.append(f"| {name} | {item.get('folds_with_metrics', 0)} | {item.get('both_class_folds', 0)} | "
                         f"{item.get('macro_f1_mean_fold')} | {item.get('macro_f1_mean_both_class_folds')} |")
        lines += ["", "## Per-vendor folds", "",
                  "| Held-out vendor | Status | n test | TECHNICAL / VAGUE | Rule macro-F1 | TF-IDF macro-F1 | Missing classes |",
                  "|---|---|---:|---|---:|---:|---|"]
        for vendor, fold in result.get("per_vendor_folds", {}).items():
            rule, tfidf = fold.get("rule_based", {}), fold.get("tfidf_logistic_regression", {})
            counts = fold.get("class_counts_test", {})
            lines.append(f"| {vendor} | {fold.get('status')} | {fold.get('n_test')} | "
                         f"{counts.get('TECHNICAL', 0)} / {counts.get('VAGUE', 0)} | {rule.get('macro_f1', 'n/a')} | "
                         f"{tfidf.get('macro_f1', tfidf.get('status', 'n/a'))} | {', '.join(rule.get('missing_classes', [])) or 'none'} |")
        calibration = summary.get("tfidf_logistic_regression", {}).get("calibration_out_of_fold")
        if calibration:
            lines += ["", "## TF-IDF out-of-fold calibration (descriptive, no recalibration applied)", "",
                      f"Brier score: {calibration['brier_score']} over {calibration['n']} held-out predictions.", "",
                      "| Bin | n | Mean predicted | Observed TECHNICAL fraction |", "|---|---:|---:|---:|"]
            for row in calibration["bins"]:
                lines.append(f"| {row['bin']} | {row['n']} | {row.get('mean_predicted', '')} | {row.get('observed_technical_fraction', '')} |")
    decision = result.get("selection_decision")
    if decision:
        lines += ["", "## Selection decision", "", f"- Selected: `{decision['selected']}`", f"- {decision['rule']}",
                  f"- Mean macro-F1 (both-class folds): rule {decision['rule_macro_f1_mean_both_class_folds']}, "
                  f"TF-IDF {decision['tfidf_macro_f1_mean_both_class_folds']}; both-class folds required: "
                  f"{decision['both_class_folds_required']} (rule {decision['both_class_folds_rule']}, TF-IDF {decision['both_class_folds_tfidf']}).",
                  f"- Evaluation favours TF-IDF: {decision.get('evaluation_favours_tfidf')}; "
                  f"deployment blocked: {decision.get('deployment_blocked_reason') or 'no'}."]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in result.get("limitations", [])] + [""]
    return "\n".join(lines)


def _remove_artifacts(output: Path) -> None:
    """Delete any saved model so a run that did not select one can never leave a stale model in use."""
    for stale in (output / MODEL_FILENAME, output / MANIFEST_FILENAME):
        stale.unlink(missing_ok=True)


def _write_outputs(result: dict[str, Any], metrics_file: Path, output: Path) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    # Keep a previously generated synthetic section under its own key; it is never merged
    # into the real-data fields or the real-data report.
    if metrics_file.exists():
        try:
            previous = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and "synthetic_experiment" in previous:
                result["synthetic_experiment"] = previous["synthetic_experiment"]
        except (OSError, json.JSONDecodeError):
            pass
    metrics_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "real_evaluation_report.md").write_text(make_real_report(result), encoding="utf-8")


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
        _remove_artifacts(output)
        result["status"] = "awaiting_manual_labels"
        result["limitations"].append("No reviewed cached-document sentences were present in labels.csv; no model was trained and no performance metrics are available.")
        result["limitations"].append("Collect sentences from actual cached vendor documents and manually label them. Do not fabricate rows to reach 300.")
        result["limitations"].append("The workspace currently has no cached vendor documents, so candidate collection cannot be performed from evidence here.")
        _write_outputs(result, metrics_file, output)
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
        _remove_artifacts(output)
        result["status"] = "INSUFFICIENT_DATA"
        result["limitations"].extend(readiness_reasons)
        result["limitations"].append(
            "No baseline evaluation or learned-model training was performed."
        )
        _write_outputs(result, metrics_file, output)
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

    # Evaluation and deployment are separate decisions. The learned model must (a) be evaluable on all
    # six folds, (b) have enough both-class folds, and (c) beat the rule baseline by the plan's margin.
    # Even then it is only deployed once enough reviewed rows exist. No tuning uses held-out labels.
    rule_mean = rule_summary.get("macro_f1_mean_fold")
    tfidf_mean = tfidf_summary.get("macro_f1_mean_fold")
    rule_both = rule_summary.get("macro_f1_mean_both_class_folds")
    tfidf_both = tfidf_summary.get("macro_f1_mean_both_class_folds")
    evaluation_favours_tfidf = bool(
        tfidf_summary.get("status") == "evaluated" and rule_mean is not None and tfidf_mean is not None
        and rule_both is not None and tfidf_both is not None
        and rule_summary.get("folds_with_metrics") == len(EXPECTED_VENDORS)
        and tfidf_summary.get("folds_with_metrics") == len(EXPECTED_VENDORS)
        and rule_summary.get("both_class_folds", 0) >= MIN_BOTH_CLASS_FOLDS
        and tfidf_summary.get("both_class_folds", 0) >= MIN_BOTH_CLASS_FOLDS
        and tfidf_both >= rule_both + MIN_MACRO_F1_MARGIN - 1e-9
        and tfidf_mean >= rule_mean)
    deployment_blocked = None
    if evaluation_favours_tfidf and len(rows) < MIN_ROWS_TO_DEPLOY:
        deployment_blocked = (f"Evaluation favours TF-IDF, but only {len(rows)} reviewed rows exist; "
                              f"at least {MIN_ROWS_TO_DEPLOY} are required before a learned model is deployed.")
    selected = "rule_based_fallback"
    if evaluation_favours_tfidf and deployment_blocked is None:
        selected = "tfidf_logistic_regression"
        final_model = make_tfidf_model()
        final_model.fit([r["sentence"] for r in rows], [int(r["label"]) for r in rows])
        save_model_artifact(final_model, output, n_rows=len(rows), labels_sha256=_file_sha256(labels_path))
    else:
        _remove_artifacts(output)
    result["selection_decision"] = {
        "selected": selected,
        "evaluation_favours_tfidf": evaluation_favours_tfidf,
        "deployment_blocked_reason": deployment_blocked,
        "min_rows_to_deploy": MIN_ROWS_TO_DEPLOY,
        "required_margin_over_rule": MIN_MACRO_F1_MARGIN,
        "rule_macro_f1_mean_fold": rule_mean, "tfidf_macro_f1_mean_fold": tfidf_mean,
        "rule_macro_f1_mean_both_class_folds": rule_both,
        "tfidf_macro_f1_mean_both_class_folds": tfidf_both,
        "both_class_folds_required": MIN_BOTH_CLASS_FOLDS,
        "both_class_folds_rule": rule_summary.get("both_class_folds", 0),
        "both_class_folds_tfidf": tfidf_summary.get("both_class_folds", 0),
        "rule": (f"TF-IDF is selected only if all six folds are evaluable, at least {MIN_BOTH_CLASS_FOLDS} folds contain both "
                 f"classes, its mean macro-F1 on both-class folds beats the rule baseline by at least {MIN_MACRO_F1_MARGIN}, "
                 f"and at least {MIN_ROWS_TO_DEPLOY} reviewed rows exist. Otherwise the rule score is used."),
    }
    if deployment_blocked:
        result["limitations"].append(deployment_blocked)
    result["model_name"] = selected
    result["status"] = "evaluated_model_selected" if selected != "rule_based_fallback" else "evaluated_rule_fallback_retained"
    result["limitations"].append("Sentence-embedding and fine-tuned encoder baselines were not run; consider them only if TF-IDF clearly underperforms on a sufficiently large reviewed sample.")
    _write_outputs(result, metrics_file, output)
    return result


def _file_sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def save_model_artifact(model: Pipeline, output_dir: str | Path, *, n_rows: int,
                        labels_sha256: str | None) -> Path:
    """Persist a model with a manifest so inference can refuse an incompatible artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / MODEL_FILENAME
    joblib.dump(model, model_path)
    manifest = {"model_name": "tfidf_logistic_regression", "scikit_learn_version": sklearn.__version__,
                "seed": SEED, "n_training_rows": n_rows, "labels_sha256": labels_sha256,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return model_path


def _load_manifest(model_dir: Path) -> dict[str, Any] | None:
    """Return the manifest only if it exists, parses, matches this scikit-learn and names a model hash.

    A missing or malformed manifest means the artifact's origin is unknown, so it is not used.
    """
    try:
        manifest = json.loads((model_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (isinstance(manifest, dict) and manifest.get("scikit_learn_version") == sklearn.__version__
            and isinstance(manifest.get("model_sha256"), str) and manifest["model_sha256"]):
        return manifest
    return None


def _manifest_compatible(model_dir: Path) -> bool:
    return _load_manifest(Path(model_dir)) is not None


@lru_cache(maxsize=4)
def _load_model_cached(path: str, mtime_ns: int, size: int, expected_sha256: str) -> Pipeline | None:
    """Load once per file version. Returns None for any unusable or hash-mismatched artifact."""
    try:
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            return None
        model = joblib.load(io.BytesIO(payload))
        classes = list(model.named_steps["classifier"].classes_)
        if 1 in classes and callable(getattr(model, "predict_proba", None)):
            return model
    except Exception:
        pass
    return None


def score_specificity_with_method(sentence: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple[float, str]:
    """Return (specificity, method) where method is ``ML`` or ``RULE``.

    The persisted TF-IDF pipeline is local-only and runs on CPU without internet.
    Missing, corrupt, incompatible, or unusable artifacts fall back to the rule
    score, which is a heuristic and not a calibrated probability.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    manifest = _load_manifest(model_dir) if sentence.strip() and model_path.is_file() else None
    if manifest is not None:
        try:
            stat = model_path.stat()
            model = _load_model_cached(str(model_path), stat.st_mtime_ns, stat.st_size, manifest["model_sha256"])
            if model is not None:
                positive_index = list(model.named_steps["classifier"].classes_).index(1)
                probability = float(model.predict_proba([sentence])[0][positive_index])
                if np.isfinite(probability):
                    return float(np.clip(probability, 0.0, 1.0)), "ML"
        except Exception:
            # A local artifact is an optional optimization, never a hard dependency.
            pass
    rule_score = float(score_specificity_rule(sentence))
    return min(1.0, max(0.0, rule_score)), "RULE"


def score_specificity(sentence: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> float:
    """Return estimated P(TECHNICAL), safely falling back to rules on bad artifacts."""
    return score_specificity_with_method(sentence, model_dir)[0]


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
