"""End-to-end functional tests. All sentences are invented; models here are trained on the
synthetic generator purely to exercise inference code. No result is real-vendor performance."""
import json

import joblib
import pytest

from m4_intel import train_specificity as ts
from m4_intel.analyse import analyse, analyse_detailed
from m4_intel.generate_synthetic_data import build_rows, write_dataset
from m4_intel.run_synthetic_experiment import run as run_synthetic
from m4_intel.specificity import score_specificity_rule

TECH = "The service currently uses a Hugging Face transformer for invoice classification through a versioned inference API."
VAGUE = "We are exploring how AI could transform workflows in the future."
DOCS = [
    {"vendor_id": "V-001", "doc_id": "D1", "source_type": "TRUST", "url": "https://example.invalid/t",
     "fetched_at": "2026-01-01T00:00:00Z", "content_hash": "h1", "text": f"Overview. {TECH} {VAGUE}"},
    {"vendor_id": "V-001", "doc_id": "D2", "source_type": "WEB", "url": "https://example.invalid/w",
     "fetched_at": "2026-01-01T00:00:00Z", "content_hash": "h2", "text": "Our AI-powered platform is next-generation."},
]
REPORTS = [{"vendor_id": "V-001", "source_type": s, "status": "OK", "pages_fetched": 1, "note": ""} for s in ("TRUST", "WEB")]


@pytest.fixture()
def functional_model(tmp_path):
    rows = [r for r in build_rows() if r["label"] in {"TECHNICAL", "VAGUE"}]
    model = ts.make_tfidf_model()
    model.fit([r["sentence"] for r in rows], [1 if r["label"] == "TECHNICAL" else 0 for r in rows])
    ts.save_model_artifact(model, tmp_path, n_rows=len(rows), labels_sha256=None)
    return tmp_path


def test_end_to_end_with_local_model_scores_by_classifier(functional_model):
    signals, verdict = analyse("V-001", DOCS, REPORTS, model_dir=functional_model)
    by_text = {s.excerpt: s for s in signals}
    assert by_text[TECH].method == "ML" and by_text[VAGUE].method == "ML"
    assert 0.0 <= by_text[VAGUE].specificity < by_text[TECH].specificity <= 1.0
    for s in signals:
        assert s.score == pytest.approx(s.weight * s.specificity)
        doc = next(d for d in DOCS if d["doc_id"] == s.doc_id)
        assert doc["text"][s.char_start:s.char_end] == s.excerpt and s.verified
    assert verdict.coverage == 1.0 and verdict.ai_usage in {"Yes", "Inconclusive", "No"}
    again, again_verdict = analyse("V-001", DOCS, REPORTS, model_dir=functional_model)
    assert [(s.signal_id, s.specificity, s.tag) for s in again] == [(s.signal_id, s.specificity, s.tag) for s in signals]
    assert again_verdict == verdict


def test_end_to_end_without_model_matches_rule_scores(tmp_path):
    signals, _ = analyse("V-001", DOCS, REPORTS, model_dir=tmp_path)
    assert signals and all(s.method == "RULE" and s.specificity == score_specificity_rule(s.excerpt) for s in signals)
    off, _ = analyse("V-001", DOCS, REPORTS, use_model=False)
    assert [s.specificity for s in off] == [s.specificity for s in signals]


@pytest.mark.parametrize("breakage", ["corrupt", "truncated", "version_mismatch", "wrong_object"])
def test_unusable_artifacts_fall_back_to_rule(functional_model, breakage):
    model_path = functional_model / ts.MODEL_FILENAME
    if breakage == "corrupt":
        model_path.write_bytes(b"not a model")
    elif breakage == "truncated":
        model_path.write_bytes(model_path.read_bytes()[:50])
    elif breakage == "wrong_object":
        joblib.dump({"not": "a pipeline"}, model_path)
    else:
        manifest = functional_model / ts.MANIFEST_FILENAME
        data = json.loads(manifest.read_text())
        data["scikit_learn_version"] = "0.0.0"
        manifest.write_text(json.dumps(data))
    result = analyse_detailed("V-001", DOCS, REPORTS, model_dir=functional_model)
    assert result.signals and all(s.method == "RULE" for s in result.signals)


def test_replaced_artifact_is_reloaded_not_served_from_cache(functional_model):
    first = analyse("V-001", DOCS, REPORTS, model_dir=functional_model)[0]
    (functional_model / ts.MODEL_FILENAME).write_bytes(b"replaced with junk")
    second = analyse("V-001", DOCS, REPORTS, model_dir=functional_model)[0]
    assert all(s.method == "ML" for s in first) and all(s.method == "RULE" for s in second)


def test_synthetic_rows_cannot_enter_real_labels(tmp_path, monkeypatch):
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\nV-001,SYN-0001,Invented sentence about AI,TECHNICAL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic"):
        ts.load_labels(labels)
    labels.write_text("vendor_id,doc_id,sentence,label,provenance\nV-001,D1,Invented,TECHNICAL,SYNTHETIC_GENERATED_NOT_VENDOR_EVIDENCE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic"):
        ts.load_labels(labels)
    synthetic = write_dataset(tmp_path / "synthetic_labels.csv")
    monkeypatch.setattr(ts, "SYNTHETIC_LABELS_PATH", synthetic)
    with pytest.raises(ValueError, match="synthetic"):
        ts.load_labels(synthetic)


def test_real_report_has_no_metrics_without_labels_and_keeps_synthetic_separate(tmp_path):
    synthetic = write_dataset(tmp_path / "synthetic.csv")
    run_synthetic(synthetic, tmp_path / "m")
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\n", encoding="utf-8")
    result = ts.run_training(labels, tmp_path / "m")
    assert result["status"] == "awaiting_manual_labels"
    report = (tmp_path / "m" / "real_evaluation_report.md").read_text()
    assert "No performance metrics are available" in report and "macro-F1" not in report.split("## Limitations")[0]
    metrics = json.loads((tmp_path / "m" / "metrics.json").read_text())
    assert "synthetic_experiment" in metrics and "model_summary" in metrics and metrics["model_summary"] == {}
    assert not (tmp_path / "m" / ts.MODEL_FILENAME).exists()


def test_unselected_run_removes_stale_artifact_and_records_decision(tmp_path):
    out = tmp_path / "m"
    out.mkdir()
    (out / ts.MODEL_FILENAME).write_bytes(b"stale")
    (out / ts.MANIFEST_FILENAME).write_text("{}")
    tasks = ["invoice parsing", "chargeback review", "ticket routing", "claims triage", "lead scoring", "address cleanup"]
    data = ["ledger entries", "call notes", "scanned forms", "email threads", "price lists", "audit trails"]
    lines = ["vendor_id,doc_id,sentence,label"]
    for v in range(1, 7):
        for k in range(6):
            if v <= 3:
                lines.append(f"V-00{v},D{v}{k},Vendor {v} currently runs OpenAI for {tasks[k]} on {data[(k+v)%6]} via API,TECHNICAL")
            else:
                lines.append(f"V-00{v},D{v}{k},Company {v} plans to explore {data[k]} {tasks[(k+v)%6]} innovation someday,VAGUE")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ts.run_training(labels, out)
    assert result["selection_decision"]["selected"] == "rule_based_fallback"
    assert result["status"] == "evaluated_rule_fallback_retained"
    assert not (out / ts.MODEL_FILENAME).exists() and not (out / ts.MANIFEST_FILENAME).exists()
    report = (out / "real_evaluation_report.md").read_text()
    assert "Selection decision" in report and "rule_based_fallback" in report


# ---- artifact lifecycle (task 2) -------------------------------------------------------------

def _plant_artifact(model_dir):
    rows = [r for r in build_rows() if r["label"] in {"TECHNICAL", "VAGUE"}]
    model = ts.make_tfidf_model()
    model.fit([r["sentence"] for r in rows], [1 if r["label"] == "TECHNICAL" else 0 for r in rows])
    ts.save_model_artifact(model, model_dir, n_rows=len(rows), labels_sha256=None)


@pytest.mark.parametrize("labels_body", ["", "V-001,D1,Generic AI innovation for the future,VAGUE\n"])
def test_runs_without_enough_data_remove_any_previous_artifact(tmp_path, labels_body):
    out = tmp_path / "m"
    _plant_artifact(out)
    assert (out / ts.MODEL_FILENAME).exists() and (out / ts.MANIFEST_FILENAME).exists()
    labels = tmp_path / "labels.csv"
    labels.write_text("vendor_id,doc_id,sentence,label\n" + labels_body, encoding="utf-8")
    result = ts.run_training(labels, out)
    assert result["status"] in {"awaiting_manual_labels", "INSUFFICIENT_DATA"}
    assert not (out / ts.MODEL_FILENAME).exists() and not (out / ts.MANIFEST_FILENAME).exists()
    assert ts.score_specificity_with_method(TECH, out)[1] == "RULE"


def test_artifact_without_usable_manifest_or_with_wrong_hash_is_not_used(functional_model):
    assert ts.score_specificity_with_method(TECH, functional_model)[1] == "ML"
    manifest = functional_model / ts.MANIFEST_FILENAME
    good = manifest.read_text()
    data = json.loads(good)
    assert len(data["model_sha256"]) == 64

    data_no_hash = {k: v for k, v in data.items() if k != "model_sha256"}
    manifest.write_text(json.dumps(data_no_hash))
    assert ts.score_specificity_with_method(TECH, functional_model)[1] == "RULE"

    manifest.write_text(good)
    assert ts.score_specificity_with_method(TECH, functional_model)[1] == "ML"
    data["model_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    assert ts.score_specificity_with_method(TECH, functional_model)[1] == "RULE"

    manifest.unlink()
    assert ts.score_specificity_with_method(TECH, functional_model)[1] == "RULE"


# ---- selection and deployment honesty (task 3) -----------------------------------------------

TASKS = ["invoice parsing", "chargeback review", "ticket routing", "claims triage", "lead scoring", "address cleanup"]
DATA = ["ledger entries", "call notes", "scanned forms", "email threads", "price lists", "audit trails"]


def _write_labels(path, tech_text, vague_text):
    lines = ["vendor_id,doc_id,sentence,label"]
    for v in range(1, 7):
        for k in range(3):
            lines.append(f"V-00{v},D{v}T{k},{tech_text(v, k)},TECHNICAL")
            lines.append(f"V-00{v},D{v}V{k},{vague_text(v, k)},VAGUE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tie_with_rules_keeps_rules_even_though_the_learned_model_is_perfect(tmp_path):
    # Both the rule score and TF-IDF separate these classes perfectly: no margin, so no learned model.
    _write_labels(tmp_path / "l.csv",
                  lambda v, k: f"Vendor {v} currently runs OpenAI for {TASKS[(k + v) % 6]} on {DATA[(k * 2 + v) % 6]} via API",
                  lambda v, k: f"Company {v} plans to explore {DATA[(k + v) % 6]} {TASKS[(k * 2 + v) % 6]} innovation someday")
    result = ts.run_training(tmp_path / "l.csv", tmp_path / "m")
    decision = result["selection_decision"]
    assert decision["evaluation_favours_tfidf"] is False and decision["selected"] == "rule_based_fallback"
    assert result["status"] == "evaluated_rule_fallback_retained"
    assert not (tmp_path / "m" / ts.MODEL_FILENAME).exists()


def _learnable_but_rule_blind(tmp_path):
    # No lexicon terms at all, so the rule score is 0 for everything while TF-IDF can learn the wording.
    _write_labels(tmp_path / "l.csv",
                  lambda v, k: f"Vendor {v} relies on zebra lattice scoring for {TASKS[(k + v) % 6]} using {DATA[(k * 2 + v) % 6]}",
                  lambda v, k: f"Company {v} feels happy about {DATA[(k + v) % 6]} {TASKS[(k * 2 + v) % 6]} someday maybe")
    return tmp_path / "l.csv"


def test_learned_model_that_wins_is_not_deployed_below_the_row_threshold(tmp_path):
    result = ts.run_training(_learnable_but_rule_blind(tmp_path), tmp_path / "m")
    decision = result["selection_decision"]
    assert decision["evaluation_favours_tfidf"] is True and decision["selected"] == "rule_based_fallback"
    assert "reviewed rows exist" in decision["deployment_blocked_reason"]
    assert result["status"] == "evaluated_rule_fallback_retained" and result["model_name"] == "rule_based_fallback"
    assert not (tmp_path / "m" / ts.MODEL_FILENAME).exists()


def test_learned_model_is_deployed_only_when_it_wins_and_enough_rows_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "MIN_ROWS_TO_DEPLOY", 10)
    result = ts.run_training(_learnable_but_rule_blind(tmp_path), tmp_path / "m")
    assert result["status"] == "evaluated_model_selected" and result["model_name"] == "tfidf_logistic_regression"
    assert (tmp_path / "m" / ts.MODEL_FILENAME).exists() and (tmp_path / "m" / ts.MANIFEST_FILENAME).exists()
    assert ts.score_specificity_with_method("Vendor 9 relies on zebra lattice scoring for invoice parsing", tmp_path / "m")[1] == "ML"


# ---- plan "done when": every vendor gets a verdict with and without model files (task 5) -----

@pytest.mark.parametrize("vendor_id", [f"V-00{n}" for n in range(1, 7)])
def test_every_vendor_gets_a_valid_repeatable_verdict_with_and_without_model_files(vendor_id, functional_model, tmp_path):
    docs = [dict(d, vendor_id=vendor_id, doc_id=f"{vendor_id}-{d['doc_id']}") for d in DOCS]
    reports = [dict(r, vendor_id=vendor_id) for r in REPORTS]
    for model_dir, use_model in ((functional_model, True), (tmp_path / "no_model", True), (functional_model, False)):
        signals, verdict = analyse(vendor_id, docs, reports, model_dir=model_dir, use_model=use_model)
        assert verdict.vendor_id == vendor_id and verdict.ai_usage in {"Yes", "No", "Inconclusive"}
        assert 0.0 <= verdict.coverage <= 1.0 and all(s.vendor_id == vendor_id and s.verified for s in signals)
        again_signals, again_verdict = analyse(vendor_id, docs, reports, model_dir=model_dir, use_model=use_model)
        assert again_verdict == verdict and [s.signal_id for s in again_signals] == [s.signal_id for s in signals]
