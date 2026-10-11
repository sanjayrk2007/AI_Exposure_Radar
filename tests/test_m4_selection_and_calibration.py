"""Mock-only tests (invented sentences) for fold-quality gating and calibration reporting."""
import pytest

from m4_intel.specificity import score_signal
from m4_intel.train_specificity import calibration_table, run_training
from radar.contracts.models import Signal

TASKS = ["invoice parsing", "chargeback review", "ticket routing", "claims triage", "lead scoring", "address cleanup"]
DATA = ["ledger entries", "call notes", "scanned forms", "email threads", "price lists", "audit trails"]


def _write(path, rows):
    path.write_text("vendor_id,doc_id,sentence,label\n" + "".join(
        f"{v},{d},{s},{l}\n" for v, d, s, l in rows), encoding="utf-8")


def test_single_class_vendor_folds_cannot_select_a_learned_model(tmp_path):
    rows = []
    for v in range(1, 7):
        for k in range(6):
            if v <= 3:   # these vendors contribute only TECHNICAL rows
                rows.append((f"V-00{v}", f"D{v}{k}", f"Vendor {v} currently runs OpenAI for {TASKS[k]} on {DATA[(k+v)%6]} via API", "TECHNICAL"))
            else:        # these vendors contribute only VAGUE rows
                rows.append((f"V-00{v}", f"D{v}{k}", f"Company {v} plans to explore {DATA[k]} {TASKS[(k+v)%6]} innovation someday", "VAGUE"))
    _write(tmp_path / "labels.csv", rows)
    result = run_training(tmp_path / "labels.csv", tmp_path / "m")
    assert result["model_name"] == "rule_based_fallback"
    assert result["model_summary"]["tfidf_logistic_regression"]["both_class_folds"] == 0
    assert not (tmp_path / "m" / "tfidf_logistic_regression.joblib").exists()


def test_calibration_table_is_descriptive_and_validates_input():
    table = calibration_table([0, 1, 1, 0], [0.1, 0.9, 0.7, 1.0])
    assert sum(b["n"] for b in table["bins"]) == 4
    assert table["brier_score"] == pytest.approx(((0.1) ** 2 + 0.1 ** 2 + 0.3 ** 2 + 1.0) / 4, abs=1e-4)
    with pytest.raises(ValueError):
        calibration_table([1], [0.5, 0.6])


def test_score_signal_accepts_classifier_probability_and_rejects_bad_values():
    def sig():
        return Signal("S", "V-001", "D", "TRUST", "u", "x", 0, 1, "c", [], 0.0, 0.0, 0.0, "NONE", True, "RULE")
    scored = score_signal(sig(), specificity=0.8, method="ML")
    assert scored.specificity == 0.8 and scored.method == "ML" and scored.score == pytest.approx(4.0)
    with pytest.raises(ValueError):
        score_signal(sig(), specificity=1.5)
    with pytest.raises(ValueError):
        score_signal(sig(), method="GUESS")
