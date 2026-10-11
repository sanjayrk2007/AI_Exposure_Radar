from pathlib import Path

import pytest

from m4_intel.specificity import score_specificity_rule
from m4_intel.train_specificity import score_specificity


def test_missing_model_uses_rule_fallback(tmp_path):
    sentence = "OpenAI uses an API for fraud detection in customer data."
    assert score_specificity(sentence, tmp_path) == pytest.approx(score_specificity_rule(sentence))


def test_corrupt_model_uses_rule_fallback(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tfidf_logistic_regression.joblib").write_bytes(b"not a joblib artifact")
    sentence = "OpenAI uses an API for fraud detection in customer data."
    assert score_specificity(sentence, model_dir) == pytest.approx(score_specificity_rule(sentence))


def test_unfamiliar_text_and_empty_text_return_bounded_scores(tmp_path):
    assert 0.0 <= score_specificity("Zyxwv quokka flarn", tmp_path) <= 1.0
    assert score_specificity("", tmp_path) == 0.0


def test_non_string_input_still_fails_clearly(tmp_path):
    with pytest.raises(TypeError, match="sentence must be a string"):
        score_specificity(None, tmp_path)  # type: ignore[arg-type]
