import pytest

from m4_intel.specificity import (
    extract_providers,
    get_source_weight,
    map_use_cases,
    score_signal,
    score_specificity_rule,
    tag_signal_score,
)
from radar.contracts.models import Signal


def test_each_specificity_factor_independently():
    assert score_specificity_rule("OpenAI") == pytest.approx(0.4)
    assert score_specificity_rule("Fraud detection") == pytest.approx(0.3)
    assert score_specificity_rule("API endpoint specification") == pytest.approx(0.2)
    assert score_specificity_rule("The service uses a system.") == pytest.approx(0.1)
    assert score_specificity_rule("We will explore options.") == pytest.approx(0.0)


def test_factors_combine_and_clamp_to_unit_interval():
    sentence = "OpenAI uses an API for fraud detection in customer data."
    assert score_specificity_rule(sentence) == pytest.approx(1.0)
    assert 0.0 <= score_specificity_rule("We may explore the future roadmap.") <= 1.0


def test_future_hedge_penalty_applies_to_positive_score():
    assert score_specificity_rule("OpenAI may be used.") == pytest.approx(0.1)


def test_provider_extraction_is_case_insensitive_and_deduplicated():
    assert extract_providers("OPENAI and Anthropic; OpenAI") == ["Anthropic", "OpenAI"]


def test_use_case_mapping():
    assert map_use_cases("Uses AI for fraud detection and transaction scoring") == [
        "fraud_detection", "transaction_scoring"
    ]


@pytest.mark.parametrize("score,expected", [
    (0, "NONE"), (0.0001, "MARKETING_ONLY"), (1.9999, "MARKETING_ONLY"),
    (2.0, "PROBABLE"), (2.9999, "PROBABLE"), (3.0, "STRONG"),
    (4.4999, "STRONG"), (4.5, "CONFIRMED"), (5.0, "CONFIRMED"),
])
def test_tag_threshold_boundaries(score, expected):
    assert tag_signal_score(score) == expected


def test_source_weights_and_manual_source_classes():
    assert get_source_weight("TRUST") == 5.0
    assert get_source_weight("MANUAL", "primary") == 3.5
    assert get_source_weight("MANUAL", "third_party") == 2.0
    with pytest.raises(ValueError):
        get_source_weight("MANUAL")
    with pytest.raises(ValueError):
        get_source_weight("UNKNOWN")


def test_score_signal_uses_source_weight_separately_and_requires_verified():
    signal = Signal(
        signal_id="SIG-MOCK", vendor_id="V-MOCK", doc_id="DOC-MOCK",
        source_type="DOCS", url="https://example.invalid/mock", excerpt="OpenAI uses an API for fraud detection.",
        char_start=0, char_end=39, category="CANDIDATE", providers=[], specificity=0,
        weight=0, score=0, tag="NONE", verified=True, method="RULE",
    )
    result = score_signal(signal)
    assert result is signal
    assert result.specificity == 1.0
    assert result.weight == 4.0
    assert result.score == 4.0
    assert result.tag == "STRONG"
    assert result.providers == ["OpenAI"]
    assert result.category == "fraud_detection"

    signal.verified = False
    with pytest.raises(ValueError):
        score_signal(signal)
