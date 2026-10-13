import pytest

from m4_intel.verdict import calculate_coverage, generate_verdict
from radar.contracts.models import Document, Signal, SourceReport, VendorVerdict


def doc(doc_id="D1", source_type="DOCS", text="OpenAI uses an API for fraud detection."):
    return Document(doc_id, "V-MOCK", source_type, "https://example.invalid/" + doc_id,
                    "2026-01-01T00:00:00Z", 200, "hash", "Mock", text, "", {})


def signal(signal_id, source_type, tag, *, verified=True, vendor_id="V-MOCK", providers=None, category="fraud_detection",
           specificity=0.8):
    return Signal(signal_id, vendor_id, "D-" + signal_id, source_type,
                  "https://example.invalid/" + signal_id,
                  "OpenAI uses an API for fraud detection.", 0, 12,
                  category, providers or [], specificity, 4.0, 3.2, tag, verified, "RULE")


def report(source_type, status="OK"):
    return SourceReport("V-MOCK", source_type, status, 1, "mock report")


def verdict(planned, docs, signals, reports=None):
    return generate_verdict("V-MOCK", planned, docs, reports or [], signals)


def test_coverage_counts_source_family_once_not_documents():
    coverage = calculate_coverage(["DOCS", "TRUST", "WEB"],
                                  [doc("D1", "DOCS"), doc("D2", "DOCS"), doc("D3", "DOCS")])
    assert coverage == pytest.approx(4.0 / 10.5)


def test_coverage_duplicate_planned_families_count_once():
    assert calculate_coverage(["DOCS", "DOCS"], [doc()]) == 1.0


def test_coverage_requires_usable_document_not_success_report_alone():
    assert calculate_coverage(["DOCS"], [], [report("DOCS", "OK")]) == 0.0


def test_coverage_empty_plan_is_zero():
    assert calculate_coverage([], [doc()]) == 0.0


def test_confirmed_signal_means_yes():
    result = verdict(["DOCS"], [doc()], [signal("S1", "DOCS", "CONFIRMED")])
    assert result.ai_usage == "Yes"
    assert result.strength == "Confirmed"


def test_strong_plus_probable_from_independent_source_means_yes():
    result = verdict(["DOCS", "TRUST"], [doc(source_type="DOCS"), doc("D2", "TRUST")],
                     [signal("S1", "DOCS", "STRONG"), signal("S2", "TRUST", "PROBABLE")])
    assert result.ai_usage == "Yes"


def test_two_strong_from_different_sources_means_yes():
    result = verdict(["DOCS", "TRUST"], [doc(source_type="DOCS"), doc("D2", "TRUST")],
                     [signal("S1", "DOCS", "STRONG"), signal("S2", "TRUST", "STRONG")])
    assert result.ai_usage == "Yes"


def test_strong_and_probable_from_same_source_is_not_independent():
    result = verdict(["DOCS", "TRUST"], [doc()],
                     [signal("S1", "DOCS", "STRONG"), signal("S2", "DOCS", "PROBABLE")])
    assert result.ai_usage == "Inconclusive"


def test_single_strong_does_not_mean_yes():
    result = verdict(["DOCS", "TRUST"], [doc()], [signal("S1", "DOCS", "STRONG")])
    assert result.ai_usage == "Inconclusive"


def test_no_requires_high_coverage_and_no_probable_or_better_signal():
    docs = [doc("D1", "DOCS"), doc("D2", "TRUST"), doc("D3", "WEB")]
    result = verdict(["DOCS", "TRUST", "WEB"], docs,
                     [signal("S1", "WEB", "MARKETING_ONLY", specificity=0.1)])
    assert result.ai_usage == "No"
    assert result.coverage == 1.0


def test_no_is_blocked_when_marketing_tagged_signal_is_specific():
    docs = [doc("D1", "DOCS"), doc("D2", "TRUST"), doc("D3", "WEB")]
    result = verdict(["DOCS", "TRUST", "WEB"], docs,
                     [signal("S1", "WEB", "MARKETING_ONLY", specificity=0.9)])
    assert result.ai_usage == "Inconclusive"
    assert any("possible AI use" in reason for reason in result.reasons)


def test_no_is_blocked_when_marketing_tagged_signal_names_a_provider():
    docs = [doc("D1", "DOCS"), doc("D2", "TRUST"), doc("D3", "WEB")]
    result = verdict(["DOCS", "TRUST", "WEB"], docs,
                     [signal("S1", "WEB", "MARKETING_ONLY", specificity=0.1, providers=["OpenAI"])])
    assert result.ai_usage == "Inconclusive"


def test_low_coverage_never_means_no():
    result = verdict(["DOCS", "TRUST", "WEB"], [doc()],
                     [signal("S1", "DOCS", "MARKETING_ONLY")])
    assert result.ai_usage == "Inconclusive"
    assert any("missing evidence is not treated" in reason for reason in result.reasons)


def test_unverified_signals_are_excluded_from_verdict_and_fields():
    result = verdict(["DOCS"], [doc()], [signal("S1", "DOCS", "CONFIRMED", verified=False)])
    assert result.ai_usage == "No"  # full coverage, no verified signal above marketing-only
    assert result.signal_ids == []
    assert result.providers == []


def test_duplicate_documents_do_not_inflate_coverage():
    coverage = calculate_coverage(["DOCS", "TRUST"], [doc("D1", "DOCS"), doc("D1", "DOCS")])
    assert coverage == pytest.approx(4.0 / 9.0)


def test_empty_reports_and_no_signals_are_explicitly_inconclusive_when_no_plan():
    result = verdict([], [], [], reports=[])
    assert result.ai_usage == "Inconclusive"
    assert result.coverage == 0.0
    assert any("No valid verified signals" in reason for reason in result.reasons)


def test_malformed_individual_signal_is_ignored_and_other_signal_survives():
    malformed = {"signal_id": "bad", "verified": True, "tag": "NOT_A_TAG"}
    valid = signal("S1", "DOCS", "CONFIRMED")
    result = verdict(["DOCS"], [doc()], [malformed, valid])
    assert result.ai_usage == "Yes"
    assert result.signal_ids == ["S1"]


def test_wrong_vendor_signal_does_not_affect_verdict():
    result = generate_verdict("V-OTHER", ["DOCS", "TRUST"], [doc()], [], [signal("S1", "DOCS", "CONFIRMED")])
    assert result.ai_usage == "Inconclusive"
    assert result.signal_ids == []


def test_malformed_collections_raise_clear_error():
    with pytest.raises(ValueError, match="planned_source_types"):
        generate_verdict("V-MOCK", "DOCS", [], [], [])


def test_unknown_planned_source_type_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown planned source type"):
        calculate_coverage(["UNKNOWN"], [])


def test_vendor_verdict_contract_fields_are_populated():
    result = verdict(["DOCS"], [doc()], [signal("S1", "DOCS", "CONFIRMED", providers=["OpenAI"])])
    assert isinstance(result, VendorVerdict)
    assert result.vendor_id == "V-MOCK"
    assert result.providers == ["OpenAI"]
    assert result.use_cases == ["fraud_detection"]
    assert result.reasons
