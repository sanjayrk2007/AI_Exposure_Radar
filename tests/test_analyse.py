"""Tests use invented mock sentences only; they are not vendor evidence."""
import pytest

from m4_intel import analyse as analyse_module
from m4_intel.analyse import analyse, analyse_detailed
from radar.contracts.models import Document, SourceReport

STRONG = "Azure OpenAI is used through an API for fraud detection on customer data."
HYPE = "Our AI-powered platform is next-generation."


def doc(doc_id, source_type, text, vendor="V-001", meta=None):
    return Document(doc_id, vendor, source_type, f"https://example.invalid/{doc_id}",
                    "2026-01-01T00:00:00Z", 200, "h", "t", text, "", meta or {})


def run(docs, reports=(), planned=None, **kw):
    return analyse({"vendor_id": "V-001"}, docs, reports, planned, use_model=False, **kw)


def test_confirmed_trust_signal_gives_yes_and_every_excerpt_is_verbatim():
    d = doc("D1", "TRUST", "Intro. " + STRONG)
    signals, verdict = run([d], planned=["TRUST"])
    assert verdict.ai_usage == "Yes" and verdict.strength == "Confirmed"
    assert signals and all(s.verified and d.text[s.char_start:s.char_end] == s.excerpt for s in signals)
    assert all(s.method == "RULE" for s in signals)


def test_marketing_only_with_high_coverage_is_no_but_low_coverage_is_inconclusive():
    d = doc("D1", "DOCS", HYPE)
    assert run([d], planned=["DOCS"])[1].ai_usage == "No"
    assert run([d], planned=["DOCS", "TRUST"])[1].ai_usage == "Inconclusive"


def test_no_documents_is_inconclusive_never_no():
    _, verdict = run([], reports=[SourceReport("V-001", "TRUST", "BLOCKED", 0, "blocked")])
    assert verdict.ai_usage == "Inconclusive" and verdict.coverage == 0.0


def test_planned_sources_derived_from_reports_skip_not_applicable():
    reports = [SourceReport("V-001", "DOCS", "OK", 1, ""), SourceReport("V-001", "FILINGS", "NOT_APPLICABLE", 0, "")]
    _, verdict = run([doc("D1", "DOCS", HYPE)], reports)
    assert verdict.coverage == 1.0


def test_example_vendor_is_refused_and_other_vendor_documents_are_ignored():
    with pytest.raises(ValueError, match="V-000"):
        analyse("V-000", [], [])
    signals, _ = run([doc("D1", "TRUST", STRONG, vendor="V-002")], planned=["TRUST"])
    assert signals == []


def test_output_is_deterministic():
    docs = [doc("D1", "TRUST", STRONG), doc("D2", "WEB", HYPE)]
    first, second = run(docs, planned=["TRUST", "WEB"]), run(docs, planned=["TRUST", "WEB"])
    assert [s.signal_id for s in first[0]] == [s.signal_id for s in second[0]]
    assert first[1] == second[1]


def test_manual_without_source_class_uses_lower_third_party_weight():
    signals, _ = run([doc("D1", "MANUAL", STRONG)], planned=["MANUAL"])
    assert signals[0].weight == 2.0


def test_missing_model_falls_back_to_rule(tmp_path):
    signals, _ = analyse("V-001", [doc("D1", "TRUST", STRONG)], planned_sources=["TRUST"], model_dir=tmp_path)
    assert all(s.method == "RULE" for s in signals)


def test_classifier_rule_disagreement_is_flagged_for_review(monkeypatch):
    monkeypatch.setattr(analyse_module, "score_specificity_with_method", lambda sentence, model_dir: (0.95, "ML"))
    result = analyse_detailed("V-001", [doc("D1", "WEB", HYPE)], planned_sources=["WEB"])
    assert result.signals[0].method == "ML" and result.signals[0].specificity == 0.95
    assert result.signals[0].signal_id in result.ambiguous
