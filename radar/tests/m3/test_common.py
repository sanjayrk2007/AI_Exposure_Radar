import pytest

from radar.m3_sources.common import (
    AiTermMatcher, NoNewTermsTracker, get_mode, load_lexicon, normalize_company_name, planned_source_types,
    make_report, to_iso_utc, validate_vendor_id,
)
from radar.tests.m3.fakes import plan


def test_planned_types_normalised_with_aliases():
    p = plan(sources=["Trust", "careers", "GitHub", "SEC", "manual inbox", {"type": "repos"}, "bogus", 5])
    assert planned_source_types(p) == {"TRUST", "CAREERS", "REPOS", "FILINGS", "MANUAL"}
    assert planned_source_types(plan(sources=[])) == set()


def test_mode_default_cache_and_validation(monkeypatch):
    assert get_mode() == "cache"
    monkeypatch.setenv("RADAR_MODE", "LIVE")
    assert get_mode() == "live"
    assert get_mode("cache") == "cache"
    with pytest.raises(ValueError):
        get_mode("turbo")


def test_report_status_validation():
    assert make_report("V-001", "CAREERS", "OK", 1, "n").status == "OK"
    with pytest.raises(ValueError):
        make_report("V-001", "CAREERS", "SKIPPED", 0, "n")      # not an allowed status
    with pytest.raises(ValueError):
        make_report("V-001", "NEWS", "OK", 0, "n")


def test_vendor_id_validation():
    assert validate_vendor_id("V-002") == "V-002"
    for bad in ("V-000", "v-001", "V-1", None):
        with pytest.raises(ValueError):
            validate_vendor_id(bad)


def test_dates():
    assert to_iso_utc("04-02-2026") == "2026-02-04T00:00:00Z"        # day-first (V-000 style)
    assert to_iso_utc("1700000000000") == "2023-11-14T22:13:20Z"      # epoch ms (Lever)
    assert to_iso_utc("2026-01-02T03:04:05Z") == "2026-01-02T03:04:05Z"
    with pytest.raises(ValueError):
        to_iso_utc("31-31-2026")


def test_matcher_case_rules_and_hyphen():
    m = AiTermMatcher(["AI", "machine learning", "LLM"])
    assert m.terms_in("We use AI and machine-learning; also an LLM.") == ["AI", "machine learning", "LLM"]
    assert m.terms_in("He said it was fair and paid. ml, llm and ai lowercase") == []   # short terms are case-sensitive
    assert not m.has_match("MAINTAIN")                                                   # word boundaries


def test_lexicon_builtin_fallback_and_custom(tmp_path):
    lex = load_lexicon(tmp_path)           # no lexicon.yaml here
    assert "OpenAI" in lex.ai_terms and "built-in" in lex.source
    assert "AI-powered" in lex.hype_terms and "AI-powered" not in lex.ai_terms
    (tmp_path / "lexicon.yaml").write_text("providers: [Acme AI]\nhype_terms: [shiny]\nuse_cases:\n  fraud: [fraud scoring]\n")
    lex = load_lexicon(tmp_path, ["extra"])
    assert "Acme AI" in lex.provider_terms and "fraud scoring" in lex.ai_terms and "extra" in lex.ai_terms
    assert lex.hype_terms == ("shiny",)


def test_stop_tracker_three_pages_without_new_terms():
    t = NoNewTermsTracker(3)
    assert [t.observe(x) for x in (["a"], [], ["a"], ["a"], ["b"], [], [], [])] == \
        [False, False, False, True, False, False, False, True]


def test_company_name_normalisation():
    suf = ["inc", "llc", "l l c", "company", "the", "corp"]
    assert normalize_company_name("Fiserv, Inc.", suf) == "fiserv"
    assert normalize_company_name("The Clearing House Payments Company L.L.C.", suf) == "clearing house payments"
    assert normalize_company_name("Financial Statement Services, Inc. (FSSI)", suf) == "financial statement services"
