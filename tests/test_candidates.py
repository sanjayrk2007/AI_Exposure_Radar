from m4_intel.candidates import extract_candidates, load_lexicon


LEXICON = {
    "providers": ["OpenAI", "Azure OpenAI"],
    "techniques": ["machine learning", "LLM", "RAG"],
    "use_cases": ["fraud detection"],
    "hype_terms": ["AI-powered"],
    "future_oriented_terms": ["exploring", "will"],
}


def test_offsets_point_to_exact_original_sentence():
    text = "Intro.  Our service uses OpenAI for fraud detection! Closing text."
    candidates = extract_candidates(text, LEXICON)
    candidate = candidates[0]
    assert candidate["text"] == "Our service uses OpenAI for fraud detection!"
    assert text[candidate["start"]:candidate["end"]] == candidate["text"]
    assert candidate["categories"] == ["providers", "use_cases"]


def test_sentence_boundaries_and_categories_are_case_insensitive():
    text = "We use machine learning. An LLM helps. Nothing relevant here."
    candidates = extract_candidates(text, LEXICON)
    assert [item["text"] for item in candidates] == ["We use machine learning.", "An LLM helps."]
    assert candidates[0]["categories"] == ["techniques"]
    assert candidates[1]["categories"] == ["techniques"]


def test_duplicate_terms_and_matches_do_not_duplicate_candidate():
    lexicon = {"techniques": ["LLM", "llm", "LLM"]}
    candidates = extract_candidates("An LLM is used.", lexicon)
    assert len(candidates) == 1
    assert candidates[0]["categories"] == ["techniques"]


def test_empty_and_whitespace_text_return_empty_list():
    assert extract_candidates("", LEXICON) == []
    assert extract_candidates(" \n\t", LEXICON) == []


def test_newlines_and_table_rows_are_supported():
    text = "Provider | Use\nOpenAI | fraud detection\nOther | no match"
    candidates = extract_candidates(text, LEXICON)
    assert [item["text"] for item in candidates] == [
        "OpenAI | fraud detection",
    ]
    for candidate in candidates:
        assert text[candidate["start"]:candidate["end"]] == candidate["text"]


def test_future_and_hype_terms_are_categories_not_verdicts():
    candidates = extract_candidates("We are exploring an AI-powered assistant.", LEXICON)
    assert candidates[0]["categories"] == ["hype_terms", "future_oriented_terms"]
    assert "verdict" not in candidates[0]
    assert "ai_usage" not in candidates[0]


def test_default_lexicon_loads_from_yaml():
    lexicon = load_lexicon()
    assert "providers" in lexicon
    assert "frameworks" in lexicon
    assert "techniques" in lexicon
    assert "use_cases" in lexicon
    assert "hype_terms" in lexicon
    assert "future_oriented_terms" in lexicon


def test_sentence_boundary_after_closing_quote():
    text = 'The provider said "we use OpenAI." Next sentence.'
    candidates = extract_candidates(text, LEXICON)
    assert candidates[0]["text"] == 'The provider said "we use OpenAI."'
    assert text[candidates[0]["start"]:candidates[0]["end"]] == candidates[0]["text"]


def test_generic_intelligent_language_is_not_a_candidate_by_itself():
    text = (
        "Intelligent mail inserters improve data integrity. "
        "Our platform uses OpenAI for document classification."
    )
    candidates = extract_candidates(text, load_lexicon())

    assert all("intelligent" not in item["text"].lower() for item in candidates)
    assert any("OpenAI" in item["text"] for item in candidates)


def test_candidate_offsets_remain_exact_after_lexicon_update():
    text = "General company overview. We use generative AI for document processing."
    candidates = extract_candidates(text, load_lexicon())

    for candidate in candidates:
        assert text[candidate["start"]:candidate["end"]] == candidate["text"]

