from dataclasses import asdict

from m4_intel.candidates import extract_candidates
from m4_intel.verify import verify_candidate, verify_candidates
from radar.contracts.models import Document, Signal


TEXT = "Our service uses OpenAI for fraud detection. No AI here."
DOC = Document(
    doc_id="DOC-MOCK-1",
    vendor_id="V-MOCK-1",
    source_type="DOCS",
    url="https://example.invalid/mock-doc",
    fetched_at="2026-01-01T00:00:00Z",
    http_status=200,
    content_hash="mock-hash-not-real-evidence",
    title="Mock fixture",
    text=TEXT,
    snapshot_path="",
)
CANDIDATE = {"text": TEXT[:43], "start": 0, "end": 43, "categories": ["providers", "use_cases"]}


def test_exact_match_creates_signal_with_traceability():
    signal = verify_candidate(DOC, CANDIDATE)
    assert isinstance(signal, Signal)
    assert signal.verified is True
    assert signal.excerpt == TEXT[signal.char_start:signal.char_end]
    assert (signal.vendor_id, signal.doc_id, signal.source_type, signal.url) == (
        DOC.vendor_id, DOC.doc_id, DOC.source_type, DOC.url
    )
    assert signal.method == "RULE"
    assert signal.specificity == signal.weight == signal.score == 0.0
    assert signal.tag == "NONE"


def test_altered_quote_is_rejected():
    altered = {**CANDIDATE, "text": "Our service uses OpenAI for fraud detection!"}
    assert verify_candidate(DOC, altered) is None


def test_incorrect_offsets_are_rejected():
    assert verify_candidate(DOC, {**CANDIDATE, "start": 1}) is None
    assert verify_candidate(DOC, {**CANDIDATE, "start": -1}) is None
    assert verify_candidate(DOC, {**CANDIDATE, "end": len(TEXT) + 1}) is None
    assert verify_candidate(DOC, {**CANDIDATE, "end": 0}) is None


def test_malformed_candidates_are_rejected():
    for candidate in (None, [], {}, {"text": 2, "start": 0, "end": 1},
                      {"text": CANDIDATE["text"], "start": True, "end": 43},
                      {**CANDIDATE, "categories": "providers"}):
        assert verify_candidate(DOC, candidate) is None


def test_empty_excerpt_is_rejected():
    assert verify_candidate(DOC, {**CANDIDATE, "text": ""}) is None
    assert verify_candidate(DOC, {**CANDIDATE, "text": "   "}) is None


def test_missing_document_metadata_is_not_invented():
    incomplete = {"vendor_id": DOC.vendor_id, "doc_id": DOC.doc_id, "source_type": DOC.source_type,
                  "text": TEXT}
    assert verify_candidate(incomplete, CANDIDATE) is None


def test_mixed_valid_and_invalid_candidates_continue_independently():
    candidates = [
        {**CANDIDATE, "text": "altered quote"},
        CANDIDATE,
        None,
        CANDIDATE,
    ]
    signals = verify_candidates(DOC, candidates)
    assert len(signals) == 1
    assert signals[0].verified is True


def test_candidates_from_phase1_pass_verification():
    candidates = extract_candidates(TEXT, {"providers": ["OpenAI"], "use_cases": ["fraud detection"]})
    signals = verify_candidates(DOC, candidates)
    assert len(signals) == 1
    assert TEXT[signals[0].char_start:signals[0].char_end] == signals[0].excerpt


def test_mapping_document_is_supported():
    document = {**asdict(DOC)}
    signal = verify_candidate(document, CANDIDATE)
    assert signal is not None
    assert signal.doc_id == "DOC-MOCK-1"
