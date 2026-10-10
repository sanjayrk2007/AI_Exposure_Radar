"""Extract terminology-matching text spans from document text.

This module only proposes text candidates. A match is not a finding or verdict
about actual AI usage. Candidate offsets refer to the original input string.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import yaml

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parents[1] / "config" / "lexicon.yaml"

# Public return contract: original excerpt and its exact character offsets,
# plus every lexicon category matched within the excerpt.
Candidate = dict[str, object]


def load_lexicon(path: str | Path = DEFAULT_LEXICON_PATH) -> dict[str, list[str]]:
    """Load and validate a category-to-terms lexicon from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Lexicon YAML must contain a mapping of categories to term lists")

    lexicon: dict[str, list[str]] = {}
    for category, terms in raw.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Lexicon category names must be non-empty strings")
        if not isinstance(terms, list) or any(not isinstance(term, str) or not term.strip() for term in terms):
            raise ValueError(f"Lexicon category {category!r} must contain a list of non-empty strings")
        # Preserve config order while removing duplicate terms case-insensitively.
        seen: set[str] = set()
        unique_terms = []
        for term in terms:
            key = term.casefold().strip()
            if key not in seen:
                seen.add(key)
                unique_terms.append(term.strip())
        lexicon[category] = unique_terms
    return lexicon


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed sentence/line spans while preserving original offsets.

    Punctuation boundaries are recognized at ., !, or ? followed by whitespace
    or end-of-text. Newlines are also boundaries, which accommodates extracted
    table rows and line-oriented PDF/HTML text. Decimal points and punctuation
    embedded in tokens are not treated as boundaries without following space.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"'’”\)\]\}]*?(?=\s|$)|\r\n|\r|\n", text):
        end = match.end()
        # For punctuation, end is already at punctuation; for newline, exclude it.
        if match.group(0) in {"\n", "\r", "\r\n"}:
            end = match.start()
        _append_trimmed_span(text, start, end, spans)
        start = match.end()
    _append_trimmed_span(text, start, len(text), spans)
    return spans


def _append_trimmed_span(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _term_pattern(term: str) -> re.Pattern[str]:
    # Use word boundaries around alphanumeric terms, while permitting phrases
    # and punctuation-bearing terms such as AI-powered and scikit-learn.
    escaped = re.escape(term.strip())
    left = r"(?<!\w)" if term[:1].isalnum() else ""
    right = r"(?!\w)" if term[-1:].isalnum() else ""
    return re.compile(left + escaped + right, flags=re.IGNORECASE)


def extract_candidates(
    document_text: str,
    lexicon: Mapping[str, Sequence[str]] | None = None,
) -> list[Candidate]:
    """Extract unique sentence/row candidates that match configured terms.

    Each result has ``text``, ``start``, ``end``, and ``categories``. ``text`` is
    exactly ``document_text[start:end]``. Categories are returned in lexicon
    insertion order. Empty text and text without terminology matches return [].
    """
    if not isinstance(document_text, str):
        raise TypeError("document_text must be a string")
    if not document_text.strip():
        return []

    active_lexicon = lexicon if lexicon is not None else load_lexicon()
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for category, terms in active_lexicon.items():
        for term in terms:
            if isinstance(term, str) and term.strip():
                compiled.append((str(category), _term_pattern(term)))

    candidates: list[Candidate] = []
    seen: set[tuple[int, int, str]] = set()
    for start, end in _sentence_spans(document_text):
        excerpt = document_text[start:end]
        categories = list(dict.fromkeys(
            category
            for category, pattern in compiled
            if pattern.search(excerpt)
        ))
        if not categories:
            continue
        identity = (start, end, excerpt)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append({"text": excerpt, "start": start, "end": end, "categories": categories})
    return candidates


def extract_document_candidates(document: Mapping[str, object], lexicon: Mapping[str, Sequence[str]] | None = None) -> list[Candidate]:
    """Convenience adapter for the shared Document mapping contract."""
    text = document.get("text", "")
    if not isinstance(text, str):
        raise TypeError("Document.text must be a string")
    return extract_candidates(text, lexicon)
