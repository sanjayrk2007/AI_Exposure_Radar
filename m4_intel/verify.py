"""Exact-quote verification and structured Signal creation.

A candidate is accepted only when its excerpt is a non-empty string and its
integer offsets identify that exact excerpt in the original Document.text.
Invalid candidates are skipped independently; their quotes are never repaired.
This module performs no scoring, provider inference, or verdict assignment.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from radar.contracts.models import Signal

_REQUIRED_DOCUMENT_FIELDS = ("vendor_id", "doc_id", "source_type", "url", "text")


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _valid_document(document: object) -> tuple[str, str, str, str, str] | None:
    values = [_field(document, name) for name in _REQUIRED_DOCUMENT_FIELDS]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    vendor_id, doc_id, source_type, url, text = values
    return vendor_id, doc_id, source_type, url, text


def verify_candidate(document: object, candidate: object) -> Signal | None:
    """Return a verified Signal, or ``None`` if document/candidate is invalid.

    ``start`` is inclusive and ``end`` exclusive. No whitespace trimming,
    normalization, quote correction, or case folding is applied to excerpts.
    """
    doc = _valid_document(document)
    if doc is None or not isinstance(candidate, Mapping):
        return None

    vendor_id, doc_id, source_type, url, original_text = doc
    excerpt = candidate.get("text")
    start = candidate.get("start")
    end = candidate.get("end")
    categories = candidate.get("categories", [])

    # bool is a subclass of int, so reject it explicitly as an offset.
    if not isinstance(excerpt, str) or not excerpt.strip():
        return None
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if isinstance(end, bool) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(original_text):
        return None
    if original_text[start:end] != excerpt:
        return None
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        return None
    if any(not isinstance(category, str) or not category.strip() for category in categories):
        return None

    # Stable ID derived only from source identity and verified span.
    identity = "\0".join((vendor_id, doc_id, source_type, url, str(start), str(end), excerpt))
    signal_id = "SIG-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return Signal(
        signal_id=signal_id,
        vendor_id=vendor_id,
        doc_id=doc_id,
        source_type=source_type,
        url=url,
        excerpt=excerpt,
        char_start=start,
        char_end=end,
        category=", ".join(dict.fromkeys(categories)) or "CANDIDATE",
        providers=[],
        specificity=0.0,
        weight=0.0,
        score=0.0,
        tag="NONE",
        verified=True,
        method="RULE",
    )


def verify_candidates(document: object, candidates: object) -> list[Signal]:
    """Verify a sequence of candidates, skipping malformed items independently."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    signals: list[Signal] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            signal = verify_candidate(document, candidate)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            # Malformed candidate must not prevent processing the rest.
            continue
        if signal is not None and signal.signal_id not in seen:
            signals.append(signal)
            seen.add(signal.signal_id)
    return signals
