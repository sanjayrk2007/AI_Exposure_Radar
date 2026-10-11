"""Local-only LLM adjudication of ambiguous candidate sentences.

The model runs on this machine (an Ollama-compatible HTTP server on a loopback
address). Output is proposal-only: it is validated by Python, never marked as
human-reviewed, and can raise but never lower a sentence's specificity. A quote
is accepted only when it is an exact substring of the already verified excerpt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

PROMPT_VERSION = "v1"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_CACHE_DIR = Path("cache") / "llm"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
LLM_TECHNICAL_SPECIFICITY = 0.7
REASON_CODES = {
    "CURRENT_SPECIFIC_USE", "NAMES_PROVIDER", "VAGUE_MARKETING",
    "FUTURE_OR_HEDGED", "NOT_VENDOR_OWN_OFFERING", "NOT_AI",
}
REVIEW_STATUS = "LLM_PROPOSED_UNREVIEWED"

SYSTEM_PROMPT = (
    "You classify one sentence taken from a vendor's public web page. "
    "Decide whether it is TECHNICAL evidence: a current, concrete statement that AI or machine learning "
    "is used in the vendor's own product, service or operations, naming a provider, model or technique, "
    "or saying what the AI does and with what data or task. Generic claims, hype, benefits without "
    "mechanism, and future or hedged statements are not technical. "
    "Answer with JSON only, using exactly these keys: "
    '{"is_technical_evidence": true|false, '
    '"reason_code": one of CURRENT_SPECIFIC_USE, NAMES_PROVIDER, VAGUE_MARKETING, FUTURE_OR_HEDGED, '
    "NOT_VENDOR_OWN_OFFERING, NOT_AI, "
    '"providers": [names of third-party AI providers or models stated in the sentence], '
    '"quote": the shortest exact span copied from the sentence that supports your answer, or ""}. '
    "Never add information that is not in the sentence."
)


class LocalLLMError(RuntimeError):
    pass


def require_local_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in LOCAL_HOSTS:
        raise ValueError("Only a local LLM server (127.0.0.1, localhost or ::1) is permitted")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in the LLM URL are not permitted")
    return url.rstrip("/")


def validate_proposal(raw: Any, excerpt: str) -> dict[str, Any] | None:
    """Return a normalised proposal or None; the quote must be an exact substring of excerpt."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    technical = raw.get("is_technical_evidence")
    reason = raw.get("reason_code")
    quote = raw.get("quote")
    providers = raw.get("providers", [])
    if not isinstance(technical, bool) or reason not in REASON_CODES:
        return None
    if not isinstance(quote, str) or not isinstance(providers, list):
        return None
    if technical and (not quote.strip() or quote not in excerpt):
        return None
    if quote and quote not in excerpt:
        return None
    haystack = excerpt.casefold()
    kept: list[str] = []
    for provider in providers:
        if isinstance(provider, str) and provider.strip() and provider.casefold() in haystack:
            if provider.strip() not in kept:
                kept.append(provider.strip())
    return {
        "is_technical_evidence": technical,
        "reason_code": reason,
        "providers": kept,
        "quote": quote,
        "review_status": REVIEW_STATUS,
    }


class LocalLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 cache_dir: str | Path | None = None, timeout_seconds: float = 120.0,
                 session: requests.Session | None = None) -> None:
        self.base_url = require_local_url(base_url or os.environ.get("RADAR_LLM_URL", DEFAULT_BASE_URL))
        self.model = model or os.environ.get("RADAR_LLM_MODEL", DEFAULT_MODEL)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.available = True
        self.stats = {"calls": 0, "cache_hits": 0, "invalid": 0, "unavailable": 0, "accepted": 0}

    def _cache_path(self, sentence: str) -> Path:
        key = hashlib.sha256("\0".join((self.model, PROMPT_VERSION, sentence)).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _chat(self, sentence: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
        }
        try:
            response = self.session.post(self.base_url + "/api/chat", json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise LocalLLMError(f"Local LLM unreachable: {exc}") from exc
        if response.status_code != 200:
            raise LocalLLMError(f"Local LLM returned HTTP {response.status_code}")
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LocalLLMError("Local LLM response was not in the expected format") from exc
        if not isinstance(content, str):
            raise LocalLLMError("Local LLM response content was not text")
        return content

    def propose(self, sentence: str) -> dict[str, Any] | None:
        if not isinstance(sentence, str) or not sentence.strip():
            return None
        path = self._cache_path(sentence)
        content: str | None = None
        if path.is_file():
            try:
                content = json.loads(path.read_text(encoding="utf-8"))["content"]
                self.stats["cache_hits"] += 1
            except (OSError, ValueError, KeyError, TypeError):
                content = None
        if content is None:
            if not self.available:
                self.stats["unavailable"] += 1
                return None
            self.stats["calls"] += 1
            try:
                content = self._chat(sentence)
            except LocalLLMError:
                self.available = False
                self.stats["unavailable"] += 1
                return None
            proposal = validate_proposal(content, sentence)
            if proposal is None:
                self.stats["invalid"] += 1
                return None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "prompt_version": PROMPT_VERSION,
                                        "content": content}, ensure_ascii=False), encoding="utf-8")
        else:
            proposal = validate_proposal(content, sentence)
            if proposal is None:
                self.stats["invalid"] += 1
                return None
        self.stats["accepted"] += 1
        proposal["model"] = self.model
        proposal["prompt_version"] = PROMPT_VERSION
        return proposal


def make_adjudicator(llm: LocalLLM) -> Callable[[str], dict[str, Any] | None]:
    return llm.propose
