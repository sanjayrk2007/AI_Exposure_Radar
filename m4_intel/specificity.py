"""Deterministic rule-based specificity and signal metadata helpers.

Specificity is derived from sentence text only. Source credibility is applied
separately as a configurable source weight and is never included in the
specificity score itself.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from m4_intel.candidates import DEFAULT_LEXICON_PATH, load_lexicon
from radar.contracts.models import Signal

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "config" / "source_weights.yaml"
DEFAULT_USE_CASES = {
    "fraud detection": "fraud_detection",
    "transaction scoring": "transaction_scoring",
    "document classification": "document_classification",
    "chatbot": "chatbot",
    "virtual assistant": "virtual_assistant",
    "summarisation": "summarisation",
    "summarization": "summarisation",
    "auto-drafting": "auto_drafting",
    "forecasting": "forecasting",
}

# Evidence that a sentence refers to a technical artifact, rather than generic hype.
_TECHNICAL_ARTIFACTS = re.compile(
    r"\b(?:API|SDK|endpoint|REST|GraphQL|version\s*\d+(?:\.\d+)*|"
    r"configuration|config(?:uration)? file|model card|requirements\.txt|"
    r"pyproject\.toml|package\.json|inference pipeline|deployment manifest)\b",
    re.IGNORECASE,
)
# Current-use wording is deliberately a modest signal; it does not prove truth.
_CURRENT_USE = re.compile(
    r"\b(?:uses?|using|processes?|generates?|classifies?|scores?|detects?|"
    r"analy[sz]es?|summari[sz]es?|deploys?|integrates?|runs?|hosts?|"
    r"routes?|retrieves?|embeds?|trains?|supports?|provides?|is powered by|"
    r"are powered by|is integrated with|are integrated with)\b",
    re.IGNORECASE,
)
_FUTURE_OR_HEDGED = re.compile(
    r"\b(?:will|may|might|could|plans? to|planning to|exploring|intends? to|"
    r"roadmap|future|considering|expects? to|aims? to|proposes? to)\b",
    re.IGNORECASE,
)
_CONCRETE_DATA_TASK = re.compile(
    r"\b(?:customer data|transaction data|payment data|account data|documents?|"
    r"records?|invoices?|claims?|chargebacks?|transactions?|fraud|risk scoring|"
    r"classification|summari[sz]ation|forecast(?:ing)?|recommendations?|"
    r"anomaly detection|customer support|support tickets|images?|speech|"
    r"text|data sets?|datasets?)\b",
    re.IGNORECASE,
)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration {path!s} must contain a YAML mapping")
    return raw


def load_source_weights(path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, Any]:
    """Load and minimally validate the source weights and tag thresholds."""
    config = _load_yaml_mapping(path)
    weights = config.get("source_weights")
    thresholds = config.get("tag_thresholds")
    factors = config.get("specificity_factors")
    if not isinstance(weights, dict) or not isinstance(thresholds, dict) or not isinstance(factors, dict):
        raise ValueError("source_weights.yaml requires source_weights, specificity_factors, and tag_thresholds mappings")
    for key, value in weights.items():
        values = value.values() if isinstance(value, dict) else (value,)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in values):
            raise ValueError(f"Source weights for {key!r} must be non-negative numbers")
    for key, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Tag threshold {key!r} must be a non-negative number")
    return config


def _contains_term(sentence: str, term: str) -> bool:
    escaped = re.escape(term.strip())
    left = r"(?<!\w)" if term[:1].isalnum() else ""
    right = r"(?!\w)" if term[-1:].isalnum() else ""
    return re.search(left + escaped + right, sentence, flags=re.IGNORECASE) is not None


def _lexicon(path: str | Path | None = None) -> dict[str, list[str]]:
    return load_lexicon(path or DEFAULT_LEXICON_PATH)


def score_specificity_rule(sentence: str, lexicon_path: str | Path | None = None,
                            weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> float:
    """Return a deterministic text-only specificity score clamped to [0, 1].

    Each factor is counted at most once. Provider/model/framework terminology
    contributes +0.4; a concrete task/data mention +0.3; a technical artifact
    +0.2; current-use wording +0.1; future/hedged wording subtracts 0.3.
    Source type and source credibility are intentionally not inputs.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    if not sentence.strip():
        return 0.0
    config = load_source_weights(weights_path)
    factors = config["specificity_factors"]
    lexicon = _lexicon(lexicon_path)
    provider_terms = list(lexicon.get("providers", [])) + list(lexicon.get("frameworks", []))
    provider_factor = any(_contains_term(sentence, term) for term in provider_terms)
    task_factor = any(_contains_term(sentence, term) for term in lexicon.get("use_cases", [])) or bool(_CONCRETE_DATA_TASK.search(sentence))
    artifact_factor = bool(_TECHNICAL_ARTIFACTS.search(sentence))
    current_factor = bool(_CURRENT_USE.search(sentence))
    hedge_factor = bool(_FUTURE_OR_HEDGED.search(sentence))

    score = 0.0
    if provider_factor:
        score += float(factors["provider_model_framework"])
    if task_factor:
        score += float(factors["concrete_task_or_data"])
    if artifact_factor:
        score += float(factors["technical_artifact"])
    if current_factor:
        score += float(factors["present_tense_current_use"])
    if hedge_factor:
        score += float(factors["future_or_hedged"])
    return min(1.0, max(0.0, round(score, 10)))


def extract_providers(sentence: str, lexicon_path: str | Path | None = None) -> list[str]:
    """Extract provider names present in text, preserving configured lexicon order."""
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    lexicon = _lexicon(lexicon_path)
    terms = list(lexicon.get("providers", []))
    found = [term for term in terms if _contains_term(sentence, term)]
    # Prefer longer names first when one configured provider is a substring of another.
    found.sort(key=lambda term: (-len(term), terms.index(term)))
    result: list[str] = []
    for term in found:
        if not any(term.casefold() == existing.casefold() for existing in result):
            result.append(term)
    return result


def map_use_cases(sentence: str, lexicon_path: str | Path | None = None) -> list[str]:
    """Map configured use-case phrases to normalized category identifiers."""
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    lexicon = _lexicon(lexicon_path)
    aliases = dict(DEFAULT_USE_CASES)
    aliases.update({term.casefold(): re.sub(r"[^a-z0-9]+", "_", term.casefold()).strip("_")
                    for term in lexicon.get("use_cases", [])})
    results: list[str] = []
    for term in lexicon.get("use_cases", []):
        if _contains_term(sentence, term):
            category = aliases.get(term.casefold(), re.sub(r"[^a-z0-9]+", "_", term.casefold()).strip("_"))
            if category not in results:
                results.append(category)
    return results


def get_source_weight(source_type: str, source_class: str | None = None,
                      weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> float:
    """Return a configured source weight; MANUAL requires primary/third_party."""
    if not isinstance(source_type, str):
        raise TypeError("source_type must be a string")
    weights = load_source_weights(weights_path)["source_weights"]
    if source_type not in weights:
        raise ValueError(f"Unknown source type: {source_type!r}")
    value = weights[source_type]
    if isinstance(value, dict):
        if source_type != "MANUAL":
            raise ValueError(f"Unexpected tiered weight configuration for {source_type!r}")
        if source_class not in value:
            raise ValueError("MANUAL source requires source_class='primary' or 'third_party'")
        value = value[source_class]
    return float(value)


def tag_signal_score(score: float, weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> str:
    """Map a non-negative signal score to the configured draft evidence tag."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 0:
        raise ValueError("score must be a non-negative number")
    thresholds = load_source_weights(weights_path)["tag_thresholds"]
    if score == 0:
        return "NONE"
    if score >= thresholds["CONFIRMED"]:
        return "CONFIRMED"
    if score >= thresholds["STRONG"]:
        return "STRONG"
    if score >= thresholds["PROBABLE"]:
        return "PROBABLE"
    return "MARKETING_ONLY"


def score_signal(signal: Signal, source_class: str | None = None,
                 weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
                 lexicon_path: str | Path | None = None) -> Signal:
    """Populate specificity, providers, weight, score and tag on a verified Signal.

    The Signal's source_type stays unchanged. For MANUAL evidence, pass
    ``source_class`` from document metadata (``primary`` or ``third_party``).
    This function mutates and returns the same Signal object for convenient
    pipeline use. Unverified signals are rejected rather than scored.
    """
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal")
    if signal.verified is not True:
        raise ValueError("Cannot score an unverified Signal")
    config = load_source_weights(weights_path)
    specificity = score_specificity_rule(signal.excerpt, lexicon_path, weights_path)
    weight = get_source_weight(signal.source_type, source_class, weights_path)
    score = round(weight * specificity, 10)
    signal.specificity = specificity
    signal.providers = extract_providers(signal.excerpt, lexicon_path)
    signal.weight = weight
    signal.score = score
    signal.tag = tag_signal_score(score, weights_path)
    signal.category = ", ".join(map_use_cases(signal.excerpt, lexicon_path)) or signal.category
    return signal
