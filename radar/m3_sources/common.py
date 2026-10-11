"""Shared helpers for the M3 source collectors.

Nothing here redefines a shared contract: ``VendorProfile``, ``DepthPlan``,
``Document`` and ``SourceReport`` come from ``radar.contracts.models``.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import yaml

from radar.contracts.models import DepthPlan, SourceReport

# --------------------------------------------------------------------------- #
# Allowed values (plan 4.3). Kept as plain strings, exactly as specified.
# --------------------------------------------------------------------------- #
SOURCE_TYPES: tuple[str, ...] = (
    "TRUST", "DOCS", "WEB", "FINGERPRINT", "CAREERS", "REPOS", "FILINGS", "MANUAL",
)
M3_SOURCE_TYPES: tuple[str, ...] = ("CAREERS", "REPOS", "FILINGS", "MANUAL")

OK = "OK"
EMPTY = "EMPTY"
BLOCKED = "BLOCKED"
ERROR = "ERROR"
NOT_APPLICABLE = "NOT_APPLICABLE"
SKIPPED_BY_PLAN = "SKIPPED_BY_PLAN"
STATUSES: tuple[str, ...] = (OK, EMPTY, BLOCKED, ERROR, NOT_APPLICABLE, SKIPPED_BY_PLAN)

MANUAL_SOURCE_CLASSES: tuple[str, ...] = ("primary", "third_party")
COLLECTOR_VERSION = "m3/1.0"

VENDOR_ID_RE = re.compile(r"^V-\d{3}$")

# Evidence-integrity caveats stored in Document.meta so downstream modules and
# the UI never present capability/intent evidence as proof of deployment.
CAVEAT_CAREERS = (
    "Job postings show capability and intent, not deployment: a posting that "
    "mentions AI is not proof that AI is used in a service delivered to Meridian."
)
CAVEAT_REPOS = (
    "A dependency in a public repository may belong to samples, tooling or "
    "internal projects; it is not proof that the library is used in a service "
    "delivered to Meridian."
)
CAVEAT_FILINGS = (
    "Statements in regulatory filings are corporate disclosures (often general "
    "or risk-factor language); they are not proof that AI is used in a specific "
    "service delivered to Meridian."
)
CAVEAT_MANUAL = (
    "Analyst-submitted evidence; the text was pasted by an analyst and has not "
    "been independently re-fetched by the tool."
)


# --------------------------------------------------------------------------- #
# Time, hashing, ids
# --------------------------------------------------------------------------- #
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """ISO 8601 UTC with a trailing Z and second precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_iso() -> str:
    return iso_utc(utc_now())


def to_iso_utc(value: str) -> str:
    """Normalise a user/feed supplied date or datetime to ISO 8601 UTC.

    Accepted: ISO dates/datetimes (with or without offset, 'Z' allowed),
    day-first ``DD-MM-YYYY`` / ``DD/MM/YYYY`` (the format used in the V-000
    worked example), and epoch milliseconds / seconds as ints or digit strings.
    Naive datetimes are interpreted as UTC. Raises ``ValueError`` otherwise.
    """
    if value is None:
        raise ValueError("empty date")
    if isinstance(value, (int, float)):
        return _epoch_to_iso(float(value))
    s = str(value).strip()
    if not s:
        raise ValueError("empty date")
    if re.fullmatch(r"\d{10,13}", s):
        return _epoch_to_iso(float(s))
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return iso_utc(datetime(year, month, day, tzinfo=timezone.utc))
        except ValueError as exc:
            raise ValueError(f"invalid day-first date {s!r}") from exc
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        return iso_utc(datetime.fromisoformat(iso))
    except ValueError:
        pass
    try:
        return iso_utc(datetime.combine(date.fromisoformat(s), datetime.min.time()))
    except ValueError as exc:
        raise ValueError(f"unrecognised date {s!r}") from exc


def _epoch_to_iso(num: float) -> str:
    if num > 1e11:  # milliseconds
        num /= 1000.0
    return iso_utc(datetime.fromtimestamp(num, tz=timezone.utc))


def sha256_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_doc_id(vendor_id: str, source_type: str, *parts: str) -> str:
    """Deterministic document id: same evidence -> same id (idempotent re-runs)."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{vendor_id}-{source_type}-{digest}"


def validate_vendor_id(vendor_id: str) -> str:
    """Vendor ids look like ``V-002``. ``V-000`` is the worked example and is
    never processed (plan 4.1)."""
    if not isinstance(vendor_id, str) or not VENDOR_ID_RE.match(vendor_id):
        raise ValueError(f"invalid vendor_id {vendor_id!r}; expected like 'V-002'")
    if vendor_id == "V-000":
        raise ValueError("V-000 is the worked example and must never be processed")
    return vendor_id


# --------------------------------------------------------------------------- #
# Paths / mode
# --------------------------------------------------------------------------- #
def package_root() -> Path:
    """The ``radar/`` directory (contains config/, cache/, fixtures/)."""
    return Path(__file__).resolve().parents[1]


def config_dir() -> Path:
    return Path(os.environ.get("RADAR_CONFIG_DIR") or package_root() / "config")


def cache_root() -> Path:
    return Path(os.environ.get("RADAR_CACHE_DIR") or package_root() / "cache")


def get_mode(explicit: Optional[str] = None) -> str:
    """``RADAR_MODE`` = cache | live. Default is ``cache`` so a demo can never
    depend on live websites by accident."""
    raw = explicit if explicit is not None else os.environ.get("RADAR_MODE", "cache")
    mode = (raw or "cache").strip().lower()
    if mode not in ("cache", "live"):
        raise ValueError(f"RADAR_MODE must be 'cache' or 'live', got {raw!r}")
    return mode


# --------------------------------------------------------------------------- #
# Config + lexicon
# --------------------------------------------------------------------------- #
def load_m3_config(directory: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    path = Path(directory) if directory else config_dir()
    file = path / "m3_sources.yaml"
    if not file.is_file():
        raise FileNotFoundError(f"M3 config not found: {file}")
    with file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for section in ("common", "fetch", "careers", "github", "edgar"):
        if section not in data:
            raise ValueError(f"m3_sources.yaml is missing section {section!r}")
    return data


# Appendix C starter lexicon, used ONLY when M4's config/lexicon.yaml is absent.
_DEFAULT_LEXICON: dict[str, list[str]] = {
    "providers": ["OpenAI", "Azure OpenAI", "Anthropic", "Claude", "Google Vertex AI",
                  "Gemini", "AWS Bedrock", "SageMaker", "Hugging Face", "Cohere",
                  "Mistral", "Databricks"],
    "frameworks": ["TensorFlow", "PyTorch", "scikit-learn", "LangChain", "LlamaIndex",
                   "MLflow", "Kubeflow"],
    "techniques": ["machine learning", "large language model", "LLM", "generative AI",
                   "GenAI", "retrieval-augmented", "embeddings", "fine-tuning", "NLP",
                   "computer vision", "anomaly detection", "predictive model",
                   "neural network", "agentic", "copilot", "intelligent document processing"],
    "use_cases": ["fraud detection", "transaction scoring", "document classification",
                  "chatbot", "virtual assistant", "summarisation", "auto-drafting",
                  "forecasting"],
    "hype_terms": ["AI-powered", "next-generation", "intelligent", "smart",
                   "revolutionary", "cutting-edge"],
    "hedge_terms": ["will", "plans to", "exploring", "may", "roadmap"],
}


@dataclass(frozen=True)
class Lexicon:
    ai_terms: tuple[str, ...]
    provider_terms: tuple[str, ...]
    hype_terms: tuple[str, ...]
    hedge_terms: tuple[str, ...]
    source: str


def _flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_flatten_strings(k))
            out.extend(_flatten_strings(v))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_flatten_strings(item))
    return out


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        key = item.lower() if not (item.isupper() and len(item) <= 4) else item
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return tuple(out)


def load_lexicon(directory: Optional[Union[str, Path]] = None,
                 extra_terms: Sequence[str] = ()) -> Lexicon:
    """Read M4's ``lexicon.yaml`` tolerantly (its exact schema is M4's to define).

    Any top-level key containing ``hype`` or ``hedge`` feeds those lists; every
    other key's strings (lists, or dict keys/values) become AI terms; keys that
    mention ``provider``/``platform`` additionally feed ``provider_terms``.
    Falls back to the plan's Appendix C starter list when the file is missing.
    """
    path = (Path(directory) if directory else config_dir()) / "lexicon.yaml"
    data: Any = None
    source = "built-in Appendix C starter list (config/lexicon.yaml not found)"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            source = str(path)
        except yaml.YAMLError:
            data = None
            source = "built-in Appendix C starter list (lexicon.yaml unreadable)"
    if not isinstance(data, dict):
        data = _DEFAULT_LEXICON if data is None else {"terms": _flatten_strings(data)}
    ai: list[str] = []
    providers: list[str] = []
    hype: list[str] = []
    hedge: list[str] = []
    for key, value in data.items():
        k = str(key).lower()
        terms = _flatten_strings(value)
        if "hype" in k:
            hype.extend(terms)
        elif "hedge" in k:
            hedge.extend(terms)
        else:
            ai.extend(terms)
            if "provider" in k or "platform" in k:
                providers.extend(terms)
    ai.extend(extra_terms)
    return Lexicon(_dedupe(ai), _dedupe(providers), _dedupe(hype), _dedupe(hedge), source)


class AiTermMatcher:
    """Find AI-related terms in text.

    Short all-capital terms (AI, ML, LLM, NLP ...) are matched case-sensitively;
    everything else case-insensitively. Spaces in multi-word terms also match a
    hyphen ("machine learning" matches "machine-learning").
    """

    def __init__(self, terms: Iterable[str]) -> None:
        cs: list[str] = []
        ci: list[str] = []
        for raw in terms:
            term = raw.strip()
            if not term:
                continue
            (cs if (len(term) <= 4 and term.isalpha() and term.isupper()) else ci).append(term)
        self._cs = self._compile(cs, re.UNICODE)
        self._ci = self._compile(ci, re.UNICODE | re.IGNORECASE)

    @staticmethod
    def _compile(terms: list[str], flags: int) -> Optional[re.Pattern[str]]:
        if not terms:
            return None
        parts = []
        for term in sorted(set(terms), key=len, reverse=True):
            parts.append(re.escape(term).replace(r"\ ", r"[\s\-]+"))
        return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])", flags)

    def find(self, text: str) -> list[tuple[int, int, str]]:
        """All matches as ``(start, end, canonical_term)`` sorted by position."""
        hits: list[tuple[int, int, str]] = []
        if self._cs:
            hits.extend((m.start(), m.end(), m.group(0)) for m in self._cs.finditer(text))
        if self._ci:
            hits.extend((m.start(), m.end(), re.sub(r"[\s\-]+", " ", m.group(0).lower()))
                        for m in self._ci.finditer(text))
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        return hits

    def terms_in(self, text: str) -> list[str]:
        """Distinct canonical terms in order of first appearance."""
        seen: dict[str, None] = {}
        for _, _, term in self.find(text):
            seen.setdefault(term, None)
        return list(seen)

    def has_match(self, text: str) -> bool:
        return bool((self._cs and self._cs.search(text)) or (self._ci and self._ci.search(text)))


def build_matcher(config: dict[str, Any], directory: Optional[Union[str, Path]] = None
                  ) -> tuple[AiTermMatcher, Lexicon]:
    lex = load_lexicon(directory, config["common"].get("extra_ai_terms", []))
    return AiTermMatcher(lex.ai_terms), lex


class NoNewTermsTracker:
    """Plan A.3 stop rule: stop after N consecutive pages that add no new
    AI-related terms. (The third A.3 condition, "a Confirmed signal is found",
    needs M4's output and is therefore not evaluated inside M3.)"""

    def __init__(self, patience: int = 3) -> None:
        self.patience = max(1, int(patience))
        self.seen: set[str] = set()
        self.streak = 0

    def observe(self, terms: Iterable[str]) -> bool:
        """Record one page; return True when collection should stop."""
        new = {t.lower() for t in terms} - self.seen
        if new:
            self.seen |= new
            self.streak = 0
        else:
            self.streak += 1
        return self.streak >= self.patience


# --------------------------------------------------------------------------- #
# DepthPlan handling
# --------------------------------------------------------------------------- #
_PLAN_ALIASES = {
    "MANUAL_INBOX": "MANUAL", "MANUAL_INBOX_PROMPT": "MANUAL", "MANUAL_EVIDENCE": "MANUAL",
    "GITHUB": "REPOS", "REPOSITORIES": "REPOS", "REPO": "REPOS",
    "EDGAR": "FILINGS", "SEC": "FILINGS", "SEC_FILINGS": "FILINGS",
    "FILINGS_IF_FILER": "FILINGS", "FILING": "FILINGS",
    "CAREER": "CAREERS", "JOBS": "CAREERS", "JOB_POSTINGS": "CAREERS",
}


def planned_source_types(plan: DepthPlan) -> set[str]:
    """Normalise ``DepthPlan.sources`` (a list; element format is M5's) to the
    canonical upper-case source types. Dict items may carry the type under
    ``source_type``/``type``/``name``. Unknown items are ignored."""
    out: set[str] = set()
    for item in plan.sources or []:
        if isinstance(item, dict):
            item = item.get("source_type") or item.get("type") or item.get("name")
        if not isinstance(item, str):
            continue
        key = re.sub(r"[\s\-]+", "_", item.strip().upper())
        key = _PLAN_ALIASES.get(key, key)
        if key in SOURCE_TYPES:
            out.add(key)
    return out


def make_report(vendor_id: str, source_type: str, status: str, pages: int, note: str) -> SourceReport:
    if status not in STATUSES:
        raise ValueError(f"illegal SourceReport.status {status!r}")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"illegal source_type {source_type!r}")
    return SourceReport(vendor_id=vendor_id, source_type=source_type, status=status,
                        pages_fetched=int(pages), note=note)


class PageBudget:
    """Counts retrieved pages against ``DepthPlan.max_pages`` (per source family)."""

    def __init__(self, max_pages: int) -> None:
        self.max_pages = max(0, int(max_pages))
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_pages - self.used)

    def can_spend(self, n: int = 1) -> bool:
        return self.used + n <= self.max_pages

    def spend(self, n: int = 1) -> None:
        self.used += n


# --------------------------------------------------------------------------- #
# Names / URLs
# --------------------------------------------------------------------------- #
def normalize_website(website: str) -> Optional[str]:
    """Return ``https://host`` for a profile website such as ``www.fiserv.com``."""
    if not website or not website.strip():
        return None
    raw = website.strip()
    if "://" not in raw:
        raw = "https://" + raw
    from urllib.parse import urlparse
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if not host or "." not in host:
        return None
    return f"https://{host}"


def host_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def base_domain(host: str) -> str:
    host = host.lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def same_site(url: str, site_host: str) -> bool:
    h = host_of(url)
    b = base_domain(site_host)
    return bool(h) and (h == b or h.endswith("." + b))


def host_in(url: str, hosts: Iterable[str]) -> bool:
    h = host_of(url)
    return any(h == x or h.endswith("." + x) for x in hosts)


def normalize_company_name(name: str, suffixes: Iterable[str]) -> str:
    """Lower-case, strip punctuation and trailing legal suffixes."""
    s = re.sub(r"\(.*?\)", " ", name.lower())
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    suffix_set = {re.sub(r"[^a-z0-9 ]+", " ", x.lower()).strip() for x in suffixes}
    tokens = s.split(" ")
    # strip leading "the"
    while tokens and tokens[0] in {"the"}:
        tokens = tokens[1:]
    changed = True
    while changed and tokens:
        changed = False
        for k in (3, 2, 1):
            if len(tokens) > k and " ".join(tokens[-k:]) in suffix_set:
                tokens = tokens[:-k]
                changed = True
                break
    return " ".join(tokens)
