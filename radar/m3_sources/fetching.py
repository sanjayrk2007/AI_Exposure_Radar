"""HTTP access for M3.

Plan 5/M3: "Use the same politeness rules as M2 (share fetcher.py)".

M2's ``fetcher.py`` is not part of the material M3 was given and the shared
contracts do not define its call signature. M3 therefore depends only on the
small :class:`Fetcher` protocol below:

* ``load_shared_fetcher()`` wraps M2's fetcher (``radar.m2_web.fetcher``) through
  :class:`M2FetcherAdapter` when it is importable;
* otherwise :class:`PoliteFetcher` is used. It implements the SAME rules the
  plan gives M2 (declared user-agent, robots.txt check, per-host rate limit of
  at least 1.5 s, timeouts, retries, SHA-256 content hash, UTC timestamp,
  snapshot of the raw response) and never logs in, keeps cookies, or submits a
  form (GET only).

Replacing it with M2's implementation requires no change to the collectors.
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import importlib
import inspect
import os
import re
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable
from urllib.parse import urlparse

import requests

from radar.m3_sources.common import cache_root, host_in, utc_now_iso


# --------------------------------------------------------------------------- #
# Result type + protocol
# --------------------------------------------------------------------------- #
@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int                                # HTTP status; 0 = no response
    text: str = ""
    content: bytes = b""
    content_type: str = ""
    fetched_at: str = ""
    content_hash: str = ""                     # SHA-256 of the raw body
    snapshot_path: str = ""                    # relative to the cache root
    from_cache: bool = False
    headers: dict[str, str] = field(default_factory=dict)   # lower-cased keys
    blocked_reason: Optional[str] = None       # robots | access_denied | login_required | rate_limited
    error: Optional[str] = None
    truncated: bool = False

    def outcome(self) -> str:
        """``ok`` | ``blocked`` | ``missing`` | ``error`` (drives SourceReport status)."""
        if self.blocked_reason or self.status in (401, 403, 429):
            return "blocked"
        if self.status in (404, 410):
            return "missing"
        if 200 <= self.status < 300 and not self.error and not self.truncated:
            return "ok"
        return "error"

    @property
    def reason(self) -> str:
        if self.blocked_reason:
            return self.blocked_reason
        if self.error:
            return self.error
        return f"HTTP {self.status}"


@runtime_checkable
class Fetcher(Protocol):
    def fetch(self, url: str, *, vendor_id: str, kind: str = "page",
              headers: Optional[dict[str, str]] = None,
              min_interval: Optional[float] = None,
              max_bytes: Optional[int] = None) -> FetchResult:
        """GET ``url``. ``kind`` is ``"page"`` (robots.txt is enforced) or
        ``"api"`` (documented public API / feed host). Must never raise for
        network problems: report them in the result."""
        ...


# --------------------------------------------------------------------------- #
# Polite built-in fetcher
# --------------------------------------------------------------------------- #
class _NoCookies(http.cookiejar.CookiePolicy):
    """Never store or send cookies (no session state, nothing login-like)."""
    netscape = True
    rfc2965 = False
    hide_cookie2 = False

    def set_ok(self, cookie: Any, request: Any) -> bool:
        return False

    def return_ok(self, cookie: Any, request: Any) -> bool:
        return False

    def domain_return_ok(self, domain: str, request: Any) -> bool:
        return False

    def path_return_ok(self, path: str, request: Any) -> bool:
        return False


_LOGIN_PATH = re.compile(r"/(log-?in|sign-?in|sso|auth(?:enticate)?|saml|oauth)(/|$|\?)", re.I)
_EXT_BY_TYPE = (("json", "json"), ("html", "html"), ("xml", "xml"), ("pdf", "pdf"), ("text", "txt"))


class PoliteFetcher:
    """GET-only, passive, rate-limited, robots-aware HTTP client."""

    def __init__(self, *, cache_dir: Optional[Union[str, Path]] = None,
                 user_agent: str = "RadarBot (student case study, passive public fetch)",
                 min_interval: float = 1.5, timeout: float = 20.0, retries: int = 2,
                 backoff: float = 1.0, max_bytes: int = 5_000_000,
                 api_hosts: tuple[str, ...] = (), max_redirects: int = 5,
                 session: Optional[requests.Session] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 write_snapshots: bool = True) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else cache_root()
        self.user_agent = user_agent
        self.min_interval = float(min_interval)
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff = backoff
        self.max_bytes = max_bytes
        self.api_hosts = tuple(h.lower() for h in api_hosts)
        self.write_snapshots = write_snapshots
        self._clock = clock
        self._sleep = sleep
        self._session = session or requests.Session()
        self._session.cookies.set_policy(_NoCookies())
        self._session.max_redirects = max_redirects
        self._last: dict[str, float] = {}
        self._host_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._robots: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._robots_problem: dict[str, str] = {}

    @classmethod
    def from_config(cls, fetch_cfg: dict[str, Any], **overrides: Any) -> "PoliteFetcher":
        kwargs: dict[str, Any] = dict(
            user_agent=fetch_cfg.get("user_agent", "RadarBot (student case study, passive public fetch)"),
            min_interval=fetch_cfg.get("min_interval_seconds", 1.5),
            timeout=fetch_cfg.get("timeout_seconds", 20),
            retries=fetch_cfg.get("retries", 2),
            backoff=fetch_cfg.get("backoff_seconds", 1.0),
            max_bytes=fetch_cfg.get("max_bytes_default", 5_000_000),
            api_hosts=tuple(fetch_cfg.get("api_hosts", ())),
            max_redirects=fetch_cfg.get("max_redirects", 5),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- public API ----------------------------------------------------------
    def fetch(self, url: str, *, vendor_id: str, kind: str = "page",
              headers: Optional[dict[str, str]] = None,
              min_interval: Optional[float] = None,
              max_bytes: Optional[int] = None) -> FetchResult:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return FetchResult(url, url, 0, error=f"unsupported URL {url!r}", fetched_at=utc_now_iso())
        host = parsed.hostname.lower()
        interval = self.min_interval if min_interval is None else max(0.0, float(min_interval))
        # A caller may not make the client MORE aggressive than the configured floor
        # for ordinary pages; documented API hosts may use a shorter interval.
        if kind != "api" or not host_in(url, self.api_hosts):
            interval = max(interval, self.min_interval)

        if kind != "api" or not host_in(url, self.api_hosts):
            allowed, why = self._robots_allows(url, vendor_id)
            if not allowed:
                return FetchResult(url, url, 0, fetched_at=utc_now_iso(), blocked_reason="robots",
                                   error=f"robots.txt: {why}")
        return self._get(url, vendor_id, headers or {}, interval, max_bytes or self.max_bytes)

    # -- internals -----------------------------------------------------------
    def _host_lock(self, host: str) -> threading.Lock:
        with self._guard:
            return self._host_locks.setdefault(host, threading.Lock())

    def _throttle(self, host: str, interval: float) -> None:
        with self._host_lock(host):
            last = self._last.get(host)
            if last is not None:
                wait = last + interval - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last[host] = self._clock()

    def _robots_allows(self, url: str, vendor_id: str) -> tuple[bool, str]:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        with self._guard:
            known = origin in self._robots
        if not known:
            self._load_robots(origin, vendor_id)
        parser = self._robots.get(origin)
        if parser is None:
            problem = self._robots_problem.get(origin)
            return (False, problem) if problem else (True, "")
        token = self.user_agent.split()[0] if self.user_agent else "*"
        if parser.can_fetch(token, url):
            return True, ""
        return False, f"disallowed for {token}"

    def _load_robots(self, origin: str, vendor_id: str) -> None:
        raw = self._get(origin + "/robots.txt", vendor_id, {}, self.min_interval, 500_000,
                        snapshot=False)
        parser: Optional[urllib.robotparser.RobotFileParser] = None
        problem = ""
        if raw.status == 200 and not raw.error:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(raw.text.splitlines())
        elif 400 <= raw.status < 500:
            parser = None                       # RFC 9309: unavailable -> no restrictions
        else:                                   # 5xx / network failure -> be conservative
            problem = f"robots.txt unreachable ({raw.reason})"
        with self._guard:
            self._robots[origin] = parser
            if problem:
                self._robots_problem[origin] = problem

    def _get(self, url: str, vendor_id: str, extra_headers: dict[str, str], interval: float,
             max_bytes: int, snapshot: bool = True) -> FetchResult:
        host = (urlparse(url).hostname or "").lower()
        headers = {"User-Agent": self.user_agent, "Accept": "*/*", "Accept-Encoding": "gzip, deflate"}
        headers.update(extra_headers)
        last_error = ""
        for attempt in range(self.retries + 1):
            self._throttle(host, interval)
            try:
                resp = self._session.get(url, headers=headers, timeout=self.timeout,
                                         stream=True, allow_redirects=True)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    self._sleep(self.backoff * (2 ** attempt))
                    continue
                return FetchResult(url, url, 0, fetched_at=utc_now_iso(), error=last_error)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                delay = self._retry_after(resp.headers.get("Retry-After")) or self.backoff * (2 ** attempt)
                resp.close()
                self._sleep(min(delay, 30.0))
                continue
            return self._finish(url, vendor_id, resp, max_bytes, snapshot)
        return FetchResult(url, url, 0, fetched_at=utc_now_iso(), error=last_error or "exhausted retries")

    @staticmethod
    def _retry_after(value: Optional[str]) -> float:
        try:
            return float(value) if value else 0.0
        except ValueError:
            return 0.0

    def _finish(self, url: str, vendor_id: str, resp: requests.Response, max_bytes: int,
                snapshot: bool) -> FetchResult:
        body = bytearray()
        truncated = False
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > max_bytes:
                    truncated = True
                    del body[max_bytes:]
                    break
        except requests.RequestException as exc:
            resp.close()
            return FetchResult(url, resp.url or url, resp.status_code, fetched_at=utc_now_iso(),
                               error=f"{type(exc).__name__}: {exc}")
        finally:
            resp.close()
        raw = bytes(body)
        ctype = resp.headers.get("Content-Type", "")
        encoding = resp.encoding or "utf-8"
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        final_url = resp.url or url
        result = FetchResult(
            url=url, final_url=final_url, status=resp.status_code, text=text, content=raw,
            content_type=ctype, fetched_at=utc_now_iso(), content_hash=hashlib.sha256(raw).hexdigest(),
            headers={k.lower(): v for k, v in resp.headers.items()}, truncated=truncated,
        )
        if truncated:
            result.error = f"response larger than {max_bytes} bytes"
        if resp.status_code in (401, 403):
            result.blocked_reason = "access_denied"
        elif resp.status_code == 429:
            result.blocked_reason = "rate_limited"
        elif (final_url != url and _LOGIN_PATH.search(urlparse(final_url).path or "")
              and not _LOGIN_PATH.search(urlparse(url).path or "")):
            result.blocked_reason = "login_required"
        if snapshot and self.write_snapshots and 200 <= resp.status_code < 300 and not truncated:
            result.snapshot_path = self._write_snapshot(vendor_id, url, raw, ctype)
        return result

    def _write_snapshot(self, vendor_id: str, url: str, raw: bytes, ctype: str) -> str:
        ext = next((e for key, e in _EXT_BY_TYPE if key in ctype.lower()), "bin")
        name = f"{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}.{ext}"
        rel = Path(vendor_id) / "snapshots" / name
        target = self.cache_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, target)
        return rel.as_posix()


# --------------------------------------------------------------------------- #
# Bridge to M2's shared fetcher
# --------------------------------------------------------------------------- #
class M2FetcherAdapter:
    """Wrap M2's fetcher object (whatever its exact class) behind :class:`Fetcher`.

    The adapter calls the first available of ``fetch`` / ``fetch_url`` / ``get``,
    passing only keyword arguments the callable accepts, and maps the returned
    object's attributes (several common spellings are understood) onto
    :class:`FetchResult`. A result that is already a ``FetchResult`` is passed
    through unchanged.
    """

    _FIELD_NAMES = {
        "status": ("http_status", "status_code", "status"),
        "text": ("text", "body", "html"),
        "content": ("content", "raw", "body_bytes"),
        "content_type": ("content_type", "mimetype"),
        "fetched_at": ("fetched_at", "retrieved_at", "timestamp"),
        "content_hash": ("content_hash", "sha256", "hash"),
        "snapshot_path": ("snapshot_path", "snapshot"),
        "final_url": ("final_url", "url"),
        "from_cache": ("from_cache", "cached"),
        "headers": ("headers",),
        "error": ("error",),
        "blocked_reason": ("blocked_reason",),
    }

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        for name in ("fetch", "fetch_url", "get"):
            fn = getattr(inner, name, None)
            if callable(fn):
                self._fn = fn
                break
        else:
            raise TypeError("M2 fetcher exposes none of fetch/fetch_url/get")
        try:
            self._params = set(inspect.signature(self._fn).parameters)
        except (TypeError, ValueError):
            self._params = set()

    def fetch(self, url: str, *, vendor_id: str, kind: str = "page",
              headers: Optional[dict[str, str]] = None, min_interval: Optional[float] = None,
              max_bytes: Optional[int] = None) -> FetchResult:
        wanted = {"vendor_id": vendor_id, "kind": kind, "headers": headers,
                  "min_interval": min_interval, "max_bytes": max_bytes}
        kwargs = {k: v for k, v in wanted.items() if k in self._params and v is not None}
        try:
            raw = self._fn(url, **kwargs)
        except Exception as exc:  # noqa: BLE001 - M2 errors must not crash M3
            return FetchResult(url, url, 0, fetched_at=utc_now_iso(),
                               error=f"{type(exc).__name__}: {exc}")
        if isinstance(raw, FetchResult):
            return raw
        return self._convert(url, raw)

    def _pick(self, raw: Any, key: str, default: Any = None) -> Any:
        for name in self._FIELD_NAMES[key]:
            value = raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)
            if value is not None:
                return value
        return default

    def _convert(self, url: str, raw: Any) -> FetchResult:
        status = int(self._pick(raw, "status", 0) or 0)
        content = self._pick(raw, "content", b"")
        text = self._pick(raw, "text", "")
        if isinstance(content, str):
            text, content = content, content.encode("utf-8")
        if not text and isinstance(content, (bytes, bytearray)):
            text = bytes(content).decode("utf-8", errors="replace")
        final = self._pick(raw, "final_url", url)
        hdrs = self._pick(raw, "headers", {}) or {}
        return FetchResult(
            url=url, final_url=str(final), status=status, text=str(text or ""),
            content=bytes(content or b""), content_type=str(self._pick(raw, "content_type", "") or ""),
            fetched_at=str(self._pick(raw, "fetched_at", "") or utc_now_iso()),
            content_hash=str(self._pick(raw, "content_hash", "") or ""),
            snapshot_path=str(self._pick(raw, "snapshot_path", "") or ""),
            from_cache=bool(self._pick(raw, "from_cache", False)),
            headers={str(k).lower(): str(v) for k, v in dict(hdrs).items()},
            blocked_reason=self._pick(raw, "blocked_reason"),
            error=self._pick(raw, "error"),
        )


_shared: Optional[Fetcher] = None
_shared_lock = threading.Lock()


def load_shared_fetcher(fetch_cfg: dict[str, Any], cache_dir: Optional[Union[str, Path]] = None) -> Fetcher:
    """Process-wide fetcher: M2's if importable, else :class:`PoliteFetcher`.

    One shared instance means per-host rate limits hold across collectors and
    across vendors processed on M1's thread pool."""
    global _shared
    with _shared_lock:
        if _shared is not None:
            return _shared
        fetcher: Optional[Fetcher] = None
        try:
            mod = importlib.import_module("radar.m2_web.fetcher")
        except ImportError:
            mod = None
        if mod is not None:
            try:
                factory = getattr(mod, "get_fetcher", None)
                if callable(factory):
                    inner: Any = factory()
                elif callable(getattr(mod, "Fetcher", None)):
                    inner = getattr(mod, "Fetcher")()
                else:
                    inner = mod
                fetcher = M2FetcherAdapter(inner)
            except Exception:  # noqa: BLE001 - incompatible M2 API -> use the built-in client
                fetcher = None
        if fetcher is None:
            fetcher = PoliteFetcher.from_config(fetch_cfg, cache_dir=cache_dir)
        _shared = fetcher
        return _shared


def reset_shared_fetcher() -> None:
    """Forget the process-wide fetcher (used by tests)."""
    global _shared
    with _shared_lock:
        _shared = None
