
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


CACHE_DIR = Path("cache")
USER_AGENT = "AIExposureRadarResearchBot/1.0"
MIN_HOST_DELAY = 1.5
TIMEOUT = 15
MAX_ATTEMPTS = 3

_last_request = {}
_robots_cache = {}
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _cache_path(url):
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _host_key(url):
    parts = urlparse(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _request(url):
    host = _host_key(url)
    elapsed = time.monotonic() - _last_request.get(host, 0)

    if elapsed < MIN_HOST_DELAY:
        time.sleep(MIN_HOST_DELAY - elapsed)

    response = _session.get(url, timeout=TIMEOUT)
    _last_request[host] = time.monotonic()
    return response


def _robots_allowed(url):
    host = _host_key(url)
    robots_url = host + "/robots.txt"

    if robots_url not in _robots_cache:
        try:
            response = _request(robots_url)

            if response.status_code in (404, 410):
                _robots_cache[robots_url] = True
            elif response.status_code >= 400:
                _robots_cache[robots_url] = False
            else:
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
                _robots_cache[robots_url] = parser

        except requests.RequestException:
            _robots_cache[robots_url] = False

    policy = _robots_cache[robots_url]

    if policy is True:
        return True

    if policy is False:
        return False

    return policy.can_fetch(USER_AGENT, url)


def _error_result(url, message, blocked=False, from_cache=False):
    return {
        "url": url,
        "status": 0,
        "html": "",
        "pdf_bytes": None,
        "content_hash": "",
        "fetched_at": "",
        "snapshot_path": "",
        "content_type": "",
        "error": message,
        "blocked": blocked,
        "from_cache": from_cache,
    }


def _load_cached(path, url):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))

        encoded_pdf = result.pop("pdf_base64", None)
        result["pdf_bytes"] = (
            base64.b64decode(encoded_pdf) if encoded_pdf else None
        )
        result.setdefault("url", url)
        result.setdefault("html", "")
        result.setdefault("content_hash", "")
        result.setdefault("fetched_at", "")
        result.setdefault("snapshot_path", str(path))
        result.setdefault("content_type", "")
        result.setdefault("error", "")
        result.setdefault("blocked", False)
        result["from_cache"] = True
        return result

    except (OSError, json.JSONDecodeError, ValueError):
        return _error_result(
            url, "Invalid cache entry", from_cache=True
        )


def fetch_url(url):
    parts = urlparse(url)

    if (
        parts.scheme.lower() not in ("http", "https")
        or not parts.netloc
        or parts.username
        or parts.password
    ):
        return _error_result(url, "Invalid URL")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url)

    if os.getenv("RADAR_MODE", "live").lower() == "cache":
        if not path.exists():
            return _error_result(
                url, "Not cached", from_cache=True
            )
        return _load_cached(path, url)

    if not _robots_allowed(url):
        return _error_result(
            url,
            "Robots policy disallows crawling or could not be checked",
            blocked=True,
        )

    response = None
    error = ""

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _request(url)

            if (
                response.status_code in (429, 500, 502, 503, 504)
                and attempt < MAX_ATTEMPTS - 1
            ):
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 0), 30)
                except ValueError:
                    delay = 2 ** attempt

                time.sleep(delay)
                continue

            break

        except requests.RequestException as exc:
            error = str(exc)

            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)

    if response is None:
        return _error_result(url, error or "Request failed")

    status = response.status_code
    body = response.content
    content_type = response.headers.get("Content-Type", "")
    is_pdf = (
        "application/pdf" in content_type.lower()
        or parts.path.lower().endswith(".pdf")
    )
    digest = hashlib.sha256(body).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()

    result = {
        "url": url,
        "status": status,
        "html": (
            response.text
            if status < 400 and not is_pdf
            else ""
        ),
        "pdf_bytes": body if status < 400 and is_pdf else None,
        "content_hash": digest,
        "fetched_at": fetched_at,
        "snapshot_path": str(path),
        "content_type": content_type,
        "error": "" if status < 400 else f"HTTP {status}",
        "blocked": status in (401, 403),
        "from_cache": False,
    }

    if status < 400:
        saved = dict(result)
        pdf_bytes = saved.pop("pdf_bytes", None)

        if pdf_bytes is not None:
            saved["pdf_base64"] = base64.b64encode(
                pdf_bytes
            ).decode("ascii")

        path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result
