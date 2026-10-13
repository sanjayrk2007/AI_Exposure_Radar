
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from xml.etree import ElementTree

import yaml
from bs4 import BeautifulSoup

from radar.m2_web.fetcher import fetch_url


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "m2_discovery.yaml"

COMMON_PATHS = [
    "/trust",
    "/security",
    "/privacy",
    "/legal",
    "/subprocessors",
    "/ai",
    "/responsible-ai",
    "/developers",
    "/docs",
    "/api",
    "/products",
]


def _load_keywords():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        data = {}

    keywords = {}
    for source_type in ("TRUST", "DOCS", "WEB"):
        values = data.get(source_type, [])
        if isinstance(values, list):
            keywords[source_type] = [
                str(value).strip().lower()
                for value in values
                if str(value).strip()
            ]
        else:
            keywords[source_type] = []

    return keywords


def _normalize_url(url):
    url, _ = urldefrag(url.strip())
    parts = urlparse(url)

    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return ""

    if parts.username or parts.password:
        return ""

    return url


def _same_host(first, second):
    first_host = (urlparse(first).hostname or "").lower()
    second_host = (urlparse(second).hostname or "").lower()
    return bool(first_host) and first_host == second_host


def _classify(url, label="", keywords=None):
    keywords = keywords if keywords is not None else _load_keywords()
    value = f"{url} {label}".lower()

    for source_type in ("TRUST", "DOCS", "WEB"):
        if any(word in value for word in keywords.get(source_type, [])):
            return source_type

    return "WEB"


def _sitemap_urls(website):
    parts = urlparse(website)
    sitemap_url = f"{parts.scheme}://{parts.netloc}/sitemap.xml"
    result = fetch_url(sitemap_url)

    if result.get("status") != 200 or not result.get("html"):
        return []

    try:
        root = ElementTree.fromstring(result["html"])
    except ElementTree.ParseError:
        return []

    urls = []
    for element in root.iter():
        if not element.tag.endswith("loc") or not element.text:
            continue

        url = _normalize_url(element.text)
        if url and _same_host(website, url):
            urls.append(url)

    return urls


def _with_scheme(website):
    """Vendor sheets often hold bare domains ("www.example.com"); assume https."""
    website = (website or "").strip()
    if website and "://" not in website:
        website = "https://" + website.lstrip("/")
    return website


def discover_urls(website, max_pages=20, max_depth=2):
    website = _normalize_url(_with_scheme(website))
    if not website:
        return []

    max_pages = max(1, int(max_pages))
    max_depth = max(0, int(max_depth))
    website = website.rstrip("/") + "/"
    keywords = _load_keywords()

    homepage = fetch_url(website)
    if homepage.get("status") != 200 or not homepage.get("html"):
        return [(website, 0, "WEB")]

    queue = deque([(website, 0, "WEB")])
    seen = {website}
    results = []
    visited = set()

    candidates = []

    for url in _sitemap_urls(website):
        candidates.append((url, 1, _classify(url, keywords=keywords)))

    for path in COMMON_PATHS:
        url = _normalize_url(urljoin(website, path.lstrip("/")))
        if url:
            candidates.append((url, 1, _classify(url, keywords=keywords)))

    soup = BeautifulSoup(homepage["html"], "lxml")
    for link in soup.find_all("a", href=True):
        url = _normalize_url(urljoin(website, link["href"]))
        label = link.get_text(" ", strip=True)

        if url and _same_host(website, url):
            candidates.append((url, 1, _classify(url, label, keywords)))

    for url, depth, source_type in candidates:
        if len(seen) >= max_pages:
            break

        if url not in seen:
            seen.add(url)
            queue.append((url, depth, source_type))

    while queue and len(results) < max_pages:
        url, depth, source_type = queue.popleft()

        if url in visited:
            continue

        visited.add(url)
        results.append((url, depth, source_type))

        if depth >= max_depth or len(results) >= max_pages:
            continue

        if url == website:
            page = homepage
        else:
            page = fetch_url(url)

        if page.get("status") != 200 or not page.get("html"):
            continue

        soup = BeautifulSoup(page["html"], "lxml")

        for link in soup.find_all("a", href=True):
            next_url = _normalize_url(urljoin(url, link["href"]))
            label = link.get_text(" ", strip=True)

            if (
                not next_url
                or not _same_host(website, next_url)
                or next_url in seen
                or len(seen) >= max_pages
            ):
                continue

            seen.add(next_url)
            queue.append(
                (
                    next_url,
                    depth + 1,
                    _classify(next_url, label, keywords),
                )
            )

    return results
