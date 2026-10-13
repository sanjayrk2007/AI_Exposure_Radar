"""HTML and text helpers for M3 (no network access here)."""
from __future__ import annotations

import bisect
import html as html_lib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urldefrag, urljoin

import lxml.html
from lxml import etree

from radar.m3_sources.common import AiTermMatcher

_BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
    "table", "section", "article", "header", "footer", "blockquote", "pre", "dl",
    "dt", "dd", "hr", "main", "aside", "nav", "form", "fieldset", "figure",
    "figcaption", "tbody", "thead", "tfoot", "address", "details", "summary",
}
_CELL_TAGS = {"td", "th"}
_DROP_TAGS = ("script", "style", "noscript", "svg", "template", "iframe", "head")


_XML_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff]")


def normalize_ws(text: str) -> str:
    """Collapse runs of spaces/tabs; keep single newlines; squeeze blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def _parse_html(html: str) -> Optional[lxml.html.HtmlElement]:
    if not html or not html.strip():
        return None
    html = _XML_ILLEGAL_CHAR_RE.sub("", html)
    try:
        return lxml.html.fromstring(html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return None


def html_to_text(html: str) -> str:
    """Visible text of an HTML document or fragment, with a newline at block
    boundaries so paragraphs stay separate (and inline tags never split a
    sentence). Scripts, styles and hidden inline-XBRL headers are dropped."""
    root = _parse_html(html)
    if root is None:
        return ""
    for el in list(root.iter()):
        tag = el.tag
        if not isinstance(tag, str):  # comments / processing instructions
            if el.getparent() is not None:
                _drop(el)
            continue
        low = tag.lower()
        style = (el.get("style") or "").replace(" ", "").lower()
        if low in _DROP_TAGS or low == "ix:header" or "display:none" in style:
            if el.getparent() is not None or el is root:
                _drop(el)
    # Source-code whitespace (including wrapped lines) is not a line break in HTML.
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        parent = el.getparent()
        if el.tag.lower() != "pre":
            el.text = re.sub(r"\s+", " ", el.text) if el.text else el.text
        if el.tail and not (parent is not None and isinstance(parent.tag, str) and parent.tag.lower() == "pre"):
            el.tail = re.sub(r"\s+", " ", el.tail)
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        low = tag.lower()
        if low in _BLOCK_TAGS:
            el.text = "\n" + (el.text or "")
            el.tail = "\n" + (el.tail or "")
        elif low in _CELL_TAGS:
            el.tail = " " + (el.tail or "")
    return normalize_ws(root.text_content())


def _drop(el: Any) -> None:
    """Remove ``el`` but keep its tail text."""
    parent = el.getparent()
    if parent is None:
        return
    tail = _XML_ILLEGAL_CHAR_RE.sub("", el.tail or "")
    if tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = _XML_ILLEGAL_CHAR_RE.sub("", (prev.tail or "") + tail)
        else:
            parent.text = _XML_ILLEGAL_CHAR_RE.sub("", (parent.text or "") + tail)
    parent.remove(el)


def html_title(html: str) -> str:
    root = _parse_html(html)
    if root is None:
        return ""
    found = root.xpath("//title")
    if found and found[0].text_content().strip():
        return normalize_ws(found[0].text_content())
    h1 = root.xpath("//h1")
    return normalize_ws(h1[0].text_content()) if h1 else ""


@dataclass(frozen=True)
class Link:
    url: str
    text: str


def extract_links(html: str, base_url: str) -> list[Link]:
    """Absolute http(s) links (anchors only), fragment-stripped, de-duplicated."""
    root = _parse_html(html)
    if root is None:
        return []
    seen: set[str] = set()
    out: list[Link] = []
    for a in root.xpath("//a[@href]"):
        href = (a.get("href") or "").strip()
        if not href or href.lower().startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        url = urldefrag(urljoin(base_url, href))[0]
        if not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append(Link(url=url, text=normalize_ws(a.text_content())[:200]))
    return out


def extract_embedded_urls(html: str, base_url: str) -> list[str]:
    """Every URL-bearing reference (anchors, iframes, scripts, links) for ATS detection."""
    root = _parse_html(html)
    if root is None:
        return []
    urls: list[str] = []
    for el in root.xpath("//*[@href or @src]"):
        for attr in ("href", "src"):
            val = (el.get(attr) or "").strip()
            if val and not val.lower().startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
                urls.append(urljoin(base_url, val))
    return urls


def extract_jobposting_jsonld(html: str) -> Optional[dict[str, Any]]:
    """First schema.org ``JobPosting`` object found in JSON-LD blocks, if any."""
    root = _parse_html(html)
    if root is None:
        return None
    for script in root.xpath('//script[@type="application/ld+json"]'):
        raw = script.text_content().strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        found = _find_jobposting(data)
        if found is not None:
            return found
    return None


def _find_jobposting(node: Any) -> Optional[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            hit = _find_jobposting(item)
            if hit is not None:
                return hit
    elif isinstance(node, dict):
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if "JobPosting" in types:
            return node
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                hit = _find_jobposting(node[key])
                if hit is not None:
                    return hit
    return None


def unescape_html_fragment(value: str) -> str:
    """Greenhouse ``content`` is HTML that has been entity-escaped; unescape then
    convert to text. Plain strings pass through unchanged."""
    if not value:
        return ""
    return html_to_text(html_lib.unescape(value))


# --------------------------------------------------------------------------- #
# Sentence splitting with character offsets (for keyword-in-context chunks)
# --------------------------------------------------------------------------- #
_ABBREVIATIONS = {
    "u.s.", "u.k.", "inc.", "corp.", "co.", "ltd.", "no.", "nos.", "vs.", "e.g.", "i.e.",
    "mr.", "ms.", "mrs.", "dr.", "st.", "jr.", "sr.", "n.a.", "l.l.c.", "cos.", "approx.",
    "etc.", "fig.", "sec.", "al.", "u.s.c.", "s.a.", "l.p.", "p.l.c.", "inc", "dept.",
}
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of sentences such that ``text[start:end]``
    is the exact sentence. Newlines are hard boundaries (paragraphs, headings,
    table rows)."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        line_start = pos
        pos += len(line) + 1
        if not line.strip():
            continue
        spans.extend(_split_line(line, line_start))
    return spans


def _split_line(line: str, offset: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(line):
        end = m.start() + len(m.group(0).rstrip())  # sentence end, before the whitespace
        nxt = line[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'\u201c\u2018(["):
            continue
        words = line[start:end].split()
        token = words[-1].lower() if words else ""
        if token in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", token):
            continue
        spans.append((offset + start, offset + end))
        start = m.end()
    tail_end = len(line.rstrip())
    if line[start:tail_end].strip():
        spans.append((offset + start, offset + tail_end))
    return spans


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    terms: tuple[str, ...]
    hit_count: int


def keyword_in_context(text: str, matcher: AiTermMatcher, window: int = 2,
                       max_chunks: int = 30, max_chars: int = 3000) -> list[Chunk]:
    """Chunks of ``window`` sentences either side of every sentence containing an
    AI term. Overlapping/adjacent windows are merged (up to ``max_chars``). When
    more than ``max_chunks`` exist, those with the most distinct AI terms win;
    the result is returned in document order. ``text[c.start:c.end]`` is always
    an exact substring of ``text``."""
    spans = split_sentences(text)
    if not spans:
        return []
    starts = [s for s, _ in spans]
    hit_sentences: dict[int, list[str]] = {}
    for h_start, _h_end, term in matcher.find(text):
        idx = bisect.bisect_right(starts, h_start) - 1
        if idx < 0 or h_start >= spans[idx][1]:
            continue
        hit_sentences.setdefault(idx, []).append(term)
    if not hit_sentences:
        return []
    windows: list[list[Any]] = []  # [lo, hi, terms, hits]
    for idx in sorted(hit_sentences):
        lo, hi = max(0, idx - window), min(len(spans) - 1, idx + window)
        terms = hit_sentences[idx]
        if windows and lo <= windows[-1][1] + 1 and spans[hi][1] - spans[windows[-1][0]][0] <= max_chars:
            windows[-1][1] = max(windows[-1][1], hi)
            windows[-1][2].extend(terms)
            windows[-1][3] += len(terms)
        else:
            windows.append([lo, hi, list(terms), len(terms)])
    chunks = [Chunk(spans[lo][0], spans[hi][1], tuple(dict.fromkeys(t)), n)
              for lo, hi, t, n in windows]
    if len(chunks) > max_chunks:
        ranked = sorted(range(len(chunks)),
                        key=lambda i: (-len(chunks[i].terms), -chunks[i].hit_count, chunks[i].start))
        keep = sorted(ranked[:max_chunks])
        chunks = [chunks[i] for i in keep]
    return chunks


def reflow_plain_text(text: str) -> str:
    """For legacy ``.txt`` filings: join hard-wrapped lines inside paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    return normalize_ws("\n".join(re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in paragraphs))


def first_nonempty(values: Iterable[Optional[str]]) -> str:
    for v in values:
        if v and v.strip():
            return v.strip()
    return ""


def main_content_text(html: str) -> str:
    """Text of the page's main content. Prefers ``<main>``/``<article>``/
    ``role=main``; otherwise strips navigation, header, footer and aside
    elements. Keeps site chrome (menus mentioning "AI solutions") out of job
    postings so it cannot be mistaken for posting text."""
    root = _parse_html(html)
    if root is None:
        return ""
    for xp in ("//main", "//article", "//*[@role='main']"):
        found = root.xpath(xp)
        if found:
            return html_to_text(etree.tostring(found[0], encoding="unicode", method="html"))
    for el in root.xpath("//nav|//header|//footer|//aside"):
        _drop(el)
    return html_to_text(etree.tostring(root, encoding="unicode", method="html"))
