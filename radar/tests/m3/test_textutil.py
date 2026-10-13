from radar.m3_sources.common import AiTermMatcher
from radar.m3_sources.textutil import (
    extract_embedded_urls, extract_jobposting_jsonld, extract_links, html_to_text, keyword_in_context, split_sentences,
    unescape_html_fragment,
)

M = AiTermMatcher(["machine learning", "AI"])


def test_sentence_offsets_are_exact_and_abbreviations_survive():
    t = "Acme Corp. is in the U.S. today. See No. 5 for e.g. details. J. Smith agrees! Done."
    got = [t[s:e] for s, e in split_sentences(t)]
    assert got == ["Acme Corp. is in the U.S. today.", "See No. 5 for e.g. details.", "J. Smith agrees!", "Done."]


def test_newlines_are_hard_boundaries():
    t = "Heading without period\nNext paragraph here."
    assert [t[s:e] for s, e in split_sentences(t)] == ["Heading without period", "Next paragraph here."]


def test_kwic_window_merge_and_cap():
    sents = [f"Sentence {i} is plain." for i in range(30)]
    sents[5] = "We use machine learning here."
    sents[7] = "AI is mentioned again."          # overlaps first window -> merged
    sents[25] = "AI appears late."
    t = " ".join(sents)
    ch = keyword_in_context(t, M, window=2)
    assert len(ch) == 2
    assert t[ch[0].start:ch[0].end].startswith("Sentence 3") and t[ch[0].start:ch[0].end].endswith("Sentence 9 is plain.")
    assert ch[0].terms == ("machine learning", "AI")
    assert len(keyword_in_context(t, M, window=2, max_chunks=1)) == 1
    assert keyword_in_context("Nothing relevant here.", M) == []


def test_html_to_text_drops_hidden_and_keeps_inline_sentences_whole():
    t = html_to_text("<div style='display:none'>secret</div><p>We use <b>AI</b> daily.</p><script>x=1</script><p>Two</p>")
    assert "secret" not in t and "x=1" not in t and "We use AI daily." in t and t.endswith("Two")


def test_links_dedup_fragments_and_skips_mailto():
    html = "<a href='/a#x'>A</a><a href='/a'>again</a><a href='mailto:x@y'>m</a><a href='https://o.example/b'>B</a>"
    assert [l.url for l in extract_links(html, "https://s.example/p")] == ["https://s.example/a", "https://o.example/b"]
    assert "https://s.example/js.js" in extract_embedded_urls("<script src='/js.js'></script>", "https://s.example/")


def test_jsonld_graph_and_bad_json():
    ok = '<script type="application/ld+json">{"@graph":[{"@type":"Org"},{"@type":["JobPosting"],"title":"T"}]}</script>'
    assert extract_jobposting_jsonld(ok)["title"] == "T"
    assert extract_jobposting_jsonld('<script type="application/ld+json">{bad</script>') is None
    assert extract_jobposting_jsonld("") is None


def test_greenhouse_double_escaped_content():
    assert unescape_html_fragment("&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;") == "Hello & welcome"
    assert unescape_html_fragment("") == ""


def test_html_to_text_with_null_bytes_and_control_chars():
    bad_html = "<div><!-- comment\x00 -->Hello\x00 world\x0c!\x08<script>drop\x00me</script><p>Tail\x0b text</p></div>"
    text = html_to_text(bad_html)
    assert "Hello world!" in text
    assert "Tail text" in text
    assert "\x00" not in text
    assert "\x0c" not in text

