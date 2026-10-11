import json

from radar.m3_sources.github import AiPackageMatcher, candidate_slugs, collect_repos, extract_ai_lines, org_matches
from radar.tests.m3.fakes import FakeFetcher, plan, resp, vendor

API = "https://api.github.com"
ORG = {"login": "acme", "type": "Organization", "name": "Acme Test Vendor", "blog": "https://acme-vendor.example"}
REPOS_URL = f"{API}/orgs/acme/repos?type=public&sort=pushed&direction=desc&per_page=100"
REQ = "# core\nflask==3.0.0\nopenai==1.30.0\nscikit-learn>=1.4  # models\n-r other.txt\nlangchain-core==0.2.1\n"
PYPROJECT = '''[project]
name = "svc"
keywords = ["openai", "ml"]
dependencies = [
  "requests>=2",
  "torch>=2.2",
]
[tool.poetry.dependencies]
anthropic = "^0.30"
'''
PKG = '{\n  "name": "web",\n  "scripts": {"openai": "x"},\n  "dependencies": {\n    "react": "^18.0.0",\n    "@anthropic-ai/sdk": "^0.20.0"\n  }\n}\n'


def tree(*paths):
    return {"truncated": False, "tree": [{"path": p, "type": "blob", "size": 100} for p in paths]}


def routes(extra=None, repos=None, org=ORG):
    repos = repos if repos is not None else [
        {"name": "svc", "default_branch": "main", "pushed_at": "2026-09-30T00:00:00Z"},
        {"name": "forked", "default_branch": "main", "fork": True},
        {"name": "old", "default_branch": "main", "archived": True}]
    r = {f"{API}/orgs/acme-vendor": resp("", 404), f"{API}/orgs/acmevendor": resp("", 404), f"{API}/orgs/acme": resp(org),
         REPOS_URL: resp(repos),
         f"{API}/repos/acme/svc/git/trees/main?recursive=1": resp(tree(
             "requirements.txt", "pyproject.toml", "web/package.json", "node_modules/x/package.json", "a/b/c/d/requirements.txt")),
         f"{API}/repos/acme/svc/contents/requirements.txt?ref=main": resp(REQ, ctype="text/plain"),
         f"{API}/repos/acme/svc/contents/pyproject.toml?ref=main": resp(PYPROJECT, ctype="text/plain"),
         f"{API}/repos/acme/svc/contents/web/package.json?ref=main": resp(PKG, ctype="text/plain")}
    r.update(extra or {})
    return r


def run(r, **kw):
    f = FakeFetcher(r)
    pl = kw.pop("plan_", plan())
    docs, rep = collect_repos(vendor(), pl, f, kw.pop("cfg"), **kw)
    return docs, rep, f


def test_happy_path_keeps_only_ai_lines_verbatim(cfg):
    docs, rep, f = run(routes(), cfg=cfg)
    assert rep.status == "OK" and rep.source_type == "REPOS" and rep.pages_fetched == 3
    by = {d.meta["path"]: d for d in docs}
    assert set(by) == {"requirements.txt", "pyproject.toml", "web/package.json"}
    assert by["requirements.txt"].text == "openai==1.30.0\nscikit-learn>=1.4  # models\nlangchain-core==0.2.1"
    assert by["pyproject.toml"].meta["matched_packages"] == ["anthropic", "torch"]    # 'keywords' line is NOT a dependency
    assert '"openai"' not in by["pyproject.toml"].text and '"torch>=2.2",' in by["pyproject.toml"].text
    assert by["web/package.json"].text == '"@anthropic-ai/sdk": "^0.20.0"'            # scripts key not matched
    d = by["requirements.txt"]
    assert d.url == "https://github.com/acme/svc/blob/main/requirements.txt" and d.source_type == "REPOS"
    assert d.http_status == 200 and d.content_hash and d.meta["org"] == "acme" and d.meta["synthetic"] is False
    assert "not proof" in d.meta["evidence_caveat"] and "website" in d.meta["org_match_reason"]
    urls = f.urls()
    assert not any("forked" in u or "/old/" in u for u in urls)                         # forks/archived skipped
    assert not any("node_modules" in u or "/d/requirements" in u for u in urls)         # ignored / too deep
    contents = [c for c in f.calls if "/contents/" in c["url"]]
    assert all(c["headers"]["Accept"] == "application/vnd.github.raw+json" for c in contents)
    assert all(c["kind"] == "api" and "Authorization" not in c["headers"] for c in f.calls)


def test_token_sent_as_bearer_only_when_given(cfg):
    _, _, f = run(routes(), cfg=cfg, token="ghp_test")
    assert all(c["headers"]["Authorization"] == "Bearer ghp_test" for c in f.calls)


def test_org_accepted_by_name_when_website_missing(cfg):
    org = dict(ORG, blog="", name="Acme Test Vendor Inc")
    docs, rep, _ = run(routes(org=org), cfg=cfg)
    assert rep.status == "OK" and "name" in docs[0].meta["org_match_reason"]


def test_mismatched_org_is_not_accepted(cfg):
    org = dict(ORG, blog="https://totally-different.example", name="Something Else")
    r = routes(org=org); r[f"{API}/search/users?q=acme+test+vendor+type%3Aorg&per_page=5"] = resp({"items": []})
    docs, rep, f = run(r, cfg=cfg)
    assert docs == [] and rep.status == "NOT_APPLICABLE" and "rejected" in rep.note
    assert REPOS_URL not in f.urls()                                                    # nothing read from a wrong org


def test_no_org_at_all_is_not_applicable(cfg):
    docs, rep, _ = run({}, cfg=cfg)
    assert rep.status == "NOT_APPLICABLE" and "no public GitHub organisation" in rep.note and docs == []


def test_search_fallback_accepts_only_matching_org(cfg):
    other = {"login": "weird-name", "type": "Organization", "name": "x", "blog": "acme-vendor.example"}
    r = {f"{API}/search/users?q=acme+test+vendor+type%3Aorg&per_page=5": resp({"items": [{"login": "weird-name"}]}),
         f"{API}/orgs/weird-name": resp(other), f"{API}/orgs/weird-name/repos?type=public&sort=pushed&direction=desc&per_page=100": resp([])}
    docs, rep, _ = run(r, cfg=cfg)
    assert rep.status == "EMPTY" and "weird-name" in rep.note


def test_no_ai_libraries_is_empty(cfg):
    r = routes(); r[f"{API}/repos/acme/svc/contents/requirements.txt?ref=main"] = resp("flask\nrequests\n")
    r[f"{API}/repos/acme/svc/contents/pyproject.toml?ref=main"] = resp("[project]\nname='x'\n")
    r[f"{API}/repos/acme/svc/contents/web/package.json?ref=main"] = resp('{"dependencies": {"react": "1"}}')
    docs, rep, _ = run(r, cfg=cfg)
    assert docs == [] and rep.status == "EMPTY" and rep.pages_fetched == 3 and "no AI libraries" in rep.note


def test_rate_limit_hit_is_blocked(cfg):
    r = {f"{API}/orgs/acme-vendor": resp("rate limit", 403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1791623493"})}
    docs, rep, f = run(r, cfg=cfg)
    assert rep.status == "BLOCKED" and docs == [] and len(f.calls) == 1               # stops immediately
    assert "rate limit exhausted, resets at 2026-" in rep.note and "GITHUB_TOKEN" in rep.note


def test_low_remaining_stops_politely_with_partial_note(cfg):
    r = routes(); r[REPOS_URL] = resp([{"name": "svc", "default_branch": "main"}], headers={"x-ratelimit-remaining": "2"})
    docs, rep, f = run(r, cfg=cfg)
    assert "partial result" in rep.note and not any("/git/trees/" in u for u in f.urls())


def test_unauthenticated_call_ceiling(cfg):
    small = json.loads(json.dumps(cfg)); small["github"]["max_api_calls_unauthenticated"] = 4
    docs, rep, f = run(routes(), cfg=small)
    assert len(f.calls) == 4 and "ceiling" in rep.note


def test_page_budget_limits_manifests(cfg):
    docs, rep, _ = run(routes(), cfg=cfg, plan_=plan(max_pages=1))
    assert len(docs) == 1 and rep.pages_fetched == 1 and "page budget" in rep.note


def test_api_failure_is_error(cfg):
    r = {f"{API}/orgs/{s}": resp("", 502) for s in ("acme-vendor", "acmevendor", "acme", "acme-test-vendor", "acmetestvendor")}
    r[f"{API}/search/users?q=acme+test+vendor+type%3Aorg&per_page=5"] = resp("", 502)
    docs, rep, _ = run(r, cfg=cfg)
    assert rep.status == "ERROR"


def test_analyst_override_bypasses_name_match(cfg):
    c = json.loads(json.dumps(cfg)); c["github"]["org_overrides"] = {"V-002": "acme"}
    org = dict(ORG, blog="https://elsewhere.example", name="Unrelated Display Name")
    docs, rep, _ = run(routes(org=org), cfg=c)
    assert rep.status == "OK" and "analyst" in docs[0].meta["org_match_reason"]


def test_extract_lines_and_matcher(cfg):
    m = AiPackageMatcher(cfg["github"])
    assert m.is_ai("OpenAI") and m.is_ai("langchain_community") and m.is_ai("@langchain/core") and not m.is_ai("requests")
    hits, mode = extract_ai_lines("a/requirements-dev.txt", "pytest\ntorch==2.0\n-e git+https://x/y.git#egg=transformers\n", m)
    assert [p for _, p in hits] == ["torch", "transformers"] and mode == "requirements"
    hits, mode = extract_ai_lines("pyproject.toml", 'dependencies = ["openai"', m)   # invalid TOML -> line scan fallback
    assert "invalid" in mode
    assert extract_ai_lines("package.json", "{not json", m) == ([], "invalid JSON")


def test_org_matching_rules(cfg):
    sfx = cfg["edgar"]["legal_suffixes"]
    v = vendor(name="The Clearing House Payments Company L.L.C.", website="www.theclearinghouse.org")
    assert org_matches(v, {"blog": "https://www.theclearinghouse.org/about"}, [], sfx)[0]
    assert org_matches(v, {"name": "Clearing House Payments"}, [], sfx)[0]
    assert not org_matches(v, {"name": "Clearing House", "blog": "https://clearinghouse.example"}, [], sfx)[0]
    bny = vendor(name="BNY", website="www.bny.com")
    assert org_matches(bny, {"name": "BNY Mellon"}, cfg["github"]["name_aliases"]["V-005"], sfx)[0]
    assert candidate_slugs(vendor(name="Financial Statement Services, Inc. (FSSI)", website="www.fssi-ca.com"), [], sfx)[:3] == \
        ["fssi-ca", "fssica", "fssi"]
