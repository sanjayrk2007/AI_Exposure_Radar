"""REPOS collector (plan 5/M3 Day 3, SHOULD).

* Finds the vendor's public GitHub *organisation* and accepts it ONLY when the
  organisation's website field matches the vendor's profile website, or its
  display name equals the vendor name (or a configured alias). Anything else is
  not accepted (reported as ``NOT_APPLICABLE``; see README for why not the
  non-enumerated "SKIPPED").
* Reads dependency manifests (``requirements*.txt``, ``pyproject.toml``,
  ``package.json``) and keeps ONLY the lines that name an AI library.
* Uses the documented GitHub REST API (optional ``GITHUB_TOKEN`` for the higher
  rate limit), tracks ``X-RateLimit-Remaining`` and stops politely when low.
  No cloning, no authentication bypass, public repositories only.

A dependency is *capability* evidence only (it may sit in a sample or an
internal tool); every Document says so in ``meta``.
"""
from __future__ import annotations

import json
import re
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote, urlencode

from radar.contracts.models import DepthPlan, Document, SourceReport, VendorProfile
from radar.m3_sources.common import (
    BLOCKED, CAVEAT_REPOS, COLLECTOR_VERSION, EMPTY, ERROR, NOT_APPLICABLE, OK,
    NoNewTermsTracker, PageBudget, base_domain, host_of, make_doc_id, make_report,
    normalize_company_name, normalize_website, to_iso_utc,
)
from radar.m3_sources.fetching import FetchResult, Fetcher

SOURCE = "REPOS"
_EXAMPLE_PARTS = {"example", "examples", "sample", "samples", "demo", "demos", "docs", "doc",
                  "test", "tests", "fixtures", "tutorial", "tutorials"}


# --------------------------------------------------------------------------- #
# AI package matching
# --------------------------------------------------------------------------- #
def _norm_pkg(name: str) -> str:
    return re.sub(r"[_.]+", "-", name.strip().lower())


class AiPackageMatcher:
    def __init__(self, gcfg: dict[str, Any]) -> None:
        self.exact = {_norm_pkg(x) for x in gcfg.get("ai_packages_exact", [])}
        self.prefixes = tuple(_norm_pkg(x) for x in gcfg.get("ai_packages_prefixes", []))

    def is_ai(self, name: str) -> bool:
        n = _norm_pkg(name)
        return n in self.exact or (bool(self.prefixes) and n.startswith(self.prefixes))


_REQ_NAME = re.compile(r"^\s*(?:-e\s+)?(?:git\+\S*?#egg=)?([A-Za-z0-9@][A-Za-z0-9._\-/@]*)")


def _requirements_hits(text: str, m: AiPackageMatcher) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-r", "--", "-c", "-i", "-f")):
            continue
        mm = _REQ_NAME.match(line)
        if mm and m.is_ai(mm.group(1)):
            hits.append((line, _norm_pkg(mm.group(1))))
    return hits


_PEP508_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)")
_LINE_NAME = re.compile(r"""^\s*["']?(@?[A-Za-z0-9][A-Za-z0-9._\-/]*)["']?\s*(?:=|:|[\[<>=!~;,"'@\s]|$)""")


def _pyproject_dep_names(data: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    def from_specs(items: Any) -> None:
        if isinstance(items, list):
            for it in items:
                if isinstance(it, str):
                    mm = _PEP508_NAME.match(it)
                    if mm:
                        names.add(_norm_pkg(mm.group(1)))

    project = data.get("project", {}) if isinstance(data.get("project"), dict) else {}
    from_specs(project.get("dependencies"))
    for group in (project.get("optional-dependencies") or {}).values():
        from_specs(group)
    for group in (data.get("dependency-groups") or {}).values():
        from_specs(group)
    tool = data.get("tool", {}) if isinstance(data.get("tool"), dict) else {}
    poetry = tool.get("poetry", {}) if isinstance(tool.get("poetry"), dict) else {}
    tables = [poetry.get("dependencies"), poetry.get("dev-dependencies")]
    for g in (poetry.get("group") or {}).values():
        if isinstance(g, dict):
            tables.append(g.get("dependencies"))
    for t in tables:
        if isinstance(t, dict):
            names.update(_norm_pkg(k) for k in t)
    for path in (("build-system", "requires"),):
        from_specs(data.get(path[0], {}).get(path[1]) if isinstance(data.get(path[0]), dict) else None)
    return names


def _pyproject_hits(text: str, m: AiPackageMatcher) -> tuple[list[tuple[str, str]], str]:
    try:
        declared = _pyproject_dep_names(tomllib.loads(text))
        mode = "toml"
    except tomllib.TOMLDecodeError:
        declared, mode = None, "line-scan (invalid TOML)"  # type: ignore[assignment]
    hits: list[tuple[str, str]] = []
    for raw in text.splitlines():
        mm = _LINE_NAME.match(raw)
        if not mm:
            continue
        name = _norm_pkg(mm.group(1))
        if m.is_ai(name) and (declared is None or name in declared):
            hits.append((raw.strip(), name))
    return hits, mode


_PKG_SECTIONS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")


def _package_json_hits(text: str, m: AiPackageMatcher) -> list[tuple[str, str]]:
    data = json.loads(text)
    declared: set[str] = set()
    for sec in _PKG_SECTIONS:
        block = data.get(sec) if isinstance(data, dict) else None
        if isinstance(block, dict):
            declared.update(_norm_pkg(k) for k in block)
    hits: list[tuple[str, str]] = []
    for raw in text.splitlines():
        mm = re.match(r'^\s*"(@?[^"]+)"\s*:\s*"', raw)
        if mm:
            name = _norm_pkg(mm.group(1))
            if name in declared and m.is_ai(name):
                hits.append((raw.strip(), name))
    return hits


def extract_ai_lines(path: str, text: str, m: AiPackageMatcher) -> tuple[list[tuple[str, str]], str]:
    """Return ``([(verbatim_line, package)], parse_mode)`` for AI dependency lines."""
    base = path.rsplit("/", 1)[-1].lower()
    if base == "package.json":
        try:
            return _package_json_hits(text, m), "json"
        except json.JSONDecodeError:
            return [], "invalid JSON"
    if base == "pyproject.toml":
        return _pyproject_hits(text, m)
    return _requirements_hits(text, m), "requirements"


# --------------------------------------------------------------------------- #
# Organisation matching
# --------------------------------------------------------------------------- #
def _sld(host: str) -> str:
    labels = base_domain(host).split(".")
    if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "net", "gov", "ac"}:
        return labels[-3]
    return labels[-2] if len(labels) >= 2 else labels[0]


def candidate_slugs(vendor: VendorProfile, aliases: list[str], suffixes: list[str]) -> list[str]:
    out: list[str] = []

    def add(x: str) -> None:
        x = re.sub(r"[^a-z0-9\-]", "", x.lower()).strip("-")
        if 2 <= len(x) <= 39 and x not in out:
            out.append(x)

    site = normalize_website(vendor.website)
    if site:
        sld = _sld(host_of(site))
        add(sld)
        add(sld.replace("-", ""))
        if "-" in sld:
            add(sld.split("-")[0])
    for name in [vendor.name, *aliases]:
        norm = normalize_company_name(name, suffixes)
        tokens = norm.split()
        if 0 < len(tokens) <= 3:
            add("-".join(tokens))
            add("".join(tokens))
    return out[:6]


def org_matches(vendor: VendorProfile, org: dict[str, Any], aliases: list[str],
                suffixes: list[str]) -> tuple[bool, str]:
    """Plan: accept an organisation only if the profile website or name clearly matches."""
    vsite = normalize_website(vendor.website)
    vhost = base_domain(host_of(vsite)) if vsite else ""
    blog = str(org.get("blog") or "").strip()
    bsite = normalize_website(blog) if blog else None
    bhost = base_domain(host_of(bsite)) if bsite else ""
    if vhost and bhost and (bhost == vhost or bhost.endswith("." + vhost) or vhost.endswith("." + bhost)):
        return True, f"organisation website '{blog}' matches vendor website '{vendor.website}'"
    wanted = {normalize_company_name(n, suffixes) for n in [vendor.name, *aliases] if n}
    wanted.discard("")
    org_name = str(org.get("name") or "").strip()
    if org_name and normalize_company_name(org_name, suffixes) in wanted:
        return True, f"organisation name '{org_name}' equals vendor name"
    return False, (f"website '{blog or 'none'}' and name '{org_name or 'none'}' do not match "
                   f"vendor '{vendor.name}' / '{vendor.website}'")


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class _GitHubApi:
    def __init__(self, fetcher: Fetcher, vendor_id: str, gcfg: dict[str, Any], token: Optional[str]) -> None:
        self.fetcher, self.vendor_id, self.token = fetcher, vendor_id, token
        self.base = gcfg.get("api_base", "https://api.github.com").rstrip("/")
        self.max_calls = int(gcfg["max_api_calls_authenticated" if token else "max_api_calls_unauthenticated"])
        self.min_remaining = int(gcfg.get("min_remaining_before_stop", 2))
        self.calls = 0
        self.remaining: Optional[int] = None
        self.stop_reason = ""          # why the API is no longer usable (partial result)
        self.blocked_reason = ""       # access denied / rate limit hit

    def usable(self) -> bool:
        if self.stop_reason:
            return False
        if self.calls >= self.max_calls:
            self.stop_reason = f"API call ceiling reached ({self.max_calls}); set GITHUB_TOKEN for a higher limit"
            return False
        if self.remaining is not None and self.remaining <= self.min_remaining:
            self.stop_reason = "GitHub API rate limit nearly exhausted"
            return False
        return True

    def get(self, path: str, params: Optional[dict[str, Any]] = None, raw: bool = False
            ) -> tuple[Optional[FetchResult], Any]:
        if not self.usable():
            return None, None
        url = self.base + path + (("?" + urlencode(params)) if params else "")
        headers = {"Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        res = self.fetcher.fetch(url, vendor_id=self.vendor_id, kind="api", headers=headers)
        self.calls += 1
        rem = res.headers.get("x-ratelimit-remaining")
        if rem is not None and str(rem).isdigit():
            self.remaining = int(rem)
        if res.outcome() == "blocked":
            self.blocked_reason = res.blocked_reason or f"HTTP {res.status}"
            if self.remaining == 0:
                reset = res.headers.get("x-ratelimit-reset", "")
                when = f", resets at {to_iso_utc(reset)}" if reset.isdigit() else ""
                self.stop_reason = f"GitHub API rate limit exhausted{when}; set GITHUB_TOKEN for a higher limit"
            elif res.status in (403, 429):
                self.stop_reason = f"GitHub API refused the request ({self.blocked_reason}); likely rate limit"
            return res, None
        if res.outcome() != "ok":
            return res, None
        if raw:
            return res, res.text
        try:
            return res, json.loads(res.text)
        except json.JSONDecodeError:
            return res, None


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
def collect_repos(vendor: VendorProfile, plan: DepthPlan, fetcher: Fetcher, cfg: dict[str, Any],
                  *, token: Optional[str] = None) -> tuple[list[Document], SourceReport]:
    vid = vendor.vendor_id
    gcfg = cfg["github"]
    suffixes = cfg["edgar"].get("legal_suffixes", [])
    aliases = list(gcfg.get("name_aliases", {}).get(vid, []))
    api = _GitHubApi(fetcher, vid, gcfg, token)
    matcher = AiPackageMatcher(gcfg)

    # 1. organisation -------------------------------------------------------------
    org: Optional[dict[str, Any]] = None
    reason = ""
    rejected: list[str] = []
    probe_errors: list[str] = []
    probed: set[str] = set()
    override = gcfg.get("org_overrides", {}).get(vid)

    def probe(slug: str, analyst_confirmed: bool = False) -> Optional[tuple[dict[str, Any], str]]:
        if slug.lower() in probed:
            return None
        probed.add(slug.lower())
        res, data = api.get(f"/orgs/{quote(slug)}")
        if res is None:
            return None
        if res.outcome() == "error":
            probe_errors.append(f"{slug}: {res.reason}")
            return None
        if not isinstance(data, dict) or data.get("type", "Organization") != "Organization":
            return None
        if analyst_confirmed:
            return data, "organisation slug confirmed by analyst (config org_overrides)"
        ok, why = org_matches(vendor, data, aliases, suffixes)
        if ok:
            return data, why
        rejected.append(f"'{slug}' rejected: {why}")
        return None

    found = probe(str(override), True) if override else None
    if not found:
        for slug in candidate_slugs(vendor, aliases, suffixes):
            found = probe(slug)
            if found or api.stop_reason:
                break
    if not found and not api.stop_reason and gcfg.get("use_search_fallback", True):
        query = normalize_company_name(vendor.name, suffixes) or vendor.name
        res, data = api.get("/search/users", {"q": f"{query} type:org", "per_page": 5})
        if isinstance(data, dict):
            for item in (data.get("items") or [])[:3]:
                login = str(item.get("login", ""))
                if login:
                    found = probe(login)
                    if found or api.stop_reason:
                        break
    if found:
        org, reason = found

    if org is None:
        return [], _no_org_report(vid, api, rejected, probe_errors)

    login = str(org.get("login"))
    # 2. repositories ----------------------------------------------------------------
    res, repos = api.get(f"/orgs/{quote(login)}/repos",
                         {"type": "public", "sort": "pushed", "direction": "desc", "per_page": 100})
    if not isinstance(repos, list):
        status = BLOCKED if (res is not None and res.outcome() == "blocked") or api.blocked_reason else ERROR
        why = res.reason if res is not None else (api.stop_reason or "no response")
        return [], make_report(vid, SOURCE, status, 0,
                               f"organisation '{login}' accepted ({reason}) but its repository list could not be read: {why}")
    usable = [r for r in repos if isinstance(r, dict) and not r.get("private")
              and not (gcfg.get("exclude_forks", True) and r.get("fork"))
              and not (gcfg.get("exclude_archived", True) and r.get("archived"))
              and not r.get("disabled")]
    max_repos = min(int(gcfg.get("max_repos", 25)), max(1, plan.max_pages))
    usable = usable[:max_repos]

    budget = PageBudget(plan.max_pages)
    tracker = NoNewTermsTracker(int(gcfg.get("stop_patience", 6)))
    docs: list[Document] = []
    repos_scanned = manifests_fetched = 0
    stopped = ""
    ignored = {x.lower() for x in gcfg.get("ignored_path_parts", [])}
    names = {x.lower() for x in gcfg.get("manifest_basenames", [])}
    patterns = [re.compile(p, re.I) for p in gcfg.get("manifest_basename_patterns", [])]
    max_depth_seg = int(gcfg.get("max_tree_depth", 3))
    truncated_trees = 0

    for repo in usable:
        if stopped or not api.usable():
            break
        name, branch = str(repo.get("name")), str(repo.get("default_branch") or "main")
        res, tree = api.get(f"/repos/{quote(login)}/{quote(name)}/git/trees/{quote(branch)}", {"recursive": "1"})
        if not isinstance(tree, dict):
            continue
        repos_scanned += 1
        if tree.get("truncated"):
            truncated_trees += 1
        manifests = []
        for node in tree.get("tree", []):
            path = str(node.get("path", ""))
            parts = path.split("/")
            if node.get("type") != "blob" or len(parts) > max_depth_seg:
                continue
            if any(p.lower() in ignored for p in parts[:-1]):
                continue
            base = parts[-1]
            if base.lower() in names or any(p.match(base) for p in patterns):
                if int(node.get("size") or 0) <= int(gcfg.get("max_manifest_bytes", 400000)):
                    manifests.append((len(parts), path))
        manifests.sort()
        for _, path in manifests[: int(gcfg.get("max_manifests_per_repo", 5))]:
            if not budget.can_spend(1):
                stopped = f"page budget of {plan.max_pages} reached"
                break
            res, body = api.get(f"/repos/{quote(login)}/{quote(name)}/contents/{quote(path)}",
                                {"ref": branch}, raw=True)
            if res is None or not isinstance(body, str):
                if api.stop_reason:
                    break
                continue
            budget.spend(1)
            manifests_fetched += 1
            hits, mode = extract_ai_lines(path, body, matcher)
            seen_lines: dict[str, str] = {}
            for line, pkg in hits:
                seen_lines.setdefault(line, pkg)
            if seen_lines:
                lines = list(seen_lines)
                pkgs = sorted(set(seen_lines.values()))
                url = f"https://github.com/{login}/{name}/blob/{branch}/{path}"
                docs.append(Document(
                    doc_id=make_doc_id(vid, SOURCE, url),
                    vendor_id=vid, source_type=SOURCE, url=url, fetched_at=res.fetched_at,
                    http_status=res.status, content_hash=res.content_hash,
                    title=f"{login}/{name}: {path}", text="\n".join(lines), snapshot_path=res.snapshot_path,
                    meta={
                        "org": login, "repo": name, "path": path, "branch": branch,
                        "repo_url": f"https://github.com/{login}/{name}", "pushed_at": repo.get("pushed_at"),
                        "manifest_type": path.rsplit("/", 1)[-1], "parse_mode": mode,
                        "matched_packages": pkgs, "org_match_reason": reason,
                        "is_example_path": any(p.lower() in _EXAMPLE_PARTS for p in path.split("/")[:-1]),
                        "evidence_class": "capability_dependency", "evidence_caveat": CAVEAT_REPOS,
                        "api_url": res.url, "synthetic": False, "collector": COLLECTOR_VERSION,
                    }))
            if tracker.observe(seen_lines.values()):
                stopped = f"stopped after {tracker.patience} consecutive manifests with no new AI libraries"
                break

    docs.sort(key=lambda d: d.doc_id)
    parts = [f"organisation '{login}' accepted ({reason})", f"{repos_scanned} repos scanned",
             f"{manifests_fetched} manifests read", f"{api.calls} API calls"]
    if truncated_trees:
        parts.append(f"{truncated_trees} repo tree(s) truncated by GitHub")
    if stopped:
        parts.append(stopped)
    if api.stop_reason:
        parts.append(f"partial result: {api.stop_reason}")
    if docs:
        parts.append(f"{len(docs)} manifests contain AI libraries")
        return docs, make_report(vid, SOURCE, OK, manifests_fetched, "; ".join(parts))
    if manifests_fetched == 0 and api.blocked_reason and api.stop_reason:
        return docs, make_report(vid, SOURCE, BLOCKED, 0, "; ".join(parts))
    parts.append("no AI libraries found in public dependency manifests")
    return docs, make_report(vid, SOURCE, EMPTY, manifests_fetched, "; ".join(parts))


def _no_org_report(vid: str, api: _GitHubApi, rejected: list[str], errors: list[str]) -> SourceReport:
    if api.blocked_reason and api.stop_reason:
        return make_report(vid, SOURCE, BLOCKED, 0, f"organisation lookup blocked: {api.stop_reason}")
    if errors and not rejected:
        return make_report(vid, SOURCE, ERROR, 0, "organisation lookup failed: " + "; ".join(errors[:3]))
    detail = ("; ".join(rejected[:3]) if rejected else "no organisation found for the vendor's name or domain")
    return make_report(
        vid, SOURCE, NOT_APPLICABLE, 0,
        "no public GitHub organisation clearly matches this vendor (accepted only on website/name match), "
        f"so repositories were skipped: {detail}")
