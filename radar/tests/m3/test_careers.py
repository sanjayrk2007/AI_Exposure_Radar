import html as h
import json

from radar.m3_sources.careers import (
    AtsRef, collect_careers, detect_ats, feed_url, is_technical_role, parse_ashby, parse_greenhouse, parse_lever,
)
from radar.tests.m3.fakes import FakeFetcher, plan, resp, vendor

HOME = f"https://www.acme-vendor.example/"
CAREERS = "https://www.acme-vendor.example/careers"
GH_FEED = "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true"


def home(links="<a href='/careers'>Careers</a>"):
    return resp(f"<html><body><nav>{links}</nav><p>Welcome</p></body></html>")


def gh_job(i, title, body, dept="Engineering"):
    return {"id": i, "title": title, "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{i}",
            "location": {"name": "Remote"}, "departments": [{"name": dept}], "first_published": "2026-09-01T10:00:00-04:00",
            "content": h.escape(f"<p>{body}</p><ul><li>Python</li></ul>")}


def gh_routes(jobs):
    return {HOME: home(), CAREERS: resp("<a href='https://boards.greenhouse.io/acme'>Open roles</a>"),
            GH_FEED: resp({"jobs": jobs})}


def run(routes, cfg, matcher, **plan_kw):
    f = FakeFetcher(routes)
    docs, rep = collect_careers(vendor(), plan(**plan_kw), f, cfg, matcher)
    return docs, rep, f


def test_greenhouse_feed_keeps_technical_roles_ai_first(cfg, matcher):
    jobs = [gh_job(1, "Staff Data Engineer", "Build pipelines with SQL."),
            gh_job(2, "Senior Machine Learning Engineer", "You will build machine learning models using PyTorch for fraud scoring."),
            gh_job(3, "Technical Recruiter", "Source engineering candidates.", "People"),
            gh_job(4, "Account Executive", "Sell products.", "Sales")]
    docs, rep, f = run(gh_routes(jobs), cfg, matcher)
    assert rep.status == "OK" and rep.source_type == "CAREERS" and rep.vendor_id == "V-002"
    assert [d.doc_id for d in docs] == sorted(d.doc_id for d in docs)   # deterministic order
    titles = {d.title for d in docs}
    assert titles == {"Staff Data Engineer", "Senior Machine Learning Engineer"}     # recruiter / sales dropped
    ml = next(d for d in docs if d.title.startswith("Senior"))
    assert "<" not in ml.text and "Python" in ml.text and ml.text.startswith("Senior Machine Learning Engineer\n")
    assert ml.source_type == "CAREERS" and ml.url == "https://boards.greenhouse.io/acme/jobs/2"
    assert ml.http_status == 200 and ml.fetched_at == "2026-10-10T08:00:00Z" and ml.content_hash
    assert ml.meta["ai_terms"] and ml.meta["ats"] == "greenhouse" and ml.meta["synthetic"] is False
    assert "not proof" in ml.meta["evidence_caveat"] and ml.meta["evidence_class"] == "capability_and_intent"
    assert ml.meta["posted_at"] == "2026-09-01T14:00:00Z"
    assert rep.pages_fetched == 3 and "greenhouse" in rep.note and "1 mention AI" in rep.note
    assert f.calls[-1]["kind"] == "api"                                              # feed uses the API route


def test_stop_rule_three_postings_without_new_terms(cfg, matcher):
    jobs = [gh_job(i, f"Machine Learning Engineer {i}", "We apply machine learning to payments.") for i in range(1, 9)]
    docs, rep, _ = run(gh_routes(jobs), cfg, matcher)
    assert len(docs) == 4 and "stopped after 3 consecutive postings" in rep.note


def test_page_budget_caps_documents(cfg, matcher):
    terms = ["machine learning", "PyTorch", "NLP", "LLM", "embeddings", "fine-tuning", "computer vision"]
    jobs = [gh_job(i, f"Engineer {i}", f"We use {t} daily.") for i, t in enumerate(terms, 1)]
    docs, rep, _ = run(gh_routes(jobs), cfg, matcher, max_pages=5)
    assert len(docs) == 5 and "capped at 5" in rep.note


def test_lever_and_ashby_parsers():
    lever = json.dumps([{"id": "u1", "text": "ML Engineer", "hostedUrl": "https://jobs.lever.co/acme/u1", "createdAt": 1790000000000,
                         "categories": {"department": "Engineering", "location": "NYC"}, "descriptionPlain": "Build LLM apps.",
                         "lists": [{"text": "Requirements", "content": "<li>Python</li>"}], "additionalPlain": "Equal opportunity."}])
    p = parse_lever(lever)[0]
    assert p.title == "ML Engineer" and "Python" in p.text and "Requirements" in p.text and p.posted_at.startswith("2026-")
    ashby = json.dumps({"jobs": [{"id": "a1", "title": "Data Scientist", "department": "Data", "location": "Remote",
                                  "descriptionHtml": "<p>Forecasting models</p>", "jobUrl": "https://jobs.ashbyhq.com/acme/a1",
                                  "publishedAt": "2026-08-01T00:00:00Z"}, {"id": "a2", "title": "Hidden", "isListed": False}]})
    ps = parse_ashby(ashby)
    assert len(ps) == 1 and ps[0].text == "Forecasting models" and ps[0].department == "Data"


def test_lever_feed_end_to_end_and_eu_host(cfg, matcher):
    routes = {HOME: home(), CAREERS: resp("<a href='https://jobs.lever.co/acme'>Jobs</a>"),
              "https://api.lever.co/v0/postings/acme?mode=json": resp([{"id": "u1", "text": "AI Engineer",
                  "hostedUrl": "https://jobs.lever.co/acme/u1", "categories": {}, "descriptionPlain": "Deploy LLM services."}])}
    docs, rep, _ = run(routes, cfg, matcher)
    assert rep.status == "OK" and docs[0].meta["ats"] == "lever"
    assert feed_url(AtsRef("lever", "x", True)).startswith("https://api.eu.lever.co/")


def test_detect_ats_patterns():
    refs = detect_ats(["https://boards.greenhouse.io/embed/job_board?for=acme", "https://x.example/a",
                       "https://jobs.lever.co/beta/123", "https://acme.wd5.myworkdayjobs.com/en-US/External",
                       "https://boards.greenhouse.io/embed/js"])
    assert [(r.ats, r.token) for r in refs] == [("greenhouse", "acme"), ("lever", "beta"), ("workday", "acme.wd5.myworkdayjobs.com")]
    assert feed_url(refs[2]) is None


def test_html_fallback_with_jsonld_and_non_tech_links_not_fetched(cfg, matcher):
    ld = {"@context": "https://schema.org", "@type": "JobPosting", "title": "Data Scientist",
          "description": h.escape("<p>" + "Train predictive model prototypes with scikit-learn. " * 5 + "</p>"),
          "datePosted": "2026-09-10", "identifier": {"@type": "PropertyValue", "value": "R-77"},
          "jobLocation": {"address": {"addressLocality": "Austin", "addressCountry": "US"}}}
    page = f"<html><script type='application/ld+json'>{json.dumps(ld)}</script><body>x</body></html>"
    routes = {HOME: home(), CAREERS: resp("<a href='/careers/jobs/77'>Data Scientist</a><a href='/careers/jobs/78'>Recruiter</a>"),
              "https://www.acme-vendor.example/careers/jobs/77": resp(page)}
    docs, rep, f = run(routes, cfg, matcher)
    assert rep.status == "OK" and len(docs) == 1
    d = docs[0]
    assert d.url.endswith("/careers/jobs/77") and d.meta["job_id"] == "R-77" and d.meta["location"] == "Austin, US"
    assert "predictive model" in d.meta["ai_terms"]
    assert not any(u.endswith("/jobs/78") for u in f.urls())                       # clearly non-technical link skipped
    assert "individual posting pages" in rep.note


def test_html_main_content_excludes_site_chrome(cfg, matcher):
    body = "<nav>Our AI solutions</nav><main><h1>Platform Engineer</h1><p>" + "Operate Kubernetes clusters. " * 8 + "</p></main>"
    routes = {HOME: home(), CAREERS: resp("<a href='/jobs/9'>Platform Engineer</a>"),
              "https://www.acme-vendor.example/jobs/9": resp(f"<html><body>{body}</body></html>")}
    docs, rep, _ = run(routes, cfg, matcher)
    assert len(docs) == 1 and docs[0].meta["ai_terms"] == [] and "AI solutions" not in docs[0].text


def test_no_careers_page_is_empty_not_error(cfg, matcher):
    docs, rep, _ = run({HOME: home("<a href='/about'>About</a>")}, cfg, matcher)
    assert docs == [] and rep.status == "EMPTY" and "no careers page found" in rep.note


def test_blocked_by_robots(cfg, matcher):
    r = resp("", 0, blocked="robots", error="robots.txt: disallowed")
    docs, rep, _ = run({HOME: r, CAREERS: r}, cfg, matcher)
    assert docs == [] and rep.status == "BLOCKED" and "robots" in rep.note


def test_all_requests_failing_is_error(cfg, matcher):
    f = FakeFetcher({HOME: resp("", 0, error="timeout")})
    f.routes.update({"https://www.acme-vendor.example" + p: resp("", 0, error="timeout") for p in cfg["careers"]["common_paths"]})
    docs, rep = collect_careers(vendor(), plan(), f, cfg, matcher)
    assert rep.status == "ERROR" and "timeout" in rep.note


def test_unsupported_ats_reported_not_scraped(cfg, matcher):
    routes = {HOME: home(), CAREERS: resp("<a href='https://acme.wd5.myworkdayjobs.com/en-US/External'>Apply</a>")}
    docs, rep, f = run(routes, cfg, matcher)
    assert rep.status == "EMPTY" and "workday" in rep.note and "no public feed implemented" in rep.note
    assert not any("myworkdayjobs" in u for u in f.urls())


def test_malformed_feed_is_reported_and_falls_back(cfg, matcher):
    routes = gh_routes([]); routes[GH_FEED] = resp("<html>oops</html>")
    docs, rep, _ = run(routes, cfg, matcher)
    assert docs == [] and rep.status == "EMPTY" and "not understood" in rep.note


def test_linkedin_is_never_requested(cfg, matcher):
    routes = {HOME: home("<a href='https://www.linkedin.com/company/acme/jobs'>Careers on LinkedIn</a><a href='/careers'>Careers</a>"),
              CAREERS: resp("<a href='https://www.linkedin.com/jobs/view/1'>Data Engineer</a>")}
    _, rep, f = run(routes, cfg, matcher)
    assert not any("linkedin" in u for u in f.urls())


def test_max_depth_limits_feed_access(cfg, matcher):
    docs, rep, f = run(gh_routes([gh_job(1, "ML Engineer", "machine learning")]), cfg, matcher, max_depth=1)
    assert docs == [] and GH_FEED not in f.urls() and "max_depth" in rep.note


def test_bad_website_is_error(cfg, matcher):
    docs, rep = collect_careers(vendor(website=""), plan(), FakeFetcher(), cfg, matcher)
    assert rep.status == "ERROR" and docs == []


def test_technical_filter(cfg, matcher):
    t = lambda title, dept="": is_technical_role(title, dept, cfg["careers"], matcher)
    assert t("Senior Software Engineer") and t("Data Analyst") and t("Analyst", "Technology")
    assert not t("Talent Acquisition Partner") and not t("Marketing Manager")
    assert t("AI Recruiting Tools Engineer") and t("Recruiter, AI Platform")   # AI term in title keeps it
    assert not t("Receptionist")


def test_deterministic_output(cfg, matcher):
    jobs = [gh_job(i, f"Engineer {i}", "machine learning") for i in range(1, 4)]
    a, _, _ = run(gh_routes(jobs), cfg, matcher)
    b, _, _ = run(gh_routes(jobs), cfg, matcher)
    assert [(d.doc_id, d.text) for d in a] == [(d.doc_id, d.text) for d in b]
