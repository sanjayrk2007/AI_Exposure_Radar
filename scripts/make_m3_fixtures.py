"""Regenerate the M3 fixtures (run from the repo root: python scripts/make_m3_fixtures.py).

* radar/fixtures/vendors.json      real input data: columns B-K of rows V-001..V-006
                                    of Meridian_Vendor_Input.xlsx (V-000 is skipped).
* radar/fixtures/m3/depth_plans.json  plan appendix A.3 tiers (draft numbers).
* radar/fixtures/m3/cache/<id>/    SYNTHETIC M3 output (documents.json, source_reports.json,
                                    snapshots/) for standalone testing of M3 and downstream modules.

EVERYTHING under fixtures/m3/cache is synthetic: URLs use the reserved ".invalid"
TLD, titles start with "[SYNTHETIC FIXTURE]", meta.synthetic is true and the text
never names a real vendor. It must never be presented as collected evidence.
"""
import hashlib, json, shutil, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar.contracts.models import Document
from radar.m3_sources import store
from radar.m3_sources.common import make_doc_id, make_report, sha256_hex
from radar.m3_sources.manual_inbox import add_manual_evidence

FX = ROOT / "radar" / "fixtures"
CACHE = FX / "m3" / "cache"
TS = "2026-10-09T09:00:00Z"
PREFIX = "[SYNTHETIC FIXTURE] "
CAV = {"CAREERS": "Job postings show capability and intent, not deployment: a posting that mentions AI is not proof that AI is used in a service delivered to Meridian.",
       "REPOS": "A dependency in a public repository may belong to samples, tooling or internal projects; it is not proof that the library is used in a service delivered to Meridian.",
       "FILINGS": "Statements in regulatory filings are corporate disclosures (often general or risk-factor language); they are not proof that AI is used in a specific service delivered to Meridian."}


def vendors():
    """Validate existing vendor profiles without overwriting them."""
    path = FX / "vendors.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Supply the original vendor workbook "
            "or restore vendors.json before generating fixtures."
        )

    profiles = json.loads(path.read_text(encoding="utf-8"))
    expected = {f"V-{i:03d}" for i in range(1, 7)}
    actual = {profile.get("vendor_id") for profile in profiles}

    if len(profiles) != 6 or actual != expected:
        raise ValueError(
            f"Expected vendor IDs {sorted(expected)}, got {sorted(actual)}"
        )

    print("Validated existing vendor profiles; left vendors.json unchanged.")


def depth_plans():
    plans = {
        "Critical": ["TRUST", "DOCS", "WEB", "CAREERS", "REPOS", "FILINGS", "MANUAL", "FINGERPRINT"],
        "High": ["TRUST", "DOCS", "WEB", "CAREERS", "REPOS", "FILINGS"],
        "Medium": ["TRUST", "DOCS", "WEB", "CAREERS"],
        "Low": ["TRUST", "DOCS"],
    }
    budget = {"Critical": (40, 3), "High": (25, 2), "Medium": (12, 2), "Low": (6, 1)}
    out = {t: {"tier": t, "sources": s, "max_pages": budget[t][0], "max_depth": budget[t][1],
               "stop_rule": "page budget, depth, or 3 consecutive pages without new AI terms; non-Critical also stops at a Confirmed signal"}
           for t, s in plans.items()}
    (FX / "m3" / "depth_plans.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def doc(vid, stype, key, title, text, meta):
    url = f"https://fixtures.invalid/{vid}/{stype.lower()}/{key}"
    snap_rel = f"{vid}/snapshots/fixture-{stype.lower()}-{key}.txt"
    p = CACHE / snap_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("SYNTHETIC FIXTURE - not real vendor evidence\n\n" + text + "\n", encoding="utf-8")
    m = {"synthetic": True, "fixture_note": "Synthetic test data; not collected from any real source.",
         "evidence_caveat": CAV.get(stype, ""), "collector": "m3/1.0-fixture", **meta}
    return Document(doc_id=make_doc_id(vid, stype, url), vendor_id=vid, source_type=stype, url=url,
                    fetched_at=TS, http_status=200, content_hash=sha256_hex(text), title=PREFIX + title,
                    text=text, snapshot_path=snap_rel, meta=m)


def put(vid, stype, docs, status, pages, note):
    store.write_source_results(vid, stype, docs, make_report(vid, stype, status, pages, note), cache_dir=CACHE)


def cache():
    if CACHE.exists():
        shutil.rmtree(CACHE)
    jp = lambda vid, k, t, body, ai, dept="Engineering": doc(
        vid, "CAREERS", k, t, f"{t}\n{body}",
        {"ats": "greenhouse", "job_id": k, "department": dept, "location": "Remote", "ai_terms": ai,
         "is_technical_role": True, "evidence_class": "capability_and_intent"})
    # V-001 High: no careers page, repos/filings not applicable
    put("V-001", "CAREERS", [], "EMPTY", 1, "no careers page found (homepage links and common paths returned nothing); a missing page is a finding, not a failure")
    put("V-001", "REPOS", [], "NOT_APPLICABLE", 0, "no public GitHub organisation clearly matches this vendor (accepted only on website/name match), so repositories were skipped")
    put("V-001", "FILINGS", [], "NOT_APPLICABLE", 1, "no SEC registrant with this exact company name in EDGAR's company list; treated as a non-filer")
    # V-002 Critical: everything OK
    put("V-002", "CAREERS", [
        jp("V-002", "1001", "Senior Machine Learning Engineer, Fraud",
           "You will build and deploy machine learning models that score card transactions for fraud. We use PyTorch and Python. You will own model monitoring and retraining pipelines.",
           ["machine learning", "pytorch"]),
        jp("V-002", "1002", "Applied AI Engineer, Document Processing",
           "Design retrieval-augmented generation services using large language model APIs for document summarisation. Experience with embeddings and vector databases is required.",
           ["large language model", "retrieval-augmented", "embeddings"]),
        jp("V-002", "1003", "Staff Platform Engineer",
           "Operate the Kubernetes platform that hosts our core processing services. Strong Go and Linux skills required.", [])],
        "OK", 3, "careers page https://fixtures.invalid/V-002/careers; ATS detected: greenhouse; read via public greenhouse job-board feed; 3 postings seen, 3 technical/data roles (2 mention AI terms), 3 documents kept")
    put("V-002", "REPOS", [doc("V-002", "REPOS", "sdk-requirements", "example-org/risk-sdk: requirements.txt",
        "scikit-learn==1.4.2\nlangchain>=0.2.0\nopenai==1.30.0",
        {"org": "example-org", "repo": "risk-sdk", "path": "requirements.txt", "branch": "main", "manifest_type": "requirements.txt",
         "matched_packages": ["langchain", "openai", "scikit-learn"], "is_example_path": False, "evidence_class": "capability_dependency"})],
        "OK", 4, "organisation 'example-org' accepted (organisation website matches vendor website); 6 repos scanned; 4 manifests read; 1 manifests contain AI libraries")
    put("V-002", "FILINGS", [
        doc("V-002", "FILINGS", "chunk-1", "10-K excerpt 1 of 2",
            "Our clients depend on the availability of our platforms. We use machine learning models to detect anomalous payment activity across our processing systems. These models are trained on historical transaction data. We review model outputs regularly. Regulators are increasing scrutiny of automated decision-making.",
            {"form": "10-K", "accession": "0000000000-26-000001", "item": "Item 1. Business", "matched_terms": ["machine learning"], "evidence_class": "regulatory_disclosure"}),
        doc("V-002", "FILINGS", "chunk-2", "10-K excerpt 2 of 2",
            "Competition is intense. New technologies such as generative AI may change client expectations. We may not successfully adopt these technologies. Failure to do so could harm our business. Our risk management framework addresses emerging technology risk.",
            {"form": "10-K", "accession": "0000000000-26-000001", "item": "Item 1A. Risk Factors", "matched_terms": ["generative ai"], "evidence_class": "regulatory_disclosure"})],
        "OK", 3, "synthetic filer; 2 keyword-in-context excerpts around AI terms")
    # V-003 High: careers error
    put("V-003", "CAREERS", [], "ERROR", 1, "careers page not reachable: timeout at https://fixtures.invalid/V-003/careers")
    put("V-003", "REPOS", [], "NOT_APPLICABLE", 0, "no public GitHub organisation clearly matches this vendor (accepted only on website/name match), so repositories were skipped")
    put("V-003", "FILINGS", [], "NOT_APPLICABLE", 1, "no SEC registrant with this exact company name in EDGAR's company list; treated as a non-filer")
    # V-004 Medium: careers OK; repos & filings skipped by plan (computed at run time)
    put("V-004", "CAREERS", [jp("V-004", "2001", "Data Engineer",
        "Build ETL pipelines that consolidate advisor commission data. Python, SQL and cloud data warehouses. Nothing in this role involves AI.", [], "Data")],
        "OK", 2, "careers page https://fixtures.invalid/V-004/careers; read via individual posting pages; 1 postings seen, 1 technical/data roles (0 mention AI terms), 1 documents kept")
    # V-005 High
    put("V-005", "CAREERS", [jp("V-005", "3001", "Machine Learning Engineer, Payments Analytics",
        "Develop predictive model features for payment risk analytics using Python and gradient boosting. Collaborate with data scientists on model validation.", ["machine learning", "predictive model"])],
        "OK", 3, "careers page https://fixtures.invalid/V-005/careers; ATS detected: lever; read via public lever job-board feed; 1 postings seen, 1 technical/data roles (1 mention AI terms), 1 documents kept")
    put("V-005", "REPOS", [], "EMPTY", 5, "organisation 'example-org' accepted (organisation name equals vendor name); 7 repos scanned; 5 manifests read; no AI libraries found in public dependency manifests")
    put("V-005", "FILINGS", [doc("V-005", "FILINGS", "chunk-1", "10-K excerpt 1 of 1",
        "We operate a global technology and control environment. Our teams apply artificial intelligence to improve operational efficiency. Governance committees oversee the use of these tools. Use cases are reviewed before deployment. Third-party model providers are subject to our vendor risk process.",
        {"form": "10-K", "accession": "0000000000-26-000002", "item": "Item 1. Business", "matched_terms": ["artificial intelligence"], "evidence_class": "regulatory_disclosure"})],
        "OK", 3, "synthetic filer; 1 keyword-in-context excerpts around AI terms")
    # V-006 Critical: careers blocked
    put("V-006", "CAREERS", [], "BLOCKED", 1, "careers page not readable: blocked (robots) at https://fixtures.invalid/V-006/careers")
    put("V-006", "REPOS", [], "NOT_APPLICABLE", 0, "no public GitHub organisation clearly matches this vendor (accepted only on website/name match), so repositories were skipped")
    put("V-006", "FILINGS", [], "NOT_APPLICABLE", 1, "no SEC registrant with this exact company name in EDGAR's company list; treated as a non-filer")
    # MANUAL examples (synthetic): primary for V-003, third_party for V-006
    for vid, cls, url, text in [
        ("V-003", "primary", "https://fixtures.invalid/V-003/exec-post", "Our team has rolled out an assistant that drafts customer notices and summarises case notes for our operations staff."),
        ("V-006", "third_party", "https://fixtures.invalid/V-006/press-article", "An industry article reports that the operator is piloting machine learning to flag unusual payment patterns.")]:
        d = add_manual_evidence(vid, url, text, cls, "2026-10-08", "", cache_dir=CACHE)
        d.title = PREFIX + d.title
        d.meta["synthetic"] = True
        d.meta["fixture_note"] = "Synthetic test data; not collected from any real source."
        store.upsert_document(d, cache_dir=CACHE)


if __name__ == "__main__":
    vendors(); depth_plans(); cache()
    print("fixtures written")
