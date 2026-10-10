# AI Exposure Radar: Final Plan (Case Study 1, 4-day build)

**Client scenario:** Meridian Financial, vendor AI usage analysis (Optiv Consulting case study)
**Team:** 5 people, 5 modules, about 4 days
**Status:** Draft v2 (adds the fine-tuned classifier plan). Numbers marked *draft* must be calibrated against Optiv's expected outcomes when they are shared.

> **How to use this document**
> Everyone reads sections 1 to 4 (about 10 minutes). After that, each person reads only their own module in section 5. Modules talk to each other only through the contracts in section 4, so you can build and test your module alone using fixtures.

---

## 1. The idea in one screen

**Problem.** Vendors increasingly use AI inside the services they sell to Meridian, often without saying so. Meridian's third-party risk team needs a repeatable, evidence-based way to tell real AI use from AI marketing, and to understand the risk.

**Our solution.** A tool that reads the Meridian vendor Excel and, for each vendor:
1. **Rates criticality** with fixed, explainable rules (not a gut call).
2. **Sets the investigation depth** from that tier (critical vendors get more sources and pages).
3. **Collects public evidence** with purpose-built collectors: trust and sub-processor pages, docs and product pages, job postings, public code repositories, SEC filings, plus a manual evidence inbox.
4. **Extracts and verifies signals.** Every piece of evidence is an exact quote from a real page, checked by code. A small fine-tuned classifier scores how technical each sentence is, and an optional LLM looks only at ambiguous sentences. Neither can invent evidence.
5. **Tags evidence strength** (Confirmed, Strong, Probable, Marketing-only) and gives a verdict: **Yes, No or Inconclusive**, adjusted by how much we actually managed to collect (coverage).
6. **Rates AI security risk** to Meridian and writes a recommended action.
7. **Fills the provided Excel** (columns L to V) and shows everything in a simple Streamlit app.

### Design principles
1. **No claim without a verified excerpt.** If the quote is not found on the page, it is rejected.
2. **Criticality drives depth.** Effort follows the tier, not how easy the vendor is to research.
3. **Absence of evidence is not evidence of absence.** Low coverage gives "Inconclusive", never "No".
4. **Deterministic first, LLM last.** Rules and a small fine-tuned classifier (benchmarked against simpler baselines) do the bulk; the LLM handles a narrow ambiguous band and can be switched off.
5. **Reproducible.** All rules, weights and thresholds live in config files, so another analyst gets the same result.
6. **Compliant.** Passive public fetching only. No logins, no contact with vendors, no public repository.

### Why this is impressive for Optiv
- It answers exactly what the brief asks (criticality, depth, sources, evidence weight, traceability, risk) as one connected method, not separate scripts.
- It is audit-grade: every finding has a verified quote, URL, date and page hash.
- It is honest: coverage-aware verdicts and a clear "Inconclusive" instead of over-claiming.
- It has a real, honestly evaluated ML component: a small fine-tuned classifier compared with simpler baselines using leave-one-vendor-out testing, not an LLM wrapper.
- It produces consultant-ready output: ratings with rationale and specific follow-up questions per vendor.
- It is a repeatable method Optiv could reuse on any vendor list, not a one-off demo.

### What we are deliberately NOT building
Real-time monitoring, patent and Wayback analysis, social-media scraping, fine-tuning a generative LLM (for example QLoRA on a 7B model), a database server, user accounts. These go on a "phase 2" slide.

---

## 2. Scope

| Level | Item |
|---|---|
| **MUST** | Rule-based criticality with explainable rationale and floors |
| **MUST** | Tier-driven depth plan (sources, page budgets, stop rule) |
| **MUST** | Web collector for trust, sub-processor, docs and product pages (HTML and PDF) |
| **MUST** | Careers and job-posting collector |
| **MUST** | Manual evidence inbox (analyst pastes URL, excerpt, screenshot path) |
| **MUST** | Candidate extraction, exact-quote verification, evidence tags, verdict with coverage |
| **MUST** | Risk matrix with rationale and recommended-action templates |
| **MUST** | Streamlit app and Excel export (L to V) with an Evidence Log sheet |
| **MUST** | Cache mode so the demo never depends on live websites |
| **SHOULD** | Public GitHub and dependency-manifest collector |
| **SHOULD** | SEC EDGAR collector for vendors that file (expected: Fiserv and BNY, verify) |
| **SHOULD** | Small fine-tuned classifier (technical vs marketing sentences), benchmarked against simpler baselines (Appendix E) |
| **SHOULD** | LLM adjudicator for the ambiguous band, with `--no-llm` fallback |
| **STRETCH** | Tech fingerprint (scan page and JS for AI provider strings) |
| **STRETCH** | Provider concentration view across vendors |
| **STRETCH** | Mitigating-signal detection (for example, "no training on customer data" statements) |
| **OUT** | Everything in the "not building" list above |

**Fallback ladder if behind schedule:** drop STRETCH first, then the ML classifier (use rules), then EDGAR, then GitHub. The MUST list alone is a complete, demo-able product.

---

## 3. Pipeline overview

```
Excel upload (vendors V-001 to V-006)
   |
   v
[M5] Criticality + Depth Plan  ->  tier, rationale, sources, page budget
   |
   v
[M2 web] + [M3 careers/repos/filings/manual]  ->  Documents + SourceReports  (cached)
   |
   v
[M4] Candidates -> Verify quotes -> Specificity -> Tags -> Verdict (with coverage)
   |
   v
[M5] Risk matrix + rationale + recommended action
   |
   v
[M1] Streamlit UI + Excel export (L to V) + Evidence Log + Coverage sheets
```

Module map:

| Module | Name | Skills | One-line goal |
|---|---|---|---|
| **M1** | Frontend and Orchestration | Streamlit, Python, openpyxl | Upload, run, show, export |
| **M2** | Web Collection | Scraping, HTTP, HTML and PDF parsing | Reliable, polite, cached web collection |
| **M3** | Source Collectors | APIs, scraping | Careers, GitHub, SEC filings, manual inbox |
| **M4** | ML and AI Intelligence | NLP, scikit-learn, Hugging Face transformers, LLM prompting | Turn documents into verified, tagged signals and a verdict |
| **M5** | Criticality, Risk and Strategy | Analysis, YAML, diagrams, writing | Rules, risk logic, templates, diagrams, deck |

---

## 4. Shared rules and contracts (everyone reads this)

### 4.1 Ground rules
- Python 3.11, type hints, small functions. One shared requirements file.
- Private Git repo or shared folder only. **Never public.** Do not commit API keys; use environment variables.
- `RADAR_MODE=cache` reads from `cache/`; `RADAR_MODE=live` fetches and writes to `cache/`.
- All IDs use the vendor ID from the Excel (for example `V-002`). All timestamps are ISO 8601 UTC.
- The Excel's `V-000` row is a worked example. Read it for tone and detail. Never process or edit it.
- Do not edit provided columns B to K. Do not add or remove vendors.
- Contract changes require agreement from all affected owners and a note in the group chat.

### 4.2 Repo layout

```
radar/
  config/        rubric.yaml  depth_plans.yaml  source_weights.yaml
                 lexicon.yaml  risk_matrix.yaml  templates.yaml
  contracts/     models.py
  m1_app/        app.py  pipeline.py  excel_io.py
  m2_web/        fetcher.py  discover.py  parse.py  fingerprint.py
  m3_sources/    careers.py  github.py  edgar.py  manual_inbox.py
  m4_intel/      candidates.py  verify.py  specificity.py
                 adjudicate.py  verdict.py  train_classifier.py
                 models/  (saved classifier and metrics, private only)
  m5_risk/       criticality.py  planner.py  risk.py  render.py
  cache/         <vendor_id>/documents.json  source_reports.json  snapshots/
  fixtures/      vendors.json  criticality.json  documents.json
                 signals.json  verdicts.json  risks.json
  data/          labels.csv  label_guide.md
  tests/
```

### 4.3 Data contracts (`contracts/models.py`)

```python
from dataclasses import dataclass, field

@dataclass
class VendorProfile:
    vendor_id: str
    name: str
    description: str
    service: str
    category: str
    website: str
    process: str
    dependency: str
    data_class: str
    data_volume: str

@dataclass
class CriticalityResult:
    vendor_id: str
    tier: str
    scores: dict
    floors_applied: list
    rationale: str

@dataclass
class DepthPlan:
    vendor_id: str
    tier: str
    sources: list
    max_pages: int
    max_depth: int
    stop_rule: str

@dataclass
class Document:
    doc_id: str
    vendor_id: str
    source_type: str
    url: str
    fetched_at: str
    http_status: int
    content_hash: str
    title: str
    text: str
    snapshot_path: str
    meta: dict = field(default_factory=dict)

@dataclass
class SourceReport:
    vendor_id: str
    source_type: str
    status: str
    pages_fetched: int
    note: str

@dataclass
class Signal:
    signal_id: str
    vendor_id: str
    doc_id: str
    source_type: str
    url: str
    excerpt: str
    char_start: int
    char_end: int
    category: str
    providers: list
    specificity: float
    weight: float
    score: float
    tag: str
    verified: bool
    method: str

@dataclass
class VendorVerdict:
    vendor_id: str
    ai_usage: str
    strength: str
    coverage: float
    signal_ids: list
    providers: list
    use_cases: list
    reasons: list

@dataclass
class RiskAssessment:
    vendor_id: str
    risk_class: str
    rationale: str
    recommended_action: str
```

**Allowed values**
- `source_type`: `TRUST`, `DOCS`, `WEB`, `FINGERPRINT`, `CAREERS`, `REPOS`, `FILINGS`, `MANUAL`
- `SourceReport.status`: `OK`, `EMPTY`, `BLOCKED`, `ERROR`, `NOT_APPLICABLE`, `SKIPPED_BY_PLAN`
- `Signal.tag`: `CONFIRMED`, `STRONG`, `PROBABLE`, `MARKETING_ONLY`, `NONE`
- `Signal.method`: `RULE`, `ML`, `LLM`, `MANUAL`
- `ai_usage`: `Yes`, `No`, `Inconclusive`
- `tier`: `Critical`, `High`, `Medium`, `Low`
- `risk_class`: `Critical`, `High`, `Medium`, `Low`, `None identified`

**Function signatures each module must expose**

```python
def assess_criticality(vendor: VendorProfile) -> CriticalityResult
def plan_depth(c: CriticalityResult, vendor: VendorProfile) -> DepthPlan
def collect_web(vendor: VendorProfile, plan: DepthPlan) -> tuple[list[Document], list[SourceReport]]
def collect_sources(vendor: VendorProfile, plan: DepthPlan) -> tuple[list[Document], list[SourceReport]]
def add_manual_evidence(vendor_id: str, url: str, text: str, source_class: str, captured_at: str, screenshot_path: str) -> Document
def analyse(vendor: VendorProfile, docs: list[Document], reports: list[SourceReport]) -> tuple[list[Signal], VendorVerdict]
def assess_risk(vendor: VendorProfile, c: CriticalityResult, v: VendorVerdict) -> RiskAssessment
```

### 4.4 Contract-first protocol (important)
Within the **first 2 hours of Day 1**, every owner writes a small hand-made fixture for **their output** in `fixtures/` (3 to 6 realistic records). Downstream owners start from those fixtures, so nobody waits.

### 4.5 Evidence tags, weights and verdict (owned by M4 and M5, all read)

**Score per signal:** `score = source_weight x specificity` (specificity from 0 to 1)

| Source type | Starting weight (*draft*) | Why |
|---|---|---|
| `TRUST` (sub-processor lists, DPA, AI policy) | 5.0 | Vendor's own formal disclosure |
| `DOCS` (API and product docs) | 4.0 | Specific, technical, vendor-authored |
| `FILINGS` (SEC) | 4.0 | Formal company statements |
| `FINGERPRINT` | 4.0 | Technical proof of deployed AI endpoints |
| `MANUAL` primary vendor statement | 3.5 | For example, executive post (analyst-set) |
| `CAREERS` | 2.5 | Shows capability and intent, not deployment |
| `REPOS` | 2.5 | May be samples, not the product |
| `MANUAL` third party | 2.0 | Secondhand |
| `WEB` generic (homepage, blog) | 1.5 | Often marketing |

**Tag from score**

| Score | Tag |
|---|---|
| 4.5 or more | `CONFIRMED` |
| 3.0 to 4.49 | `STRONG` |
| 2.0 to 2.99 | `PROBABLE` |
| above 0 and below 2.0 | `MARKETING_ONLY` |
| 0 | `NONE` |

**Verdict rules (draft, config-driven)**
- **Yes:** at least one `CONFIRMED`, or one `STRONG` plus one independent `PROBABLE` or better (different source types), or two `STRONG` from different source types.
- **No:** coverage at or above 0.7 **and** no signal above `MARKETING_ONLY`.
- **Inconclusive:** everything else (including only marketing signals, a single weak source, or low coverage).

**Coverage** = sum of weights of planned source types that returned at least one usable document, divided by the sum of weights of all planned source types.

**Specificity (0 to 1)** is the classifier probability (fine-tuned encoder if it wins the baseline ladder in Appendix E, otherwise the best simpler model). If no model file is present it falls back to the rule score below. The model sees the sentence text only, never the source type, so source weight is not counted twice. Rule factors:
- +0.4 names a provider, model or framework
- +0.3 describes a function in the service (what data, what task)
- +0.2 cites a technical artifact (API, SDK, version, configuration)
- +0.1 present-tense statement of current use
- minus 0.3 if future or hedged ("will", "exploring", "may")

**Disagreement rule:** if the classifier and the rule score differ by more than 0.4, treat the sentence as ambiguous: it goes to the LLM adjudicator (if enabled) and is flagged for analyst review in the app.

### 4.6 Config files and owners

| File | Owner | Shared with |
|---|---|---|
| `rubric.yaml` (criticality scoring and floors) | M5 | M1 (display) |
| `depth_plans.yaml` (tier to sources and budgets) | M5 | M2, M3, M1 |
| `source_weights.yaml` (weights, tag thresholds, verdict rules) | M4 | M5, M1 |
| `lexicon.yaml` (AI terms, providers, use-cases, hype terms) | M4 | M2, M3 (for link scoring) |
| `risk_matrix.yaml` | M5 | M1 |
| `templates.yaml` (rationale and action text) | M5 | M1 |

---

## 5. Module specs (read only your own)

### M1: Frontend and Orchestration

**Goal.** One simple app that runs the whole pipeline and produces the filled Excel.
**You need to know.** Sections 1 to 4, and the Excel mapping in section 6.
**Build against.** `fixtures/` from every module (you can finish the UI before others are done).

**Inputs.** The uploaded Excel; function outputs from M2 to M5.
**Outputs.** The running app; a filled workbook; Evidence Log and Coverage sheets.

**Tasks**
- **Day 1:** Streamlit skeleton with pages below, running entirely on fixtures. `excel_io.py`: read vendors from rows 6 to 11 (columns B to K), skip `V-000`.
- **Day 2:** `pipeline.py` orchestrator: per vendor, run criticality, plan, collect, analyse, risk. Isolate errors per vendor. Use a thread pool for collection. Respect `RADAR_MODE`. Add live progress callbacks.
- **Day 3:** Excel export: write L to V, keep formatting, add `Evidence Log` and `Coverage` sheets, then verify B to K are byte-identical to the input. Polish UI.
- **Day 4:** Bug fixes, demo rehearsal, backup screen recording.

**Pages**
1. **Upload and Vendors:** table of provided vendor data.
2. **Criticality and Plan:** tier badges, score breakdown, floors applied, depth plan. Analyst override with a mandatory justification (stored in the rationale).
3. **Collection Console:** per-vendor, per-source status (OK, EMPTY, BLOCKED), pages fetched, coverage bar, toggle for cache or live.
4. **Vendor Detail:** evidence cards with the excerpt highlighted, tag chip, source link, retrieval date, hash. Approve or reject per signal. Manual evidence form (calls `add_manual_evidence`).
5. **Risk Overview:** tier by risk heatmap and a vendor table.
6. **Export:** download the workbook.

**Tech.** Streamlit, pandas, openpyxl, plotly.
**Done when**
- Upload to export works in cache mode for all 6 vendors with no manual steps.
- Columns B to K of the output are unchanged.
- Evidence view shows highlighted quotes with link and date.
- Rejecting a signal re-computes the verdict.

---

### M2: Web Collection (scraping core)

**Goal.** A polite, reliable collector that discovers and caches the web part of each vendor's public footprint.
**You need to know.** Sections 1 to 4, `lexicon.yaml` (for link scoring), `depth_plans.yaml`.
**Build against.** `fixtures/vendors.json`, a hand-made `DepthPlan`.

**Inputs.** `VendorProfile` (website), `DepthPlan`.
**Outputs.** `Document` list with `source_type` in `TRUST`, `DOCS`, `WEB`, `FINGERPRINT`; `SourceReport` per source type; cache files and snapshots.

**Tasks**
- **Day 1:**
  - `fetcher.py`: HTTP client with a clear user-agent (for example, "RadarBot (student case study, passive public fetch)"), robots.txt check, per-host rate limit (at least 1.5 seconds), timeouts, retries, caching by URL, snapshot of raw response, SHA-256 content hash, UTC timestamp.
  - `discover.py`: from the vendor domain, read `sitemap.xml` and try candidate paths (`/trust`, `/security`, `/privacy`, `/legal`, `/subprocessors`, `/ai`, `/responsible-ai`, `/developers`, `/docs`, `/api`, `/products`). Score homepage links with keywords from `lexicon.yaml` and follow the best ones within `max_depth` and `max_pages`.
  - Run all 6 vendors in live mode and **freeze the cache by end of Day 1 or early Day 2.**
- **Day 2:**
  - `parse.py`: main-text extraction (trafilatura or BeautifulSoup), PDF text (pdfplumber), table-to-text for sub-processor lists, page classification (`TRUST` / `DOCS` / `WEB`) by URL and title keywords.
  - Optional browser-rendering fallback (Playwright) for JavaScript-only pages. Skip if time is short.
- **Day 3:**
  - Stop rule from plan (page budget, depth, or three consecutive pages with no new AI-related terms).
  - Emit `SourceReport` for every planned source (OK, EMPTY, BLOCKED, ERROR) with a short note. A missing trust page is a finding, not a failure.
  - STRETCH `fingerprint.py`: fetch homepage plus up to 10 linked scripts, grep for provider strings from `lexicon.yaml`, emit `FINGERPRINT` documents.
- **Day 4:** fix site-specific problems, support M1 with live demos.

**Tech.** requests or httpx, BeautifulSoup/lxml, trafilatura, pdfplumber, optional Playwright.
**Done when**
- Every document has URL, timestamp, status, hash and snapshot.
- Re-running in cache mode returns identical output.
- Robots.txt and rate limits are honoured; no login or form submission anywhere.
- Each vendor has either documents or a `SourceReport` explaining why not.

---

### M3: Source Collectors (careers, repos, filings, manual)

**Goal.** Add the other evidence families, so verdicts rest on independent sources.
**You need to know.** Sections 1 to 4, `depth_plans.yaml`, `lexicon.yaml`.
**Build against.** `fixtures/vendors.json`.

**Inputs.** `VendorProfile`, `DepthPlan`.
**Outputs.** `Document` list with `CAREERS`, `REPOS`, `FILINGS`, `MANUAL`; `SourceReport` per source.

**Tasks**
- **Day 1:** `manual_inbox.py` (**MUST**): `add_manual_evidence(...)` creates a `MANUAL` document with `meta.source_class` of `primary` or `third_party`, optional screenshot path, captured date. M1 builds the form; you own the function and its tests. Start `careers.py`.
- **Day 2:** `careers.py` (**MUST**):
  - Find the careers page from the homepage links or common paths.
  - Detect the applicant tracking system from URL patterns and use any public job feed it exposes (verify per vendor).
  - Fetch individual postings; keep technical and data roles; one `CAREERS` document per posting. Use the same politeness rules as M2 (share `fetcher.py`).
- **Day 3:** `github.py` (**SHOULD**): find the vendor's public organisation (accept it only if the profile website or name clearly matches; otherwise `SKIPPED`). Read dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`) and keep lines matching AI libraries. Respect API rate limits.
- **Day 3 to 4:** `edgar.py` (**SHOULD**): for vendors that file with the SEC, fetch the latest annual report text and return keyword-in-context chunks (two sentences either side of AI terms) as `FILINGS` documents with the filing URL. Follow the SEC's current access rules (declared user-agent, rate limits; verify). Mark `NOT_APPLICABLE` for non-filers.

**Tech.** requests, BeautifulSoup, GitHub REST API, SEC public data.
**Done when**
- Each collector returns a `SourceReport` even when it finds nothing.
- Manual evidence flows through to a verified, tagged signal.
- No LinkedIn or other login-gated scraping (executive posts go through the manual inbox, as in the V-000 example).

---

### M4: ML and AI Intelligence

**Goal.** Turn raw documents into a small set of verified, tagged signals and a defensible verdict, with a small fine-tuned classifier at the core of the specificity score.
**You need to know.** Sections 1 to 4, especially 4.5, and Appendices C, D and E.
**Build against.** `fixtures/documents.json` (hand-write 10 documents, including marketing-only pages and one sub-processor table).

**Inputs.** `Document` list, `SourceReport` list, `VendorProfile`.
**Outputs.** `Signal` list, `VendorVerdict`, and the saved classifier with its metrics file.

**Tasks**
- **Day 1:**
  - `lexicon.yaml`: providers, frameworks, techniques, use-case phrases, hype terms (starter list in Appendix C).
  - `candidates.py`: sentence split with character offsets; keep sentences matching the lexicon; for tables, keep rows.
  - `verify.py`: exact substring check against the document text; `verified=True` only if `text[char_start:char_end] == excerpt`. Reject everything else.
  - Run a 30-minute session with the team to agree the labelling rules in Appendix D.
- **Day 2:**
  - `specificity.py`: rule-based score from 4.5 (this is the baseline and the fallback); provider extraction; use-case category mapping.
  - `verdict.py`: tags, coverage calculation and verdict rules read from `source_weights.yaml`. Unit tests for every branch (confirmed, two strong, marketing only, low coverage, no signals).
  - Pre-label candidates from the real cache using the rule score and export `data/labels.csv`.
  - **Team labelling session (evening, all five people):** correct the labels. Target **at least 300 labelled sentences**, with at least 20 percent labelled twice to check agreement.
- **Day 3 morning:**
  - `train_classifier.py` (**SHOULD**): run the baseline ladder from Appendix E with leave-one-vendor-out evaluation (rules, TF-IDF plus logistic regression, sentence embeddings plus logistic regression, fine-tuned small encoder). Train the encoder on a free Colab GPU. Save the winner to `m4_intel/models/specificity_v1/` with `metrics.json`.
  - **Noon: choose the model** using the decision rule in Appendix E.
- **Day 3 afternoon:**
  - Plug the chosen model into `specificity.py` with automatic fallback to rules if model files are missing.
  - `adjudicate.py` (**SHOULD**): LLM call only for ambiguous candidates (specificity between 0.3 and 0.6, or classifier and rules disagree). Strict JSON output (`is_technical_evidence`, `reason_code`, `providers`, `quote`). Temperature 0. The returned quote must be a substring of the document, otherwise discard. Cache outputs by input hash so demos are deterministic. Provide `--no-llm` mode.
- **Day 4:** error analysis on all 6 vendors, tune thresholds, and produce the **evaluation table** for the deck (baseline ladder results, per-vendor folds, precision and recall).

**Tech.** Python, PyYAML, scikit-learn, Hugging Face transformers and PyTorch (training on Colab GPU, CPU inference), optional sentence-transformers, rapidfuzz (optional), one LLM API (optional).
**Done when**
- No unverified excerpt ever leaves the module.
- All 6 vendors produce a verdict in `--no-llm` mode, with and without the model files.
- The same input gives the same output twice.
- The model loads and scores a sentence on CPU with no internet.
- The evaluation table exists, with honest numbers, including the simpler baselines.
- Labelled data and model files stay in private storage only.

---

### M5: Criticality, Risk and Strategy

**Goal.** Make the reasoning explicit, reproducible and consultant-ready, and own the design deliverables.
**You need to know.** Sections 1 to 4, the Excel field guide, Appendices A and B.
**Build against.** `fixtures/vendors.json`, hand-written `VendorVerdict` fixtures (one for each branch).

**Inputs.** `VendorProfile`; later `VendorVerdict`.
**Outputs.** `CriticalityResult`, `DepthPlan`, `RiskAssessment`; diagrams; deck.

**Tasks**
- **Day 1:**
  - `rubric.yaml` and `criticality.py`: score the five criteria from the brief with fixed rules (Appendix A), apply escalation floors, produce a rationale from a template (so another analyst reaches the same answer).
  - `depth_plans.yaml` and `planner.py`: tier to sources, page budgets and stop rule (Appendix A.3).
  - Publish fixtures for criticality and depth plans in the first 2 hours.
- **Day 2:**
  - `risk_matrix.yaml` and `risk.py`: tier by evidence strength lookup, plus aggravators (Appendix B).
  - `templates.yaml` and `render.py`: text for Excel columns M, N, Q, T and U, using hedged language ("may be using") and specific follow-up questions per risk type.
  - Draft **Functional Design Diagram** (components and data flow).
- **Day 3:** **Architecture Diagram** (components inside each module, with interfaces). **Source Strategy** slide (sources, weights, why). Calibrate thresholds against Optiv's expected outcomes if shared. Run all 6 vendors end to end with M1.
- **Day 4:** assemble the deck (each owner supplies their slides, see section 8), rehearse, final proofreading of all Excel text.

**Tech.** Python, PyYAML, draw.io or Excalidraw, PowerPoint.
**Done when**
- The same row always yields the same tier, and the rationale names every criterion and floor.
- Depth in column N follows the tier, with a stop reason.
- Every risk has a rationale tied to data, dependency and evidence strength.
- Diagrams and deck are ready.

---

## 6. Excel mapping (columns L to V)

| Col | Field | Produced by | Content |
|---|---|---|---|
| L | Criticality Tier | M5 | `Critical`, `High`, `Medium`, `Low` |
| M | Criticality Rationale | M5 | Scores per criterion, floors applied |
| N | Assessment Depth Applied | M5 + M2/M3 | Sources actually consulted, pages, stop reason |
| O | AI Usage Detected | M4 | `Yes`, `No`, `Inconclusive` |
| P | Evidence Description and Source | M4 | Exact excerpt(s) in quotes, then source type, URL, retrieval date. No interpretation. |
| Q | How Vendor May Be Using AI | M4 + M5 | Use-case categories, hedged wording |
| R | Named AI Sub-processors / Model Providers | M4 | Providers named, or "None named" with reason |
| S | AI Security Risk Classification | M5 | `Critical`, `High`, `Medium`, `Low`, `None identified` |
| T | Risk Rationale | M5 | Data, dependency, evidence strength |
| U | Recommended Action | M5 | Concrete next step and questions |
| V | Assessed By / Date | M1 | Team name and date |

Extra sheets (confirm with Optiv first, see section 10): `Evidence Log` (every signal with excerpt, URL, retrieved date, hash, tag, score, verified), `Coverage` (per vendor and source: status, pages), `Config` (rubric and weights used).

---

## 7. Timeline and checkpoints

| When | Everyone | Checkpoint |
|---|---|---|
| **Day 1, hour 0 to 1** | Read sections 1 to 4, agree contracts | Contracts confirmed |
| **Day 1, hour 2** | Publish output fixtures | All fixtures in `fixtures/` |
| **Day 1, end** | M2 cache for first vendors, M5 rubric draft, M1 UI on fixtures | **Contracts frozen** |
| **Day 2** | Modules finished stand-alone, team labelling session in the evening (target 300 labels) | Integration run 1 (2 vendors, cache mode) |
| **Day 3** | Train and evaluate classifier (morning), pick the model (noon), integrate all 6, tune thresholds, export, polish | **Feature freeze** at end of day |
| **Day 4** | Bug fixes, rehearse twice, deck final, backup recording, package privately | Dry run for the demo |

**Daily:** a 15-minute sync (blockers and contract questions only).
**Interface change protocol:** post the change, wait for acknowledgement from affected owners, update the fixture.

---

## 8. Deck and demo

### 8.1 Brief outcomes to owners

| Expected outcome | Owner | Support |
|---|---|---|
| Functional Design Diagram | M5 | all |
| Architecture Diagram | M5 | M1 to M4 provide component lists |
| Source Strategy and evidence weights | M5 + M4 | M2, M3 |
| Criticality Ratings (criteria, depth) | M5 | M1 (display) |
| Working Demonstration | M1 | all |

Each owner supplies **two slides** about their module (what it does, one result or chart). M5 assembles.

### 8.2 Suggested 12 slides (10 to 15 minutes)
1. Problem and approach
2. Functional design diagram
3. Architecture diagram
4. Source strategy and evidence weights
5. Criticality criteria and depth by tier
6. Evidence tags and verdict logic
7. Collection engine and coverage
8. Intelligence layer and evaluation table
9. Results on the 6 vendors
10. Live demo
11. Risk and recommended actions
12. Limitations and phase 2

### 8.3 Demo script (5 minutes live, cache mode)
1. Upload the Excel and show tiers with reasons.
2. Show one Critical and one Low vendor: different depth plans.
3. Run collection and show the console (sources, coverage).
4. Open a vendor: evidence card with highlighted quote and tag. Reject a signal and show the verdict change.
5. Show an Inconclusive vendor caused by low coverage.
6. Add a manual evidence item.
7. Export the Excel and show columns L to V.

Rehearse twice. Have a screen recording as backup.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Site blocks or changes | Cache on Day 1; blocked sources are reported and lower coverage |
| LLM unavailable or inconsistent | `--no-llm` mode; cached outputs; LLM only in ambiguous band |
| Wrong GitHub organisation matched | Accept only if website or name clearly matches |
| Too little labelled data for fine-tuning | Aim for 300 or more labels, use the baseline ladder, and ship the simpler model if the encoder does not clearly win |
| Classifier looks good only because vendor pages are similar | Leave-one-vendor-out evaluation; drop near-duplicate boilerplate before splitting |
| Classifier is hard to explain | Show rule factors beside the probability; analyst can reject any signal |
| Weights feel arbitrary | Config-driven; calibrate on Optiv's expected outcomes; document rationale |
| Over-claiming | Verified quotes only; "Inconclusive" is a valid, defended outcome |
| Integration surprises | Fixtures, frozen contracts, integration run on Day 2 |
| Time pressure | Fallback ladder in section 2 |

**Ethics and rules.** Public pages only, polite rates, no logins, no form submissions, no vendor contact, no public repo, no LinkedIn scraping. Executive posts go through the manual inbox with a screenshot, as in the worked example.

---

## 10. Questions for the Optiv survey

1. The Field Guide says Evidence should be the exact excerpt only, but V-000 shows narrative evidence. Is excerpt plus source citation (URL, date) acceptable in column P?
2. The note says columns L to P are for students but the headers and Field Guide go to V. Confirm all of L to V are to be filled.
3. Does passive fetching of a vendor's public pages count as "contact"? (We assume not.)
4. Can we add extra sheets (Evidence Log, Coverage) to the workbook?
5. Is manual capture of executive posts (as in V-000) acceptable instead of automated collection?
6. Will expected outcomes be shared for calibration?

---

## Appendix A: Criticality rubric (*draft*)

### A.1 Criteria (each scored 0 to 3)

| Criterion | Weight | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|
| **D** Operational dependency (col I) | 0.25 | Critical or Total | High | Moderate | Low |
| **S** Data sensitivity (col J) | 0.25 | Customer master or PII, tax IDs, at population scale | Payment or financial data (instructions, account identifiers, holdings, compensation) | Business data or credentials only | None |
| **P** Payment-flow involvement (cols E, H) | 0.20 | Processes, transmits or settles payments | Handles payment-related data | Indirect (statements, batch jobs) | None |
| **R** Regulatory exposure (cols D to H) | 0.15 | Core banking or payment infrastructure | Regulated reports or notices (for example tax documents) | Indirect | None |
| **V** Volume (col K) | 0.15 | 1M or more per year | 100k to under 1M | Under 100k or periodic cycles | N/A |

**Weighted score** = 0.25D + 0.25S + 0.20P + 0.15R + 0.15V (range 0 to 3).

**Thresholds (*draft*):** Critical at 2.6 or more; High at 1.8 or more; Medium at 1.0 or more; otherwise Low.

### A.2 Escalation floors (explainable overrides)
- **F1:** privileged access to production systems or credentials means at least **High**.
- **F2:** operational dependency Critical or Total means at least **High**.
- **F3:** S = 3 and V = 3 (customer PII at population scale) means at least **High**.

### A.3 Depth plans (*draft*)

| Tier | Sources | Page budget | Notes |
|---|---|---|---|
| Critical | Trust, docs, web, careers, repos, filings (if filer), manual inbox prompt, fingerprint (stretch) | 40 pages, depth 3 | Keep going even after a Confirmed signal to find providers and use-cases |
| High | Trust, docs, web, careers, repos, filings (if filer) | 25 pages, depth 2 | Stop a source family at a Confirmed signal |
| Medium | Trust, docs, web, careers | 12 pages, depth 2 | |
| Low | Trust and docs only | 6 pages, depth 1 | |

**Stop rule:** stop a source family when the page budget is reached, or three consecutive pages add no new AI-related terms, or (non-Critical tiers) a Confirmed signal is found.

### A.4 Worked draft scoring on the six vendors (sanity check, not conclusions)

| Vendor | D | S | P | R | V | Score | Floors | Tier |
|---|---|---|---|---|---|---|---|---|
| V-001 AutomWorx | 2 | 1 | 1 | 1 | 0 | 1.10 | F1 (privileged access) | High |
| V-002 Fiserv | 3 | 3 | 3 | 3 | 3 | 3.00 | | Critical |
| V-003 FSSI | 1 | 3 | 1 | 2 | 3 | 1.95 | F3 | High |
| V-004 Terrapin | 1 | 2 | 0 | 1 | 1 | 1.05 | | Medium |
| V-005 BNY | 2 | 2 | 3 | 3 | 3 | 2.50 | | High |
| V-006 The Clearing House | 3 | 2 | 3 | 3 | 3 | 2.75 | | Critical |

Re-check these against the Excel text and Optiv's expected outcomes before using them.

---

## Appendix B: Risk matrix (*draft*)

### B.1 Base lookup (tier by evidence strength)

| Tier | CONFIRMED / STRONG | PROBABLE | MARKETING_ONLY / Inconclusive | No AI (high coverage) |
|---|---|---|---|---|
| Critical | Critical | High | Medium | None identified |
| High | High | Medium | Low | None identified |
| Medium | Medium | Low | Low | None identified |
| Low | Low | Low | Low | None identified |

### B.2 Aggravators (move up one band, cap at Critical)
- **A1:** AI confirmed or strong, **provider not named**, and data sensitivity S is 2 or more.
- **A2:** AI used for automated decisions or agent-style actions affecting customers.

### B.3 Risk types to cover in rationale and actions
1. Sensitive data exposed to AI models
2. Reliance on undisclosed AI sub-processors
3. Reduced transparency over how decisions affecting Meridian are made

Recommended actions are specific: a targeted questionnaire (model provider identity, hosting region, retention, training use of Meridian data, human review) with a time frame, and registration on the AI sub-processor register where applicable.

---

## Appendix C: Starter lexicon (extend freely, M4 owns)

- **Providers and platforms:** OpenAI, Azure OpenAI, Anthropic, Claude, Google Vertex AI, Gemini, AWS Bedrock, SageMaker, Hugging Face, Cohere, Mistral, Databricks
- **Frameworks and tooling:** TensorFlow, PyTorch, scikit-learn, LangChain, LlamaIndex, MLflow, Kubeflow
- **Techniques:** machine learning, large language model, LLM, generative AI, GenAI, retrieval-augmented, embeddings, fine-tuning, NLP, computer vision, anomaly detection, predictive model, neural network, agentic, copilot, intelligent document processing
- **Use-case phrases:** fraud detection, transaction scoring, document classification, chatbot, virtual assistant, summarisation, auto-drafting, forecasting
- **Hype terms (lower specificity):** AI-powered, next-generation, intelligent, smart, revolutionary, cutting-edge
- **Hedge terms (lower specificity):** will, plans to, exploring, may, roadmap

---

## Appendix D: Labelling guide (for the classifier)

**Unit.** One candidate sentence, shown with the sentence before and after. Label by what the sentence says, not by how trustworthy the source seems (source weight is handled separately).

**Labels**
- **TECHNICAL (1):** a current, concrete statement about AI or ML in the vendor's own product, service or operations. It names a provider, model or technique, or says what the AI does and with what data or task.
- **VAGUE (0):** generic claims, hype, benefits without mechanism, or future and hedged statements.
- **EXCLUDE:** not about AI, not about this vendor's own offering (general industry commentary), or boilerplate and duplicates. Excluded sentences are not used for training or evaluation.

**Decision questions (ask in order)**
1. Is it about AI or ML? If no, EXCLUDE.
2. Is it about this vendor's own product, service or operations? If no, EXCLUDE.
3. Is it current (present or past tense, not "will", "plans to", "exploring")? If no, VAGUE.
4. Does it say what the AI does, or name a provider, model or technique? If yes, TECHNICAL. Otherwise VAGUE.

**Invented examples (not from any vendor)**

| Sentence | Label |
|---|---|
| "Our fraud engine scores each card transaction with a gradient-boosted model trained on past chargebacks." | TECHNICAL |
| "Document summaries are generated by a large language model hosted by a third-party provider." | TECHNICAL |
| "Build and deploy machine learning models for our payments risk platform." (job posting) | TECHNICAL (the source weight, not the label, reflects that job postings show capability) |
| "We harness the power of AI to transform your business." | VAGUE |
| "We are exploring generative AI to improve client experience." | VAGUE |
| "The AI industry is evolving rapidly." | EXCLUDE |

**Process**
- Rules pre-label the sentences; humans correct them.
- At least 20 percent of sentences are labelled by two people. Compute Cohen's kappa; the target is 0.6 or higher (*draft*). If lower, refine this guide.
- Resolve disagreements by discussion and write down the rule you agreed.
- Hand-written or synthetic sentences may be used in training only, never in evaluation.

---

## Appendix E: Classifier training and evaluation spec

**Task.** Input: a sentence (text only, no source type). Output: probability that it is TECHNICAL.

**Baseline ladder (all evaluated the same way)**

| Rung | Model |
|---|---|
| 1 | Rule score from section 4.5 |
| 2 | TF-IDF plus logistic regression |
| 3 | Sentence embeddings (a small model such as MiniLM) plus logistic regression |
| 4 | Fine-tuned small encoder (for example DistilBERT or DeBERTa-small) |

**Training setup for rung 4 (*draft* starting values).** 3 to 5 epochs, learning rate between 2e-5 and 5e-5, batch size 16, max length 128, class weights, fixed random seed, early stopping.

**Evaluation**
- **Leave-one-vendor-out:** 6 folds (V-001 to V-006). Train on five vendors, test on the sixth.
- Report precision, recall and F1 for TECHNICAL, per fold and averaged. Expect noisy folds because vendors have different numbers of sentences; show that honestly.
- Check calibration with a simple binned table, because the probability is multiplied into the evidence score. Recalibrate (Platt or temperature scaling) if it is badly off.

**Decision rule (*draft*).** Ship the fine-tuned encoder only if it beats the best simpler rung by at least 3 macro-F1 points **and** is not worse than that rung by more than 10 points in any fold. Otherwise ship the simpler model and say so in the deck.

**Data hygiene**
- Remove near-duplicate boilerplate before splitting, and split by `vendor_id`.
- **Keep labelled data and model files private.** Do not upload them to Hugging Face Hub or any public site (the case rules forbid public repositories).

**Artifacts.** `m4_intel/models/specificity_v1/` (model, tokenizer or sklearn pipeline), `metrics.json`, the exact `labels.csv` used, and the training command with its seed.

**Inference contract.** `score_specificity(sentence: str) -> float`. It must work on CPU with no internet and fall back to the rule score if model files are missing.

---

## Appendix F: Tool list

| Area | Tools |
|---|---|
| App and export | Streamlit, pandas, openpyxl, plotly |
| Web collection | requests or httpx, BeautifulSoup with lxml, trafilatura, pdfplumber, optional Playwright |
| Other sources | GitHub REST API, SEC EDGAR public data, careers and job-board feeds, manual-evidence form |
| ML and AI | PyYAML, scikit-learn, Hugging Face transformers, PyTorch, optional sentence-transformers, rapidfuzz, hashlib, optional LLM API |
| Rules and design | PyYAML, Python rule engine, draw.io or Excalidraw, PowerPoint |
| Team | Python 3.11, pytest, private Git repo or shared drive, environment variables for keys, Google Antigravity |

Keep it lean: no Docker, Postgres, vector database or cloud deployment for this 4-day build.
