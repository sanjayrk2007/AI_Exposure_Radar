# AI Exposure Radar — Phase 0 scaffold

Private Optiv Consulting case-study project. This phase establishes the repository and input contract only; it does **not** collect public pages, label evidence, train models, or write results back to the supplied workbook.

## Requirements

- Python 3.11 (the shared project plan specifies Python 3.11)
- Git recommended; keep the repository private.

## Local setup

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Put the provided workbook in a private local location outside version control. To inspect vendor profiles from Python:

```python
from radar.input_inspection import read_vendor_profiles

vendors = read_vendor_profiles("/path/to/Meridian_Vendor_Input.xlsx")
print(len(vendors))  # expected: 6 non-example vendors
```

The reader opens the workbook read-only, reads columns B–K starting at row 5 using row 4 as headers, and excludes `V-000`. It does not save the workbook or edit any cells.

## Phase 0 boundaries

- No web scraping, network calls, public-source collection, model training, or fabricated fixture data.
- Never process or modify `V-000`.
- Never overwrite the original workbook. Excel export belongs to a later phase and must write to a separate output file.
- Do not commit the source workbook, case brief, private cache, labelled data, credentials, or model artifacts.
- The workbook's own instruction says findings must remain inside the exercise; no public repository, published presentation, or vendor contact.

## Fixture contracts

See [`fixtures/README.md`](fixtures/README.md) for `documents.json` and `source_reports.json` formats. The shared plan's `Document` contract uses `fetched_at` for the retrieved timestamp. M4's minimum input fields are `vendor_id`, `doc_id`, `source_type`, `url`, `fetched_at`, `content_hash`, and `text`.

## Current layout

```text
ai_exposure_radar_phase0/
├── .gitignore
├── README.md
├── requirements.txt
├── config/                  # M5 & shared configuration files
│   ├── rubric.yaml          # Criticality weights, criteria, and escalation floors (Appendix A)
│   ├── depth_plans.yaml     # Per-tier investigation budgets and stop rules (Appendix A.3)
│   ├── risk_matrix.yaml     # Tier × Evidence strength lookup and aggravators (Appendix B)
│   ├── templates.yaml       # Audit-grade prose templates for Excel columns M, N, Q, T, U
│   ├── lexicon.yaml         # AI provider, technique, and use-case dictionaries
│   └── source_weights.yaml  # Source credibility weights and verdict thresholds
├── radar/
│   ├── __init__.py
│   ├── input_inspection.py
│   └── contracts/
│       ├── __init__.py
│       └── models.py        # Shared dataclasses (VendorProfile, CriticalityResult, DepthPlan, etc.)
├── fixtures/
│   ├── README.md
│   ├── vendors.json         # Reference vendor profiles
│   ├── criticality.json     # Sample CriticalityResult and DepthPlan fixtures
│   ├── documents.json       # Mock document inputs
│   ├── signals.json         # Mock candidate signals
│   ├── verdicts.json        # Mock M4 vendor verdicts
│   └── risks.json           # Sample RiskAssessment fixtures
├── m1_app/                  # Frontend and pipeline orchestrator
│   ├── app.py               # Streamlit application
│   ├── pipeline.py          # End-to-end multi-module runner
│   └── excel_io.py          # Excel reader/exporter preserving columns B–K
├── m2_web/                  # Public web footprint collector
│   ├── fetcher.py           # Polite HTTP client with rate-limiting & caching
│   ├── discover.py          # Trust, docs, and product page discovery
│   ├── parse.py             # Main text, table, and PDF extraction
│   └── fingerprint.py       # JavaScript & endpoint AI provider scanner
├── m3_sources/              # Auxiliary source collectors
│   ├── careers.py           # Job board & ATS posting extractor
│   ├── github.py            # Public organization manifest collector
│   ├── edgar.py             # SEC annual filing analyzer
│   └── manual_inbox.py      # Analyst inbox for primary vendor statements
├── m4_intel/                # Intelligence and ML layer
│   ├── candidates.py        # Sentence extraction and character offset indexing
│   ├── verify.py            # Exact substring quote integrity verification
│   ├── specificity.py       # Sentence specificity scoring (classifier & rules)
│   ├── adjudicate.py        # Ambiguous band handling & LLM adjudicator
│   ├── verdict.py           # Tagging, coverage calculation, and final verdict
│   ├── train_specificity.py # Classifier baseline evaluation ladder
│   └── models/              # Local trained models and metrics (git-ignored)
├── m5_risk/                 # Criticality, Risk, and Strategy Engine
│   ├── criticality.py       # Rule-based criticality scoring & floor enforcement
│   ├── planner.py           # Tier-to-depth allocation and crawl budget planner
│   ├── risk.py              # Risk matrix lookup & aggravator processor
│   └── render.py            # Consultant-grade prose renderer for Excel export
├── tests/
│   ├── test_input_inspection.py
│   └── test_criticality.py
├── cache/                   # Ignored; raw cached responses and SHA-256 snapshots
└── data/                    # Ignored; private candidate labels (labels.csv)
```

## Phase 1 — candidate extraction

M4 candidate extraction is implemented in `m4_intel/candidates.py`, with configurable terminology in `config/lexicon.yaml`. See [`m4_intel/README.md`](m4_intel/README.md) for the output contract and offset invariant. Candidate extraction only identifies terminology-matching sentences/rows; it does not establish AI usage or assign evidence verdicts.

## Phase 5 — specificity classifier (manual labels required)

Phase 5 evaluates the existing deterministic rule score against TF-IDF + Logistic Regression using six leave-one-vendor-out folds. It does not create or infer labels. `data/labels.csv` is a private, header-only template until candidate sentences from actual cached vendor documents have been manually reviewed. Use `TECHNICAL` (or `1`), `VAGUE` (or `0`), and `EXCLUDE`; EXCLUDE rows are never trained or evaluated. Only `V-001` through `V-006` are accepted; the worked example `V-000` is prohibited.

After adding reviewed rows, run:

```bash
python -m m4_intel.train_specificity
```

The command writes `m4_intel/models/specificity_v1/metrics.json` and only saves a local TF-IDF pipeline if it performs at least as well as the rule baseline on mean fold macro-F1. The public inference helper is `from m4_intel.scoring import score_specificity`; it returns a classifier probability when a saved pipeline is available, otherwise it falls back to `score_specificity_rule`. The fallback is a heuristic score, not a calibrated probability. Model files and metrics stay local and are ignored by Git.

No cached vendor documents are currently present in this workspace, so no candidate sentences or labels have been fabricated. Do not train a Transformer or use Colab until the reviewed sample is large and balanced enough to justify it. Public datasets are not required for this Optiv-specific evaluation.

## Module M5 — Criticality, Risk, and Strategy

Module M5 executes the rules-based criticality scoring engine, determines the evidence collection depth budget, maps vendor exposure through the AI Security Risk Matrix, and populates consultant-grade narrative rationales for Excel columns L–V.

### Core Architecture & Responsibilities

- **`m5_risk/criticality.py`**: Implements `assess_criticality(vendor: VendorProfile) -> CriticalityResult`. Evaluates the five dimensions specified in the brief using configuration parameters in `config/rubric.yaml`:
  - $D$ (Operational Dependency, Col I): Weighted at $0.25$.
  - $S$ (Data Sensitivity, Col J): Weighted at $0.25$.
  - $P$ (Payment-Flow Involvement, Cols E, H): Weighted at $0.20$.
  - $R$ (Regulatory Exposure, Cols D–H): Weighted at $0.15$.
  - $V$ (Transaction/Record Volume, Col K): Weighted at $0.15$.
  - Evaluates weighted score: $\text{Score} = 0.25D + 0.25S + 0.20P + 0.15R + 0.15V$ (range $0.0$ to $3.0$).
  - Base classification bands: $\text{Critical} \ge 2.6$, $\text{High} \ge 1.8$, $\text{Medium} \ge 1.0$, $\text{Low} < 1.0$.
  - Applies mandatory escalation floor rules:
    - **$F_1$**: Privileged access to production environments, job schedules, or service credentials escalates tier to at least **High**.
    - **$F_2$**: Operational dependency rated Critical or Total escalates tier to at least **High** (and **Critical** if score $\ge 2.6$).
    - **$F_3$**: Population-scale PII ($S=3$ and $V=3$) escalates tier to at least **High**.

- **`m5_risk/planner.py`**: Implements `plan_depth(c: CriticalityResult, vendor: VendorProfile) -> DepthPlan`. Enforces tier-driven investigation bounds from `config/depth_plans.yaml`:
  - **Critical**: Budget of 40 pages, crawl depth 3 across all sources (Trust, Docs, Web, Careers, Repos, Filings, Manual inbox, Fingerprint). Stop rule: Continue beyond Confirmed signal to identify providers and use cases.
  - **High**: Budget of 25 pages, crawl depth 2 across standard sources. Stop rule: Stop a source family upon discovering a Confirmed signal.
  - **Medium**: Budget of 12 pages, crawl depth 2 across Trust, Docs, Web, and Careers.
  - **Low**: Budget of 6 pages, crawl depth 1 restricted to Trust and Docs.

- **`m5_risk/risk.py`**: Implements `assess_risk(vendor: VendorProfile, c: CriticalityResult, v: VendorVerdict) -> RiskAssessment`. Executes the 2D risk lookup defined in `config/risk_matrix.yaml`:
  - Intersects Criticality Tier with evidence strength (`CONFIRMED`, `STRONG`, `PROBABLE`, `MARKETING_ONLY`, `Inconclusive`, `No AI with high coverage`).
  - Evaluates risk aggravators to increase classification by one severity band:
    - **$A_1$**: AI Confirmed or Strong, model provider is not named/disclosed, and data sensitivity $S \ge 2$.
    - **$A_2$**: AI utilized in automated customer-affecting decisions or autonomous agentic workflows.

- **`m5_risk/render.py`**: Synthesizes structured data and context using `config/templates.yaml` into consultant-ready narratives:
  - Renders detailed numerical breakdowns and floor audit trails for Column M.
  - Formats actual sources consulted, page counts, and coverage ratios for Column N.
  - Formats data dependency risk explanations for Column T.
  - Generates actionable vendor questionnaire requests, sub-processor registration tasks, and business-day SLAs (15/30/60/90 days) for Column U.

### Excel Export Mapping (Columns L–V)

The table below outlines the ownership and structure of the fields populated by M5 in the master workbook:

| Column | Field Name | Produced By | Required Content & Structure |
| :---: | :--- | :---: | :--- |
| **L** | Criticality Tier | **M5** | Categorical tier: `Critical`, `High`, `Medium`, or `Low`. |
| **M** | Criticality Rationale | **M5** | Mathematical score breakdown ($D, S, P, R, V$), floor triggers ($F_1, F_2, F_3$), and process scope. |
| **N** | Assessment Depth Applied | **M5 + M2/M3** | Planned tier budget vs. actual sources queried, pages fetched, coverage ratio, and stop reason. |
| **O** | AI Usage Detected | M4 | Final intelligence verdict: `Yes`, `No`, or `Inconclusive`. |
| **P** | Evidence Description & Source | M4 | Exact verbatim quotes in quotation marks, source URL, retrieval date, and content hash. |
| **Q** | How Vendor May Be Using AI | **M4 + M5** | Hedged domain use cases based on detected category keywords. |
| **R** | Named AI Sub-processors | M4 | Identified 3rd-party foundation model providers or explicit absence statement. |
| **S** | AI Security Risk Classification | **M5** | Categorical risk: `Critical`, `High`, `Medium`, `Low`, or `None identified`. |
| **T** | Risk Rationale | **M5** | Audit narrative linking data classification and operational dependency to AI evidence strength. |
| **U** | Recommended Action | **M5** | Targeted questionnaire focus areas (retention, training exclusions), inventory update, and timeline. |
| **V** | Assessed By / Date | M1 | Assessment stamp and ISO timestamp. |

### Standalone M5 Execution

To run the M5 pipeline locally and generate completed audit workbooks without relying on C-compiler dependencies or external services:

```bash
python run_m5.py
```

The execution pipeline processes `verify_no_llm.xlsx` and `verify_with_llm.xlsx`, guarantees byte-identical preservation of vendor columns B–K, and outputs `verify_no_llm_M5_completed.xlsx` and `verify_with_llm_M5_completed.xlsx`.