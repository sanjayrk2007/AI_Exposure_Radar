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
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
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
├── radar/
│   ├── __init__.py
│   ├── input_inspection.py
│   └── contracts/
│       └── __init__.py
├── fixtures/
│   └── README.md
├── tests/
│   └── test_input_inspection.py
├── cache/       # ignored; private generated data only
├── data/        # ignored; private labels only
└── m4_intel/
    └── models/  # ignored; local model artifacts only
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

## M4 integration entry point and evaluation hardening

- `m4_intel/analyse.py` implements the shared contract `analyse(vendor, docs, reports, planned_sources=None) -> (signals, verdict)`. It extracts candidates, keeps only quote-verified signals, scores specificity with the saved classifier when a usable local artifact exists (otherwise the rule score), and returns a coverage-aware verdict. `analyse_detailed` also returns signals flagged for review (classifier/rule disagreement above 0.4, or specificity in the 0.3–0.6 band). No LLM or network call is made; the flags are not auto-resolved. V-000 is refused.
- Model selection now also requires at least 3 leave-one-vendor-out folds whose held-out vendor has both classes, because single-class folds give degenerate macro-F1. This only tightens the existing gate.
- When real labels are evaluated, `metrics.json` includes an out-of-fold calibration table and Brier score for the TF-IDF probabilities. It is descriptive only and no recalibration is applied.
- Synthetic results remain in the separate `synthetic_experiment` section. Synthetic families are label-pure and built around the same cues as the rule score, so near-perfect synthetic TF-IDF scores are not evidence for model selection.

### Reproducible real-label reporting

`python -m m4_intel.train_specificity` writes `metrics.json`, `real_evaluation_report.md` and, only when TF-IDF is selected, `tfidf_logistic_regression.joblib` plus `model_manifest.json` (scikit-learn version, seed, row count, labels hash) into `m4_intel/models/specificity_v1/` (git-ignored). `selection_decision` in the metrics records why a model was or was not chosen. Inference (`score_specificity`, `analyse`) loads the artifact once per file version and falls back to the rule score if the file is missing, corrupt, the wrong object, or was built with a different scikit-learn version. A stale artifact is deleted when a run does not select TF-IDF. `load_labels` rejects synthetic rows (`SYN*` doc ids, a `provenance` column marked synthetic, or `data/synthetic_labels.csv` itself). The synthetic report (`experiment_report.md`) stays separate and is never used for selection.


## End-to-end demo (M4 scope)

```bash
python -m m4_intel.run_demo --workbook /path/to/Meridian_Vendor_Input.xlsx --output /path/to/report.xlsx [--no-llm] [--mode live|cache]
```

Flow: read the workbook (V-000 is never processed) -> collect public pages (robots.txt honoured, at least 1.5 s between requests, pages typed TRUST/DOCS/WEB, well-known trust and docs paths probed) -> optional local-LLM review of ambiguous sentences -> exact-quote verification, integrity and attribution checks -> coverage-aware verdict -> a **new** workbook. The input workbook is never modified, and the exporter refuses an output path equal to the input.

- Only columns O, P, Q and R of the six real vendors are filled; B to K are verified unchanged after saving. L to N and S to V are not assessed by this module.
- Extra sheets: `Evidence Log` (every signal with offsets, URL, retrieval date, hash, tag, proposer, `review_status` of `RULE_PROPOSED_UNREVIEWED`, `ML_PROPOSED_UNREVIEWED` or `LLM_PROPOSED_UNREVIEWED`, and a blank `human_decision` for the analyst), `Coverage` (per-source status and verdict reasons), `Run Notes`.
- Coverage is computed against the planned families (TRUST, DOCS, WEB). Blocked, failed or missing sources lower coverage and are reported; they never produce a `No` verdict. `No` is also refused whenever a verified signal names a provider or has specificity of 0.5 or more.
- Local LLM: Ollama-compatible server on a loopback address only (`RADAR_LLM_URL`, default `http://127.0.0.1:11434`; `RADAR_LLM_MODEL`, default `qwen2.5:7b-instruct`). Non-local URLs are refused. It sees only ambiguous sentences, uses temperature 0 and JSON output, and a quote is accepted only as an exact substring of the verified sentence. It can raise a sentence's specificity to 0.7 but never lower it, and its output is labelled `LLM_PROPOSED_UNREVIEWED`. Responses are cached under `<cache-dir>/llm`. If the server is unreachable the run continues with rule/ML scores.
- Use `--mode cache` to re-run from saved collections without touching the network. Cache, outputs and `*.xlsx` are git-ignored.
- A page statement is a vendor claim, not independent proof of how or whether a capability is deployed.

