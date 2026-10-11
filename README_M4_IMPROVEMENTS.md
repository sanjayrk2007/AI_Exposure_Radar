# M4 Evaluation Improvements Patch

This patch extends the existing M4 experiment while preserving real labels and the production readiness gate.

## Included files

- `m4_intel/generate_synthetic_data.py` — reproducible synthetic examples with `template_family` metadata and harder future/marketing examples.
- `m4_intel/run_synthetic_experiment.py` — same-fold rule/TF-IDF comparison, row-stratified CV, template-family-held-out `StratifiedGroupKFold`, per-class metrics, confusion matrices, invalid-fold reporting, error examples, model configuration, and a separate real-data status section in the report.
- `m4_intel/build_real_label_review_queue.py` — builds a traceable human review queue from cached candidates, skips existing labels and V-000, adds context only when offsets verify, and never auto-labels.
- `tests/test_synthetic_experiment.py` — synthetic generation, provenance, row/grouped evaluation, leakage, output/report and edge-case tests.
- `tests/test_real_label_review_queue.py` — review queue provenance/deduplication/blank-label tests.
- `tests/test_specificity_fallback.py` — missing/corrupt model fallback and input edge-case tests.
- `patches/score_specificity_fallback.patch` — a narrow patch to make the existing `score_specificity` function fall back safely if a saved model artifact is missing, corrupt, incompatible, or unusable. This patch deliberately avoids replacing the whole training module.

## Apply safely

From the root of the existing AI Exposure Radar repository, merge the included files. The generator, runner, and synthetic tests are updated versions of files from the earlier synthetic experiment patch; back them up first if you have local edits not yet saved.

Apply the narrow fallback change from the project root:

```bash
git apply patches/score_specificity_fallback.patch
```

If `git apply` reports a context mismatch, do not force it. Open `m4_intel/train_specificity.py` and apply the small change to `score_specificity()` manually, preserving the rest of your current training/readiness-gate code.

## Run

```bash
python -m m4_intel.generate_synthetic_data
python -m m4_intel.run_synthetic_experiment
python -m m4_intel.build_real_label_review_queue \
  --candidates data/candidates_pilot_v2.jsonl \
  --labels data/labels.csv \
  --manifest cache_pilot_v2/manifest.jsonl \
  --output data/real_label_review_queue.csv
python -m pytest -q tests/test_synthetic_experiment.py tests/test_real_label_review_queue.py tests/test_specificity_fallback.py
python -m pytest -q
```

The review queue is for human review only. Manually review each row in context before appending a decision to `data/labels.csv`; do not treat candidate extraction as a label. Keep vendor data and evidence files out of public repositories.

## Outputs and interpretation

- `data/synthetic_labels.csv` is generated locally with a fixed seed. It is synthetic only and must not be merged into `data/labels.csv`.
- `m4_intel/models/specificity_v1/metrics.json` preserves existing real-data metrics and adds/updates a separate `synthetic_experiment` section.
- `m4_intel/models/specificity_v1/experiment_report.md` reports stratified and template-family-held-out synthetic metrics, per-class scores, confusion matrices, grouped-fold validity/coverage, selected error examples, model configuration, and real-data readiness separately.
- Template-family-held-out evaluation is a stronger synthetic stress test, not evidence of generalization to real vendors. If families are correlated with labels or remain linguistically simplistic, grouped scores can still be optimistic.
- The real-data readiness gate remains unchanged. Synthetic scores do not qualify the learned model for production; retain the rule fallback until real labels from multiple vendors support defensible held-out evaluation.
