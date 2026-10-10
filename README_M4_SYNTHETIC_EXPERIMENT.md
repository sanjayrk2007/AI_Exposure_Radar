# M4 synthetic experiment patch

This patch adds a reproducible synthetic-data experiment to the existing AI Exposure Radar project. It does **not** modify `data/labels.csv` or change the real-data readiness gate.

## Files added

- `m4_intel/generate_synthetic_data.py` — creates 120 labelled illustrative rows (48 `TECHNICAL`, 48 `VAGUE`, 24 `EXCLUDE`) using seed 42 by default.
- `m4_intel/run_synthetic_experiment.py` — compares the rule-based scorer with TF-IDF + Logistic Regression using stratified 5-fold cross-validation. The TF-IDF vectorizer is fitted inside each training fold.
- `tests/test_synthetic_experiment.py` — deterministic generation, labels/provenance, EXCLUDE handling, edge cases, output/report, and leakage tests.
- `data/synthetic_labels.csv` — generated example dataset.
- `m4_intel/models/specificity_v1/experiment_report.md` — example report.
- `m4_intel/models/specificity_v1/synthetic_metrics_example.json` — example synthetic metrics, named separately to avoid overwriting existing real-data metrics.

## Run in the project virtual environment

```bash
python -m m4_intel.generate_synthetic_data
python -m m4_intel.run_synthetic_experiment
python -m pytest -q tests/test_synthetic_experiment.py
python -m pytest -q
```

The runner writes a `synthetic_experiment` section into `m4_intel/models/specificity_v1/metrics.json`, preserving existing keys when that file is valid JSON. It writes `experiment_report.md` separately. The supplied `synthetic_metrics_example.json` is only an example output; regenerate metrics in your own environment.

## Important limitations

All generated sentences are synthetic, not public vendor evidence. Synthetic cross-validation only measures performance on these generated patterns. It does not validate real-world vendor generalization. Keep synthetic rows out of `data/labels.csv` and do not use the synthetic scores as the headline result for the real case study. The script does not persist a synthetic-trained model as the production inference model.
