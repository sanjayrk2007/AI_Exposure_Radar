# AI Exposure Radar integration patch

Copy these files into your existing project, preserving all other files:
- `m4_intel/run_pipeline.py` -> `<project>/m4_intel/run_pipeline.py`
- `tests/test_run_pipeline.py` -> `<project>/tests/test_run_pipeline.py`

Run from the project root with your virtual environment active:

```bash
python -m pytest -q
python -m m4_intel.run_pipeline \
  --manifest cache_pilot_v2/manifest.jsonl \
  --candidates data/candidates_pilot_v2.jsonl \
  --output-dir outputs/pilot_v2
```

The runner writes `candidate_signals.csv` and `pipeline_summary.json`. It does not
open or change the Excel workbook. It excludes V-000, adapts the actual
`char_start`/`char_end` candidate schema to the verifier's `start`/`end` schema,
and refuses to overwrite outputs unless `--overwrite` is passed.

Important: `quote_integrity=PASS` means the excerpt exactly matches the cached
document at the recorded offsets. It does not establish that the claim is true
or that it demonstrates the vendor's actual AI use. All factual review statuses
remain `UNREVIEWED`, and final vendor verdicts remain `Inconclusive` until
evidence is manually reviewed. A 403 or absent source is not evidence of no AI.
