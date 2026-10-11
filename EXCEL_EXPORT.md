# Excel export patch

Copy:
- `m4_intel/export_workbook.py` into your project's `m4_intel/`
- `tests/test_export_workbook.py` into your project's `tests/`

Then run:
```bash
python -m pytest -q
python -m m4_intel.export_workbook \
  --workbook "Meridian_Vendor_Input.xlsx" \
  --manifest cache_pilot_v2/manifest.jsonl \
  --candidates data/candidates_pilot_v2.jsonl \
  --output outputs/AI_Exposure_Radar_Provisional.xlsx
```

This creates a separate workbook copy with an `AI Radar Summary` sheet. It does not fill the original vendor output columns because their exact mapping has not been confirmed from the local workbook headers. All verdicts remain `Inconclusive` pending factual evidence review. If the output already exists, choose another path or add `--overwrite`.
