# Evidence review and verdict integration patch

Copy:
- `m4_intel/build_evidence_review.py`
- `m4_intel/run_reviewed_verdicts.py`
- `tests/test_evidence_review_verdicts.py`

Run:
```bash
python -m pytest -q
python -m m4_intel.build_evidence_review \
  --manifest cache_pilot_v2/manifest.jsonl \
  --signals outputs/pilot_v2/candidate_signals.csv \
  --output data/evidence_review.csv
```

Open `data/evidence_review.csv`. For each row, change `factual_review_decision` from `PENDING` to:
- `APPROVED`: the surrounding source context supports a current AI-use claim by this vendor.
- `REJECTED`: it does not support that claim, is unrelated, or is misleading in context.
Leave uncertain rows as `PENDING`. Add a concise reason in `review_reason`.

Then run:
```bash
python -m m4_intel.run_reviewed_verdicts \
  --manifest cache_pilot_v2/manifest.jsonl \
  --review data/evidence_review.csv \
  --output outputs/pilot_v2/vendor_verdicts.json
```

Only APPROVED rows are passed to `generate_verdict`. This patch deliberately treats all current collection as WEB-only and plans TRUST, DOCS, FILINGS, FINGERPRINT, CAREERS, REPOS, WEB source families, so low coverage should yield Inconclusive rather than an unsupported No. It does not create final findings from the earlier classifier labels. Review decisions are human-entered, and final results remain subject to review of weights/coverage plans. Do not commit review data or outputs to a public repo.
