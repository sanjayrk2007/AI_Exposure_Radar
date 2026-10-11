# M4 intelligence module

## Candidate extraction (Phase 1)

`extract_candidates(document_text, lexicon=None)` returns a list of dictionaries:

- `text`: original sentence or line/table row, with surrounding whitespace trimmed
- `start`: inclusive character offset into the original `Document.text`
- `end`: exclusive character offset into the original `Document.text`
- `categories`: configured lexicon categories matched in the candidate

The required invariant is `document_text[start:end] == text`. Sentence/row spans are not rewritten. Terminology matches are case-insensitive. The default lexicon is `config/lexicon.yaml`; callers may pass a mapping for testing or custom configuration. `extract_document_candidates` accepts a mapping-shaped Document contract and reads its `text` field.

Candidate extraction is a recall-oriented filtering step only. It does not verify source authenticity, infer actual AI use, score specificity, label evidence, or assign a verdict. Tests use mock text only; it is not vendor evidence.

## Status and limitations (read before relying on M4)

**Code is complete; data is not.** Everything below describes what the code does today, not a trained result.

- **Specificity scoring.** `analyse()` scores every verified sentence with the rule score (`specificity.py`). A learned
  TF-IDF + logistic-regression model is used only if a saved artifact passes every check; otherwise the rule score is
  used and the signal's `method` is `RULE`. Nothing here claims a classifier has been trained.
- **When a learned model is used.** `python -m m4_intel.train_specificity` evaluates the rule baseline and TF-IDF with six
  leave-one-vendor-out folds and writes `metrics.json` and `real_evaluation_report.md`. It saves a model only if all of these hold:
  all six vendor folds are evaluable, at least 3 folds contain both classes, TF-IDF beats the rule baseline by at least
  0.03 macro-F1 on those folds, and at least `MIN_ROWS_TO_DEPLOY` (300, the plan's labelling target; draft/config) reviewed
  rows exist. Below that the status is `evaluated_rule_fallback_retained` and the report says why. Statuses:
  `awaiting_manual_labels`, `INSUFFICIENT_DATA` (fewer than 30 rows, 5 per class, or a class covering fewer than 2 vendors),
  `evaluated_rule_fallback_retained`, `evaluated_model_selected`.
- **Artifact safety.** `model_manifest.json` records the scikit-learn version and the SHA-256 of the model file. Inference
  refuses (falls back to rules) when the manifest is missing, malformed, from another scikit-learn version, or does not match
  the model file. Every training run that does not select a model deletes any previous artifact.
- **Not implemented (by decision).** Sentence-embedding and fine-tuned-encoder baselines (Appendix E rungs 3 and 4), probability
  recalibration (the calibration table is descriptive only) and `adjudicate.py`. `analyse()` never calls an LLM or the
  network, so the plan's `--no-llm` behaviour is the only behaviour. Ambiguous sentences (classifier/rule disagreement above
  0.4, or specificity in 0.3 to 0.6) are flagged by `analyse_detailed(...).ambiguous`; the contract-shaped `analyse()` returns
  only `(signals, verdict)`.
- **Coverage.** If `analyse()` is given neither `planned_sources` nor `SourceReport`s, coverage is 0.0 and the verdict can
  never be `No`. Collectors must send a `SourceReport` for every planned source.
- **Quote verification** proves an excerpt exists verbatim in a document, not that the vendor's claim is true.

### Adding real labels safely

1. Build the review queue from cached candidates: `python -m m4_intel.build_real_label_review_queue`. Rows are ordered so
   vendors with the fewest reviewed labels come first, alternating `rule_high` and `rule_low` candidates (columns
   `rule_score` and `queue_stratum`; a reading aid only, never a label, never merged). Review both strata: labelling
   only sentences the rule already likes would flatter the rule baseline. Re-running the builder refuses to overwrite a
   queue file that already holds decisions (merge them first, or pass `--force`).
2. A human reads each sentence in context and fills `reviewer_label` (TECHNICAL, VAGUE or EXCLUDE) and sets
   `review_status` to `REVIEWED`. Do not auto-label. Do not relabel rows already in `data/labels.csv`.
3. Check, then merge: `python -m m4_intel.merge_reviewed_labels --dry-run`, then without `--dry-run`. The merge only appends
   new rows, writes nothing if any row is invalid or conflicts with an existing review, and rejects V-000 and synthetic rows.
4. `load_labels` raises if the same sentence carries two different labels; resolve that by hand.

`data/synthetic_labels.csv` is for development only and is never real evidence of performance. The review queue and
`data/labels*.csv` are git-ignored because they contain real vendor text.
