# Phase 5 operating notes

1. Collect sentence candidates only from actual cached documents for vendors V-001 through V-006. Preserve their true `doc_id` values.
2. Manually review each sentence. Use `TECHNICAL` only for concrete, current AI use in the vendor's own product/service/operations; use `VAGUE` for generic marketing or future-oriented claims; use `EXCLUDE` for irrelevant, industry-level, or duplicate content.
3. Put reviewed labels in `data/labels.csv`. Do not use `V-000`, synthetic examples, or unreviewed labels.
4. Run `python -m m4_intel.train_specificity`. Exact and near-duplicate boilerplate is removed before folds. Each test fold holds out one vendor. The TF-IDF vocabulary and classifier are fitted only on the other five vendors.
5. Review `metrics.json` for sample counts, missing classes, fold metrics and limitations. No performance claim is valid when the status is `awaiting_manual_labels`.

## Current limitation

This workspace contains no cached vendor-document snapshots, so the first step is to collect/cache real source documents through the project's evidence-collection phase, then manually label candidate sentences. The current CSV is deliberately header-only. A Transformer is not justified until a meaningful reviewed dataset exists; Colab GPU is not needed for the current rule and TF-IDF baselines.
