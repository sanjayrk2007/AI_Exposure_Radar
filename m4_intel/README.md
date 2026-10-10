# M4 intelligence module

## Candidate extraction (Phase 1)

`extract_candidates(document_text, lexicon=None)` returns a list of dictionaries:

- `text`: original sentence or line/table row, with surrounding whitespace trimmed
- `start`: inclusive character offset into the original `Document.text`
- `end`: exclusive character offset into the original `Document.text`
- `categories`: configured lexicon categories matched in the candidate

The required invariant is `document_text[start:end] == text`. Sentence/row spans are not rewritten. Terminology matches are case-insensitive. The default lexicon is `config/lexicon.yaml`; callers may pass a mapping for testing or custom configuration. `extract_document_candidates` accepts a mapping-shaped Document contract and reads its `text` field.

Candidate extraction is a recall-oriented filtering step only. It does not verify source authenticity, infer actual AI use, score specificity, label evidence, or assign a verdict. Tests use mock text only; it is not vendor evidence.
