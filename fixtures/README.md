# Fixture format (Phase 0)

Fixtures are JSON files used to test module contracts without live collection. Do not place private vendor data, real collected pages, screenshots, API keys, or model artifacts in a public repository. This project is intended to remain private for the Optiv case study.

## `documents.json`

Top-level JSON array. Each document must contain:

- `doc_id` (string): stable document identifier.
- `vendor_id` (string): vendor ID from the input workbook; never `V-000`.
- `source_type` (string): one of `TRUST`, `DOCS`, `WEB`, `FINGERPRINT`, `CAREERS`, `REPOS`, `FILINGS`, `MANUAL`.
- `url` (string): canonical source URL (manual evidence may retain the supplied URL).
- `fetched_at` (string): ISO 8601 UTC timestamp, e.g. `2026-10-10T12:30:00Z`; this is the retrieved timestamp.
- `content_hash` (string): SHA-256 hex digest of the stored document text or defined canonical content.
- `text` (string): extracted document text used by M4.

Recommended additional fields, matching the shared plan contract: `http_status` (integer or null), `title` (string), `snapshot_path` (string or null), and `meta` (object). Keep raw snapshots in private cache storage and do not commit them.

## `source_reports.json`

Top-level JSON array. Each report must contain:

- `vendor_id` (string): vendor ID from the workbook; never `V-000`.
- `source_type` (string): one of the allowed source types above.
- `status` (string): one of `OK`, `EMPTY`, `BLOCKED`, `ERROR`, `NOT_APPLICABLE`, `SKIPPED_BY_PLAN`.
- `pages_fetched` (integer, zero or greater).
- `note` (string): concise explanation of the result or limitation.

## Integrity rules

- Do not fabricate document text, URLs, hashes, timestamps, source reports, or labels.
- Use an empty array (`[]`) when no genuine fixture records are available yet.
- Exact excerpts emitted by M4 must be substrings of the corresponding `text`, verified using character offsets.
- Do not process or modify the workbook's `V-000` worked-example row.
