"""Write M4 results to a new workbook; the input workbook is never modified.

Only columns O, P, Q and R of the six real vendor rows are filled. V-000 and every
provided column (B to K) are left untouched, and that is verified after saving.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

from radar.contracts.models import Signal, SourceReport, VendorVerdict

SHEET_NAME = "Vendor Inventory"
HEADER_ROW = 4
EXAMPLE_VENDOR_ID = "V-000"
FIRST_PROVIDED_COLUMN, LAST_PROVIDED_COLUMN = 2, 11
FILLED_COLUMNS = {
    15: "AI Usage Detected",
    16: "Evidence Description",
    17: "How Vendor May Be Using AI",
    18: "Named AI Sub-processors",
}
MAX_EVIDENCE_EXCERPTS = 3
PROPOSAL_STATUS = {
    "LLM": "LLM_PROPOSED_UNREVIEWED",
    "ML": "ML_PROPOSED_UNREVIEWED",
    "RULE": "RULE_PROPOSED_UNREVIEWED",
}


def review_status(signal: Signal, llm_proposed: Mapping[str, Any]) -> str:
    """Status derived from who proposed the signal; code never assigns a human-reviewed status."""
    if signal.method == "LLM" or signal.signal_id in llm_proposed:
        return PROPOSAL_STATUS["LLM"]
    return PROPOSAL_STATUS.get(signal.method, PROPOSAL_STATUS["RULE"])


@dataclass
class VendorResult:
    vendor_id: str
    verdict: VendorVerdict
    signals: list[Signal] = field(default_factory=list)
    reports: list[SourceReport] = field(default_factory=list)
    documents: list[Mapping[str, Any]] = field(default_factory=list)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    llm_proposed: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str = ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieved(documents: Sequence[Mapping[str, Any]], doc_id: str) -> str:
    for document in documents:
        if document.get("doc_id") == doc_id:
            return str(document.get("fetched_at") or "")
    return ""


def _document(documents: Sequence[Mapping[str, Any]], doc_id: str) -> Mapping[str, Any]:
    for document in documents:
        if document.get("doc_id") == doc_id:
            return document
    return {}


def _gap_note(result: VendorResult) -> str:
    grouped: dict[tuple[str, str], list[str]] = {}
    for report in result.reports:
        if report.status != "OK":
            grouped.setdefault((report.status, report.note[:160]), []).append(report.source_type)
    return "; ".join(
        f"{', '.join(types)}: {status}" + (f" ({note})" if note else "")
        for (status, note), types in grouped.items()
    )


def evidence_text(result: VendorResult) -> str:
    verified = [s for s in result.signals if s.verified and s.tag != "NONE"]
    verified.sort(key=lambda s: (-s.score, s.signal_id))
    if not verified:
        if result.error:
            return f"No verified AI-related excerpt collected. Processing failed: {result.error}."
        reason = _gap_note(result) or "no candidate sentence matched"
        return f"No verified AI-related excerpt collected. Collection status: {reason}."
    lines = []
    for signal in verified[:MAX_EVIDENCE_EXCERPTS]:
        retrieved = _retrieved(result.documents, signal.doc_id)[:10]
        lines.append(f"\"{signal.excerpt}\" ({signal.source_type}, {signal.url}, retrieved {retrieved})")
    return "\n".join(lines)


def usage_text(result: VendorResult) -> str:
    cases = result.verdict.use_cases
    if result.verdict.ai_usage == "No":
        return "No AI use indicated in the sources collected; coverage-qualified and not proof of absence."
    if not cases:
        return "Not determined from the public sources collected."
    return ("May be using AI for: " + ", ".join(cases)
            + ". Proposed from public text; not reviewed by an analyst.")


_SUBPROCESSOR_CONTEXT = re.compile(
    r"\b(?:sub-?processor|sub-?contractor|third-?party processor|vendor processor|service provider)\b",
    re.IGNORECASE,
)


def extract_explicit_subprocessors(result: VendorResult) -> list[str]:
    """Extract providers only when verified evidence explicitly asserts a subprocessor relationship."""
    found: list[str] = []
    for signal in result.signals:
        if signal.verified and signal.providers and _SUBPROCESSOR_CONTEXT.search(signal.excerpt):
            for provider in signal.providers:
                if provider not in found:
                    found.append(provider)
    return found


def providers_text(result: VendorResult) -> str:
    """Populate the subprocessor column conservatively, not with model/product names.

    Model names and general provider mentions are preserved in the Evidence Log and
    verdict providers list. The subprocessor field is only populated when explicit
    subprocessor or service-provider context is present in verified evidence.
    """
    explicit = extract_explicit_subprocessors(result)
    if explicit:
        return f"Explicitly verified AI subprocessor(s): {', '.join(explicit)}."
    reason = result.error or _gap_note(result)
    suffix = f" Source gaps: {reason}." if reason else ""
    return "No named AI subprocessor explicitly verified in the collected evidence." + suffix


def _verify_unchanged(input_path: Path, output_path: Path) -> None:
    source = load_workbook(input_path, data_only=False)
    produced = load_workbook(output_path, data_only=False)
    try:
        src, out = source[SHEET_NAME], produced[SHEET_NAME]
        for row in range(1, src.max_row + 1):
            vendor_id = src.cell(row=row, column=2).value
            columns = range(1, src.max_column + 1)
            if row > HEADER_ROW and vendor_id != EXAMPLE_VENDOR_ID:
                columns = [c for c in columns if c not in FILLED_COLUMNS]
            for column in columns:
                if src.cell(row=row, column=column).value != out.cell(row=row, column=column).value:
                    raise RuntimeError(
                        f"Provided cell {src.cell(row=row, column=column).coordinate} changed during export")
    finally:
        source.close()
        produced.close()


def export_workbook(input_path: str | Path, output_path: str | Path,
                    results: Mapping[str, VendorResult], notes: Sequence[str] = ()) -> Path:
    source, target = Path(input_path).resolve(), Path(output_path).resolve()
    if source == target or (target.exists() and source.samefile(target)):
        raise ValueError("Output path must differ from the input workbook; the original is never overwritten")
    before = _sha256(source)

    workbook = load_workbook(source)
    sheet = workbook[SHEET_NAME]
    wrap = Alignment(wrap_text=True, vertical="top")
    filled: list[str] = []
    for row in range(HEADER_ROW + 1, sheet.max_row + 1):
        vendor_id = sheet.cell(row=row, column=2).value
        if not isinstance(vendor_id, str) or vendor_id == EXAMPLE_VENDOR_ID:
            continue
        result = results.get(vendor_id)
        if result is None:
            continue
        values = {
            15: result.verdict.ai_usage,
            16: evidence_text(result),
            17: usage_text(result),
            18: providers_text(result),
        }
        for column, value in values.items():
            cell = sheet.cell(row=row, column=column, value=value)
            cell.alignment = wrap
        filled.append(vendor_id)

    log = workbook.create_sheet("Evidence Log")
    log.append(["vendor_id", "signal_id", "doc_id", "source_type", "url", "retrieved_at", "content_hash",
                "char_start", "char_end", "excerpt", "tag", "specificity", "source_weight", "score",
                "providers", "proposed_by", "llm_reason_code", "llm_quote", "ambiguity_reasons",
                "review_status", "human_decision"])
    for vendor_id in sorted(results):
        result = results[vendor_id]
        for signal in result.signals:
            proposal = result.llm_proposed.get(signal.signal_id, {})
            document = _document(result.documents, signal.doc_id)
            log.append([
                vendor_id, signal.signal_id, signal.doc_id, signal.source_type, signal.url,
                _retrieved(result.documents, signal.doc_id), document.get("content_hash", ""),
                signal.char_start, signal.char_end, signal.excerpt, signal.tag,
                round(signal.specificity, 4), signal.weight, round(signal.score, 4),
                ", ".join(signal.providers), signal.method, proposal.get("reason_code", ""),
                proposal.get("quote", ""), "; ".join(result.ambiguous.get(signal.signal_id, [])),
                review_status(signal, result.llm_proposed), "",
            ])

    coverage = workbook.create_sheet("Coverage")
    coverage.append(["vendor_id", "ai_usage", "strength", "coverage", "source_type", "status",
                     "pages_fetched", "note", "verdict_reasons"])
    for vendor_id in sorted(results):
        result = results[vendor_id]
        reasons = " | ".join(result.verdict.reasons + ([f"Run error: {result.error}"] if result.error else []))
        rows = result.reports or [SourceReport(vendor_id, "", "", 0, "")]
        for index, report in enumerate(rows):
            coverage.append([
                vendor_id, result.verdict.ai_usage, result.verdict.strength,
                round(result.verdict.coverage, 4), report.source_type, report.status,
                report.pages_fetched, report.note, reasons if index == 0 else "",
            ])

    about = workbook.create_sheet("Run Notes")
    about.append(["Notes"])
    about["A1"].font = Font(bold=True)
    about.append(["Columns O to R are machine-proposed from public pages and are not analyst-reviewed."])
    about.append(["Every excerpt was verified as an exact substring of the collected page text."])
    about.append(["A page statement is a vendor claim, not independent proof of how or whether a capability is deployed."])
    about.append(["Missing, blocked or failed sources lower coverage and never indicate that a vendor does not use AI."])
    about.append(["Columns L to N and S to V were not assessed by this module."])
    for note in notes:
        about.append([note])

    target.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(target)
    workbook.close()
    if _sha256(source) != before:
        raise RuntimeError("Input workbook changed during export")
    _verify_unchanged(source, target)
    return target
