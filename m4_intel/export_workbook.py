"""Create a separate Excel copy with provisional AI Exposure Radar results."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from m4_intel.run_pipeline import run_pipeline
from radar.input_inspection import read_vendor_profiles


SUMMARY_HEADERS = [
    "Vendor ID", "Vendor Name", "Domain", "Documents Collected",
    "Quote-Verified Candidates", "AI Usage Verdict", "Review Status", "Notes",
]


def export_workbook(
    workbook_path: str | Path,
    manifest_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(workbook_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Output workbook must be different from the original workbook.")
    if not source.exists():
        raise FileNotFoundError(f"Workbook not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output workbook already exists: {destination}. Choose another path or pass --overwrite."
        )

    # Read the profiles through the established read-only input contract.
    profiles = read_vendor_profiles(source)
    if any(profile.get("vendor_id") == "V-000" for profile in profiles):
        raise ValueError("Input reader unexpectedly returned V-000; refusing to continue.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and overwrite:
        destination.unlink()
    shutil.copy2(source, destination)

    # Pipeline output is temporary/intermediate and kept beside the Excel output.
    intermediate_dir = destination.parent / f"{destination.stem}_pipeline"
    summary = run_pipeline(
        manifest_path, candidates_path, intermediate_dir, overwrite=overwrite
    )
    summary_by_vendor = {row["vendor_id"]: row for row in summary["vendor_summary"]}

    workbook = load_workbook(destination)
    sheet_name = "AI Radar Summary"
    if sheet_name in workbook.sheetnames:
        if not overwrite:
            workbook.close()
            raise FileExistsError(
                f"Sheet {sheet_name!r} already exists in the source copy. Use a fresh output path."
            )
        del workbook[sheet_name]
    sheet = workbook.create_sheet(sheet_name)
    sheet.append(SUMMARY_HEADERS)

    docs_by_vendor: dict[str, int] = {}
    for doc in _read_jsonl(manifest_path):
        vendor_id = doc.get("vendor_id")
        if vendor_id and vendor_id != "V-000":
            docs_by_vendor[vendor_id] = docs_by_vendor.get(vendor_id, 0) + 1

    for profile in profiles:
        vendor_id = profile.get("Vendor ID", "")
        item = summary_by_vendor.get(vendor_id, {})
        sheet.append([
            vendor_id,
            profile.get("Vendor Name", ""),
            profile.get("Public Website", ""),
            docs_by_vendor.get(vendor_id, 0),
            item.get("quote_verified_candidates", 0),
            "Inconclusive",
            "Pending factual review",
            "Quote matching is not factual verification; missing/inaccessible pages are not evidence of no AI.",
        ])

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    widths = [14, 36, 34, 20, 25, 20, 24, 90]
    for column_index, width in enumerate(widths, 1):
        sheet.column_dimensions[chr(64 + column_index)].width = width
    workbook.save(destination)
    workbook.close()

    return {
        "status": "PROVISIONAL_ONLY",
        "output_workbook": str(destination),
        "vendors_exported": len(profiles),
        "original_workbook_modified": False,
        "summary_sheet": sheet_name,
        "pipeline_output_dir": str(intermediate_dir),
        "note": "All verdicts remain Inconclusive pending factual review.",
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", default="outputs/AI_Exposure_Radar_Provisional.xlsx")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = export_workbook(
        args.workbook, args.manifest, args.candidates, args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
