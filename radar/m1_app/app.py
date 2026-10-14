"""M1 Streamlit interface for AI Exposure Radar."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from openpyxl import load_workbook

from radar.contracts.models import SourceReport, VendorVerdict
from radar.export import VendorResult, export_workbook
from radar.m1_app.dashboard import record_from_error, record_from_result, render_dashboard
from radar.m1_app.excel_io import EXAMPLE_VENDOR_ID, read_vendor_profiles
from radar.m1_app.pipeline import run_vendor
from radar.m1_app.theme import inject_theme

st.set_page_config(page_title="AI Exposure Radar", layout="wide")
inject_theme()

st.markdown(
    '<div class="hero"><div class="eyebrow">Vendor risk · AI exposure screening</div>'
    '<div class="hero-title">AI Exposure Radar</div>'
    '<div class="hero-sub">Screen a vendor inventory for evidence of AI use, rate the resulting risk, '
    "and export an audit-ready workbook.</div></div>",
    unsafe_allow_html=True,
)

input_col, option_col = st.columns([3, 2], gap="large")
with input_col:
    uploaded = st.file_uploader("Vendor inventory (.xlsx)", type=["xlsx"])
with option_col:
    mode = st.radio("Collection mode", ["cache", "live"], horizontal=True, index=0)
    use_model = st.checkbox("Use M4 model where available", value=False)

st.markdown(
    '<div class="notice">Machine-proposed findings require analyst review. The exporter fills '
    "columns L–V when assessment results are available; missing M5 results stay blank. "
    "Review the Evidence Log, Coverage and Run Notes sheets before relying on conclusions.</div>",
    unsafe_allow_html=True,
)

if uploaded is not None:
    input_bytes = uploaded.getvalue()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.xlsx"
        output_path = Path(temp_dir) / "AI_Exposure_Radar_results.xlsx"
        input_path.write_bytes(input_bytes)

        try:
            profiles = read_vendor_profiles(input_path)
        except Exception as exc:
            st.error(f"Could not read workbook: {exc}")
            st.stop()

        st.write(f"**Vendors to process:** {len(profiles)} (V-000 is excluded)")

        if st.button("Run AI Exposure Radar", type="primary", disabled=not profiles):
            export_results = {}
            display_rows = []
            progress = st.progress(0)
            status = st.empty()

            for index, vendor in enumerate(profiles):
                status.write(f"Processing {vendor.vendor_id} — {vendor.name}…")

                try:
                    result = run_vendor(
                        vendor,
                        mode=mode,
                        cache_dir="radar/fixtures/m3/cache",
                        use_model=use_model,
                    )

                    export_results[vendor.vendor_id] = VendorResult(
                        vendor_id=vendor.vendor_id,
                        verdict=result.verdict,
                        signals=result.signals,
                        reports=result.reports,
                        documents=[
                            {
                                "doc_id": doc.doc_id,
                                "vendor_id": doc.vendor_id,
                                "source_type": doc.source_type,
                                "url": doc.url,
                                "fetched_at": doc.fetched_at,
                                "content_hash": doc.content_hash,
                                "text": doc.text,
                            }
                            for doc in result.documents
                        ],
                        ambiguous=result.ambiguous,
                        llm_proposed=result.llm_proposed,
                        criticality=result.criticality,
                        depth_plan=result.depth_plan,
                        risk=result.risk,
                        vendor_profile=vendor,
                    )

                    display_rows.append(record_from_result(vendor, result))

                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    export_results[vendor.vendor_id] = VendorResult(
                        vendor_id=vendor.vendor_id,
                        verdict=VendorVerdict(
                            vendor_id=vendor.vendor_id,
                            ai_usage="Inconclusive",
                            strength="None",
                            coverage=0.0,
                            signal_ids=[],
                            providers=[],
                            use_cases=[],
                            reasons=["Pipeline failed; no conclusion can be drawn."],
                        ),
                        reports=[
                            SourceReport(
                                vendor_id=vendor.vendor_id,
                                source_type="PIPELINE",
                                status="ERROR",
                                pages_fetched=0,
                                note=message[:500],
                            )
                        ],
                        error=message,
                    )
                    display_rows.append(record_from_error(vendor, message))

                progress.progress((index + 1) / len(profiles))

            try:
                export_workbook(
                    input_path,
                    output_path,
                    export_results,
                    notes=[
                        f"Collection mode: {mode}",
                        "M1 orchestration run; results are machine-proposed and not analyst-reviewed.",
                        "M5 fields are populated when assessment results are available; blank fields indicate unavailable results.",
                    ],
                )
                st.session_state["m1_results"] = display_rows
                st.session_state["m1_output"] = output_path.read_bytes()
                st.session_state["m1_export_error"] = None
                st.success("Run finished and Excel export created.")
            except Exception as exc:
                st.session_state["m1_results"] = display_rows
                st.session_state["m1_output"] = None
                st.session_state["m1_export_error"] = str(exc)
                st.error(f"Excel export failed: {exc}")

            status.empty()

if "m1_results" in st.session_state:
    title_col, download_col = st.columns([4, 1], vertical_alignment="bottom")
    title_col.markdown('<div class="section">Run results</div>', unsafe_allow_html=True)
    if st.session_state.get("m1_export_error"):
        download_col.button(
            "Download workbook",
            disabled=True,
            use_container_width=True,
            help=f"Excel export failed: {st.session_state['m1_export_error']}",
        )
    elif st.session_state.get("m1_output"):
        download_col.download_button(
            "Download workbook",
            data=st.session_state["m1_output"],
            file_name="AI_Exposure_Radar_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    render_dashboard(st.session_state["m1_results"])
