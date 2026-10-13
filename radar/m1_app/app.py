"""M1 Streamlit interface for AI Exposure Radar."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from openpyxl import load_workbook

from radar.contracts.models import RiskAssessment, SourceReport, VendorVerdict
from radar.export import VendorResult, export_workbook
from radar.m1_app.excel_io import EXAMPLE_VENDOR_ID, read_vendor_profiles
from radar.m1_app.pipeline import run_vendor

st.set_page_config(page_title="AI Exposure Radar", layout="wide")
st.title("AI Exposure Radar")
st.caption("Vendor AI exposure screening — machine-proposed findings require analyst review.")

uploaded = st.file_uploader("Upload vendor inventory (.xlsx)", type=["xlsx"])
mode = st.radio("Collection mode", ["cache", "live"], horizontal=True, index=0)
use_model = st.checkbox("Use M4 model where available", value=False)

st.info(
    "The current exporter fills columns O–R and adds Evidence Log, Coverage, "
    "and Run Notes. Columns L–N and S–V are not assessed by this version."
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
                    )

                    display_rows.append({
                        "Vendor ID": vendor.vendor_id,
                        "Vendor": vendor.name,
                        "Criticality": result.criticality.tier,
                        "AI Usage": result.verdict.ai_usage,
                        "Strength": result.verdict.strength,
                        "Coverage": round(result.verdict.coverage * 100, 1),
                        "Risk": result.risk.risk_class,
                        "Documents": len(result.documents),
                        "Signals": len(result.signals),
                        "Status": "Completed",
                    })

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
                    display_rows.append({
                        "Vendor ID": vendor.vendor_id,
                        "Vendor": vendor.name,
                        "Criticality": "Not assessed",
                        "AI Usage": "Inconclusive",
                        "Strength": "None",
                        "Coverage": 0.0,
                        "Risk": "Not assessed",
                        "Documents": 0,
                        "Signals": 0,
                        "Status": f"Error: {message}",
                    })

                progress.progress((index + 1) / len(profiles))

            try:
                export_workbook(
                    input_path,
                    output_path,
                    export_results,
                    notes=[
                        f"Collection mode: {mode}",
                        "M1 orchestration run; results are machine-proposed and not analyst-reviewed.",
                        "L–N and S–V remain unassessed by the current exporter.",
                    ],
                )
                st.session_state["m1_results"] = display_rows
                st.session_state["m1_output"] = output_path.read_bytes()
                st.success("Run finished and Excel export created.")
            except Exception as exc:
                st.error(f"Excel export failed: {exc}")

            status.empty()

if "m1_results" in st.session_state:
    st.subheader("Run results")
    st.dataframe(st.session_state["m1_results"], use_container_width=True, hide_index=True)

    st.download_button(
        "Download results workbook",
        data=st.session_state["m1_output"],
        file_name="AI_Exposure_Radar_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
