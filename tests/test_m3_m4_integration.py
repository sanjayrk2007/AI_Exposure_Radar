from __future__ import annotations

import json
from pathlib import Path
import pytest

from radar.contracts.models import (
    VendorProfile, DepthPlan, CriticalityResult, RiskAssessment,
    Document, Signal, SourceReport, VendorVerdict
)
from radar.m3_sources.manual_inbox import add_manual_evidence
from radar.m3_sources.store import export_cache_to_manifest, read_all_documents
from m4_intel.candidates import extract_candidates
from m4_intel.verify import verify_candidate
from m4_intel.specificity import score_signal, get_source_weight
from m4_intel.run_pipeline import run_pipeline
from m4_intel.verdict import generate_verdict, calculate_coverage


def test_contract_models_instantiation():
    profile = VendorProfile(
        vendor_id="V-001", name="Acme Corp", description="Payments", service="Processing",
        category="FinTech", website="https://acme.test", process="Core",
        dependency="High", data_class="PII", data_volume="High"
    )
    assert profile.vendor_id == "V-001"

    crit = CriticalityResult(vendor_id="V-001", tier="High", scores={"data": 3}, rationale="Test")
    assert crit.tier == "High"

    plan = DepthPlan(vendor_id="V-001", tier="High", sources=["WEB", "CAREERS"], max_pages=4, max_depth=2, stop_rule="stop")
    assert "CAREERS" in plan.sources

    risk = RiskAssessment(vendor_id="V-001", risk_class="High", rationale="Risk", recommended_action="Review")
    assert risk.risk_class == "High"


def test_m3_manual_evidence_scored_by_m4(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    primary_text = "We deploy OpenAI models to detect fraud in transaction data."
    third_party_text = "Acme is reportedly using OpenAI for transaction fraud detection."

    doc_primary = add_manual_evidence(
        vendor_id="V-001",
        url="https://acme.test/blog/ai-release",
        text=primary_text,
        source_class="primary",
        captured_at="2026-03-01T10:00:00Z",
        cache_dir=cache_dir,
    )
    doc_third_party = add_manual_evidence(
        vendor_id="V-001",
        url="https://news.test/article",
        text=third_party_text,
        source_class="third_party",
        captured_at="2026-03-01T10:00:00Z",
        cache_dir=cache_dir,
    )

    assert doc_primary.meta["source_class"] == "primary"
    assert doc_third_party.meta["source_class"] == "third_party"

    # Verify candidate extraction from M3 manual docs
    cands_primary = extract_candidates(doc_primary.text)
    assert len(cands_primary) >= 1
    cand_p = cands_primary[0]

    # Verify candidate quote integrity
    sig_primary = verify_candidate(doc_primary, cand_p)
    assert sig_primary is not None
    assert sig_primary.verified is True

    # Test M4 score_signal with source_class
    score_signal(sig_primary, source_class=doc_primary.meta["source_class"])
    assert sig_primary.weight == 3.5
    assert sig_primary.score > 0
    assert "OpenAI" in sig_primary.providers

    # Test third-party scoring
    cands_third = extract_candidates(doc_third_party.text)
    sig_third = verify_candidate(doc_third_party, cands_third[0])
    assert sig_third is not None
    score_signal(sig_third, source_class=doc_third_party.meta["source_class"])
    assert sig_third.weight == 2.0


def test_m3_cache_to_m4_pipeline_run(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    text = "We use an API with OpenAI for fraud detection in customer data."
    add_manual_evidence(
        vendor_id="V-002",
        url="https://vendor2.test/announcement",
        text=text,
        source_class="primary",
        captured_at="2026-01-01T00:00:00Z",
        cache_dir=cache_dir,
    )

    # Export cache to manifest.jsonl
    manifest_path = tmp_path / "manifest.jsonl"
    count = export_cache_to_manifest(manifest_path, cache_dir=cache_dir)
    assert count == 1

    # Extract candidate and write candidates.jsonl
    docs = read_all_documents(cache_dir=cache_dir)
    cands = extract_candidates(docs[0].text)
    assert len(cands) >= 1

    candidate_row = {
        "vendor_id": "V-002",
        "doc_id": docs[0].doc_id,
        "source_type": docs[0].source_type,
        "url": docs[0].url,
        "title": docs[0].title,
        "text": cands[0]["text"],
        "char_start": cands[0]["start"],
        "char_end": cands[0]["end"],
        "categories": cands[0]["categories"],
    }
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")

    # Run M4 pipeline on the exported M3 evidence
    out_dir = tmp_path / "pipeline_out"
    result = run_pipeline(manifest_path, candidates_path, out_dir)
    assert result["quote_verified_candidates"] == 1
    assert result["rejected_missing_document"] == 0
    assert result["rejected_invalid_quote_or_offsets"] == 0

    # Ensure output CSV got generated with verified signals
    signals_csv = out_dir / "candidate_signals.csv"
    assert signals_csv.exists()
    csv_content = signals_csv.read_text(encoding="utf-8-sig")
    assert "V-002" in csv_content
    assert "MANUAL" in csv_content
