import csv,json
from pathlib import Path
import pytest
from m4_intel.build_evidence_review import build_review_sheet
from m4_intel.run_reviewed_verdicts import run_reviewed_verdicts

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")

def test_review_sheet_includes_context_and_pending_default(tmp_path):
    text="Prefix context. Vendor uses AI for document classification. Suffix context."
    start=text.index("Vendor")
    m=tmp_path/"manifest.jsonl"; s=tmp_path/"signals.csv"; out=tmp_path/"review.csv"
    write_jsonl(m,[{"vendor_id":"V-001","doc_id":"D1","text":text}])
    with s.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["vendor_id","doc_id","signal_id","source_type","url","title","excerpt","char_start","char_end","score","tag"])
        w.writeheader();w.writerow({"vendor_id":"V-001","doc_id":"D1","signal_id":"S1","source_type":"WEB","url":"https://x.test","title":"T","excerpt":text[start:start+len("Vendor uses AI for document classification.")],"char_start":start,"char_end":start+len("Vendor uses AI for document classification."),"score":1.5,"tag":"MARKETING_ONLY"})
    result=build_review_sheet(m,s,out)
    assert result["quote_integrity_pass"]==1
    row=next(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert row["factual_review_decision"]=="PENDING"
    assert "Prefix context." in row["context_before"]

def test_only_approved_rows_feed_verdict(tmp_path):
    text="A sentence about AI."
    m=tmp_path/"manifest.jsonl"; write_jsonl(m,[{"vendor_id":"V-001","doc_id":"D1","source_type":"WEB","text":text,"url":"https://x.test"}])
    review=tmp_path/"review.csv"
    fields=["vendor_id","doc_id","signal_id","source_type","url","excerpt","quote_integrity","provisional_score","provisional_tag","factual_review_decision"]
    with review.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerow({"vendor_id":"V-001","doc_id":"D1","signal_id":"S1","source_type":"WEB","url":"https://x.test","excerpt":text,"quote_integrity":"PASS","provisional_score":"1.5","provisional_tag":"MARKETING_ONLY","factual_review_decision":"PENDING"})
    result=run_reviewed_verdicts(m,review,tmp_path/"out.json")
    assert result["approved_signal_count"]==0
    assert result["pending_signal_count"]==1
    assert result["verdicts"][0]["ai_usage"]=="Inconclusive"
