"""Generate vendor verdicts from only explicitly approved evidence-review rows."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import Counter
from m4_intel.verdict import generate_verdict

PLANNED_SOURCE_TYPES=["TRUST","DOCS","FILINGS","FINGERPRINT","CAREERS","REPOS","WEB"]

def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def run_reviewed_verdicts(manifest_path, review_path, output_path, weights_path=None):
    docs=read_jsonl(manifest_path)
    vendors=[f"V-{i:03d}" for i in range(1,7)]
    approved=[]; pending=Counter(); rejected=Counter()
    with Path(review_path).open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vendor=row.get("vendor_id","")
            if vendor not in vendors: continue
            decision=row.get("factual_review_decision","PENDING").strip().upper()
            if decision not in {"APPROVED","REJECTED","PENDING"}:
                raise ValueError(f"Invalid factual_review_decision {decision!r}; use APPROVED, REJECTED, or PENDING")
            if row.get("quote_integrity") != "PASS":
                if decision=="APPROVED":
                    raise ValueError(f"Cannot approve quote-integrity failure: {row.get('signal_id')}")
                continue
            if decision=="PENDING": pending[vendor]+=1; continue
            if decision=="REJECTED": rejected[vendor]+=1; continue
            def number(k):
                try: return float(row.get(k) or 0)
                except ValueError: return 0.0
            # This 'verified' flag is only reached after both quote integrity and
            # an explicit human APPROVED decision in the review sheet.
            approved.append({
                "signal_id":row.get("signal_id"),"vendor_id":vendor,
                "doc_id":row.get("doc_id"),"source_type":row.get("source_type"),
                "url":row.get("url"),"excerpt":row.get("excerpt"),
                "char_start":0,"char_end":len(row.get("excerpt","")),
                "category":"reviewed_evidence","providers":[],
                "specificity":number("provisional_score"),"weight":1.0,
                "score":number("provisional_score"),"tag":row.get("provisional_tag","NONE"),
                "verified":True,"method":"rule_score+human_factual_review"
            })
    result=[]
    for vendor in vendors:
        vendor_docs=[d for d in docs if d.get("vendor_id")==vendor]
        vendor_signals=[s for s in approved if s["vendor_id"]==vendor]
        verdict=generate_verdict(vendor,PLANNED_SOURCE_TYPES,vendor_docs,[],vendor_signals,**({"weights_path":weights_path} if weights_path else {}))
        result.append({
            "vendor_id":vendor,"ai_usage":verdict.ai_usage,"strength":verdict.strength,
            "coverage":round(verdict.coverage,3),"approved_signals":len(vendor_signals),
            "pending_review":pending[vendor],"rejected_signals":rejected[vendor],
            "signal_ids":verdict.signal_ids,"reasons":verdict.reasons,
            "review_status":"PENDING" if pending[vendor] else "REVIEWED_CANDIDATES_ONLY"
        })
    payload={
        "status":"PROVISIONAL_EVIDENCE_REVIEW",
        "approved_signal_count":len(approved),
        "pending_signal_count":sum(pending.values()),
        "rejected_signal_count":sum(rejected.values()),
        "verdicts":result,
        "limitations":[
            "Only explicitly APPROVED rows are used as signals.",
            "This pilot has only WEB documents; source coverage is low and verdicts are expected to remain Inconclusive.",
            "No evidence from unavailable source families is assumed absent.",
            "Verdicts depend on analyst factual-review decisions and draft weights/thresholds."
        ]
    }
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return payload

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--review",default="data/evidence_review.csv")
    p.add_argument("--output",default="outputs/pilot_v2/vendor_verdicts.json")
    p.add_argument("--weights")
    a=p.parse_args()
    r=run_reviewed_verdicts(a.manifest,a.review,a.output,a.weights)
    print(json.dumps({"status":r["status"],"approved_signal_count":r["approved_signal_count"],"pending_signal_count":r["pending_signal_count"],"output":a.output,"verdicts":[{"vendor_id":v["vendor_id"],"ai_usage":v["ai_usage"],"coverage":v["coverage"]} for v in r["verdicts"]]},indent=2))
if __name__=="__main__":main()
