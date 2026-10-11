"""Build a contextual factual-review sheet from provisional candidate signals."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def build_review_sheet(manifest_path, signals_path, output_path, overwrite=False):
    out=Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; choose a new path or pass --overwrite")
    docs={}
    for d in read_jsonl(manifest_path):
        if d.get("vendor_id") != "V-000":
            docs[(d.get("vendor_id"),d.get("doc_id"))]=d
    rows=[]
    with Path(signals_path).open(newline="",encoding="utf-8-sig") as f:
        for s in csv.DictReader(f):
            if s.get("vendor_id") == "V-000": continue
            d=docs.get((s.get("vendor_id"),s.get("doc_id")), {})
            text=d.get("text","")
            try:
                start=int(s.get("char_start",""))
                end=int(s.get("char_end",""))
            except (TypeError,ValueError):
                start=end=0
            # Prefer the exact stored excerpt but show source context from the manifest.
            excerpt=s.get("excerpt","")
            if text and 0 <= start <= end <= len(text):
                before=text[max(0,start-350):start]
                after=text[end:min(len(text),end+350)]
                exact=(text[start:end] == excerpt)
            else:
                before=after=""
                exact=False
            rows.append({
                "vendor_id":s.get("vendor_id",""), "doc_id":s.get("doc_id",""),
                "signal_id":s.get("signal_id",""), "source_type":s.get("source_type",""),
                "url":s.get("url",""), "title":s.get("title",""),
                "context_before":before, "excerpt":excerpt, "context_after":after,
                "quote_integrity": "PASS" if exact else "FAIL",
                "provisional_score":s.get("score",""), "provisional_tag":s.get("tag",""),
                "factual_review_decision":"PENDING",
                "review_reason":""
            })
    fields=["vendor_id","doc_id","signal_id","source_type","url","title","context_before","excerpt","context_after","quote_integrity","provisional_score","provisional_tag","factual_review_decision","review_reason"]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return {"review_rows":len(rows),"quote_integrity_pass":sum(r["quote_integrity"]=="PASS" for r in rows),"output":str(out)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--signals",required=True)
    p.add_argument("--output",default="data/evidence_review.csv")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    print(json.dumps(build_review_sheet(a.manifest,a.signals,a.output,a.overwrite),indent=2))
if __name__=="__main__": main()
