import csv
import json
import re
from pathlib import Path

from m4_intel.candidates import load_lexicon, extract_document_candidates

MANIFEST = Path("cache_pilot_v2/manifest.jsonl")
EXISTING = Path("data/candidates_pilot_v2.jsonl")
OUTPUT_JSONL = Path("data/candidates_expansion_v1.jsonl")
OUTPUT_CSV = Path("data/candidate_review_expansion_v1.csv")

# Additional discovery terms for a second pass.
EXTRA_TERMS = {
    "techniques": [
        "machine learning", "machine-learning", "ML model",
        "natural language processing", "NLP", "computer vision",
        "large language model", "LLM", "LLMs", "neural network",
        "predictive model", "classification model", "AI-powered",
        "AI-enabled", "AI-driven", "generative AI",
        "conversational AI", "intelligent automation",
    ],
    "use_cases": [
        "chatbot", "virtual assistant", "copilot",
        "fraud detection", "predictive analytics",
        "document intelligence", "automated decision",
        "personalized recommendation", "text generation",
    ],
}

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def main():
    for path in (MANIFEST, EXISTING):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    lexicon = load_lexicon()
    lexicon = {key: list(value) for key, value in lexicon.items()}
    for category, terms in EXTRA_TERMS.items():
        lexicon.setdefault(category, [])
        known = {normalize(term) for term in lexicon[category]}
        for term in terms:
            if normalize(term) not in known:
                lexicon[category].append(term)
                known.add(normalize(term))

    with EXISTING.open(encoding="utf-8") as f:
        old_candidates = [json.loads(line) for line in f if line.strip()]

    # Exclude any sentence already present in the original candidate pool.
    existing_texts = {normalize(row["text"]) for row in old_candidates}

    with MANIFEST.open(encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]

    new_rows = []
    seen_new = set()

    for document in documents:
        for candidate in extract_document_candidates(document, lexicon):
            row = dict(candidate)
            text = normalize(row["text"])
            if not text or text in existing_texts or text in seen_new:
                continue
            seen_new.add(text)
            new_rows.append(row)

    fields = [
        "vendor_id", "doc_id", "source_type", "url", "title",
        "text", "char_start", "char_end", "categories",
    ]

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields + ["label", "notes"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in new_rows:
            writer.writerow({
                **row,
                "categories": "; ".join(row.get("categories", []))
                    if isinstance(row.get("categories"), list)
                    else row.get("categories", ""),
                "label": "",
                "notes": "",
            })

    print("Documents scanned:", len(documents))
    print("Original candidates:", len(old_candidates))
    print("New unique candidates:", len(new_rows))
    print("JSONL:", OUTPUT_JSONL)
    print("Review CSV:", OUTPUT_CSV)
    print("Labels were not changed.")

if __name__ == "__main__":
    main()
