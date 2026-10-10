import csv
import json
from pathlib import Path

CANDIDATES = Path("data/candidates_pilot_v2.jsonl")
LABELS = Path("data/labels.csv")
OUTPUT = Path("data/candidate_review_queue_v1.csv")


def normalize(text):
    return " ".join(text.split()).casefold()


def main():
    with CANDIDATES.open(encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    with LABELS.open(newline="", encoding="utf-8-sig") as f:
        labels = list(csv.DictReader(f))

    labeled_keys = {
        (row["vendor_id"], row["doc_id"], normalize(row["sentence"]))
        for row in labels
    }

    queue = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate["vendor_id"],
            candidate["doc_id"],
            normalize(candidate["text"]),
        )

        if key in labeled_keys or key in seen:
            continue

        seen.add(key)
        queue.append({
            "vendor_id": candidate["vendor_id"],
            "doc_id": candidate["doc_id"],
            "source_type": candidate.get("source_type", ""),
            "url": candidate.get("url", ""),
            "sentence": candidate["text"],
            "label": "",
            "notes": "",
        })

    fields = [
        "vendor_id", "doc_id", "source_type",
        "url", "sentence", "label", "notes",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)

    print("Candidates:", len(candidates))
    print("Existing labels:", len(labels))
    print("New review rows:", len(queue))
    print("Output:", OUTPUT)

    from collections import Counter
    print("Review rows by vendor:", dict(Counter(
        row["vendor_id"] for row in queue
    )))


if __name__ == "__main__":
    main()