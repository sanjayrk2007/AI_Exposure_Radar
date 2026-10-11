"""Generate clearly synthetic examples for controlled M4 pipeline experiments.

These sentences are invented for software testing. They are not vendor evidence,
not sourced from websites, and must never be merged into data/labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "synthetic_labels.csv"
SEED = 42
FIELDS = ("synthetic_id", "sentence", "label", "scenario", "template_family", "provenance")

PROVIDERS = ["an OpenAI model", "a Hugging Face transformer", "a locally hosted language model",
             "a scikit-learn classifier", "a computer-vision model", "a speech-recognition model"]
TASKS = ["invoice classification", "payment anomaly detection", "support-ticket summarization",
         "transaction risk scoring", "document extraction", "customer-message routing",
         "fraud detection", "demand forecasting"]
ARTIFACTS = ["through a versioned inference API", "inside a production inference pipeline",
             "using a documented model endpoint", "with model outputs logged for review",
             "through a deployed service with a defined input schema", "with batch predictions recorded"]

TECH_TEMPLATES = [
    "The service currently uses {provider} for {task} {artifact}.",
    "In production, {provider} classifies {task} {artifact}.",
    "The deployed workflow runs {provider} to perform {task} {artifact}.",
    "Our current system integrates {provider} for {task} {artifact}.",
    "The application sends incoming records to {provider} for {task} {artifact}.",
    "A live service uses {provider} to automate {task} {artifact}.",
    "Production requests are passed to {provider}, which performs {task}; outputs are retained for audit.",
    "The current release invokes {provider} to process {task} through a documented service endpoint.",
    "Operational logs record {provider} predictions for {task} from the deployed workflow.",
    "The application routes eligible records to {provider} and stores the resulting {task} decisions.",
    "A running inference component applies {provider} to {task} during normal processing.",
    "The production runbook documents how {provider} performs {task} and how failures are retried.",
]

VAGUE_TEMPLATES = [
    "Our platform is designed to deliver smarter experiences through AI innovation.",
    "We are exploring how AI could transform workflows in the future.",
    "AI-powered capabilities may help customers unlock new possibilities.",
    "Our roadmap includes plans to investigate generative AI opportunities.",
    "We believe intelligent technology can support better business outcomes.",
    "The company is considering potential AI features for a future release.",
    "Our vision is to bring next-generation intelligence to customer experiences.",
    "AI may eventually enhance the platform, subject to further evaluation.",
    "We aim to explore responsible AI as part of our long-term innovation strategy.",
    "The product offers a modern, smart, and innovative digital experience.",
    "Future releases could include AI-enabled functionality if appropriate.",
    "We are committed to evaluating emerging technologies for future use cases.",
    "We are evaluating whether {provider} might support {task} in a future release; no production deployment is described.",
    "The roadmap mentions a possible {provider} integration for {task}, pending feasibility review.",
    "Customers have asked whether {provider} could assist with {task}; the team has not committed to implementation.",
    "The company is researching potential uses of {provider} for {task}, with no current operational use announced.",
    "Marketing materials describe AI as an opportunity, but do not identify a deployed model or workflow.",
    "A future pilot may consider {provider} for {task}, subject to approval and testing.",
]

EXCLUDE_TEMPLATES = [
    "The article discusses how AI adoption is changing the financial-services industry.",
    "A third-party supplier advertises an AI assistant, but this sentence makes no claim about our system.",
    "The conference agenda includes a panel about machine learning research.",
    "The report summarizes general AI market trends and analyst forecasts.",
    "The job applicant lists Python and machine learning among personal interests.",
    "The company commented on a public debate about AI regulation without describing its own use.",
    "A third-party supplier advertises {provider} for {task}; this sentence does not establish our own implementation.",
    "The analyst report says companies may use {provider} for {task}, without making a claim about this vendor.",
]


def build_rows(seed: int = SEED, technical_count: int = 48, vague_count: int = 48,
               exclude_count: int = 24) -> list[dict[str, str]]:
    """Build deterministic, balanced binary examples plus separate EXCLUDE examples."""
    if min(technical_count, vague_count, exclude_count) < 0:
        raise ValueError("example counts must be non-negative")
    if technical_count == 0 or vague_count == 0:
        raise ValueError("technical_count and vague_count must both be positive")
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    vague_contexts = ["For enterprise customers, ", "In a product overview, ",
                      "As a strategic priority, ", "In its public messaging, ",
                      "At this stage, ", "As part of its long-term vision, "]
    exclude_contexts = ["For context, ", "Separately, ", "In a general industry discussion, ",
                        "In an unrelated announcement, ", "According to the market report, ",
                        "In a conference summary, "]

    def add(count: int, label: str, scenario: str, templates: list[str]) -> None:
        attempts = 0
        while sum(1 for row in rows if row["label"] == label) < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"Could not generate {count} unique {label} examples")
            template_index = rng.randrange(len(templates))
            template = templates[template_index]
            sentence = template.format(provider=rng.choice(PROVIDERS), task=rng.choice(TASKS),
                                       artifact=rng.choice(ARTIFACTS))
            if label == "VAGUE":
                sentence = rng.choice(vague_contexts) + sentence[0].lower() + sentence[1:]
            elif label == "EXCLUDE":
                sentence = rng.choice(exclude_contexts) + sentence[0].lower() + sentence[1:]
            identity = (label, sentence.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            # Group by source template, not by generated wording, so grouped CV can
            # hold out an entire family and expose template memorization.
            family = f"{label.lower()}_template_{template_index + 1:02d}"
            rows.append({"synthetic_id": f"SYN-{len(rows)+1:04d}", "sentence": sentence,
                         "label": label, "scenario": scenario, "template_family": family,
                         "provenance": "SYNTHETIC_GENERATED_NOT_VENDOR_EVIDENCE"})

    add(technical_count, "TECHNICAL", "current_specific_implementation", TECH_TEMPLATES)
    add(vague_count, "VAGUE", "generic_or_future_marketing", VAGUE_TEMPLATES)
    add(exclude_count, "EXCLUDE", "irrelevant_or_third_party_reference", EXCLUDE_TEMPLATES)
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["synthetic_id"] = f"SYN-{index:04d}"
    return rows


def write_dataset(path: str | Path = DEFAULT_OUTPUT, seed: int = SEED,
                  technical_count: int = 48, vague_count: int = 48,
                  exclude_count: int = 24) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(seed, technical_count, vague_count, exclude_count)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--technical", type=int, default=48)
    parser.add_argument("--vague", type=int, default=48)
    parser.add_argument("--exclude", type=int, default=24)
    args = parser.parse_args(argv)
    path = write_dataset(args.output, args.seed, args.technical, args.vague, args.exclude)
    print(f"Wrote {path} (seed={args.seed}; TECHNICAL={args.technical}; VAGUE={args.vague}; EXCLUDE={args.exclude})")
    print("WARNING: synthetic examples are not real vendor evidence and must not be merged into data/labels.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
