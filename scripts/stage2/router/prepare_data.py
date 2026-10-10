"""Download, inspect, cap, deduplicate, and split public router datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEED = 20260905
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / ".tools/stage2/data/router"
os.environ.setdefault("HF_HOME", str(ROOT / ".tools/stage2/hf"))

from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

TEXT_QUERIES = ("Explain this.", "Summarize this.", "What does this mean?")
CODE_QUERIES = ("Explain this code.", "Why is this failing?", "What language is this?")
MATH_QUERIES = ("Solve this.", "Explain this problem.", "What type of problem is this?")


def joined(query: str, content: str, modality: str) -> str:
    return (
        f"intent_explain modality_{modality} representation_unknown application_unknown "
        f"has_ocr_true has_table_false has_math_{modality == 'math'} "
        f"has_code_analysis_{modality == 'code'} raw_pixels_true\n"
        f"QUESTION: {query}\nCONTENT: {content[:12000]}"
    )


def stable_choice(values: tuple[str, ...], text: str) -> str:
    index = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % len(values)
    return values[index]


def text_field(field: str) -> Callable[[dict[str, Any]], str]:
    return lambda row: str(row[field])


def pubmed(row: dict[str, Any]) -> str:
    context = row.get("context", {}).get("contexts", [])
    return f"{row.get('question', '')}\n{' '.join(map(str, context))}"


def sql(row: dict[str, Any]) -> str:
    return f"{row.get('context', '')}\n{row.get('question', '')}\n{row.get('answer', '')}"


SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "qiaojin/PubMedQA",
        "config": "pqa_labeled",
        "label": "text/biomedical",
        "extract": pubmed,
        "query": TEXT_QUERIES,
        "modality": "text",
    },
    {
        "id": "coastalcph/lex_glue",
        "config": "ledgar",
        "label": "text/legal",
        "extract": text_field("text"),
        "query": TEXT_QUERIES,
        "modality": "text",
    },
    {
        "id": "zeroshot/twitter-financial-news-sentiment",
        "config": None,
        "label": "text/finance",
        "extract": text_field("text"),
        "query": TEXT_QUERIES,
        "modality": "text",
    },
    {
        "id": "code-search-net/code_search_net",
        "config": "python",
        "label": "code/python",
        "extract": text_field("whole_func_string"),
        "query": CODE_QUERIES,
        "modality": "code",
    },
    {
        "id": "code-search-net/code_search_net",
        "config": "java",
        "label": "code/java",
        "extract": text_field("whole_func_string"),
        "query": CODE_QUERIES,
        "modality": "code",
    },
    {
        "id": "b-mc2/sql-create-context",
        "config": None,
        "label": "code/sql",
        "extract": sql,
        "query": CODE_QUERIES,
        "modality": "code",
    },
    {
        "id": "DigitalLearningGmbH/MATH-lighteval",
        "config": None,
        "label": "math/algebra",
        "extract": text_field("problem"),
        "query": MATH_QUERIES,
        "modality": "math",
        "filter": lambda r: "algebra" in str(r.get("type", "")).lower(),
    },
    {
        "id": "DigitalLearningGmbH/MATH-lighteval",
        "config": None,
        "label": "math/geometry",
        "extract": text_field("problem"),
        "query": MATH_QUERIES,
        "modality": "math",
        "filter": lambda r: "geometry" in str(r.get("type", "")).lower(),
    },
    {
        "id": "DigitalLearningGmbH/MATH-lighteval",
        "config": None,
        "label": "math/probability",
        "extract": text_field("problem"),
        "query": MATH_QUERIES,
        "modality": "math",
        "filter": lambda r: (
            "counting" in str(r.get("type", "")).lower()
            or "probability" in str(r.get("type", "")).lower()
        ),
    },
)


def split_for(content: str) -> str:
    bucket = int(hashlib.sha256(f"{SEED}:{content}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-class", type=int, default=2500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    records: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    for source in SOURCES:
        info = api.dataset_info(source["id"])
        dataset = load_dataset(source["id"], source["config"], split="train", streaming=True)
        seen: set[str] = set()
        count = 0
        for row in dataset:
            predicate = source.get("filter")
            if predicate and not predicate(row):
                continue
            content = " ".join(source["extract"](row).split())
            digest = hashlib.sha256(content.encode()).hexdigest()
            if len(content) < 20 or digest in seen:
                continue
            seen.add(digest)
            query = stable_choice(source["query"], content)
            records.append(
                {
                    "text": joined(query, content, source["modality"]),
                    "label": source["label"],
                    "source": source["id"],
                    "split": split_for(content),
                }
            )
            count += 1
            if count >= args.max_per_class:
                break
        card = info.card_data or {}
        audit.append(
            {
                "source": source["id"],
                "config": source["config"],
                "revision": info.sha,
                "license": card.get("license", "not declared in dataset card"),
                "schema": {key: str(value) for key, value in dataset.features.items()},
                "mapped_route": source["label"],
                "accepted_examples": count,
            }
        )
    records.sort(key=lambda item: hashlib.sha256(item["text"].encode()).hexdigest())
    for split in ("train", "validation", "test"):
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter((record["label"], record["split"]) for record in records)
    manifest = {
        "seed": SEED,
        "max_per_class": args.max_per_class,
        "sources": audit,
        "counts": {f"{label}:{split}": count for (label, split), count in sorted(counts.items())},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
