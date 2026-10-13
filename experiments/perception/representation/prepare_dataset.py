"""Prepare deterministic four-class proxy labels from DocLayNet-small annotations.

This explicit experiment command downloads one pinned 381 MB archive. It is
never imported by the perception runtime. The derived labels are proxies for
layout sufficiency, not a human-verified gold set for arbitrary screenshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_REPO = "pierreguillou/DocLayNet-small"
SOURCE_REVISION = "389e375"
SOURCE_VERSION = "DocLayNet 1.1.0; small subset published 2023-01"
SOURCE_LICENSE = "CDLA-Permissive-1.0 (upstream DocLayNet license)"
SOURCE_URL = (
    "https://huggingface.co/datasets/pierreguillou/DocLayNet-small/resolve/"
    f"{SOURCE_REVISION}/data/dataset_small.zip"
)
CLASSES = ("text_sufficient", "structure_sufficient", "pixels_required", "mixed")
CATEGORY_NAMES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)
TEXT_CATEGORIES = {0, 1, 4, 5, 7, 9, 10}
STRUCTURE_CATEGORIES = {2, 3, 8}
PIXEL_CATEGORIES = {6}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / PurePosixPath(member.filename)).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe path in dataset archive: {member.filename}")
    archive.extractall(destination)


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "mvs-stage2/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _category_ids(form: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for item in form:
        category = item.get("category")
        if isinstance(category, int) and 0 <= category < len(CATEGORY_NAMES):
            result.add(category)
        elif isinstance(category, str):
            normalized = category.strip().lower()
            result.update(
                index for index, name in enumerate(CATEGORY_NAMES) if name.lower() == normalized
            )
    return result


def proxy_label(category_ids: set[int], picture_fraction: float) -> str | None:
    """Map existing layout annotations into transparent task-label proxies."""
    has_text = bool(category_ids & TEXT_CATEGORIES)
    has_structure = bool(category_ids & STRUCTURE_CATEGORIES)
    has_pixel_element = bool(category_ids & PIXEL_CATEGORIES)
    if has_pixel_element and picture_fraction >= 0.20:
        return "pixels_required"
    if has_pixel_element and (has_text or has_structure):
        return "mixed"
    if has_pixel_element:
        return "pixels_required"
    if has_structure:
        return "structure_sufficient"
    if has_text:
        return "text_sufficient"
    return None


def prepare(output_dir: Path, archive_path: Path, refresh: bool = False) -> dict[str, Any]:
    if refresh or not archive_path.is_file():
        _download(archive_path)
    with archive_path.open("rb") as source_archive:
        archive_digest = hashlib.file_digest(source_archive, "sha256").hexdigest()
    extraction = output_dir / "source"
    extraction.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, extraction)

    source_root = extraction / "small_dataset"
    split_map = {"train": "train", "val": "validation", "test": "test"}
    manifest_path = output_dir / "manifest.jsonl"
    counts: dict[str, Counter[str]] = {split: Counter() for split in split_map.values()}
    source_counts: Counter[str] = Counter()
    total = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for archive_split, canonical_split in split_map.items():
            annotation_dir = source_root / archive_split / "annotations"
            image_dir = source_root / archive_split / "images"
            if not annotation_dir.is_dir() or not image_dir.is_dir():
                raise FileNotFoundError(f"unexpected DocLayNet-small split layout: {archive_split}")
            for annotation_path in sorted(annotation_dir.glob("*.json")):
                source_counts[canonical_split] += 1
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                form = annotation.get("form", [])
                metadata = annotation.get("metadata", {})
                page_width = float(metadata.get("coco_width", 1025))
                page_height = float(metadata.get("coco_height", 1025))
                picture_area = 0.0
                for item in form:
                    if item.get("category") != "Picture":
                        continue
                    box = item.get("box", [])
                    if len(box) == 4:
                        x, y, width, height = (float(value) for value in box)
                        clipped_width = max(0.0, min(page_width, x + width) - max(0.0, x))
                        clipped_height = max(0.0, min(page_height, y + height) - max(0.0, y))
                        picture_area += clipped_width * clipped_height
                picture_fraction = min(
                    1.0, picture_area / max(1.0, page_width * page_height)
                )
                label = proxy_label(_category_ids(form), picture_fraction)
                if label is None:
                    continue
                image_path = image_dir / f"{annotation_path.stem}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(f"annotation image missing: {image_path}")
                record = {
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "label": label,
                    "split": canonical_split,
                    "source_annotation": annotation_path.relative_to(output_dir).as_posix(),
                    "source_page_hash": annotation.get("metadata", {}).get("page_hash"),
                    "doc_category": annotation.get("metadata", {}).get("doc_category"),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[canonical_split][label] += 1
                total += 1

    summary: dict[str, Any] = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_version": SOURCE_VERSION,
        "source_license": SOURCE_LICENSE,
        "source_url": SOURCE_URL,
        "archive_sha256": archive_digest,
        "archive_bytes": archive_path.stat().st_size,
        "manifest": "manifest.jsonl",
        "class_order": list(CLASSES),
        "proxy_rules": {
            "mixed": "Picture plus any text or structure annotation",
            "pixels_required": "Picture bounding boxes cover at least 20% of page area",
            "structure_sufficient": "Table, Formula, or List-item annotation without Picture",
            "text_sufficient": "Text-like annotation without Picture or structure",
        },
        "annotation_categories": list(CATEGORY_NAMES),
        "source_split_page_counts": dict(source_counts),
        "split_counts": {split: dict(counter) for split, counter in counts.items()},
        "split_totals": {split: sum(counter.values()) for split, counter in counts.items()},
        "total_samples": total,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/perception/cache/doclaynet-small-389e375.zip"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.output_dir, args.archive, args.refresh)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
