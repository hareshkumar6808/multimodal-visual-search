"""Evaluate the frozen representation selector on conservative cross-source proxies.

This explicit research command downloads small, capped public samples. The mappings are
not task-level sufficiency ground truth and no downloaded image is committed or used at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".tools/huggingface"))

from datasets import load_dataset  # noqa: E402

from services.perception.representation.mobilenet import (  # noqa: E402
    REPRESENTATION_CLASSES,
    MobileNetV3SmallSelector,
)

CRITICAL = {"pixels_required", "mixed"}
DISCARD = {"text_sufficient", "structure_sufficient"}
SOURCE_METADATA = {
    "chartqa": {
        "dataset": "HuggingFaceM4/ChartQA",
        "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "license": "GPL-3.0 dataset card",
        "proxy": "pixels_required",
    },
    "cifar10": {
        "dataset": "uoft-cs/cifar10",
        "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
        "license": "dataset card declares unknown; benchmark images are not redistributed",
        "proxy": "pixels_required",
    },
    "screenspot": {
        "dataset": "HongxinLi/ScreenSpot_v2 mirror of ScreenSpot v2",
        "revision": "7386eb340a78e29c44add97a5958915864e7f3af",
        "license": "mirror card undeclared; upstream ScreenSpot v2 card is Apache-2.0",
        "proxy": "mixed",
    },
    "rvl_cdip_demo": {
        "dataset": "nielsr/rvl_cdip_10_examples_per_class",
        "revision": "94115dbd325e701c0a95abf292e0d119a0c83a57",
        "license": "dataset card undeclared; images are evaluation-only and not redistributed",
        "proxy": "class-name mapping to text/structure/mixed",
    },
    "codesearchnet_render": {
        "dataset": "code-search-net/code_search_net (Python)",
        "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
        "license": "dataset card declares other; rendered samples are not redistributed",
        "proxy": "mixed",
    },
}


def _image_bytes(image: Image.Image, *, quality: int = 90) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _render_code(code: str) -> Image.Image:
    image = Image.new("RGB", (1280, 720), (25, 27, 31))
    draw = ImageDraw.Draw(image)
    y = 20
    for line in code.expandtabs(4).splitlines()[:42]:
        draw.text((20, y), line[:150], fill=(224, 224, 224))
        y += 16
    return image


def _take(dataset: Any, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in dataset:
        result.append(row)
        if len(result) >= limit:
            break
    return result


def load_samples(limit_per_source: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    chart_rows = _take(
        load_dataset(
            "HuggingFaceM4/ChartQA",
            revision=SOURCE_METADATA["chartqa"]["revision"],
            split="test",
            streaming=True,
        ),
        limit_per_source,
    )
    samples.extend(
        _sample("chartqa", str(i), "pixels_required", row["image"])
        for i, row in enumerate(chart_rows)
    )

    photo_rows = _take(
        load_dataset(
            "uoft-cs/cifar10",
            revision=SOURCE_METADATA["cifar10"]["revision"],
            split="test",
            streaming=True,
        ),
        limit_per_source,
    )
    samples.extend(
        _sample("cifar10", str(i), "pixels_required", row["img"])
        for i, row in enumerate(photo_rows)
    )

    ui_rows = _take(
        load_dataset(
            "HongxinLi/ScreenSpot_v2",
            revision=SOURCE_METADATA["screenspot"]["revision"],
            split="test",
            streaming=True,
        ),
        limit_per_source,
    )
    samples.extend(
        _sample("screenspot", str(row["file_name"]), "mixed", row["image"]) for row in ui_rows
    )

    rvl = load_dataset(
        "nielsr/rvl_cdip_10_examples_per_class",
        revision=SOURCE_METADATA["rvl_cdip_demo"]["revision"],
        split="train",
        streaming=True,
    )
    names = rvl.features["label"]
    rvl_map = {
        "letter": "text_sufficient",
        "email": "text_sufficient",
        "scientific report": "text_sufficient",
        "scientific publication": "text_sufficient",
        "specification": "text_sufficient",
        "resume": "text_sufficient",
        "memo": "text_sufficient",
        "form": "structure_sufficient",
        "budget": "structure_sufficient",
        "invoice": "structure_sufficient",
        "questionnaire": "structure_sufficient",
        "handwritten": "mixed",
        "advertisement": "mixed",
        "news article": "mixed",
        "presentation": "mixed",
        "file folder": "mixed",
    }
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(rvl):
        name = names.int2str(row["label"])
        if source_counts[name] >= min(10, limit_per_source):
            continue
        source_counts[name] += 1
        samples.append(_sample("rvl_cdip_demo", f"{name}:{index}", rvl_map[name], row["image"]))

    code_rows = _take(
        load_dataset(
            "code-search-net/code_search_net",
            "python",
            revision=SOURCE_METADATA["codesearchnet_render"]["revision"],
            split="train",
            streaming=True,
        ),
        limit_per_source,
    )
    samples.extend(
        _sample(
            "codesearchnet_render",
            hashlib.sha256(row["func_code_string"].encode()).hexdigest()[:16],
            "mixed",
            _render_code(row["func_code_string"]),
        )
        for row in code_rows
    )
    return samples


def _sample(source: str, sample_id: str, label: str, image: Image.Image) -> dict[str, Any]:
    encoded = _image_bytes(image)
    return {
        "source": source,
        "sample_id": sample_id,
        "label": label,
        "image": encoded,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = list(REPRESENTATION_CLASSES)
    matrix = [[0] * len(labels) for _ in labels]
    for row in rows:
        matrix[labels.index(row["label"])][labels.index(row["prediction"])] += 1
    supports = {label: sum(matrix[i]) for i, label in enumerate(labels)}
    f1s: list[float] = []
    per_class: dict[str, Any] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        predicted = sum(line[i] for line in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / supports[label] if supports[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if supports[label]:
            f1s.append(f1)
        per_class[label] = {
            "support": supports[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    critical_rows = [row for row in rows if row["label"] in CRITICAL]
    false_discards = [row for row in critical_rows if row["prediction"] in DISCARD]
    correct = sum(row["prediction"] == row["label"] for row in rows)
    return {
        "sample_count": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1_present_classes": statistics.mean(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": matrix},
        "critical_false_discard": {
            "count": len(false_discards),
            "support": len(critical_rows),
            "rate": len(false_discards) / len(critical_rows) if critical_rows else 0.0,
        },
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lower in [i / 10 for i in range(10)]:
        upper = lower + 0.1
        selected = [
            row
            for row in rows
            if lower <= row["confidence"] < upper or (upper == 1.0 and row["confidence"] == 1.0)
        ]
        if not selected:
            continue
        accuracy = sum(row["prediction"] == row["label"] for row in selected) / len(selected)
        confidence = statistics.mean(row["confidence"] for row in selected)
        ece += len(selected) / len(rows) * abs(accuracy - confidence)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "accuracy": accuracy,
                "mean_confidence": confidence,
            }
        )
    brier = statistics.mean(
        sum(
            (probability - float(index == REPRESENTATION_CLASSES.index(row["label"]))) ** 2
            for index, probability in enumerate(row["probabilities"])
        )
        for row in rows
    )
    return {"ece_10_bin": ece, "multiclass_brier": brier, "bins": bins}


def _thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for threshold in (0.0, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99):
        predictions = []
        for row in rows:
            item = dict(row)
            if item["confidence"] < threshold:
                item["prediction"] = "pixels_required"
            predictions.append(item)
        metrics = _metrics(predictions)
        abstained = sum(row["confidence"] < threshold for row in rows)
        unnecessary = sum(
            row["label"] in DISCARD and predicted["prediction"] in CRITICAL
            for row, predicted in zip(rows, predictions, strict=True)
        )
        result.append(
            {
                "threshold": threshold,
                "abstained": abstained,
                "coverage": 1 - abstained / len(rows),
                "critical_false_discard": metrics["critical_false_discard"],
                "unnecessary_preservation": unnecessary,
            }
        )
    return result


def _degraded(image_bytes: bytes, kind: str) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        if kind == "downscale_50":
            small = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
            image = small.resize(image.size)
        elif kind == "jpeg_40":
            return _image_bytes(image, quality=40)
        elif kind == "mild_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif kind == "brightness_70":
            image = ImageEnhance.Brightness(image).enhance(0.7)
        elif kind == "hidpi_downsample":
            small = image.resize((max(1, image.width // 4), max(1, image.height // 4)))
            image = small.resize(image.size)
        return _image_bytes(image)


def benchmark(limit_per_source: int) -> dict[str, Any]:
    selector = MobileNetV3SmallSelector()
    rows = load_samples(limit_per_source)
    degradation_kinds = (
        "downscale_50",
        "jpeg_40",
        "mild_blur",
        "brightness_70",
        "hidpi_downsample",
    )
    for row in rows:
        image_bytes = row.pop("image")
        probabilities = selector.score_image(image_bytes)
        row["probabilities"] = probabilities
        row["confidence"] = max(probabilities)
        row["prediction"] = REPRESENTATION_CLASSES[probabilities.index(max(probabilities))]
        row["degradation_predictions"] = {}
        for kind in degradation_kinds:
            degraded_probabilities = selector.score_image(_degraded(image_bytes, kind))
            row["degradation_predictions"][kind] = REPRESENTATION_CLASSES[
                degraded_probabilities.index(max(degraded_probabilities))
            ]

    by_source = {
        source: _metrics([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    confidence = {
        "correct": [row["confidence"] for row in rows if row["prediction"] == row["label"]],
        "incorrect": [row["confidence"] for row in rows if row["prediction"] != row["label"]],
        "critical_false_discard": [
            row["confidence"]
            for row in rows
            if row["label"] in CRITICAL and row["prediction"] in DISCARD
        ],
    }
    critical_errors = [
        {
            key: row[key]
            for key in (
                "source",
                "sample_id",
                "sha256",
                "label",
                "prediction",
                "confidence",
                "probabilities",
            )
        }
        for row in rows
        if row["label"] in CRITICAL and row["prediction"] in DISCARD
    ][:50]
    degradation = {}
    for kind in degradation_kinds:
        degraded_rows = [
            {**row, "prediction": row["degradation_predictions"][kind]} for row in rows
        ]
        degradation[kind] = {
            "prediction_agreement": sum(
                row["prediction"] == row["degradation_predictions"][kind] for row in rows
            )
            / len(rows),
            "metrics": _metrics(degraded_rows),
        }
    return {
        "benchmark_kind": "conservative proxy classification; not task-level sufficiency",
        "model_id": selector.model_id,
        "source_metadata": SOURCE_METADATA,
        "overall": _metrics(rows),
        "by_source": by_source,
        "calibration": _calibration(rows),
        "threshold_sweep": _thresholds(rows),
        "confidence": {
            key: {
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "max": max(values) if values else None,
            }
            for key, values in confidence.items()
        },
        "critical_errors": critical_errors,
        "degradation": degradation,
        "limitations": [
            "Mappings are conservative source/category proxies, not query-conditioned "
            "sufficiency labels.",
            "Dataset cards for CIFAR-10, RVL-CDIP demo, and the ScreenSpot mirror do not "
            "declare complete redistribution terms; no raw image is committed.",
            "The benchmark is capped and does not estimate production prevalence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-source", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=Path("models/perception/generalization_benchmark.json")
    )
    args = parser.parse_args()
    report = benchmark(args.limit_per_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "by_source": report["by_source"],
                "confidence": report["confidence"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
