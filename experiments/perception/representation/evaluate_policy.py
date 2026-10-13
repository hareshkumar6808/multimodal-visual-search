"""Compare raw model, heuristic, and composed perception policy on a fixed split.

This evaluator only scores existing proxy labels. It never downloads data or
models and must not be interpreted as a task-sufficiency benchmark.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.perception.ocr.factory import get_default_ocr_engine  # noqa: E402
from services.perception.pipeline import PerceptionPipeline  # noqa: E402
from services.perception.representation.mobilenet import (  # noqa: E402
    REPRESENTATION_CLASSES,
    MobileNetV3SmallSelector,
)
from services.perception.representation.selector import (  # noqa: E402
    HeuristicRepresentationSelector,
)


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    size = len(REPRESENTATION_CLASSES)
    matrix = [[0] * size for _ in range(size)]
    for row in rows:
        matrix[REPRESENTATION_CLASSES.index(row["label"])][
            REPRESENTATION_CLASSES.index(row[key])
        ] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, name in enumerate(REPRESENTATION_CLASSES):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(line[index] for line in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    critical: dict[str, dict[str, int | float]] = {}
    for label in ("pixels_required", "mixed"):
        index = REPRESENTATION_CLASSES.index(label)
        support = sum(matrix[index])
        false_discard = sum(
            matrix[index][j]
            for j, pred in enumerate(REPRESENTATION_CLASSES)
            if pred in {"text_sufficient", "structure_sufficient"}
        )
        critical[label] = {
            "false_pixel_discard_count": false_discard,
            "support": support,
            "rate": false_discard / support if support else 0.0,
        }
    total = len(rows)
    return {
        "sample_count": total,
        "accuracy": sum(matrix[i][i] for i in range(size)) / total if total else 0.0,
        "macro_precision": statistics.mean(precisions),
        "macro_recall": statistics.mean(recalls),
        "macro_f1": statistics.mean(f1s),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(REPRESENTATION_CLASSES),
            "rows_true_columns_predicted": matrix,
        },
        "critical_errors": critical,
        "combined_critical_errors": {
            "count": sum(int(item["false_pixel_discard_count"]) for item in critical.values()),
            "support": sum(int(item["support"]) for item in critical.values()),
            "rate": (
                sum(int(item["false_pixel_discard_count"]) for item in critical.values())
                / max(1, sum(int(item["support"]) for item in critical.values()))
            ),
        },
        "pixel_preservation": {
            "count": sum(row[key] in {"pixels_required", "mixed"} for row in rows),
            "rate": sum(row[key] in {"pixels_required", "mixed"} for row in rows)
            / max(1, total),
            "unnecessary_count": sum(
                row["label"] in {"text_sufficient", "structure_sufficient"}
                and row[key] in {"pixels_required", "mixed"}
                for row in rows
            ),
            "unnecessary_support": sum(
                row["label"] in {"text_sufficient", "structure_sufficient"}
                for row in rows
            ),
        },
    }


def _calibration(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    ece = 0.0
    bin_rows: list[dict[str, Any]] = []
    brier = 0.0
    for row in rows:
        truth = REPRESENTATION_CLASSES.index(row["label"])
        probabilities = row["probabilities"]
        brier += sum(
            (prob - float(index == truth)) ** 2
            for index, prob in enumerate(probabilities)
        )
    brier /= len(rows)
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            row for row in rows
            if lower <= row["confidence"] < upper
            or (bin_index == bins - 1 and row["confidence"] == 1.0)
        ]
        if not selected:
            continue
        mean_confidence = statistics.mean(row["confidence"] for row in selected)
        accuracy = statistics.mean(row["label"] == row["raw_prediction"] for row in selected)
        ece += len(selected) / len(rows) * abs(mean_confidence - accuracy)
        bin_rows.append({
            "lower": lower,
            "upper": upper,
            "count": len(selected),
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })
    confidence_distribution: dict[str, Any] = {}
    for correctness, name in ((True, "correct"), (False, "incorrect")):
        confidence_values = [
            float(row["confidence"])
            for row in rows
            if row["raw_correct"] is correctness
        ]
        confidence_distribution[name] = {
            "count": len(confidence_values),
            "mean": statistics.mean(confidence_values) if confidence_values else None,
            "median": statistics.median(confidence_values) if confidence_values else None,
            "min": min(confidence_values) if confidence_values else None,
            "max": max(confidence_values) if confidence_values else None,
        }
    return {
        "sample_count": len(rows),
        "multiclass_brier": brier,
        "ece_10_bin": ece,
        "confidence_by_correctness": confidence_distribution,
        "bins": bin_rows,
    }


def _degradation_robustness(
    evaluated: list[dict[str, Any]],
    image_bytes_by_hash: dict[str, bytes],
    selector: MobileNetV3SmallSelector,
) -> dict[str, Any]:
    variants = ("quarter_resolution_jpeg_q40", "native_resolution_jpeg_q40")
    outputs: dict[str, Any] = {}
    for variant in variants:
        degraded_rows: list[dict[str, Any]] = []
        agreements = 0
        for row in evaluated:
            with Image.open(io.BytesIO(image_bytes_by_hash[row["page_hash"]])) as source:
                image = source.convert("RGB")
                if variant.startswith("quarter"):
                    reduced = image.resize(
                        (max(1, image.width // 4), max(1, image.height // 4)),
                        Image.Resampling.BILINEAR,
                    )
                    image = reduced.resize(image.size, Image.Resampling.BILINEAR)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=40)
            probabilities = selector.score_image(buffer.getvalue())
            prediction = REPRESENTATION_CLASSES[probabilities.index(max(probabilities))]
            agreements += prediction == row["raw_prediction"]
            degraded_rows.append({"label": row["label"], "prediction": prediction})
        outputs[variant] = {
            "raw_model_metrics": _metrics(degraded_rows, "prediction"),
            "agreement_with_original_count": agreements,
            "agreement_with_original_rate": agreements / max(1, len(evaluated)),
            "sample_count": len(evaluated),
        }
    return outputs


def _threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for threshold in (0.0, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.99):
        for row in rows:
            row["threshold_prediction"] = (
                row["raw_prediction"]
                if row["confidence"] >= threshold
                else "pixels_required"
            )
        metrics = _metrics(rows, "threshold_prediction")
        preserved = metrics["pixel_preservation"]
        output.append({
            "threshold": threshold,
            "abstained_count": sum(row["confidence"] < threshold for row in rows),
            "abstention_rate": sum(row["confidence"] < threshold for row in rows) / max(
                1, len(rows)
            ),
            "critical_errors": metrics["critical_errors"],
            "combined_critical_errors": metrics["combined_critical_errors"],
            "preserved_count": preserved["count"],
            "preserved_rate": preserved["rate"],
            "unnecessary_preserves": preserved["unnecessary_count"],
            "unnecessary_preservation_rate": preserved["unnecessary_count"]
            / max(1, int(preserved["unnecessary_support"])),
        })
    return output


def evaluate(
    manifest_path: Path,
    split: str,
    limit: int | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    manifest_root = manifest_path.parent
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == split
    ]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows for split {split!r}")
    engine = get_default_ocr_engine()
    if not engine.is_available():
        raise RuntimeError("A local OCR engine is required for the composed-policy comparison")
    selector = (
        MobileNetV3SmallSelector(checkpoint_path, onnx_path=ROOT / "models/perception/no-onnx")
        if checkpoint_path
        else MobileNetV3SmallSelector()
    )
    pipeline = PerceptionPipeline(ocr_engine=engine, representation_selector=selector)
    heuristic = HeuristicRepresentationSelector()
    evaluated: list[dict[str, Any]] = []
    image_bytes_by_hash: dict[str, bytes] = {}
    pipeline_ms: list[float] = []
    for row in rows:
        image_bytes = (manifest_root / row["image"]).read_bytes()
        image_bytes_by_hash[row["source_page_hash"]] = image_bytes
        started = time.perf_counter()
        mir = pipeline.analyze(image_bytes, request_id=row["source_page_hash"] or "eval")
        pipeline_ms.append((time.perf_counter() - started) * 1000)
        probabilities = selector.score_image(image_bytes)
        raw_prediction = REPRESENTATION_CLASSES[probabilities.index(max(probabilities))]
        heuristic_result = heuristic.select(
            mir.modalities,
            mir.primary_modality,
            mir.overall_confidence,
            mir.ocr,
            mir.visual_required,
            image_bytes,
        )
        evaluated.append({
            "page_hash": row["source_page_hash"],
            "doc_category": row.get("doc_category"),
            "label": row["label"],
            "raw_prediction": raw_prediction,
            "raw_correct": raw_prediction == row["label"],
            "probabilities": probabilities,
            "confidence": max(probabilities),
            "heuristic_prediction": heuristic_result.kind,
            "composed_prediction": mir.representation.kind if mir.representation else None,
            "visual_required": mir.visual_required,
            "primary_modality": mir.primary_modality,
            "ocr_confidence": mir.ocr.confidence,
            "pipeline_ms": pipeline.last_telemetry["total_pipeline_ms"],
        })
    source_metrics: dict[str, Any] = {}
    for category in sorted({row["doc_category"] for row in evaluated}):
        subset = [row for row in evaluated if row["doc_category"] == category]
        source_metrics[str(category)] = {
            "count": len(subset),
            "raw_model": _metrics(subset, "raw_prediction"),
            "heuristic": _metrics(subset, "heuristic_prediction"),
            "composed": _metrics(subset, "composed_prediction"),
        }
    return {
        "experiment_id": f"POLICY-{split.upper()}-01",
        "split": split,
        "status": (
            "historical diagnostic; validation supports threshold analysis; test split "
            "previously consulted and not an untouched final evaluation"
        ),
        "dataset_manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path or ROOT / "models/perception/mobilenet_v3_small.onnx"),
        "sample_count": len(evaluated),
        "class_counts": {
            name: sum(row["label"] == name for row in evaluated)
            for name in REPRESENTATION_CLASSES
        },
        "raw_model": _metrics(evaluated, "raw_prediction"),
        "heuristic": _metrics(evaluated, "heuristic_prediction"),
        "composed_runtime_selector": _metrics(evaluated, "composed_prediction"),
        "raw_model_calibration": _calibration(evaluated),
        "pixel_preservation_signals": {
            "stage1_visual_required_true": sum(row["visual_required"] for row in evaluated),
            "stage1_visual_required_false": sum(not row["visual_required"] for row in evaluated),
            "critical_proxy_cases_with_visual_required_false": sum(
                row["label"] in {"pixels_required", "mixed"} and not row["visual_required"]
                for row in evaluated
            ),
            "note": (
                "Stage 1 signal only; query-specific router determines actual provider "
                "image forwarding."
            ),
        },
        "synthetic_degradation_sensitivity": (
            _degradation_robustness(evaluated, image_bytes_by_hash, selector)
            if split == "validation"
            else None
        ),
        "validation_threshold_sweep": (
            _threshold_sweep(evaluated) if split == "validation" else None
        ),
        "by_document_category": source_metrics,
        "pipeline_latency_ms": {
            "measurements": len(pipeline_ms),
            "median": statistics.median(pipeline_ms),
            "p95": sorted(pipeline_ms)[
                min(len(pipeline_ms) - 1, math.ceil(0.95 * len(pipeline_ms)) - 1)
            ],
            "scope": (
                "warm local model; includes preflight, OCR, profiling, extraction, and selector"
            ),
        },
        "pixel_forwarding_policy": (
            "representation metadata is advisory; orchestrator image_required follows "
            "Stage 1 routing, not this prediction"
        ),
        "limitations": [
            "proxy labels do not establish task-specific sufficiency",
            "held-out test was already consulted in the previous development milestone",
            "small minority supports, particularly mixed and pixels_required",
            "validation threshold sweep is exploratory and not a confidence interval",
        ],
        "samples": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.split, args.limit, args.checkpoint)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if not args.output:
        print(serialized, end="")
    else:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
