"""Measure Stage 2 learned-selector overhead against a Stage 1-equivalent path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.models import RepresentationInfo, RepresentationScores  # noqa: E402
from services.perception import PerceptionPipeline  # noqa: E402
from services.perception.models import OCRResult  # noqa: E402
from services.perception.ocr.base import OCREngine  # noqa: E402
from services.perception.representation import (  # noqa: E402
    MobileNetV3SmallSelector,
    RepresentationSelector,
)


class FixtureOCR(OCREngine):
    """Remove OCR variability so the paired result isolates selector cost."""

    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        del image_bytes
        return OCRResult(text="Deterministic benchmark text", confidence=0.95, regions=[])


class Stage1EquivalentSelector(RepresentationSelector):
    """Near-zero-cost adapter needed because MIR v0.2 requires representation data."""

    def select(self, *args: object, **kwargs: object) -> RepresentationInfo:
        del args, kwargs
        return RepresentationInfo(
            kind="pixels_required",
            confidence=1.0,
            pixel_dependency_prior=1.0,
            scores=RepresentationScores(pixels_required=1.0),
            source="stage1_equivalent_fixture",
            available_representations=["pixels"],
            local_capabilities={"ml_model": False},
        )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
    }


def benchmark(manifest: Path, repeats: int, sample_limit: int) -> dict[str, Any]:
    root = manifest.parent
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "validation"
    ][:sample_limit]
    images = [(root / row["image"]).read_bytes() for row in rows]
    if not images:
        raise ValueError("no validation images were found")

    baseline = PerceptionPipeline(
        ocr_engine=FixtureOCR(), representation_selector=Stage1EquivalentSelector()
    )
    learned_selector = MobileNetV3SmallSelector()
    stage2 = PerceptionPipeline(ocr_engine=FixtureOCR(), representation_selector=learned_selector)
    stage2.analyze(images[0], request_id="warmup")

    baseline_ms: list[float] = []
    stage2_ms: list[float] = []
    representation_ms: list[float] = []
    for repeat in range(repeats):
        for index, image in enumerate(images):
            started = time.perf_counter()
            baseline.analyze(image, request_id=f"baseline-{repeat}-{index}")
            baseline_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            stage2.analyze(image, request_id=f"stage2-{repeat}-{index}")
            stage2_ms.append((time.perf_counter() - started) * 1000)
            representation_ms.append(float(stage2.last_telemetry["representation_ms"]))

    paired = [new - old for new, old in zip(stage2_ms, baseline_ms, strict=True)]
    baseline_median = statistics.median(baseline_ms)
    added_median = statistics.median(paired)
    return {
        "method": "paired controlled pipeline benchmark",
        "backend": learned_selector._backend,
        "sample_count": len(images),
        "repeats_per_sample": repeats,
        "stage1_equivalent_total_ms": _summary(baseline_ms),
        "stage2_total_ms": _summary(stage2_ms),
        "stage2_representation_telemetry_ms": _summary(representation_ms),
        "added_overhead_ms": _summary(paired),
        "added_overhead_percent_of_controlled_baseline_median": (
            added_median / baseline_median * 100 if baseline_median else None
        ),
        "scope": (
            "Both paths use identical deterministic zero-I/O OCR. The baseline emits a "
            "constant pixel-preserving representation only to satisfy MIR v0.2. This isolates "
            "learned selector cost and is not an end-to-end OCR latency claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.manifest, args.repeats, args.samples)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
