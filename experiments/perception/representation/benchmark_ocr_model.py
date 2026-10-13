"""Measure sequential versus concurrent local OCR and MobileNet inference."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.perception.ocr.factory import get_default_ocr_engine  # noqa: E402
from services.perception.representation.mobilenet import MobileNetV3SmallSelector  # noqa: E402


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def _timed(action: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1000


def benchmark(manifest: Path, sample_limit: int, repeats: int) -> dict[str, Any]:
    root = manifest.parent
    test_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "test"
    ][:sample_limit]
    if not test_rows:
        raise ValueError("no held-out images found in the manifest")

    ocr = get_default_ocr_engine()
    if not ocr.is_available():
        raise RuntimeError(
            "local Tesseract OCR is unavailable; OCR/model concurrency was not measured"
        )
    selector = MobileNetV3SmallSelector()
    images = [(root / row["image"]).read_bytes() for row in test_rows]
    selector.score_image(images[0])
    ocr.extract(images[0])

    sequential: list[float] = []
    parallel: list[float] = []
    ocr_ms: list[float] = []
    model_ms: list[float] = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="perception-bench") as pool:
        for image in images:
            for _ in range(repeats):
                start = time.perf_counter()
                _, ocr_elapsed = _timed(partial(ocr.extract, image))
                _, model_elapsed = _timed(partial(selector.score_image, image))
                sequential.append((time.perf_counter() - start) * 1000)
                ocr_ms.append(ocr_elapsed)
                model_ms.append(model_elapsed)

                start = time.perf_counter()
                ocr_future = pool.submit(ocr.extract, image)
                model_future = pool.submit(selector.score_image, image)
                ocr_future.result()
                model_future.result()
                parallel.append((time.perf_counter() - start) * 1000)

    sequential_median = statistics.median(sequential)
    parallel_median = statistics.median(parallel)
    return {
        "device": str(selector._device),
        "ocr_engine": type(ocr).__name__,
        "sample_count": len(images),
        "repeats_per_sample": repeats,
        "measurements": len(sequential),
        "sequential_ms": {
            "median": sequential_median,
            "p95": _percentile(sequential, 0.95),
        },
        "parallel_ms": {"median": parallel_median, "p95": _percentile(parallel, 0.95)},
        "ocr_only_ms": {"median": statistics.median(ocr_ms), "p95": _percentile(ocr_ms, 0.95)},
        "model_only_ms": {
            "median": statistics.median(model_ms),
            "p95": _percentile(model_ms, 0.95),
        },
        "parallel_speedup": sequential_median / parallel_median if parallel_median else None,
        "note": "Warm model, same held-out local images; includes thread submission and joins.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = benchmark(args.manifest, args.samples, args.repeats)
    output = Path("models/perception/ocr_model_benchmark.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
