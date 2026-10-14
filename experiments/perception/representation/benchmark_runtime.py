"""Measure cold selector startup, warm ONNX inference, and process memory."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.perception.models import Modalities, OCRResult  # noqa: E402
from services.perception.representation.mobilenet import (  # noqa: E402
    MobileNetV3SmallSelector,
)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _memory_bytes() -> dict[str, int] | None:
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process = windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    process = get_process()
    get_memory = windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    success = get_memory(process, ctypes.byref(counters), counters.cb)
    if not success:
        return None
    return {
        "working_set": int(counters.WorkingSetSize),
        "peak_working_set": int(counters.PeakWorkingSetSize),
        "pagefile_usage": int(counters.PagefileUsage),
        "peak_pagefile_usage": int(counters.PeakPagefileUsage),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


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

    memory_before = _memory_bytes()
    selector = MobileNetV3SmallSelector()
    cold_started = time.perf_counter()
    selector.score_image(images[0])
    cold_ms = (time.perf_counter() - cold_started) * 1000
    memory_after_load = _memory_bytes()

    measurements: list[float] = []
    decision_measurements: list[float] = []
    for index in range(repeats):
        image = images[index % len(images)]
        started = time.perf_counter()
        selector.score_image(image)
        measurements.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        selector.select(
            Modalities(text=0.8),
            "text",
            0.9,
            OCRResult(text="Warm local OCR fixture", confidence=0.9),
            False,
            image,
        )
        decision_measurements.append((time.perf_counter() - started) * 1000)
    overhead = [decision - inference for decision, inference in zip(
        decision_measurements, measurements, strict=True
    )]
    return {
        "backend": selector._backend,
        "device": str(selector._device),
        "model_id": selector.model_id,
        "sample_count": len(images),
        "warm_repeats": repeats,
        "cold_first_inference_ms": cold_ms,
        "warm_inference_ms": {
            "count": len(measurements),
            "median": statistics.median(measurements),
            "p95": _percentile(measurements, 0.95),
        },
        "warm_selector_decision_ms": {
            "count": len(decision_measurements),
            "median": statistics.median(decision_measurements),
            "p95": _percentile(decision_measurements, 0.95),
        },
        "decision_gate_overhead_estimate_ms": {
            "median_paired_difference": statistics.median(overhead),
            "p95_paired_difference": _percentile(overhead, 0.95),
            "scope": (
                "selector.select minus score_image; paired run order may add CPU scheduling noise"
            ),
        },
        "process_memory_bytes": {
            "before_selector_load": memory_before,
            "after_cold_load_and_inference": memory_after_load,
            "limitation": (
                "process working set includes Python, Pillow, and ONNX Runtime; "
                "not an isolated model allocation"
            ),
        },
        "scope": (
            "selector inference includes image decode, resize, normalization, and backend "
            "inference; excludes OCR"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/perception/dataset/doclaynet-small/manifest.jsonl"),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
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
