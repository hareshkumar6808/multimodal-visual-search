"""Measure production pipeline inference latency without network/provider noise."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    model = joblib.load(ROOT / "services/orchestrator/artifacts/task_router_v1.joblib")["pipeline"]
    sample = (
        "intent_debug modality_code representation_ocr has_ocr_true\n"
        "QUESTION: Why is this failing?\nCONTENT: print(items[9])"
    )
    model.predict_proba([sample])
    values = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        model.predict_proba([sample])
        values.append((time.perf_counter() - started) * 1000)
    report = {
        "iterations": args.iterations,
        "classifier_ms": {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        },
    }
    output = ROOT / ".tools/stage2/reports/router_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
