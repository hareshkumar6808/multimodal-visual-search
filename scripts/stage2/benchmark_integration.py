from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from contracts.models import MIR
from services.orchestrator.execution.engine import ExecutionEngine, RouteDecisionV2
from services.orchestrator.execution.tools import math_tool, table_tool
from services.orchestrator.routing import UnifiedTaskRouter
from services.perception.representation.mobilenet import MobileNetV3SmallSelector


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


def measure(action: Callable[[], object], repeats: int) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        values.append((time.perf_counter() - started) * 1000)
    return values


def summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
    }


def fixture_mir() -> MIR:
    return MIR.model_validate(
        {
            "mir_version": "0.2",
            "request_id": "benchmark",
            "image": {"sha256": "a" * 64, "width": 320, "height": 180},
            "modalities": {"text": 0.98},
            "primary_modality": "text",
            "ocr": {
                "text": "Revenue increased while operating margin declined.",
                "confidence": 0.98,
            },
            "overall_confidence": 0.98,
        }
    )


def main() -> None:
    image = Image.new("RGB", (320, 180), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    selector = MobileNetV3SmallSelector()
    probabilities = selector.score_image(image_bytes)
    representation = measure(lambda: selector.score_image(image_bytes), 30)

    mir = fixture_mir()
    router = UnifiedTaskRouter()
    decision = router.route(mir, "explain", "Explain this")
    routing = measure(lambda: router.route(mir, "explain", "Explain this"), 300)

    engine = ExecutionEngine()
    direct_decision = RouteDecisionV2(
        expert="text",
        leaf_route="text/general",
        execution_mode="direct",
        requested_tool="return_ocr",
    )
    _, direct_policy = engine.adapt(direct_decision, "text-expert")
    direct = measure(
        lambda: engine.execute_local(direct_decision, direct_policy, ocr_text="Hello world"),
        300,
    )
    sympy_values = measure(lambda: math_tool("solve", "x^2 - 5x + 6 = 0"), 100)
    table_values = measure(lambda: table_tool("max", [10, 20, 7, 13]), 300)

    report = {
        "representation_model": {
            "loaded": selector.model_available,
            "model_id": selector.model_id,
            "scores": probabilities,
            "scores_sum": sum(probabilities),
            **summary(representation),
        },
        "router_model": {
            "loaded": router.ready,
            "leaf_route": decision.leaf_route,
            **summary(routing),
        },
        "direct_ocr_overhead": summary(direct),
        "sympy_solve": summary(sympy_values),
        "table_max": summary(table_values),
    }
    destination = Path(".tools/stage2/benchmark.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
