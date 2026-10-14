from pathlib import Path

import joblib
import pytest

from contracts.models import MIR
from services.orchestrator.experts import EXPERTS
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import UnifiedTaskRouter
from services.orchestrator.routing.provider_performance import ProviderPerformanceStore


def mir(modality: str, text: str = "visible text", **extra: object) -> MIR:
    data: dict[str, object] = {
        "mir_version": "0.2",
        "request_id": "stage2-test",
        "image": {"sha256": "a" * 64, "width": 100, "height": 50},
        "modalities": {modality: 1.0},
        "primary_modality": modality,
        "ocr": {"text": text, "confidence": 0.96},
        "objects": [],
        "overall_confidence": 0.95,
    }
    data.update(extra)
    return MIR.model_validate(data)


@pytest.mark.parametrize(
    ("modality", "content", "expected"),
    [
        (
            "text",
            "A randomized clinical study of gout patients examined serum urate and "
            "urate-lowering therapy. Patients received allopurinol prescriptions and "
            "questionnaires were linked to medical records. Multivariable logistic "
            "regression measured predictors, odds ratios, confidence intervals, and "
            "health-related quality of life outcomes.",
            "text/biomedical",
        ),
        (
            "text",
            "This agreement is binding and governed by applicable laws. The parties "
            "consent to jurisdiction, arbitration, and indemnification.",
            "text/legal",
        ),
        (
            "text",
            "Quarterly revenue, operating profit, earnings per share and market sales rose.",
            "text/finance",
        ),
        ("code", "def fibonacci(n):\n return fibonacci(n-1) + fibonacci(n-2)", "code/python"),
        (
            "code",
            'public static void main(String[] args) { System.out.println("hello"); }',
            "code/java",
        ),
        ("code", "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id", "code/sql"),
        (
            "math",
            "Let x and y be real numbers satisfying x+y=10 and xy=21. Compute x^2+y^2.",
            "math/algebra",
        ),
        (
            "math",
            "In triangle ABC angle A is 60 degrees and sides AB=5 and AC=7. Find the area.",
            "math/geometry",
        ),
        (
            "math",
            "A fair coin is tossed five times. Find the probability of exactly three heads.",
            "math/probability",
        ),
    ],
)
def test_production_model_specialized_routes(modality: str, content: str, expected: str) -> None:
    decision = UnifiedTaskRouter().route(mir(modality, content), "explain", "Explain this.")
    assert decision.leaf_route == expected
    assert decision.router_source == "ml"


@pytest.mark.parametrize(
    ("modality", "expected"),
    [
        ("table", "table/general"),
        ("chart", "chart/general"),
        ("image", "vision/general"),
        ("ui", "ui/general"),
    ],
)
def test_single_child_families_are_deterministic(modality: str, expected: str) -> None:
    decision = UnifiedTaskRouter().route(mir(modality), "explain", "Explain this")
    assert decision.leaf_route == expected
    assert decision.router_source == "deterministic"


def test_low_confidence_abstains_to_family_general(tmp_path: Path) -> None:
    artifact = joblib.load(UnifiedTaskRouter().artifact_path)
    artifact["thresholds"] = {"text": 1.01, "code": 1.01, "math": 1.01}
    path = tmp_path / "strict.joblib"
    joblib.dump(artifact, path)
    decision = UnifiedTaskRouter(path).route(mir("text", "ambiguous words"), "explain", "Explain")
    assert decision.leaf_route == "text/general"
    assert decision.fallback_used is True
    assert decision.reason_code == "ML_LOW_CONFIDENCE_GENERAL_FALLBACK"


@pytest.mark.parametrize("contents", [None, b"not a joblib artifact"])
def test_missing_or_corrupt_model_uses_rule_router(tmp_path: Path, contents: bytes | None) -> None:
    path = tmp_path / "router.joblib"
    if contents is not None:
        path.write_bytes(contents)
    decision = UnifiedTaskRouter(path).route(mir("code", "print(value)"), "debug", "Why?")
    assert decision.leaf_route == "code/general"
    assert decision.router_source == "rule"
    assert decision.fallback_used is True


def test_stage1_mir_without_v02_fields_routes_safely() -> None:
    stage1 = mir("text", "ordinary paragraph").model_copy(update={"mir_version": "0.1"})
    assert (
        UnifiedTaskRouter().route(stage1, "summarize", "Summarize").leaf_route.startswith("text/")
    )


def test_execution_and_query_aware_pixels() -> None:
    router = UnifiedTaskRouter()
    direct = router.route(mir("text", "copy me"), "extract", "Copy this text")
    assert (direct.execution_mode, direct.requested_tool, direct.send_pixels) == (
        "direct",
        "return_ocr",
        False,
    )
    math = router.route(
        mir("math", "Solve x^2 - 5x + 6", math_expression_candidate=True),
        "calculate",
        "Solve this",
    )
    assert (math.execution_mode, math.requested_tool, math.send_pixels) == (
        "tool",
        "sympy_math",
        False,
    )
    table = mir(
        "table",
        "A 1 B 2",
        objects=[{"type": "table", "content": "A,1"}],
    )
    table_decision = router.route(table, "calculate", "Find the maximum")
    assert table_decision.requested_tool == "table_aggregate"
    assert router.route(mir("code", "print(a[4])"), "debug", "Why failing?").send_pixels is False
    assert router.route(mir("image"), "identify", "Identify this").send_pixels is True
    assert (
        router.route(
            mir(
                "ui",
                "Save",
            ),
            "extract",
            "Read the label",
        ).send_pixels
        is False
    )
    assert (
        router.route(mir("ui", "Save"), "identify", "What color is this button?").send_pixels
        is True
    )


def test_provider_performance_persists_and_adapts(tmp_path: Path) -> None:
    database = tmp_path / "performance.sqlite3"
    store = ProviderPerformanceStore(str(database), alpha=0.5)
    for _ in range(3):
        store.record("nvidia-fast", "text", total_ms=100, ttft_ms=40, success=True)
        store.record("gemini-slow", "text", total_ms=900, ttft_ms=400, success=True)
    fast = MockProvider(name="nvidia-fast", family="nvidia")
    slow = MockProvider(name="gemini-slow", family="gemini")
    registry = ProviderRegistry([slow, fast], store)
    assert registry.ranked(EXPERTS["text-expert"], False, task_kind="text")[0].name == "nvidia-fast"
    store.close()
    reopened = ProviderPerformanceStore(str(database))
    assert reopened.get("nvidia-fast", "text").sample_count == 3  # type: ignore[union-attr]
    reopened.close()


def test_provider_filters_capability_privacy_and_budget(tmp_path: Path) -> None:
    vision = MockProvider(name="nvidia-vision", vision=True, family="nvidia")
    text_only = MockProvider(name="local", vision=False)
    unavailable = MockProvider(name="gemini", vision=True, family="gemini")
    unavailable.daily_budget = 1
    unavailable.stats.requests_today = 1
    registry = ProviderRegistry([unavailable, text_only, vision])
    assert [p.name for p in registry.ranked(EXPERTS["vision-expert"], True)] == ["nvidia-vision"]
    assert [p.name for p in registry.ranked(EXPERTS["text-expert"], False, local_only=True)] == [
        "local"
    ]


def test_failure_ewma_penalizes_otherwise_fast_provider(tmp_path: Path) -> None:
    store = ProviderPerformanceStore(str(tmp_path / "performance.sqlite3"), alpha=0.5)
    for _ in range(3):
        store.record("nvidia-fast-failing", "text", total_ms=50, success=False)
        store.record("gemini-reliable", "text", total_ms=300, success=True)
    failing = MockProvider(name="nvidia-fast-failing", family="nvidia")
    reliable = MockProvider(name="gemini-reliable", family="gemini")
    ranked = ProviderRegistry([failing, reliable], store).ranked(
        EXPERTS["text-expert"], False, task_kind="text"
    )
    assert ranked[0].name == "gemini-reliable"
    store.close()
