from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from services.orchestrator.execution.engine import ExecutionEngine, RouteDecisionV2
from services.orchestrator.execution.registry import CANONICAL_LEAVES, EXPERT_REGISTRY
from services.orchestrator.execution.tools import (
    math_tool,
    python_static_tool,
    sql_static_tool,
    table_tool,
)
from services.orchestrator.experts import EXPERTS
from services.orchestrator.providers.base import Provider, ProviderError
from services.orchestrator.providers.registry import (
    ProviderRegistry,
    StreamInterruptedError,
)


def test_every_canonical_leaf_resolves() -> None:
    assert {EXPERT_REGISTRY.resolve(leaf).canonical_leaf for leaf in CANONICAL_LEAVES} == set(
        CANONICAL_LEAVES
    )


@pytest.mark.parametrize(
    ("operation", "expression", "expected"),
    [
        ("arithmetic", "17 * 23", "391"),
        ("simplify", "x + x", "2*x"),
        ("factor", "x**2 - 1", "(x - 1)*(x + 1)"),
        ("expand", "(x + 1)**2", "x**2 + 2*x + 1"),
        ("differentiate", "x**3", "3*x**2"),
        ("integrate", "2*x", "x**2"),
    ],
)
def test_math_operations(operation: str, expression: str, expected: str) -> None:
    result = math_tool(operation, expression)
    assert result.success
    assert result.result == expected


def test_math_solves_quadratic() -> None:
    result = math_tool("solve", "x**2 - 5*x + 6 = 0")
    assert result.success
    assert result.result == "[2, 3]"


def test_math_accepts_caret_and_implicit_coefficient() -> None:
    result = math_tool("solve", "x^2 - 5x + 6 = 0")
    assert result.success
    assert result.result == "[2, 3]"


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "x.__class__", "lambda: 1", "[x for x in [1]]", "open('x')"],
)
def test_math_rejects_unsafe_ast(expression: str) -> None:
    assert not math_tool("simplify", expression).success


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("count", 3), ("sum", 6.0), ("mean", 2.0), ("average", 2.0), ("min", 1.0), ("max", 3.0)],
)
def test_table_operations(operation: str, expected: float) -> None:
    result = table_tool(operation, [1, 2, 3])
    assert result.success and result.result == expected


def test_table_rejects_non_numeric_values() -> None:
    result = table_tool("sum", [1, "not-a-number"])
    assert not result.success


def test_python_static_valid_invalid_and_never_executes() -> None:
    valid = python_static_tool("answer = 1 + 2")
    invalid = python_static_tool("if:")
    dangerous = python_static_tool("raise RuntimeError('must not execute')")
    assert valid.result["valid"] is True
    assert invalid.result["valid"] is False and invalid.result["line"] == 1
    assert dangerous.success and dangerous.metadata["executed"] is False


def test_sql_static_parse_does_not_execute() -> None:
    result = sql_static_tool("SELECT id FROM users WHERE active = 1")
    assert result.success
    assert result.result[0]["type"] == "SELECT"
    assert result.metadata["executed"] is False
    assert "not database-specific" in result.metadata["validation_scope"]


def test_direct_ocr_has_zero_api_calls() -> None:
    engine = ExecutionEngine()
    decision = RouteDecisionV2(
        expert="text",
        leaf_route="text/general",
        execution_mode="direct",
        requested_tool="return_ocr",
    )
    _, policy = engine.adapt(decision, "text-expert")
    result = engine.execute_local(decision, policy, ocr_text="Exact OCR")
    assert result.answer == "Exact OCR"
    assert result.api_calls == 0
    assert "direct_ms" in result.metrics


def test_failed_deterministic_tool_falls_through_to_provider() -> None:
    engine = ExecutionEngine()
    decision = RouteDecisionV2(
        expert="math",
        leaf_route="math/general",
        execution_mode="tool",
        requested_tool="math",
    )
    _, policy = engine.adapt(decision, "math-expert")
    result = engine.execute_local(
        decision,
        policy,
        ocr_text="noisy OCR",
        tool_input={"operation": "simplify", "expression": "Problem: [oe +4x —5) dx"},
    )
    assert result.kind == "provider"
    assert result.tool_result and result.tool_result["success"] is False


def test_stage1_experts_map_to_general_leaves() -> None:
    engine = ExecutionEngine()
    for expert, expected in {
        "text-expert": "text/general",
        "code-expert": "code/general",
        "table-expert": "table/general",
        "chart-expert": "chart/general",
        "vision-expert": "vision/general",
        "general-expert": "general/general",
    }.items():
        _, policy = engine.adapt(None, expert)
        assert policy.canonical_leaf == expected


class StreamingProvider(Provider):
    def __init__(self, name: str, chunks: list[str], fail_after: int | None = None) -> None:
        super().__init__(family=name)
        self.name = name
        self.chunks = chunks
        self.fail_after = fail_after

    @property
    def configured(self) -> bool:
        return True

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"text"})

    async def _generate(self, prompt: str, image_bytes: bytes | None, mime_type: str) -> str:
        return "".join(self.chunks)

    async def stream_generate(
        self, prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/png"
    ) -> AsyncIterator[str]:
        for index, chunk in enumerate(self.chunks):
            if self.fail_after == index:
                raise ProviderError("planned stream failure")
            yield chunk
        if self.fail_after == len(self.chunks):
            raise ProviderError("planned stream failure")


@pytest.mark.asyncio
async def test_stream_emits_start_progress_multiple_deltas_done_and_ttft() -> None:
    registry = ProviderRegistry([StreamingProvider("nvidia", ["one ", "two"])])
    events = [
        event
        async for event in registry.stream_with_fallback(
            EXPERTS["text-expert"], "prompt", False, b"", "image/png"
        )
    ]
    assert [event.type for event in events] == ["progress", "delta", "delta", "done"]
    assert events[1].metrics and events[1].metrics["provider_ttft_ms"] >= 0
    assert events[-1].metrics and events[-1].metrics["stream_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_stream_falls_back_only_before_first_delta() -> None:
    registry = ProviderRegistry(
        [
            StreamingProvider("local", [], fail_after=0),
            StreamingProvider("gemini", ["fallback"]),
        ]
    )
    events = [
        event
        async for event in registry.stream_with_fallback(
            EXPERTS["text-expert"], "prompt", False, b"", "image/png"
        )
    ]
    assert [event.text for event in events if event.type == "delta"] == ["fallback"]


@pytest.mark.asyncio
async def test_stream_falls_back_when_provider_returns_no_text() -> None:
    registry = ProviderRegistry(
        [StreamingProvider("nvidia", []), StreamingProvider("local", ["local answer"])]
    )
    events = [
        event
        async for event in registry.stream_with_fallback(
            EXPERTS["text-expert"], "prompt", False, b"", "image/png"
        )
    ]
    assert [event.text for event in events if event.type == "delta"] == ["local answer"]


@pytest.mark.asyncio
async def test_stream_does_not_fallback_after_partial_output() -> None:
    registry = ProviderRegistry(
        [
            StreamingProvider("local", ["partial"], fail_after=1),
            StreamingProvider("gemini", ["duplicate"]),
        ]
    )
    with pytest.raises(StreamInterruptedError):
        _ = [
            event
            async for event in registry.stream_with_fallback(
                EXPERTS["text-expert"], "prompt", False, b"", "image/png"
            )
        ]
