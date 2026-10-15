from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from services.orchestrator.execution.registry import EXPERT_REGISTRY, ExpertPolicy
from services.orchestrator.execution.tools import (
    math_tool,
    python_static_tool,
    sql_static_tool,
    table_tool,
)

ExecutionMode = Literal["direct", "tool", "local_llm", "cloud_text", "cloud_vision"]


@dataclass(frozen=True)
class RouteDecisionV2:
    expert: str
    sub_expert: str = "general"
    leaf_route: str | None = None
    routing_confidence: float = 1.0
    execution_mode: ExecutionMode = "cloud_text"
    requested_tool: str | None = None
    send_pixels: bool = False
    ranked_provider_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    kind: Literal["direct", "tool", "provider"]
    answer: str | None
    policy: ExpertPolicy
    tool_result: dict[str, Any] | None = None
    api_calls: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class ExecutionEngine:
    def adapt(
        self, decision: RouteDecisionV2 | None, stage1_expert: str
    ) -> tuple[RouteDecisionV2, ExpertPolicy]:
        if decision is None:
            policy = EXPERT_REGISTRY.adapt_stage1(stage1_expert)
            return RouteDecisionV2(stage1_expert, leaf_route=policy.canonical_leaf), policy
        leaf = decision.leaf_route or f"{decision.expert}/{decision.sub_expert}"
        try:
            policy = EXPERT_REGISTRY.resolve(leaf)
        except KeyError:
            policy = EXPERT_REGISTRY.adapt_stage1(decision.expert)
        return decision, policy

    def execute_local(
        self,
        decision: RouteDecisionV2,
        policy: ExpertPolicy,
        *,
        ocr_text: str,
        tool_input: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        started = perf_counter()
        if decision.execution_mode == "direct":
            if decision.requested_tool != "return_ocr":
                raise ValueError("unsupported direct operation")
            elapsed = (perf_counter() - started) * 1000
            return ExecutionResult(
                "direct",
                ocr_text,
                policy,
                api_calls=0,
                metrics={"direct_ms": elapsed, "time_to_first_useful_output_ms": elapsed},
            )
        if decision.execution_mode != "tool" or not decision.requested_tool:
            return ExecutionResult("provider", None, policy)
        args = tool_input or {}
        tools = {"math", "table", "python_static", "sql_static"}
        if (
            decision.requested_tool not in policy.tool_policy
            or decision.requested_tool not in tools
        ):
            raise ValueError(
                f"tool {decision.requested_tool} is not allowed for {policy.canonical_leaf}"
            )
        if decision.requested_tool == "math":
            result = math_tool(**args)
        elif decision.requested_tool == "table":
            result = table_tool(**args)
        elif decision.requested_tool == "python_static":
            result = python_static_tool(**args)
        else:
            result = sql_static_tool(**args)
        if not result.success:
            return ExecutionResult(
                "provider",
                None,
                policy,
                result.model_dump(),
                0,
                {"tool_ms": result.duration_ms},
            )
        answer = str(result.result) if result.success else None
        return ExecutionResult(
            "tool",
            answer,
            policy,
            result.model_dump(),
            0,
            {
                "tool_ms": result.duration_ms,
                "time_to_first_useful_output_ms": (perf_counter() - started) * 1000,
            },
        )
