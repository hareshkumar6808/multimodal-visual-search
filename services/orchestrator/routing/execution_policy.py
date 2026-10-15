from __future__ import annotations

import re
from dataclasses import dataclass

from contracts.models import MIR, ExecutionMode, Intent


@dataclass(frozen=True)
class ExecutionPolicyResult:
    execution_mode: ExecutionMode
    send_pixels: bool
    requested_tool: str | None
    reason_code: str


class ExecutionPolicy:
    """Transparent cheapest-valid execution and query-aware pixel policy."""

    _visual_terms = frozenset(
        {"color", "colour", "position", "layout", "shape", "look like", "where", "button"}
    )

    def decide(
        self, mir: MIR, query: str | None, intent: Intent, leaf_route: str
    ) -> ExecutionPolicyResult:
        normalized = " ".join((query or "").lower().split())
        has_ocr = bool(mir.ocr.text.strip())
        has_table = self._has_object(mir, "table")
        extra = mir.model_extra or {}
        math_candidate = bool(extra.get("math_expression_candidate")) or self._looks_math(
            mir.ocr.text
        )

        if (
            has_ocr
            and intent == "extract"
            and any(term in normalized for term in ("copy", "transcribe", "extract", "read"))
        ):
            return ExecutionPolicyResult("direct", False, "return_ocr", "DIRECT_OCR_SUFFICIENT")

        if (
            leaf_route.startswith("math/")
            and math_candidate
            and intent
            in {
                "calculate",
                "general",
                "explain",
            }
        ):
            return ExecutionPolicyResult("tool", False, "sympy_math", "TOOL_MATH_CANDIDATE")

        if leaf_route == "table/general" and has_table and intent in {"calculate", "extract"}:
            return ExecutionPolicyResult("tool", False, "table_aggregate", "TOOL_STRUCTURED_TABLE")

        send_pixels = self.should_send_pixels(mir, normalized, leaf_route)
        if send_pixels:
            return ExecutionPolicyResult("cloud_vision", True, None, "PIXELS_REQUIRED")

        complex_route = leaf_route.startswith(("code/", "math/")) or intent in {
            "debug",
            "calculate",
            "compare",
            "search",
        }
        if complex_route:
            return ExecutionPolicyResult("cloud_text", False, None, "CLOUD_REASONING_PREFERRED")
        return ExecutionPolicyResult("cloud_text", False, None, "CLOUD_GENERATION_PREFERRED")

    def should_send_pixels(self, mir: MIR, query: str, leaf_route: str) -> bool:
        extra = mir.model_extra or {}
        prior = (
            mir.representation.pixel_dependency_prior
            if mir.representation
            else extra.get("pixel_dependency_prior")
        )
        representation = str(
            (mir.representation.kind if mir.representation else None)
            or extra.get("representation_mode")
            or ""
        )
        ocr_confidence = mir.ocr.confidence or 0.0
        asks_visual = any(term in query for term in self._visual_terms)

        # Perception has already determined that the reduced OCR/structure is
        # insufficient. Never let a downstream representation label discard
        # the only remaining evidence: the original screenshot.
        if mir.visual_required:
            return True
        if leaf_route in {"vision/general", "chart/general"}:
            return True
        if leaf_route == "ui/general":
            if asks_visual:
                return True
            return not (mir.ocr.text and ocr_confidence >= 0.75)
        if asks_visual and mir.primary_modality in {"image", "ui", "mixed"}:
            return True
        if representation in {"pixels", "raw_pixels", "pixels_required"} and isinstance(
            prior, (int, float)
        ):
            return float(prior) >= 0.65
        return False

    @staticmethod
    def _has_object(mir: MIR, kind: str) -> bool:
        return any(str(obj.get("type", "")).startswith(kind) for obj in mir.objects)

    @staticmethod
    def _looks_math(text: str) -> bool:
        return bool(re.search(r"(?:\d\s*[+*/=^]|[a-zA-Z]\s*\^\s*\d|\bsolve\b)", text))
