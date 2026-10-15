from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import joblib

from contracts.models import MIR, ExecutionMode, Intent
from services.orchestrator.experts import EXPERTS, Expert
from services.orchestrator.routing.execution_policy import ExecutionPolicy
from services.orchestrator.routing.feature_builder import RouterFeatureBuilder
from services.orchestrator.routing.rule_router import RuleRouter

RouterSource = Literal["ml", "rule", "deterministic"]

PARENT_EXPERTS = {
    "text": "text-expert",
    "code": "code-expert",
    "math": "math-expert",
    "table": "table-expert",
    "chart": "chart-expert",
    "vision": "vision-expert",
    "ui": "ui-expert",
    "general": "general-expert",
}
FAMILY_CLASSES = {
    "text": ("text/biomedical", "text/legal", "text/finance"),
    "code": ("code/python", "code/java", "code/sql"),
    "math": ("math/algebra", "math/geometry", "math/probability"),
}
GENERAL_LEAVES = {family: f"{family}/general" for family in ("text", "code", "math")}
DETERMINISTIC_LEAVES = {
    "table": "table/general",
    "chart": "chart/general",
    "image": "vision/general",
    "ui": "ui/general",
}


@dataclass(frozen=True)
class RouteDecisionV2:
    expert: Expert
    sub_expert: str
    leaf_route: str
    routing_confidence: float
    router_source: RouterSource
    fallback_used: bool
    reason_code: str
    image_required: bool
    routing_certainty: float
    execution_mode: ExecutionMode
    send_pixels: bool
    requested_tool: str | None
    feature_build_ms: float
    classifier_ms: float
    router_total_ms: float


class UnifiedTaskRouter:
    """One learned semantic router plus deterministic families and Stage 1 fallback."""

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        *,
        enabled: bool = True,
        fallback: RuleRouter | None = None,
        feature_builder: RouterFeatureBuilder | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        default_path = Path(__file__).parents[1] / "artifacts" / "task_router_v1.joblib"
        self.artifact_path = Path(artifact_path) if artifact_path else default_path
        self.enabled = enabled
        self.fallback = fallback or RuleRouter()
        self.feature_builder = feature_builder or RouterFeatureBuilder()
        self.execution_policy = execution_policy or ExecutionPolicy()
        self._artifact: dict[str, Any] | None = None
        self.load_error: str | None = None
        if enabled:
            self._load()

    @property
    def ready(self) -> bool:
        return self._artifact is not None

    def _load(self) -> None:
        try:
            loaded = joblib.load(self.artifact_path)
            if not isinstance(loaded, dict) or "pipeline" not in loaded:
                raise ValueError("artifact does not contain a pipeline")
            self._artifact = loaded
        except Exception as exc:
            self._artifact = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def route(self, mir: MIR, intent: Intent, query: str | None = None) -> RouteDecisionV2:
        started = perf_counter()
        family = self._family(mir, query)
        if family in DETERMINISTIC_LEAVES:
            leaf = DETERMINISTIC_LEAVES[family]
            return self._finish(
                mir,
                query,
                intent,
                leaf,
                1.0,
                "deterministic",
                False,
                f"DETERMINISTIC_{leaf.replace('/', '_').upper()}",
                0.0,
                0.0,
                started,
            )

        if family not in FAMILY_CLASSES or not self.enabled or not self.ready:
            return self._rule_fallback(mir, query, intent, started, "ML_ROUTER_UNAVAILABLE")

        built = self.feature_builder.build(mir, query, intent)
        classifier_started = perf_counter()
        try:
            assert self._artifact is not None
            pipeline = self._artifact["pipeline"]
            probabilities = pipeline.predict_proba([built.text])[0]
            classes = [str(value) for value in pipeline.classes_]
            probability_by_class = dict(zip(classes, probabilities, strict=True))
            candidates = FAMILY_CLASSES[family]
            leaf = max(candidates, key=lambda value: probability_by_class.get(value, 0.0))
            confidence = float(probability_by_class.get(leaf, 0.0))
            classifier_ms = (perf_counter() - classifier_started) * 1000
            thresholds = self._artifact.get("thresholds", {})
            threshold = float(thresholds.get(family, thresholds.get("global", 0.5)))
            evidence_leaf = self._specialized_evidence_leaf(built.text, candidates)
            if evidence_leaf is None:
                return self._finish(
                    mir,
                    query,
                    intent,
                    GENERAL_LEAVES[family],
                    confidence,
                    "ml",
                    True,
                    (
                        "ML_LOW_CONFIDENCE_GENERAL_FALLBACK"
                        if confidence < threshold
                        else "ML_NO_SPECIALIZED_EVIDENCE_GENERAL_FALLBACK"
                    ),
                    built.feature_build_ms,
                    classifier_ms,
                    started,
                )
            evidence_overrode_prediction = evidence_leaf != leaf
            if evidence_overrode_prediction or confidence < threshold:
                leaf = evidence_leaf
                confidence = float(probability_by_class.get(leaf, 0.0))
            evidence_supported = self._supports_specialized_prediction(built.text, leaf)
            if confidence < threshold and not evidence_supported:
                return self._finish(
                    mir,
                    query,
                    intent,
                    GENERAL_LEAVES[family],
                    confidence,
                    "ml",
                    True,
                    "ML_LOW_CONFIDENCE_GENERAL_FALLBACK",
                    built.feature_build_ms,
                    classifier_ms,
                    started,
                )
            return self._finish(
                mir,
                query,
                intent,
                leaf,
                confidence,
                "ml",
                False,
                (
                    "ML_DOMAIN_EVIDENCE_OVERRIDE"
                    if evidence_overrode_prediction
                    else "ML_SPECIALIZED_ROUTE_EVIDENCE"
                    if evidence_supported
                    else "ML_SPECIALIZED_ROUTE"
                ),
                built.feature_build_ms,
                classifier_ms,
                started,
            )
        except Exception:
            return self._rule_fallback(mir, query, intent, started, "ML_PREDICTION_FAILED")

    def _rule_fallback(
        self, mir: MIR, query: str | None, intent: Intent, started: float, reason: str
    ) -> RouteDecisionV2:
        broad = self.fallback.route(mir, intent)
        parent = broad.expert.name.removesuffix("-expert")
        if parent == "vision":
            leaf = "vision/general"
        elif parent in PARENT_EXPERTS:
            leaf = f"{parent}/general"
        else:
            leaf = "general/general"
        return self._finish(
            mir,
            query,
            intent,
            leaf,
            broad.routing_certainty,
            "rule",
            True,
            f"{reason}:{broad.reason_code}",
            0.0,
            0.0,
            started,
        )

    def _finish(
        self,
        mir: MIR,
        query: str | None,
        intent: Intent,
        leaf: str,
        confidence: float,
        source: RouterSource,
        fallback_used: bool,
        reason: str,
        feature_build_ms: float,
        classifier_ms: float,
        started: float,
    ) -> RouteDecisionV2:
        parent, child = leaf.split("/", 1)
        expert = EXPERTS[PARENT_EXPERTS[parent]]
        policy = self.execution_policy.decide(mir, query, intent, leaf)
        total_ms = (perf_counter() - started) * 1000
        return RouteDecisionV2(
            expert=expert,
            sub_expert=child,
            leaf_route=leaf,
            routing_confidence=max(0.0, min(confidence, 1.0)),
            router_source=source,
            fallback_used=fallback_used,
            reason_code=reason,
            image_required=policy.send_pixels,
            routing_certainty=max(0.0, min(confidence, 1.0)),
            execution_mode=policy.execution_mode,
            send_pixels=policy.send_pixels,
            requested_tool=policy.requested_tool,
            feature_build_ms=feature_build_ms,
            classifier_ms=classifier_ms,
            router_total_ms=total_ms,
        )

    @classmethod
    def _family(cls, mir: MIR, query: str | None = None) -> str:
        primary = mir.primary_modality
        if primary in {"text", "mixed"}:
            evidence = f"{mir.ocr.text}\n{query or ''}".lower()
            if cls._looks_like_code(evidence):
                return "code"
            if cls._looks_like_math(evidence):
                return "math"
        if primary in {"text", "code", "math", "table", "chart", "image", "ui"}:
            return primary
        return "general"

    @staticmethod
    def _looks_like_code(evidence: str) -> bool:
        strong = (
            r"\b(?:async\s+)?def\s+[a-z_]\w*\s*\(",
            r"\bclass\s+[a-z_]\w*\s*[:(]",
            r"\bimport\s+[a-z_]\w*(?:\.[a-z_]\w*)*",
            r"\bfrom\s+[a-z_]\w*(?:\.[a-z_]\w*)*\s+import\s+[a-z_*]",
            r"\bprint\s*\(",
            r"\bpublic\s+static\s+void\s+main\s*\(",
            r"\bselect\b[\s\S]+\bfrom\b",
            r"\b(?:const|let|var)\s+[a-z_$]\w*\s*=",
        )
        return any(re.search(pattern, evidence, re.IGNORECASE) for pattern in strong)

    @staticmethod
    def _looks_like_math(evidence: str) -> bool:
        return bool(
            re.search(
                r"\b(?:algebra|angle|calculus|calculate|derivative|differentiate|equation|"
                r"factor|geometry|indefinite\s+integral|integral|integrate|matrix|probability|"
                r"quadratic|simplify|solve|theorem|triangle)\b",
                evidence,
            )
            or re.search(r"(?:\b[a-z]\b|\d+)\s*(?:\^|\*\*)\s*\d+", evidence)
        )

    @staticmethod
    def _supports_specialized_prediction(features: str, leaf: str) -> bool:
        """Accept a confident ML prediction when multiple domain terms support it."""
        terms = {
            "text/biomedical": {
                "biology",
                "biological",
                "cell",
                "cellular respiration",
                "glucose",
                "atp",
                "aerobic",
                "anaerobic",
                "fermentation",
                "gas exchange",
                "diffusion",
                "alveoli",
                "capillaries",
                "stomata",
                "photosynthesis",
                "enzyme",
                "protein",
                "dna",
                "rna",
                "organism",
                "living organism",
                "gene",
                "genes",
                "heredity",
                "evolution",
                "botany",
                "zoology",
                "ecology",
                "molecular biology",
                "patient",
                "metformin",
                "insulin",
                "clinical",
                "therapy",
                "diagnosis",
                "disease",
            },
            "text/legal": {
                "agreement",
                "contract",
                "clause",
                "lessee",
                "lessor",
                "indemnify",
                "jurisdiction",
                "statute",
                "liability",
                "plaintiff",
                "defendant",
                "arbitration",
                "governing law",
            },
            "text/finance": {
                "finance",
                "personal finance",
                "corporate finance",
                "public finance",
                "budget",
                "budgeting",
                "saving",
                "mortgage",
                "insurance",
                "retirement",
                "investment",
                "capital",
                "fiscal",
                "tax",
                "interest",
                "liquidity",
                "risk",
                "revenue",
                "profit",
                "margin",
                "earnings",
                "cash flow",
                "assets",
                "liabilities",
                "equity",
                "dividend",
                "balance sheet",
                "income statement",
                "quarterly",
            },
            "code/python": {
                "python",
                "def",
                "elif",
                "import",
                "print",
                "return",
                "none",
                "traceback",
                "self",
                "__init__",
            },
            "code/java": {
                "public static void main",
                "system.out",
                "string[]",
                "public class",
                "private class",
                "extends",
                "implements",
                "new",
            },
            "code/sql": {
                "select",
                "from",
                "where",
                "join",
                "group by",
                "order by",
                "insert into",
                "update",
                "delete from",
            },
            "math/algebra": {
                "equation",
                "polynomial",
                "factor",
                "variable",
                "solve for",
                "quadratic",
                "linear",
                "algebra",
            },
            "math/geometry": {
                "triangle",
                "circle",
                "angle",
                "radius",
                "diameter",
                "perimeter",
                "area",
                "hypotenuse",
                "parallel",
            },
            "math/probability": {
                "probability",
                "random",
                "sample space",
                "expected value",
                "coin",
                "dice",
                "outcomes",
                "permutation",
                "combination",
            },
        }.get(leaf)
        if not terms:
            return False
        normalized = features.lower()
        strong_patterns = {
            "code/python": (
                r"(?m)^\s*(?:async\s+)?def\s+[a-z_]\w*\s*\(",
                r"\bprint\s*\(",
                r"\bf?['\"][^'\"]*\{[^{}]+\}[^'\"]*['\"]",
            ),
            "code/java": (r"\bpublic\s+static\s+void\s+main\s*\(",),
            "code/sql": (r"\bselect\b[\s\S]+\bfrom\b",),
            "math/algebra": (
                r"\b[a-z]\s*(?:\^|\*\*)\s*\d+",
                r"\b[a-z]\s*[+\-*/]\s*[a-z]\s*=",
            ),
        }.get(leaf, ())
        if any(re.search(pattern, normalized) for pattern in strong_patterns):
            return True
        matches = 0
        for term in terms:
            pattern = rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])"
            matches += bool(re.search(pattern, normalized))
        return matches >= 2

    @classmethod
    def _specialized_evidence_leaf(cls, features: str, candidates: tuple[str, ...]) -> str | None:
        supported = [
            leaf for leaf in candidates if cls._supports_specialized_prediction(features, leaf)
        ]
        return supported[0] if len(supported) == 1 else None
