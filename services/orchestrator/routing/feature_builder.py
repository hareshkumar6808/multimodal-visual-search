from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from contracts.models import MIR, Intent


@dataclass(frozen=True)
class BuiltFeatures:
    text: str
    feature_build_ms: float


class RouterFeatureBuilder:
    """Builds leak-free textual features from optional MIR v0.1/v0.2 fields."""

    def build(self, mir: MIR, query: str | None, intent: Intent) -> BuiltFeatures:
        started = perf_counter()
        extra = mir.model_extra or {}
        representation_mode = str(
            (mir.representation.kind if mir.representation else None)
            or extra.get("representation_mode")
            or "unknown"
        )
        app = str(mir.context.get("active_app") or mir.context.get("application") or "unknown")
        boolean_tokens = [
            f"has_ocr_{bool(mir.ocr.text)}",
            f"has_table_{self._has_object(mir, 'table')}",
            f"has_math_{bool(extra.get('math_expression_candidate'))}",
            f"has_code_analysis_{bool(extra.get('code_static_analysis_candidate'))}",
            f"raw_pixels_{bool(mir.image)}",
        ]
        metadata = " ".join(
            [
                f"intent_{intent}",
                f"modality_{mir.primary_modality}",
                f"representation_{representation_mode}",
                f"application_{app.lower().replace(' ', '_')[:80]}",
                *boolean_tokens,
            ]
        )
        content = mir.ocr.text[:12_000]
        question = (query or "").strip()[:2_000]
        text = f"{metadata}\nQUESTION: {question}\nCONTENT: {content}".strip()
        return BuiltFeatures(text=text, feature_build_ms=(perf_counter() - started) * 1000)

    @staticmethod
    def _has_object(mir: MIR, kind: str) -> bool:
        return any(str(obj.get("type", "")).startswith(kind) for obj in mir.objects)
