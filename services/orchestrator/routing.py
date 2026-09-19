from dataclasses import dataclass

from contracts.models import MIR, Intent
from services.orchestrator.experts import EXPERTS, Expert


@dataclass(frozen=True)
class ExpertRoute:
    expert: Expert
    reason_code: str
    image_required: bool
    routing_certainty: float


class RuleRouter:
    """Replaceable Stage 1 rule router with explicit, auditable decisions."""

    def __init__(self, confidence_threshold: float = 0.45) -> None:
        self.confidence_threshold = confidence_threshold

    def route(self, mir: MIR, intent: Intent) -> ExpertRoute:
        if mir.overall_confidence < self.confidence_threshold:
            return ExpertRoute(
                EXPERTS["general-expert"], "LOW_PERCEPTION_CONFIDENCE", mir.visual_required, 0.55
            )

        primary = mir.primary_modality
        if primary == "code":
            return ExpertRoute(
                EXPERTS["code-expert"], f"CODE_{intent.upper()}_TEXT_SUFFICIENT", False, 0.95
            )
        if primary == "table":
            return ExpertRoute(
                EXPERTS["table-expert"],
                f"TABLE_{intent.upper()}_STRUCTURED",
                mir.visual_required,
                0.92,
            )
        if primary == "chart":
            return ExpertRoute(
                EXPERTS["chart-expert"],
                f"CHART_{intent.upper()}_{'VISUAL' if mir.visual_required else 'EXTRACTED'}",
                mir.visual_required,
                0.9,
            )
        if primary == "image":
            return ExpertRoute(EXPERTS["vision-expert"], "IMAGE_VISUAL_REQUIRED", True, 0.96)
        if primary == "text":
            return ExpertRoute(
                EXPERTS["text-expert"],
                f"TEXT_{intent.upper()}_OCR_SUFFICIENT",
                mir.visual_required,
                0.94,
            )
        if primary == "mixed" and mir.visual_required:
            return ExpertRoute(EXPERTS["vision-expert"], "MIXED_VISUAL_REQUIRED", True, 0.84)
        return ExpertRoute(EXPERTS["general-expert"], "MIXED_GENERAL_FALLBACK", False, 0.7)
