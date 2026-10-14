"""Conservative runtime gate between representation labels and image forwarding."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.models import MIR


@dataclass(frozen=True)
class ImagePreservationDecision:
    preserve_original: bool
    reason_code: str
    reason: str


def decide_image_preservation(mir: MIR, query: str | None) -> ImagePreservationDecision:
    """Keep capture pixels until query-specific sufficiency is independently verified.

    A content-only class, including ``text_sufficient`` and
    ``structure_sufficient``, is not evidence that the current request can be
    answered from reduced content. Query is accepted here to keep the policy
    boundary explicit; no task-sufficiency model currently evaluates it.
    """
    _ = query
    if mir.representation is None:
        return ImagePreservationDecision(
            True,
            "REPRESENTATION_MISSING_OR_LEGACY",
            "No Stage 2 representation evidence; retaining original capture pixels.",
        )
    if mir.visual_required or mir.representation.kind in {"pixels_required", "mixed"}:
        return ImagePreservationDecision(
            True,
            "PERCEPTION_REQUIRES_PIXELS",
            "Perception reports visual dependency; retaining original capture pixels.",
        )
    return ImagePreservationDecision(
        True,
        "TASK_SUFFICIENCY_UNPROVEN",
        "Content class does not establish query-specific sufficiency; retaining original pixels.",
    )
