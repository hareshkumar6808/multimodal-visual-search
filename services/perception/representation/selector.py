"""Representation selection interface and conservative local fallback.

Learned selectors and the pipeline fallback share this MIR contract. The
heuristic implementation remains deterministic and preserves pixels when
evidence is uncertain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts.models import RepresentationInfo, RepresentationKind, RepresentationScores
from services.perception.models import Modalities, OCRResult


class RepresentationSelector(ABC):
    """Select the least lossy representation supported by perception evidence."""

    @abstractmethod
    def select(
        self,
        modalities: Modalities,
        primary_modality: str,
        overall_confidence: float,
        ocr: OCRResult,
        visual_required: bool,
        image_bytes: bytes,
    ) -> RepresentationInfo:
        """Return a validated selection, including confidence and provenance."""


class HeuristicRepresentationSelector(RepresentationSelector):
    """OCR/layout heuristic that retains pixels whenever evidence is uncertain."""

    def select(
        self,
        modalities: Modalities,
        primary_modality: str,
        overall_confidence: float,
        ocr: OCRResult,
        visual_required: bool,
        image_bytes: bytes,
    ) -> RepresentationInfo:
        ocr_confidence = ocr.confidence or 0.0
        has_text = bool(ocr.text.strip())
        pixel_prior = max(
            modalities.image,
            modalities.chart,
            0.85 if visual_required else 0.0,
            1.0 - ocr_confidence,
        )
        structural_evidence = primary_modality in {"code", "table"}
        visual_evidence = max(modalities.image, modalities.chart) >= 0.35 or visual_required
        confidence = min(1.0, max(0.0, overall_confidence))

        scores = RepresentationScores(
            text_sufficient=min(1.0, ocr_confidence * (0.9 if has_text else 0.0)),
            structure_sufficient=min(
                1.0, ocr_confidence * (0.95 if structural_evidence else 0.0)
            ),
            pixels_required=min(1.0, max(pixel_prior, 0.0)),
            mixed=min(
                1.0,
                max(
                    modalities.mixed,
                    min(ocr_confidence if has_text else 0.0, pixel_prior),
                ),
            ),
        )

        # Low OCR/perception confidence and any strong visual evidence must keep
        # the original pixels available. Never discard detail on a weak heuristic.
        if confidence < 0.55 or (has_text and ocr_confidence < 0.55):
            kind: RepresentationKind = "pixels_required"
        elif modalities.mixed >= 0.50 or (visual_evidence and has_text):
            kind = "mixed"
        elif visual_evidence or not has_text:
            kind = "pixels_required"
        elif structural_evidence:
            kind = "structure_sufficient"
        elif has_text and ocr_confidence >= 0.65:
            kind = "text_sufficient"
        else:
            kind = "pixels_required"

        return RepresentationInfo(
            kind=kind,
            scores=scores,
            confidence=confidence,
            pixel_dependency_prior=min(1.0, max(0.0, pixel_prior)),
            source="heuristic",
            model=None,
            available_representations=["pixels", "ocr_text", "layout"],
            local_capabilities={"ocr": has_text, "layout": bool(ocr.regions), "ml_model": False},
        )
