"""Stage 1 deterministic heuristic for visual_required.

NOTE: This is a basic boolean heuristic, not a trained classifier or the full
future Visual Dependency Score. It determines whether downstream processing
strictly requires original image pixels or if extracted text/code/table is sufficient.
"""

from __future__ import annotations

from typing import Any

from services.perception.models import Modalities, Modality, OCRResult


def evaluate_visual_required(
    modalities: Modalities,
    primary_modality: Modality,
    ocr_result: OCRResult,
    objects: list[dict[str, Any]] | None = None,
) -> bool:
    """Evaluate whether the screenshot requires raw visual pixels for downstream reasoning.

    Returns:
        False if clean text, code, or structured tabular representation is sufficient.
        True if the image contains photographs, charts, visual elements, or low-confidence OCR.
    """
    ocr_conf = ocr_result.confidence if ocr_result.confidence is not None else 0.0
    text = (ocr_result.text or "").strip()

    # 1. Pure or heavy image/photography always requires visual pixels
    if primary_modality == "image" or modalities.image >= 0.50:
        return True

    # 2. Charts and diagrams require visual inspection of graphics and layout
    if primary_modality == "chart" or modalities.chart >= 0.45:
        return True

    # 3. Empty OCR text or very low OCR confidence requires visual fallback
    if not text or ocr_conf < 0.60:
        return True

    # 4. Mixed modality with significant visual or chart component
    if primary_modality == "mixed" and (modalities.chart >= 0.35 or modalities.image >= 0.35):
        return True

    # 5. Clean code screenshots with reliable OCR
    if primary_modality == "code" and ocr_conf >= 0.60:
        return False

    # 6. Clean plain text screenshots with reliable OCR
    if primary_modality == "text" and ocr_conf >= 0.60:
        return False

    # 7. Clean tabular data with reliable OCR
    return not (primary_modality == "table" and ocr_conf >= 0.65)
