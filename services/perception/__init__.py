"""Context + Multimodal Perception Subsystem.

Provides local deterministic pre-flight, real OCR, heuristic multimodal profiling,
OCR/layout-based object extraction, lightweight privacy scanning, and canonical
Stage 1 MIR v0.1 generation.
"""

from __future__ import annotations

from typing import Any

from services.perception.models import (
    MIR,
    ExtractedObject,
    ImageInfo,
    Modalities,
    Modality,
    OCRRegion,
    OCRResult,
)
from services.perception.ocr import (
    OCREngine,
    OCREngineUnavailableError,
    OCRError,
    TesseractOCREngine,
    WindowsMediaOCREngine,
    get_default_ocr_engine,
)
from services.perception.pipeline import PerceptionPipeline
from services.perception.preflight import (
    ImagePreflightResult,
    PreflightError,
    validate_and_extract_metadata,
)

# Cached pipeline instance to avoid repeatedly allocating resources
_DEFAULT_PIPELINE: PerceptionPipeline | None = None


def get_pipeline(tesseract_path: str | None = None) -> PerceptionPipeline:
    """Return the cached default perception pipeline instance."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None or tesseract_path is not None:
        _DEFAULT_PIPELINE = PerceptionPipeline(tesseract_path=tesseract_path)
    return _DEFAULT_PIPELINE


def analyze_capture(
    image_bytes: bytes,
    context: dict[str, Any] | None = None,
    request_id: str | None = None,
    tesseract_path: str | None = None,
) -> dict[str, Any]:
    """Primary public entry point for the perception subsystem.

    Intended usage by orchestration layer:
        from services.perception import analyze_capture
        mir_dict = analyze_capture(image_bytes, context)

    Returns:
        JSON-serializable dictionary representation of canonical MIR v0.1.
    """
    pipeline = get_pipeline(tesseract_path=tesseract_path)
    mir_model = pipeline.analyze(image_bytes, context=context, request_id=request_id)
    return mir_model.model_dump(mode="json")


__all__ = [
    "ExtractedObject",
    "ImageInfo",
    "ImagePreflightResult",
    "MIR",
    "Modalities",
    "Modality",
    "OCREngine",
    "OCREngineUnavailableError",
    "OCRError",
    "OCRRegion",
    "OCRResult",
    "PerceptionPipeline",
    "PreflightError",
    "TesseractOCREngine",
    "WindowsMediaOCREngine",
    "analyze_capture",
    "get_default_ocr_engine",
    "get_pipeline",
    "validate_and_extract_metadata",
]

