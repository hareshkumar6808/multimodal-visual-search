"""Perception pipeline coordinating image pre-flight, local OCR, profiling,
object extraction, privacy scanning, visual dependency, and canonical MIR v0.1.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from services.perception.extraction import extract_objects
from services.perception.models import MIR, OCRResult
from services.perception.ocr.base import OCREngine, OCRError
from services.perception.ocr.factory import get_default_ocr_engine
from services.perception.preflight import validate_and_extract_metadata
from services.perception.privacy import scan_privacy
from services.perception.profiling import profile_multimodal
from services.perception.visual_dependency import evaluate_visual_required

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """End-to-end perception pipeline for Multimodal Visual Search."""

    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        tesseract_path: str | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine or get_default_ocr_engine(tesseract_path=tesseract_path)
        self.last_telemetry: dict[str, int] = {}

    def analyze(
        self,
        image_bytes: bytes,
        context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> MIR:
        """Process screenshot bytes and context into a canonical MIR v0.1 model."""
        start_time = time.perf_counter()
        req_id = request_id.strip() if request_id and request_id.strip() else uuid.uuid4().hex
        ctx = dict(context or {})

        # 1. Local Image Pre-flight
        t0 = time.perf_counter()
        preflight = validate_and_extract_metadata(image_bytes)
        preflight_ms = int((time.perf_counter() - t0) * 1000)

        # 2. Local OCR Extraction
        t1 = time.perf_counter()
        ocr_degraded = 0
        try:
            ocr_result = self.ocr_engine.extract(image_bytes)
        except OCRError as exc:
            # A valid screenshot can still be understood by a vision provider. Keep the
            # request alive and let modality routing select the vision path.
            logger.warning("ocr_failed_falling_back_to_vision error=%s", exc)
            ocr_result = OCRResult(text="", confidence=None, regions=[])
            ocr_degraded = 1
        ocr_ms = int((time.perf_counter() - t1) * 1000)

        # 3. Multimodal Profiling (Deterministic Heuristic)
        t2 = time.perf_counter()
        modalities, primary_modality, overall_confidence = profile_multimodal(
            ocr_result,
            ctx,
            preflight,
        )
        profiling_ms = int((time.perf_counter() - t2) * 1000)

        # 4. Basic Layout/OCR Object Extraction
        t3 = time.perf_counter()
        objects = extract_objects(
            ocr_result,
            primary_modality,
            preflight.width,
            preflight.height,
        )
        extraction_ms = int((time.perf_counter() - t3) * 1000)

        # 5. Lightweight Privacy Scanning
        t4 = time.perf_counter()
        privacy_flags = scan_privacy(ocr_result.text, ctx)
        privacy_ms = int((time.perf_counter() - t4) * 1000)

        # 6. Visual Required Evaluation
        visual_required = evaluate_visual_required(
            modalities,
            primary_modality,
            ocr_result,
            objects,
        )

        total_ms = int((time.perf_counter() - start_time) * 1000)
        self.last_telemetry = {
            "preflight_ms": preflight_ms,
            "ocr_ms": ocr_ms,
            "ocr_degraded": ocr_degraded,
            "profiling_ms": profiling_ms,
            "extraction_ms": extraction_ms,
            "privacy_ms": privacy_ms,
            "total_pipeline_ms": total_ms,
        }

        # 7. Assemble Canonical MIR v0.1
        return MIR(
            mir_version="0.1",
            request_id=req_id,
            image=preflight.to_image_info(),
            context=ctx,
            modalities=modalities,
            primary_modality=primary_modality,
            ocr=ocr_result,
            objects=objects,
            visual_required=visual_required,
            privacy_flags=privacy_flags,
            overall_confidence=overall_confidence,
        )
