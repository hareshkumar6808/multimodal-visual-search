from __future__ import annotations

import io

from PIL import Image

from contracts.models import MIR, RepresentationInfo
from services.perception import PerceptionPipeline
from services.perception.models import Modalities, OCRResult
from services.perception.ocr.base import OCREngine
from services.perception.representation import HeuristicRepresentationSelector


def _select(
    kind: str,
    *,
    primary: str = "text",
    confidence: float = 0.9,
    ocr: OCRResult | None = None,
    visual: bool = False,
) -> RepresentationInfo:
    scores = {
        "text": 0.9 if primary == "text" else 0.1,
        "code": 0.9 if primary == "code" else 0.0,
        "table": 0.9 if primary == "table" else 0.0,
        "chart": 0.9 if primary == "chart" else 0.0,
        "image": 0.9 if primary == "image" else 0.0,
        "mixed": 0.8 if primary == "mixed" else 0.0,
    }
    result = HeuristicRepresentationSelector().select(
        Modalities(**scores),
        primary,
        confidence,
        ocr or OCRResult(text="Readable content", confidence=0.92),
        visual,
    )
    assert result.kind == kind
    return result


def test_selector_emits_text_sufficient() -> None:
    result = _select("text_sufficient")
    assert 0 <= result.confidence <= 1
    assert 0 <= result.pixel_dependency_prior <= 1
    assert result.source == "heuristic"
    assert result.model is None
    assert result.available_representations == ["pixels", "ocr_text", "layout"]
    assert all(0 <= value <= 1 for value in result.scores.model_dump().values())


def test_selector_emits_structure_sufficient() -> None:
    _select("structure_sufficient", primary="table")


def test_selector_emits_pixels_required_for_visual_content() -> None:
    _select(
        "pixels_required",
        primary="image",
        ocr=OCRResult(text="", confidence=None),
        visual=True,
    )


def test_selector_emits_mixed_when_text_and_visual_evidence_coexist() -> None:
    _select("mixed", primary="mixed", visual=True)


def test_low_confidence_conservatively_preserves_pixels() -> None:
    result = _select(
        "pixels_required",
        confidence=0.4,
        ocr=OCRResult(text="uncertain", confidence=0.4),
    )
    assert result.pixel_dependency_prior >= 0.6


class StaticOCR(OCREngine):
    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        return OCRResult(text="Readable paragraph content", confidence=0.9, regions=[])


class BrokenSelector:
    def select(self, *args: object, **kwargs: object) -> RepresentationInfo:
        raise RuntimeError("optional model unavailable")


def test_unavailable_model_uses_heuristic_and_emits_v02() -> None:
    image = Image.new("RGB", (20, 20), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    pipeline = PerceptionPipeline(ocr_engine=StaticOCR(), representation_selector=BrokenSelector())  # type: ignore[arg-type]

    mir = pipeline.analyze(stream.getvalue(), request_id="fallback")

    assert mir.mir_version == "0.2"
    assert mir.representation is not None
    assert mir.representation.source == "heuristic"
    assert pipeline.last_telemetry["representation_ms"] >= 0


def test_mir_v01_payload_remains_valid_and_v02_serializes_representation() -> None:
    digest = "a" * 64
    legacy = MIR.model_validate(
        {
            "mir_version": "0.1",
            "request_id": "legacy",
            "image": {"sha256": digest, "width": 1, "height": 1},
            "modalities": {"text": 0.1},
            "primary_modality": "text",
            "overall_confidence": 0.5,
        }
    )
    current = MIR(
        mir_version="0.2",
        request_id="current",
        image={"sha256": digest, "width": 1, "height": 1},
        modalities=Modalities(text=0.8),
        primary_modality="text",
        overall_confidence=0.8,
        representation=_select("text_sufficient"),
    )

    assert legacy.mir_version == "0.1"
    assert current.model_dump(mode="json")["representation"]["kind"] == "text_sufficient"
