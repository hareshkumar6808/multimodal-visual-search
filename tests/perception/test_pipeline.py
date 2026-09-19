from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from services.perception import (
    MIR,
    PerceptionPipeline,
    analyze_capture,
)
from services.perception.models import OCRResult
from services.perception.ocr.base import OCREngine
from services.perception.ocr.tesseract_engine import TesseractOCREngine

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class DeterministicMockOCR(OCREngine):
    def __init__(self, text: str = "def hello(): pass", conf: float = 0.95) -> None:
        self.text = text
        self.conf = conf

    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        return OCRResult(
            text=self.text,
            confidence=self.conf,
            regions=[
                {
                    "bbox": {"x": 10, "y": 10, "width": 100, "height": 20},
                    "text": self.text,
                    "confidence": self.conf,
                }
            ],
        )


def make_png(
    width: int = 400, height: int = 200, color: tuple[int, int, int] = (255, 255, 255)
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_pipeline_end_to_end_with_mock_ocr() -> None:
    mock_ocr = DeterministicMockOCR(
        text="def compute_answer(x, y):\n    return x * y",
        conf=0.96,
    )
    pipeline = PerceptionPipeline(ocr_engine=mock_ocr)

    context = {
        "active_app": "Visual Studio Code",
        "window_title": "math_utils.py",
        "selection_mode": "rectangle",
        "bounds": {"x": 100, "y": 100, "width": 400, "height": 200},
    }

    mir = pipeline.analyze(make_png(), context=context, request_id="custom-req-42")

    assert isinstance(mir, MIR)
    assert mir.request_id == "custom-req-42"
    assert mir.image.width == 400
    assert mir.image.height == 200
    assert mir.primary_modality == "code"
    assert mir.visual_required is False
    assert mir.context["active_app"] == "Visual Studio Code"
    assert mir.context["bounds"]["x"] == 100
    assert len(mir.objects) > 0
    assert mir.objects[0]["type"] == "code_block"

    # Telemetry check
    assert "total_pipeline_ms" in pipeline.last_telemetry
    assert pipeline.last_telemetry["total_pipeline_ms"] >= 0


def test_analyze_capture_convenience_function() -> None:
    # Uses real local Tesseract engine
    if not TesseractOCREngine(executable_path=TESSERACT_EXE).is_available():
        pytest.skip("Tesseract is not installed at the documented Windows path")
    img = Image.new("RGB", (500, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), "print('Stage 1 Perception')", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    mir_dict = analyze_capture(
        buf.getvalue(),
        context={"active_app": "Code", "window_title": "test.py"},
        request_id="convenience-test",
        tesseract_path=TESSERACT_EXE,
    )

    assert isinstance(mir_dict, dict)
    assert mir_dict["mir_version"] == "0.1"
    assert mir_dict["request_id"] == "convenience-test"
    assert "image" in mir_dict
    assert "aspect_ratio" not in mir_dict["image"]
    assert mir_dict["image"]["width"] == 500
    assert mir_dict["image"]["height"] == 100
    assert "modalities" in mir_dict
    assert "primary_modality" in mir_dict
    assert "ocr" in mir_dict
    assert "objects" in mir_dict
    assert "visual_required" in mir_dict
    assert "privacy_flags" in mir_dict
    assert "overall_confidence" in mir_dict
