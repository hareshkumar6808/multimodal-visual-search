from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from services.perception.ocr.tesseract_engine import TesseractOCREngine


def make_synthetic_text_image(text: str = "Multimodal Visual Search 2026") -> bytes:
    """Render high-contrast synthetic image containing known clear text."""
    img = Image.new("RGB", (600, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Default bitmap font is always available in PIL
    draw.text((30, 45), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tesseract_engine() -> TesseractOCREngine:
    engine = TesseractOCREngine()
    if not engine.is_available():
        pytest.skip("Tesseract is not available through TESSERACT_CMD, PATH, or a standard path")
    return engine


def test_real_tesseract_engine_is_available(tesseract_engine: TesseractOCREngine) -> None:
    assert tesseract_engine.is_available() is True


def test_real_tesseract_extracts_rendered_text(tesseract_engine: TesseractOCREngine) -> None:
    image_bytes = make_synthetic_text_image("Multimodal Visual Search 2026")
    result = tesseract_engine.extract(image_bytes)

    assert result.text != ""
    assert "Multimodal" in result.text or "Visual" in result.text or "Search" in result.text
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.regions) > 0

    first_region = result.regions[0]
    assert "bbox" in first_region
    assert "text" in first_region
    assert first_region["bbox"]["width"] > 0
    assert first_region["bbox"]["height"] > 0


def test_real_tesseract_blank_image_yields_empty_result(
    tesseract_engine: TesseractOCREngine,
) -> None:
    # Blank white image with no text
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = tesseract_engine.extract(buf.getvalue())
    assert result.text.strip() == ""
    assert len(result.regions) == 0
