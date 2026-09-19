from __future__ import annotations

from services.perception.models import Modalities, OCRResult
from services.perception.visual_dependency import evaluate_visual_required


def test_visual_required_clean_text() -> None:
    modalities = Modalities(text=0.85, code=0.10, table=0.0, chart=0.0, image=0.10, mixed=0.0)
    ocr = OCRResult(text="This is a clean paragraph of text.", confidence=0.92)
    assert evaluate_visual_required(modalities, "text", ocr) is False


def test_visual_required_clean_code() -> None:
    modalities = Modalities(text=0.20, code=0.92, table=0.0, chart=0.0, image=0.05, mixed=0.0)
    ocr = OCRResult(text="def add(a, b):\n    return a + b", confidence=0.95)
    assert evaluate_visual_required(modalities, "code", ocr) is False


def test_visual_required_clean_table() -> None:
    modalities = Modalities(text=0.20, code=0.0, table=0.80, chart=0.0, image=0.0, mixed=0.0)
    ocr = OCRResult(text="| Col 1 | Col 2 |\n| 100   | 200   |", confidence=0.90)
    assert evaluate_visual_required(modalities, "table", ocr) is False


def test_visual_required_chart_true() -> None:
    modalities = Modalities(text=0.20, code=0.0, table=0.0, chart=0.85, image=0.20, mixed=0.0)
    ocr = OCRResult(text="Quarterly Revenue Trend axis %", confidence=0.85)
    assert evaluate_visual_required(modalities, "chart", ocr) is True


def test_visual_required_photograph_true() -> None:
    modalities = Modalities(text=0.0, code=0.0, table=0.0, chart=0.0, image=0.95, mixed=0.0)
    ocr = OCRResult(text="", confidence=None)
    assert evaluate_visual_required(modalities, "image", ocr) is True


def test_visual_required_low_confidence_ocr_forces_true() -> None:
    modalities = Modalities(text=0.70, code=0.0, table=0.0, chart=0.0, image=0.10, mixed=0.0)
    # Low confidence OCR (below 0.60 threshold)
    ocr = OCRResult(text="garbled text 123", confidence=0.45)
    assert evaluate_visual_required(modalities, "text", ocr) is True

