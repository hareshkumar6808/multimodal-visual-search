from __future__ import annotations

import pytest

from services.perception.models import OCRResult
from services.perception.ocr.base import OCREngine, OCREngineUnavailableError
from services.perception.ocr.windows_media_engine import WindowsMediaOCREngine


class MockOCREngine(OCREngine):
    """Deterministic mock OCR engine for isolated unit tests."""

    def __init__(self, result: OCRResult | None = None) -> None:
        self.result = result or OCRResult(
            text="print('hello world')",
            confidence=0.95,
            regions=[
                {"bbox": {"x": 10, "y": 10, "width": 50, "height": 20}, "text": "print", "confidence": 0.96},
                {"bbox": {"x": 65, "y": 10, "width": 120, "height": 20}, "text": "('hello world')", "confidence": 0.94},
            ],
        )

    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        return self.result


def test_mock_ocr_returns_expected_structure() -> None:
    engine = MockOCREngine()
    result = engine.extract(b"dummy")

    assert result.text == "print('hello world')"
    assert result.confidence == 0.95
    assert len(result.regions) == 2
    assert result.regions[0]["text"] == "print"
    assert result.regions[0]["bbox"]["width"] == 50


def test_mock_ocr_empty_text_handled() -> None:
    engine = MockOCREngine(OCRResult(text="", confidence=None, regions=[]))
    result = engine.extract(b"dummy")

    assert result.text == ""
    assert result.confidence is None
    assert result.regions == []


def test_windows_media_ocr_gracefully_reports_unavailable_without_winrt() -> None:
    engine = WindowsMediaOCREngine()
    # In standard Python environments without WinRT bindings, it should report False
    assert engine.is_available() is False
    with pytest.raises(OCREngineUnavailableError, match="Windows.Media.Ocr"):
        engine.extract(b"dummy")

