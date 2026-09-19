from __future__ import annotations

from services.perception.ocr.base import OCREngine, OCREngineUnavailableError, OCRError
from services.perception.ocr.factory import get_default_ocr_engine
from services.perception.ocr.tesseract_engine import TesseractOCREngine
from services.perception.ocr.windows_media_engine import WindowsMediaOCREngine

__all__ = [
    "OCREngine",
    "OCRError",
    "OCREngineUnavailableError",
    "TesseractOCREngine",
    "WindowsMediaOCREngine",
    "get_default_ocr_engine",
]
