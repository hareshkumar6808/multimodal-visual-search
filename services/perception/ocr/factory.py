from __future__ import annotations

from services.perception.ocr.base import OCREngine, OCREngineUnavailableError
from services.perception.ocr.tesseract_engine import TesseractOCREngine
from services.perception.ocr.windows_media_engine import WindowsMediaOCREngine


def get_default_ocr_engine(
    tesseract_path: str | None = None,
    prefer_windows: bool = False,
) -> OCREngine:
    """Detect and return the active real local OCR engine."""
    if prefer_windows:
        win_engine = WindowsMediaOCREngine()
        if win_engine.is_available():
            return win_engine

    tess_engine = TesseractOCREngine(executable_path=tesseract_path)
    if tess_engine.is_available():
        return tess_engine

    raise OCREngineUnavailableError(
        "No real local OCR engine is available. Please ensure Tesseract OCR is installed "
        "at 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' or added to your system PATH."
    )

