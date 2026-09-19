from __future__ import annotations

from services.perception.models import OCRResult
from services.perception.ocr.base import OCREngine, OCREngineUnavailableError


class WindowsMediaOCREngine(OCREngine):
    """Optional native Windows 10/11 OCR engine using Windows.Media.Ocr WinRT runtime.

    Only active when appropriate Python WinRT bindings (such as winsdk or winrt)
    are installed in the Python environment. Does NOT generate fake or simulated OCR text.
    """

    def __init__(self) -> None:
        self._winrt_available = self._check_winrt()

    @staticmethod
    def _check_winrt() -> bool:
        try:
            # Check for winsdk or winrt bindings
            import importlib.util

            if importlib.util.find_spec("winsdk") is not None:
                from winsdk.windows.media.ocr import OcrEngine  # type: ignore[import-not-found]

                return OcrEngine.is_language_supported("en-US")  # type: ignore[no-any-return]
            if importlib.util.find_spec("winrt") is not None:
                from winrt.windows.media import ocr  # type: ignore[import-not-found]

                return ocr.OcrEngine.is_language_supported("en-US")  # type: ignore[no-any-return]
            return False
        except Exception:  # noqa: BLE001
            return False

    def is_available(self) -> bool:
        """Return True only if Windows WinRT OCR bindings are verified and functional."""
        return self._winrt_available

    def extract(self, image_bytes: bytes) -> OCRResult:
        if not self.is_available():
            raise OCREngineUnavailableError(
                "Windows.Media.Ocr is not available in this Python environment because "
                "Python WinRT bindings ('winsdk' or 'winrt') are not installed. "
                "Use TesseractOCREngine as the verified local OCR engine."
            )

        raise NotImplementedError("Windows.Media.Ocr extraction is not configured.")
