from __future__ import annotations

import io
import os
import shutil
from typing import Any

import pytesseract  # type: ignore[import-untyped]
from PIL import Image

from services.perception.models import OCRResult
from services.perception.ocr.base import OCREngine, OCRError

# Known standard installation paths on Windows
_DEFAULT_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


class TesseractOCREngine(OCREngine):
    """Real local OCR engine using Tesseract OCR via pytesseract."""

    def __init__(self, executable_path: str | None = None) -> None:
        self.executable_path = self._resolve_executable(executable_path)
        if self.executable_path:
            pytesseract.pytesseract.tesseract_cmd = self.executable_path

    @staticmethod
    def _resolve_executable(custom_path: str | None) -> str | None:
        if custom_path and os.path.exists(custom_path):
            return custom_path

        # Check PATH first
        path_in_env = shutil.which("tesseract")
        if path_in_env:
            return path_in_env

        # Check common Windows locations
        for win_path in _DEFAULT_WINDOWS_PATHS:
            if os.path.exists(win_path):
                return win_path

        return None

    def is_available(self) -> bool:
        """Check if Tesseract binary can be executed successfully."""
        if not self.executable_path or not os.path.exists(self.executable_path):
            return False
        try:
            pytesseract.pytesseract.tesseract_cmd = self.executable_path
            version = pytesseract.get_tesseract_version()
            return bool(version)
        except Exception:  # noqa: BLE001
            return False

    def extract(self, image_bytes: bytes) -> OCRResult:
        """Execute Tesseract OCR extraction and produce normalized regions and confidences."""
        if not self.is_available():
            raise OCRError(
                "Tesseract OCR executable is not available or failed to execute at: "
                f"{self.executable_path}"
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Use image_to_data to obtain structured token positions and confidences
                data: dict[str, list[Any]] = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                )

            words: list[str] = []
            regions: list[dict[str, Any]] = []
            confidences: list[float] = []

            num_boxes = len(data.get("text", []))
            for i in range(num_boxes):
                raw_text = str(data["text"][i]).strip()
                if not raw_text:
                    continue

                raw_conf = float(data["conf"][i])
                # Tesseract returns -1 when confidence is undetermined
                conf_val = max(0.0, min(1.0, raw_conf / 100.0)) if raw_conf >= 0 else None

                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])

                words.append(raw_text)
                if conf_val is not None:
                    confidences.append(conf_val)

                regions.append(
                    {
                        "bbox": {"x": left, "y": top, "width": width, "height": height},
                        "text": raw_text,
                        "confidence": conf_val,
                    }
                )

            full_text = " ".join(words)
            overall_confidence: float | None = None
            if confidences:
                overall_confidence = round(sum(confidences) / len(confidences), 4)

            return OCRResult(
                text=full_text,
                confidence=overall_confidence,
                regions=regions,
            )

        except Exception as exc:
            raise OCRError(f"Tesseract OCR extraction failed: {exc}") from exc
