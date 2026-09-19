from __future__ import annotations

from abc import ABC, abstractmethod

from services.perception.models import OCRResult


class OCRError(RuntimeError):
    """Raised when an OCR extraction operation fails."""

    pass


class OCREngineUnavailableError(OCRError):
    """Raised when no requested or default OCR engine is installed or runnable."""

    pass


class OCREngine(ABC):
    """Abstract interface for local optical character recognition engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying engine binary/library is installed and accessible."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, image_bytes: bytes) -> OCRResult:
        """Run character recognition on raw image bytes and return normalized OCRResult."""
        raise NotImplementedError

