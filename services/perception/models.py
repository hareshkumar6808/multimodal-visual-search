"""Compatibility imports for the canonical Stage 1 contracts.

All shared MIR definitions live in :mod:`contracts.models`.
"""

from contracts.models import (
    MIR,
    ExtractedObject,
    ImageInfo,
    Modalities,
    Modality,
    OCRRegion,
    OCRResult,
)

__all__ = [
    "ExtractedObject",
    "ImageInfo",
    "MIR",
    "Modalities",
    "Modality",
    "OCRRegion",
    "OCRResult",
]
