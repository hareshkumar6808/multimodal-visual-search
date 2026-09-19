from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["text", "code", "table", "chart", "image", "mixed"]


class ImageInfo(BaseModel):
    """Canonical MIR image representation.

    Note: aspect_ratio is calculated internally in pre-flight but omitted here
    to strictly adhere to the canonical Stage 1 MIR v0.1 contract.
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(min_length=64, max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Modalities(BaseModel):
    """Multi-label modality scores in the closed interval [0.0, 1.0]."""

    model_config = ConfigDict(extra="forbid")

    text: float = Field(default=0.0, ge=0.0, le=1.0)
    code: float = Field(default=0.0, ge=0.0, le=1.0)
    table: float = Field(default=0.0, ge=0.0, le=1.0)
    chart: float = Field(default=0.0, ge=0.0, le=1.0)
    image: float = Field(default=0.0, ge=0.0, le=1.0)
    mixed: float = Field(default=0.0, ge=0.0, le=1.0)


class OCRRegion(BaseModel):
    """Spatial bounding box and recognized text for a token or line."""

    model_config = ConfigDict(extra="allow")

    bbox: dict[str, int]
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OCRResult(BaseModel):
    """Aggregated OCR output containing concatenated text and token regions."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    regions: list[dict[str, Any]] = Field(default_factory=list)


class ExtractedObject(BaseModel):
    """Basic OCR/layout-based structural element (block, region, grid)."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    bbox: dict[str, int]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MIR(BaseModel):
    """Canonical Multimodal Intermediate Representation v0.1."""

    model_config = ConfigDict(extra="allow")

    mir_version: Literal["0.1"] = "0.1"
    request_id: str = Field(min_length=1)
    image: ImageInfo
    context: dict[str, Any] = Field(default_factory=dict)
    modalities: Modalities
    primary_modality: Modality
    ocr: OCRResult = Field(default_factory=OCRResult)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    visual_required: bool = False
    privacy_flags: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_request_id(self) -> MIR:
        if not self.request_id.strip():
            raise ValueError("request_id cannot be empty or whitespace")
        return self

