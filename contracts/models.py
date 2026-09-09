from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["text", "code", "table", "chart", "image", "mixed"]
Intent = Literal[
    "explain",
    "identify",
    "summarize",
    "debug",
    "extract",
    "compare",
    "calculate",
    "search",
    "general",
    "suggest",
]


class Bounds(BaseModel):
    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CaptureContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    active_app: str | None = None
    window_title: str | None = None
    selection_mode: str | None = None
    bounds: Bounds | None = None


class AnalyzePayload(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    query: str | None = None
    context: CaptureContext = Field(default_factory=CaptureContext)


class ImageInfo(BaseModel):
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Modalities(BaseModel):
    text: float = Field(default=0, ge=0, le=1)
    code: float = Field(default=0, ge=0, le=1)
    table: float = Field(default=0, ge=0, le=1)
    chart: float = Field(default=0, ge=0, le=1)
    image: float = Field(default=0, ge=0, le=1)
    mixed: float = Field(default=0, ge=0, le=1)


class OCRResult(BaseModel):
    text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    regions: list[dict[str, Any]] = Field(default_factory=list)


class MIR(BaseModel):
    model_config = ConfigDict(extra="allow")

    mir_version: Literal["0.1"] = "0.1"
    request_id: str
    image: ImageInfo
    context: dict[str, Any] = Field(default_factory=dict)
    modalities: Modalities
    primary_modality: Modality
    ocr: OCRResult = Field(default_factory=OCRResult)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    visual_required: bool = False
    privacy_flags: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def request_matches(self) -> MIR:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        return self


class TraceEvent(BaseModel):
    stage: str
    status: Literal["complete", "failed", "skipped"]
    message: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_ms: int | None = Field(default=None, ge=0)


class MIRSummary(BaseModel):
    primary_modality: Modality
    confidence: float


class RouteInfo(BaseModel):
    intent: Intent
    expert: str
    provider: str | None
    reason_code: str


class Metrics(BaseModel):
    latency_ms: int
    perception_ms: int
    routing_ms: int
    provider_ms: int
    cloud_image_uploaded: bool
    api_calls: int


class AnalyzeResponse(BaseModel):
    request_id: str
    answer: str | None
    suggested_actions: list[str] = Field(default_factory=list)
    mir_summary: MIRSummary
    route: RouteInfo
    trace: list[TraceEvent]
    metrics: Metrics


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    available: bool
    capabilities: list[str]
    session_request_count: int
    requests_today: int
    failures: int
    average_latency_ms: float | None
