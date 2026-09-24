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
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    query: str | None = None
    context: CaptureContext = Field(default_factory=CaptureContext)


class ChatPayload(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=20_000)
    context: CaptureContext = Field(default_factory=CaptureContext)


class ImageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(min_length=64, max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Modalities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: float = Field(default=0, ge=0, le=1)
    code: float = Field(default=0, ge=0, le=1)
    table: float = Field(default=0, ge=0, le=1)
    chart: float = Field(default=0, ge=0, le=1)
    image: float = Field(default=0, ge=0, le=1)
    mixed: float = Field(default=0, ge=0, le=1)


class OCRResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    regions: list[dict[str, Any]] = Field(default_factory=list)


class OCRRegion(BaseModel):
    model_config = ConfigDict(extra="allow")

    bbox: dict[str, int]
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class ExtractedObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    bbox: dict[str, int]
    confidence: float = Field(default=1, ge=0, le=1)
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


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
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty or whitespace")
        return self


class TraceEvent(BaseModel):
    stage: str
    status: Literal["complete", "failed", "skipped"]
    message: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_ms: int | None = Field(default=None, ge=0)


class ProgressEvent(BaseModel):
    stage: str
    status: Literal["running", "complete", "failed"]
    message: str


class ProgressSnapshot(BaseModel):
    request_id: str
    complete: bool
    events: list[ProgressEvent] = Field(default_factory=list)


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
    conversation_id: str | None = None
    message_id: str | None = None
    answer: str | None
    suggested_actions: list[str] = Field(default_factory=list)
    mir_summary: MIRSummary
    route: RouteInfo
    trace: list[TraceEvent]
    metrics: Metrics


class ConversationMessage(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: int
    has_image: bool = False
    response: AnalyzeResponse | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    capture_id: str
    primary_modality: Modality
    created_at: int
    updated_at: int


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage]
    image_data_url: str | None = None


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    available: bool
    capabilities: list[str]
    session_request_count: int
    requests_today: int
    failures: int
    average_latency_ms: float | None
