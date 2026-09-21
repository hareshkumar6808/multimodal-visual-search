import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from contracts.models import (
    AnalyzePayload,
    AnalyzeResponse,
    ChatPayload,
    ConversationDetail,
    ConversationSummary,
    ProviderStatus,
)
from services.orchestrator.config import Settings, get_settings
from services.orchestrator.conversations import ConversationNotFoundError, ConversationStore
from services.orchestrator.images import InvalidImageError, validate_image
from services.orchestrator.perception import PerceptionUnavailableError, build_perception_adapter
from services.orchestrator.providers.registry import (
    NoCompatibleProviderError,
    ProviderRegistry,
)
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator
from services.orchestrator.validation import ResponseValidationError

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def build_orchestrator(settings: Settings) -> Orchestrator:
    perception = build_perception_adapter(
        settings.perception_mode, settings.app_env, settings.perception_module
    )
    return Orchestrator(
        perception,
        RuleRouter(settings.perception_confidence_threshold),
        ProviderRegistry.from_settings(settings),
        ConversationStore(settings.conversation_db_path),
    )


def create_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        active = cast(Orchestrator | None, application.state.orchestrator)
        if active is not None:
            await active.providers.aclose()
            active.conversations.close()

    app = FastAPI(title="Multimodal Visual Search Orchestrator", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator

    def get_orchestrator(settings: Annotated[Settings, Depends(get_settings)]) -> Orchestrator:
        if app.state.orchestrator is None:
            app.state.orchestrator = build_orchestrator(settings)
        return cast(Orchestrator, app.state.orchestrator)

    @app.get("/api/health")
    async def health(
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> dict[str, object]:
        perception = service.perception.health()
        return {
            "status": "ok" if perception.get("available") else "degraded",
            "orchestrator": {"available": True, "version": "0.1.0"},
            "perception": perception,
            "providers": service.providers.safe_status(),
        }

    @app.get("/api/providers", response_model=list[ProviderStatus])
    async def providers(
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> list[dict[str, object]]:
        return service.providers.safe_status()

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(
        image: Annotated[UploadFile, File()],
        payload_json: Annotated[str, Form()],
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> AnalyzeResponse:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "image must be an image file"
            )
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "image must not be empty")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "image exceeds 20 MiB")
        try:
            detected_mime_type = validate_image(image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"malformed image: {exc}"
            ) from exc
        try:
            payload = AnalyzePayload.model_validate(json.loads(payload_json))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"invalid payload_json: {exc}"
            ) from exc
        try:
            return await service.analyze(image_bytes, detected_mime_type, payload)
        except PerceptionUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except NoCompatibleProviderError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ResponseValidationError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    @app.post("/api/chat", response_model=AnalyzeResponse)
    async def chat(
        payload: ChatPayload,
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> AnalyzeResponse:
        try:
            return await service.chat(payload)
        except ConversationNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except NoCompatibleProviderError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ResponseValidationError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    @app.get("/api/conversations", response_model=list[ConversationSummary])
    async def conversations(
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> list[ConversationSummary]:
        return service.conversations.list_conversations()

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
    async def conversation(
        conversation_id: str,
        service: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> ConversationDetail:
        try:
            return service.conversations.get_conversation(conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    return app


app = create_app()
