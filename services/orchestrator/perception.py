from __future__ import annotations

import hashlib
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

from anyio import to_thread
from pydantic import ValidationError

from contracts.models import MIR, CaptureContext, ImageInfo, Modalities, OCRResult


class PerceptionUnavailableError(RuntimeError):
    pass


class PerceptionAdapter(ABC):
    @abstractmethod
    async def analyze_capture(
        self, image_bytes: bytes, context: CaptureContext, request_id: str
    ) -> MIR:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError


class RealPerceptionAdapter(PerceptionAdapter):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def _function(self) -> Any:
        try:
            module = importlib.import_module(self.module_name)
            return module.analyze_capture
        except (ImportError, AttributeError) as exc:
            raise PerceptionUnavailableError(
                f"Real perception module '{self.module_name}' is not available"
            ) from exc

    async def analyze_capture(
        self, image_bytes: bytes, context: CaptureContext, request_id: str
    ) -> MIR:
        function = self._function()
        context_data = context.model_dump(mode="json")
        arguments = (image_bytes, context_data)
        keyword_arguments = (
            {"request_id": request_id}
            if "request_id" in inspect.signature(function).parameters
            else {}
        )
        try:
            if inspect.iscoroutinefunction(function):
                result = await function(*arguments, **keyword_arguments)
            else:
                result = await to_thread.run_sync(lambda: function(*arguments, **keyword_arguments))
        except Exception as exc:
            raise PerceptionUnavailableError("Perception analysis failed") from exc
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise PerceptionUnavailableError("Perception returned a non-object result")
        result.setdefault("request_id", request_id)
        try:
            mir = MIR.model_validate(result)
        except ValidationError as exc:
            raise PerceptionUnavailableError("Perception returned invalid MIR v0.1") from exc
        if mir.request_id != request_id:
            raise PerceptionUnavailableError("Perception request_id does not match request")
        return mir

    def health(self) -> dict[str, Any]:
        try:
            self._function()
            module = importlib.import_module(self.module_name)
            health_function = getattr(module, "health", None)
            component_health = (
                health_function() if callable(health_function) else {"available": True}
            )
            return {"mode": "real", "module": self.module_name, **component_health}
        except PerceptionUnavailableError as exc:
            return {"mode": "real", "available": False, "message": str(exc)}


class MockPerceptionAdapter(PerceptionAdapter):
    """TEST/DEVELOPMENT ONLY. Never selected by default or allowed in production."""

    def __init__(self, mir: MIR | None = None) -> None:
        self._mir = mir

    async def analyze_capture(
        self, image_bytes: bytes, context: CaptureContext, request_id: str
    ) -> MIR:
        if self._mir:
            return self._mir.model_copy(update={"request_id": request_id})
        text = context.window_title or "Mock perception content"
        return MIR(
            request_id=request_id,
            image=ImageInfo(sha256=hashlib.sha256(image_bytes).hexdigest(), width=1, height=1),
            context=context.model_dump(mode="json"),
            modalities=Modalities(text=1),
            primary_modality="text",
            ocr=OCRResult(text=text, confidence=1),
            overall_confidence=1,
        )

    def health(self) -> dict[str, Any]:
        return {"mode": "mock", "available": True, "warning": "TEST/DEVELOPMENT ONLY"}


def build_perception_adapter(mode: str, app_env: str, module_name: str) -> PerceptionAdapter:
    if mode == "mock":
        if app_env.lower() not in {"test", "development"}:
            raise RuntimeError("Mock perception is forbidden outside test/development environments")
        return MockPerceptionAdapter()
    if mode != "real":
        raise ValueError("PERCEPTION_MODE must be 'real' or 'mock'")
    return RealPerceptionAdapter(module_name)
