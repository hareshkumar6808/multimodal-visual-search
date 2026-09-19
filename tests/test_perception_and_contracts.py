import sys
import threading
from types import ModuleType

import pytest
from pydantic import ValidationError

from contracts.models import MIR, CaptureContext
from services.orchestrator.perception import (
    PerceptionUnavailableError,
    RealPerceptionAdapter,
    build_perception_adapter,
)


def mir_dict(request_id: str = "request") -> dict[str, object]:
    return {
        "mir_version": "0.1",
        "request_id": request_id,
        "image": {"sha256": "a" * 64, "width": 800, "height": 500},
        "context": {"active_app": "Visual Studio Code"},
        "modalities": {
            "text": 0.92,
            "code": 0.95,
            "table": 0.02,
            "chart": 0.01,
            "image": 0.05,
            "mixed": 0.08,
        },
        "primary_modality": "code",
        "ocr": {"text": "print(arr[4])", "confidence": 0.96, "regions": []},
        "objects": [],
        "visual_required": False,
        "privacy_flags": [],
        "overall_confidence": 0.92,
    }


def test_canonical_mir_v01_accepts_agreed_contract_and_rejects_incompatible_values() -> None:
    assert MIR.model_validate(mir_dict()).mir_version == "0.1"
    for field, value in (("mir_version", "0.2"), ("primary_modality", "unknown")):
        invalid = mir_dict()
        invalid[field] = value
        with pytest.raises(ValidationError):
            MIR.model_validate(invalid)


async def test_sync_perception_runs_off_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("test_sync_perception")
    worker_thread: list[int] = []

    def analyze_capture(image_bytes: bytes, context: dict[str, object]) -> dict[str, object]:
        worker_thread.append(threading.get_ident())
        return mir_dict("request")

    module.analyze_capture = analyze_capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calling_thread = threading.get_ident()
    result = await RealPerceptionAdapter(module.__name__).analyze_capture(
        b"image", CaptureContext(), "request"
    )
    assert result.request_id == "request"
    assert worker_thread[0] != calling_thread


@pytest.mark.parametrize(
    "output",
    ["not a dictionary", {"mir_version": "0.2"}, mir_dict("wrong-request")],
)
async def test_perception_contract_failures_are_controlled(
    monkeypatch: pytest.MonkeyPatch, output: object
) -> None:
    module = ModuleType("test_bad_perception")

    async def analyze_capture(image_bytes: bytes, context: dict[str, object]) -> object:
        return output

    module.analyze_capture = analyze_capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(PerceptionUnavailableError):
        await RealPerceptionAdapter(module.__name__).analyze_capture(
            b"image", CaptureContext(), "request"
        )


def test_mock_perception_is_rejected_in_production() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        build_perception_adapter("mock", "production", "unused")


@pytest.mark.parametrize(
    "result",
    [
        "not a dictionary",
        {"mir_version": "0.2"},
        {**mir_dict("different-request")},
    ],
)
async def test_bad_perception_output_is_a_controlled_contract_failure(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    module = ModuleType("test_bad_perception")

    async def analyze_capture(image_bytes: bytes, context: dict[str, object]) -> object:
        return result

    module.analyze_capture = analyze_capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(PerceptionUnavailableError):
        await RealPerceptionAdapter(module.__name__).analyze_capture(
            b"image", CaptureContext(), "request"
        )
