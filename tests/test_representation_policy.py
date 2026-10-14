from __future__ import annotations

import hashlib

import pytest

from contracts.models import (
    MIR,
    AnalyzePayload,
    ImageInfo,
    Modalities,
    OCRResult,
    RepresentationInfo,
    RepresentationScores,
)
from services.orchestrator.perception import (
    MockPerceptionAdapter,
    PerceptionUnavailableError,
)
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.representation_policy import decide_image_preservation
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator

KINDS = ("text_sufficient", "structure_sufficient", "pixels_required", "mixed")


def make_mir(kind: str | None, *, version: str = "0.2", confidence: float = 0.99) -> MIR:
    representation = None
    if kind is not None:
        representation = RepresentationInfo(
            kind=kind,  # type: ignore[arg-type]
            scores=RepresentationScores(
                text_sufficient=0.9 if kind == "text_sufficient" else 0.0,
                structure_sufficient=0.9 if kind == "structure_sufficient" else 0.0,
                pixels_required=0.9 if kind == "pixels_required" else 0.0,
                mixed=0.9 if kind == "mixed" else 0.0,
            ),
            confidence=confidence,
            pixel_dependency_prior=(
                0.0 if kind in {"text_sufficient", "structure_sufficient"} else 1.0
            ),
            source="test",
            available_representations=["pixels", "ocr_text", "layout"],
        )
    return MIR(
        mir_version=version,  # type: ignore[arg-type]
        request_id="policy-test",
        image=ImageInfo(sha256="a" * 64, width=320, height=200),
        modalities=Modalities(text=1.0),
        primary_modality="text",
        ocr=OCRResult(text="Readable content", confidence=0.99),
        visual_required=False,
        overall_confidence=0.99,
        representation=representation,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_no_content_class_authorizes_query_independent_pixel_discard(kind: str) -> None:
    mir = make_mir(kind)
    decision = decide_image_preservation(mir, "What does this screenshot mean?")
    assert decision.preserve_original is True
    expected_reason = (
        "PERCEPTION_REQUIRES_PIXELS"
        if kind in {"pixels_required", "mixed"}
        else "TASK_SUFFICIENCY_UNPROVEN"
    )
    assert decision.reason_code == expected_reason


@pytest.mark.parametrize("kind", KINDS)
async def test_each_representation_forwards_original_capture_to_vision_provider(kind: str) -> None:
    image_bytes = f"original screenshot bytes for {kind}".encode()
    provider = MockProvider(name="local", vision=True)
    service = Orchestrator(
        MockPerceptionAdapter(make_mir(kind)),
        RuleRouter(),
        ProviderRegistry([provider]),
    )

    response = await service.analyze(
        image_bytes,
        "image/png",
        AnalyzePayload(request_id=f"forwarding-{kind}", query="Read the small label"),
    )

    assert provider.received_images == [True]
    assert provider.received_image_sha256 == [hashlib.sha256(image_bytes).hexdigest()]
    assert response.route.reason_code.endswith("_PIXELS_PRESERVED")
    policy_event = next(event for event in response.trace if event.stage == "representation_policy")
    decision = decide_image_preservation(make_mir(kind), "Read the small label")
    assert decision.reason_code in (policy_event.message or "")


async def test_mir_v01_without_stage2_metadata_preserves_image() -> None:
    provider = MockProvider(name="local", vision=True)
    service = Orchestrator(
        MockPerceptionAdapter(make_mir(None, version="0.1")),
        RuleRouter(),
        ProviderRegistry([provider]),
    )
    image_bytes = b"legacy capture"

    await service.analyze(
        image_bytes,
        "image/png",
        AnalyzePayload(request_id="legacy", query="Summarize this"),
    )

    assert provider.received_images == [True]
    assert provider.received_image_sha256 == [hashlib.sha256(image_bytes).hexdigest()]
    decision = decide_image_preservation(make_mir(None, version="0.1"), "Summarize")
    assert decision.reason_code == "REPRESENTATION_MISSING_OR_LEGACY"


def test_visual_required_overrides_a_text_sufficient_prediction() -> None:
    mir = make_mir("text_sufficient").model_copy(update={"visual_required": True})
    decision = decide_image_preservation(mir, "Summarize the page")
    assert decision.preserve_original is True
    assert decision.reason_code == "PERCEPTION_REQUIRES_PIXELS"


def test_low_confidence_prediction_still_preserves_pixels() -> None:
    mir = make_mir("text_sufficient", confidence=0.2)
    decision = decide_image_preservation(mir, "Read the small label")
    assert decision.preserve_original is True
    assert decision.reason_code == "TASK_SUFFICIENCY_UNPROVEN"


async def test_invalid_perception_contract_fails_before_provider_invocation() -> None:
    class InvalidPerception(MockPerceptionAdapter):
        async def analyze_capture(self, image_bytes, context, request_id):  # type: ignore[no-untyped-def]
            raise PerceptionUnavailableError("invalid MIR")

    provider = MockProvider(name="local", vision=True)
    service = Orchestrator(
        InvalidPerception(make_mir(None)),
        RuleRouter(),
        ProviderRegistry([provider]),
    )

    with pytest.raises(PerceptionUnavailableError, match="invalid MIR"):
        await service.analyze(
            b"capture retained by caller",
            "image/png",
            AnalyzePayload(request_id="invalid-mir", query="Explain this"),
        )

    assert provider.stats.session_requests == 0
    assert provider.received_images == []
