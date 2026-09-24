import asyncio

import pytest

from contracts.models import MIR, AnalyzePayload, ChatPayload, ImageInfo, Modalities, OCRResult
from services.orchestrator.conversations import ConversationStore
from services.orchestrator.perception import MockPerceptionAdapter
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator


def make_mir(modality: str, *, visual_required: bool = False, confidence: float = 0.95) -> MIR:
    scores = {name: 0.0 for name in ("text", "code", "table", "chart", "image", "mixed")}
    scores[modality] = confidence
    return MIR(
        request_id="original",
        image=ImageInfo(sha256="a" * 64, width=100, height=100),
        modalities=Modalities(**scores),
        primary_modality=modality,
        ocr=OCRResult(
            text="print(undefined_name)" if modality == "code" else "A | 10", confidence=confidence
        ),
        objects=[{"type": modality, "content": "A | 10"}],
        visual_required=visual_required,
        overall_confidence=confidence,
    )


def service_for(mir: MIR, providers: list[MockProvider] | None = None) -> Orchestrator:
    return Orchestrator(
        MockPerceptionAdapter(mir),
        RuleRouter(),
        ProviderRegistry(providers or [MockProvider(name="local")]),
    )


@pytest.mark.parametrize(
    ("modality", "query", "expert"),
    [
        ("text", "Explain this", "text-expert"),
        ("code", "Why is this code failing?", "code-expert"),
        ("table", "Find the maximum", "table-expert"),
        ("image", "What is this?", "vision-expert"),
    ],
)
async def test_routes_modality_to_expected_expert(modality: str, query: str, expert: str) -> None:
    response = await service_for(make_mir(modality)).analyze(
        b"image", "image/png", AnalyzePayload(request_id="r1", query=query)
    )
    assert response.route.expert == expert


async def test_provider_failure_uses_fallback() -> None:
    first = MockProvider(name="nvidia", fail=True)
    second = MockProvider(name="gemini", response="fallback answer")
    response = await service_for(make_mir("text"), [first, second]).analyze(
        b"image", "image/png", AnalyzePayload(request_id="r2", query="Explain")
    )
    assert response.answer == "fallback answer"
    assert response.route.provider == "gemini"
    assert response.metrics.api_calls == 2
    assert any(
        event.status == "failed" and "nvidia" in (event.message or "") for event in response.trace
    )


async def test_empty_query_returns_local_suggestions_without_provider_call() -> None:
    provider = MockProvider(name="local")
    response = await service_for(make_mir("code"), [provider]).analyze(
        b"image", "image/png", AnalyzePayload(request_id="r3", query="")
    )
    assert "Debug this" in response.suggested_actions
    assert response.answer is None
    assert response.metrics.api_calls == 0
    assert provider.stats.session_requests == 0


async def test_visual_not_required_does_not_send_image() -> None:
    provider = MockProvider(name="local")
    response = await service_for(make_mir("text", visual_required=False), [provider]).analyze(
        b"private pixels", "image/png", AnalyzePayload(request_id="r4", query="Summarize")
    )
    assert provider.received_images == [False]
    assert response.metrics.cloud_image_uploaded is False


@pytest.mark.parametrize("modality", ["code", "chart"])
async def test_nonvisual_code_and_chart_routes_do_not_send_image(modality: str) -> None:
    provider = MockProvider(name="local")
    response = await service_for(make_mir(modality, visual_required=False), [provider]).analyze(
        b"private pixels", "image/png", AnalyzePayload(request_id=modality, query="Explain")
    )
    assert provider.received_images == [False]
    assert response.metrics.cloud_image_uploaded is False


async def test_visual_required_selects_vision_capable_route_and_sends_image() -> None:
    text_only = MockProvider(name="nvidia", vision=False)
    vision = MockProvider(name="gemini", vision=True)
    response = await service_for(
        make_mir("chart", visual_required=True), [text_only, vision]
    ).analyze(b"chart pixels", "image/png", AnalyzePayload(request_id="r5", query="Explain trend"))
    assert response.route.expert == "chart-expert"
    assert response.route.provider == "gemini"
    assert text_only.stats.session_requests == 0
    assert vision.received_images == [True]
    assert response.metrics.cloud_image_uploaded is True


async def test_visual_route_falls_back_to_text_only_local_without_image() -> None:
    vision = MockProvider(name="nvidia-vision-agent", vision=True, fail=True)
    local = MockProvider(name="local", vision=False, response="Offline answer from OCR.")
    response = await service_for(make_mir("image", visual_required=True), [vision, local]).analyze(
        b"private pixels", "image/png", AnalyzePayload(request_id="offline", query="Explain")
    )

    assert response.answer == "Offline answer from OCR."
    assert response.route.provider == "local"
    assert vision.received_images == [True, True]
    assert local.received_images == [False]
    assert "OFFLINE TEXT FALLBACK" in local.prompts[0]
    assert response.metrics.cloud_image_uploaded is True


async def test_low_perception_confidence_routes_to_general_expert() -> None:
    response = await service_for(make_mir("text", confidence=0.2)).analyze(
        b"image", "image/png", AnalyzePayload(request_id="r6", query="Explain")
    )
    assert response.route.expert == "general-expert"
    assert response.route.reason_code == "LOW_PERCEPTION_CONFIDENCE"


async def test_trace_metrics_and_concurrent_request_isolation() -> None:
    provider = MockProvider(name="local")
    service = service_for(make_mir("code"), [provider])
    responses = await asyncio.gather(
        *[
            service.analyze(
                b"pixels",
                "image/png",
                AnalyzePayload(request_id=f"concurrent-{index}", query="Why is this failing?"),
            )
            for index in range(20)
        ]
    )

    assert {response.request_id for response in responses} == {
        f"concurrent-{index}" for index in range(20)
    }
    assert provider.stats.session_requests == 20
    for response in responses:
        assert [event.stage for event in response.trace] == [
            "capture_received",
            "perception",
            "intent",
            "routing",
            "provider",
            "validation",
        ]
        assert response.metrics.latency_ms + 2 >= (
            response.metrics.perception_ms
            + response.metrics.routing_ms
            + response.metrics.provider_ms
        )


async def test_three_turn_conversation_reuses_mir_and_preserves_history() -> None:
    class CountingPerception(MockPerceptionAdapter):
        calls = 0

        async def analyze_capture(self, image_bytes, context, request_id):  # type: ignore[no-untyped-def]
            self.calls += 1
            return await super().analyze_capture(image_bytes, context, request_id)

    perception = CountingPerception(make_mir("code"))
    provider = MockProvider(name="local", response="Use arr[2].")
    service = Orchestrator(
        perception,
        RuleRouter(),
        ProviderRegistry([provider]),
        ConversationStore(),
    )
    first = await service.analyze(
        b"code image",
        "image/png",
        AnalyzePayload(
            request_id="capture-1",
            conversation_id="conversation-1",
            query="Why is this failing?",
        ),
    )
    second = await service.chat(
        ChatPayload(
            request_id="follow-up-1",
            conversation_id="conversation-1",
            query="How do I fix it?",
        )
    )
    third = await service.chat(
        ChatPayload(
            request_id="follow-up-2",
            conversation_id="conversation-1",
            query="Give me the full corrected code.",
        )
    )

    assert perception.calls == 1
    assert {first.conversation_id, second.conversation_id, third.conversation_id} == {
        "conversation-1"
    }
    assert second.metrics.perception_ms == third.metrics.perception_ms == 0
    assert second.trace[0].stage == third.trace[0].stage == "context_reused"
    assert "Why is this failing?" in provider.prompts[1]
    assert "How do I fix it?" in provider.prompts[2]
    assert "Use arr[2]." in provider.prompts[2]
    detail = service.conversations.get_conversation("conversation-1")
    assert len(detail.messages) == 6
    assistant_traces = [message.response.trace for message in detail.messages if message.response]
    assert len(assistant_traces) == 3
    assert assistant_traces[0] is not assistant_traces[1]
