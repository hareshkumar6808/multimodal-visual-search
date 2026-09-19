import httpx
import pytest

from services.orchestrator.experts import EXPERTS
from services.orchestrator.providers.gemini import GeminiProvider
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.openai_compatible import OpenAICompatibleProvider
from services.orchestrator.providers.registry import ProviderRegistry


def nvidia_provider(transport: httpx.AsyncBaseTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        model="configured-test-model",
        api_key="test-key",
        supports_vision=True,
        daily_budget=0,
        timeout=0.1,
        transport=transport,
    )


@pytest.mark.parametrize("failure", ["connection", "timeout", "429", "500", "malformed"])
async def test_nvidia_failures_activate_compatible_fallback(failure: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "connection":
            raise httpx.ConnectError("offline", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("late", request=request)
        if failure in {"429", "500"}:
            return httpx.Response(int(failure), request=request)
        return httpx.Response(200, json={"unexpected": True}, request=request)

    registry = ProviderRegistry(
        [
            nvidia_provider(httpx.MockTransport(handler)),
            MockProvider(name="gemini", response="controlled fallback"),
        ]
    )
    try:
        name, result, failures, _ = await registry.generate_with_fallback(
            EXPERTS["text-expert"], "prompt", False, b"pixels", "image/png", "failure-test"
        )
    finally:
        await registry.aclose()

    assert name == "gemini"
    assert result.text == "controlled fallback"
    assert failures == ["nvidia"]


async def test_gemini_uses_generate_content_and_distinguishes_multimodal_input() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
            request=request,
        )

    provider = GeminiProvider(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="configured-test-model",
        daily_budget=0,
        timeout=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        text_result = await provider.generate("text only")
        image_result = await provider.generate("with image", b"pixels", "image/png")
    finally:
        await provider.aclose()

    assert text_result.cloud_image_uploaded is False
    assert image_result.cloud_image_uploaded is True
    assert all(":generateContent" in str(request.url) for request in requests)
    assert b"inline_data" not in requests[0].content
    assert b"inline_data" in requests[1].content


async def test_cloud_upload_metric_includes_failed_cloud_attempt() -> None:
    first = MockProvider(name="nvidia", vision=True, fail=True, is_cloud=True)
    second = MockProvider(name="gemini", vision=True, response="answer", is_cloud=True)
    registry = ProviderRegistry([first, second])
    _, _, failures, uploaded = await registry.generate_with_fallback(
        EXPERTS["vision-expert"], "prompt", True, b"pixels", "image/png", "image-test"
    )
    assert failures == ["nvidia"]
    assert uploaded is True
