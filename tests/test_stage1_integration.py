from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import services.perception as perception_module
from contracts.models import OCRResult
from services.orchestrator.main import create_app
from services.orchestrator.perception import RealPerceptionAdapter
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator
from services.perception.ocr.base import OCREngine
from services.perception.pipeline import PerceptionPipeline


class StaticOCREngine(OCREngine):
    def __init__(self, text: str) -> None:
        self.text = text

    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        return OCRResult(text=self.text, confidence=0.96, regions=[])


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (400, 200), color=(245, 245, 245)).save(output, format="PNG")
    return output.getvalue()


def integrated_client(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    *,
    vision: bool = False,
    cloud: bool = False,
) -> tuple[TestClient, MockProvider]:
    monkeypatch.setattr(
        perception_module,
        "_DEFAULT_PIPELINE",
        PerceptionPipeline(ocr_engine=StaticOCREngine(text)),
    )
    provider = MockProvider(name="gemini" if cloud else "local", vision=vision, is_cloud=cloud)
    service = Orchestrator(
        RealPerceptionAdapter("services.perception"),
        RuleRouter(),
        ProviderRegistry([provider]),
    )
    return TestClient(create_app(service)), provider


@pytest.mark.parametrize(
    ("text", "context", "query", "expected_modality", "expected_expert"),
    [
        (
            "The capital of France is Paris. This sentence states a geographic fact.",
            {"active_app": "Word", "window_title": "notes.docx"},
            "Explain this.",
            "text",
            "text-expert",
        ),
        (
            "arr = [1, 2, 3]\nprint(arr[4])",
            {"active_app": "Visual Studio Code", "window_title": "main.py"},
            "Why is this failing?",
            "code",
            "code-expert",
        ),
        (
            "| Name | Value |\n| A | 10 |\n| B | 20 |",
            {"active_app": "Excel", "window_title": "values.csv"},
            "Extract this table",
            "table",
            "table-expert",
        ),
        (
            "Sales trend chart 2024 2025 axis percent increase",
            {"active_app": "Power BI", "window_title": "sales dashboard"},
            "Explain this chart",
            "chart",
            "chart-expert",
        ),
    ],
)
def test_real_perception_to_router_textual_routes(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    context: dict[str, str],
    query: str,
    expected_modality: str,
    expected_expert: str,
) -> None:
    image_required = expected_modality == "chart"
    client, provider = integrated_client(monkeypatch, text, vision=image_required)
    response = client.post(
        "/api/analyze",
        files={"image": ("capture.png", png_bytes(), "image/png")},
        data={
            "payload_json": json.dumps(
                {
                    "request_id": f"integration-{expected_modality}",
                    "query": query,
                    "context": context,
                }
            )
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["request_id"] == f"integration-{expected_modality}"
    assert body["mir_summary"]["primary_modality"] == expected_modality
    assert body["route"]["expert"] == expected_expert
    assert body["metrics"]["cloud_image_uploaded"] is False
    assert provider.received_images == [image_required]


def test_real_perception_visual_route_forwards_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    client, provider = integrated_client(monkeypatch, "", vision=True, cloud=True)
    response = client.post(
        "/api/analyze",
        files={"image": ("photo.png", png_bytes(), "image/png")},
        data={
            "payload_json": json.dumps(
                {"request_id": "integration-image", "query": "What is this?", "context": {}}
            )
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["mir_summary"]["primary_modality"] == "image"
    assert body["route"]["expert"] == "vision-expert"
    assert body["metrics"]["cloud_image_uploaded"] is True
    assert provider.received_images == [True]


def test_real_perception_empty_query_returns_suggestions(monkeypatch: pytest.MonkeyPatch) -> None:
    client, provider = integrated_client(
        monkeypatch,
        "arr = [1, 2, 3]\nprint(arr[4])",
    )
    response = client.post(
        "/api/analyze",
        files={"image": ("code.png", png_bytes(), "image/png")},
        data={
            "payload_json": json.dumps(
                {
                    "request_id": "integration-suggest",
                    "query": None,
                    "context": {"active_app": "Visual Studio Code", "window_title": "main.py"},
                }
            )
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["route"]["intent"] == "suggest"
    assert body["suggested_actions"] == [
        "Explain this",
        "Debug this",
        "Optimize this",
        "What does this output?",
    ]
    assert body["metrics"]["api_calls"] == 0
    assert provider.stats.session_requests == 0
