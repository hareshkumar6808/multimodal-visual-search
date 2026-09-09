import json

from fastapi.testclient import TestClient

from contracts.models import MIR, ImageInfo, Modalities, OCRResult
from services.orchestrator.main import create_app
from services.orchestrator.perception import MockPerceptionAdapter
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator


def test_api_contract_health_providers_and_analyze() -> None:
    mir = MIR(
        request_id="seed",
        image=ImageInfo(sha256="abc", width=10, height=10),
        modalities=Modalities(text=0.98),
        primary_modality="text",
        ocr=OCRResult(text="Hello", confidence=0.98),
        overall_confidence=0.98,
    )
    service = Orchestrator(
        MockPerceptionAdapter(mir),
        RuleRouter(),
        ProviderRegistry([MockProvider(name="local", response="Hello explained")]),
    )
    client = TestClient(create_app(service))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["perception"]["mode"] == "mock"

    providers = client.get("/api/providers")
    assert providers.status_code == 200
    assert "api_key" not in json.dumps(providers.json()).lower()

    payload = {"request_id": "api-1", "query": "Explain this", "context": {}}
    response = client.post(
        "/api/analyze",
        files={"image": ("capture.png", b"pixels", "image/png")},
        data={"payload_json": json.dumps(payload)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "api-1"
    assert body["route"]["expert"] == "text-expert"
    assert body["metrics"]["cloud_image_uploaded"] is False


def test_api_rejects_non_image_upload() -> None:
    service = Orchestrator(
        MockPerceptionAdapter(),
        RuleRouter(),
        ProviderRegistry([MockProvider(name="local")]),
    )
    response = TestClient(create_app(service)).post(
        "/api/analyze",
        files={"image": ("capture.txt", b"text", "text/plain")},
        data={"payload_json": '{"request_id":"api-2","query":"Explain","context":{}}'},
    )
    assert response.status_code == 415
