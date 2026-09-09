import base64
import json

from fastapi.testclient import TestClient

from contracts.models import MIR, ImageInfo, Modalities, OCRResult
from services.orchestrator.main import create_app
from services.orchestrator.perception import MockPerceptionAdapter, RealPerceptionAdapter
from services.orchestrator.providers.mock import MockProvider
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import RuleRouter
from services.orchestrator.service import Orchestrator

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZlHYAAAAASUVORK5CYII="
)


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
        files={"image": ("capture.png", PNG, "image/png")},
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


def test_api_rejects_spoofed_or_corrupt_image_content() -> None:
    service = Orchestrator(
        MockPerceptionAdapter(),
        RuleRouter(),
        ProviderRegistry([MockProvider(name="local")]),
    )
    client = TestClient(create_app(service))
    payload = '{"request_id":"bad-image","query":"Explain","context":{}}'
    response = client.post(
        "/api/analyze",
        files={"image": ("capture.png", b"not an image", "image/png")},
        data={"payload_json": payload},
    )
    assert response.status_code == 422
    assert "malformed image" in response.json()["detail"]


def test_analyze_input_contract_errors_and_nullable_query() -> None:
    service = Orchestrator(
        MockPerceptionAdapter(),
        RuleRouter(),
        ProviderRegistry([MockProvider(name="local")]),
    )
    client = TestClient(create_app(service))

    missing_image = client.post("/api/analyze", data={"payload_json": "{}"})
    malformed_json = client.post(
        "/api/analyze",
        files={"image": ("capture.png", PNG, "image/png")},
        data={"payload_json": "{"},
    )
    invalid_types = client.post(
        "/api/analyze",
        files={"image": ("capture.png", PNG, "image/png")},
        data={"payload_json": json.dumps({"request_id": [], "context": {}})},
    )
    nullable_query = client.post(
        "/api/analyze",
        files={"image": ("capture.png", PNG, "image/png")},
        data={"payload_json": json.dumps({"request_id": "nullable", "query": None})},
    )

    assert missing_image.status_code == 422
    assert malformed_json.status_code == 422
    assert invalid_types.status_code == 422
    assert nullable_query.status_code == 200
    assert nullable_query.json()["route"]["intent"] == "suggest"
    assert nullable_query.json()["metrics"]["api_calls"] == 0


def test_no_provider_is_a_controlled_failure() -> None:
    service = Orchestrator(MockPerceptionAdapter(), RuleRouter(), ProviderRegistry([]))
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/analyze",
            files={"image": ("capture.png", PNG, "image/png")},
            data={"payload_json": json.dumps({"request_id": "none", "query": "Explain"})},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "No configured, available provider supports this route"


def test_missing_real_perception_returns_controlled_503() -> None:
    service = Orchestrator(
        RealPerceptionAdapter("module_that_does_not_exist"), RuleRouter(), ProviderRegistry([])
    )
    response = TestClient(create_app(service)).post(
        "/api/analyze",
        files={"image": ("capture.png", PNG, "image/png")},
        data={"payload_json": json.dumps({"request_id": "no-perception", "query": None})},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Real perception module 'module_that_does_not_exist' is not available"
    )


def test_upload_size_limit_is_enforced(monkeypatch) -> None:
    from services.orchestrator import main as main_module

    monkeypatch.setattr(main_module, "MAX_IMAGE_BYTES", 8)
    service = Orchestrator(MockPerceptionAdapter(), RuleRouter(), ProviderRegistry([]))
    response = TestClient(create_app(service)).post(
        "/api/analyze",
        files={"image": ("capture.png", PNG, "image/png")},
        data={"payload_json": json.dumps({"request_id": "large", "query": None})},
    )
    assert response.status_code == 413


def test_complete_code_flow_uses_extracted_text_without_uploading_image() -> None:
    code = "arr = [1, 2, 3]\nprint(arr[4])"
    mir = MIR(
        request_id="original",
        image=ImageInfo(sha256="abc", width=100, height=100),
        modalities=Modalities(code=0.98, text=0.9),
        primary_modality="code",
        ocr=OCRResult(text=code, confidence=0.98),
        visual_required=False,
        overall_confidence=0.98,
    )
    provider = MockProvider(name="local", response="IndexError: index 4 is out of range.")
    service = Orchestrator(MockPerceptionAdapter(mir), RuleRouter(), ProviderRegistry([provider]))
    response = TestClient(create_app(service)).post(
        "/api/analyze",
        files={"image": ("code.png", PNG, "image/png")},
        data={
            "payload_json": json.dumps(
                {"request_id": "code-e2e", "query": "Why is this failing?", "context": {}}
            )
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["request_id"] == "code-e2e"
    assert body["route"]["intent"] == "debug"
    assert body["route"]["expert"] == "code-expert"
    assert body["metrics"]["cloud_image_uploaded"] is False
    assert provider.received_images == [False]
    assert code in provider.prompts[0]
