from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from contracts.models import MIR, RepresentationInfo
from services.perception import PerceptionPipeline
from services.perception.models import Modalities, OCRResult
from services.perception.ocr.base import OCREngine
from services.perception.representation import HeuristicRepresentationSelector
from services.perception.representation.mobilenet import MobileNetV3SmallSelector


def make_png(width: int = 40, height: int = 30) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _select(
    kind: str,
    *,
    primary: str = "text",
    confidence: float = 0.9,
    ocr: OCRResult | None = None,
    visual: bool = False,
) -> RepresentationInfo:
    scores = {
        "text": 0.9 if primary == "text" else 0.1,
        "code": 0.9 if primary == "code" else 0.0,
        "table": 0.9 if primary == "table" else 0.0,
        "chart": 0.9 if primary == "chart" else 0.0,
        "image": 0.9 if primary == "image" else 0.0,
        "mixed": 0.8 if primary == "mixed" else 0.0,
    }
    result = HeuristicRepresentationSelector().select(
        Modalities(**scores),
        primary,
        confidence,
        ocr or OCRResult(text="Readable content", confidence=0.92),
        visual,
        b"fake image bytes",
    )
    assert result.kind == kind
    return result


def test_selector_emits_text_sufficient() -> None:
    result = _select("text_sufficient")
    assert 0 <= result.confidence <= 1
    assert 0 <= result.pixel_dependency_prior <= 1
    assert result.source == "heuristic"
    assert result.model is None
    assert result.available_representations == ["pixels", "ocr_text", "layout"]
    assert all(0 <= value <= 1 for value in result.scores.model_dump().values())


def test_selector_emits_structure_sufficient() -> None:
    _select("structure_sufficient", primary="table")


def test_selector_emits_pixels_required_for_visual_content() -> None:
    _select(
        "pixels_required",
        primary="image",
        ocr=OCRResult(text="", confidence=None),
        visual=True,
    )


def test_selector_emits_mixed_when_text_and_visual_evidence_coexist() -> None:
    _select("mixed", primary="mixed", visual=True)


def test_low_confidence_conservatively_preserves_pixels() -> None:
    result = _select(
        "pixels_required",
        confidence=0.4,
        ocr=OCRResult(text="uncertain", confidence=0.4),
    )
    assert result.pixel_dependency_prior >= 0.6


class StaticOCR(OCREngine):
    def is_available(self) -> bool:
        return True

    def extract(self, image_bytes: bytes) -> OCRResult:
        return OCRResult(text="Readable paragraph content", confidence=0.9, regions=[])


class BrokenSelector:
    def select(self, *args: object, **kwargs: object) -> RepresentationInfo:
        raise RuntimeError("optional model unavailable")


def test_missing_model_falls_back_without_runtime_download(tmp_path: Path) -> None:
    selector = MobileNetV3SmallSelector(
        tmp_path / "missing.pth", onnx_path=tmp_path / "missing.onnx"
    )
    result = selector.select(
        Modalities(text=0.8),
        "text",
        0.9,
        OCRResult(text="Readable content", confidence=0.9),
        False,
        b"not decoded because no checkpoint exists",
    )

    assert result.source == "heuristic"
    assert result.model is None
    assert result.local_capabilities["ml_model"] is False
    assert selector._load_attempted is True
    assert selector.model_available is False


def test_corrupt_model_falls_back_conservatively(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    model_path = tmp_path / "corrupt.pth"
    model_path.write_bytes(b"not a pytorch checkpoint")
    selector = MobileNetV3SmallSelector(
        model_path, onnx_path=tmp_path / "missing.onnx"
    )
    result = selector.select(
        Modalities(image=0.9),
        "image",
        0.9,
        OCRResult(text="", confidence=None),
        True,
        b"pixels are still retained by the fallback",
    )

    assert result.kind == "pixels_required"
    assert result.source == "heuristic"
    assert selector.model_available is False


def test_model_probabilities_normalize_and_checkpoint_loads_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "models/perception/mobilenet_v3_small.pth"
    if not checkpoint.is_file():
        pytest.skip("trained local MobileNet artifact is not present")
    local_checkpoint = tmp_path / "local-only.pth"
    local_checkpoint.write_bytes(checkpoint.read_bytes())

    def no_remote_weights(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime must never fetch model weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_remote_weights)
    image = make_png()
    selector = MobileNetV3SmallSelector(
        local_checkpoint, onnx_path=tmp_path / "missing.onnx"
    )
    first = selector.score_image(image)
    model = selector._model
    second = selector.score_image(image)
    second_selector = MobileNetV3SmallSelector(
        local_checkpoint, onnx_path=tmp_path / "missing.onnx"
    )
    second_selector.score_image(image)

    assert len(first) == 4
    assert sum(first) == pytest.approx(1.0, abs=1e-6)
    assert sum(second) == pytest.approx(1.0, abs=1e-6)
    assert selector.model_available is True
    assert selector._model is model
    assert second_selector._model is model


def test_onnx_model_normalizes_and_runs_without_network(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parents[2]
    onnx_model = root / "models/perception/mobilenet_v3_small.onnx"
    if not onnx_model.is_file():
        pytest.skip("validated ONNX artifact is not present")
    local_model = tmp_path / "local-only.onnx"
    local_model.write_bytes(onnx_model.read_bytes())
    selector = MobileNetV3SmallSelector(
        tmp_path / "missing.pth", onnx_path=local_model
    )

    probabilities = selector.score_image(make_png())

    assert len(probabilities) == 4
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert selector.model_available is True
    assert selector._backend == "onnxruntime"
    assert isinstance(selector._model, ort.InferenceSession)


def test_uncertain_model_prediction_preserves_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    selector = MobileNetV3SmallSelector("unused.pth")
    monkeypatch.setattr(selector, "_predict", lambda _image: [0.3, 0.3, 0.2, 0.2])
    result = selector.select(
        Modalities(text=0.8),
        "text",
        0.9,
        OCRResult(text="Readable content", confidence=0.9),
        False,
        b"image",
    )

    assert result.kind == "pixels_required"
    assert result.scores.model_dump() == {
        "text_sufficient": 0.3,
        "structure_sufficient": 0.3,
        "pixels_required": 0.2,
        "mixed": 0.2,
    }
    assert sum(result.scores.model_dump().values()) == pytest.approx(1.0)


def test_unavailable_model_uses_heuristic_and_emits_v02() -> None:
    image = Image.new("RGB", (20, 20), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    pipeline = PerceptionPipeline(ocr_engine=StaticOCR(), representation_selector=BrokenSelector())  # type: ignore[arg-type]

    mir = pipeline.analyze(stream.getvalue(), request_id="fallback")

    assert mir.mir_version == "0.2"
    assert mir.representation is not None
    assert mir.representation.source == "heuristic"
    assert pipeline.last_telemetry["representation_ms"] >= 0


def test_mir_v01_payload_remains_valid_and_v02_serializes_representation() -> None:
    digest = "a" * 64
    legacy = MIR.model_validate(
        {
            "mir_version": "0.1",
            "request_id": "legacy",
            "image": {"sha256": digest, "width": 1, "height": 1},
            "modalities": {"text": 0.1},
            "primary_modality": "text",
            "overall_confidence": 0.5,
        }
    )
    current = MIR(
        mir_version="0.2",
        request_id="current",
        image={"sha256": digest, "width": 1, "height": 1},
        modalities=Modalities(text=0.8),
        primary_modality="text",
        overall_confidence=0.8,
        representation=_select("text_sufficient"),
    )

    assert legacy.mir_version == "0.1"
    assert current.model_dump(mode="json")["representation"]["kind"] == "text_sufficient"
