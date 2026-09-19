from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from services.perception.models import (
    MIR,
    ImageInfo,
    Modalities,
    OCRResult,
)


def valid_mir_dict() -> dict[str, object]:
    return {
        "mir_version": "0.1",
        "request_id": "test-req-001",
        "image": {
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "width": 800,
            "height": 500,
        },
        "context": {
            "active_app": "Visual Studio Code",
            "window_title": "main.py",
            "selection_mode": "rectangle",
            "bounds": {"x": 100, "y": 100, "width": 800, "height": 500},
        },
        "modalities": {
            "text": 0.45,
            "code": 0.95,
            "table": 0.0,
            "chart": 0.0,
            "image": 0.05,
            "mixed": 0.0,
        },
        "primary_modality": "code",
        "ocr": {
            "text": "print('hello')",
            "confidence": 0.98,
            "regions": [
                {
                    "bbox": {"x": 10, "y": 10, "width": 80, "height": 20},
                    "text": "print('hello')",
                    "confidence": 0.98,
                }
            ],
        },
        "objects": [
            {
                "id": "obj_1",
                "type": "code_block",
                "bbox": {"x": 10, "y": 10, "width": 80, "height": 20},
                "confidence": 0.98,
                "content": "print('hello')",
                "metadata": {"line_count": 1},
            }
        ],
        "visual_required": False,
        "privacy_flags": [],
        "overall_confidence": 0.95,
    }


def test_mir_v01_valid_dict_passes() -> None:
    raw = valid_mir_dict()
    mir = MIR.model_validate(raw)

    assert mir.mir_version == "0.1"
    assert mir.request_id == "test-req-001"
    assert mir.image.width == 800
    assert mir.primary_modality == "code"
    assert mir.visual_required is False
    assert mir.overall_confidence == 0.95


def test_mir_v01_json_serializable() -> None:
    raw = valid_mir_dict()
    mir = MIR.model_validate(raw)
    json_str = json.dumps(mir.model_dump(mode="json"))
    loaded = json.loads(json_str)

    assert loaded["mir_version"] == "0.1"
    assert loaded["image"]["sha256"] == raw["image"]["sha256"]  # type: ignore[index]
    assert loaded["modalities"]["code"] == 0.95


def test_mir_v01_rejects_aspect_ratio_in_canonical_image() -> None:
    raw = valid_mir_dict()
    raw["image"]["aspect_ratio"] = 1.6  # type: ignore[index]
    # Extra field forbidden on ImageInfo
    with pytest.raises(ValidationError):
        MIR.model_validate(raw)


def test_mir_v01_rejects_empty_request_id() -> None:
    raw = valid_mir_dict()
    raw["request_id"] = "   "
    with pytest.raises(ValidationError):
        MIR.model_validate(raw)


def test_mir_v01_rejects_out_of_bounds_modality_score() -> None:
    raw = valid_mir_dict()
    raw["modalities"]["code"] = 1.5  # type: ignore[index]
    with pytest.raises(ValidationError):
        MIR.model_validate(raw)


def test_mir_v01_rejects_invalid_modality_name() -> None:
    raw = valid_mir_dict()
    raw["primary_modality"] = "video"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        MIR.model_validate(raw)

