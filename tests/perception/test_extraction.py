from __future__ import annotations

from services.perception.extraction.extractor import extract_objects
from services.perception.models import OCRResult


def test_extraction_code_block() -> None:
    ocr = OCRResult(
        text="def foo():\n    return 42",
        confidence=0.96,
        regions=[
            {"bbox": {"x": 20, "y": 20, "width": 80, "height": 20}, "text": "def foo():"},
            {"bbox": {"x": 40, "y": 45, "width": 90, "height": 20}, "text": "return 42"},
        ],
    )
    objects = extract_objects(ocr, primary_modality="code", img_width=800, img_height=500)

    assert len(objects) >= 1
    code_objs = [o for o in objects if o["type"] == "code_block"]
    assert len(code_objs) == 1
    assert "def foo()" in code_objs[0]["content"]
    assert code_objs[0]["bbox"]["x"] == 20
    assert code_objs[0]["bbox"]["width"] >= 100


def test_extraction_table_region() -> None:
    ocr = OCRResult(
        text="A | B\n1 | 2",
        confidence=0.91,
        regions=[
            {"bbox": {"x": 50, "y": 50, "width": 40, "height": 18}, "text": "A"},
            {"bbox": {"x": 100, "y": 50, "width": 40, "height": 18}, "text": "B"},
            {"bbox": {"x": 50, "y": 80, "width": 40, "height": 18}, "text": "1"},
            {"bbox": {"x": 100, "y": 80, "width": 40, "height": 18}, "text": "2"},
        ],
    )
    objects = extract_objects(ocr, primary_modality="table", img_width=600, img_height=400)

    table_objs = [o for o in objects if o["type"] == "table_region"]
    assert len(table_objs) == 1
    assert "estimated_rows" in table_objs[0]["metadata"]


def test_extraction_visual_region_for_empty_ocr() -> None:
    ocr = OCRResult(text="", confidence=None, regions=[])
    objects = extract_objects(ocr, primary_modality="image", img_width=1000, img_height=700)

    assert len(objects) == 1
    assert objects[0]["type"] == "visual_region"
    assert objects[0]["bbox"] == {"x": 0, "y": 0, "width": 1000, "height": 700}

