"""Basic OCR and layout-based object extraction.

NOTE: This module performs rule-based spatial grouping of OCR bounding boxes.
It does NOT perform learned object detection, semantic segmentation, or
vision-language reasoning.
"""

from __future__ import annotations

from typing import Any

from services.perception.models import ExtractedObject, Modality, OCRResult


def _union_bboxes(bboxes: list[dict[str, int]]) -> dict[str, int]:
    """Compute the bounding box union enclosing all provided bounding boxes."""
    if not bboxes:
        return {"x": 0, "y": 0, "width": 0, "height": 0}

    min_x = min(b.get("x", 0) for b in bboxes)
    min_y = min(b.get("y", 0) for b in bboxes)
    max_x = max(b.get("x", 0) + b.get("width", 0) for b in bboxes)
    max_y = max(b.get("y", 0) + b.get("height", 0) for b in bboxes)

    return {
        "x": min_x,
        "y": min_y,
        "width": max(0, max_x - min_x),
        "height": max(0, max_y - min_y),
    }


def extract_objects(
    ocr_result: OCRResult,
    primary_modality: Modality,
    img_width: int,
    img_height: int,
) -> list[dict[str, Any]]:
    """Group OCR regions into basic layout objects (blocks, grids, visual regions)."""
    regions = ocr_result.regions or []
    extracted: list[ExtractedObject] = []
    obj_counter = 1

    if not regions:
        # If no OCR regions exist, the entire screenshot is treated as a single visual element
        extracted.append(
            ExtractedObject(
                id=f"obj_{obj_counter}",
                type="visual_region",
                bbox={"x": 0, "y": 0, "width": img_width, "height": img_height},
                confidence=1.0,
                content="",
                metadata={"reason": "No text detected by local OCR; full area visual region"},
            )
        )
        return [obj.model_dump() for obj in extracted]

    # 1. Group regions vertically into approximate lines/paragraphs
    # Sort regions primarily by Y, secondarily by X
    sorted_regions = sorted(
        regions,
        key=lambda r: (r.get("bbox", {}).get("y", 0), r.get("bbox", {}).get("x", 0)),
    )

    # Line grouping: cluster tokens with similar vertical Y baseline
    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = []
    current_y: int | None = None
    y_tolerance = 10  # pixels

    for reg in sorted_regions:
        y = reg.get("bbox", {}).get("y", 0)
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current_line.append(reg)
            current_y = y
        else:
            if current_line:
                lines.append(current_line)
            current_line = [reg]
            current_y = y

    if current_line:
        lines.append(current_line)

    # 2. Block grouping based on modality context
    if primary_modality == "code":
        # Group all lines into a code block
        all_bboxes = [r.get("bbox", {}) for r in sorted_regions]
        union_box = _union_bboxes(all_bboxes)
        code_text = "\n".join(" ".join(r.get("text", "") for r in line) for line in lines)
        extracted.append(
            ExtractedObject(
                id=f"obj_{obj_counter}",
                type="code_block",
                bbox=union_box,
                confidence=ocr_result.confidence or 0.85,
                content=code_text,
                metadata={"line_count": len(lines)},
            )
        )
        obj_counter += 1

    elif primary_modality == "table":
        # Group aligned lines into a table region
        all_bboxes = [r.get("bbox", {}) for r in sorted_regions]
        union_box = _union_bboxes(all_bboxes)
        table_text = "\n".join(" | ".join(r.get("text", "") for r in line) for line in lines)
        extracted.append(
            ExtractedObject(
                id=f"obj_{obj_counter}",
                type="table_region",
                bbox=union_box,
                confidence=ocr_result.confidence or 0.80,
                content=table_text,
                metadata={"estimated_rows": len(lines)},
            )
        )
        obj_counter += 1

    elif primary_modality == "chart":
        # Group into a chart layout region with label metadata
        all_bboxes = [r.get("bbox", {}) for r in sorted_regions]
        union_box = _union_bboxes(all_bboxes)
        extracted.append(
            ExtractedObject(
                id=f"obj_{obj_counter}",
                type="chart_region",
                bbox=union_box,
                confidence=ocr_result.confidence or 0.75,
                content=ocr_result.text,
                metadata={"label_count": len(sorted_regions)},
            )
        )
        obj_counter += 1

    else:
        # Default text/mixed: cluster contiguous lines into text blocks (paragraphs)
        # Separate paragraphs when vertical gap > 2x average line height
        current_block: list[list[dict[str, Any]]] = []
        last_line_bottom: int | None = None

        for line in lines:
            line_bboxes = [r.get("bbox", {}) for r in line]
            line_box = _union_bboxes(line_bboxes)
            line_top = line_box.get("y", 0)
            line_height = max(12, line_box.get("height", 16))

            if last_line_bottom is not None and (line_top - last_line_bottom) > (line_height * 2):
                # New paragraph / block
                if current_block:
                    block_bboxes = [r.get("bbox", {}) for l in current_block for r in l]
                    block_text = "\n".join(" ".join(r.get("text", "") for r in l) for l in current_block)
                    extracted.append(
                        ExtractedObject(
                            id=f"obj_{obj_counter}",
                            type="text_block",
                            bbox=_union_bboxes(block_bboxes),
                            confidence=ocr_result.confidence or 0.90,
                            content=block_text,
                            metadata={"line_count": len(current_block)},
                        )
                    )
                    obj_counter += 1
                current_block = [line]
            else:
                current_block.append(line)

            last_line_bottom = line_top + line_height

        if current_block:
            block_bboxes = [r.get("bbox", {}) for l in current_block for r in l]
            block_text = "\n".join(" ".join(r.get("text", "") for r in l) for l in current_block)
            extracted.append(
                ExtractedObject(
                    id=f"obj_{obj_counter}",
                    type="text_block",
                    bbox=_union_bboxes(block_bboxes),
                    confidence=ocr_result.confidence or 0.90,
                    content=block_text,
                    metadata={"line_count": len(current_block)},
                )
            )
            obj_counter += 1

    # 3. Check for substantial untexted visual region (e.g. diagram/photo beside or below text)
    all_text_bboxes = [r.get("bbox", {}) for r in sorted_regions]
    text_union = _union_bboxes(all_text_bboxes)
    text_area = text_union.get("width", 0) * text_union.get("height", 0)
    image_area = max(1, img_width * img_height)

    # If text occupies less than 60% of the screenshot area, record the remaining major area
    if (text_area / image_area) < 0.60:
        extracted.append(
            ExtractedObject(
                id=f"obj_{obj_counter}",
                type="visual_region",
                bbox={"x": 0, "y": 0, "width": img_width, "height": img_height},
                confidence=0.85,
                content="",
                metadata={"description": "Visual/graphical area surrounding or accompanying text"},
            )
        )

    return [obj.model_dump() for obj in extracted]

