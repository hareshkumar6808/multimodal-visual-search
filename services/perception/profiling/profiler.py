"""Deterministic heuristic multimodal profiler.

Evaluates OCR tokens, spatial bounding boxes, and desktop context to score
modalities: text, code, table, chart, image, mixed.
"""

from __future__ import annotations

from typing import Any

from services.perception.models import Modalities, Modality, OCRResult
from services.perception.preflight import ImagePreflightResult
from services.perception.profiling.rules import (
    check_context_boost,
    count_chart_keywords,
    count_code_matches,
    count_table_matches,
)


def _check_table_alignment(regions: list[dict[str, Any]], img_w: int, img_h: int) -> float:
    """Heuristic check for vertical column alignment among OCR bounding boxes."""
    if len(regions) < 4:
        return 0.0

    # Cluster x-coordinates within a 15-pixel tolerance
    tolerance = max(8, int(img_w * 0.015))
    x_positions: list[int] = []
    for reg in regions:
        bbox = reg.get("bbox") or {}
        x = bbox.get("x")
        if x is not None:
            x_positions.append(x)

    if not x_positions:
        return 0.0

    aligned_columns = 0
    used = [False] * len(x_positions)
    for i in range(len(x_positions)):
        if used[i]:
            continue
        cluster_size = 1
        for j in range(i + 1, len(x_positions)):
            if not used[j] and abs(x_positions[i] - x_positions[j]) <= tolerance:
                cluster_size += 1
                used[j] = True
        used[i] = True
        if cluster_size >= 3:
            aligned_columns += 1

    # If at least 2 distinct columns have 3+ aligned items, score table alignment high
    if aligned_columns >= 3:
        return 0.7
    if aligned_columns >= 2:
        return 0.45
    if aligned_columns == 1:
        return 0.2
    return 0.0


def profile_multimodal(
    ocr_result: OCRResult,
    context: dict[str, Any],
    preflight: ImagePreflightResult,
) -> tuple[Modalities, Modality, float]:
    """Calculate multi-label modality scores and derive primary_modality deterministically.

    Returns:
        (modalities, primary_modality, overall_confidence)
    """
    text = ocr_result.text or ""
    regions = ocr_result.regions or []
    word_count = len(text.split())
    ocr_conf = ocr_result.confidence if ocr_result.confidence is not None else 0.0

    total_image_area = max(1, preflight.width * preflight.height)
    total_text_area = 0
    for reg in regions:
        bbox = reg.get("bbox") or {}
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)
        total_text_area += w * h

    text_coverage = min(1.0, total_text_area / total_image_area)
    boosts = check_context_boost(context)

    # -------------------------------------------------------------
    # 1. Code Modality Score
    # -------------------------------------------------------------
    pattern_count, symbol_count = count_code_matches(text)
    code_raw = 0.0
    if pattern_count > 0:
        code_raw += min(0.65, pattern_count * 0.20)
    if symbol_count > 0:
        code_raw += min(0.25, symbol_count * 0.03)

    # Check line indentation
    indented_lines = sum(
        1 for line in text.splitlines() if line.startswith(("    ", "\t"))
    )
    if indented_lines >= 2:
        code_raw += 0.15

    code_score = min(1.0, code_raw + boosts["code"])

    # -------------------------------------------------------------
    # 2. Table Modality Score
    # -------------------------------------------------------------
    table_delims = count_table_matches(text)
    alignment_score = _check_table_alignment(regions, preflight.width, preflight.height)
    table_raw = 0.0
    if table_delims > 0:
        table_raw += min(0.60, table_delims * 0.25)
    table_raw += alignment_score
    table_score = min(1.0, table_raw + boosts["table"])

    # -------------------------------------------------------------
    # 3. Chart Modality Score
    # -------------------------------------------------------------
    chart_kw_count = count_chart_keywords(text)
    chart_raw = min(0.60, chart_kw_count * 0.20)
    # Charts typically have moderate-to-low text coverage with numbers/%
    if 0.02 <= text_coverage <= 0.35 and chart_kw_count >= 1:
        chart_raw += 0.25
    chart_score = min(1.0, chart_raw + boosts["chart"])

    # -------------------------------------------------------------
    # 4. Text Modality Score
    # -------------------------------------------------------------
    text_raw = 0.0
    if word_count >= 50:
        text_raw = 0.85
    elif word_count >= 20:
        text_raw = 0.65
    elif word_count >= 5:
        text_raw = 0.40
    elif word_count >= 1:
        text_raw = 0.20

    # Penalize plain text score if strongly identified as code
    if code_score >= 0.6:
        text_raw = min(text_raw, 0.45)

    text_score = min(1.0, text_raw + boosts["text"])

    # -------------------------------------------------------------
    # 5. Image Modality Score
    # -------------------------------------------------------------
    image_raw = 0.0
    if word_count == 0:
        image_raw = 0.90
    elif word_count < 5 and text_coverage < 0.05:
        image_raw = 0.65
    elif word_count < 15 and text_coverage < 0.15:
        image_raw = 0.35
    elif word_count >= 15:
        image_raw = 0.05
    else:
        image_raw = max(0.0, 0.20 - text_coverage)

    image_score = min(1.0, image_raw + boosts["image"])

    # -------------------------------------------------------------
    # 6. Mixed Modality Score
    # -------------------------------------------------------------
    # Mixed is elevated when heterogeneous modalities co-occur:
    # e.g., significant structured text (text/code/table) alongside visual elements (chart/image)
    # or multiple non-text modalities (table + chart)
    mixed_score = 0.0
    has_text_modality = max(text_score, code_score, table_score) >= 0.40
    has_visual_element = max(chart_score, image_score) >= 0.35
    if has_text_modality and has_visual_element:
        mixed_score = round(
            min(1.0, (max(text_score, code_score, table_score) + max(chart_score, image_score)) / 2.0 + 0.10),
            2,
        )
    elif chart_score >= 0.40 and table_score >= 0.40:
        mixed_score = round((chart_score + table_score) / 2.0, 2)

    modalities = Modalities(
        text=round(text_score, 2),
        code=round(code_score, 2),
        table=round(table_score, 2),
        chart=round(chart_score, 2),
        image=round(image_score, 2),
        mixed=round(mixed_score, 2),
    )

    # -------------------------------------------------------------
    # Determine Primary Modality
    # -------------------------------------------------------------
    primary_modality: Modality
    max_individual = max(text_score, code_score, table_score, chart_score, image_score)
    if mixed_score >= 0.65 and mixed_score >= max_individual:
        primary_modality = "mixed"
    else:
        # Pick the highest individual scoring modality
        # Priority on ties: code > table > chart > text > image
        priority_order: list[Modality] = ["code", "table", "chart", "text", "image"]
        best_modality = priority_order[0]
        max_val = -1.0
        for mod in priority_order:
            val = getattr(modalities, mod)
            if val > max_val:
                max_val = val
                best_modality = mod

        primary_modality = best_modality

    # -------------------------------------------------------------
    # Calculate Overall Confidence
    # -------------------------------------------------------------
    scores = {
        "text": text_score,
        "code": code_score,
        "table": table_score,
        "chart": chart_score,
        "image": image_score,
    }
    sorted_scores = sorted(scores.values(), reverse=True)
    top_score = sorted_scores[0]
    second_score = sorted_scores[1]

    # Confidence factors:
    # 1. OCR confidence (if text present)
    # 2. Score prominence (difference between highest and 2nd highest)
    # 3. Base detection reliability
    separation = top_score - second_score
    if word_count > 0 and ocr_conf > 0.0:
        overall_conf = (ocr_conf * 0.5) + (top_score * 0.3) + (min(1.0, separation + 0.2) * 0.2)
    else:
        # Non-text or image-only scenario
        overall_conf = (top_score * 0.7) + (preflight.contrast * 0.3)

    overall_confidence = round(max(0.10, min(0.99, overall_conf)), 2)

    return modalities, primary_modality, overall_confidence
