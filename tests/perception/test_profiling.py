from __future__ import annotations

from services.perception.models import OCRResult
from services.perception.preflight import ImagePreflightResult
from services.perception.profiling.profiler import profile_multimodal


def dummy_preflight(width: int = 800, height: int = 500) -> ImagePreflightResult:
    return ImagePreflightResult(
        sha256="a" * 64,
        width=width,
        height=height,
        format="PNG",
        aspect_ratio=round(width / height, 4),
        brightness=0.8,
        contrast=0.5,
        sharpness=0.6,
    )


def test_profiling_python_code() -> None:
    code_text = (
        "def calculate_metrics(data):\n"
        "    result = {}\n"
        "    for item in data:\n"
        "        result[item.id] = item.value * 2\n"
        "    return result\n"
    )
    ocr = OCRResult(
        text=code_text,
        confidence=0.95,
        regions=[
            {"bbox": {"x": 20, "y": 20, "width": 150, "height": 20}, "text": "def calculate_metrics(data):"},
            {"bbox": {"x": 40, "y": 50, "width": 120, "height": 20}, "text": "result = {}"},
        ],
    )
    context = {"active_app": "Visual Studio Code", "window_title": "metrics.py"}

    modalities, primary, conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary == "code"
    assert modalities.code >= 0.60
    assert 0.0 <= conf <= 1.0


def test_profiling_plain_text() -> None:
    prose = (
        "Multimodal visual search enables users to interact with arbitrary screen content. "
        "The architecture separates representation, expert routing, tool execution, and local caching. "
        "It provides a unified conversational intelligence layer across Windows applications and workflows."
    )
    ocr = OCRResult(text=prose, confidence=0.92, regions=[])
    context = {"active_app": "Google Chrome", "window_title": "Architecture Overview - Docs"}

    modalities, primary, conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary == "text"
    assert modalities.text >= 0.50
    assert modalities.code < 0.30
    assert 0.0 <= conf <= 1.0


def test_profiling_table_layout() -> None:
    table_text = (
        "| Quarter | Revenue | Growth |\n"
        "| Q1 2026 | $12.4M  | +14%   |\n"
        "| Q2 2026 | $15.1M  | +22%   |\n"
    )
    ocr = OCRResult(
        text=table_text,
        confidence=0.90,
        regions=[
            {"bbox": {"x": 50, "y": 30, "width": 60, "height": 20}, "text": "Quarter"},
            {"bbox": {"x": 150, "y": 30, "width": 60, "height": 20}, "text": "Revenue"},
            {"bbox": {"x": 50, "y": 60, "width": 60, "height": 20}, "text": "Q1 2026"},
            {"bbox": {"x": 150, "y": 60, "width": 60, "height": 20}, "text": "$12.4M"},
            {"bbox": {"x": 50, "y": 90, "width": 60, "height": 20}, "text": "Q2 2026"},
            {"bbox": {"x": 150, "y": 90, "width": 60, "height": 20}, "text": "$15.1M"},
        ],
    )
    context = {"active_app": "Microsoft Excel", "window_title": "financial_summary.xlsx"}

    modalities, primary, _conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary == "table"
    assert modalities.table >= 0.50


def test_profiling_chart() -> None:
    chart_text = "Revenue Growth YoY chart axis x-axis Q1 Q2 Q3 Q4 trend 85%"
    ocr = OCRResult(text=chart_text, confidence=0.88, regions=[])
    context = {"active_app": "PowerBI", "window_title": "Sales Dashboard"}

    modalities, primary, _conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary in ("chart", "mixed")
    assert modalities.chart >= 0.45


def test_profiling_photograph_low_text() -> None:
    ocr = OCRResult(text="", confidence=None, regions=[])
    context = {"active_app": "Photos", "window_title": "vacation.png"}

    modalities, primary, _conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary == "image"
    assert modalities.image >= 0.70
    assert modalities.text <= 0.20


def test_profiling_mixed_content() -> None:
    # Substantial text alongside strong chart terms and layout
    mixed_text = (
        "Figure 3. Quarterly system performance comparison across test benchmarks. "
        "The chart below displays latency trends and throughput metrics. "
        "axis x-axis y-axis trend growth 45% YoY Q1 Q2"
    )
    ocr = OCRResult(text=mixed_text, confidence=0.90, regions=[])
    context = {"active_app": "Acrobat Reader", "window_title": "Paper_Draft.pdf"}

    modalities, primary, _conf = profile_multimodal(ocr, context, dummy_preflight())

    assert modalities.text >= 0.40
    assert modalities.chart >= 0.35
    assert modalities.mixed >= 0.50
    assert primary in ("mixed", "chart", "text")
