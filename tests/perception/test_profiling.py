from __future__ import annotations

import pytest

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


def paragraph_regions(lines: list[str]) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for row, line in enumerate(lines):
        x = 40
        for word in line.split():
            width = max(18, len(word) * 8)
            regions.append(
                {"bbox": {"x": x, "y": 30 + row * 32, "width": width, "height": 20}, "text": word}
            )
            x += width + 7
    return regions


def table_regions(rows: list[list[str]]) -> list[dict[str, object]]:
    columns = [40, 260, 470]
    return [
        {
            "bbox": {"x": columns[column], "y": 30 + row * 38, "width": 80, "height": 20},
            "text": cell,
        }
        for row, cells in enumerate(rows)
        for column, cell in enumerate(cells)
    ]


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
            {
                "bbox": {"x": 20, "y": 20, "width": 150, "height": 20},
                "text": "def calculate_metrics(data):",
            },
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
        "The architecture separates representation, expert routing, tool execution, "
        "and local caching. It provides a unified conversational intelligence layer "
        "across Windows applications and workflows."
    )
    ocr = OCRResult(text=prose, confidence=0.92, regions=[])
    context = {"active_app": "Google Chrome", "window_title": "Architecture Overview - Docs"}

    modalities, primary, conf = profile_multimodal(ocr, context, dummy_preflight())

    assert primary == "text"
    assert modalities.text >= 0.50
    assert modalities.code < 0.30
    assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize(
    "lines",
    [
        ["This is normal text.", "This is another normal line.", "This is another sentence."],
        ["Operating systems manage hardware resources.", "They provide services to applications."],
        ["Paris is the capital of France.", "It is known for art and culture."],
        ["A process is a program in execution.", "The scheduler allocates CPU time."],
        [
            "Memory stores active program data.",
            "Virtual memory extends the available address space.",
        ],
        ["Files organize persistent information.", "Directories group related files together."],
        ["Networks connect independent computers.", "Protocols define how messages are exchanged."],
        ["Databases store structured records.", "Queries retrieve selected information."],
        ["Compilers translate source code.", "The resulting program can then execute."],
        ["Security controls access to resources.", "Authentication verifies a user's identity."],
    ],
)
def test_paragraph_lines_with_identical_left_margins_are_text(lines: list[str]) -> None:
    text = "\n".join(lines)
    modalities, primary, _ = profile_multimodal(
        OCRResult(text=text, confidence=0.94, regions=paragraph_regions(lines)),
        {"active_app": "Notepad", "window_title": "notes.txt"},
        dummy_preflight(),
    )
    assert primary == "text"
    assert modalities.text > modalities.table


@pytest.mark.parametrize(
    "rows",
    [
        [["Name", "Age", "Marks"], ["John", "20", "90"], ["Sam", "21", "84"], ["Raj", "19", "88"]],
        [["Item", "Qty", "Price"], ["Pen", "3", "30"], ["Book", "2", "120"]],
        [["City", "Temp", "Rain"], ["Paris", "20", "No"], ["Rome", "24", "No"]],
        [["Team", "Won", "Lost"], ["Blue", "8", "2"], ["Red", "6", "4"]],
        [["Quarter", "Sales", "Growth"], ["Q1", "120", "10"], ["Q2", "145", "21"]],
    ],
)
def test_repeated_multicolumn_rows_are_tables(rows: list[list[str]]) -> None:
    text = "\n".join("   ".join(row) for row in rows)
    modalities, primary, _ = profile_multimodal(
        OCRResult(text=text, confidence=0.94, regions=table_regions(rows)),
        {},
        dummy_preflight(),
    )
    assert primary == "table"
    assert modalities.table > modalities.text


@pytest.mark.parametrize(
    "code",
    [
        "arr = [1, 2, 3]\nprint(arr[4])",
        "def add(a, b):\n    return a + b",
        "for item in values:\n    print(item)",
        "try:\n    run()\nexcept ValueError as error:\n    print(error)",
        "const value = items.map((item) => item.id);",
    ],
)
def test_code_cases_remain_code(code: str) -> None:
    modalities, primary, _ = profile_multimodal(
        OCRResult(text=code, confidence=0.94, regions=paragraph_regions(code.splitlines())),
        {"active_app": "Visual Studio Code", "window_title": "main.py"},
        dummy_preflight(),
    )
    assert primary == "code"
    assert modalities.code > modalities.table


def test_modality_state_does_not_leak_between_captures() -> None:
    table = [["Name", "Age", "Marks"], ["John", "20", "90"], ["Sam", "21", "84"]]
    paragraph = ["This is a normal paragraph.", "It has complete sentences and wrapped lines."]
    cases = [
        (
            OCRResult(
                text="\n".join("   ".join(row) for row in table),
                confidence=0.95,
                regions=table_regions(table),
            ),
            {},
            "table",
        ),
        (
            OCRResult(
                text="\n".join(paragraph), confidence=0.95, regions=paragraph_regions(paragraph)
            ),
            {},
            "text",
        ),
        (
            OCRResult(text="arr = [1, 2, 3]\nprint(arr[4])", confidence=0.95, regions=[]),
            {"active_app": "Code", "window_title": "main.py"},
            "code",
        ),
        (
            OCRResult(
                text="\n".join(paragraph), confidence=0.95, regions=paragraph_regions(paragraph)
            ),
            {},
            "text",
        ),
    ]
    assert [
        profile_multimodal(ocr, context, dummy_preflight())[1] for ocr, context, _ in cases
    ] == [expected for _, _, expected in cases]


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
