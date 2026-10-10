from dataclasses import dataclass

from contracts.models import Intent, Modality


@dataclass(frozen=True)
class Expert:
    name: str
    modalities: frozenset[Modality]
    intents: frozenset[Intent]
    requires_image: bool
    preferred_representation: str
    compatible_providers: frozenset[str]


ALL_INTENTS: frozenset[Intent] = frozenset(
    {
        "explain",
        "identify",
        "summarize",
        "debug",
        "extract",
        "compare",
        "calculate",
        "search",
        "general",
        "suggest",
    }
)

EXPERTS: dict[str, Expert] = {
    "text-expert": Expert(
        "text-expert",
        frozenset({"text", "mixed"}),
        ALL_INTENTS,
        False,
        "ocr_text",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "code-expert": Expert(
        "code-expert",
        frozenset({"code", "text"}),
        frozenset({"debug", "explain", "identify", "general"}),
        False,
        "extracted_code",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "math-expert": Expert(
        "math-expert",
        frozenset({"math", "text"}),
        ALL_INTENTS,
        False,
        "math_expression_or_ocr",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "table-expert": Expert(
        "table-expert",
        frozenset({"table", "text"}),
        frozenset({"extract", "summarize", "compare", "calculate", "general"}),
        False,
        "structured_or_ocr_table",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "chart-expert": Expert(
        "chart-expert",
        frozenset({"chart", "image"}),
        ALL_INTENTS,
        False,
        "extracted_labels_or_image",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "vision-expert": Expert(
        "vision-expert",
        frozenset({"image", "chart", "mixed"}),
        ALL_INTENTS,
        True,
        "image",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "ui-expert": Expert(
        "ui-expert",
        frozenset({"ui", "image", "mixed"}),
        ALL_INTENTS,
        False,
        "ocr_or_pixels",
        frozenset({"nvidia", "gemini", "local"}),
    ),
    "general-expert": Expert(
        "general-expert",
        frozenset({"text", "code", "math", "table", "chart", "image", "ui", "mixed"}),
        ALL_INTENTS,
        False,
        "best_available",
        frozenset({"nvidia", "gemini", "local"}),
    ),
}


SUGGESTIONS: dict[Modality, list[str]] = {
    "code": ["Explain this", "Debug this", "Optimize this", "What does this output?"],
    "math": ["Solve this", "Explain this", "Check the result", "Show the steps"],
    "table": ["Summarize", "Extract data", "Compare values", "Find maximum"],
    "chart": ["Explain trend", "Extract data", "Find anomaly", "Compare values"],
    "image": ["Identify", "Explain", "Describe", "Visual search"],
    "ui": ["Read this", "Explain this screen", "Find the control", "Describe the layout"],
    "text": ["Explain", "Summarize", "Translate", "Verify"],
    "mixed": ["Explain", "Summarize", "Compare components", "Extract information"],
}
