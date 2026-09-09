import json

from contracts.models import MIR, AnalyzePayload
from services.orchestrator.experts import Expert

SYSTEM_INSTRUCTIONS = {
    "text-expert": "Answer from the extracted text. State when the text is insufficient.",
    "code-expert": (
        "Analyze the extracted code precisely. Do not invent missing code or runtime output."
    ),
    "table-expert": (
        "Use only the extracted table data for factual values. State extraction limitations."
    ),
    "chart-expert": (
        "Analyze the chart and extracted labels. Distinguish visible facts from inference."
    ),
    "vision-expert": "Answer from the supplied image and concise desktop context.",
    "general-expert": "Answer using the supplied evidence. State when evidence is insufficient.",
}


def _structured_objects(mir: MIR) -> str:
    relevant = [obj for obj in mir.objects if obj.get("type") in {"code", "table", "text", "chart"}]
    return json.dumps(relevant, ensure_ascii=False, separators=(",", ":")) if relevant else ""


def build_prompt(expert: Expert, mir: MIR, payload: AnalyzePayload) -> str:
    lines = [SYSTEM_INSTRUCTIONS[expert.name]]
    context = payload.context
    context_bits = [
        f"application={context.active_app}" if context.active_app else "",
        f"window={context.window_title}" if context.window_title else "",
    ]
    if compact_context := ", ".join(bit for bit in context_bits if bit):
        lines.append(f"Context: {compact_context}")
    if expert.name in {"text-expert", "code-expert", "table-expert", "general-expert"}:
        if mir.ocr.text:
            lines.append(f"Extracted content:\n{mir.ocr.text}")
        if objects := _structured_objects(mir):
            lines.append(f"Structured objects:\n{objects}")
    elif expert.name == "chart-expert" and mir.ocr.text:
        lines.append(f"Extracted chart labels:\n{mir.ocr.text}")
    lines.append(f"Question: {payload.query.strip() or 'Describe the relevant content.'}")
    return "\n\n".join(lines)
