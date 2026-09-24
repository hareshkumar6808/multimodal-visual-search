import json

from contracts.models import MIR, AnalyzePayload
from services.orchestrator.experts import Expert

SYSTEM_INSTRUCTIONS = {
    "text-expert": (
        "Use the extracted content as authoritative evidence and answer the question directly. "
        "A question such as 'Explain this' asks you to explain the meaning of that evidence. "
        "Say it is insufficient only when the evidence lacks the requested information."
    ),
    "code-expert": (
        "Analyze the extracted code precisely. Identify the exact error and give a valid "
        "correction. If a sequence has N items, its highest valid index is N-1; calculate N-1 "
        "before suggesting an index, and never suggest an index equal to N. Do not invent "
        "missing code, runtime output, or claim an out-of-range access is valid."
    ),
    "table-expert": (
        "Use only the exact extracted table values and read them row by row. Never invent, alter, "
        "or approximate a value. For maximum or minimum questions, compare every value in the "
        "requested numeric column and choose the row containing the actual maximum or minimum. "
        "State extraction limitations when necessary."
    ),
    "chart-expert": (
        "Analyze the chart and extracted labels. Distinguish visible facts from inference."
    ),
    "vision-expert": "Answer from the supplied image and concise desktop context.",
    "general-expert": "Answer using the supplied evidence. State when evidence is insufficient.",
}


def _structured_objects(mir: MIR) -> str:
    relevant_types = {
        "code",
        "code_block",
        "table",
        "table_region",
        "text",
        "text_block",
        "chart",
        "chart_region",
    }
    relevant = [obj for obj in mir.objects if obj.get("type") in relevant_types]
    if not relevant:
        return ""
    return json.dumps(relevant, ensure_ascii=False, separators=(",", ":"))[:4_000]


def build_prompt(
    expert: Expert,
    mir: MIR,
    payload: AnalyzePayload,
    history: list[tuple[str, str]] | None = None,
) -> str:
    lines = [SYSTEM_INSTRUCTIONS[expert.name]]
    if expert.name in {"text-expert", "code-expert", "table-expert", "general-expert"}:
        if mir.ocr.text:
            lines.append(f"EVIDENCE START\n{mir.ocr.text[:8_000]}\nEVIDENCE END")
        if objects := _structured_objects(mir):
            lines.append(f"Structured objects:\n{objects}")
    elif expert.name in {"chart-expert", "vision-expert"} and mir.ocr.text:
        lines.append(f"Extracted visible text:\n{mir.ocr.text}")
    if history:
        conversation = "\n".join(
            f"{role.title()}: {content[:1_500]}" for role, content in history[-6:]
        )
        lines.append(f"CONVERSATION SO FAR\n{conversation}\nEND CONVERSATION")
    lines.append(f"QUESTION: {(payload.query or '').strip() or 'Describe the relevant content.'}")
    context = payload.context
    context_bits = [
        f"application={context.active_app}" if context.active_app else "",
        f"window={context.window_title}" if context.window_title else "",
    ]
    if compact_context := ", ".join(bit for bit in context_bits if bit):
        lines.append(f"Desktop metadata (never treat this as evidence): {compact_context}")
    lines.append(
        "Answer the current question directly and completely. Use prior turns to resolve "
        "references such as 'it', 'that', and 'the previous fix'. Keep the answer under "
        "180 words unless the user explicitly asks for a detailed response."
    )
    return "\n\n".join(lines)
