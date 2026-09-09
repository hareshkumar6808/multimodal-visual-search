from contracts.models import MIR


def validate_answer(answer: str, expert_name: str, mir: MIR) -> tuple[bool, str]:
    if not answer.strip():
        return False, "Provider response was empty"
    if expert_name == "table-expert" and not mir.ocr.text and not mir.objects:
        return False, "No extracted table evidence was available"
    return True, "Deterministic response checks passed"
