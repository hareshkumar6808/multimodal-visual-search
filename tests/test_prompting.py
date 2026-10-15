from contracts.models import MIR, AnalyzePayload
from services.orchestrator.execution.registry import EXPERT_REGISTRY
from services.orchestrator.experts import EXPERTS
from services.orchestrator.prompting import build_prompt


def biology_mir() -> MIR:
    return MIR.model_validate(
        {
            "request_id": "biology-prompt",
            "image": {"sha256": "a" * 64, "width": 800, "height": 300},
            "modalities": {"text": 0.97},
            "primary_modality": "text",
            "ocr": {
                "text": "Cellular respiration breaks down glucose to produce ATP.",
                "confidence": 0.96,
            },
            "objects": [],
            "overall_confidence": 0.96,
        }
    )


def test_generic_explain_prompt_does_not_turn_window_context_into_the_question() -> None:
    prompt = build_prompt(
        EXPERTS["text-expert"],
        biology_mir(),
        AnalyzePayload(
            request_id="biology-prompt",
            query="explain",
            context={"active_app": "Visual Search", "window_title": "Desktop metadata"},
        ),
    )
    assert "Desktop metadata" not in prompt
    assert prompt.endswith("CURRENT USER QUESTION: explain")
    assert "Cellular respiration" in prompt


def test_explicit_app_question_can_use_application_context() -> None:
    prompt = build_prompt(
        EXPERTS["text-expert"],
        biology_mir(),
        AnalyzePayload(
            request_id="biology-prompt",
            query="Which app is this?",
            context={"active_app": "Notepad", "window_title": "notes.txt"},
        ),
    )
    assert "Application context: application=Notepad, window=notes.txt" in prompt
    assert prompt.endswith("CURRENT USER QUESTION: Which app is this?")


def test_specialized_leaf_policy_precedes_current_question() -> None:
    policy = EXPERT_REGISTRY.resolve("text/biomedical")
    prompt = build_prompt(
        EXPERTS["text-expert"],
        biology_mir(),
        AnalyzePayload(request_id="biology-prompt", query="Explain this"),
        specialized_policy=policy.prompt_fragment,
    )
    assert "Act as a biology and biomedical subject expert" in prompt
    assert prompt.endswith("CURRENT USER QUESTION: Explain this")
