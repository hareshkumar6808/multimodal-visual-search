from services.orchestrator.intent import classify_intent


def test_stage_one_intents() -> None:
    assert classify_intent("Why is this code failing?", "code") == "debug"
    assert classify_intent("Summarize this", "text") == "summarize"
    assert classify_intent("What is this?", "image") == "identify"
    assert classify_intent("", "table") == "suggest"
