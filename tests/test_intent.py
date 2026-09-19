from services.orchestrator.intent import classify_intent


def test_stage_one_intents() -> None:
    cases = [
        ("Explain this", "text", "explain"),
        ("Why is this code failing?", "code", "debug"),
        ("Summarize this paragraph", "text", "summarize"),
        ("Extract this table", "table", "extract"),
        ("Compare these values", "table", "compare"),
        ("Calculate the total", "table", "calculate"),
        ("Find information about this", "text", "search"),
        ("What is this?", "image", "identify"),
        ("Please inspect this", "text", "general"),
    ]
    for query, modality, expected in cases:
        assert classify_intent(query, modality) == expected
    assert classify_intent("", "table") == "suggest"
    assert classify_intent(None, "image") == "suggest"
