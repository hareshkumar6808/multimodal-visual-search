from contracts.models import Intent, Modality

KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    ("debug", ("debug", "failing", "failure", "error", "exception", "bug", "wrong")),
    ("summarize", ("summarize", "summary", "brief", "tldr", "tl;dr")),
    ("extract", ("extract", "transcribe", "copy", "convert to", "read the")),
    ("compare", ("compare", "difference", "versus", " vs ", "agree")),
    ("calculate", ("calculate", "compute", "total", "average", "maximum", "minimum", "sum")),
    ("search", ("search", "find online", "find information", "look up", "similar")),
    ("explain", ("explain", "why", "how does", "what does", "meaning")),
    ("identify", ("what is this", "identify", "who is", "recognize")),
]


def classify_intent(query: str | None, modality: Modality) -> Intent:
    normalized = " ".join((query or "").lower().split())
    if not normalized:
        return "suggest"
    for intent, terms in KEYWORDS:
        if any(term in normalized for term in terms):
            return intent
    if normalized.startswith(("what is", "what's", "which")):
        return "identify" if modality == "image" else "explain"
    return "general"
