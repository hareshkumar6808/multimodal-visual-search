from __future__ import annotations

from dataclasses import dataclass

CANONICAL_LEAVES = (
    "text/general",
    "text/biomedical",
    "text/legal",
    "text/finance",
    "code/general",
    "code/python",
    "code/java",
    "code/sql",
    "math/general",
    "math/algebra",
    "math/geometry",
    "math/probability",
    "table/general",
    "chart/general",
    "vision/general",
    "ui/general",
    "general/general",
)

PROMPTS = {
    "text/biomedical": (
        "Explain only the supplied biomedical source. Do not invent studies or diagnose."
    ),
    "text/legal": (
        "Ground the answer in the supplied source. Do not invent cases or statutes; "
        "state jurisdiction uncertainty."
    ),
    "text/finance": "Preserve every number and unit. Do not invent current prices or market data.",
    "code/general": (
        "Ground conclusions in the shown code and respect authoritative static-tool results."
    ),
    "code/python": "Analyze Python statically; never claim that selected code was executed.",
    "code/java": "Analyze the shown Java evidence and distinguish compiler facts from inference.",
    "code/sql": "Use SQL structure only; do not claim database-specific validation or execution.",
    "math/general": "Treat a successful deterministic math-tool result as authoritative.",
    "math/algebra": "Treat a successful deterministic algebra result as authoritative.",
    "math/geometry": "Use only supplied geometric facts; deterministic results are authoritative.",
    "math/probability": "State assumptions and preserve exact deterministic results.",
}


@dataclass(frozen=True)
class ExpertPolicy:
    parent_expert: str
    sub_expert: str
    canonical_leaf: str
    supported_intents: frozenset[str]
    preferred_representation: str
    tool_policy: frozenset[str]
    provider_capabilities: frozenset[str]
    prompt_fragment: str
    validation_policy: str


class ExpertRegistry:
    def __init__(self) -> None:
        self._policies = {leaf: self._build(leaf) for leaf in CANONICAL_LEAVES}

    @staticmethod
    def _build(leaf: str) -> ExpertPolicy:
        parent, sub = leaf.split("/", 1)
        tools: dict[str, frozenset[str]] = {
            "math": frozenset({"math"}),
            "table": frozenset({"table"}),
            "code/python": frozenset({"python_static"}),
            "code/sql": frozenset({"sql_static"}),
        }
        tool_policy = tools.get(leaf, tools.get(parent, frozenset()))
        capability = "vision" if parent in {"chart", "vision", "ui"} else "text"
        representation = "pixels" if capability == "vision" else "ocr_text"
        return ExpertPolicy(
            parent_expert=parent,
            sub_expert=sub,
            canonical_leaf=leaf,
            supported_intents=frozenset(
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
                }
            ),
            preferred_representation=representation,
            tool_policy=tool_policy,
            provider_capabilities=frozenset({capability}),
            prompt_fragment=PROMPTS.get(
                leaf, "Stay grounded in the supplied capture and extracted content."
            ),
            validation_policy="nonempty_grounded_answer",
        )

    def resolve(self, leaf_route: str) -> ExpertPolicy:
        try:
            return self._policies[leaf_route]
        except KeyError as exc:
            raise KeyError(f"unknown leaf route: {leaf_route}") from exc

    def adapt_stage1(self, expert: str) -> ExpertPolicy:
        mapping = {
            "text-expert": "text/general",
            "code-expert": "code/general",
            "table-expert": "table/general",
            "chart-expert": "chart/general",
            "vision-expert": "vision/general",
            "general-expert": "general/general",
        }
        return self.resolve(mapping.get(expert, "general/general"))


EXPERT_REGISTRY = ExpertRegistry()
