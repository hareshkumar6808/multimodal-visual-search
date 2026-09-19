"""Deterministic heuristic rule sets for multimodal profiling.

NOTE: These rules are transparent, deterministic heuristics based on OCR text,
layout geometry, and desktop context. They do NOT use machine learning or trained
classifiers.
"""

from __future__ import annotations

import re
from typing import Any

# Programming language keywords
PYTHON_PATTERNS = [
    r"\bdef\s+[a-zA-Z_]\w*\s*\(",
    r"\bclass\s+[a-zA-Z_]\w*[:\(]",
    r"\bimport\s+[a-zA-Z_]",
    r"\bfrom\s+[a-zA-Z_].*import",
    r"\bself\.[a-zA-Z_]",
    r"\breturn\b",
    r"\bif\s+.*:",
    r"\belif\s+.*:",
    r"\bprint\s*\(",
    r"\b__init__\b",
    r"\bexcept\s+.*:",
    r"\bwith\s+.*as\s+.*:",
]

JAVA_CPP_C_PATTERNS = [
    r"\bpublic\s+(static\s+)?(void|int|class|String|boolean)",
    r"\bprivate\s+[a-zA-Z_]",
    r"#include\s+[<\"].*[>\"]",
    r"\bstd::",
    r"\bSystem\.out\.println",
    r"\bprintf\s*\(",
    r"\bcout\s*<<",
    r"\bnullptr\b",
    r"\btemplate\s*<",
    r"\bnamespace\s+[a-zA-Z_]",
]

JS_TS_PATTERNS = [
    r"\bconst\s+[a-zA-Z_]\w*\s*=",
    r"\blet\s+[a-zA-Z_]\w*\s*=",
    r"\bfunction\s+[a-zA-Z_]\w*\s*\(",
    r"\bexport\s+(default\s+)?(const|function|class|type|interface)",
    r"\bconsole\.(log|error|warn)\s*\(",
    r"=>",
    r"\binterface\s+[a-zA-Z_]\w*\s*\{",
    r"\btype\s+[a-zA-Z_]\w*\s*=",
    r"===",
    r"!==",
]

SQL_PATTERNS = [
    r"\bSELECT\s+.*FROM\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+.*SET\b",
    r"\bDELETE\s+FROM\b",
    r"\bCREATE\s+TABLE\b",
    r"\bGROUP\s+BY\b",
    r"\bORDER\s+BY\b",
    r"\bINNER\s+JOIN\b",
    r"\bLEFT\s+JOIN\b",
    r"\bWHERE\s+.*(=|<|>|LIKE|IN)\b",
]

# Code structural characters
CODE_SYMBOLS = ["{", "}", ";", "()", "[]", "->", "/*", "*/", "//", "&&", "||", "!="]

# Table delimiter patterns
TABLE_DELIMITER_PATTERNS = [
    r"\|.*\|.*\|",  # Markdown pipe table
    r"\+[-+]+\+",  # ASCII grid table
    r"\t",  # Tab separated
]

# Chart terminology
CHART_KEYWORDS = [
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
    r"\baxis\b",
    r"\bx-axis\b",
    r"\by-axis\b",
    r"\blegend\b",
    r"\bscatter\b",
    r"\bhistogram\b",
    r"\btrend\b",
    r"\bgrowth\b",
    r"\byoy\b",
    r"\bq[1-4]\b",
    r"%",
]

# Desktop context keyword mappings
CONTEXT_APP_HINTS = {
    "code": [
        "code",
        "visual studio",
        "vscode",
        "pycharm",
        "intellij",
        "sublime",
        "vim",
        "nvim",
        "terminal",
        "powershell",
        "cmd",
        "git",
    ],
    "table": [
        "excel",
        "sheets",
        "calc",
        "grid",
        "workbench",
        "dbeaver",
        "pgadmin",
    ],
    "chart": [
        "tableau",
        "power bi",
        "grafana",
        "analytics",
    ],
    "text": [
        "word",
        "docs",
        "writer",
        "reader",
        "acrobat",
        "notion",
        "obsidian",
        "onenote",
    ],
    "image": [
        "photos",
        "paint",
        "photoshop",
        "lightroom",
        "gimp",
        "gallery",
    ],
}

CONTEXT_FILE_EXTENSIONS = {
    "code": [
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".sql",
        ".html",
        ".css",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    ],
    "table": [".csv", ".tsv", ".xlsx", ".xls"],
    "text": [".txt", ".md", ".pdf", ".docx", ".doc"],
}


def count_code_matches(text: str) -> tuple[int, int]:
    """Return (pattern_matches_count, symbol_matches_count)."""
    pattern_count = 0
    for group in (PYTHON_PATTERNS, JAVA_CPP_C_PATTERNS, JS_TS_PATTERNS, SQL_PATTERNS):
        for pattern in group:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_count += 1

    symbol_count = sum(text.count(sym) for sym in CODE_SYMBOLS)
    return pattern_count, symbol_count


def count_table_matches(text: str) -> int:
    """Check for table delimiter matches in OCR text."""
    matches = 0
    lines = text.splitlines()
    for line in lines:
        for pattern in TABLE_DELIMITER_PATTERNS:
            if re.search(pattern, line):
                matches += 1
                break
    return matches


def count_chart_keywords(text: str) -> int:
    """Count occurrences of chart-related terminology."""
    count = 0
    for pattern in CHART_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


def check_context_boost(
    context: dict[str, Any],
) -> dict[str, float]:
    """Derive heuristic score boosts in [0.0, 0.25] from active app and window title."""
    boosts = {"text": 0.0, "code": 0.0, "table": 0.0, "chart": 0.0, "image": 0.0}

    active_app = str(context.get("active_app") or "").lower()
    window_title = str(context.get("window_title") or "").lower()
    combined = f"{active_app} {window_title}"

    for modality, keywords in CONTEXT_APP_HINTS.items():
        if any(kw in combined for kw in keywords):
            boosts[modality] += 0.15

    for modality, extensions in CONTEXT_FILE_EXTENSIONS.items():
        if any(ext in combined for ext in extensions):
            boosts[modality] += 0.15

    # Clamp boosts to 0.25 max
    return {k: min(0.25, v) for k, v in boosts.items()}
