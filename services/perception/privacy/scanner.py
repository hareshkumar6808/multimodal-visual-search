"""Lightweight conservative privacy and PII scanner.

NOTE: This module uses deterministic heuristic regular expressions to detect
common privacy-sensitive patterns (emails, phone numbers, API keys, credentials).
It does NOT use a trained privacy classification model.
"""

from __future__ import annotations

import re
from typing import Any

# Conservative regex patterns to avoid false positives
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

# Standard phone number formats: e.g., +1-800-555-0199, (555) 019-2834, 555-123-4567
_PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Obvious API keys and tokens
_API_KEY_PATTERNS = [
    re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"),  # OpenAI style keys
    re.compile(r"\bghp_[a-zA-Z0-9]{36}\b"),  # GitHub personal access tokens
    re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"),  # Authorization Bearer tokens
    re.compile(
        r"(?i)['\"]?(?:api[_-]?key|secret[_-]?key|access[_-]?token)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"
    ),
]

# Obvious password / credential assignments
_CREDENTIAL_PATTERNS = [
    re.compile(r"(?i)['\"]?(?:password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"),
]


def scan_privacy(text: str, context: dict[str, Any] | None = None) -> list[str]:
    """Scan OCR text and desktop context for privacy flags.

    Returns:
        List of distinct detected flag labels (e.g. ['contains_email', 'contains_api_key']).
    """
    flags: set[str] = set()

    combined_text = text or ""
    if context:
        window_title = str(context.get("window_title") or "")
        active_app = str(context.get("active_app") or "")
        combined_text = f"{combined_text}\n{window_title}\n{active_app}"

    if not combined_text.strip():
        return []

    # 1. Email detection
    if _EMAIL_PATTERN.search(combined_text):
        flags.add("contains_email")

    # 2. Phone number detection
    if _PHONE_PATTERN.search(combined_text):
        flags.add("contains_phone")

    # 3. API key and token detection
    for pattern in _API_KEY_PATTERNS:
        if pattern.search(combined_text):
            flags.add("contains_api_key")
            break

    # 4. Password and credential detection
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(combined_text):
            flags.add("contains_credential")
            break

    return sorted(flags)
