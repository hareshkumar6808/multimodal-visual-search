from __future__ import annotations

from services.perception.privacy.scanner import scan_privacy


def test_privacy_email_detection() -> None:
    text = "Contact our support team at security@example.com for access."
    flags = scan_privacy(text)
    assert "contains_email" in flags


def test_privacy_phone_detection() -> None:
    text = "Call customer support at +1 (800) 555-0199 during business hours."
    flags = scan_privacy(text)
    assert "contains_phone" in flags


def test_privacy_api_key_detection() -> None:
    text = "client = OpenAI(api_key='sk-abcdef1234567890abcdef1234567890')"
    flags = scan_privacy(text)
    assert "contains_api_key" in flags


def test_privacy_github_token_detection() -> None:
    text = "export GITHUB_TOKEN=ghp_123456789012345678901234567890123456"
    flags = scan_privacy(text)
    assert "contains_api_key" in flags


def test_privacy_credential_detection() -> None:
    text = "db_config = {'password': 'super_secret_password_123'}"
    flags = scan_privacy(text)
    assert "contains_credential" in flags


def test_privacy_context_scanning() -> None:
    text = "Welcome to your inbox"
    context = {"window_title": "Inbox for user.name@domain.org - Outlook"}
    flags = scan_privacy(text, context)
    assert "contains_email" in flags


def test_privacy_clean_code_no_false_positives() -> None:
    text = "def compute_hash(data: bytes) -> str:\n    return hashlib.sha256(data).hexdigest()\n"
    flags = scan_privacy(text)
    assert flags == []
