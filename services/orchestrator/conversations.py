from __future__ import annotations

import base64
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

from contracts.models import (
    MIR,
    AnalyzeResponse,
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
)


class ConversationNotFoundError(LookupError):
    pass


class ConversationStore:
    """Small local SQLite store for capture context and bounded chat history."""

    def __init__(self, database_path: str = ":memory:") -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL,
                    mir_json TEXT NOT NULL,
                    image_bytes BLOB NOT NULL,
                    mime_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    has_image INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                """
            )

    @staticmethod
    def _now() -> int:
        return int(time.time() * 1000)

    def save_capture(
        self,
        conversation_id: str,
        capture_id: str,
        mir: MIR,
        image_bytes: bytes,
        mime_type: str,
    ) -> None:
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversations
                    (id, capture_id, mir_json, image_bytes, mime_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    capture_id=excluded.capture_id,
                    mir_json=excluded.mir_json,
                    image_bytes=excluded.image_bytes,
                    mime_type=excluded.mime_type,
                    updated_at=excluded.updated_at
                """,
                (
                    conversation_id,
                    capture_id,
                    mir.model_dump_json(),
                    image_bytes,
                    mime_type,
                    now,
                    now,
                ),
            )

    def capture_context(self, conversation_id: str) -> tuple[MIR, bytes, str, str]:
        with self._lock:
            row = self._connection.execute(
                "SELECT capture_id, mir_json, image_bytes, mime_type FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")
        return (
            MIR.model_validate_json(row["mir_json"]),
            bytes(row["image_bytes"]),
            str(row["mime_type"]),
            str(row["capture_id"]),
        )

    def prompt_history(self, conversation_id: str, limit: int = 12) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, content FROM messages
                WHERE conversation_id=? AND role IN ('user', 'assistant')
                ORDER BY created_at DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [(str(row["role"]), str(row["content"])) for row in reversed(rows)]

    def append_exchange(
        self,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        response: AnalyzeResponse,
        *,
        user_has_image: bool,
        assistant_id: str,
    ) -> None:
        user_id = str(uuid4())
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO messages
                    (id, conversation_id, role, content, has_image, response_json, created_at)
                    VALUES (?, ?, 'user', ?, ?, NULL, ?)""",
                (user_id, conversation_id, user_content, int(user_has_image), now),
            )
            self._connection.execute(
                """INSERT INTO messages
                    (id, conversation_id, role, content, has_image, response_json, created_at)
                    VALUES (?, ?, 'assistant', ?, 0, ?, ?)""",
                (
                    assistant_id,
                    conversation_id,
                    assistant_content,
                    response.model_dump_json(),
                    now + 1,
                ),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now + 1, conversation_id)
            )

    def list_conversations(self) -> list[ConversationSummary]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.id, c.capture_id, c.mir_json, c.created_at, c.updated_at,
                    COALESCE((SELECT NULLIF(content, '') FROM messages
                              WHERE conversation_id=c.id AND role='user'
                              ORDER BY created_at LIMIT 1), '') AS first_user
                FROM conversations c ORDER BY c.updated_at DESC
                """
            ).fetchall()
        summaries: list[ConversationSummary] = []
        for row in rows:
            mir = MIR.model_validate_json(row["mir_json"])
            first_user = str(row["first_user"] or "").strip()
            title = first_user[:72] if first_user else f"{mir.primary_modality.title()} capture"
            summaries.append(
                ConversationSummary(
                    id=row["id"],
                    title=title,
                    capture_id=row["capture_id"],
                    primary_modality=mir.primary_modality,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return summaries

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        with self._lock:
            conversation = self._connection.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            rows = self._connection.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")
        mir = MIR.model_validate_json(conversation["mir_json"])
        messages = [
            ConversationMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["created_at"],
                has_image=bool(row["has_image"]),
                response=(
                    AnalyzeResponse.model_validate_json(row["response_json"])
                    if row["response_json"]
                    else None
                ),
            )
            for row in rows
        ]
        first_user = next(
            (message.content for message in messages if message.role == "user" and message.content),
            "",
        )
        encoded = base64.b64encode(bytes(conversation["image_bytes"])).decode("ascii")
        return ConversationDetail(
            id=conversation["id"],
            title=first_user[:72] or f"{mir.primary_modality.title()} capture",
            capture_id=conversation["capture_id"],
            primary_modality=mir.primary_modality,
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            messages=messages,
            image_data_url=f"data:{conversation['mime_type']};base64,{encoded}",
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
