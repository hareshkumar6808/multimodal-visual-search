from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderPerformance:
    provider: str
    task_kind: str
    ewma_ttft_ms: float | None
    ewma_total_ms: float | None
    success_ewma: float
    sample_count: int
    updated_at: int


class ProviderPerformanceStore:
    """Bounded SQLite EWMA statistics shared across backend restarts."""

    def __init__(self, database_path: str, *, alpha: float = 0.25) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_performance (
                    provider TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    ewma_ttft_ms REAL,
                    ewma_total_ms REAL,
                    success_ewma REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(provider, task_kind)
                )
                """
            )

    def get(self, provider: str, task_kind: str) -> ProviderPerformance | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM provider_performance WHERE provider=? AND task_kind=?",
                (provider, task_kind),
            ).fetchone()
        if row is None:
            return None
        return ProviderPerformance(
            provider=str(row["provider"]),
            task_kind=str(row["task_kind"]),
            ewma_ttft_ms=(float(row["ewma_ttft_ms"]) if row["ewma_ttft_ms"] is not None else None),
            ewma_total_ms=(
                float(row["ewma_total_ms"]) if row["ewma_total_ms"] is not None else None
            ),
            success_ewma=float(row["success_ewma"]),
            sample_count=int(row["sample_count"]),
            updated_at=int(row["updated_at"]),
        )

    def record(
        self,
        provider: str,
        task_kind: str,
        *,
        total_ms: float,
        success: bool,
        ttft_ms: float | None = None,
    ) -> ProviderPerformance:
        current = self.get(provider, task_kind)
        alpha = self.alpha

        def ewma(previous: float | None, value: float | None) -> float | None:
            if value is None:
                return previous
            return value if previous is None else alpha * value + (1 - alpha) * previous

        total = ewma(current.ewma_total_ms if current else None, total_ms)
        ttft = ewma(current.ewma_ttft_ms if current else None, ttft_ms)
        previous_success = current.success_ewma if current else None
        success_value = 1.0 if success else 0.0
        success_ewma = (
            success_value
            if previous_success is None
            else alpha * success_value + (1 - alpha) * previous_success
        )
        sample_count = (current.sample_count if current else 0) + 1
        updated_at = int(time.time() * 1000)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO provider_performance
                    (provider, task_kind, ewma_ttft_ms, ewma_total_ms,
                     success_ewma, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, task_kind) DO UPDATE SET
                    ewma_ttft_ms=excluded.ewma_ttft_ms,
                    ewma_total_ms=excluded.ewma_total_ms,
                    success_ewma=excluded.success_ewma,
                    sample_count=excluded.sample_count,
                    updated_at=excluded.updated_at
                """,
                (provider, task_kind, ttft, total, success_ewma, sample_count, updated_at),
            )
        return ProviderPerformance(
            provider, task_kind, ttft, total, success_ewma, sample_count, updated_at
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
