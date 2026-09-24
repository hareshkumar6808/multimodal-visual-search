from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import time

from contracts.models import ProgressEvent, ProgressSnapshot


@dataclass
class _RequestProgress:
    events: list[ProgressEvent] = field(default_factory=list)
    complete: bool = False
    updated_at: float = field(default_factory=time)


class ProgressStore:
    """In-memory request progress for the desktop client's short polling loop."""

    def __init__(self, max_requests: int = 100) -> None:
        self._requests: dict[str, _RequestProgress] = {}
        self._max_requests = max_requests
        self._lock = Lock()

    def start(self, request_id: str) -> None:
        with self._lock:
            if len(self._requests) >= self._max_requests:
                oldest = min(self._requests, key=lambda key: self._requests[key].updated_at)
                self._requests.pop(oldest, None)
            self._requests[request_id] = _RequestProgress()

    def append(self, request_id: str, stage: str, status: str, message: str) -> None:
        event = ProgressEvent(stage=stage, status=status, message=message)
        with self._lock:
            progress = self._requests.setdefault(request_id, _RequestProgress())
            progress.events.append(event)
            progress.updated_at = time()

    def finish(self, request_id: str) -> None:
        with self._lock:
            progress = self._requests.setdefault(request_id, _RequestProgress())
            progress.complete = True
            progress.updated_at = time()

    def fail(self, request_id: str, message: str) -> None:
        self.append(request_id, "request", "failed", message)
        self.finish(request_id)

    def snapshot(self, request_id: str) -> ProgressSnapshot:
        with self._lock:
            progress = self._requests.get(request_id)
            if progress is None:
                return ProgressSnapshot(request_id=request_id, complete=False, events=[])
            return ProgressSnapshot(
                request_id=request_id,
                complete=progress.complete,
                events=list(progress.events),
            )
