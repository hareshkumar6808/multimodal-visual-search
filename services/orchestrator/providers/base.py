from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from time import perf_counter


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    text: str
    latency_ms: int
    cloud_image_uploaded: bool


@dataclass
class ProviderStats:
    session_requests: int = 0
    requests_today: int = 0
    failures: int = 0
    total_latency_ms: int = 0
    tracking_date: date = date.today()

    def roll_day(self) -> None:
        today = date.today()
        if self.tracking_date != today:
            self.tracking_date = today
            self.requests_today = 0

    @property
    def average_latency_ms(self) -> float | None:
        successful = self.session_requests - self.failures
        return round(self.total_latency_ms / successful, 1) if successful > 0 else None


class Provider(ABC):
    name: str

    def __init__(
        self,
        *,
        daily_budget: int = 0,
        is_cloud: bool = True,
        family: str | None = None,
        preferred_experts: frozenset[str] | None = None,
    ) -> None:
        self.daily_budget = daily_budget
        self.is_cloud = is_cloud
        self.family = family
        self.preferred_experts = preferred_experts or frozenset()
        self.stats = ProviderStats()

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        raise NotImplementedError

    def available(self) -> bool:
        self.stats.roll_day()
        return self.configured and (
            self.daily_budget <= 0 or self.stats.requests_today < self.daily_budget
        )

    async def generate(
        self, prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/png"
    ) -> ProviderResult:
        if not self.available():
            raise ProviderError(f"Provider {self.name} is unavailable or over its local budget")
        if image_bytes is not None and "vision" not in self.capabilities:
            raise ProviderError(f"Provider {self.name} does not support vision")
        self.stats.session_requests += 1
        self.stats.requests_today += 1
        started = perf_counter()
        try:
            text = (await self._generate(prompt, image_bytes, mime_type)).strip()
            if not text:
                raise ProviderError(f"Provider {self.name} returned an empty response")
        except Exception as exc:
            self.stats.failures += 1
            if isinstance(exc, ProviderError):
                raise
            detail = str(exc).strip()
            suffix = f": {detail}" if detail else ""
            raise ProviderError(
                f"Provider {self.name} request failed ({type(exc).__name__}){suffix}"
            ) from exc
        latency_ms = round((perf_counter() - started) * 1000)
        self.stats.total_latency_ms += latency_ms
        return ProviderResult(text, latency_ms, image_bytes is not None and self.is_cloud)

    async def aclose(self) -> None:
        """Release provider resources when the application shuts down."""
        return None

    @abstractmethod
    async def _generate(self, prompt: str, image_bytes: bytes | None, mime_type: str) -> str:
        raise NotImplementedError
