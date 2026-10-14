import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import perf_counter

from services.orchestrator.config import Settings
from services.orchestrator.experts import Expert
from services.orchestrator.providers.base import Provider, ProviderError, ProviderResult
from services.orchestrator.providers.gemini import GeminiProvider
from services.orchestrator.providers.openai_compatible import OpenAICompatibleProvider


class NoCompatibleProviderError(RuntimeError):
    pass


class StreamInterruptedError(RuntimeError):
    """A provider failed after user-visible output; fallback would duplicate text."""


@dataclass(frozen=True)
class ProviderStreamEvent:
    type: str
    text: str = ""
    provider: str | None = None
    metrics: dict[str, float] | None = None


logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRegistry":
        return cls(
            [
                OpenAICompatibleProvider(
                    name="nvidia-general-agent",
                    base_url=settings.nvidia_base_url,
                    model=settings.nvidia_model,
                    api_key=settings.nvidia_api_key,
                    supports_vision=settings.nvidia_supports_vision,
                    daily_budget=settings.nvidia_daily_budget,
                    timeout=settings.provider_timeout_seconds,
                    max_tokens=settings.nvidia_max_tokens,
                    temperature=0.2,
                    family="nvidia",
                    preferred_experts=frozenset({"text-expert", "general-expert"}),
                ),
                OpenAICompatibleProvider(
                    name="nvidia-reasoning-agent",
                    base_url=settings.nvidia_base_url,
                    model=settings.nvidia_reasoning_model,
                    api_key=settings.nvidia_api_key,
                    supports_vision=False,
                    daily_budget=settings.nvidia_daily_budget,
                    timeout=settings.provider_timeout_seconds,
                    max_tokens=settings.nvidia_max_tokens,
                    temperature=0.2,
                    family="nvidia",
                    preferred_experts=frozenset({"code-expert", "table-expert"}),
                ),
                OpenAICompatibleProvider(
                    name="nvidia-vision-agent",
                    base_url=settings.nvidia_base_url,
                    model=settings.nvidia_vision_model,
                    api_key=settings.nvidia_api_key,
                    supports_vision=True,
                    daily_budget=settings.nvidia_daily_budget,
                    timeout=settings.provider_timeout_seconds,
                    max_tokens=settings.nvidia_max_tokens,
                    temperature=0.2,
                    family="nvidia",
                    preferred_experts=frozenset({"chart-expert", "vision-expert"}),
                ),
                GeminiProvider(
                    api_key=settings.gemini_api_key,
                    base_url=settings.gemini_base_url,
                    model=settings.gemini_model,
                    daily_budget=settings.gemini_daily_budget,
                    timeout=settings.provider_timeout_seconds,
                ),
                OpenAICompatibleProvider(
                    name="local",
                    base_url=settings.local_base_url,
                    model=settings.local_model,
                    api_key=settings.local_api_key,
                    supports_vision=settings.local_supports_vision,
                    daily_budget=settings.local_daily_budget,
                    timeout=settings.local_provider_timeout_seconds,
                    max_tokens=settings.local_max_tokens,
                ),
            ]
        )

    def ranked(
        self, expert: Expert, image_required: bool, *, prefer_cloud: bool = False
    ) -> list[Provider]:
        compatible = [
            provider
            for provider in self.providers
            if (provider.family or provider.name) in expert.compatible_providers
            and provider.available()
        ]

        if prefer_cloud:
            preferred_names = (
                ("gemini", "nvidia", "local")
                if image_required or expert.name in {"vision-expert", "chart-expert"}
                else ("nvidia", "gemini", "local")
            )
        else:
            preferred_names = ("local", "gemini", "nvidia")
        provider_priority = {name: index for index, name in enumerate(preferred_names)}

        # Apply the route-specific quality/privacy preference first, then favor healthy,
        # in-budget, low-latency providers. Unavailable providers were already filtered.
        def score(provider: Provider) -> tuple[float, float, float, float, float, float, float]:
            failure_rate = provider.stats.failures / max(provider.stats.session_requests, 1)
            latency = provider.stats.average_latency_ms or 0
            bounded = 1.0 if provider.daily_budget > 0 else 0.0
            budget_used = (
                provider.stats.requests_today / provider.daily_budget
                if provider.daily_budget > 0
                else 0.0
            )
            family = provider.family or provider.name
            priority = float(provider_priority.get(family, len(provider_priority)))
            specialization = 0.0 if expert.name in provider.preferred_experts else 1.0
            vision_penalty = (
                1.0 if image_required and "vision" not in provider.capabilities else 0.0
            )
            return (
                vision_penalty,
                priority,
                failure_rate,
                specialization,
                bounded,
                budget_used,
                latency,
            )

        return sorted(compatible, key=score)

    async def generate_with_fallback(
        self,
        expert: Expert,
        prompt: str,
        image_required: bool,
        image_bytes: bytes,
        mime_type: str,
        request_id: str,
        *,
        prefer_cloud: bool = False,
        on_progress: Callable[[str, str, str], None] | None = None,
        ranked_provider_candidates: tuple[str, ...] = (),
    ) -> tuple[str, ProviderResult, list[str], bool]:
        candidates = self.ranked(expert, image_required, prefer_cloud=prefer_cloud)
        if ranked_provider_candidates:
            order = {name: index for index, name in enumerate(ranked_provider_candidates)}
            candidates.sort(key=lambda provider: order.get(provider.name, len(order)))
        if not candidates:
            raise NoCompatibleProviderError("No configured, available provider supports this route")
        failures: list[str] = []
        cloud_image_uploaded = False
        attempted_families: set[str] = set()
        for provider in candidates:
            family = provider.family or provider.name
            if family in attempted_families:
                continue
            attempted_families.add(family)
            send_image = image_required and "vision" in provider.capabilities
            cloud_image_uploaded = cloud_image_uploaded or (send_image and provider.is_cloud)
            if on_progress:
                target = "cloud" if provider.is_cloud else "local fallback"
                on_progress("provider", "running", f"Calling {provider.name} ({target})")
            provider_prompt = prompt
            if image_required and not send_image:
                provider_prompt += (
                    "\n\nOFFLINE TEXT FALLBACK: Raw image pixels are unavailable to this "
                    "text-only model. Answer only from extracted text and desktop context "
                    "included above. If they are insufficient, say what could not be "
                    "determined from the local extraction."
                )
            attempts = 2 if provider.is_cloud else 1
            for attempt in range(attempts):
                try:
                    result = await provider.generate(
                        provider_prompt, image_bytes if send_image else None, mime_type
                    )
                    if on_progress:
                        on_progress(
                            "provider",
                            "complete",
                            f"{provider.name} responded in {result.latency_ms / 1000:.1f}s",
                        )
                    return provider.name, result, failures, cloud_image_uploaded
                except ProviderError as exc:
                    retrying = attempt + 1 < attempts
                    logger.warning(
                        "provider_failed request_id=%s provider=%s category=%s "
                        "retrying=%s error=%s",
                        request_id,
                        provider.name,
                        type(exc).__name__,
                        retrying,
                        exc,
                    )
                    if retrying:
                        if on_progress:
                            on_progress(
                                "provider",
                                "running",
                                f"{provider.name} was delayed; retrying cloud request",
                            )
                        continue
                    failures.append(provider.name)
                    if on_progress:
                        on_progress(
                            "provider",
                            "failed",
                            f"{provider.name} did not respond; trying the next provider",
                        )
        raise NoCompatibleProviderError(f"All compatible providers failed: {', '.join(failures)}")

    async def stream_with_fallback(
        self,
        expert: Expert,
        prompt: str,
        image_required: bool,
        image_bytes: bytes,
        mime_type: str,
        *,
        prefer_cloud: bool = False,
        ranked_provider_candidates: tuple[str, ...] = (),
    ) -> AsyncIterator[ProviderStreamEvent]:
        candidates = self.ranked(expert, image_required, prefer_cloud=prefer_cloud)
        if ranked_provider_candidates:
            order = {name: index for index, name in enumerate(ranked_provider_candidates)}
            candidates.sort(key=lambda provider: order.get(provider.name, len(order)))
        if not candidates:
            raise NoCompatibleProviderError("No configured, available provider supports this route")
        failed: list[str] = []
        stream_started = perf_counter()
        attempted_families: set[str] = set()
        for provider in candidates:
            family = provider.family or provider.name
            if family in attempted_families:
                continue
            attempted_families.add(family)
            provider_started = perf_counter()
            emitted = False
            try:
                yield ProviderStreamEvent("progress", provider=provider.name)
                send_image = image_required and "vision" in provider.capabilities
                async for delta in provider.stream_generate(
                    prompt, image_bytes if send_image else None, mime_type
                ):
                    if not delta:
                        continue
                    now = perf_counter()
                    metrics = None
                    if not emitted:
                        metrics = {
                            "provider_start_ms": (provider_started - stream_started) * 1000,
                            "provider_ttft_ms": (now - provider_started) * 1000,
                            "time_to_first_useful_output_ms": (now - stream_started) * 1000,
                        }
                    emitted = True
                    yield ProviderStreamEvent("delta", delta, provider.name, metrics)
                total = (perf_counter() - provider_started) * 1000
                yield ProviderStreamEvent(
                    "done",
                    provider=provider.name,
                    metrics={
                        "provider_total_ms": total,
                        "stream_duration_ms": (perf_counter() - stream_started) * 1000,
                    },
                )
                return
            except Exception as exc:
                if emitted:
                    raise StreamInterruptedError(
                        f"{provider.name} stream interrupted after partial output: {exc}"
                    ) from exc
                failed.append(provider.name)
        raise NoCompatibleProviderError(f"All compatible providers failed: {', '.join(failed)}")

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    def safe_status(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for provider in self.providers:
            provider.stats.roll_day()
            result.append(
                {
                    "name": provider.name,
                    "configured": provider.configured,
                    "available": provider.available(),
                    "capabilities": sorted(provider.capabilities),
                    "session_request_count": provider.stats.session_requests,
                    "requests_today": provider.stats.requests_today,
                    "failures": provider.stats.failures,
                    "average_latency_ms": provider.stats.average_latency_ms,
                }
            )
        return result
