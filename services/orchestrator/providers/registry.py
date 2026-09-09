from services.orchestrator.config import Settings
from services.orchestrator.experts import Expert
from services.orchestrator.providers.base import Provider, ProviderError, ProviderResult
from services.orchestrator.providers.gemini import GeminiProvider
from services.orchestrator.providers.openai_compatible import OpenAICompatibleProvider


class NoCompatibleProviderError(RuntimeError):
    pass


class ProviderRegistry:
    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRegistry":
        return cls(
            [
                OpenAICompatibleProvider(
                    name="nvidia",
                    base_url=settings.nvidia_base_url,
                    model=settings.nvidia_model,
                    api_key=settings.nvidia_api_key,
                    supports_vision=settings.nvidia_supports_vision,
                    daily_budget=settings.nvidia_daily_budget,
                    timeout=settings.provider_timeout_seconds,
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
                    timeout=settings.provider_timeout_seconds,
                ),
            ]
        )

    def ranked(self, expert: Expert, image_required: bool) -> list[Provider]:
        compatible = [
            provider
            for provider in self.providers
            if provider.name in expert.compatible_providers
            and provider.available()
            and (not image_required or "vision" in provider.capabilities)
        ]

        # Prefer local processing, then low failure rates, unbounded/local quota,
        # remaining configured budget, and observed latency.
        def score(provider: Provider) -> tuple[float, float, float, float, float]:
            failure_rate = provider.stats.failures / max(provider.stats.session_requests, 1)
            latency = provider.stats.average_latency_ms or 0
            local_bonus = 1.0 if provider.name == "local" else 0.0
            bounded = 1.0 if provider.daily_budget > 0 else 0.0
            budget_used = (
                provider.stats.requests_today / provider.daily_budget
                if provider.daily_budget > 0
                else 0.0
            )
            return (-local_bonus, failure_rate, bounded, budget_used, latency)

        return sorted(compatible, key=score)

    async def generate_with_fallback(
        self,
        expert: Expert,
        prompt: str,
        image_required: bool,
        image_bytes: bytes,
        mime_type: str,
    ) -> tuple[str, ProviderResult, list[str]]:
        candidates = self.ranked(expert, image_required)
        if not candidates:
            raise NoCompatibleProviderError("No configured, available provider supports this route")
        failures: list[str] = []
        for provider in candidates:
            try:
                result = await provider.generate(
                    prompt, image_bytes if image_required else None, mime_type
                )
                return provider.name, result, failures
            except ProviderError:
                failures.append(provider.name)
        raise NoCompatibleProviderError(f"All compatible providers failed: {', '.join(failures)}")

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
