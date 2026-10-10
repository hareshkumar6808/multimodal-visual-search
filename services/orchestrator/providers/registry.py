import logging
from collections.abc import Callable
from time import perf_counter

from services.orchestrator.config import Settings
from services.orchestrator.experts import Expert
from services.orchestrator.providers.base import Provider, ProviderError, ProviderResult
from services.orchestrator.providers.gemini import GeminiProvider
from services.orchestrator.providers.openai_compatible import OpenAICompatibleProvider
from services.orchestrator.routing.adaptive_provider_policy import (
    AdaptiveProviderPolicy,
    ProviderPolicySettings,
)
from services.orchestrator.routing.provider_performance import ProviderPerformanceStore


class NoCompatibleProviderError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(
        self,
        providers: list[Provider],
        performance: ProviderPerformanceStore | None = None,
        policy_settings: ProviderPolicySettings | None = None,
    ) -> None:
        self.providers = providers
        self.performance = performance
        self.policy = AdaptiveProviderPolicy(performance, policy_settings)
        self.last_policy_reason = "NOT_RANKED"

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRegistry":
        performance = ProviderPerformanceStore(
            settings.provider_performance_db_path, alpha=settings.provider_ewma_alpha
        )
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
            ],
            performance,
            ProviderPolicySettings(
                ttft_weight=settings.provider_ttft_weight,
                total_latency_weight=settings.provider_total_latency_weight,
                failure_penalty_ms=settings.provider_failure_penalty_ms,
                min_samples=settings.provider_min_samples,
            ),
        )

    def ranked(
        self,
        expert: Expert,
        image_required: bool,
        *,
        prefer_cloud: bool = False,
        local_only: bool = False,
        task_kind: str | None = None,
    ) -> list[Provider]:
        ranked, reason = self.policy.rank(
            self.providers,
            expert,
            image_required=image_required,
            prefer_cloud=prefer_cloud,
            local_only=local_only,
            task_kind=task_kind or ("vision" if image_required else "text"),
        )
        self.last_policy_reason = reason
        return ranked

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
        local_only: bool = False,
        task_kind: str | None = None,
        on_progress: Callable[[str, str, str], None] | None = None,
    ) -> tuple[str, ProviderResult, list[str], bool]:
        resolved_task_kind = task_kind or ("vision" if image_required else "text")
        candidates = self.ranked(
            expert,
            image_required,
            prefer_cloud=prefer_cloud,
            local_only=local_only,
            task_kind=resolved_task_kind,
        )
        # A visual route first considers only providers that can receive pixels. If all
        # of those fail, preserve the Stage 1 offline path by allowing a local text model
        # to answer from OCR/context without ever receiving the image.
        if image_required:
            text_fallbacks = self.ranked(
                expert,
                False,
                prefer_cloud=False,
                local_only=True,
                task_kind=f"{resolved_task_kind}:ocr-fallback",
            )
            candidate_names = {provider.name for provider in candidates}
            candidates.extend(
                provider
                for provider in text_fallbacks
                if provider.name not in candidate_names and "vision" not in provider.capabilities
            )
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
                attempt_started = perf_counter()
                try:
                    result = await provider.generate(
                        provider_prompt, image_bytes if send_image else None, mime_type
                    )
                    if self.performance:
                        self.performance.record(
                            provider.name,
                            resolved_task_kind,
                            total_ms=(perf_counter() - attempt_started) * 1000,
                            success=True,
                        )
                    if on_progress:
                        on_progress(
                            "provider",
                            "complete",
                            f"{provider.name} responded in {result.latency_ms / 1000:.1f}s",
                        )
                    return provider.name, result, failures, cloud_image_uploaded
                except ProviderError as exc:
                    if self.performance:
                        self.performance.record(
                            provider.name,
                            resolved_task_kind,
                            total_ms=(perf_counter() - attempt_started) * 1000,
                            success=False,
                        )
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

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()
        if self.performance:
            self.performance.close()

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
