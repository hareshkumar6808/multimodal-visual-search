from __future__ import annotations

from dataclasses import dataclass

from services.orchestrator.experts import Expert
from services.orchestrator.providers.base import Provider
from services.orchestrator.routing.provider_performance import ProviderPerformanceStore


@dataclass(frozen=True)
class ProviderPolicySettings:
    ttft_weight: float = 0.6
    total_latency_weight: float = 0.4
    failure_penalty_ms: float = 8_000
    min_samples: int = 3


class AdaptiveProviderPolicy:
    """Filters invalid providers first, then ranks sufficiently measured providers by EWMA."""

    def __init__(
        self,
        performance: ProviderPerformanceStore | None,
        settings: ProviderPolicySettings | None = None,
    ) -> None:
        self.performance = performance
        self.settings = settings or ProviderPolicySettings()

    def rank(
        self,
        providers: list[Provider],
        expert: Expert,
        *,
        image_required: bool,
        prefer_cloud: bool,
        local_only: bool,
        task_kind: str,
    ) -> tuple[list[Provider], str]:
        eligible = [
            provider
            for provider in providers
            if (provider.family or provider.name) in expert.compatible_providers
            and provider.available()
            and (not image_required or "vision" in provider.capabilities)
            and (not local_only or not provider.is_cloud)
        ]
        if not eligible:
            return [], "NO_ELIGIBLE_PROVIDER"

        defaults = self._default_order(expert, image_required, prefer_cloud)
        default_priority = {name: index for index, name in enumerate(defaults)}

        measured = {
            provider.name: self.performance.get(provider.name, task_kind)
            if self.performance
            else None
            for provider in eligible
        }
        if not any(
            stat is not None and stat.sample_count >= self.settings.min_samples
            for stat in measured.values()
        ):
            return sorted(eligible, key=lambda p: self._static_key(p, default_priority, expert)), (
                "STAGE1_DEFAULT_INSUFFICIENT_HISTORY"
            )

        def adaptive_key(provider: Provider) -> tuple[int, int, float, int, float]:
            stat = measured[provider.name]
            preferred_tier = int(provider.is_cloud != prefer_cloud)
            specialization = 0 if expert.name in provider.preferred_experts else 1
            if stat is None or stat.sample_count < self.settings.min_samples:
                static = self._static_key(provider, default_priority, expert)
                return (preferred_tier, specialization, float("inf"), static[0], static[1])
            total = stat.ewma_total_ms or 0.0
            ttft = stat.ewma_ttft_ms if stat.ewma_ttft_ms is not None else total
            failure = (1.0 - stat.success_ewma) * self.settings.failure_penalty_ms
            score = (
                self.settings.ttft_weight * ttft
                + self.settings.total_latency_weight * total
                + failure
            )
            static = self._static_key(provider, default_priority, expert)
            return (preferred_tier, specialization, score, static[0], static[1])

        return sorted(eligible, key=adaptive_key), "EWMA_LATENCY_SUCCESS_RANKING"

    @staticmethod
    def _default_order(expert: Expert, image_required: bool, prefer_cloud: bool) -> tuple[str, ...]:
        if prefer_cloud:
            if image_required or expert.name in {"vision-expert", "chart-expert", "ui-expert"}:
                return ("gemini", "nvidia", "local")
            return ("nvidia", "gemini", "local")
        return ("local", "gemini", "nvidia")

    @staticmethod
    def _static_key(
        provider: Provider, priorities: dict[str, int], expert: Expert
    ) -> tuple[int, float, int]:
        family = provider.family or provider.name
        specialization = 0 if expert.name in provider.preferred_experts else 1
        failure_rate = provider.stats.failures / max(provider.stats.session_requests, 1)
        return (priorities.get(family, len(priorities)), failure_rate, specialization)
