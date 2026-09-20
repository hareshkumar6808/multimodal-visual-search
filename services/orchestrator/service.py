import logging
from time import perf_counter

from contracts.models import (
    AnalyzePayload,
    AnalyzeResponse,
    Metrics,
    MIRSummary,
    RouteInfo,
    TraceEvent,
)
from services.orchestrator.experts import SUGGESTIONS
from services.orchestrator.intent import classify_intent
from services.orchestrator.perception import PerceptionAdapter
from services.orchestrator.prompting import build_prompt
from services.orchestrator.providers.registry import ProviderRegistry
from services.orchestrator.routing import RuleRouter
from services.orchestrator.validation import ResponseValidationError, validate_answer

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self, perception: PerceptionAdapter, router: RuleRouter, providers: ProviderRegistry
    ) -> None:
        self.perception = perception
        self.router = router
        self.providers = providers

    async def analyze(
        self, image_bytes: bytes, mime_type: str, payload: AnalyzePayload
    ) -> AnalyzeResponse:
        total_started = perf_counter()
        trace = [TraceEvent(stage="capture_received", status="complete")]

        perception_started = perf_counter()
        mir = await self.perception.analyze_capture(
            image_bytes, payload.context, payload.request_id
        )
        logger.info(
            "PERCEPTION_COMPLETE request_id=%s modality=%s confidence=%.2f",
            payload.request_id,
            mir.primary_modality,
            mir.overall_confidence,
        )
        logger.info("MIR_CREATED request_id=%s version=%s", payload.request_id, mir.mir_version)
        perception_ms = round((perf_counter() - perception_started) * 1000)
        trace.append(
            TraceEvent(
                stage="perception",
                status="complete",
                message=f"{mir.primary_modality.title()} detected",
                confidence=mir.overall_confidence,
                duration_ms=perception_ms,
            )
        )

        intent = classify_intent(payload.query, mir.primary_modality)
        logger.info("INTENT_SELECTED request_id=%s intent=%s", payload.request_id, intent)
        trace.append(TraceEvent(stage="intent", status="complete", message=intent.title()))

        routing_started = perf_counter()
        expert_route = self.router.route(mir, intent)
        logger.info(
            "EXPERT_SELECTED request_id=%s expert=%s",
            payload.request_id,
            expert_route.expert.name,
        )
        routing_ms = round((perf_counter() - routing_started) * 1000)
        trace.append(
            TraceEvent(
                stage="routing",
                status="complete",
                message=f"{expert_route.expert.name} selected",
                confidence=expert_route.routing_certainty,
                duration_ms=routing_ms,
            )
        )

        if intent == "suggest":
            trace.append(
                TraceEvent(
                    stage="provider", status="skipped", message="Suggestions generated locally"
                )
            )
            return AnalyzeResponse(
                request_id=payload.request_id,
                answer=None,
                suggested_actions=SUGGESTIONS[mir.primary_modality],
                mir_summary=MIRSummary(
                    primary_modality=mir.primary_modality, confidence=mir.overall_confidence
                ),
                route=RouteInfo(
                    intent=intent,
                    expert=expert_route.expert.name,
                    provider=None,
                    reason_code="EMPTY_QUERY_LOCAL_SUGGESTIONS",
                ),
                trace=trace,
                metrics=Metrics(
                    latency_ms=round((perf_counter() - total_started) * 1000),
                    perception_ms=perception_ms,
                    routing_ms=routing_ms,
                    provider_ms=0,
                    cloud_image_uploaded=False,
                    api_calls=0,
                ),
            )

        prompt = build_prompt(expert_route.expert, mir, payload)
        (
            provider_name,
            result,
            failed_providers,
            cloud_image_uploaded,
        ) = await self.providers.generate_with_fallback(
            expert_route.expert,
            prompt,
            expert_route.image_required,
            image_bytes,
            mime_type,
            payload.request_id,
        )
        logger.info(
            "PROVIDER_SELECTED request_id=%s provider=%s", payload.request_id, provider_name
        )
        logger.info(
            "PROVIDER_RESPONSE request_id=%s provider=%s latency_ms=%d",
            payload.request_id,
            provider_name,
            result.latency_ms,
        )
        for failed in failed_providers:
            trace.append(
                TraceEvent(
                    stage="provider", status="failed", message=f"{failed} failed; trying fallback"
                )
            )
        trace.append(
            TraceEvent(
                stage="provider",
                status="complete",
                message=f"{provider_name} completed",
                duration_ms=result.latency_ms,
            )
        )
        valid, validation_message = validate_answer(result.text, expert_route.expert.name, mir)
        trace.append(
            TraceEvent(
                stage="validation",
                status="complete" if valid else "failed",
                message=validation_message,
            )
        )
        if not valid:
            raise ResponseValidationError(validation_message)

        total_ms = round((perf_counter() - total_started) * 1000)
        logger.info(
            "REQUEST_COMPLETE request_id=%s expert=%s provider=%s latency_ms=%d",
            payload.request_id,
            expert_route.expert.name,
            provider_name,
            total_ms,
        )

        return AnalyzeResponse(
            request_id=payload.request_id,
            answer=result.text,
            suggested_actions=[],
            mir_summary=MIRSummary(
                primary_modality=mir.primary_modality, confidence=mir.overall_confidence
            ),
            route=RouteInfo(
                intent=intent,
                expert=expert_route.expert.name,
                provider=provider_name,
                reason_code=expert_route.reason_code,
            ),
            trace=trace,
            metrics=Metrics(
                latency_ms=total_ms,
                perception_ms=perception_ms,
                routing_ms=routing_ms,
                provider_ms=result.latency_ms,
                cloud_image_uploaded=cloud_image_uploaded,
                api_calls=len(failed_providers) + 1,
            ),
        )
