import logging
from time import perf_counter
from uuid import uuid4

from contracts.models import (
    MIR,
    AnalyzePayload,
    AnalyzeResponse,
    ChatPayload,
    Metrics,
    MIRSummary,
    RouteInfo,
    TraceEvent,
)
from services.orchestrator.conversations import ConversationStore
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
        self,
        perception: PerceptionAdapter,
        router: RuleRouter,
        providers: ProviderRegistry,
        conversations: ConversationStore | None = None,
    ) -> None:
        self.perception = perception
        self.router = router
        self.providers = providers
        self.conversations = conversations or ConversationStore()

    async def analyze(
        self, image_bytes: bytes, mime_type: str, payload: AnalyzePayload
    ) -> AnalyzeResponse:
        total_started = perf_counter()
        conversation_id = payload.conversation_id or str(uuid4())
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
        self.conversations.save_capture(
            conversation_id, payload.request_id, mir, image_bytes, mime_type
        )
        return await self._answer_turn(
            mir=mir,
            payload=payload,
            conversation_id=conversation_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            history=[],
            trace=trace,
            perception_ms=perception_ms,
            total_started=total_started,
            user_has_image=True,
        )

    async def chat(self, payload: ChatPayload) -> AnalyzeResponse:
        total_started = perf_counter()
        mir, image_bytes, mime_type, _capture_id = self.conversations.capture_context(
            payload.conversation_id
        )
        history = self.conversations.prompt_history(payload.conversation_id)
        analyze_payload = AnalyzePayload(
            request_id=payload.request_id,
            conversation_id=payload.conversation_id,
            query=payload.query,
            context=payload.context,
        )
        trace = [
            TraceEvent(
                stage="context_reused",
                status="complete",
                message="Existing OCR and MIR reused; perception skipped",
                duration_ms=0,
            )
        ]
        return await self._answer_turn(
            mir=mir,
            payload=analyze_payload,
            conversation_id=payload.conversation_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            history=history,
            trace=trace,
            perception_ms=0,
            total_started=total_started,
            user_has_image=False,
        )

    async def _answer_turn(
        self,
        *,
        mir: MIR,
        payload: AnalyzePayload,
        conversation_id: str,
        image_bytes: bytes,
        mime_type: str,
        history: list[tuple[str, str]],
        trace: list[TraceEvent],
        perception_ms: int,
        total_started: float,
        user_has_image: bool,
    ) -> AnalyzeResponse:
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
        message_id = str(uuid4())

        if intent == "suggest":
            suggestions = SUGGESTIONS[mir.primary_modality]
            trace.append(
                TraceEvent(
                    stage="provider", status="skipped", message="Suggestions generated locally"
                )
            )
            response = AnalyzeResponse(
                request_id=payload.request_id,
                conversation_id=conversation_id,
                message_id=message_id,
                answer=None,
                suggested_actions=suggestions,
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
            self.conversations.append_exchange(
                conversation_id,
                payload.query or "",
                "Suggested actions: " + ", ".join(suggestions),
                response,
                user_has_image=user_has_image,
                assistant_id=message_id,
            )
            return response

        prompt = build_prompt(expert_route.expert, mir, payload, history)
        prefer_cloud = (
            expert_route.image_required
            or expert_route.expert.name
            in {"code-expert", "table-expert", "chart-expert", "vision-expert"}
            or intent in {"explain", "debug", "compare", "calculate", "search", "general"}
            or len(history) >= 4
            or len(mir.ocr.text) > 1_200
        )
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
            prefer_cloud=prefer_cloud,
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
        response = AnalyzeResponse(
            request_id=payload.request_id,
            conversation_id=conversation_id,
            message_id=message_id,
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
        self.conversations.append_exchange(
            conversation_id,
            payload.query or "",
            result.text,
            response,
            user_has_image=user_has_image,
            assistant_id=message_id,
        )
        return response
