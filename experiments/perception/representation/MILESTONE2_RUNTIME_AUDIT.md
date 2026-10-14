# Milestone 2 runtime decision-path audit

## Baseline behavior verified in code

The capture endpoint passes the uploaded image bytes to Orchestrator.analyze. RealPerceptionAdapter calls the local services.perception.analyze_capture; PerceptionPipeline runs preflight, OCR, modality profiling, extraction, Stage 1 visual_required, and the Stage 2 selector. It serializes both signals into MIR v0.2; MIR v0.1 remains accepted with representation=None.

Before this milestone, Orchestrator._answer_turn sent MIR only to RuleRouter. The router set ExpertRoute.image_required from Stage 1 visual_required and modality-specific rules; it did not read MIR.representation. ProviderRegistry.generate_with_fallback sent image bytes only when that route flag was true and a selected provider advertised vision capability. Therefore Stage 2 predictions did not affect actual forwarding, while Stage 1 could suppress image forwarding even when Stage 2 indicated mixed or pixels_required. When no visual route was selected, providers received OCR/context text only. Conversation storage independently retained the original capture bytes.

Relevant code: services/perception/pipeline.py, services/orchestrator/service.py, services/orchestrator/routing.py, services/orchestrator/providers/registry.py, and services/orchestrator/conversations.py.

## Milestone 2 policy and precedence

The orchestrator now evaluates decide_image_preservation(mir, query) after intent/expert selection and before provider invocation. For every non-empty content request, the policy retains original pixels:

1. Missing representation metadata, including valid MIR v0.1: preserve.
2. Stage 1 visual dependency or Stage 2 pixels_required/mixed: preserve.
3. text_sufficient or structure_sufficient: also preserve because content classification is not query-specific sufficiency evidence.
4. The selector does not receive the query and no validated answer-equivalence benchmark exists. Consequently, no class currently authorizes omission.

When the route did not already require an image, the policy upgrades its image_required flag and records a representation_policy trace event and route reason. The existing provider registry forwards the unchanged original capture bytes to a vision-capable selected provider. If it falls back to a text-only provider, that provider receives no image and receives the existing explicit offline prompt that it must answer only from OCR/context or say what is unknown. The image remains stored with the conversation for follow-up. This text-only fallback is a capability limitation; successful forwarding is asserted for a vision-capable route.

Empty-query local suggestions make no provider request; capture storage still retains the image. Preflight/MIR validation failures fail the request instead of silently routing an uncertain or malformed representation as text.

The policy is deliberately conservative and does not reduce resolution, crop, recompress, or delete pixels. It separates content class from task sufficiency while preserving the existing expert choice, MIR schemas, and provider ranking.

## Evaluation validity and model decision

The existing MobileNetV3-Small and ONNX artifacts are retained. Existing DocLayNet-small labels are layout-derived proxies, not task-sufficiency ground truth. The source validation report has only 9 proxy-critical examples (pixels_required 7, mixed 2), with 2 composed-policy critical misses; the historical 49-page test split has been exposed during earlier development and is diagnostic. Existing report examples include a portrait-heavy directory and an aircraft photograph misclassified as text-sufficient while Stage 1 also returned visual_required=False.

No retraining, threshold tuning, or new dataset is justified by these findings: the missing evidence is task-specific sufficiency, not model capacity. Dataset provenance, license review, split counts, prior model metrics, ONNX parity, and measured latency remain in RESEARCH_PLAN.md, RESEARCH_REPORT.md, README.md, and the committed JSON reports. No new dataset was fetched for this milestone.

## Verification scope and limitations

Automated tests exercise all four MIR representation classes, legacy MIR v0.1 without the Stage 2 field, the policy trace/reason, and passage of the original byte payload to a vision-capable mock provider (verified by SHA-256 at provider invocation). Existing selector tests cover missing/corrupt artifacts, invalid scores and conservative pixel fallback; model/data fetching stays in explicit experiment scripts, outside runtime imports.

This proves the local orchestration boundary with test doubles, not external provider ingestion, semantic answer equivalence, or real-user query sufficiency. Text-only fallback cannot see original pixels. The application therefore does not claim representation reduction is safe or integration-ready for tasks requiring a vision-capable provider. Existing sample sizes and proxy labels cannot support a real-world false-discard risk estimate.
