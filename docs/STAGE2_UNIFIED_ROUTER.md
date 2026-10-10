# Stage 2 Unified Task Router

The router answers one latency-oriented question: what is the cheapest valid way to handle the current query and screen representation? It consumes MIR v0.1 or v0.2, the current query, and the existing Stage 1 intent. It returns a `RouteDecisionV2` with the broad expert, canonical leaf route, confidence, source, fallback state, execution mode, pixel decision, requested tool, and measured router timings.

There is one learned semantic model. A TF-IDF pipeline classifies the nine ambiguous specialized leaves. Table, chart, image, and UI have only one child and therefore route deterministically. Text, code, and math use validation-selected confidence thresholds; a low-confidence result becomes that family's `general` leaf. This keeps uncertainty visible and avoids a second broad-modality model.

`RuleRouter` remains the fallback when ML is disabled, the artifact is absent or corrupt, prediction fails, or a family is unsupported. MIR v0.2-only fields are optional and feature absence is represented explicitly. Dataset names, source paths, expected routes, and expert names never enter model features.

For a follow-up, `OrchestrationService` loads the stored MIR and calls the router again with the new intent and query. It does not rerun OCR. Empty-query suggestion requests retain the Stage 1 local path and make zero provider calls.

The production artifact is `services/orchestrator/artifacts/task_router_v1.joblib`. Regenerate it with:

```powershell
python scripts/stage2/router/prepare_data.py
python scripts/stage2/router/train.py
python scripts/stage2/router/evaluate.py
python scripts/stage2/router/benchmark.py
```

Person 3 should consume `execution_mode`, `requested_tool`, and `send_pixels` directly: execute `direct` or `tool` locally; use ranked candidates for provider modes; attach pixels only when `send_pixels` is true. The current Stage 1 service exposes this handoff while retaining its provider execution path until Person 3 integrates direct/tool execution and streaming.
