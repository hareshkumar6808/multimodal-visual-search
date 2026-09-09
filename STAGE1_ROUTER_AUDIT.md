# Stage 1 Adaptive Router Audit

## Overall Status

**PASS WITH WARNINGS**

The Stage 1 orchestration architecture is implemented, runnable, tested without cloud credentials, and ready for integration work. Production use still requires the real `context-perception` implementation and configured providers. Live cloud calls were intentionally skipped because no provider keys were present.

Audited branch: `adaptive-router`  
Original published head: `e01e879f026b6c1d9ee2dd82f440d6a33aebd910`

## Tests Performed

| Check | Result | Notes |
| --- | --- | --- |
| Git branch and working tree | PASS | Began on clean `adaptive-router`; no branch switch or reset performed |
| Branch diff and tracked files | PASS | Stage 1 source, tests, documentation, and empty `.env.example` only |
| Secret and unsafe-code scan | PASS | No credential values, `eval`, `exec`, `shell=True`, unsafe pickle use, personal paths, or `/tmp` assumptions found |
| Clean environment installation | PASS | Editable install with development dependencies succeeded in a new repository-local virtual environment |
| Import and server startup | PASS | Documented module imported and Uvicorn started on `127.0.0.1:8765` without cloud keys |
| Live `/api/health` | PASS | HTTP 200; orchestrator, explicit mock perception, and provider configuration status returned without secrets |
| Live `/api/providers` | PASS | HTTP 200; NVIDIA, Gemini, and local providers safely reported unconfigured/unavailable |
| Live `/api/analyze` suggestion request | PASS | HTTP 200; request ID preserved, nullable query mapped to `suggest`, zero provider calls |
| Analyze input errors | PASS | Missing image and malformed JSON returned 422; oversized input returned 413; spoofed/malformed image bytes returned 422 |
| No-provider behavior | PASS | HTTP 503 with controlled message; no fabricated answer |
| MIR v0.1 contract | PASS | Agreed structure accepted; version `0.2`, unknown modality, malformed output, and request-ID mismatch rejected |
| Intent classification | PASS | All ten Stage 1 intents verified locally without model calls |
| Expert routing | PASS | Text, code, table, chart, vision, mixed, and low-confidence/general behavior verified |
| Minimum-sufficient representation | PASS | Text, code, and extracted-chart routes withheld bytes; visual routes required a vision-capable provider |
| Provider API shape | PASS | NVIDIA OpenAI-compatible and Gemini `generateContent` request shapes exercised with mock HTTP transports |
| Provider fallback | PASS | Connection error, timeout, HTTP 429, HTTP 500, and malformed response all activated a compatible fallback |
| Provider exhaustion | PASS | All unavailable/failed cases return controlled failures |
| Trace and/duration contract | PASS | Events remain request-local, ordered, factual, and contain measured non-negative timing values |
| Sequential performance smoke | PASS | 500 mock requests in 113.41 ms total; 0.227 ms mean orchestration time; provider count exactly 500 |
| Sequential memory smoke | PASS | About 94 KB retained and 107 KB peak under `tracemalloc`; no repeated registry/provider initialization observed |
| Concurrency smoke | PASS | Concurrent requests retained distinct IDs and traces; shared provider count remained consistent |
| Automated test suite | PASS | 37 passed, 0 failed; no real API keys or quota used |
| Ruff lint and format | PASS | No findings |
| Strict mypy | PASS | No findings across `contracts` and `services` |

The test environment used Python 3.14. Dependency deprecation warnings came from the current FastAPI TestClient/Starlette and pytest-asyncio compatibility layers; they did not indicate application test failures.

## Architecture Verification

The implementation follows the required pipeline:

```text
multipart desktop request
  -> PerceptionAdapter
  -> validated MIR v0.1
  -> local intent classifier
  -> isolated RuleRouter
  -> expert-specific prompt/representation
  -> ProviderRegistry
  -> provider adapter and fallback
  -> deterministic validation
  -> structured response, trace, and measured metrics
```

Provider-specific HTTP syntax remains inside provider adapters. Routing does not import OCR or provider HTTP details. Experts are registry descriptions of specialized pipelines; there is no agent-to-agent framework or autonomous behavior.

The real perception boundary supports synchronous or asynchronous `analyze_capture(image_bytes, context)` functions. Synchronous perception now runs in a worker thread so it cannot block the FastAPI event loop. Mock perception remains explicitly labeled and is forbidden outside development/test mode.

## API Verification

| Endpoint | Result | Verified behavior |
| --- | --- | --- |
| `GET /api/health` | PASS | Safe orchestrator, perception, and provider state; degraded state when real perception is absent |
| `GET /api/providers` | PASS | Safe capability and local metric fields; absent keys do not crash startup |
| `POST /api/analyze` | PASS | Multipart contract, image and JSON validation, request-ID preservation, nullable query, routing, fallback, response schema, and controlled errors |

The maximum image body is 20 MiB. PNG, JPEG, GIF, BMP, and WebP content signatures receive basic structural validation. The detected content type, rather than the user-provided filename, is passed to providers.

## Routing Matrix

| MIR / condition | Intent example | Verified expert | Image sent |
| --- | --- | --- | --- |
| `text`, visual false | summarize | `text-expert` | No |
| `code`, visual false | debug | `code-expert` | No |
| `table`, visual false | extract | `table-expert` | No |
| `chart`, visual false | explain | `chart-expert` | No |
| `chart`, visual true | explain | `chart-expert` with vision provider | Yes |
| `image`, visual true | identify | `vision-expert` | Yes |
| `mixed`, visual true | general | `vision-expert` | Yes |
| Low perception confidence | any | `general-expert` | Mirrors MIR visual requirement |
| Empty or null query | suggest | modality expert, local suggestions | No provider call |

The complete code scenario used `arr = [1, 2, 3]` and `print(arr[4])`. It produced `debug -> code-expert -> extracted text -> provider`, preserved the request ID, and reported `cloud_image_uploaded=false`.

## Provider Status

| Provider | Implementation | Text | Vision | Configuration during audit |
| --- | --- | --- | --- | --- |
| NVIDIA | OpenAI-compatible async HTTP adapter | Yes | Configurable per selected model | Not configured; live call skipped |
| Gemini | REST `generateContent` async HTTP adapter | Yes | Yes | Not configured; live call skipped |
| Local | OpenAI-compatible async HTTP adapter | Yes | Configurable per selected model | Not configured |
| Mock | Deterministic test adapter | Yes | Configurable | Test/development only |

Every external HTTP client has a finite configurable timeout and is reused across requests, then closed during FastAPI shutdown. Provider selection considers compatibility, vision requirements, local budget, failures, and observed latency. Local providers are preferred when compatible. Fallback errors are logged by request ID, provider, and error category without prompts, OCR content, images, headers, or credentials.

`cloud_image_uploaded` is derived from actual cloud provider attempts. It remains true if a cloud image attempt fails and a later provider succeeds, avoiding a false privacy claim.

## Security Findings

- No secrets or populated `.env` file are tracked.
- `.env` and local virtual environments are ignored.
- Provider keys come only from environment-backed settings and are absent from status responses and logs.
- The API does not use uploaded filenames as filesystem paths and performs no upload file writes.
- Upload size and supported image-content checks are enforced before perception.
- Default documentation binds Uvicorn to loopback address `127.0.0.1`.
- User-facing provider and perception errors are controlled; exception causes remain server-side.

## Performance Findings

Provider registries, provider adapters, and HTTP clients are initialized once per application instance. A 500-request sequential mock run completed in 113.41 ms total with 0.227 ms average orchestration overhead. The 94 KB retained allocation is consistent with interpreter/test bookkeeping and did not grow per request in an obviously unbounded way. These figures measure mock orchestration only and do not predict perception or network latency.

Concurrent mock requests completed without mixed IDs or traces. Shared counters were correct for the tested single-process async execution model.

## Issues Found

| Severity | Problem | Root cause | Fix |
| --- | --- | --- | --- |
| Medium | `query: null` failed validation | Request schema allowed only strings | Made query nullable and normalized null/empty values to `suggest` |
| Medium | Declared MIME type was trusted | Upload path checked `Content-Type` only | Added bounded content-signature and basic structure validation with canonical MIME detection |
| Medium | Sync perception could block all async requests | Adapter invoked a sync function on the event-loop thread | Run synchronous perception in AnyIO's worker thread |
| Medium | Malformed perception output could become an uncontrolled 500 | Pydantic validation errors escaped the adapter | Wrap malformed/incompatible MIR as a safe perception failure |
| Medium | Failed cloud vision attempt could be omitted from upload metric | Metric reflected only the successful provider | Aggregate actual cloud image attempts across the fallback chain |
| Low | Provider clients were recreated for every request | `AsyncClient` lived inside each generate call | Reuse per-provider clients and close them on application shutdown |
| Low | Chart expert metadata said images were always required | Metadata conflicted with extracted-chart routing | Marked chart images conditional and retained MIR-driven visual routing |
| Low | “Find information” missed the search intent | Keyword set lacked the agreed phrase | Added the phrase and complete intent matrix tests |
| Low | Suggested-action labels differed from the agreed UI contract | Initial labels/order were approximate | Aligned code, table, and image suggestions |
| Low | Numeric runtime settings accepted invalid ranges | Settings lacked bounds | Added Pydantic bounds for thresholds, budgets, and timeout |

## Fixes Applied

- Hardened request, image, MIR, and configuration validation.
- Made the perception integration async-safe and failure-safe.
- Corrected nullable empty-query behavior and suggestion labels.
- Corrected cloud upload accounting across fallbacks.
- Reused and cleanly closed external HTTP clients.
- Added safe operational logs using identifiers and categories only.
- Expanded tests for all intents, route variants, real adapter contracts, input failures, five fallback failure modes, concurrency, metrics, and full mock code flow.
- Updated documentation and generated this audit report.

## Remaining Limitations

- Real perception is deliberately absent from this component branch and must be supplied by `context-perception`.
- No cloud provider was called because no keys were present; request formats were verified with mock transports against current documented APIs.
- Provider counters and quota budgets are process-local and reset after restart.
- Health availability means configured and within the local budget; it is not a live provider network probe.
- Image checks provide bounded Stage 1 structural validation, not full forensic decoding. The perception implementation must still handle decoder errors safely.
- Intent and route selection remain explicit Stage 1 heuristics.
- Trace uses a single `provider` stage with factual failure/completion events rather than separate selection and execution stage names.
- Multi-process counter synchronization, persistent metrics, authentication, and advanced privacy routing remain outside Stage 1.
- Python 3.14 currently emits dependency deprecation warnings in tests; Python 3.11-3.13 is the lower-risk runtime range until those libraries complete their 3.14 transitions.

## Stage 1 Merge Readiness

**READY TO MERGE WITH context-perception / desktop-interface**

The contracts and adapter boundary are ready for integration. Integration testing must confirm the real perception function returns MIR v0.1 with the same request ID and that the desktop sends the documented multipart fields. Production release remains conditional on those integration tests and at least one configured provider or local inference service.
