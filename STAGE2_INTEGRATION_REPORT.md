# Stage 2 Integration Report

## Integrated sources

The `stage2-integration` branch was created from Stage 1 commit
`66e07f5023b51e1e53eecb131da2e8eea4498ab1` and contains normal merge commits
for all three Stage 2 components:

- Representation selector: `6e80c6a6f3d41ecaaf142ea21d4b0aa3d2d56022`
- Unified ML router: `c8af37c6f27633709f7dd49a09e4bc7ce5f911a5`
- Expert execution and streaming: `9a47ad7927aa7040aa157e44247d9e8e10ae2de2`

All component branches have the Stage 1 commit as their merge base.

## Conflict resolution and architecture

The overlapping contracts, dependency declarations, orchestrator service, and
provider registry were reconciled semantically. MIR v0.1 remains loadable, while
MIR v0.2 retains representation metadata. The final route contract carries the
parent expert, sub-expert, canonical leaf route, confidence and provenance,
fallback reason, execution mode, pixel decision, requested tool, and ranked
provider candidates.

The runtime flow is capture and local perception, advisory representation
selection, MIR v0.2, query-aware unified routing, execution policy, expert/tool
execution or adaptive provider selection, and streamed output. Person 2 decides
the route and execution policy; Person 3 executes it. The original pixels remain
available until the query-aware policy makes the final `send_pixels` decision.

The provider registry preserves capability, health, privacy, budget, and vision
filters before EWMA performance ranking. Streaming can fall back before visible
output; after visible output it preserves the partial answer and reports the
interruption rather than silently restarting.

## Representation selector

A lightweight representation selector was trained and deployed with low
inference overhead. Cross-source evaluation revealed that query-independent
pixel sufficiency did not generalize safely enough for authoritative pixel
discard decisions. The model is therefore retained as advisory perception
metadata, while the query-aware router makes the final image-transmission
decision.

The packaged MobileNetV3-Small ONNX artifact loads offline and returned four
normalized scores summing to 1.0. The observed local inference latency was
2.280 ms p50 and 2.723 ms p95.

## Unified router

The packaged calibrated TF-IDF/LinearSVC artifact loads offline. Its recorded
evaluation report is approximately 0.9834 accuracy, 0.9717 macro F1, 0.9486
coverage, and 0.9965 accepted-route accuracy. Integration smoke tests exercised
biomedical, legal, finance, SQL, geometry, and probability prompts. The observed
warm route latency was 1.800 ms p50 and 3.453 ms p95.

Low-confidence decisions abstain to the family-general route. A narrow
high-signal evidence check prevents short biomedical captures with multiple
domain terms from being discarded by the stored text threshold while retaining
the ML artifact as the semantic router.

## Expert execution

The integrated Expert Registry supports direct, tool, local LLM, cloud text,
and cloud vision execution. Deterministic paths include OCR return, SymPy math,
table aggregation, Python AST analysis, and SQL static analysis. User code is
never executed. The math normalizer accepts common OCR notation such as
`x^2 - 5x + 6 = 0`; the verified solution is 2 and 3 without an API call.

Initial and follow-up streaming endpoints preserve conversation context, reuse
the MIR on follow-up, rerun query-aware routing, and emit provider TTFT, provider
total latency, and time-to-first-useful-output metrics. Existing non-streaming
endpoints remain available.

## Validation

- Backend: 199 passed, 7 skipped.
- Ruff: all lint checks passed.
- Mypy: strict mode passed for 53 source files.
- Frontend typecheck and ESLint: passed.
- Frontend tests: 3 files and 19 tests passed.
- Vite production build: passed.
- Cargo check: passed with the Windows GNU toolchain.
- SQLite migration and adaptive provider persistence: covered by passing tests.
- MobileNet ONNX and task-router joblib artifacts: loaded and executed offline.

Repository-wide format checking still identifies 11 pre-existing files from the
representation research component. Integration-touched Python files pass the
formatter; unrelated research files were intentionally not mass-formatted.

## Local latency sample

The lightweight benchmark used warm local runs and records its machine-readable
output under `.tools/stage2/benchmark.json`:

| Operation | p50 | p95 |
| --- | ---: | ---: |
| Representation selection | 2.280 ms | 2.723 ms |
| Unified route decision | 1.800 ms | 3.453 ms |
| Direct OCR return overhead | 0.002 ms | 0.002 ms |
| SymPy solve | 1.035 ms | 1.728 ms |
| Table maximum | 0.001 ms | 0.001 ms |

Provider network latency is environment-dependent and was not presented as a
controlled benchmark.

## Launch and operational notes

Stage 2 provides setup, run, stop, full restart, and backend-only restart
scripts. They track project-owned PIDs, reject occupied ports rather than killing
unrelated processes, retain logs under `.tools/stage2/logs`, reuse the existing
Stage 1 Qwen/llama.cpp assets, wait for local provider and backend health, and
then start the native Tauri desktop.

The current managed Windows environment required local compatibility adjustments
inside its cached Rust crates because sandbox path canonicalization returns
`Access is denied`. This is an execution-host limitation rather than a repository
runtime behavior; no cached-crate modification is committed to the project.

Known limits are that OCR quality depends on capture resolution and installed
Tesseract language data, cloud answers depend on provider availability and
credentials, and a text-only local Qwen fallback cannot truthfully inspect image
pixels. Vision requests retain their pixels and report that limitation when no
vision-capable cloud provider is available.
