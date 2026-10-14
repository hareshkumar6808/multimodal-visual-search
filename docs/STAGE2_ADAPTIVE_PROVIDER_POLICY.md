# Stage 2 Adaptive Provider Policy

Provider ranking first excludes providers that are unconfigured, unavailable, over budget, vision-incompatible, outside the expert's compatible families, or disallowed by local-only privacy. A fast invalid provider can never win. The Stage 1 static family/specialization/health order remains active until any eligible provider has at least three samples.

After warm-up, ranking uses persisted per-provider, per-task-kind exponentially weighted moving averages:

`score = ttft_weight × EWMA(TTFT) + total_weight × EWMA(total) + failure_penalty × (1 − success_EWMA)`

Defaults are alpha 0.25, TTFT weight 0.6, total-latency weight 0.4, failure penalty 8,000 ms, and minimum samples 3. All values are configurable through environment settings. If TTFT is unavailable, total latency substitutes for it. Statistics live in the bounded SQLite table `provider_performance`, keyed by provider and task kind, so learning survives backend restarts.

Visual routes rank pixel-capable providers only. If every pixel provider fails, Stage 1's offline safety path may append a local text model that receives OCR/context and no pixels. This fallback is not eligible to win the visual ranking itself.

No live provider comparison is reported because a reproducible credentialed benchmark was not part of this isolated training run. Deterministic tests verify capability and privacy filtering, static fallback, persisted EWMA ordering, fast-provider preference, and failure penalties. Real ranking adapts only from actual application requests.
