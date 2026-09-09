# Multimodal Visual Search

> Select anything on your screen, understand what it contains, and route it through the most appropriate AI model or tool.

A year-long academic and engineering project by a team of three, building a **Windows desktop application for conversational interaction with arbitrary screen content**.

The application is designed to analyze a selection, decompose mixed content into meaningful objects, choose the minimum sufficient representation, and route the user's request through specialized expert pipelines. The long-term goal is a desktop visual intelligence layer that makes visible content understandable, searchable, and interactive.

## Project status

**Planning stage.** This repository currently contains the project overview. The features and architecture below describe the intended system; application code, installation instructions, and runnable releases will be added as development progresses.

## Intended user experience

1. Click a small, draggable floating desktop icon.
2. Select a region using a Windows Snipping Tool-style overlay.
3. Open a compact chat window with the captured image already attached.
4. Ask a question or choose a context-specific suggested action.
5. Receive an answer, with an optional **Show Process** panel displaying real execution events.

Planned capture modes include rectangle, free-form, window, and full-screen selection. The desktop experience will also include history, pinned captures, settings, keyboard shortcuts, and privacy controls.

| Selected content | Example actions |
| --- | --- |
| Text or document | Explain, summarize, translate, verify |
| Code or error message | Explain, debug, identify errors |
| Table | Extract structured data, compare values, calculate |
| Chart | Explain trends, extract data, find anomalies |
| Equation | Extract notation, solve, explain |
| Image or object | Identify, search, find similar |
| Mixed content | Analyze related text, charts, and other objects together |

## Architecture

```text
Floating desktop widget
        |
Screen selection + context + optional query
        |
Local pre-flight processing
(OCR, image quality, cache lookup, privacy checks)
        |
Multimodal profiling + semantic decomposition
        |
Multimodal Intermediate Representation (MIR)
        |
Intent understanding + adaptive routing
        |
Specialized expert pipeline
        |
Model / deterministic tool / API selection
        |
Execution + conditional validation
        |
Answer + execution trace
        |
Chat interface + cache + feedback
```

The architecture separates five responsibilities:

- **Representation:** decide whether a task needs text, structured data, code, symbolic notation, original pixels, or a combination.
- **Expert:** a specialized processing pipeline for a task domain, such as tables, code, charts, mathematics, or visual search.
- **Tool:** a concrete operation such as OCR, retrieval, parsing, or symbolic calculation.
- **Model and provider:** replaceable inference capabilities and the services or local runtimes that expose them.
- **Router:** choose a representation, expert, model or tool, and fallback path using the request and available resources.

An expert may combine deterministic tools and models; it does not need to be an autonomous agent.

## Core engineering and research direction

### Semantic multimodal decomposition

Identify meaningful objects within a selection, such as a paragraph, chart, and equation, and process each with a suitable pipeline before combining the results.

### Multimodal Intermediate Representation (MIR)

Use a shared structured representation for extracted content, object relationships, context, and confidence. MIR is intended to connect perception, routing, experts, caching, and response generation without repeatedly interpreting the same screenshot.

### Minimum sufficient representation

Determine which information the question actually requires. A numerical table query may use extracted JSON, while a question about an object's appearance requires pixels. A query-dependent **visual dependency score** and information-loss estimate will guide these choices.

### Adaptive expert routing

Select pipelines using modality, intent, context, expected quality, latency, privacy, quota, reliability, and observed performance. Start with simple rules and heuristics, then evaluate whether a learned router improves decisions.

### Self-improving capability routing

Measure model and expert performance rather than relying on fixed assumptions about their strengths. Later research extensions include capability graphs, shadow benchmarking, and counterfactual comparison of routing choices.

These are proposed engineering contributions and research directions, not claims of established novelty or measured improvements. OCR, existing models, desktop frameworks, and symbolic tools are enabling technologies.

## Processing principles

- **Local processing first:** perform inexpensive extraction, quality checks, and cache lookups before cloud inference.
- **Preserve necessary information:** retain visual content when conversion would lose details required by the question.
- **Minimum necessary context:** gradually expand surrounding context only when the selection is ambiguous.
- **Privacy-aware routing:** keep sensitive content local, redact it, or transform it before external processing where appropriate.
- **Quota-aware operation:** target ₹0 in paid API spending using open-source tools, local models, and suitable free API tiers. Free quotas are finite and availability must be checked during integration.
- **Replaceable providers:** maintain practical fallback paths when a provider fails or quota is exhausted.
- **Confidence-guided escalation:** use stronger processing or additional validation when needed, rather than on every request.
- **Reusable understanding:** cache OCR, MIR, extracted structures, and other intermediate results for follow-up questions.
- **Observable execution:** display actual processing stages, routing decisions, latency, confidence, API usage, and upload status. The trace must reflect recorded system events.

## Proposed technology stack

These are preferred directions, not installed dependencies or finalized commitments.

| Layer | Candidate technologies |
| --- | --- |
| Windows desktop | Tauri, React, TypeScript, Rust |
| Backend and streaming | Python, FastAPI, WebSockets |
| Local image processing and OCR | OpenCV, PaddleOCR or another suitable OCR engine |
| Local inference | Open-source models through Ollama, llama.cpp, or Transformers |
| Cloud inference | Suitable free developer APIs, potentially including NVIDIA and Gemini, subject to availability |
| Deterministic mathematics | SymPy |
| Local storage | SQLite |
| Optional semantic retrieval | FAISS or another local vector-search solution |

Specific model names and providers should remain replaceable.

## Development roadmap

### Phase 1 — End-to-end MVP

- [ ] Floating desktop icon and screen-selection overlay
- [ ] Screenshot capture and attached-image chat interface
- [ ] Basic multimodal classification
- [ ] Basic routing to one or more local or free-tier models
- [ ] Answer display and basic execution telemetry

### Phase 2 — Structured perception and expert pipelines

- [ ] Define shared request, MIR, response, and trace schemas
- [ ] Add semantic decomposition and structured extraction
- [ ] Introduce specialized text, code, table, chart, math, and vision pipelines
- [ ] Add representation selection and extraction-confidence checks

### Phase 3 — Efficient and reliable orchestration

- [ ] Add quota tracking and provider fallbacks
- [ ] Introduce adaptive context expansion and semantic caching
- [ ] Add privacy-aware routing and confidence-guided escalation
- [ ] Validate low-confidence results when appropriate

### Phase 4 — Evaluation and advanced extensions

- [ ] Benchmark expert quality, latency, reliability, and quota efficiency
- [ ] Compare adaptive routing against a direct multimodal-model baseline
- [ ] Evaluate information loss from representation conversion
- [ ] Explore learned routing, capability graphs, and shadow benchmarking
- [ ] Add multi-selection reasoning
- [ ] Explore a persistent screen knowledge graph

The priority is a reliable end-to-end MVP before advanced extensions.

## Team ownership

| Owner | Responsibility | Main output |
| --- | --- | --- |
| Person A — Desktop Experience | Floating widget, Windows integration, capture overlay, chat, history, settings, execution-trace UI | Screenshot + context + query |
| Person B — Multimodal Perception | OCR, profiling, decomposition, structured extraction, MIR, visual dependency and loss estimation | MIR |
| Person C — Intelligence Orchestration | Backend APIs, registries, routing, provider integrations, quotas, fallbacks, caching, validation, evaluation | Answer + execution trace |

Shared schemas should be agreed early so the three subsystems can be developed independently.

## Evaluation plan

Evaluate the system on text, code, table, chart, equation, image, and mixed-content tasks. Compare routed processing with a direct screenshot-to-multimodal-model baseline using:

- Answer accuracy and task completion
- End-to-end latency
- API calls, token usage, and image uploads
- Quota consumption and paid API cost
- Information loss introduced by extraction or conversion
- Fallback success and failure rates
- Cache effectiveness and routing quality

Performance claims will be added only after measurement.

## Getting started

You can clone the repository to follow development:

```bash
git clone https://github.com/hareshkumar6808/multimodal-visual-search.git
cd multimodal-visual-search
```

There is no runnable application yet. Dependency setup, configuration, and Windows development instructions will be documented when the implementation is available.

## Stage 1 orchestrator backend

The `adaptive-router` branch contains the first runnable backend vertical slice. It receives a desktop capture, calls the perception boundary, classifies intent, selects an expert pipeline and compatible provider, and returns an answer with factual execution telemetry.

### Setup and run

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn services.orchestrator.main:app --host 127.0.0.1 --port 8765 --reload
```

The service defaults to the real perception adapter. Until the perception branch is merged, local development can explicitly set `APP_ENV=development` and `PERCEPTION_MODE=mock`. Mock perception is rejected when `APP_ENV` is neither `development` nor `test`.

### API

`POST /api/analyze` accepts `multipart/form-data` with an image file in `image` and a JSON string in `payload_json`:

```json
{
  "request_id": "abc-123",
  "query": "Why is this code failing?",
  "context": {
    "active_app": "Visual Studio Code",
    "window_title": "main.py",
    "selection_mode": "rectangle",
    "bounds": {"x": 100, "y": 100, "width": 800, "height": 500}
  }
}
```

The structured response contains `answer`, `suggested_actions`, a MIR summary, the selected intent/expert/provider and reason code, execution trace events, and latency/API/image-upload metrics. An empty query returns local modality-specific suggestions and makes no provider call.

`GET /api/health` reports orchestrator, perception adapter, and provider readiness. `GET /api/providers` returns only safe status and locally tracked counters; neither endpoint exposes credentials or claims knowledge of provider-side remaining quota.

### Perception integration

The real adapter imports the module named by `PERCEPTION_MODULE` (default `context_perception`) and calls:

```python
analyze_capture(image_bytes: bytes, context: dict) -> dict
```

The function may be synchronous or asynchronous and must return valid MIR v0.1. Canonical Pydantic contracts live in `contracts/models.py`. The adapter rejects mismatched request IDs and malformed MIR instead of substituting mock output.

### Experts and routing

The Stage 1 expert registry declares supported modalities, intents, image requirements, preferred representations, and provider compatibility for:

- `text-expert`
- `code-expert`
- `table-expert`
- `chart-expert`
- `vision-expert`
- `general-expert`

Transparent rules route code, text, tables, charts, and images to their respective pipelines. Low perception confidence routes to the general expert. Mixed visual input routes to the vision expert. The selected route uploads image bytes only when MIR and routing require pixels; OCR text and structured objects are used otherwise.

### Provider configuration and fallback

Stage 1 includes separate adapters for NVIDIA's OpenAI-compatible chat completions interface, Gemini's REST `generateContent` interface, and an optional local OpenAI-compatible service. Current model identifiers must be supplied through configuration rather than being embedded in code:

| Variable | Purpose |
| --- | --- |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL` | NVIDIA hosted endpoint credentials and selected model |
| `NVIDIA_BASE_URL` | NVIDIA OpenAI-compatible API base URL |
| `NVIDIA_SUPPORTS_VISION` | Declare whether the configured NVIDIA model accepts images |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini credentials and selected model |
| `GEMINI_BASE_URL` | Gemini API base URL |
| `LOCAL_BASE_URL`, `LOCAL_MODEL` | Optional local OpenAI-compatible provider |
| `LOCAL_API_KEY` | Optional key for a protected local endpoint |
| `LOCAL_SUPPORTS_VISION` | Declare whether the local model accepts images |
| `*_DAILY_BUDGET` | Locally enforced request ceiling; `0` means no local ceiling |

Providers are filtered by expert compatibility, image capability, configuration, and local daily budget. They are attempted in registry order, with failures recorded in the response trace before trying the next compatible provider. If every compatible provider fails or none is configured, the endpoint returns a clean `503` and never fabricates an answer.

Provider calls use async HTTP. Session requests, locally tracked daily requests, failures, and average successful latency are held in memory for Stage 1 and reset when the process restarts.

### Quality checks

Tests use deterministic mock perception and provider adapters and do not consume external quota:

```bash
pytest
ruff check .
mypy
```

The suite covers modality-to-expert routing, provider fallback, local empty-query suggestions, minimum-sufficient representation behavior, vision routing, low-confidence fallback, and the public API contract.

### Stage 1 limitations

- Provider counters are process-local rather than persisted across restarts.
- Cloud quota remaining is not fetched or estimated.
- Intent and expert routing use explicit rules, keywords, and heuristics.
- Validation is deterministic and lightweight; calibrated confidence and multi-model validation are future work.
- The real perception implementation must be supplied by the perception component before production use.
