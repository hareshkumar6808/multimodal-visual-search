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
