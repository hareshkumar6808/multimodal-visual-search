# Stage 2 Execution and Pixel Policy

The post-route policy is transparent code, not another model. It chooses the cheapest valid class in this order: direct, tool, local LLM, cloud text, cloud vision.

- “Copy/read/extract this text” with OCR returns `direct` and `return_ocr`.
- A math route with a math candidate returns `tool` and `sympy_math`.
- A calculate/extract request over a structured table returns `tool` and `table_aggregate`.
- Code debugging and other complex text reasoning prefer `cloud_text`.
- Ordinary OCR-backed explanation or summary uses `local_llm`.
- A pixel-dependent request uses `cloud_vision` and `send_pixels=true`.

Pixel use is query-aware. Text summaries, OCR-backed code debugging, and structured table operations do not send pixels. Image identification and chart interpretation do. UI text with OCR confidence at least 0.75 does not; UI color, position, layout, shape, or button questions do. MIR v0.2 `representation_mode` and `pixel_dependency_prior` are used when present and safely ignored when absent.

This component decides and reports direct/tool modes. Person 3 owns their actual execution.
