# Stage 2 Component 3 recovery report

## Recovery

The former remote `stage2-experts-tools` head was `9a3529ebb00323423d870b709bb501d85257be1e`. Its only changes were an empty `apps/desktop/src/codex` file and a root lockfile fragment. Branches, worktrees, stashes, all reflogs, path history, and unreachable Git objects were inspected. No unpushed execution-engine implementation was recoverable, so this branch was rebuilt from `origin/stage1-integration` at `66e07f5023b51e1e53eecb131da2e8eea4498ab1`.

## Implemented architecture

`RouteDecisionV2` is consumed as optional input. When those fields are absent, Stage 1 broad experts map to the corresponding `*/general` policy. This component does not implement a second router.

The expert registry covers all required text, code, math, table, chart, vision, UI, and general leaves. Each policy carries its parent, sub-expert, preferred representation, allowed deterministic tools, provider capability, prompt fragment, and validation policy.

The execution engine supports provider-free `return_ocr` and reports zero API calls. Its deterministic tools are:

- SymPy arithmetic, simplify, factor, expand, equation solving, differentiation, and basic integration through an explicit AST whitelist.
- Table count, sum, mean/average, minimum, and maximum with controlled nonnumeric errors.
- Python syntax/static inspection through `ast.parse`; selected code is never executed.
- SQL statement classification, token inspection, and formatting through `sqlparse`; SQL is never executed and database-specific validity is not claimed.

No `eval`, `exec`, arbitrary code execution, autonomous-agent framework, or runtime tool download is used.

## Streaming

The existing `/api/analyze` and `/api/chat` endpoints remain intact. `/api/analyze/stream` and `/api/chat/stream` return typed SSE events: `start`, `progress`, `direct_result`, `tool_result`, `delta`, `done`, and `error`.

OpenAI-compatible providers use `POST /chat/completions` with `stream: true` and parse standard `data:` frames. This covers local llama.cpp/Qwen-compatible servers and NVIDIA's OpenAI-compatible API. Gemini uses the official `streamGenerateContent?alt=sse` method. Other providers inherit a safe single-result streaming fallback.

Provider fallback is allowed before the first visible delta. A failure after partial output produces an interruption error and preserves the partial text; another model is not silently started. TTFT, provider start, provider total, stream duration, and time to first useful output are emitted with stream events.

The Rust bridge forwards initial-capture stream events to the webview. Follow-ups consume the chat SSE endpoint directly. The current assistant message appends deltas as they arrive, preserves history/retry/image behavior, retains partial output on interruption, and uses `AbortController` for follow-up cancellation.

## Validation notes

Python tests, Ruff, strict mypy, TypeScript typechecking, ESLint, Vitest, and the Vite production build were run. The sandbox denies Node's native Windows `realpath` and temp-directory rename operations, so Vitest/Vite were run with a process-local realpath shim and workspace-local temporary directory; no shim is shipped. Rust compilation could not run because `cargo.exe` is not installed on this host. The Rust changes are therefore source-reviewed but remain the only environment-limited check.
