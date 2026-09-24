# Multimodal Visual Search

Multimodal Visual Search is a Windows desktop app for selecting part of the screen and discussing it with a routed AI assistant. Stage 1 captures a rectangle, runs local OCR and perception, builds MIR v0.1, selects a specialized expert, and uses configured cloud providers with an offline local fallback. Images, messages, traces, and semantic context remain available in a persistent conversation.

# Run on Windows

## Requirements

- 64-bit Windows 10 or 11 with at least 5 GB of free disk space
- Python 3.11 or newer
- Node.js 20 or newer
- Rust stable, Microsoft Visual C++ Build Tools, and a Windows SDK
- Tesseract OCR available on `PATH` or installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Microsoft Edge WebView2 Runtime

The setup script installs repository dependencies and downloads the pinned llama.cpp runtime plus the 2.1 GB Qwen2.5 3B model. It does not install system-wide development tools.

## First time

```powershell
git clone https://github.com/hareshkumar6808/multimodal-visual-search.git
cd multimodal-visual-search
git switch stage1-integration
.\scripts\setup-stage1.ps1
.\scripts\run-stage1.ps1
```

If PowerShell blocks local scripts, use these equivalent commands:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-stage1.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-stage1.ps1
```

## Normal use

```powershell
.\scripts\run-stage1.ps1
```

The launcher stops stale project-owned processes, starts the local AI provider on port `11434`, starts the backend on port `8765`, waits for both health checks, and launches the native Tauri app. Runtime logs and the local conversation database are stored under `.tools\stage1`.

Stop every process started by the launcher with:

```powershell
.\scripts\stop-stage1.ps1
```

Restart the complete stack after code or desktop changes with:

```powershell
.\scripts\restart-stage1.ps1
```

Restart only the backend after Python or `.env` changes with:

```powershell
.\scripts\restart-backend.ps1
```

## Using the app

1. Click the floating widget and drag a rectangle around text, code, a table, or an image.
2. The capture appears as a removable draft attachment in the composer.
3. Enter a question, or send the image without text.
4. Watch the live process panel report capture receipt, OCR/perception, MIR creation, intent, expert routing, provider calls, retries, fallback, and validation.
5. When the answer arrives, the process panel is replaced by the normal assistant response. The image moves into the first user message and the composer clears.
6. Ask follow-up questions in the same thread. Previous messages stay visible and the backend reuses the stored MIR/OCR context.
7. Expand **Show Process** on any completed response to inspect the stored trace for that turn.
8. Use **New Chat** for an empty thread or **History** to reopen a stored conversation.

## Current Stage 1 capabilities

- Native Tauri floating widget and rectangle screen capture
- Persistent ChatGPT-style message timeline with image previews
- Image-only messages, Enter to send, and Shift+Enter for a newline
- Retryable error messages that keep the original user turn
- Live backend-derived progress while OCR, MIR, routing, and provider work is running
- Local SQLite conversation, capture, MIR, response, and trace persistence
- Real Tesseract OCR and MIR v0.1 perception
- Per-turn intent classification and text, code, table, chart, vision, or general expert routing
- Bounded conversation history in provider prompts
- Local Qwen2.5 3B inference with no cloud key required
- Separate backend and provider readiness indicators
- Per-turn latency, routing, provider, image-upload, and API-call telemetry
- Automatic cloud retry and offline local fallback

## Provider behavior

The standard launcher configures the repository-local Qwen2.5 3B Q4 model through llama.cpp. It is a small local model, so it is private and free to run but less capable than larger hosted models. Answers receive the relevant recent conversation plus the original OCR/MIR selection context and are requested to be complete.

NVIDIA and Gemini adapters are available through `.env`, but the launcher does not require or create cloud credentials. One NVIDIA key can power three model agents: `NVIDIA_MODEL` handles text and general requests, `NVIDIA_REASONING_MODEL` handles code and table reasoning, and `NVIDIA_VISION_MODEL` handles charts and images.

Lightweight identification, extraction, and short summaries prefer the local model. Explanations, debugging, calculations, comparisons, code, tables, charts, visual questions, long selections, and longer conversations prefer cloud providers. Visual routes prefer Gemini when configured and otherwise use the NVIDIA vision agent. Code and table requests prefer the NVIDIA reasoning agent.

Cloud connections fail fast when the network is unavailable. Once connected, a cloud generation may run for up to 30 seconds and is retried once before the router moves to another provider or the local fallback. The native client allows the complete fallback chain to finish while showing live progress. Provider health, capabilities, budget, recent failures, specialization, and latency affect routing. The final request fails only when every compatible configured provider, including local fallback, is unavailable or returns an error.

## Troubleshooting

- **Tesseract was not found:** install Tesseract OCR and reopen PowerShell, or install it in the default path above.
- **Rust/Cargo was not found:** install Rust with `rustup` and select the stable MSVC toolchain.
- **`link.exe` or Windows libraries are missing:** install the Visual Studio Build Tools workload **Desktop development with C++** and a Windows SDK.
- **WebView2 error:** install or repair the Microsoft Edge WebView2 Runtime.
- **Port 8765, 11434, or 1420 is busy:** run `.\scripts\stop-stage1.ps1`, then run the launcher again.
- **First launch takes several minutes:** the initial Tauri build compiles Rust dependencies; later launches reuse the build cache.
- **Local responses are slow:** inference speed depends on CPU/GPU and memory bandwidth. Follow-up requests still avoid OCR and image parsing.
- **Cloud provider is delayed:** keep the request open. The live panel shows the active provider and automatic retry; local fallback starts if cloud attempts fail.
- **Cloud always falls back locally:** confirm the model IDs and key in `.env`, restart the backend, and inspect `backend.err.log` for the exact HTTP or timeout category.
- **Startup fails:** inspect `.tools\stage1\logs\backend.err.log`, `local-provider.err.log`, and `desktop.err.log`.

## API

The backend binds to `http://127.0.0.1:8765`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Backend, perception, OCR, and provider readiness |
| `GET /api/providers` | Provider capability and request telemetry |
| `POST /api/analyze` | First turn: multipart image plus `payload_json` |
| `POST /api/chat` | Follow-up turn: conversation ID and query, without another image |
| `GET /api/progress/{request_id}` | Live OCR, MIR, routing, provider, retry, and validation progress |
| `GET /api/conversations` | List locally persisted conversations |
| `GET /api/conversations/{conversation_id}` | Load messages, image, responses, and traces |

The first request accepts a PNG, JPEG, GIF, BMP, or WebP image up to 20 MiB and metadata such as:

```json
{
  "request_id": "abc-123",
  "conversation_id": "f17f5cde-68f6-4db2-9cf2-d116315ba151",
  "query": "Why is this code failing?",
  "context": {
    "active_app": "Visual Studio Code",
    "window_title": "main.py",
    "selection_mode": "rectangle",
    "bounds": {"x": 100, "y": 100, "width": 800, "height": 500}
  }
}
```

A follow-up sends JSON only:

```json
{
  "request_id": "abc-124",
  "conversation_id": "f17f5cde-68f6-4db2-9cf2-d116315ba151",
  "query": "How do I fix it?",
  "context": {}
}
```

The backend reloads the stored MIR, capture reference, and recent messages, reroutes the current intent, and returns a new answer, route, trace, metrics, conversation ID, and message ID. It sends stored pixels later only when the selected route genuinely requires visual input.

## Project structure

| Path | Purpose |
| --- | --- |
| `apps/desktop` | React, TypeScript, Tauri, and Rust desktop client |
| `contracts/models.py` | Request, MIR v0.1, conversation, response, trace, and provider contracts |
| `services/perception` | Image validation, OCR, profiling, extraction, and MIR generation |
| `services/orchestrator` | API, live progress, persistent conversations, routing, experts, prompts, and providers |
| `scripts/setup-stage1.ps1` | One-time repository setup and local model download |
| `scripts/run-stage1.ps1` | Complete local stack launcher |
| `scripts/stop-stage1.ps1` | Project process shutdown |
| `tests` | Perception, routing, provider, API, persistence, and integration tests |

## Development checks

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy --strict contracts services
```

From `apps\desktop`:

```powershell
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
cargo check --manifest-path src-tauri\Cargo.toml
```

## Stage 1 limits

- Capture currently targets the primary display; full multi-monitor coordinate handling remains future work.
- Conversations use bounded recent history instead of long-term memory or RAG.
- Final answer bodies are non-streaming; real pipeline and provider progress is shown while they run.
- Provider counters and local budgets reset with the backend process.
- Intent, profiling, and routing use explicit Stage 1 rules and heuristics.
- Authentication, persistent metrics, and advanced privacy routing remain outside Stage 1.
