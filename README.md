# Multimodal Visual Search

Multimodal Visual Search is a Windows desktop application that captures a selected screen region, extracts local context, chooses a specialized expert pipeline, and returns an answer with factual execution telemetry.

Stage 1 implements the complete baseline vertical slice: a Tauri desktop client, local perception and OCR boundary, MIR v0.1, deterministic intent and expert routing, replaceable providers with fallback, and a React chat interface.

## Stage 1 architecture

```text
Windows floating widget
  -> screen selection and screenshot preview
  -> POST /api/analyze (image + payload_json)
  -> real services.perception adapter
  -> canonical MIR v0.1
  -> intent classifier
  -> expert and representation selection
  -> compatible provider with fallback
  -> validated response, trace, and metrics
  -> desktop chat and Show Process panel
```

Text, code, tables, and extracted charts use their local extracted representation when pixels are unnecessary. Image bytes are sent only when `visual_required` is true and the selected provider supports vision.

## Repository structure

| Path | Purpose |
| --- | --- |
| `apps/desktop` | React, TypeScript, Tauri, and Rust desktop client |
| `contracts/models.py` | Canonical request, MIR v0.1, response, trace, and provider contracts |
| `services/perception` | Image preflight, OCR boundary, profiling, extraction, and MIR generation |
| `services/orchestrator` | API, routing, experts, prompts, providers, fallback, validation, and metrics |
| `tests/perception` | Perception unit and environment-dependent OCR tests |
| `tests/test_stage1_integration.py` | Real perception-module to router integration scenarios |

## Requirements

- Windows 10 or 11
- Python 3.11-3.13 recommended
- Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` or available on `PATH`
- Node.js 20 or newer
- Rust stable with the MSVC target
- Microsoft C++ Build Tools and Windows SDK
- WebView2 Runtime

The backend starts without OCR or cloud keys and reports degraded readiness. Analysis requires a working local OCR engine. Provider-backed answers require at least one configured provider or local inference service. Empty-query suggestions require perception but make no provider request.

## Backend setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configure only the providers you intend to use. Model identifiers are explicit configuration so they can be updated without code changes.

```env
APP_ENV=development
PERCEPTION_MODE=real
PERCEPTION_MODULE=services.perception

NVIDIA_API_KEY=
NVIDIA_MODEL=

GEMINI_API_KEY=
GEMINI_MODEL=

LOCAL_BASE_URL=
LOCAL_MODEL=
```

The complete configuration template is in `.env.example`. Never commit `.env`; it is ignored by Git.

Start the backend:

```powershell
uvicorn services.orchestrator.main:app --host 127.0.0.1 --port 8765 --reload
```

## Desktop setup

In a second PowerShell window:

```powershell
cd apps\desktop
npm install
npm run tauri:dev
```

The Tauri application opens the draggable floating widget. Click it to capture a rectangle, preview the screenshot in chat, enter an optional query, and send the multipart request to the backend.

Browser-only visual development is available with `npm run dev`, but native capture commands work only inside Tauri.

## API

The backend binds to `http://127.0.0.1:8765` by default.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Orchestrator, real perception, OCR, and safe provider readiness |
| `GET /api/providers` | Provider configuration, capability, request, failure, and latency status |
| `POST /api/analyze` | Analyze a capture and return the routed response |

`POST /api/analyze` accepts `multipart/form-data`:

- `image`: PNG, JPEG, GIF, BMP, or WebP, maximum 20 MiB
- `payload_json`: serialized request metadata

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

`query` may be null or empty. That selects the local `suggest` intent and returns contextual actions with `api_calls = 0`.

## MIR v0.1

`contracts/models.py` is the single source of truth. `services.perception.models` provides compatibility imports only.

```json
{
  "mir_version": "0.1",
  "request_id": "abc-123",
  "image": {"sha256": "...", "width": 800, "height": 500},
  "context": {},
  "modalities": {
    "text": 0.92,
    "code": 0.95,
    "table": 0.02,
    "chart": 0.01,
    "image": 0.05,
    "mixed": 0.08
  },
  "primary_modality": "code",
  "ocr": {"text": "...", "confidence": 0.96, "regions": []},
  "objects": [],
  "visual_required": false,
  "privacy_flags": [],
  "overall_confidence": 0.92
}
```

Incompatible MIR versions, unknown modalities, malformed output, and mismatched request IDs are rejected.

## Experts and providers

Current expert pipelines:

- `text-expert`
- `code-expert`
- `table-expert`
- `chart-expert`
- `vision-expert`
- `general-expert`

Current provider adapters:

- NVIDIA OpenAI-compatible chat completions
- Gemini REST `generateContent`
- Optional local OpenAI-compatible endpoint
- Deterministic mock provider for tests only

Providers are filtered by configuration, capability, image support, expert compatibility, and local daily budget. Eligible failures activate the next compatible provider. If none succeeds, the API returns a controlled 503 and never fabricates an answer.

## Verification

Backend:

```powershell
python -m pytest -q
python -m ruff format --check .
python -m ruff check .
python -m mypy --strict contracts services
```

Desktop:

```powershell
cd apps\desktop
npm run lint
npm run typecheck
npm test
npm run build
cargo check --manifest-path src-tauri\Cargo.toml
```

`tests/test_stage1_integration.py` executes text, code, table, chart, image, and empty-query flows through the real perception module boundary with deterministic OCR and providers. Tests do not use cloud credentials or quota. Tests marked as real Tesseract checks skip when the system executable is absent.

## Stage 1 limitations

- Tesseract is a separately installed system dependency.
- Provider counters and local budgets are process-local and reset after restart.
- Health reflects local configuration and OCR readiness; it does not make paid provider probe calls.
- Intent, profiling, and routing use explicit Stage 1 rules and heuristics.
- Image validation is bounded structural validation, not forensic decoding.
- Capture currently targets the primary display; multi-monitor coordinate handling remains future work.
- History and shortcut persistence are not implemented.
- Streaming, authentication, persistent metrics, and advanced privacy routing are outside Stage 1.

See `STAGE1_INTEGRATION_REPORT.md` for the completed integration evidence and remaining environment-specific verification items.
