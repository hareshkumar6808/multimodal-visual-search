# Desktop Interface — Stage 1

This directory contains the Windows desktop interaction layer for Multimodal Visual Search. It owns screen interaction and presentation only; OCR, multimodal perception, MIR generation, routing, and model integrations belong to other components.

## Implemented scope

- Frameless, transparent, always-on-top floating widget
- Click-to-capture and drag-to-move behavior
- Right-click menu with New Capture, History, Pause, Settings, Keyboard Shortcuts, Hide, and Quit
- Rectangle and primary-display fullscreen capture
- Active application and foreground window title collection on Windows when access is available
- Independent chat window with attached screenshot, optional question, suggested actions, answer, and factual execution trace
- Automatic expert routing display: the desktop sends the original screenshot and optional query unchanged, then presents the expert selected by the orchestration backend
- Typed native bridge and typed backend contract
- Graceful backend-unavailable state with retry
- Runtime-configurable global capture shortcut
- Clearly labelled Stage 1 placeholder for History

No mock AI response is enabled. A missing backend always produces the documented unavailable state.

## Prerequisites (Windows)

1. Node.js 20 or newer
2. Rust stable with the MSVC target
3. Microsoft C++ Build Tools and Windows 10/11 SDK
4. WebView2 Runtime (already included on current Windows 11 installations)

Follow the official Tauri v2 Windows prerequisites if Rust or the Microsoft build tools are not installed.

## Install and run

From this directory:

```powershell
npm install
npm run tauri:dev
```

Run only the frontend in a browser for visual development:

```powershell
npm run dev
```

Native capture commands are available only inside Tauri, so browser-only mode cannot take real screenshots.

## Quality checks

```powershell
npm run lint
npm run typecheck
npm test
npm run build
npm run tauri:build
```

`tauri:build` compiles the native application but does not create an installer yet (`bundle.active` is disabled for Stage 1).

## Global shortcut

The default shortcut is:

```text
Ctrl+Shift+Period
```

It launches rectangle capture and deliberately avoids the Windows `Win+Shift+S` shortcut. The native constant in `src-tauri/src/lib.rs` is the default and is exposed to the UI through `get_capture_shortcut`. Open **Settings** from the widget menu to register a different accelerator for the current session.

## Backend contract

The native client posts to:

```text
http://127.0.0.1:8765/api/analyze
```

The request uses `multipart/form-data` with:

- `image`: the selected PNG as `capture.png`
- `payload_json`: serialized metadata and optional query

Example metadata:

```json
{
  "request_id": "generated-uuid",
  "query": "Why is this code failing?",
  "context": {
    "active_app": "Code",
    "window_title": "main.py",
    "selection_mode": "rectangle",
    "bounds": { "x": 200, "y": 120, "width": 800, "height": 500 }
  }
}
```

The response interfaces are defined in `src/types/api.ts` and match the project contract exactly.

Expert selection is intentionally not performed by the desktop application. The backend analyzes the screenshot, creates the MIR, selects the expert/provider, and returns that decision in `route`; the UI only displays this factual routing result.

## Known Stage 1 limitations

- Capture currently targets the Windows primary display. Multi-monitor coordinate handling is a later milestone.
- Rectangle and fullscreen modes are implemented; free-form, window, and smart-object modes are extension points only.
- History is an honest placeholder; no persistence is implemented yet.
- Shortcut changes are not persisted across application restarts yet.
- The chat uses a request/response call. Streaming and WebSocket traces are not part of Stage 1.
- Capture data is held in memory for the running process and is not persisted to disk.
- Some elevated or protected Windows applications may not expose foreground-process metadata; capture still succeeds and returns `null` values.
