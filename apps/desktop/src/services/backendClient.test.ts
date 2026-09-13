import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalyzePayload, AnalyzeResponse } from "../types/api";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));

import { analyzeCapture, BackendUnavailableError, normalizeBackendError } from "./backendClient";

const payload: AnalyzePayload = {
  request_id: "request-123",
  query: "Explain this",
  context: {
    active_app: "Code",
    window_title: "main.ts",
    selection_mode: "rectangle",
    bounds: { x: 20, y: 40, width: 600, height: 300 },
  },
};

const response: AnalyzeResponse = {
  request_id: "request-123",
  answer: "Example answer",
  mir_summary: { primary_modality: "code", confidence: 0.94 },
  route: { intent: "explain", expert: "code-expert", provider: "local", reason_code: "CODE_TEXT_SUFFICIENT" },
  trace: [{ stage: "capture_received", status: "complete" }],
  metrics: { latency_ms: 900, cloud_image_uploaded: false },
};

describe("analyzeCapture", () => {
  beforeEach(() => invoke.mockReset());

  it("sends the agreed capture id and payload through the native bridge", async () => {
    invoke.mockResolvedValue(response);
    await expect(analyzeCapture("capture-123", payload)).resolves.toEqual(response);
    expect(invoke).toHaveBeenCalledWith("analyze_capture", { captureId: "capture-123", payload });
  });

  it("maps connection failures to the required backend unavailable message", () => {
    const error = normalizeBackendError(new Error("connection refused"));
    expect(error).toBeInstanceOf(BackendUnavailableError);
    expect(error.message).toBe("Backend unavailable. Start the Stage 1 orchestration service and retry.");
  });
});
