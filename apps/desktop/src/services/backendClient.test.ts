import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalyzePayload, AnalyzeResponse } from "../types/api";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));

import {
  analyzeCapture,
  BackendRequestError,
  BackendUnavailableError,
  getAnalysisProgress,
  normalizeBackendError,
  ProviderUnavailableError,
} from "./backendClient";

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
  suggested_actions: [],
  mir_summary: { primary_modality: "code", confidence: 0.94 },
  route: { intent: "explain", expert: "code-expert", provider: "local", reason_code: "CODE_TEXT_SUFFICIENT" },
  trace: [{ stage: "capture_received", status: "complete" }],
  metrics: { latency_ms: 900, perception_ms: 200, routing_ms: 2, provider_ms: 690, cloud_image_uploaded: false, api_calls: 1 },
};

describe("analyzeCapture", () => {
  beforeEach(() => invoke.mockReset());

  it("sends the agreed capture id and payload through the native bridge", async () => {
    invoke.mockResolvedValue(response);
    await expect(analyzeCapture("capture-123", payload)).resolves.toEqual(response);
    expect(invoke).toHaveBeenCalledWith("analyze_capture", { captureId: "capture-123", payload });
  });

  it("maps connection failures to the required backend unavailable message", () => {
    const error = normalizeBackendError({ kind: "network", message: "connection refused" });
    expect(error).toBeInstanceOf(BackendUnavailableError);
    expect(error.message).toBe("Backend unavailable. Start the Stage 1 orchestration service and retry.");
  });

  it("reads live progress through the native bridge", async () => {
    const progress = {
      request_id: "request-123",
      complete: false,
      events: [{ stage: "provider", status: "running", message: "Calling NVIDIA" }],
    };
    invoke.mockResolvedValue(progress);
    await expect(getAnalysisProgress("request-123")).resolves.toEqual(progress);
    expect(invoke).toHaveBeenCalledWith("get_analysis_progress", { requestId: "request-123" });
  });

  it("does not classify provider HTTP 503 as a disconnected backend", () => {
    const error = normalizeBackendError({
      kind: "provider_unavailable",
      message: "AI provider is not configured.",
      status: 503,
    });
    expect(error).toBeInstanceOf(ProviderUnavailableError);
    expect(error).not.toBeInstanceOf(BackendUnavailableError);
  });

  it("keeps backend HTTP failures separate from network failures", () => {
    const error = normalizeBackendError({ kind: "server", message: "Internal error", status: 500 });
    expect(error).toBeInstanceOf(BackendRequestError);
    expect(error).not.toBeInstanceOf(BackendUnavailableError);
  });
});
