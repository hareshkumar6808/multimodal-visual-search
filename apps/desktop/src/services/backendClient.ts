import { invoke } from "@tauri-apps/api/core";
import type { AnalyzePayload, AnalyzeResponse } from "../types/api";

export class BackendUnavailableError extends Error {
  constructor() {
    super("Backend unavailable. Start the Stage 1 orchestration service and retry.");
    this.name = "BackendUnavailableError";
  }
}

export function normalizeBackendError(error: unknown): Error {
  const message = error instanceof Error ? error.message : String(error);
  if (/connect|connection|refused|unavailable|timed out/i.test(message)) {
    return new BackendUnavailableError();
  }
  return new Error(message || "The analysis request failed.");
}

export async function analyzeCapture(captureId: string, payload: AnalyzePayload): Promise<AnalyzeResponse> {
  try {
    return await invoke<AnalyzeResponse>("analyze_capture", { captureId, payload });
  } catch (error) {
    throw normalizeBackendError(error);
  }
}
