import { invoke } from "@tauri-apps/api/core";
import type {
  AnalyzePayload,
  AnalyzeResponse,
  ChatPayload,
  ConversationDetail,
  ConversationSummary,
  ProgressSnapshot,
} from "../types/api";

interface NativeCommandError {
  kind?: string;
  message?: string;
  status?: number | null;
}

export class BackendUnavailableError extends Error {
  constructor(message = "Backend unavailable. Start the Stage 1 orchestration service and retry.") {
    super(message);
    this.name = "BackendUnavailableError";
  }
}

export class BackendTimeoutError extends BackendUnavailableError {
  constructor() {
    super("The backend timed out. Check that the Stage 1 orchestration service is responsive, then retry.");
    this.name = "BackendTimeoutError";
  }
}

export class ProviderUnavailableError extends Error {
  readonly status = 503;

  constructor(message = "AI provider is not configured. OCR and routing completed successfully, but an AI provider is required to generate the final answer.") {
    super(message);
    this.name = "ProviderUnavailableError";
  }
}

export class BackendRequestError extends Error {
  constructor(message: string, readonly status: number | null) {
    super(message);
    this.name = "BackendRequestError";
  }
}

function commandError(error: unknown): NativeCommandError {
  if (typeof error === "object" && error !== null) return error as NativeCommandError;
  return { message: error instanceof Error ? error.message : String(error) };
}

export function normalizeBackendError(error: unknown): Error {
  const native = commandError(error);
  const message = native.message?.trim() || "The analysis request failed.";
  switch (native.kind) {
    case "provider_unavailable":
      return new ProviderUnavailableError(message);
    case "network":
      return new BackendUnavailableError();
    case "timeout":
      return new BackendTimeoutError();
    case "request":
    case "server":
    case "invalid_response":
    case "capture":
      return new BackendRequestError(message, native.status ?? null);
    default:
      return new BackendRequestError(message, native.status ?? null);
  }
}

export async function analyzeCapture(captureId: string, payload: AnalyzePayload): Promise<AnalyzeResponse> {
  try {
    return await invoke<AnalyzeResponse>("analyze_capture", { captureId, payload });
  } catch (error) {
    throw normalizeBackendError(error);
  }
}

export async function continueConversation(payload: ChatPayload): Promise<AnalyzeResponse> {
  try {
    return await invoke<AnalyzeResponse>("continue_conversation", { payload });
  } catch (error) {
    throw normalizeBackendError(error);
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  try {
    return await invoke<ConversationSummary[]>("list_conversations");
  } catch (error) {
    throw normalizeBackendError(error);
  }
}

export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  try {
    return await invoke<ConversationDetail>("get_conversation", { conversationId });
  } catch (error) {
    throw normalizeBackendError(error);
  }
}

export async function getAnalysisProgress(requestId: string): Promise<ProgressSnapshot> {
  try {
    return await invoke<ProgressSnapshot>("get_analysis_progress", { requestId });
  } catch (error) {
    throw normalizeBackendError(error);
  }
}
