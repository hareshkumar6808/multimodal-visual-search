import { Channel, invoke } from "@tauri-apps/api/core";
import type {
  AnalyzePayload,
  AnalyzeResponse,
  ChatPayload,
  ConversationDetail,
  ConversationSummary,
  ProgressSnapshot,
  StreamEvent,
} from "../types/api";

interface NativeCommandError {
  kind?: string;
  message?: string;
  status?: number | null;
}

export async function streamConversation(
  payload: ChatPayload,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("http://127.0.0.1:8771/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new BackendRequestError(`Streaming request failed with HTTP ${response.status}.`, response.status);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
      if (data) onEvent(JSON.parse(data) as StreamEvent);
    }
    if (done) break;
  }
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

export async function analyzeCapture(
  captureId: string,
  payload: AnalyzePayload,
  onEvent?: (event: StreamEvent) => void,
): Promise<AnalyzeResponse> {
  try {
    const eventChannel = new Channel<StreamEvent>();
    eventChannel.onmessage = (event) => onEvent?.(event);
    return await invoke<AnalyzeResponse>("analyze_capture", {
      captureId,
      payload,
      onEvent: eventChannel,
    });
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
