import { describe, expect, it } from "vitest";
import {
  appendPendingTurn,
  canSend,
  conversationFromCapture,
  emptyConversation,
  failAssistant,
  removeDraftAttachment,
  resolveAssistant,
  updateAssistantProgress,
} from "./conversationState";
import type { AnalyzeResponse, CaptureResult } from "./types/api";

const capture: CaptureResult = {
  capture_id: "capture-a",
  image_data_url: "data:image/png;base64,abc",
  captured_at_ms: 100,
  payload: {
    request_id: "request-a",
    query: null,
    context: {
      active_app: "Code",
      window_title: "main.py",
      selection_mode: "rectangle",
      bounds: { x: 1, y: 2, width: 100, height: 80 },
    },
  },
};

function response(turn: number): AnalyzeResponse {
  return {
    request_id: `request-${turn}`,
    conversation_id: "conversation-a",
    message_id: `server-assistant-${turn}`,
    answer: `answer ${turn}`,
    suggested_actions: [],
    mir_summary: { primary_modality: "code", confidence: 0.9 },
    route: { intent: "debug", expert: "code-expert", provider: "local", reason_code: "test" },
    trace: [{ stage: `turn-${turn}`, status: "complete" }],
    metrics: { latency_ms: 10, perception_ms: turn === 1 ? 5 : 0, routing_ms: 1, provider_ms: 4, cloud_image_uploaded: false, api_calls: 1 },
  };
}

describe("conversation state", () => {
  it("creates a new conversation with the capture as a draft attachment", () => {
    const state = conversationFromCapture("conversation-a", capture, "attachment-a");
    expect(state.draftAttachment?.captureId).toBe("capture-a");
    expect(state.conversation.id).toBe("conversation-a");
  });

  it("moves the image into the sent user message and clears the composer attachment", () => {
    const state = conversationFromCapture("conversation-a", capture, "attachment-a");
    const sent = appendPendingTurn(state, "Why?", "user-1", "assistant-1", 200);
    expect(sent.conversation.messages[0].attachments?.[0].captureId).toBe("capture-a");
    expect(sent.draftAttachment).toBeNull();
  });

  it("supports image-only messages but rejects a completely empty composer", () => {
    const attached = conversationFromCapture("conversation-a", capture, "attachment-a");
    expect(canSend(attached, "", false)).toBe(true);
    expect(canSend(removeDraftAttachment(attached), "", false)).toBe(false);
    expect(canSend(emptyConversation("conversation-b"), "text without a capture", false)).toBe(false);
  });

  it("keeps the user message and appends the resolved assistant response", () => {
    let state = conversationFromCapture("conversation-a", capture, "attachment-a");
    state = appendPendingTurn(state, "Why?", "user-1", "assistant-1", 200);
    state = resolveAssistant(state, "assistant-1", response(1));
    expect(state.conversation.messages.map((message) => message.content)).toEqual(["Why?", "answer 1"]);
  });

  it("shows live backend stages only while the assistant is pending", () => {
    let state = appendPendingTurn(
      conversationFromCapture("conversation-a", capture, "attachment-a"),
      "Why?",
      "user-1",
      "assistant-1",
    );
    state = updateAssistantProgress(state, "assistant-1", [
      { stage: "routing", status: "complete", message: "Expert: code-expert" },
      { stage: "provider", status: "running", message: "Calling NVIDIA" },
    ]);
    expect(state.conversation.messages[1].progress?.[1].message).toBe("Calling NVIDIA");
    state = resolveAssistant(state, "assistant-1", response(1));
    expect(state.conversation.messages[1].progress).toBeUndefined();
  });

  it("appends a second complete turn without replacing the first turn", () => {
    let state = conversationFromCapture("conversation-a", capture, "attachment-a");
    state = resolveAssistant(appendPendingTurn(state, "Why?", "user-1", "assistant-1"), "assistant-1", response(1));
    state = resolveAssistant(appendPendingTurn(state, "How do I fix it?", "user-2", "assistant-2"), "assistant-2", response(2));
    expect(state.conversation.messages.map((message) => message.content)).toEqual([
      "Why?", "answer 1", "How do I fix it?", "answer 2",
    ]);
    expect(state.conversation.id).toBe("conversation-a");
  });

  it("preserves a separate process trace on every assistant message", () => {
    let state = conversationFromCapture("conversation-a", capture, "attachment-a");
    state = resolveAssistant(appendPendingTurn(state, "one", "u1", "a1"), "a1", response(1));
    state = resolveAssistant(appendPendingTurn(state, "two", "u2", "a2"), "a2", response(2));
    const assistants = state.conversation.messages.filter((message) => message.role === "assistant");
    expect(assistants[0].response?.trace[0].stage).toBe("turn-1");
    expect(assistants[1].response?.trace[0].stage).toBe("turn-2");
  });

  it("keeps user messages when an assistant request fails", () => {
    let state = conversationFromCapture("conversation-a", capture, "attachment-a");
    state = appendPendingTurn(state, "Why?", "user-1", "assistant-1");
    state = failAssistant(state, "assistant-1", "offline");
    expect(state.conversation.messages[0].content).toBe("Why?");
    expect(state.conversation.messages[1].status).toBe("error");
  });

  it("creates a distinct conversation for a new capture", () => {
    const first = conversationFromCapture("conversation-a", capture, "attachment-a");
    const second = conversationFromCapture("conversation-b", { ...capture, capture_id: "capture-b" }, "attachment-b");
    expect(second.conversation.id).not.toBe(first.conversation.id);
    expect(second.conversation.activeCaptureId).toBe("capture-b");
  });

  it("creates an empty new chat without deleting the existing conversation object", () => {
    const current = conversationFromCapture("conversation-a", capture, "attachment-a");
    const next = emptyConversation("conversation-b");
    expect(current.conversation.id).toBe("conversation-a");
    expect(next.conversation.messages).toEqual([]);
  });
});
