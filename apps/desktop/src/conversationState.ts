import type { AnalyzeResponse, CaptureResult, ConversationDetail } from "./types/api";

export interface Attachment {
  id: string;
  type: "image";
  previewUrl: string;
  captureId?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  attachments?: Attachment[];
  timestamp: number;
  status?: "sending" | "complete" | "error";
  response?: AnalyzeResponse;
  error?: string;
}

export interface Conversation {
  id: string;
  messages: ChatMessage[];
  activeCaptureId?: string;
  createdAt: number;
  updatedAt: number;
}

export interface ConversationState {
  conversation: Conversation;
  draftAttachment: Attachment | null;
  capture: CaptureResult | null;
}

export function emptyConversation(id: string, now = Date.now()): ConversationState {
  return {
    conversation: { id, messages: [], createdAt: now, updatedAt: now },
    draftAttachment: null,
    capture: null,
  };
}

export function conversationFromCapture(
  id: string,
  capture: CaptureResult,
  attachmentId: string,
): ConversationState {
  return {
    conversation: {
      id,
      messages: [],
      activeCaptureId: capture.capture_id,
      createdAt: capture.captured_at_ms,
      updatedAt: capture.captured_at_ms,
    },
    draftAttachment: {
      id: attachmentId,
      type: "image",
      previewUrl: capture.image_data_url,
      captureId: capture.capture_id,
    },
    capture,
  };
}

export function canSend(state: ConversationState, text: string, loading: boolean): boolean {
  const hasConversationContext = state.capture !== null || state.conversation.messages.length > 0;
  return !loading
    && hasConversationContext
    && (text.trim().length > 0 || state.draftAttachment !== null);
}

export function removeDraftAttachment(state: ConversationState): ConversationState {
  return { ...state, draftAttachment: null };
}

export function appendPendingTurn(
  state: ConversationState,
  text: string,
  userId: string,
  assistantId: string,
  now = Date.now(),
): ConversationState {
  const user: ChatMessage = {
    id: userId,
    role: "user",
    content: text.trim(),
    attachments: state.draftAttachment ? [state.draftAttachment] : undefined,
    timestamp: now,
    status: "complete",
  };
  const assistant: ChatMessage = {
    id: assistantId,
    role: "assistant",
    content: "",
    timestamp: now + 1,
    status: "sending",
  };
  return {
    ...state,
    conversation: {
      ...state.conversation,
      messages: [...state.conversation.messages, user, assistant],
      updatedAt: now + 1,
    },
    draftAttachment: null,
  };
}

export function resolveAssistant(
  state: ConversationState,
  assistantId: string,
  response: AnalyzeResponse,
): ConversationState {
  const content = response.answer
    ?? (response.suggested_actions.length
      ? `Suggested actions: ${response.suggested_actions.join(", ")}`
      : "No answer was returned.");
  return {
    ...state,
    conversation: {
      ...state.conversation,
      id: response.conversation_id ?? state.conversation.id,
      messages: state.conversation.messages.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              id: response.message_id ?? message.id,
              content,
              status: "complete" as const,
              response,
            }
          : message,
      ),
      updatedAt: Date.now(),
    },
  };
}

export function failAssistant(
  state: ConversationState,
  assistantId: string,
  error: string,
): ConversationState {
  return {
    ...state,
    conversation: {
      ...state.conversation,
      messages: state.conversation.messages.map((message) =>
        message.id === assistantId
          ? { ...message, content: "Request failed.", status: "error" as const, error }
          : message,
      ),
      updatedAt: Date.now(),
    },
  };
}

export function retryAssistant(state: ConversationState, assistantId: string): ConversationState {
  return {
    ...state,
    conversation: {
      ...state.conversation,
      messages: state.conversation.messages.map((message) =>
        message.id === assistantId
          ? { ...message, content: "", status: "sending" as const, error: undefined }
          : message,
      ),
    },
  };
}

export function restoreConversation(detail: ConversationDetail): ConversationState {
  const attachment: Attachment | undefined = detail.image_data_url
    ? {
        id: `stored-${detail.capture_id}`,
        type: "image",
        previewUrl: detail.image_data_url,
        captureId: detail.capture_id,
      }
    : undefined;
  let imageAttached = false;
  return {
    conversation: {
      id: detail.id,
      activeCaptureId: detail.capture_id,
      createdAt: detail.created_at,
      updatedAt: detail.updated_at,
      messages: detail.messages.map((message) => {
        const attachments = message.has_image && attachment && !imageAttached
          ? [attachment]
          : undefined;
        if (attachments) imageAttached = true;
        return {
          id: message.id,
          role: message.role,
          content: message.content,
          attachments,
          timestamp: message.timestamp,
          status: "complete",
          response: message.response ?? undefined,
        };
      }),
    },
    draftAttachment: null,
    capture: null,
  };
}
