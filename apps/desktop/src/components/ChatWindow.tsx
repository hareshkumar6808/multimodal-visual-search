import { FormEvent, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  History,
  LoaderCircle,
  MessageSquarePlus,
  RotateCcw,
  Sparkles,
  X,
} from "lucide-react";
import { APP_CONFIG } from "../config/app";
import {
  appendPendingTurn,
  canSend,
  conversationFromCapture,
  emptyConversation,
  failAssistant,
  removeDraftAttachment,
  resolveAssistant,
  restoreConversation,
  retryAssistant,
  type ConversationState,
} from "../conversationState";
import {
  analyzeCapture,
  BackendUnavailableError,
  continueConversation,
  getConversation,
  listConversations,
  ProviderUnavailableError,
} from "../services/backendClient";
import { desktopBridge } from "../services/desktopBridge";
import type { CaptureContext, ConversationSummary } from "../types/api";
import { TracePanel } from "./TracePanel";

const fallbackContext: CaptureContext = {
  active_app: null,
  window_title: null,
  selection_mode: "rectangle",
  bounds: { x: 0, y: 0, width: 1, height: 1 },
};

function newId() {
  return crypto.randomUUID();
}

export function ChatWindow({ captureId }: { captureId: string | null }) {
  const [state, setState] = useState<ConversationState>(() => emptyConversation(newId()));
  const [query, setQuery] = useState("");
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const [aiProvider, setAiProvider] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<ConversationSummary[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const timeline = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    if (!captureId || captureId === state.conversation.activeCaptureId) return;
    void desktopBridge.getCapture(captureId)
      .then((capture) => {
        setState(conversationFromCapture(newId(), capture, newId()));
        setQuery("");
        setProcessingId(null);
        setHistoryOpen(false);
        stickToBottom.current = true;
        void desktopBridge.markCaptureDelivered(resultCaptureId(capture));
      })
      .catch(() => undefined);
  }, [captureId, state.conversation.activeCaptureId]);

  useEffect(() => input.current?.focus(), [state.draftAttachment]);

  useEffect(() => {
    if (!stickToBottom.current) return;
    timeline.current?.scrollTo({ top: timeline.current.scrollHeight, behavior: "smooth" });
  }, [state.conversation.messages]);

  useEffect(() => {
    let active = true;
    let checking = false;
    const checkHealth = async () => {
      if (checking) return;
      checking = true;
      try {
        const health = await desktopBridge.checkBackendHealth();
        if (active) {
          setBackendStatus(health.connected ? "connected" : "disconnected");
          setAiProvider(health.provider);
        }
      } catch {
        if (active) setBackendStatus("disconnected");
      } finally {
        checking = false;
      }
    };
    void checkHealth();
    const interval = window.setInterval(() => void checkHealth(), 5_000);
    window.addEventListener("focus", checkHealth);
    return () => {
      active = false;
      window.clearInterval(interval);
      window.removeEventListener("focus", checkHealth);
    };
  }, []);

  async function refreshHistory() {
    try {
      setHistory(await listConversations());
    } catch {
      setHistory([]);
    }
  }

  async function toggleHistory() {
    const opening = !historyOpen;
    setHistoryOpen(opening);
    if (opening) await refreshHistory();
  }

  async function openConversation(conversationId: string) {
    const detail = await getConversation(conversationId);
    setState(restoreConversation(detail));
    setQuery("");
    setHistoryOpen(false);
    setProcessingId(null);
    stickToBottom.current = true;
  }

  async function executeTurn(
    snapshot: ConversationState,
    text: string,
    assistantId: string,
    firstTurn: boolean,
  ) {
    setProcessingId(assistantId);
    try {
      const context = snapshot.capture?.payload.context ?? fallbackContext;
      const response = firstTurn && snapshot.capture
        ? await analyzeCapture(snapshot.capture.capture_id, {
            ...snapshot.capture.payload,
            conversation_id: snapshot.conversation.id,
            query: text || null,
          })
        : await continueConversation({
            request_id: newId(),
            conversation_id: snapshot.conversation.id,
            query: text,
            context,
          });
      setState((current) => resolveAssistant(current, assistantId, response));
      setBackendStatus("connected");
      setAiProvider(response.route.provider ?? aiProvider);
      void refreshHistory();
    } catch (reason) {
      const requestError = reason instanceof Error ? reason : new Error(String(reason));
      setState((current) => failAssistant(current, assistantId, requestError.message));
      setBackendStatus(requestError instanceof BackendUnavailableError ? "disconnected" : "connected");
      if (requestError instanceof ProviderUnavailableError) setAiProvider(null);
    } finally {
      setProcessingId(null);
    }
  }

  async function submit(event?: FormEvent, action?: string) {
    event?.preventDefault();
    const text = (action ?? query).trim();
    if (!canSend(state, text, processingId !== null)) return;
    const firstTurn = state.conversation.messages.length === 0 && state.draftAttachment !== null;
    if (!firstTurn && !text) return;
    const assistantId = newId();
    const snapshot = state;
    setState(appendPendingTurn(state, text, newId(), assistantId));
    setQuery("");
    stickToBottom.current = true;
    await executeTurn(snapshot, text, assistantId, firstTurn);
  }

  async function retry(messageId: string) {
    const index = state.conversation.messages.findIndex((message) => message.id === messageId);
    const user = index > 0 ? state.conversation.messages[index - 1] : undefined;
    if (!user || user.role !== "user" || processingId) return;
    const firstTurn = Boolean(user.attachments?.length);
    setState(retryAssistant(state, messageId));
    await executeTurn(state, user.content, messageId, firstTurn);
  }

  function startNewChat() {
    setState(emptyConversation(newId()));
    setQuery("");
    setProcessingId(null);
    setHistoryOpen(false);
  }

  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div className="app-mark"><Sparkles /></div>
        <div><strong>Visual Search</strong><small>Multimodal conversation</small></div>
        <div className="service-statuses">
          <span className={`backend-status ${backendStatus}`}>Backend: {backendStatus}</span>
          <span className={`ai-status ${aiProvider ? "ready" : "unavailable"}`}>
            AI: {aiProvider ? `${aiProvider} ready` : "not configured"}
          </span>
        </div>
        <button className="secondary icon-button" title="History" onClick={() => void toggleHistory()}><History /></button>
        <button className="secondary icon-button" title="New chat" onClick={startNewChat}><MessageSquarePlus /></button>
        <button className="secondary icon-button" title="New capture" onClick={() => desktopBridge.beginRectangleCapture()}><RotateCcw /></button>
      </header>

      {historyOpen && (
        <aside className="history-drawer" aria-label="Conversation history">
          <strong>Previous conversations</strong>
          {history.length === 0 && <small>No saved conversations yet.</small>}
          {history.map((item) => (
            <button key={item.id} onClick={() => void openConversation(item.id)}>
              <span>{item.title}</span><small>{item.primary_modality}</small>
            </button>
          ))}
        </aside>
      )}

      <section
        className="chat-body conversation-timeline"
        ref={timeline}
        onScroll={(event) => {
          const element = event.currentTarget;
          stickToBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
        }}
      >
        {state.conversation.messages.length === 0 && (
          <section className="empty-chat">
            <Sparkles />
            <strong>{state.draftAttachment ? "Screenshot ready" : "Start a new capture"}</strong>
            <p>{state.draftAttachment ? "Ask anything about the selected content, or send the image by itself." : "Use the floating capture button to begin a conversation."}</p>
            {state.draftAttachment && (
              <div className="suggestions" aria-label="Suggested actions">
                <div>{APP_CONFIG.suggestedActions.map((action) => <button key={action} onClick={() => void submit(undefined, action)}>{action}</button>)}</div>
              </div>
            )}
          </section>
        )}

        {state.conversation.messages.map((message) => (
          <article className={`chat-message ${message.role} ${message.status ?? "complete"}`} key={message.id}>
            <span className="message-role">{message.role}</span>
            <div className="message-bubble">
              {message.attachments?.map((attachment) => (
                <button className="sent-image" key={attachment.id} onClick={() => setPreview(attachment.previewUrl)}>
                  <img src={attachment.previewUrl} alt="Captured screenshot" />
                </button>
              ))}
              {message.status === "sending" ? (
                <div className="thinking"><LoaderCircle className="spin" /> Thinking…</div>
              ) : (
                message.content && <p>{message.content}</p>
              )}
              {message.status === "error" && (
                <div className="message-error"><small>{message.error}</small><button onClick={() => void retry(message.id)}>Retry</button></div>
              )}
              {message.response && <TracePanel response={message.response} />}
            </div>
          </article>
        ))}
      </section>

      <form className="composer" onSubmit={(event) => void submit(event)}>
        {state.draftAttachment && (
          <div className="composer-attachment" aria-live="polite">
            <img src={state.draftAttachment.previewUrl} alt="Screenshot attached to draft" />
            <div><strong>Screenshot attached</strong><small>Moves into your message when sent</small></div>
            <button type="button" className="remove-attachment" aria-label="Remove screenshot" onClick={() => setState(removeDraftAttachment(state))}><X /></button>
          </div>
        )}
        <textarea
          ref={input}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submit();
            }
          }}
          placeholder={state.draftAttachment ? "Ask anything…" : "Message…"}
          rows={2}
          aria-label="Chat message"
        />
        <button className="send-button" type="submit" disabled={!canSend(state, query, processingId !== null)} aria-label="Send message"><ArrowUp /></button>
      </form>

      {preview && (
        <button className="image-preview" onClick={() => setPreview(null)} aria-label="Close image preview">
          <img src={preview} alt="Captured screenshot preview" />
        </button>
      )}
    </main>
  );
}

function resultCaptureId(capture: { capture_id: string }) {
  return capture.capture_id;
}
