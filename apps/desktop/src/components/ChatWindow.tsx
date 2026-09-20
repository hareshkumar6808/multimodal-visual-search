import { FormEvent, useEffect, useRef, useState } from "react";
import { ArrowUp, Image as ImageIcon, LoaderCircle, RotateCcw, Sparkles } from "lucide-react";
import { APP_CONFIG } from "../config/app";
import {
  analyzeCapture,
  BackendUnavailableError,
  ProviderUnavailableError,
} from "../services/backendClient";
import { desktopBridge } from "../services/desktopBridge";
import type { AnalyzeResponse, CaptureResult } from "../types/api";
import { TracePanel } from "./TracePanel";

export function ChatWindow({ captureId }: { captureId: string | null }) {
  const [capture, setCapture] = useState<CaptureResult | null>(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const [aiProvider, setAiProvider] = useState<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!captureId) {
      setCapture(null);
      return;
    }
    const loadCapture = () => {
      void desktopBridge.getCapture(captureId)
        .then((result) => {
          setCapture(result);
          setQuery("");
          setResponse(null);
          setError(null);
          void desktopBridge.markCaptureDelivered(result.capture_id);
        })
        .catch((reason) => setError(new Error(String(reason))));
    };
    loadCapture();
    window.addEventListener("focus", loadCapture);
    return () => window.removeEventListener("focus", loadCapture);
  }, [captureId]);

  useEffect(() => input.current?.focus(), [capture]);

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

  async function submit(event?: FormEvent, action?: string) {
    event?.preventDefault();
    if (!capture || loading) return;
    const effectiveQuery = (action ?? query).trim() || null;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await analyzeCapture(capture.capture_id, {
        ...capture.payload,
        query: effectiveQuery,
      });
      setResponse(result);
      setBackendStatus("connected");
      setAiProvider(result.route.provider);
    } catch (reason) {
      const requestError = reason instanceof Error ? reason : new Error(String(reason));
      setError(requestError);
      setBackendStatus(requestError instanceof BackendUnavailableError ? "disconnected" : "connected");
      if (requestError instanceof ProviderUnavailableError) setAiProvider(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div className="app-mark"><Sparkles /></div>
        <div><strong>Visual Search</strong><small>Desktop capture</small></div>
        <div className="service-statuses">
          <span className={`backend-status ${backendStatus}`}>Backend: {backendStatus}</span>
          <span className={`ai-status ${aiProvider ? "ready" : "unavailable"}`}>
            AI: {aiProvider ? `${aiProvider} ready` : "not configured"}
          </span>
        </div>
        <button className="secondary icon-button" title="New capture" onClick={() => desktopBridge.beginRectangleCapture()}><RotateCcw /></button>
      </header>

      <section className="chat-body">
        {!response && !loading && (
          <section className="suggestions" aria-label="Suggested actions">
            <span>Suggested actions</span>
            <div>{APP_CONFIG.suggestedActions.map((action) => <button key={action} onClick={() => void submit(undefined, action)}>{action}</button>)}</div>
          </section>
        )}

        {loading && <div className="analysis-state"><LoaderCircle className="spin" /><div><strong>Understanding screenshot</strong><small>The backend is selecting the best expert automatically…</small></div></div>}

        {error && (
          <div className={`error-banner${error instanceof ProviderUnavailableError ? " provider-unavailable" : ""}`} role="alert">
            <strong>{error instanceof ProviderUnavailableError ? "AI provider unavailable" : "Couldn’t analyze this capture"}</strong>
            <p>{error.message}</p>
            {error instanceof ProviderUnavailableError && (
              <details className="provider-trace">
                <summary>Show Process</summary>
                <ol>
                  <li>Capture received <span>✓</span></li>
                  <li>Perception and OCR completed <span>✓</span></li>
                  <li>MIR, intent, and routing completed <span>✓</span></li>
                  <li>Provider unavailable</li>
                </ol>
              </details>
            )}
            <button onClick={() => void submit()}>Retry</button>
          </div>
        )}

        {response && (
          <section className="answer-card">
            <div className="route-summary"><span>Automatically routed</span><strong>{response.route.expert}</strong><small>{response.mir_summary.primary_modality} · {Math.round(response.mir_summary.confidence * 100)}% confidence</small></div>
            <span className="answer-label">Answer</span>
            {response.answer && <p>{response.answer}</p>}
            {response.suggested_actions.length > 0 && (
              <div className="suggestions" aria-label="Contextual suggested actions">
                <span>Suggested actions</span>
                <div>{response.suggested_actions.map((action) => <button key={action} onClick={() => void submit(undefined, action)}>{action}</button>)}</div>
              </div>
            )}
            <TracePanel response={response} />
          </section>
        )}
      </section>

      <form className="composer" onSubmit={(event) => void submit(event)}>
        <div className="composer-attachment" aria-live="polite">
          {capture ? (
            <>
              <img src={capture.image_data_url} alt="Screenshot attached to this request" />
              <div><strong>Screenshot attached</strong><small>{desktopBridge.selectionLabel(capture.payload.context.selection_mode)} · Expert selected automatically after Send</small></div>
              <ImageIcon aria-hidden="true" />
            </>
          ) : (
            <div className="attachment-loading"><LoaderCircle className="spin" /><span>Attaching screenshot…</span></div>
          )}
        </div>
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
          placeholder="Ask about this capture (optional)"
          rows={2}
          aria-label="Question about the capture"
        />
        <button className="send-button" type="submit" disabled={!capture || loading} aria-label="Send request"><ArrowUp /></button>
      </form>
    </main>
  );
}
