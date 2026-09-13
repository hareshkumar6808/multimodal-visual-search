import { useEffect, useMemo, useRef, useState } from "react";
import { Expand, LoaderCircle, Monitor, X } from "lucide-react";
import { desktopBridge } from "../services/desktopBridge";
import type { CaptureBounds, CaptureFrame } from "../types/api";

interface Point { x: number; y: number }

function boundsBetween(start: Point, end: Point): CaptureBounds {
  return {
    x: Math.round(Math.min(start.x, end.x)),
    y: Math.round(Math.min(start.y, end.y)),
    width: Math.round(Math.abs(end.x - start.x)),
    height: Math.round(Math.abs(end.y - start.y)),
  };
}

export function CaptureOverlay({ captureId }: { captureId: string | null }) {
  const [frame, setFrame] = useState<CaptureFrame | null>(null);
  const [start, setStart] = useState<Point | null>(null);
  const [end, setEnd] = useState<Point | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [finishing, setFinishing] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const selection = useMemo(() => (start && end ? boundsBetween(start, end) : null), [start, end]);

  useEffect(() => {
    if (!captureId) {
      setFrame(null);
      return;
    }
    setError(null);
    setFinishing(false);
    setStart(null);
    setEnd(null);
    const loadFrame = () => {
      void desktopBridge.getCapture(captureId)
        .then((result) => setFrame({ capture_id: result.capture_id, image_data_url: result.image_data_url, screen: result.payload.context.bounds }))
        .catch((reason) => setError(String(reason)));
    };
    loadFrame();
    window.addEventListener("focus", loadFrame);
    return () => window.removeEventListener("focus", loadFrame);
  }, [captureId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") void desktopBridge.cancelCapture();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function finish(selectedBounds: CaptureBounds) {
    if (!captureId || selectedBounds.width < 4 || selectedBounds.height < 4) {
      setStart(null);
      setEnd(null);
      return;
    }
    const scaleX = frame ? frame.screen.width / window.innerWidth : 1;
    const scaleY = frame ? frame.screen.height / window.innerHeight : 1;
    setFinishing(true);
    try {
      await desktopBridge.finalizeRectangle(captureId, {
        x: Math.round(selectedBounds.x * scaleX),
        y: Math.round(selectedBounds.y * scaleY),
        width: Math.round(selectedBounds.width * scaleX),
        height: Math.round(selectedBounds.height * scaleY),
      });
    } catch (reason) {
      setFinishing(false);
      setError(`Could not capture that area: ${String(reason)}`);
    }
  }

  async function finishFullscreen() {
    if (!captureId || finishing) return;
    setFinishing(true);
    try {
      await desktopBridge.captureFullscreen(captureId);
    } catch (reason) {
      setFinishing(false);
      setError(`Could not capture the screen: ${String(reason)}`);
    }
  }

  if (error) return <div className="capture-error">{error}</div>;

  return (
    <div
      ref={root}
      className={`capture-overlay${finishing ? " is-finishing" : ""}`}
      onPointerDown={(event) => {
        if (finishing) return;
        if ((event.target as HTMLElement).closest("button")) return;
        const point = { x: event.clientX, y: event.clientY };
        setStart(point);
        setEnd(point);
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (start) setEnd({ x: event.clientX, y: event.clientY });
      }}
      onPointerUp={(event) => {
        if (!start) return;
        const finalPoint = { x: event.clientX, y: event.clientY };
        const finalBounds = boundsBetween(start, finalPoint);
        setEnd(finalPoint);
        void finish(finalBounds);
      }}
    >
      {frame && <img className="capture-background" src={frame.image_data_url} alt="Captured desktop" draggable={false} />}
      <div className="screen-dim" />
      {selection && (
        <div className="selection-box" style={{
          left: selection.x,
          top: selection.y,
          width: selection.width,
          height: selection.height,
          backgroundImage: frame ? `url(${frame.image_data_url})` : undefined,
          backgroundSize: `${window.innerWidth}px ${window.innerHeight}px`,
          backgroundPosition: `${-selection.x}px ${-selection.y}px`,
        }}>
          <span>{selection.width} × {selection.height}</span>
        </div>
      )}
      <div className="capture-toolbar">
        <span><Expand /> Rectangle</span>
        <button disabled={!captureId || finishing} onClick={() => void finishFullscreen()} title="Capture full screen"><Monitor /> Fullscreen</button>
        <button onClick={() => desktopBridge.cancelCapture()} title="Cancel"><X /></button>
      </div>
      {!frame && <div className="capture-loading">Preparing screen…</div>}
      {finishing && <div className="capture-finishing"><LoaderCircle className="spin" /><strong>Attaching screenshot…</strong><span>Opening your agent chat</span></div>}
    </div>
  );
}
