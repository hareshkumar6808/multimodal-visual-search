import { useState } from "react";
import { Check, ChevronDown, CircleAlert, LoaderCircle } from "lucide-react";
import type { AnalyzeResponse, TraceEvent } from "../types/api";

function TraceIcon({ event }: { event: TraceEvent }) {
  if (event.status === "complete") return <Check className="trace-ok" />;
  if (event.status === "error") return <CircleAlert className="trace-error" />;
  return <LoaderCircle className={event.status === "running" ? "spin" : ""} />;
}

function label(stage: string) {
  return stage.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

export function TracePanel({ response }: { response: AnalyzeResponse }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="trace-panel">
      <button className="trace-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span>Show Process</span><ChevronDown className={open ? "rotated" : ""} />
      </button>
      {open && (
        <div className="trace-content">
          {response.trace.map((event, index) => (
            <div className="trace-row" key={`${event.stage}-${index}`}>
              <TraceIcon event={event} />
              <div><strong>{label(event.stage)}</strong>{event.message && <small>{event.message}</small>}</div>
            </div>
          ))}
          <dl className="trace-facts">
            <div><dt>Modality</dt><dd>{response.mir_summary.primary_modality} · {Math.round(response.mir_summary.confidence * 100)}%</dd></div>
            <div><dt>Intent</dt><dd>{response.route.intent}</dd></div>
            <div><dt>Expert</dt><dd>{response.route.expert}</dd></div>
            <div><dt>Provider</dt><dd>{response.route.provider}</dd></div>
            <div><dt>Total</dt><dd>{(response.metrics.latency_ms / 1000).toFixed(2)} sec</dd></div>
            <div><dt>Cloud image</dt><dd>{response.metrics.cloud_image_uploaded ? "Uploaded" : "Not uploaded"}</dd></div>
          </dl>
        </div>
      )}
    </section>
  );
}
