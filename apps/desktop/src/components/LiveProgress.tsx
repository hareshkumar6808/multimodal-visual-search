import { Check, CircleAlert, LoaderCircle } from "lucide-react";
import type { LiveProgressEvent } from "../types/api";

function icon(status: LiveProgressEvent["status"]) {
  if (status === "complete") return <Check className="live-progress-ok" />;
  if (status === "failed") return <CircleAlert className="live-progress-error" />;
  return <LoaderCircle className="spin" />;
}

export function LiveProgress({ events }: { events: LiveProgressEvent[] }) {
  const visible = events.length
    ? events
    : [{ stage: "request", status: "running" as const, message: "Starting analysis" }];
  return (
    <div className="live-progress" aria-live="polite">
      <strong>Working through the request</strong>
      {visible.map((event, index) => (
        <div className={`live-progress-row ${event.status}`} key={`${event.stage}-${index}`}>
          {icon(event.status)}
          <span>{event.message}</span>
        </div>
      ))}
    </div>
  );
}
