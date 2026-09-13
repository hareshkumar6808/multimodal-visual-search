import { useEffect, useState } from "react";
import { CaptureOverlay } from "./components/CaptureOverlay";
import { ChatWindow } from "./components/ChatWindow";
import { FloatingWidget } from "./components/FloatingWidget";
import { UtilityPage } from "./components/UtilityPage";
import { desktopBridge } from "./services/desktopBridge";

type View = "widget" | "capture" | "chat" | "history" | "settings" | "shortcuts";

function currentView(): { view: View; captureId: string | null } {
  const parameters = new URLSearchParams(window.location.search);
  const requested = parameters.get("view") as View | null;
  const supported: View[] = ["widget", "capture", "chat", "history", "settings", "shortcuts"];
  return {
    view: requested && supported.includes(requested) ? requested : "widget",
    captureId: parameters.get("capture_id"),
  };
}

export function App() {
  const current = currentView();
  const { view } = current;
  const [captureId, setCaptureId] = useState(current.captureId);

  useEffect(() => {
    if (view !== "capture" && view !== "chat") return;
    const refreshCapture = () => {
      void desktopBridge.getActiveCaptureId().then(setCaptureId).catch(() => setCaptureId(null));
    };
    refreshCapture();
    window.addEventListener("focus", refreshCapture);
    document.addEventListener("visibilitychange", refreshCapture);
    return () => {
      window.removeEventListener("focus", refreshCapture);
      document.removeEventListener("visibilitychange", refreshCapture);
    };
  }, [view]);

  if (view === "capture") return <CaptureOverlay captureId={captureId} />;
  if (view === "chat") return <ChatWindow captureId={captureId} />;
  if (view === "widget") return <FloatingWidget />;
  return <UtilityPage view={view} />;
}
