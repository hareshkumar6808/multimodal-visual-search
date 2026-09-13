import { useEffect, useRef, useState } from "react";
import { Clock3, EyeOff, Keyboard, Pause, Play, ScanLine, Settings, X } from "lucide-react";
import { desktopBridge } from "../services/desktopBridge";

export function FloatingWidget() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [paused, setPaused] = useState(false);
  const [busy, setBusy] = useState(false);
  const dragged = useRef(false);

  useEffect(() => {
    const close = () => {
      if (menuOpen) {
        setMenuOpen(false);
        void desktopBridge.setWidgetExpanded(false);
      }
    };
    window.addEventListener("blur", close);
    return () => window.removeEventListener("blur", close);
  }, [menuOpen]);

  async function startCapture() {
    if (paused || busy) return;
    setBusy(true);
    try {
      await desktopBridge.beginRectangleCapture();
    } finally {
      setBusy(false);
    }
  }

  async function openMenu(event: React.MouseEvent) {
    event.preventDefault();
    setMenuOpen(true);
    await desktopBridge.setWidgetExpanded(true);
  }

  async function menuAction(action: () => Promise<unknown>) {
    setMenuOpen(false);
    await desktopBridge.setWidgetExpanded(false);
    await action();
  }

  if (menuOpen) {
    return (
      <main className="widget-menu" role="menu" aria-label="Assistant menu">
        <div className="menu-heading">Visual Search</div>
        <button role="menuitem" onClick={() => menuAction(startCapture)}><ScanLine />New Capture</button>
        <button role="menuitem" onClick={() => menuAction(() => desktopBridge.openUtility("history"))}><Clock3 />History <small>Coming soon</small></button>
        <button role="menuitem" onClick={() => {
          const next = !paused;
          setPaused(next);
          void desktopBridge.setPaused(next);
        }}>{paused ? <Play /> : <Pause />}{paused ? "Resume Assistant" : "Pause Assistant"}</button>
        <button role="menuitem" onClick={() => menuAction(() => desktopBridge.openUtility("settings"))}><Settings />Settings</button>
        <button role="menuitem" onClick={() => menuAction(() => desktopBridge.openUtility("shortcuts"))}><Keyboard />Keyboard Shortcuts</button>
        <div className="menu-separator" />
        <button role="menuitem" onClick={() => desktopBridge.hideWidget()}><EyeOff />Hide Icon</button>
        <button className="danger" role="menuitem" onClick={() => desktopBridge.quit()}><X />Quit</button>
      </main>
    );
  }

  return (
    <button
      className={`floating-orb ${paused ? "is-paused" : ""}`}
      aria-label={paused ? "Visual Search paused" : "Start screen capture"}
      title={paused ? "Assistant paused" : "Click to capture · Right-click for menu"}
      onContextMenu={openMenu}
      onPointerDown={async () => {
        dragged.current = false;
        const timer = window.setTimeout(() => {
          dragged.current = true;
          void desktopBridge.startWidgetDrag();
        }, 160);
        const clear = () => window.clearTimeout(timer);
        window.addEventListener("pointerup", clear, { once: true });
      }}
      onClick={() => {
        if (!dragged.current) void startCapture();
      }}
    >
      <ScanLine aria-hidden="true" />
      <span className="status-dot" />
    </button>
  );
}
