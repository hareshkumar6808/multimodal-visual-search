import { Clock3, Keyboard, Settings } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { APP_CONFIG } from "../config/app";
import { desktopBridge } from "../services/desktopBridge";

type UtilityView = "history" | "settings" | "shortcuts";

const details = {
  history: { icon: Clock3, title: "History", description: "Capture history will be added after Stage 1 storage is integrated." },
  settings: { icon: Settings, title: "Settings", description: "Configure the global shortcut used to start screen capture." },
  shortcuts: { icon: Keyboard, title: "Keyboard Shortcuts", description: "Use the global shortcut from any application to begin a rectangle capture." },
};

export function UtilityPage({ view }: { view: UtilityView }) {
  const item = details[view];
  const Icon = item.icon;
  const [shortcut, setShortcut] = useState<string>(APP_CONFIG.defaultShortcut);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (view === "shortcuts" || view === "settings") void desktopBridge.getShortcut().then(setShortcut).catch(() => undefined);
  }, [view]);

  async function saveShortcut(event: FormEvent) {
    event.preventDefault();
    setSaved(false);
    setError(null);
    try {
      setShortcut(await desktopBridge.setShortcut(shortcut.trim()));
      setSaved(true);
    } catch (reason) {
      setError(`Could not register shortcut: ${String(reason)}`);
    }
  }

  return (
    <main className="utility-page">
      <div className="utility-icon"><Icon /></div>
      <h1>{item.title}</h1>
      <p>{item.description}</p>
      {view === "shortcuts" && <kbd>{shortcut}</kbd>}
      {view === "settings" && (
        <form className="shortcut-form" onSubmit={(event) => void saveShortcut(event)}>
          <label htmlFor="shortcut">Global capture shortcut</label>
          <div><input id="shortcut" value={shortcut} onChange={(event) => setShortcut(event.target.value)} /><button type="submit">Save</button></div>
          {saved && <small className="success-text">Shortcut updated.</small>}
          {error && <small className="error-text">{error}</small>}
        </form>
      )}
      {view === "history" && <span className="coming-soon">Clearly marked placeholder · Not implemented</span>}
    </main>
  );
}
