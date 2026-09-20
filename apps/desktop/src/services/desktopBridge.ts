import { invoke } from "@tauri-apps/api/core";
import type { BackendHealth, CaptureBounds, CaptureFrame, CaptureResult, SelectionMode } from "../types/api";

export const desktopBridge = {
  beginRectangleCapture: () => invoke<CaptureFrame>("begin_rectangle_capture"),
  captureFullscreen: (captureId: string) => invoke<CaptureResult>("finalize_fullscreen_capture", { captureId }),
  finalizeRectangle: (captureId: string, bounds: CaptureBounds) =>
    invoke<CaptureResult>("finalize_rectangle_capture", { captureId, bounds }),
  getCapture: (captureId: string) => invoke<CaptureResult>("get_capture", { captureId }),
  getActiveCaptureId: () => invoke<string | null>("get_active_capture_id"),
  markChatReady: () => invoke<void>("mark_chat_ready"),
  markCaptureDelivered: (captureId: string) => invoke<void>("mark_capture_delivered", { captureId }),
  checkBackendHealth: () => invoke<BackendHealth>("check_backend_health"),
  cancelCapture: () => invoke<void>("cancel_capture"),
  openUtility: (view: "history" | "settings" | "shortcuts") =>
    invoke<void>("open_utility_window", { view }),
  setPaused: (paused: boolean) => invoke<void>("set_assistant_paused", { paused }),
  setWidgetExpanded: (expanded: boolean) => invoke<void>("set_widget_expanded", { expanded }),
  startWidgetDrag: () => invoke<void>("start_widget_drag"),
  hideWidget: () => invoke<void>("hide_widget"),
  quit: () => invoke<void>("quit_app"),
  getShortcut: () => invoke<string>("get_capture_shortcut"),
  setShortcut: (shortcut: string) => invoke<string>("set_capture_shortcut", { shortcut }),
  selectionLabel: (mode: SelectionMode) => (mode === "rectangle" ? "Rectangle" : "Fullscreen"),
};
