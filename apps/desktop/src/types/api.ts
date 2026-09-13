export type SelectionMode = "rectangle" | "fullscreen";

export interface CaptureBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CaptureContext {
  active_app: string | null;
  window_title: string | null;
  selection_mode: SelectionMode;
  bounds: CaptureBounds;
}

export interface AnalyzePayload {
  request_id: string;
  query: string | null;
  context: CaptureContext;
}

export interface MirSummary {
  primary_modality: string;
  confidence: number;
}

export interface RouteDetails {
  intent: string;
  expert: string;
  provider: string;
  reason_code: string;
}

export type TraceStatus = "pending" | "running" | "complete" | "error";

export interface TraceEvent {
  stage: string;
  status: TraceStatus;
  message?: string;
}

export interface AnalyzeMetrics {
  latency_ms: number;
  cloud_image_uploaded: boolean;
}

export interface AnalyzeResponse {
  request_id: string;
  answer: string;
  mir_summary: MirSummary;
  route: RouteDetails;
  trace: TraceEvent[];
  metrics: AnalyzeMetrics;
}

export interface CaptureFrame {
  capture_id: string;
  image_data_url: string;
  screen: CaptureBounds;
}

export interface CaptureResult {
  capture_id: string;
  image_data_url: string;
  payload: AnalyzePayload;
}
