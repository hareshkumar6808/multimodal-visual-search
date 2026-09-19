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
  provider: string | null;
  reason_code: string;
}

export type TraceStatus = "complete" | "failed" | "skipped";

export interface TraceEvent {
  stage: string;
  status: TraceStatus;
  message?: string;
  confidence?: number;
  duration_ms?: number;
}

export interface AnalyzeMetrics {
  latency_ms: number;
  perception_ms: number;
  routing_ms: number;
  provider_ms: number;
  cloud_image_uploaded: boolean;
  api_calls: number;
}

export interface AnalyzeResponse {
  request_id: string;
  answer: string | null;
  suggested_actions: string[];
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
