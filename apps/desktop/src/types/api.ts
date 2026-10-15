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
  conversation_id?: string | null;
  query: string | null;
  context: CaptureContext;
}

export interface ChatPayload {
  request_id: string;
  conversation_id: string;
  query: string;
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
  sub_expert?: string | null;
  leaf_route?: string | null;
  routing_confidence?: number | null;
  router_source?: "ml" | "rule" | "deterministic";
  fallback_used?: boolean;
  execution_mode?: "direct" | "tool" | "local_llm" | "cloud_text" | "cloud_vision" | null;
  send_pixels?: boolean;
  requested_tool?: string | null;
}

export type TraceStatus = "complete" | "failed" | "skipped";

export interface TraceEvent {
  stage: string;
  status: TraceStatus;
  message?: string;
  confidence?: number;
  duration_ms?: number;
}

export interface LiveProgressEvent {
  stage: string;
  status: "running" | "complete" | "failed";
  message: string;
}

export interface ProgressSnapshot {
  request_id: string;
  complete: boolean;
  events: LiveProgressEvent[];
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
  conversation_id?: string | null;
  message_id?: string | null;
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
  captured_at_ms: number;
}

export interface StoredConversationMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  has_image: boolean;
  response: AnalyzeResponse | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  capture_id: string;
  primary_modality: string;
  created_at: number;
  updated_at: number;
}

export interface ConversationDetail extends ConversationSummary {
  messages: StoredConversationMessage[];
  image_data_url: string | null;
}

export interface BackendHealth {
  connected: boolean;
  status: number | null;
  message: string;
  provider: string | null;
}

export interface StreamEvent {
  type: "start" | "progress" | "direct_result" | "tool_result" | "delta" | "done" | "error";
  text?: string;
  message?: string;
  partial_text?: string;
  provider?: string | null;
  response?: AnalyzeResponse;
  metrics?: Record<string, number>;
}
