use std::{
    collections::HashMap,
    io::Cursor,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{DynamicImage, ImageOutputFormat};
use screenshots::Screen;
use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Emitter, LogicalSize, Manager, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, Window, WindowEvent,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use uuid::Uuid;

const BACKEND_BASE_URL: &str = "http://127.0.0.1:8765";
const DEFAULT_CAPTURE_SHORTCUT: &str = "Ctrl+Shift+Period";

struct DesktopState {
    captures: Mutex<HashMap<String, StoredCapture>>,
    active_capture: Mutex<Option<String>>,
    paused: AtomicBool,
    shortcut: Mutex<String>,
}

impl Default for DesktopState {
    fn default() -> Self {
        Self {
            captures: Mutex::new(HashMap::new()),
            active_capture: Mutex::new(None),
            paused: AtomicBool::new(false),
            shortcut: Mutex::new(DEFAULT_CAPTURE_SHORTCUT.to_string()),
        }
    }
}

#[derive(Clone)]
struct StoredCapture {
    png: Vec<u8>,
    screen: CaptureBounds,
    active_app: Option<String>,
    window_title: Option<String>,
    result: Option<CaptureResult>,
    captured_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CaptureBounds {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CaptureContext {
    active_app: Option<String>,
    window_title: Option<String>,
    selection_mode: String,
    bounds: CaptureBounds,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AnalyzePayload {
    request_id: String,
    conversation_id: Option<String>,
    query: Option<String>,
    context: CaptureContext,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatPayload {
    request_id: String,
    conversation_id: String,
    query: String,
    context: CaptureContext,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CaptureFrame {
    capture_id: String,
    image_data_url: String,
    screen: CaptureBounds,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CaptureResult {
    capture_id: String,
    image_data_url: String,
    payload: AnalyzePayload,
    captured_at_ms: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct MirSummary {
    primary_modality: String,
    confidence: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct RouteDetails {
    intent: String,
    expert: String,
    provider: Option<String>,
    reason_code: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TraceEvent {
    stage: String,
    status: String,
    message: Option<String>,
    confidence: Option<f64>,
    duration_ms: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct AnalyzeMetrics {
    latency_ms: u64,
    perception_ms: u64,
    routing_ms: u64,
    provider_ms: u64,
    cloud_image_uploaded: bool,
    api_calls: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct AnalyzeResponse {
    request_id: String,
    conversation_id: Option<String>,
    message_id: Option<String>,
    answer: Option<String>,
    suggested_actions: Vec<String>,
    mir_summary: MirSummary,
    route: RouteDetails,
    trace: Vec<TraceEvent>,
    metrics: AnalyzeMetrics,
}

#[derive(Debug, Serialize)]
struct BackendHealth {
    connected: bool,
    status: Option<u16>,
    message: String,
    provider: Option<String>,
}

#[derive(Debug, Serialize)]
struct CommandError {
    kind: &'static str,
    message: String,
    status: Option<u16>,
}

impl CommandError {
    fn new(kind: &'static str, message: impl Into<String>, status: Option<u16>) -> Self {
        Self {
            kind,
            message: message.into(),
            status,
        }
    }
}

impl DesktopState {
    fn store_capture(&self, capture_id: String, capture: StoredCapture) -> Result<(), String> {
        let mut captures = self
            .captures
            .lock()
            .map_err(|_| "Capture store is unavailable.".to_string())?;
        captures.clear();
        captures.insert(capture_id.clone(), capture);
        *self
            .active_capture
            .lock()
            .map_err(|_| "Capture state is unavailable.".to_string())? = Some(capture_id);
        Ok(())
    }

    fn active_capture_id(&self) -> Result<Option<String>, String> {
        self.active_capture
            .lock()
            .map(|capture_id| capture_id.clone())
            .map_err(|_| "Capture state is unavailable.".to_string())
    }
}

fn png_data_url(bytes: &[u8]) -> String {
    format!("data:image/png;base64,{}", BASE64.encode(bytes))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn encode_png(image: DynamicImage) -> Result<Vec<u8>, String> {
    let mut output = Cursor::new(Vec::new());
    image
        .write_to(&mut output, ImageOutputFormat::Png)
        .map_err(|error| format!("Could not encode screenshot: {error}"))?;
    Ok(output.into_inner())
}

fn capture_primary_screen() -> Result<(Vec<u8>, CaptureBounds), String> {
    let screen = Screen::all()
        .map_err(|error| format!("Could not enumerate displays: {error}"))?
        .into_iter()
        .find(|screen| screen.display_info.is_primary)
        .ok_or_else(|| "No primary display was found.".to_string())?;
    let info = screen.display_info;
    let image = screen
        .capture()
        .map_err(|error| format!("Could not capture the display: {error}"))?;
    let png = encode_png(DynamicImage::ImageRgba8(image))?;
    Ok((
        png,
        CaptureBounds {
            x: info.x,
            y: info.y,
            width: info.width,
            height: info.height,
        },
    ))
}

#[cfg(windows)]
fn active_window_context() -> (Option<String>, Option<String>) {
    use std::{ffi::OsString, os::windows::ffi::OsStringExt, path::PathBuf};
    use windows::{
        core::PWSTR,
        Win32::{
            Foundation::CloseHandle,
            System::Threading::{
                OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT,
                PROCESS_QUERY_LIMITED_INFORMATION,
            },
            UI::WindowsAndMessaging::{
                GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
            },
        },
    };

    unsafe {
        let window = GetForegroundWindow();
        if window.0.is_null() {
            return (None, None);
        }

        let title_length = GetWindowTextLengthW(window);
        let mut title_buffer = vec![0_u16; title_length.max(0) as usize + 1];
        let copied = GetWindowTextW(window, &mut title_buffer);
        let title =
            (copied > 0).then(|| String::from_utf16_lossy(&title_buffer[..copied as usize]));

        let mut process_id = 0_u32;
        GetWindowThreadProcessId(window, Some(&mut process_id));
        let app = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id)
            .ok()
            .and_then(|process| {
                let mut path_buffer = vec![0_u16; 32_768];
                let mut size = path_buffer.len() as u32;
                let result = QueryFullProcessImageNameW(
                    process,
                    PROCESS_NAME_FORMAT(0),
                    PWSTR(path_buffer.as_mut_ptr()),
                    &mut size,
                );
                let _ = CloseHandle(process);
                result.ok().and_then(|_| {
                    PathBuf::from(OsString::from_wide(&path_buffer[..size as usize]))
                        .file_stem()
                        .map(|name| name.to_string_lossy().into_owned())
                })
            });

        (app, title)
    }
}

#[cfg(not(windows))]
fn active_window_context() -> (Option<String>, Option<String>) {
    (None, None)
}

fn open_capture_window(
    app: &AppHandle,
    capture_id: &str,
    screen: &CaptureBounds,
) -> Result<(), String> {
    let _ = (capture_id, screen);
    let overlay = app
        .get_webview_window("capture")
        .ok_or_else(|| "The capture window is unavailable.".to_string())?;
    overlay
        .set_fullscreen(true)
        .map_err(|error| error.to_string())?;
    overlay.show().map_err(|error| error.to_string())?;
    overlay.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn configure_chat_window(chat: &WebviewWindow) {
    let window = chat.clone();
    chat.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            let _ = window.hide();
        }
    });
}

fn open_chat_window(app: &AppHandle, capture_id: &str) -> Result<(), String> {
    let chat = if let Some(existing) = app.get_webview_window("chat") {
        eprintln!("CHAT_WINDOW_REUSED capture_id={capture_id}");
        existing
    } else {
        let created =
            WebviewWindowBuilder::new(app, "chat", WebviewUrl::App("index.html?view=chat".into()))
                .title("Visual Search")
                .inner_size(440.0, 650.0)
                .min_inner_size(360.0, 500.0)
                .resizable(true)
                .visible(false)
                .build()
                .map_err(|error| format!("Could not create the chat window: {error}"))?;
        configure_chat_window(&created);
        eprintln!("CHAT_WINDOW_CREATED capture_id={capture_id}");
        created
    };
    let _ = chat.unminimize();
    chat.show().map_err(|error| error.to_string())?;
    chat.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn start_capture(app: &AppHandle) -> Result<CaptureFrame, String> {
    if app.state::<DesktopState>().paused.load(Ordering::Relaxed) {
        return Err("Assistant is paused.".to_string());
    }
    eprintln!("CAPTURE_STARTED");
    let (active_app, window_title) = active_window_context();
    if let Some(widget) = app.get_webview_window("widget") {
        widget.hide().map_err(|error| error.to_string())?;
    }
    std::thread::sleep(Duration::from_millis(90));
    let (png, screen) = capture_primary_screen()?;
    let capture_id = Uuid::new_v4().to_string();
    let frame = CaptureFrame {
        capture_id: capture_id.clone(),
        image_data_url: png_data_url(&png),
        screen: screen.clone(),
    };
    app.state::<DesktopState>().store_capture(
        capture_id.clone(),
        StoredCapture {
            png,
            screen: screen.clone(),
            active_app,
            window_title,
            result: None,
            captured_at_ms: unix_time_ms(),
        },
    )?;
    eprintln!("CAPTURE_STORED capture_id={capture_id}");
    open_capture_window(app, &capture_id, &screen)?;
    Ok(frame)
}

#[tauri::command]
fn begin_rectangle_capture(app: AppHandle) -> Result<CaptureFrame, String> {
    start_capture(&app)
}

fn complete_capture(
    app: &AppHandle,
    capture_id: &str,
    png: Vec<u8>,
    mode: &str,
    bounds: CaptureBounds,
) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let mut captures = state
        .captures
        .lock()
        .map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures
        .get_mut(capture_id)
        .ok_or_else(|| "Capture session was not found.".to_string())?;
    let result = CaptureResult {
        capture_id: capture_id.to_string(),
        image_data_url: png_data_url(&png),
        payload: AnalyzePayload {
            request_id: Uuid::new_v4().to_string(),
            conversation_id: None,
            query: None,
            context: CaptureContext {
                active_app: stored.active_app.clone(),
                window_title: stored.window_title.clone(),
                selection_mode: mode.to_string(),
                bounds,
            },
        },
        captured_at_ms: stored.captured_at_ms,
    };
    stored.png = png;
    stored.result = Some(result.clone());
    drop(captures);
    eprintln!("CAPTURE_COMPLETED capture_id={capture_id} mode={mode}");
    eprintln!("CAPTURE_STORED capture_id={capture_id} completed=true");
    app.emit("capture-stored", capture_id)
        .map_err(|error| format!("Could not notify the chat window: {error}"))?;

    if let Some(overlay) = app.get_webview_window("capture") {
        let _ = overlay.set_fullscreen(false);
        let _ = overlay.hide();
    }
    if let Some(widget) = app.get_webview_window("widget") {
        let _ = widget.show();
    }
    open_chat_window(app, capture_id)?;
    Ok(result)
}

#[tauri::command]
fn finalize_rectangle_capture(
    app: AppHandle,
    capture_id: String,
    bounds: CaptureBounds,
) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let captures = state
        .captures
        .lock()
        .map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures
        .get(&capture_id)
        .ok_or_else(|| "Capture session was not found.".to_string())?;
    if bounds.width == 0
        || bounds.height == 0
        || bounds.x < 0
        || bounds.y < 0
        || bounds.x as u32 + bounds.width > stored.screen.width
        || bounds.y as u32 + bounds.height > stored.screen.height
    {
        return Err("The selected rectangle is outside the captured display.".to_string());
    }
    let image = image::load_from_memory(&stored.png)
        .map_err(|error| format!("Could not read screenshot: {error}"))?;
    let cropped = image.crop_imm(
        bounds.x as u32,
        bounds.y as u32,
        bounds.width,
        bounds.height,
    );
    let png = encode_png(cropped)?;
    drop(captures);
    complete_capture(&app, &capture_id, png, "rectangle", bounds)
}

#[tauri::command]
fn finalize_fullscreen_capture(
    app: AppHandle,
    capture_id: String,
) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let captures = state
        .captures
        .lock()
        .map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures
        .get(&capture_id)
        .ok_or_else(|| "Capture session was not found.".to_string())?;
    let png = stored.png.clone();
    let bounds = stored.screen.clone();
    drop(captures);
    complete_capture(&app, &capture_id, png, "fullscreen", bounds)
}

#[tauri::command]
fn get_capture(state: State<DesktopState>, capture_id: String) -> Result<CaptureResult, String> {
    let captures = state
        .captures
        .lock()
        .map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures
        .get(&capture_id)
        .ok_or_else(|| "Capture session was not found.".to_string())?;
    if let Some(result) = &stored.result {
        return Ok(result.clone());
    }
    Ok(CaptureResult {
        capture_id,
        image_data_url: png_data_url(&stored.png),
        payload: AnalyzePayload {
            request_id: Uuid::new_v4().to_string(),
            conversation_id: None,
            query: None,
            context: CaptureContext {
                active_app: stored.active_app.clone(),
                window_title: stored.window_title.clone(),
                selection_mode: "rectangle".to_string(),
                bounds: stored.screen.clone(),
            },
        },
        captured_at_ms: stored.captured_at_ms,
    })
}

#[tauri::command]
fn get_active_capture_id(state: State<DesktopState>) -> Result<Option<String>, String> {
    state.active_capture_id()
}

#[tauri::command]
fn mark_chat_ready() {
    eprintln!("CHAT_WINDOW_READY");
}

#[tauri::command]
fn mark_capture_delivered(capture_id: String) {
    eprintln!("CAPTURE_DELIVERED capture_id={capture_id}");
}

#[tauri::command]
fn cancel_capture(app: AppHandle, window: Window) -> Result<(), String> {
    if let Some(overlay) = app.get_webview_window("capture") {
        let _ = overlay.set_fullscreen(false);
        overlay.hide().map_err(|error| error.to_string())?;
    } else {
        window.hide().map_err(|error| error.to_string())?;
    }
    *app.state::<DesktopState>()
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable.".to_string())? = None;
    if let Some(widget) = app.get_webview_window("widget") {
        widget.show().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn response_detail(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("detail")
                .and_then(|detail| detail.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.trim().to_string())
}

async fn decode_backend_response(
    response: reqwest::Response,
    request_label: &str,
) -> Result<AnalyzeResponse, CommandError> {
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        CommandError::new("invalid_response", error.to_string(), Some(status.as_u16()))
    })?;
    eprintln!(
        "ANALYZE_RESPONSE_RECEIVED request={request_label} status={}",
        status.as_u16()
    );
    if !status.is_success() {
        let detail = response_detail(&body);
        if status.as_u16() == 503 && detail.contains("No configured, available provider") {
            eprintln!("PROVIDER_UNAVAILABLE request={request_label}");
            return Err(CommandError::new(
                "provider_unavailable",
                "AI provider is not configured. OCR and routing completed successfully, but an AI provider is required to generate the final answer.",
                Some(503),
            ));
        }
        let kind = if status.is_client_error() {
            "request"
        } else {
            "server"
        };
        return Err(CommandError::new(kind, detail, Some(status.as_u16())));
    }
    serde_json::from_str::<AnalyzeResponse>(&body).map_err(|error| {
        CommandError::new(
            "invalid_response",
            format!("Backend returned an invalid response: {error}"),
            Some(status.as_u16()),
        )
    })
}

#[tauri::command]
async fn check_backend_health() -> BackendHealth {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            eprintln!("BACKEND_HEALTH_FAILED client={error}");
            return BackendHealth {
                connected: false,
                status: None,
                message: error.to_string(),
                provider: None,
            };
        }
    };
    let waits = [0_u64, 500, 1_000];
    let mut last_message = "Backend did not respond.".to_string();
    for wait_ms in waits {
        if wait_ms > 0 {
            tokio::time::sleep(Duration::from_millis(wait_ms)).await;
        }
        match client
            .get(format!("{BACKEND_BASE_URL}/api/health"))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let status = response.status().as_u16();
                let provider = response
                    .json::<serde_json::Value>()
                    .await
                    .ok()
                    .and_then(|body| {
                        body.get("providers")
                            .and_then(|value| value.as_array())
                            .cloned()
                    })
                    .and_then(|providers| {
                        providers.into_iter().find_map(|provider| {
                            provider
                                .get("available")
                                .and_then(|value| value.as_bool())
                                .unwrap_or(false)
                                .then(|| provider.get("name")?.as_str().map(str::to_owned))
                                .flatten()
                        })
                    });
                eprintln!("BACKEND_HEALTH_OK status={status}");
                return BackendHealth {
                    connected: true,
                    status: Some(status),
                    message: "Connected".to_string(),
                    provider,
                };
            }
            Ok(response) => {
                last_message =
                    format!("Health check returned HTTP {}.", response.status().as_u16());
            }
            Err(error) => last_message = error.to_string(),
        }
    }
    eprintln!("BACKEND_HEALTH_FAILED reason={last_message}");
    BackendHealth {
        connected: false,
        status: None,
        message: last_message,
        provider: None,
    }
}

#[tauri::command]
async fn analyze_capture(
    state: State<'_, DesktopState>,
    capture_id: String,
    payload: AnalyzePayload,
) -> Result<AnalyzeResponse, CommandError> {
    let png = {
        let captures = state
            .captures
            .lock()
            .map_err(|_| CommandError::new("capture", "Capture store is unavailable.", None))?;
        captures
            .get(&capture_id)
            .ok_or_else(|| CommandError::new("capture", "Capture session was not found.", None))?
            .png
            .clone()
    };
    let payload_json = serde_json::to_string(&payload).map_err(|error| {
        CommandError::new(
            "request",
            format!("Could not serialize request metadata: {error}"),
            None,
        )
    })?;
    let image_part = reqwest::multipart::Part::bytes(png)
        .file_name("capture.png")
        .mime_str("image/png")
        .map_err(|error| {
            CommandError::new(
                "request",
                format!("Could not prepare screenshot upload: {error}"),
                None,
            )
        })?;
    let form = reqwest::multipart::Form::new()
        .part("image", image_part)
        .text("payload_json", payload_json);
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|error| CommandError::new("network", error.to_string(), None))?
        .post(format!("{BACKEND_BASE_URL}/api/analyze"))
        .multipart(form)
        .send()
        .await
        .map_err(|error| {
            let kind = if error.is_timeout() {
                "timeout"
            } else {
                "network"
            };
            CommandError::new(kind, error.to_string(), None)
        })?;
    eprintln!("ANALYZE_REQUEST_SENT capture_id={capture_id}");
    decode_backend_response(response, &capture_id).await
}

#[tauri::command]
async fn continue_conversation(payload: ChatPayload) -> Result<AnalyzeResponse, CommandError> {
    let request_id = payload.request_id.clone();
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|error| CommandError::new("network", error.to_string(), None))?
        .post(format!("{BACKEND_BASE_URL}/api/chat"))
        .json(&payload)
        .send()
        .await
        .map_err(|error| {
            let kind = if error.is_timeout() {
                "timeout"
            } else {
                "network"
            };
            CommandError::new(kind, error.to_string(), None)
        })?;
    eprintln!("CHAT_REQUEST_SENT request_id={request_id}");
    decode_backend_response(response, &request_id).await
}

async fn get_backend_json(path: &str) -> Result<serde_json::Value, CommandError> {
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|error| CommandError::new("network", error.to_string(), None))?
        .get(format!("{BACKEND_BASE_URL}{path}"))
        .send()
        .await
        .map_err(|error| CommandError::new("network", error.to_string(), None))?;
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        CommandError::new("invalid_response", error.to_string(), Some(status.as_u16()))
    })?;
    if !status.is_success() {
        return Err(CommandError::new(
            "request",
            response_detail(&body),
            Some(status.as_u16()),
        ));
    }
    serde_json::from_str(&body).map_err(|error| {
        CommandError::new("invalid_response", error.to_string(), Some(status.as_u16()))
    })
}

#[tauri::command]
async fn list_conversations() -> Result<serde_json::Value, CommandError> {
    get_backend_json("/api/conversations").await
}

#[tauri::command]
async fn get_conversation(conversation_id: String) -> Result<serde_json::Value, CommandError> {
    get_backend_json(&format!("/api/conversations/{conversation_id}")).await
}

#[tauri::command]
async fn get_analysis_progress(request_id: String) -> Result<serde_json::Value, CommandError> {
    get_backend_json(&format!("/api/progress/{request_id}")).await
}

#[tauri::command]
fn open_utility_window(app: AppHandle, view: String) -> Result<(), String> {
    if !matches!(view.as_str(), "history" | "settings" | "shortcuts") {
        return Err("Unknown utility page.".to_string());
    }
    let label = format!("utility-{view}");
    if let Some(existing) = app.get_webview_window(&label) {
        existing.set_focus().map_err(|error| error.to_string())?;
        return Ok(());
    }
    WebviewWindowBuilder::new(
        &app,
        label,
        WebviewUrl::App(format!("index.html?view={view}").into()),
    )
    .title("Visual Search")
    .inner_size(430.0, 300.0)
    .resizable(false)
    .build()
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
fn set_widget_expanded(window: Window, expanded: bool) -> Result<(), String> {
    let size = if expanded {
        LogicalSize::new(224.0, 322.0)
    } else {
        LogicalSize::new(64.0, 64.0)
    };
    window.set_size(size).map_err(|error| error.to_string())
}

#[tauri::command]
fn start_widget_drag(window: Window) -> Result<(), String> {
    window.start_dragging().map_err(|error| error.to_string())
}

#[tauri::command]
fn hide_widget(window: Window) -> Result<(), String> {
    window.hide().map_err(|error| error.to_string())
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
fn get_capture_shortcut(state: State<DesktopState>) -> Result<String, String> {
    state
        .shortcut
        .lock()
        .map(|value| value.clone())
        .map_err(|_| "Shortcut state is unavailable.".to_string())
}

fn register_capture_shortcut(app: &AppHandle, shortcut: &str) -> Result<(), String> {
    app.global_shortcut()
        .on_shortcut(shortcut, |app, _, event| {
            if event.state == ShortcutState::Pressed {
                let _ = start_capture(app);
            }
        })
        .map_err(|error| format!("Invalid or unavailable shortcut: {error}"))
}

#[tauri::command]
fn set_assistant_paused(state: State<DesktopState>, paused: bool) {
    state.paused.store(paused, Ordering::Relaxed);
}

#[tauri::command]
fn set_capture_shortcut(
    app: AppHandle,
    state: State<DesktopState>,
    shortcut: String,
) -> Result<String, String> {
    let shortcut = shortcut.trim();
    if shortcut.is_empty() {
        return Err("Shortcut cannot be empty.".to_string());
    }
    let previous = state
        .shortcut
        .lock()
        .map_err(|_| "Shortcut state is unavailable.".to_string())?
        .clone();
    app.global_shortcut()
        .unregister_all()
        .map_err(|error| error.to_string())?;
    if let Err(error) = register_capture_shortcut(&app, shortcut) {
        let _ = register_capture_shortcut(&app, &previous);
        return Err(error);
    }
    *state
        .shortcut
        .lock()
        .map_err(|_| "Shortcut state is unavailable.".to_string())? = shortcut.to_string();
    Ok(shortcut.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(DesktopState::default())
        .setup(|app| {
            if let Err(error) = register_capture_shortcut(app.handle(), DEFAULT_CAPTURE_SHORTCUT) {
                eprintln!("Global capture shortcut is unavailable: {error}");
            }
            if let Some(chat) = app.get_webview_window("chat") {
                configure_chat_window(&chat);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            begin_rectangle_capture,
            finalize_rectangle_capture,
            finalize_fullscreen_capture,
            get_capture,
            get_active_capture_id,
            mark_chat_ready,
            mark_capture_delivered,
            cancel_capture,
            check_backend_health,
            analyze_capture,
            continue_conversation,
            get_analysis_progress,
            list_conversations,
            get_conversation,
            open_utility_window,
            set_widget_expanded,
            start_widget_drag,
            hide_widget,
            quit_app,
            get_capture_shortcut,
            set_capture_shortcut,
            set_assistant_paused
        ])
        .run(tauri::generate_context!())
        .expect("error while running the desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_capture(byte: u8) -> StoredCapture {
        StoredCapture {
            png: vec![byte],
            screen: CaptureBounds {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
            active_app: None,
            window_title: None,
            result: None,
            captured_at_ms: 1,
        }
    }

    #[test]
    fn latest_capture_replaces_previous_capture_state() {
        let state = DesktopState::default();
        state
            .store_capture("capture-a".to_string(), stored_capture(1))
            .unwrap();
        state
            .store_capture("capture-b".to_string(), stored_capture(2))
            .unwrap();

        assert_eq!(
            state.active_capture_id().unwrap().as_deref(),
            Some("capture-b")
        );
        let captures = state.captures.lock().unwrap();
        assert!(captures.get("capture-a").is_none());
        assert_eq!(captures.get("capture-b").unwrap().png, vec![2]);
    }

    #[test]
    fn extracts_provider_detail_and_uses_canonical_backend_url() {
        assert_eq!(
            response_detail(
                r#"{"detail":"No configured, available provider supports this route"}"#
            ),
            "No configured, available provider supports this route"
        );
        assert_eq!(BACKEND_BASE_URL, "http://127.0.0.1:8765");
    }
}
