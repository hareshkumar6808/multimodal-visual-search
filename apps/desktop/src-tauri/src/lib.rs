use std::{
    collections::HashMap,
    io::Cursor,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::{DynamicImage, ImageOutputFormat};
use screenshots::Screen;
use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, LogicalSize, Manager, State, WebviewUrl, WebviewWindowBuilder, Window,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use uuid::Uuid;

const BACKEND_ENDPOINT: &str = "http://127.0.0.1:8765/api/analyze";
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
    query: Option<String>,
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
    answer: Option<String>,
    suggested_actions: Vec<String>,
    mir_summary: MirSummary,
    route: RouteDetails,
    trace: Vec<TraceEvent>,
    metrics: AnalyzeMetrics,
}

fn png_data_url(bytes: &[u8]) -> String {
    format!("data:image/png;base64,{}", BASE64.encode(bytes))
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
    Ok((png, CaptureBounds { x: info.x, y: info.y, width: info.width, height: info.height }))
}

#[cfg(windows)]
fn active_window_context() -> (Option<String>, Option<String>) {
    use std::{ffi::OsString, os::windows::ffi::OsStringExt, path::PathBuf};
    use windows::{
        core::PWSTR,
        Win32::{
            Foundation::CloseHandle,
            System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT, PROCESS_QUERY_LIMITED_INFORMATION},
            UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId},
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
        let title = (copied > 0).then(|| String::from_utf16_lossy(&title_buffer[..copied as usize]));

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

fn open_capture_window(app: &AppHandle, capture_id: &str, screen: &CaptureBounds) -> Result<(), String> {
    let _ = (capture_id, screen);
    let overlay = app
        .get_webview_window("capture")
        .ok_or_else(|| "The capture window is unavailable.".to_string())?;
    overlay.set_fullscreen(true).map_err(|error| error.to_string())?;
    overlay.show().map_err(|error| error.to_string())?;
    overlay.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn open_chat_window(app: &AppHandle, capture_id: &str) -> Result<(), String> {
    let _ = capture_id;
    let chat = app
        .get_webview_window("chat")
        .ok_or_else(|| "The chat window is unavailable.".to_string())?;
    chat.show().map_err(|error| error.to_string())?;
    chat.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn start_capture(app: &AppHandle) -> Result<CaptureFrame, String> {
    if app.state::<DesktopState>().paused.load(Ordering::Relaxed) {
        return Err("Assistant is paused.".to_string());
    }
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
    app.state::<DesktopState>()
        .captures
        .lock()
        .map_err(|_| "Capture store is unavailable.".to_string())?
        .insert(capture_id.clone(), StoredCapture { png, screen: screen.clone(), active_app, window_title, result: None });
    *app.state::<DesktopState>()
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable.".to_string())? = Some(capture_id.clone());
    open_capture_window(app, &capture_id, &screen)?;
    Ok(frame)
}

#[tauri::command]
fn begin_rectangle_capture(app: AppHandle) -> Result<CaptureFrame, String> {
    start_capture(&app)
}

fn complete_capture(app: &AppHandle, capture_id: &str, png: Vec<u8>, mode: &str, bounds: CaptureBounds) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let mut captures = state.captures.lock().map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures.get_mut(capture_id).ok_or_else(|| "Capture session was not found.".to_string())?;
    let result = CaptureResult {
        capture_id: capture_id.to_string(),
        image_data_url: png_data_url(&png),
        payload: AnalyzePayload {
            request_id: Uuid::new_v4().to_string(),
            query: None,
            context: CaptureContext {
                active_app: stored.active_app.clone(),
                window_title: stored.window_title.clone(),
                selection_mode: mode.to_string(),
                bounds,
            },
        },
    };
    stored.png = png;
    stored.result = Some(result.clone());
    drop(captures);

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
fn finalize_rectangle_capture(app: AppHandle, capture_id: String, bounds: CaptureBounds) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let captures = state.captures.lock().map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures.get(&capture_id).ok_or_else(|| "Capture session was not found.".to_string())?;
    if bounds.width == 0 || bounds.height == 0 || bounds.x < 0 || bounds.y < 0
        || bounds.x as u32 + bounds.width > stored.screen.width
        || bounds.y as u32 + bounds.height > stored.screen.height
    {
        return Err("The selected rectangle is outside the captured display.".to_string());
    }
    let image = image::load_from_memory(&stored.png).map_err(|error| format!("Could not read screenshot: {error}"))?;
    let cropped = image.crop_imm(bounds.x as u32, bounds.y as u32, bounds.width, bounds.height);
    let png = encode_png(cropped)?;
    drop(captures);
    complete_capture(&app, &capture_id, png, "rectangle", bounds)
}

#[tauri::command]
fn finalize_fullscreen_capture(app: AppHandle, capture_id: String) -> Result<CaptureResult, String> {
    let state = app.state::<DesktopState>();
    let captures = state.captures.lock().map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures.get(&capture_id).ok_or_else(|| "Capture session was not found.".to_string())?;
    let png = stored.png.clone();
    let bounds = stored.screen.clone();
    drop(captures);
    complete_capture(&app, &capture_id, png, "fullscreen", bounds)
}

#[tauri::command]
fn get_capture(state: State<DesktopState>, capture_id: String) -> Result<CaptureResult, String> {
    let captures = state.captures.lock().map_err(|_| "Capture store is unavailable.".to_string())?;
    let stored = captures.get(&capture_id).ok_or_else(|| "Capture session was not found.".to_string())?;
    if let Some(result) = &stored.result {
        return Ok(result.clone());
    }
    Ok(CaptureResult {
        capture_id,
        image_data_url: png_data_url(&stored.png),
        payload: AnalyzePayload {
            request_id: Uuid::new_v4().to_string(),
            query: None,
            context: CaptureContext {
                active_app: stored.active_app.clone(),
                window_title: stored.window_title.clone(),
                selection_mode: "rectangle".to_string(),
                bounds: stored.screen.clone(),
            },
        },
    })
}

#[tauri::command]
fn get_active_capture_id(state: State<DesktopState>) -> Result<Option<String>, String> {
    state
        .active_capture
        .lock()
        .map(|capture_id| capture_id.clone())
        .map_err(|_| "Capture state is unavailable.".to_string())
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

#[tauri::command]
async fn analyze_capture(state: State<'_, DesktopState>, capture_id: String, payload: AnalyzePayload) -> Result<AnalyzeResponse, String> {
    let png = {
        let captures = state.captures.lock().map_err(|_| "Capture store is unavailable.".to_string())?;
        captures.get(&capture_id).ok_or_else(|| "Capture session was not found.".to_string())?.png.clone()
    };
    let payload_json = serde_json::to_string(&payload).map_err(|error| format!("Could not serialize request metadata: {error}"))?;
    let image_part = reqwest::multipart::Part::bytes(png)
        .file_name("capture.png")
        .mime_str("image/png")
        .map_err(|error| format!("Could not prepare screenshot upload: {error}"))?;
    let form = reqwest::multipart::Form::new()
        .part("image", image_part)
        .text("payload_json", payload_json);
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(45))
        .build()
        .map_err(|error| error.to_string())?
        .post(BACKEND_ENDPOINT)
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("Backend unavailable: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("Backend returned HTTP {}.", response.status()));
    }
    response.json::<AnalyzeResponse>().await.map_err(|error| format!("Backend returned an invalid response: {error}"))
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
    WebviewWindowBuilder::new(&app, label, WebviewUrl::App(format!("index.html?view={view}").into()))
        .title("Visual Search")
        .inner_size(430.0, 300.0)
        .resizable(false)
        .build()
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
fn set_widget_expanded(window: Window, expanded: bool) -> Result<(), String> {
    let size = if expanded { LogicalSize::new(224.0, 322.0) } else { LogicalSize::new(64.0, 64.0) };
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
    state.shortcut.lock().map(|value| value.clone()).map_err(|_| "Shortcut state is unavailable.".to_string())
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
fn set_capture_shortcut(app: AppHandle, state: State<DesktopState>, shortcut: String) -> Result<String, String> {
    let shortcut = shortcut.trim();
    if shortcut.is_empty() {
        return Err("Shortcut cannot be empty.".to_string());
    }
    let previous = state.shortcut.lock().map_err(|_| "Shortcut state is unavailable.".to_string())?.clone();
    app.global_shortcut().unregister_all().map_err(|error| error.to_string())?;
    if let Err(error) = register_capture_shortcut(&app, shortcut) {
        let _ = register_capture_shortcut(&app, &previous);
        return Err(error);
    }
    *state.shortcut.lock().map_err(|_| "Shortcut state is unavailable.".to_string())? = shortcut.to_string();
    Ok(shortcut.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(DesktopState::default())
        .setup(|app| {
            register_capture_shortcut(app.handle(), DEFAULT_CAPTURE_SHORTCUT)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            begin_rectangle_capture,
            finalize_rectangle_capture,
            finalize_fullscreen_capture,
            get_capture,
            get_active_capture_id,
            cancel_capture,
            analyze_capture,
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
