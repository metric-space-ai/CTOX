// ref: internal/runtime/executor/helps/logging_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Request-scoped upstream logging without Gin or package globals.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::internal::logging::{self, RequestContext};
use crate::internal::util::{hide_api_key, mask_sensitive_header_value};

pub const MAX_DEFERRED_API_REQUEST_BODY_BYTES: usize = 32 << 20;
pub type LogHeaders = BTreeMap<String, Vec<String>>;
pub type DeferredApiRequest = Arc<dyn Fn() -> Vec<u8> + Send + Sync>;

pub trait ApiLogClock: Send + Sync {
    fn now(&self) -> SystemTime;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemApiLogClock;

impl ApiLogClock for SystemApiLogClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequestLogPolicy {
    pub request_log: bool,
    pub commercial_mode: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UpstreamRequestLog {
    pub url: String,
    pub method: String,
    pub headers: LogHeaders,
    pub body: Vec<u8>,
    pub provider: String,
    pub auth_id: String,
    pub auth_label: String,
    pub auth_type: String,
    pub auth_value: String,
}

#[derive(Default)]
struct UpstreamAttempt {
    index: usize,
    request: Vec<u8>,
    response: Vec<u8>,
    response_intro_written: bool,
    status_written: bool,
    headers_written: bool,
    body_started: bool,
    body_has_content: bool,
    prev_was_sse_event: bool,
    error_written: bool,
}

#[derive(Default)]
struct ApiLogState {
    attempts: Vec<UpstreamAttempt>,
    deferred: Vec<DeferredApiRequest>,
    deferred_bytes: usize,
    websocket_timeline: Vec<u8>,
    response_timestamp: Option<SystemTime>,
    credits_used: bool,
}

pub struct ApiLogContext {
    clock: Arc<dyn ApiLogClock>,
    request_context: RequestContext,
    state: Mutex<ApiLogState>,
}

impl ApiLogContext {
    #[must_use]
    pub fn new(clock: Arc<dyn ApiLogClock>, request_context: RequestContext) -> Self {
        Self {
            clock,
            request_context,
            state: Mutex::new(ApiLogState::default()),
        }
    }

    #[must_use]
    pub fn deferred_requests(&self) -> Vec<DeferredApiRequest> {
        self.state.lock().unwrap().deferred.clone()
    }

    #[must_use]
    pub fn aggregated_request(&self) -> Vec<u8> {
        self.state
            .lock()
            .unwrap()
            .attempts
            .iter()
            .flat_map(|attempt| attempt.request.iter().copied())
            .collect()
    }

    #[must_use]
    pub fn aggregated_response(&self) -> Vec<u8> {
        aggregate_responses(&self.state.lock().unwrap().attempts)
    }

    #[must_use]
    pub fn websocket_timeline(&self) -> Vec<u8> {
        self.state.lock().unwrap().websocket_timeline.clone()
    }

    #[must_use]
    pub fn request_context(&self) -> &RequestContext {
        &self.request_context
    }
}

fn capture_enabled(policy: RequestLogPolicy) -> bool {
    policy.request_log && !policy.commercial_mode
}

pub fn record_api_request(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    info: UpstreamRequestLog,
) {
    if policy.commercial_mode {
        return;
    }
    let Some(context) = context else { return };
    let mut state = context.state.lock().unwrap();
    let index = if policy.request_log {
        state.attempts.len() + 1
    } else {
        state.deferred.len() + 1
    };
    if !policy.request_log {
        let remaining = MAX_DEFERRED_API_REQUEST_BODY_BYTES.saturating_sub(state.deferred_bytes);
        let capture_length = info.body.len().min(remaining);
        let captured_body = info.body[..capture_length].to_vec();
        let body_empty = info.body.is_empty();
        let body_truncated = capture_length < info.body.len();
        let captured_at = context.clock.now();
        let captured = UpstreamRequestLog {
            body: captured_body,
            ..info
        };
        state.deferred_bytes += capture_length;
        state.deferred.push(Arc::new(move || {
            request_log_bytes(index, &captured, captured_at, body_empty, body_truncated)
        }));
        return;
    }
    let request = request_log_bytes(
        index,
        &info,
        context.clock.now(),
        info.body.is_empty(),
        false,
    );
    state.attempts.push(UpstreamAttempt {
        index,
        request,
        ..UpstreamAttempt::default()
    });
}

fn request_log_bytes(
    index: usize,
    info: &UpstreamRequestLog,
    timestamp: SystemTime,
    body_empty: bool,
    body_truncated: bool,
) -> Vec<u8> {
    let mut output = format!(
        "=== API REQUEST {index} ===\nTimestamp: {}\nUpstream URL: {}\n",
        timestamp_string(timestamp),
        if info.url.is_empty() {
            "<unknown>"
        } else {
            &info.url
        }
    );
    if !info.method.is_empty() {
        let _ = writeln!(output, "HTTP Method: {}", info.method);
    }
    let auth = format_auth_info(info);
    if !auth.is_empty() {
        let _ = writeln!(output, "Auth: {auth}");
    }
    output.push_str("\nHeaders:\n");
    write_headers(&mut output, &info.headers);
    output.push_str("\nBody:\n");
    if body_empty {
        output.push_str("<empty>");
    } else {
        output.push_str(&String::from_utf8_lossy(&info.body));
        if body_truncated {
            let _ = write!(
                output,
                "\n[API REQUEST BODY TRUNCATED: captured first {} bytes]",
                info.body.len()
            );
        }
    }
    output.push_str("\n\n");
    output.into_bytes()
}

pub fn record_api_response_metadata(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    status: u16,
    headers: &LogHeaders,
) {
    if let Some(context) = context {
        logging::set_response_headers(Some(context.request_context()), headers);
    }
    if !capture_enabled(policy) {
        return;
    }
    let Some(context) = context else { return };
    let mut state = context.state.lock().unwrap();
    let attempt = ensure_attempt(&mut state);
    ensure_response_intro(attempt, context.clock.now());
    if status > 0 && !attempt.status_written {
        attempt
            .response
            .extend_from_slice(format!("Status: {status}\n").as_bytes());
        attempt.status_written = true;
    }
    if !attempt.headers_written {
        let mut rendered = String::from("Headers:\n");
        write_headers(&mut rendered, headers);
        rendered.push('\n');
        attempt.response.extend_from_slice(rendered.as_bytes());
        attempt.headers_written = true;
    }
}

pub fn record_api_response_error(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    error: &str,
) {
    if !capture_enabled(policy) || error.is_empty() {
        return;
    }
    let Some(context) = context else { return };
    let mut state = context.state.lock().unwrap();
    let attempt = ensure_attempt(&mut state);
    ensure_response_intro(attempt, context.clock.now());
    if attempt.body_started && !attempt.body_has_content {
        attempt.body_started = false;
    }
    if attempt.error_written {
        attempt.response.push(b'\n');
    }
    attempt
        .response
        .extend_from_slice(format!("Error: {error}\n").as_bytes());
    attempt.error_written = true;
}

pub fn append_api_response_chunk(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    chunk: &[u8],
) {
    if !capture_enabled(policy) {
        return;
    }
    let data = trim_ascii(chunk);
    if data.is_empty() {
        return;
    }
    let Some(context) = context else { return };
    let mut state = context.state.lock().unwrap();
    let attempt = ensure_attempt(&mut state);
    ensure_response_intro(attempt, context.clock.now());
    if !attempt.headers_written {
        attempt.response.extend_from_slice(b"Headers:\n<none>\n\n");
        attempt.headers_written = true;
    }
    if !attempt.body_started {
        attempt.response.extend_from_slice(b"Body:\n");
        attempt.body_started = true;
    }
    let is_event = data.starts_with(b"event:");
    let is_data = data.starts_with(b"data:");
    if attempt.body_has_content {
        attempt
            .response
            .extend_from_slice(if attempt.prev_was_sse_event && is_data {
                b"\n"
            } else {
                b"\n\n"
            });
    }
    attempt.response.extend_from_slice(data);
    attempt.body_has_content = true;
    attempt.prev_was_sse_event = is_event;
}

pub fn record_api_websocket_request(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    info: &UpstreamRequestLog,
) {
    if !capture_enabled(policy) {
        return;
    }
    let Some(context) = context else { return };
    let mut event = format!(
        "Timestamp: {}\nEvent: api.websocket.request\n",
        timestamp_string(context.clock.now())
    );
    if !info.url.is_empty() {
        let _ = writeln!(event, "Upstream URL: {}", info.url);
    }
    let auth = format_auth_info(info);
    if !auth.is_empty() {
        let _ = writeln!(event, "Auth: {auth}");
    }
    event.push_str("Headers:\n");
    write_headers(&mut event, &info.headers);
    event.push_str("\nBody:\n");
    if info.body.is_empty() {
        event.push_str("<empty>");
    } else {
        event.push_str(&String::from_utf8_lossy(&info.body));
    }
    append_websocket_timeline(context, event.as_bytes());
}

pub fn record_api_websocket_handshake(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    status: u16,
    headers: &LogHeaders,
) {
    if let Some(context) = context {
        logging::set_response_headers(Some(context.request_context()), headers);
    }
    if !capture_enabled(policy) {
        return;
    }
    let Some(context) = context else { return };
    let mut event = format!(
        "Timestamp: {}\nEvent: api.websocket.handshake\n",
        timestamp_string(context.clock.now())
    );
    if status > 0 {
        let _ = writeln!(event, "Status: {status}");
    }
    event.push_str("Headers:\n");
    write_headers(&mut event, headers);
    append_websocket_timeline(context, event.as_bytes());
}

pub fn record_api_websocket_upgrade_rejection(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    info: UpstreamRequestLog,
    status: u16,
    headers: &LogHeaders,
    body: &[u8],
) {
    record_api_request(context, policy, info);
    record_api_response_metadata(context, policy, status, headers);
    append_api_response_chunk(context, policy, body);
}

#[must_use]
pub fn websocket_upgrade_request_url(raw_url: &str) -> String {
    let raw_url = raw_url.trim();
    let Ok(mut parsed) = url::Url::parse(raw_url) else {
        return raw_url.to_owned();
    };
    match parsed.scheme().to_ascii_lowercase().as_str() {
        "ws" => {
            let _ = parsed.set_scheme("http");
        }
        "wss" => {
            let _ = parsed.set_scheme("https");
        }
        _ => {}
    }
    parsed.to_string()
}

pub fn append_api_websocket_response(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    payload: &[u8],
) {
    if !capture_enabled(policy) || trim_ascii(payload).is_empty() {
        return;
    }
    let Some(context) = context else { return };
    let now = context.clock.now();
    context
        .state
        .lock()
        .unwrap()
        .response_timestamp
        .get_or_insert(now);
    let event = format!(
        "Timestamp: {}\nEvent: api.websocket.response\n{}\n",
        timestamp_string(now),
        String::from_utf8_lossy(trim_ascii(payload))
    );
    append_websocket_timeline(context, event.as_bytes());
}

pub fn record_api_websocket_error(
    context: Option<&ApiLogContext>,
    policy: RequestLogPolicy,
    stage: &str,
    error: &str,
) {
    if !capture_enabled(policy) || error.is_empty() {
        return;
    }
    let Some(context) = context else { return };
    let now = context.clock.now();
    context
        .state
        .lock()
        .unwrap()
        .response_timestamp
        .get_or_insert(now);
    let mut event = format!(
        "Timestamp: {}\nEvent: api.websocket.error\n",
        timestamp_string(now)
    );
    if !stage.trim().is_empty() {
        let _ = writeln!(event, "Stage: {}", stage.trim());
    }
    let _ = writeln!(event, "Error: {error}");
    append_websocket_timeline(context, event.as_bytes());
}

fn ensure_attempt(state: &mut ApiLogState) -> &mut UpstreamAttempt {
    if state.attempts.is_empty() {
        state.attempts.push(UpstreamAttempt {
            index: 1,
            request: b"=== API REQUEST 1 ===\n<missing>\n\n".to_vec(),
            ..UpstreamAttempt::default()
        });
    }
    state.attempts.last_mut().expect("attempt initialized")
}

fn ensure_response_intro(attempt: &mut UpstreamAttempt, now: SystemTime) {
    if attempt.response_intro_written {
        return;
    }
    attempt.response.extend_from_slice(
        format!(
            "=== API RESPONSE {} ===\nTimestamp: {}\n\n",
            attempt.index,
            timestamp_string(now)
        )
        .as_bytes(),
    );
    attempt.response_intro_written = true;
}

fn aggregate_responses(attempts: &[UpstreamAttempt]) -> Vec<u8> {
    let mut output = Vec::new();
    for attempt in attempts {
        if attempt.response.is_empty() {
            continue;
        }
        if !output.is_empty() {
            output.push(b'\n');
        }
        output.extend_from_slice(&attempt.response);
        if !attempt.response.ends_with(b"\n") {
            output.push(b'\n');
        }
    }
    output
}

fn append_websocket_timeline(context: &ApiLogContext, chunk: &[u8]) {
    let data = trim_ascii(chunk);
    if data.is_empty() {
        return;
    }
    let mut state = context.state.lock().unwrap();
    if !state.websocket_timeline.is_empty() {
        if !state.websocket_timeline.ends_with(b"\n") {
            state.websocket_timeline.push(b'\n');
        }
        state.websocket_timeline.push(b'\n');
    }
    state.websocket_timeline.extend_from_slice(data);
}

fn write_headers(output: &mut String, headers: &LogHeaders) {
    if headers.is_empty() {
        output.push_str("<none>\n");
        return;
    }
    for (key, values) in headers {
        if values.is_empty() {
            let _ = writeln!(output, "{key}:");
        } else {
            for value in values {
                let masked = mask_sensitive_header_value(key, value);
                let _ = writeln!(output, "{key}: {}", String::from_utf8_lossy(&masked));
            }
        }
    }
}

fn format_auth_info(info: &UpstreamRequestLog) -> String {
    let mut parts = Vec::new();
    for (key, value) in [
        ("provider", info.provider.trim()),
        ("auth_id", info.auth_id.trim()),
        ("label", info.auth_label.trim()),
    ] {
        if !value.is_empty() {
            parts.push(format!("{key}={value}"));
        }
    }
    let auth_type = info.auth_type.trim().to_ascii_lowercase();
    let auth_value = info.auth_value.trim();
    match auth_type.as_str() {
        "api_key" if !auth_value.is_empty() => parts.push(format!(
            "type=api_key value={}",
            String::from_utf8_lossy(&hide_api_key(auth_value.as_bytes()))
        )),
        "api_key" => parts.push("type=api_key".into()),
        "oauth" => parts.push("type=oauth".into()),
        "" => {}
        _ if auth_value.is_empty() => parts.push(format!("type={auth_type}")),
        _ => parts.push(format!("type={auth_type} value={auth_value}")),
    }
    parts.join(", ")
}

#[must_use]
pub fn summarize_error_body(content_type: &str, body: &[u8]) -> String {
    let lower = trim_ascii(body)
        .iter()
        .map(u8::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let html = content_type.to_ascii_lowercase().contains("text/html")
        || lower.starts_with(b"<!doctype html")
        || lower.starts_with(b"<html");
    if html {
        return extract_html_title(body).unwrap_or_else(|| "[html body omitted]".into());
    }
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(message) = value
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
        {
            if !message.is_empty() {
                return message.to_owned();
            }
        }
    }
    String::from_utf8_lossy(body).into_owned()
}

fn extract_html_title(body: &[u8]) -> Option<String> {
    let lower = body.iter().map(u8::to_ascii_lowercase).collect::<Vec<_>>();
    let start = find_bytes(&lower, b"<title")?;
    let opening = lower[start..].iter().position(|byte| *byte == b'>')? + start + 1;
    let end = find_bytes(&lower[opening..], b"</title>")? + opening;
    let title = String::from_utf8_lossy(&body[opening..end]);
    let decoded = title
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    let normalized = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    (!normalized.is_empty()).then_some(normalized)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn mark_credits_used(context: Option<&ApiLogContext>) {
    if let Some(context) = context {
        context.state.lock().unwrap().credits_used = true;
    }
}

#[must_use]
pub fn credits_used(context: Option<&ApiLogContext>) -> bool {
    context.is_some_and(|context| context.state.lock().unwrap().credits_used)
}

#[must_use]
pub fn request_id(context: Option<&ApiLogContext>) -> String {
    context
        .map(|context| logging::get_request_id(Some(context.request_context())))
        .unwrap_or_default()
        .to_owned()
}

fn timestamp_string(timestamp: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(timestamp)
        .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}
