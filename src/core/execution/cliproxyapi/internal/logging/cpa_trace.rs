// ref: internal/logging/cpa_trace.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use chrono::{DateTime, NaiveDateTime, Utc};

use super::requestid::{get_handler_request_id, get_request_id};
use super::requestmeta::{read_unpoisoned, write_unpoisoned, RequestContext, ResponseHeaders};

pub const CPA_TRACE_ID_HEADER: &str = "X-CPA-TRACE-ID";
pub const HANDLER_CPA_TRACE_STATE_KEY: &str = "__cpa_trace_state__";

pub type CpaTraceIdCallback = Arc<dyn Fn(&str) + Send + Sync + 'static>;

/// Builds a trace ID from a timezone-free wall-clock value, auth index, and
/// request ID. `None` is the Rust equivalent of Go's zero `time.Time`.
#[must_use]
pub fn format_cpa_trace_id(
    selected_at: Option<NaiveDateTime>,
    auth_index: &str,
    request_id: &str,
) -> String {
    let auth_index = auth_index.trim();
    let request_id = request_id.trim();
    let Some(selected_at) = selected_at else {
        return String::new();
    };
    if auth_index.is_empty() || request_id.is_empty() {
        return String::new();
    }
    format!(
        "{}-{auth_index}-{request_id}",
        selected_at.format("%Y%m%d%H%M%S")
    )
}

/// Creates the request-local callback used by auth selection. The returned
/// closure owns only the trace holder and request ID, so it remains safe after
/// the surrounding handler context is dropped.
pub fn handler_cpa_trace_id_callback(
    context: Option<&mut RequestContext>,
) -> Option<CpaTraceIdCallback> {
    let context = context?;
    let request_id = match get_handler_request_id(Some(context)).trim() {
        "" => get_request_id(Some(context)).trim(),
        handler_request_id => handler_request_id,
    };
    if request_id.is_empty() {
        return None;
    }
    let request_id = request_id.to_owned();
    let state = ensure_trace_state(context);
    Some(Arc::new(move |auth_index| {
        let now = DateTime::<Utc>::from(SystemTime::now()).naive_utc();
        let trace_id = format_cpa_trace_id(Some(now), auth_index, &request_id);
        if !trace_id.is_empty() {
            *write_unpoisoned(&state) = trace_id;
        }
    }))
}

pub fn set_handler_cpa_trace_id(context: Option<&mut RequestContext>, auth_index: &str) {
    if let Some(callback) = handler_cpa_trace_id_callback(context) {
        callback(auth_index);
    }
}

#[must_use]
pub fn get_handler_cpa_trace_id(context: Option<&RequestContext>) -> String {
    context
        .and_then(|context| context.cpa_trace_id.as_ref())
        .map(|state| read_unpoisoned(state).clone())
        .unwrap_or_default()
}

/// Framework-neutral response-writer wrapper corresponding to the upstream
/// Gin middleware. Every committing operation injects the current trace once,
/// immediately before the response becomes immutable.
pub struct CpaTraceResponseWriter {
    state: Arc<RwLock<String>>,
    headers: ResponseHeaders,
    status_code: Option<u16>,
    body: Vec<u8>,
    written: bool,
    flushed: bool,
}

impl CpaTraceResponseWriter {
    #[must_use]
    pub fn new(context: &mut RequestContext) -> Self {
        Self {
            state: ensure_trace_state(context),
            headers: ResponseHeaders::new(),
            status_code: None,
            body: Vec::new(),
            written: false,
            flushed: false,
        }
    }

    pub fn write_header(&mut self, status_code: u16) {
        if self.written {
            return;
        }
        self.apply_trace_header();
        self.status_code = Some(status_code);
        self.written = true;
    }

    pub fn write_header_now(&mut self) {
        if !self.written {
            self.write_header(200);
        }
    }

    pub fn write(&mut self, data: &[u8]) -> usize {
        self.write_header_now();
        self.body.extend_from_slice(data);
        data.len()
    }

    pub fn write_string(&mut self, data: &str) -> usize {
        self.write(data.as_bytes())
    }

    pub fn flush(&mut self) {
        self.write_header_now();
        self.flushed = true;
    }

    #[must_use]
    pub fn headers(&self) -> &ResponseHeaders {
        &self.headers
    }

    #[must_use]
    pub fn status_code(&self) -> Option<u16> {
        self.status_code
    }

    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    #[must_use]
    pub fn written(&self) -> bool {
        self.written
    }

    #[must_use]
    pub fn flushed(&self) -> bool {
        self.flushed
    }

    fn apply_trace_header(&mut self) {
        if self.written {
            return;
        }
        let trace_id = read_unpoisoned(&self.state).clone();
        if trace_id.is_empty() {
            return;
        }
        self.headers
            .retain(|name, _| !name.eq_ignore_ascii_case(CPA_TRACE_ID_HEADER));
        self.headers
            .insert(CPA_TRACE_ID_HEADER.to_owned(), vec![trace_id]);
    }
}

fn ensure_trace_state(context: &mut RequestContext) -> Arc<RwLock<String>> {
    context
        .cpa_trace_id
        .get_or_insert_with(|| Arc::new(RwLock::new(String::new())))
        .clone()
}
