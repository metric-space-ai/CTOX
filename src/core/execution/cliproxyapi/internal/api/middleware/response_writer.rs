// ref: internal/api/middleware/response_writer.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::logging::request_logger::{
    RequestLogRecord, RequestLogger, StreamingLogWriter,
};
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestInfo {
    pub url: String,
    pub method: String,
    pub headers: BTreeMap<String, Vec<String>>,
    pub body: Vec<u8>,
    pub request_id: String,
}

pub struct ResponseWriterWrapper {
    logger: Arc<dyn RequestLogger>,
    request_info: RequestInfo,
    body: Vec<u8>,
    status_code: Option<u16>,
    headers: BTreeMap<String, Vec<String>>,
    is_streaming: bool,
    stream_writer: Option<Box<dyn StreamingLogWriter>>,
    log_on_error_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseLoggingOutcome {
    pub log_path: Option<PathBuf>,
    pub dropped_stream_chunks: u64,
    pub forced_error_log: bool,
    pub streaming: bool,
}

impl ResponseWriterWrapper {
    pub fn new(logger: Arc<dyn RequestLogger>, request_info: RequestInfo) -> Self {
        Self {
            logger,
            request_info,
            body: Vec::new(),
            status_code: None,
            headers: BTreeMap::new(),
            is_streaming: false,
            stream_writer: None,
            log_on_error_only: false,
        }
    }

    pub fn set_log_on_error_only(&mut self, enabled: bool) {
        self.log_on_error_only = enabled;
    }

    pub fn write_header(&mut self, status_code: u16, headers: BTreeMap<String, Vec<String>>) {
        self.is_streaming = detect_streaming(
            header_value(&headers, "content-type"),
            &self.request_info.body,
        );
        self.status_code = Some(status_code);
        self.headers = headers;
        if self.is_streaming && self.logger.is_enabled() {
            self.stream_writer = self
                .logger
                .log_streaming_request(
                    &self.request_info.url,
                    &self.request_info.method,
                    &self.request_info.headers,
                    &self.request_info.body,
                    &self.request_info.request_id,
                )
                .ok()
                .flatten();
            if let Some(writer) = &mut self.stream_writer {
                writer.write_status(status_code, &self.headers);
            }
        }
    }

    pub fn write(&mut self, data: &[u8]) {
        if let Some(writer) = &self.stream_writer {
            writer.write_chunk_async(data);
            return;
        }
        if self.should_buffer_response_body() {
            self.body.extend_from_slice(data);
        }
    }

    pub fn finalize(self) -> io::Result<Option<PathBuf>> {
        self.finalize_with_outcome().map(|outcome| outcome.log_path)
    }

    pub fn finalize_with_outcome(mut self) -> io::Result<ResponseLoggingOutcome> {
        if let Some(writer) = self.stream_writer.take() {
            let outcome = writer.close()?;
            return Ok(ResponseLoggingOutcome {
                log_path: Some(outcome.path),
                dropped_stream_chunks: outcome.dropped_chunks,
                forced_error_log: false,
                streaming: true,
            });
        }
        let status_code = self.status_code.unwrap_or(200);
        let force = self.log_on_error_only && status_code >= 400 && !self.logger.is_enabled();
        if !self.logger.is_enabled() && !force {
            return Ok(ResponseLoggingOutcome {
                log_path: None,
                dropped_stream_chunks: 0,
                forced_error_log: false,
                streaming: self.is_streaming,
            });
        }
        let record = RequestLogRecord {
            url: self.request_info.url,
            method: self.request_info.method,
            request_headers: self.request_info.headers,
            request_body: self.request_info.body,
            status_code,
            response_headers: self.headers,
            response_body: self.body,
            request_id: self.request_info.request_id,
            streaming: self.is_streaming,
        };
        let log_path = self.logger.log_request(&record, force)?;
        Ok(ResponseLoggingOutcome {
            log_path,
            dropped_stream_chunks: 0,
            forced_error_log: force,
            streaming: self.is_streaming,
        })
    }

    fn should_buffer_response_body(&self) -> bool {
        if self.logger.is_enabled() {
            return true;
        }
        self.log_on_error_only && self.status_code.unwrap_or(200) >= 400
    }
}

pub fn detect_streaming(content_type: Option<&str>, request_body: &[u8]) -> bool {
    if content_type.is_some_and(|value| value.contains("text/event-stream")) {
        return true;
    }
    if content_type.is_some_and(|value| !value.trim().is_empty()) {
        return false;
    }
    request_body
        .windows(br#""stream": true"#.len())
        .any(|window| window == br#""stream": true"#)
        || request_body
            .windows(br#""stream":true"#.len())
            .any(|window| window == br#""stream":true"#)
}

fn header_value<'a>(headers: &'a BTreeMap<String, Vec<String>>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.first())
        .map(String::as_str)
}
