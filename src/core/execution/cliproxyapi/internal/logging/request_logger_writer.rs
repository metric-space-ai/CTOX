// ref: internal/logging/request_logger_writer.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{FileRequestLogger, RequestLogRecord, RequestLogger};
use super::request_logger_body_source::{cleanup_file_body_sources, FileBodySource};
use super::request_logger_format::{
    decompress_response, infer_downstream_transport, infer_upstream_transport, write_api_section,
    write_request_info_at, write_response_section,
};
use super::request_logger_home::HomeRequestLogPayload;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::SystemTime;

pub struct DetailedRequestLog<'a> {
    pub url: &'a str,
    pub method: &'a str,
    pub request_headers: &'a BTreeMap<String, Vec<String>>,
    pub request_body: &'a [u8],
    pub status_code: u16,
    pub response_headers: &'a BTreeMap<String, Vec<String>>,
    pub response: &'a [u8],
    pub websocket_timeline: &'a [u8],
    pub websocket_timeline_source: Option<&'a FileBodySource>,
    pub api_request: &'a [u8],
    pub api_request_source: Option<&'a FileBodySource>,
    pub api_response: &'a [u8],
    pub api_response_source: Option<&'a FileBodySource>,
    pub api_websocket_timeline: &'a [u8],
    pub api_websocket_timeline_source: Option<&'a FileBodySource>,
    pub api_response_errors: &'a [String],
    pub force: bool,
    pub request_id: &'a str,
    pub request_timestamp: SystemTime,
    pub api_response_timestamp: Option<SystemTime>,
}

impl FileRequestLogger {
    pub fn log_detailed(&self, request: &DetailedRequestLog<'_>) -> io::Result<Option<PathBuf>> {
        let result = self.log_detailed_inner(request);
        cleanup_file_body_sources(&[
            request.websocket_timeline_source,
            request.api_request_source,
            request.api_response_source,
            request.api_websocket_timeline_source,
        ]);
        result
    }

    fn log_detailed_inner(&self, request: &DetailedRequestLog<'_>) -> io::Result<Option<PathBuf>> {
        if !self.is_enabled() && !request.force {
            return Ok(None);
        }
        let mut content = Vec::new();
        let websocket_source = source_bytes(request.websocket_timeline_source)?;
        let api_request_source = source_bytes(request.api_request_source)?;
        let api_response_source = source_bytes(request.api_response_source)?;
        let api_websocket_source = source_bytes(request.api_websocket_timeline_source)?;
        let websocket_timeline = choose_payload(request.websocket_timeline, &websocket_source);
        let api_request = choose_payload(request.api_request, &api_request_source);
        let api_response = choose_payload(request.api_response, &api_response_source);
        let api_websocket = choose_payload(request.api_websocket_timeline, &api_websocket_source);
        let downstream =
            infer_downstream_transport(request.request_headers, !websocket_timeline.is_empty());
        let upstream =
            infer_upstream_transport(api_request, api_response, !api_websocket.is_empty());
        write_request_info_at(
            &mut content,
            request.url,
            request.method,
            request.request_headers,
            request.request_body,
            downstream,
            upstream,
            Some(request.request_timestamp),
        )?;
        write_api_section(
            &mut content,
            "=== WEBSOCKET TIMELINE ===",
            websocket_timeline,
            None,
        )?;
        write_api_section(
            &mut content,
            "=== API WEBSOCKET TIMELINE ===",
            api_websocket,
            None,
        )?;
        write_api_section(
            &mut content,
            "=== API REQUEST ===",
            api_request,
            Some(request.request_timestamp),
        )?;
        write_api_section(
            &mut content,
            "=== API RESPONSE ===",
            api_response,
            request.api_response_timestamp,
        )?;
        if !request.api_response_errors.is_empty() {
            writeln!(&mut content, "=== API RESPONSE ERRORS ===")?;
            for error in request.api_response_errors {
                writeln!(&mut content, "{error}")?;
            }
        }
        let (response, decompression_error) =
            match decompress_response(request.response_headers, request.response) {
                Ok(response) => (response, None),
                Err(error) => (request.response.to_vec(), Some(error.to_string())),
            };
        write_response_section(
            &mut content,
            request.status_code,
            request.response_headers,
            &response,
            decompression_error.as_deref(),
            false,
        )?;

        if self.is_enabled() {
            if let Some(sink) = self
                .home_sink
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
            {
                if sink.heartbeat_ok() {
                    sink.push_request_log(&HomeRequestLogPayload {
                        headers: request.request_headers.clone(),
                        request_id: request.request_id.trim().to_owned(),
                        request_log: content,
                    })?;
                    return Ok(None);
                }
            }
        }
        self.storage.create_dir_all(self.logs_dir())?;
        let record = RequestLogRecord {
            url: request.url.to_owned(),
            method: request.method.to_owned(),
            request_headers: request.request_headers.clone(),
            request_body: request.request_body.to_vec(),
            status_code: request.status_code,
            response_headers: request.response_headers.clone(),
            response_body: Vec::new(),
            request_id: request.request_id.to_owned(),
            streaming: false,
        };
        let error_only = request.force && !self.is_enabled();
        let path = self
            .logs_dir()
            .join(self.generate_filename(&record, error_only));
        let mut file = self.storage.create_exclusive(&path)?;
        file.write_all(&content)?;
        file.flush()?;
        if error_only {
            self.cleanup_old_error_logs()?;
        }
        Ok(Some(path))
    }
}

fn source_bytes(source: Option<&FileBodySource>) -> io::Result<Vec<u8>> {
    source
        .map(FileBodySource::bytes)
        .transpose()
        .map(Option::unwrap_or_default)
}

fn choose_payload<'a>(inline: &'a [u8], source: &'a [u8]) -> &'a [u8] {
    if source.is_empty() {
        inline
    } else {
        source
    }
}
