// ref: internal/logging/request_logger_home.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{
    FileRequestLogger, RequestLogClock, RequestLogRecord, StreamingLogOutcome, StreamingLogWriter,
};
use super::request_logger_format::{
    infer_upstream_transport, write_api_section, write_request_info_at, write_response_section,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

pub trait HomeRequestLogSink: Send + Sync {
    fn heartbeat_ok(&self) -> bool;
    fn push_request_log(&self, payload: &HomeRequestLogPayload) -> io::Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomeRequestLogPayload {
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub headers: BTreeMap<String, Vec<String>>,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub request_id: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default, with = "payload_text")]
    pub request_log: Vec<u8>,
}

impl HomeRequestLogPayload {
    pub fn new(record: &RequestLogRecord, request_log: Vec<u8>) -> Self {
        Self {
            headers: clone_headers(&record.request_headers),
            request_id: record.request_id.trim().to_owned(),
            request_log,
        }
    }
}

mod payload_text {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&String::from_utf8_lossy(value))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        String::deserialize(deserializer).map(String::into_bytes)
    }
}

pub fn clone_headers(headers: &BTreeMap<String, Vec<String>>) -> BTreeMap<String, Vec<String>> {
    headers
        .iter()
        .filter(|(name, _)| !name.trim().is_empty())
        .map(|(name, values)| (name.clone(), values.clone()))
        .collect()
}

impl FileRequestLogger {
    pub fn bind_home_sink(&self, sink: Arc<dyn HomeRequestLogSink>) {
        *self
            .home_sink
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(sink);
    }

    pub fn unbind_home_sink(&self) {
        self.home_sink
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
    }
}

pub struct HomeStreamingLogWriter {
    url: String,
    method: String,
    timestamp: SystemTime,
    request_headers: BTreeMap<String, Vec<String>>,
    request_body: Vec<u8>,
    request_id: String,
    sink: Arc<dyn HomeRequestLogSink>,
    sender: Option<mpsc::SyncSender<Vec<u8>>>,
    worker: Option<JoinHandle<()>>,
    response_body: Arc<Mutex<Vec<u8>>>,
    response_status: u16,
    response_headers: BTreeMap<String, Vec<String>>,
    api_request: Vec<u8>,
    api_response: Vec<u8>,
    api_websocket_timeline: Vec<u8>,
    api_response_timestamp: Option<SystemTime>,
    dropped: AtomicU64,
}

impl HomeStreamingLogWriter {
    pub fn new(
        url: &str,
        method: &str,
        headers: &BTreeMap<String, Vec<String>>,
        body: &[u8],
        request_id: &str,
        sink: Arc<dyn HomeRequestLogSink>,
        clock: Arc<dyn RequestLogClock>,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(100);
        let response_body = Arc::new(Mutex::new(Vec::new()));
        let worker_body = Arc::clone(&response_body);
        let worker = thread::Builder::new()
            .name("cliproxy-home-request-log".to_owned())
            .spawn(move || {
                while let Ok(chunk) = receiver.recv() {
                    worker_body
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .extend_from_slice(&chunk);
                }
            })
            .expect("home request log worker must start");
        Self {
            url: url.to_owned(),
            method: method.to_owned(),
            timestamp: clock.now(),
            request_headers: clone_headers(headers),
            request_body: body.to_vec(),
            request_id: request_id.trim().to_owned(),
            sink,
            sender: Some(sender),
            worker: Some(worker),
            response_body,
            response_status: 0,
            response_headers: BTreeMap::new(),
            api_request: Vec::new(),
            api_response: Vec::new(),
            api_websocket_timeline: Vec::new(),
            api_response_timestamp: None,
            dropped: AtomicU64::new(0),
        }
    }

    fn finish_worker(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl StreamingLogWriter for HomeStreamingLogWriter {
    fn write_chunk_async(&self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        if self
            .sender
            .as_ref()
            .is_some_and(|sender| sender.try_send(chunk.to_vec()).is_err())
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn write_status(&mut self, status: u16, headers: &BTreeMap<String, Vec<String>>) {
        if status != 0 {
            self.response_status = status;
            self.response_headers = clone_headers(headers);
        }
    }
    fn write_api_request(&mut self, request: &[u8]) -> io::Result<()> {
        self.api_request = request.to_vec();
        Ok(())
    }
    fn write_api_response(&mut self, response: &[u8]) -> io::Result<()> {
        self.api_response = response.to_vec();
        Ok(())
    }
    fn write_api_websocket_timeline(&mut self, timeline: &[u8]) -> io::Result<()> {
        self.api_websocket_timeline = timeline.to_vec();
        Ok(())
    }
    fn set_first_chunk_timestamp(&mut self, timestamp: SystemTime) {
        self.api_response_timestamp = Some(timestamp);
    }
    fn close(mut self: Box<Self>) -> io::Result<StreamingLogOutcome> {
        self.finish_worker();
        if !self.sink.heartbeat_ok() {
            return Ok(StreamingLogOutcome {
                path: PathBuf::new(),
                dropped_chunks: self.dropped.load(Ordering::Relaxed),
            });
        }
        let transport = infer_upstream_transport(
            &self.api_request,
            &self.api_response,
            !self.api_websocket_timeline.is_empty(),
        );
        let mut content = Vec::new();
        write_request_info_at(
            &mut content,
            &self.url,
            &self.method,
            &self.request_headers,
            &self.request_body,
            "http",
            transport,
            Some(self.timestamp),
        )?;
        write_api_section(
            &mut content,
            "=== API WEBSOCKET TIMELINE ===",
            &self.api_websocket_timeline,
            None,
        )?;
        write_api_section(&mut content, "=== API REQUEST ===", &self.api_request, None)?;
        write_api_section(
            &mut content,
            "=== API RESPONSE ===",
            &self.api_response,
            self.api_response_timestamp,
        )?;
        let body = self
            .response_body
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        write_response_section(
            &mut content,
            self.response_status,
            &self.response_headers,
            &body,
            None,
            true,
        )?;
        self.sink.push_request_log(&HomeRequestLogPayload {
            headers: clone_headers(&self.request_headers),
            request_id: self.request_id.clone(),
            request_log: content,
        })?;
        Ok(StreamingLogOutcome {
            path: PathBuf::new(),
            dropped_chunks: self.dropped.load(Ordering::Relaxed),
        })
    }
}

impl Drop for HomeStreamingLogWriter {
    fn drop(&mut self) {
        self.finish_worker();
    }
}
