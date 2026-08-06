// ref: internal/logging/request_logger_streaming.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{
    temp_name, FileRequestLogger, RequestLogRecord, RequestLogStorage, StreamingLogOutcome,
    StreamingLogWriter,
};
use super::request_logger_format::{write_api_section, write_request_info, write_response_section};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

const STREAM_CHUNK_QUEUE_CAPACITY: usize = 100;

pub struct FileStreamingLogWriter {
    log_file_path: PathBuf,
    response_body_path: PathBuf,
    storage: Arc<dyn RequestLogStorage>,
    sender: Option<SyncSender<Vec<u8>>>,
    worker: Option<JoinHandle<io::Result<()>>>,
    dropped_chunks: Arc<AtomicU64>,
    record: RequestLogRecord,
    api_request: Vec<u8>,
    api_response: Vec<u8>,
    api_websocket_timeline: Vec<u8>,
    first_chunk_timestamp: Option<SystemTime>,
    closed: bool,
}

impl FileStreamingLogWriter {
    pub fn new(
        logger: &FileRequestLogger,
        url: &str,
        method: &str,
        headers: &BTreeMap<String, Vec<String>>,
        body: &[u8],
        request_id: &str,
    ) -> io::Result<Self> {
        logger.storage.create_dir_all(logger.logs_dir())?;
        let record = RequestLogRecord {
            url: url.to_owned(),
            method: method.to_owned(),
            request_headers: headers.clone(),
            request_body: body.to_vec(),
            status_code: 0,
            response_headers: BTreeMap::new(),
            response_body: Vec::new(),
            request_id: request_id.to_owned(),
            streaming: true,
        };
        let log_file_path = logger
            .logs_dir()
            .join(logger.generate_filename(&record, false));
        let response_body_path = logger.logs_dir().join(temp_name("response-body"));
        let mut response_body = logger.storage.create_exclusive(&response_body_path)?;
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(STREAM_CHUNK_QUEUE_CAPACITY);
        let worker = thread::Builder::new()
            .name("cliproxy-stream-log".to_owned())
            .spawn(move || {
                for chunk in receiver {
                    response_body.write_all(&chunk)?;
                }
                response_body.flush()
            })?;
        Ok(Self {
            log_file_path,
            response_body_path,
            storage: Arc::clone(&logger.storage),
            sender: Some(sender),
            worker: Some(worker),
            dropped_chunks: Arc::new(AtomicU64::new(0)),
            record,
            api_request: Vec::new(),
            api_response: Vec::new(),
            api_websocket_timeline: Vec::new(),
            first_chunk_timestamp: None,
            closed: false,
        })
    }

    pub fn dropped_chunks(&self) -> u64 {
        self.dropped_chunks.load(Ordering::Relaxed)
    }

    fn finish(&mut self, write_final_log: bool) -> io::Result<PathBuf> {
        self.sender.take();
        let worker_result = self.worker.take().map_or(Ok(()), |worker| {
            worker
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("stream log worker panicked")))
        });
        let final_result = if worker_result.is_ok() && write_final_log {
            (|| {
                self.record.response_body = self.storage.read(&self.response_body_path)?;
                let mut output = self.storage.create_exclusive(&self.log_file_path)?;
                write_request_info(
                    &mut output,
                    &self.record.url,
                    &self.record.method,
                    &self.record.request_headers,
                    &self.record.request_body,
                    "stream",
                    if self.api_websocket_timeline.is_empty() {
                        "http"
                    } else {
                        "websocket"
                    },
                )?;
                write_api_section(
                    &mut output,
                    "=== API WEBSOCKET TIMELINE ===",
                    &self.api_websocket_timeline,
                    None,
                )?;
                write_api_section(&mut output, "=== API REQUEST ===", &self.api_request, None)?;
                write_api_section(
                    &mut output,
                    "=== API RESPONSE ===",
                    &self.api_response,
                    self.first_chunk_timestamp,
                )?;
                write_response_section(
                    &mut output,
                    self.record.status_code,
                    &self.record.response_headers,
                    &self.record.response_body,
                    None,
                    true,
                )?;
                output.flush()
            })()
        } else {
            worker_result
        };
        let _ = self.storage.remove_file(&self.response_body_path);
        self.closed = true;
        final_result.map(|()| self.log_file_path.clone())
    }
}

impl StreamingLogWriter for FileStreamingLogWriter {
    fn write_chunk_async(&self, chunk: &[u8]) {
        if let Some(sender) = &self.sender {
            enqueue_chunk(sender, &self.dropped_chunks, chunk);
        }
    }
    fn write_status(&mut self, status: u16, headers: &BTreeMap<String, Vec<String>>) {
        if status != 0 {
            self.record.status_code = status;
            self.record.response_headers = headers.clone();
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
        self.first_chunk_timestamp = Some(timestamp);
    }
    fn close(mut self: Box<Self>) -> io::Result<StreamingLogOutcome> {
        let path = self.finish(true)?;
        Ok(StreamingLogOutcome {
            path,
            dropped_chunks: self.dropped_chunks(),
        })
    }
}

impl Drop for FileStreamingLogWriter {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.finish(false);
        }
    }
}

pub struct NoOpStreamingLogWriter;
impl StreamingLogWriter for NoOpStreamingLogWriter {
    fn write_chunk_async(&self, _chunk: &[u8]) {}
    fn write_status(&mut self, _status: u16, _headers: &BTreeMap<String, Vec<String>>) {}
    fn close(self: Box<Self>) -> io::Result<StreamingLogOutcome> {
        Ok(StreamingLogOutcome {
            path: PathBuf::new(),
            dropped_chunks: 0,
        })
    }
}

fn enqueue_chunk(sender: &SyncSender<Vec<u8>>, dropped: &AtomicU64, chunk: &[u8]) {
    match sender.try_send(chunk.to_vec()) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Disconnected(_)) => {}
    }
}

#[cfg(test)]
pub(super) fn saturated_queue_drops_without_blocking() -> u64 {
    let (sender, _receiver) = mpsc::sync_channel(1);
    let dropped = AtomicU64::new(0);
    enqueue_chunk(&sender, &dropped, b"accepted");
    enqueue_chunk(&sender, &dropped, b"dropped");
    dropped.load(Ordering::Relaxed)
}
