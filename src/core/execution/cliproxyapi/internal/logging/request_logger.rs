// ref: internal/logging/request_logger.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::logging::request_logger_streaming::FileStreamingLogWriter;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use super::request_logger_format::write_record;
use super::request_logger_home::HomeRequestLogSink;

pub const WEBSOCKET_TIMELINE_SOURCE_CONTEXT_KEY: &str = "WEBSOCKET_TIMELINE_SOURCE";
pub const API_REQUEST_SOURCE_CONTEXT_KEY: &str = "API_REQUEST_SOURCE";
pub const DEFERRED_API_REQUEST_CONTEXT_KEY: &str = "DEFERRED_API_REQUEST";
pub const API_RESPONSE_SOURCE_CONTEXT_KEY: &str = "API_RESPONSE_SOURCE";
pub const API_RESPONSE_CAPTURED_CONTEXT_KEY: &str = "API_RESPONSE_CAPTURED";
pub const API_WEBSOCKET_TIMELINE_SOURCE_CONTEXT_KEY: &str = "API_WEBSOCKET_TIMELINE_SOURCE";

static REQUEST_LOG_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestLogRecord {
    pub url: String,
    pub method: String,
    pub request_headers: BTreeMap<String, Vec<String>>,
    pub request_body: Vec<u8>,
    pub status_code: u16,
    pub response_headers: BTreeMap<String, Vec<String>>,
    pub response_body: Vec<u8>,
    pub request_id: String,
    pub streaming: bool,
}

pub trait RequestLogger: Send + Sync {
    fn is_enabled(&self) -> bool;
    fn log_request(&self, record: &RequestLogRecord, force: bool) -> io::Result<Option<PathBuf>>;
    fn log_streaming_request(
        &self,
        _url: &str,
        _method: &str,
        _headers: &BTreeMap<String, Vec<String>>,
        _body: &[u8],
        _request_id: &str,
    ) -> io::Result<Option<Box<dyn StreamingLogWriter>>> {
        Ok(None)
    }
}

pub trait StreamingLogWriter: Send {
    fn write_chunk_async(&self, chunk: &[u8]);
    fn write_status(&mut self, status: u16, headers: &BTreeMap<String, Vec<String>>);
    fn write_api_request(&mut self, _request: &[u8]) -> io::Result<()> {
        Ok(())
    }
    fn write_api_response(&mut self, _response: &[u8]) -> io::Result<()> {
        Ok(())
    }
    fn write_api_websocket_timeline(&mut self, _timeline: &[u8]) -> io::Result<()> {
        Ok(())
    }
    fn set_first_chunk_timestamp(&mut self, _timestamp: SystemTime) {}
    fn close(self: Box<Self>) -> io::Result<StreamingLogOutcome>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingLogOutcome {
    pub path: PathBuf,
    pub dropped_chunks: u64,
}

pub trait RequestLogClock: Send + Sync {
    fn now(&self) -> SystemTime;
}

#[derive(Debug, Default)]
pub struct SystemRequestLogClock;
impl RequestLogClock for SystemRequestLogClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

pub trait RequestLogStorage: Send + Sync {
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    fn create_exclusive(&self, path: &Path) -> io::Result<Box<dyn Write + Send>>;
    fn append(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn remove_dir_all(&self, path: &Path) -> io::Result<()>;
    fn error_log_files(&self, dir: &Path) -> io::Result<Vec<(PathBuf, SystemTime)>>;
}

#[derive(Debug, Default)]
pub struct NativeRequestLogStorage;
impl RequestLogStorage for NativeRequestLogStorage {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)
    }
    fn create_exclusive(&self, path: &Path) -> io::Result<Box<dyn Write + Send>> {
        Ok(Box::new(
            OpenOptions::new().create_new(true).write(true).open(path)?,
        ))
    }
    fn append(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(bytes)?;
        file.flush()
    }
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    }
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }
    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir_all(path)
    }
    fn error_log_files(&self, dir: &Path) -> io::Result<Vec<(PathBuf, SystemTime)>> {
        Ok(fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("error-") && name.ends_with(".log")
            })
            .filter_map(|entry| Some((entry.path(), entry.metadata().ok()?.modified().ok()?)))
            .collect())
    }
}

pub struct FileRequestLogger {
    enabled: AtomicBool,
    logs_dir: PathBuf,
    error_logs_max_files: AtomicUsize,
    pub(super) storage: Arc<dyn RequestLogStorage>,
    pub(super) clock: Arc<dyn RequestLogClock>,
    pub(super) home_sink: Mutex<Option<Arc<dyn HomeRequestLogSink>>>,
}

impl FileRequestLogger {
    pub fn new(
        enabled: bool,
        logs_dir: impl AsRef<Path>,
        config_dir: impl AsRef<Path>,
        error_logs_max_files: usize,
    ) -> Self {
        Self::with_runtime(
            enabled,
            logs_dir,
            config_dir,
            error_logs_max_files,
            Arc::new(NativeRequestLogStorage),
            Arc::new(SystemRequestLogClock),
        )
    }

    pub fn with_runtime(
        enabled: bool,
        logs_dir: impl AsRef<Path>,
        config_dir: impl AsRef<Path>,
        error_logs_max_files: usize,
        storage: Arc<dyn RequestLogStorage>,
        clock: Arc<dyn RequestLogClock>,
    ) -> Self {
        let logs_dir = logs_dir.as_ref();
        let logs_dir = if logs_dir.is_absolute() || config_dir.as_ref().as_os_str().is_empty() {
            logs_dir.to_path_buf()
        } else {
            config_dir.as_ref().join(logs_dir)
        };
        Self {
            enabled: AtomicBool::new(enabled),
            logs_dir,
            error_logs_max_files: AtomicUsize::new(error_logs_max_files),
            storage,
            clock,
            home_sink: Mutex::new(None),
        }
    }

    pub fn logs_dir(&self) -> &Path {
        &self.logs_dir
    }
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }
    pub fn set_error_logs_max_files(&self, max_files: usize) {
        self.error_logs_max_files
            .store(max_files, Ordering::Release);
    }
    pub fn new_file_body_source(
        &self,
        prefix: &str,
    ) -> io::Result<super::request_logger_body_source::FileBodySource> {
        super::request_logger_body_source::FileBodySource::with_storage(
            &self.logs_dir,
            prefix,
            Arc::clone(&self.storage),
        )
    }

    pub(super) fn generate_filename(&self, record: &RequestLogRecord, error_only: bool) -> String {
        let prefix = if error_only { "error" } else { "request" };
        let route = sanitize_filename(record.url.split('?').next().unwrap_or_default());
        let request_id = sanitize_filename(&record.request_id);
        let now = self
            .clock
            .now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sequence = REQUEST_LOG_ID.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{route}-{request_id}-{now}-{sequence}.log")
    }

    pub(super) fn cleanup_old_error_logs(&self) -> io::Result<()> {
        let max = self.error_logs_max_files.load(Ordering::Acquire);
        if max == 0 {
            return Ok(());
        }
        let mut files = self.storage.error_log_files(&self.logs_dir)?;
        files.sort_by_key(|(_, modified)| *modified);
        let remove_count = files.len().saturating_sub(max);
        for (path, _) in files.into_iter().take(remove_count) {
            self.storage.remove_file(&path)?;
        }
        Ok(())
    }
}

impl RequestLogger for FileRequestLogger {
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    fn log_request(&self, record: &RequestLogRecord, force: bool) -> io::Result<Option<PathBuf>> {
        let enabled = self.is_enabled();
        if !enabled && !force {
            return Ok(None);
        }
        if enabled {
            if let Some(sink) = self
                .home_sink
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
            {
                if sink.heartbeat_ok() {
                    let mut content = Vec::new();
                    write_record(&mut content, record)?;
                    sink.push_request_log(
                        &super::request_logger_home::HomeRequestLogPayload::new(record, content),
                    )?;
                    return Ok(None);
                }
            }
        }
        self.storage.create_dir_all(&self.logs_dir)?;
        let error_only = force && !enabled;
        let path = self
            .logs_dir
            .join(self.generate_filename(record, error_only));
        let mut file = self.storage.create_exclusive(&path)?;
        write_record(&mut file, record)?;
        file.flush()?;
        if error_only {
            self.cleanup_old_error_logs()?;
        }
        Ok(Some(path))
    }

    fn log_streaming_request(
        &self,
        url: &str,
        method: &str,
        headers: &BTreeMap<String, Vec<String>>,
        body: &[u8],
        request_id: &str,
    ) -> io::Result<Option<Box<dyn StreamingLogWriter>>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        if let Some(sink) = self
            .home_sink
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
        {
            if sink.heartbeat_ok() {
                return Ok(Some(Box::new(
                    super::request_logger_home::HomeStreamingLogWriter::new(
                        url,
                        method,
                        headers,
                        body,
                        request_id,
                        sink,
                        Arc::clone(&self.clock),
                    ),
                )));
            }
            return Ok(None);
        }
        FileStreamingLogWriter::new(self, url, method, headers, body, request_id)
            .map(|writer| Some(Box::new(writer) as Box<dyn StreamingLogWriter>))
    }
}

pub(super) fn temp_name(prefix: &str) -> String {
    format!("{}-{}.tmp", sanitize_filename(prefix), Uuid::new_v4())
}

pub(super) fn sanitize_filename(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "root".to_owned()
    } else {
        sanitized.chars().take(80).collect()
    }
}
