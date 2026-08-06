// ref: internal/logging/log_dir_cleaner.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogFileMetadata {
    pub path: PathBuf,
    pub size: u64,
    pub modified: SystemTime,
    pub regular: bool,
}

/// Filesystem authority used by rotation and request logging.
pub trait LogFilesystem: Send + Sync {
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    fn list(&self, path: &Path) -> io::Result<Vec<LogFileMetadata>>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
}

#[derive(Debug, Default)]
pub struct NativeLogFilesystem;

impl LogFilesystem for NativeLogFilesystem {
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)
    }

    fn list(&self, path: &Path) -> io::Result<Vec<LogFileMetadata>> {
        let entries = match fs::read_dir(path) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let metadata = entry.metadata().ok()?;
                Some(LogFileMetadata {
                    path: entry.path(),
                    size: metadata.len(),
                    modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    regular: metadata.is_file(),
                })
            })
            .collect())
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }
}

pub fn is_log_file_name(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    !lower.is_empty() && (lower.ends_with(".log") || lower.ends_with(".log.gz"))
}

pub fn enforce_log_dir_size_limit(
    filesystem: &dyn LogFilesystem,
    log_dir: &Path,
    max_bytes: u64,
    protected_path: Option<&Path>,
) -> io::Result<usize> {
    if max_bytes == 0 || log_dir.as_os_str().is_empty() {
        return Ok(0);
    }
    let protected = protected_path.map(normalized_path);
    let mut files = filesystem
        .list(log_dir)?
        .into_iter()
        .filter(|file| file.regular)
        .filter(|file| {
            file.path
                .file_name()
                .is_some_and(|name| is_log_file_name(&name.to_string_lossy()))
        })
        .collect::<Vec<_>>();
    let mut total = files.iter().map(|file| file.size).sum::<u64>();
    if total <= max_bytes {
        return Ok(0);
    }
    files.sort_by_key(|file| file.modified);
    let mut deleted = 0;
    for file in files {
        if total <= max_bytes {
            break;
        }
        if protected
            .as_ref()
            .is_some_and(|path| *path == normalized_path(&file.path))
        {
            continue;
        }
        if filesystem.remove_file(&file.path).is_ok() {
            total = total.saturating_sub(file.size);
            deleted += 1;
        }
    }
    Ok(deleted)
}

fn normalized_path(path: &Path) -> PathBuf {
    path.components().collect()
}

/// Owned cleaner lifecycle. Dropping or stopping it terminates the worker.
pub struct LogDirCleaner {
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl LogDirCleaner {
    pub fn start(
        filesystem: Arc<dyn LogFilesystem>,
        log_dir: PathBuf,
        max_bytes: u64,
        protected_path: Option<PathBuf>,
        interval: Duration,
    ) -> Option<Self> {
        if max_bytes == 0 || log_dir.as_os_str().is_empty() {
            return None;
        }
        let (stop, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("cliproxy-log-cleaner".to_owned())
            .spawn(move || loop {
                let _ = enforce_log_dir_size_limit(
                    filesystem.as_ref(),
                    &log_dir,
                    max_bytes,
                    protected_path.as_deref(),
                );
                match receiver.recv_timeout(interval) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            })
            .ok()?;
        Some(Self {
            stop: Some(stop),
            worker: Some(worker),
        })
    }

    pub fn stop(&mut self) {
        self.stop.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for LogDirCleaner {
    fn drop(&mut self) {
        self.stop();
    }
}
