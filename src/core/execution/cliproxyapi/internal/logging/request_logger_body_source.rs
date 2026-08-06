// ref: internal/logging/request_logger_body_source.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{NativeRequestLogStorage, RequestLogStorage};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Debug)]
struct State {
    paths: Vec<PathBuf>,
    cleaned: bool,
}

pub struct FileBodySource {
    dir: PathBuf,
    storage: Arc<dyn RequestLogStorage>,
    state: Mutex<State>,
}

impl FileBodySource {
    pub fn new_in_dir(base_dir: impl AsRef<Path>, prefix: &str) -> io::Result<Self> {
        Self::with_storage(base_dir, prefix, Arc::new(NativeRequestLogStorage))
    }

    pub fn with_storage(
        base_dir: impl AsRef<Path>,
        prefix: &str,
        storage: Arc<dyn RequestLogStorage>,
    ) -> io::Result<Self> {
        if base_dir.as_ref().as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "base directory is required",
            ));
        }
        storage.create_dir_all(base_dir.as_ref())?;
        let dir = base_dir.as_ref().join(format!(
            "request-log-parts-{}-{}",
            sanitize_temp_prefix(prefix),
            Uuid::new_v4()
        ));
        storage.create_dir_all(&dir)?;
        Ok(Self {
            dir,
            storage,
            state: Mutex::new(State {
                paths: Vec::new(),
                cleaned: false,
            }),
        })
    }

    pub fn create_part(&self, prefix: &str) -> io::Result<Box<dyn Write + Send>> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.cleaned {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "file body source has been cleaned",
            ));
        }
        self.storage.create_dir_all(&self.dir)?;
        let path = self.dir.join(format!(
            "{}-{}.tmp",
            sanitize_temp_prefix(prefix),
            Uuid::new_v4()
        ));
        let file = self.storage.create_exclusive(&path)?;
        state.paths.push(path);
        Ok(file)
    }

    pub fn append_part(&self, data: &[u8]) -> io::Result<()> {
        let data = trim_ascii_whitespace(data);
        if data.is_empty() {
            return Ok(());
        }
        let mut file = self.create_part("part")?;
        file.write_all(data)?;
        file.flush()
    }

    pub fn append_bytes(&self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let path = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.cleaned {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "file body source has been cleaned",
                ));
            }
            self.storage.create_dir_all(&self.dir)?;
            match state.paths.last() {
                Some(path) => path.clone(),
                None => {
                    let path = self.dir.join(format!("part-{}.tmp", Uuid::new_v4()));
                    state.paths.push(path.clone());
                    path
                }
            }
        };
        self.storage.append(&path, data)
    }

    pub fn has_payload(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        !state.cleaned && !state.paths.is_empty()
    }

    pub fn paths(&self) -> Vec<PathBuf> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .paths
            .clone()
    }

    pub fn write_to(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut wrote = false;
        for path in self.paths() {
            let bytes = match self.storage.read(&path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if wrote {
                writer.write_all(b"\n")?;
            }
            writer.write_all(&bytes)?;
            wrote = true;
        }
        Ok(())
    }

    pub fn bytes(&self) -> io::Result<Vec<u8>> {
        let mut output = Vec::new();
        self.write_to(&mut output)?;
        Ok(output)
    }

    pub fn cleanup(&self) -> io::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.cleaned {
            return Ok(());
        }
        state.cleaned = true;
        state.paths.clear();
        match self.storage.remove_dir_all(&self.dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

impl Drop for FileBodySource {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

pub fn cleanup_file_body_sources(sources: &[Option<&FileBodySource>]) {
    for source in sources.iter().flatten() {
        let _ = source.cleanup();
    }
}

fn sanitize_temp_prefix(prefix: &str) -> String {
    let output = prefix
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let output = output.trim_matches(['-', '_']);
    if output.is_empty() {
        "log".to_owned()
    } else {
        output.to_owned()
    }
}

fn trim_ascii_whitespace(mut data: &[u8]) -> &[u8] {
    while data.first().is_some_and(u8::is_ascii_whitespace) {
        data = &data[1..];
    }
    while data.last().is_some_and(u8::is_ascii_whitespace) {
        data = &data[..data.len() - 1];
    }
    data
}
