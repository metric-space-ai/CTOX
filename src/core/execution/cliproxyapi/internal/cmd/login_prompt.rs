// ref: internal/cmd/login_prompt.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::{self, BufRead, Write};
use std::sync::Mutex;
pub trait Prompt: Send + Sync {
    fn ask(&self, message: &str) -> io::Result<String>;
}
pub struct IoPrompt<R: BufRead + Send, W: Write + Send> {
    reader: Mutex<R>,
    writer: Mutex<W>,
}
impl<R: BufRead + Send, W: Write + Send> IoPrompt<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader: Mutex::new(reader),
            writer: Mutex::new(writer),
        }
    }
}
impl<R: BufRead + Send, W: Write + Send> Prompt for IoPrompt<R, W> {
    fn ask(&self, message: &str) -> io::Result<String> {
        let mut writer = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        writer.write_all(message.as_bytes())?;
        writer.flush()?;
        drop(writer);
        let mut line = String::new();
        self.reader
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .read_line(&mut line)?;
        Ok(line.trim().to_owned())
    }
}
#[derive(Debug, Default)]
pub struct RejectingPrompt;
impl Prompt for RejectingPrompt {
    fn ask(&self, _message: &str) -> io::Result<String> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "interactive prompt authority was not supplied",
        ))
    }
}
