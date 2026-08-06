// ref: internal/tui/browser.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io;
pub trait BrowserLauncher: Send + Sync {
    fn open(&self, url: &str) -> io::Result<()>;
}
#[derive(Debug, Default)]
pub struct DisabledBrowserLauncher;
impl BrowserLauncher for DisabledBrowserLauncher {
    fn open(&self, _url: &str) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "browser launch authority was not supplied",
        ))
    }
}
pub fn validate_browser_url(url: &str) -> io::Result<()> {
    let parsed =
        url::Url::parse(url).map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if matches!(parsed.scheme(), "http" | "https") {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "only http(s) browser URLs are allowed",
        ))
    }
}
