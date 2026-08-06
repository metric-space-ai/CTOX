// ref: internal/browser @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: supplemental
// License: AGPL-3.0-only

#[path = "browser.rs"]
mod implementation;

pub use implementation::{
    get_platform_info, is_available, open_url, BrowserAuthority, BrowserError, PlatformInfo,
};
