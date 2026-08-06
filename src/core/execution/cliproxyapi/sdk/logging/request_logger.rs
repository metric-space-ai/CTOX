// ref: sdk/logging/request_logger.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::path::Path;

pub use crate::internal::logging::request_logger::{
    FileRequestLogger, RequestLogger, StreamingLogWriter,
};

const DEFAULT_ERROR_LOGS_MAX_FILES: usize = 10;

/// Creates a file request logger with the upstream default retention of ten
/// forced error logs.
pub fn new_file_request_logger(
    enabled: bool,
    logs_dir: impl AsRef<Path>,
    config_dir: impl AsRef<Path>,
) -> FileRequestLogger {
    FileRequestLogger::new(enabled, logs_dir, config_dir, DEFAULT_ERROR_LOGS_MAX_FILES)
}

/// Creates a file request logger with an explicit forced-error-log retention
/// limit. A limit of zero disables cleanup, matching the upstream contract.
pub fn new_file_request_logger_with_options(
    enabled: bool,
    logs_dir: impl AsRef<Path>,
    config_dir: impl AsRef<Path>,
    error_logs_max_files: usize,
) -> FileRequestLogger {
    FileRequestLogger::new(enabled, logs_dir, config_dir, error_logs_max_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_resolve_relative_and_absolute_log_directories() {
        let relative = new_file_request_logger(false, "logs", "/tmp/config");
        assert_eq!(relative.logs_dir(), Path::new("/tmp/config/logs"));
        assert!(!relative.is_enabled());

        let absolute = new_file_request_logger_with_options(true, "/var/log/ctox", "ignored", 3);
        assert_eq!(absolute.logs_dir(), Path::new("/var/log/ctox"));
        assert!(absolute.is_enabled());
    }
}
