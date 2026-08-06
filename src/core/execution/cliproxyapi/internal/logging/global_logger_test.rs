// ref: internal/logging/global_logger_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::global_logger::{
    LogEntry, LogFormatter, LogLevel, LogSink, NativeRotationFilesystem, RotatingFileSink,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

#[test]
fn formatter_prints_ordered_and_quoted_fields_without_raw_newlines() {
    let mut entry = LogEntry::new(LogLevel::Info, "media started\n", SystemTime::UNIX_EPOCH);
    entry
        .fields
        .insert("credential".into(), "Voice credential\nsecondary".into());
    entry
        .fields
        .insert("connection".into(), "via socks5 proxy".into());
    entry.fields.insert("version".into(), "2.1.0".into());
    let line = LogFormatter.format(&entry);
    assert!(line.contains("version=2.1.0"));
    assert!(line.contains(r#"credential="Voice credential\nsecondary""#));
    assert_eq!(line.matches('\n').count(), 1);
}

#[test]
fn plugin_paths_require_plugin_identity() {
    let mut generic = LogEntry::new(LogLevel::Warn, "rollback", SystemTime::UNIX_EPOCH);
    generic
        .fields
        .insert("path".into(), "auth/private.json".into());
    assert!(!LogFormatter.format(&generic).contains("path="));
    generic.fields.insert("plugin_id".into(), "sample".into());
    generic
        .fields
        .insert("active_path".into(), "plugins/sample-v1.dll".into());
    generic.source = Some((PathBuf::from("manager.rs"), 42));
    let line = LogFormatter.format(&generic);
    assert!(line.contains("plugin_id=sample"));
    assert!(line.contains("path=auth/private.json"));
    assert!(line.contains("[manager.rs:42]"));
}

#[test]
fn rotating_sink_bounds_numbered_backups() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.log");
    let sink =
        RotatingFileSink::new(path.clone(), 5, 2, Arc::new(NativeRotationFilesystem)).unwrap();
    sink.write(b"12345").unwrap();
    sink.write(b"67890").unwrap();
    sink.write(b"abcde").unwrap();
    sink.write(b"final").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"final");
    assert_eq!(
        std::fs::read(format!("{}.1", path.display())).unwrap(),
        b"abcde"
    );
    assert_eq!(
        std::fs::read(format!("{}.2", path.display())).unwrap(),
        b"67890"
    );
    assert!(!std::path::Path::new(&format!("{}.3", path.display())).exists());
}
