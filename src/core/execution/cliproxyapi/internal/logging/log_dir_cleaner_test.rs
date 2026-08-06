// ref: internal/logging/log_dir_cleaner_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::log_dir_cleaner::{enforce_log_dir_size_limit, is_log_file_name, NativeLogFilesystem};
use std::fs;
use std::io::Write;
use std::time::{Duration, SystemTime};

fn write_log(path: &std::path::Path, size: usize, modified: SystemTime) {
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&vec![0; size]).unwrap();
    file.set_modified(modified).unwrap();
}

#[test]
fn deletes_oldest_and_skips_protected_file() {
    let dir = tempfile::tempdir().unwrap();
    write_log(
        &dir.path().join("old.log"),
        60,
        SystemTime::UNIX_EPOCH + Duration::from_secs(1),
    );
    write_log(
        &dir.path().join("mid.log"),
        60,
        SystemTime::UNIX_EPOCH + Duration::from_secs(2),
    );
    let protected = dir.path().join("main.log");
    write_log(
        &protected,
        60,
        SystemTime::UNIX_EPOCH + Duration::from_secs(3),
    );
    assert_eq!(
        enforce_log_dir_size_limit(&NativeLogFilesystem, dir.path(), 120, Some(&protected))
            .unwrap(),
        1
    );
    assert!(!dir.path().join("old.log").exists());
    assert!(dir.path().join("mid.log").exists());
    assert!(protected.exists());
}

#[test]
fn protected_file_can_leave_directory_above_limit() {
    let dir = tempfile::tempdir().unwrap();
    let protected = dir.path().join("main.log");
    write_log(
        &protected,
        200,
        SystemTime::UNIX_EPOCH + Duration::from_secs(1),
    );
    write_log(
        &dir.path().join("other.log"),
        50,
        SystemTime::UNIX_EPOCH + Duration::from_secs(2),
    );
    assert_eq!(
        enforce_log_dir_size_limit(&NativeLogFilesystem, dir.path(), 100, Some(&protected))
            .unwrap(),
        1
    );
    assert!(protected.exists());
    assert!(!dir.path().join("other.log").exists());
    assert!(is_log_file_name("archive.LOG.GZ"));
    assert!(!is_log_file_name("response.tmp"));
}
