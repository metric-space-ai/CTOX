// ref: internal/api/handlers/management/logs_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{
    ManagementLogAttachment, ManagementLogError, ManagementLogPage, ManagementLogQuery,
    ManagementLogStore, ManagementLogs, ManagementRequestErrorLog,
};

#[derive(Default)]
struct Store {
    reads: Mutex<Vec<ManagementLogQuery>>,
    cleared: Mutex<usize>,
}

impl ManagementLogStore for Store {
    fn read(&self, query: &ManagementLogQuery) -> Result<ManagementLogPage, ManagementLogError> {
        self.reads.lock().unwrap().push(query.clone());
        Ok(ManagementLogPage {
            lines: vec!["line".to_owned()],
            line_count: 1,
            latest_timestamp: 42,
            next_cursor: "opaque-cursor".to_owned(),
            cursor_reset: false,
        })
    }

    fn clear(&self) -> Result<usize, ManagementLogError> {
        *self.cleared.lock().unwrap() += 1;
        Ok(3)
    }

    fn list_request_errors(&self) -> Result<Vec<ManagementRequestErrorLog>, ManagementLogError> {
        Ok(vec![
            ManagementRequestErrorLog {
                name: "error-old.log".to_owned(),
                size: 1,
                modified: 1,
            },
            ManagementRequestErrorLog {
                name: "error-new.log".to_owned(),
                size: 2,
                modified: 2,
            },
        ])
    }

    fn request_log_by_id(
        &self,
        request_id: &str,
    ) -> Result<ManagementLogAttachment, ManagementLogError> {
        Ok(attachment(request_id))
    }

    fn request_error_log(&self, name: &str) -> Result<ManagementLogAttachment, ManagementLogError> {
        Ok(attachment(name))
    }
}

fn attachment(name: &str) -> ManagementLogAttachment {
    ManagementLogAttachment {
        name: name.to_owned(),
        content_type: "text/plain".to_owned(),
        bytes: b"private log contents".to_vec(),
    }
}

#[test]
fn limits_and_opaque_cursors_are_validated_before_store_access() {
    let store = Arc::new(Store::default());
    let logs = ManagementLogs::new(store.clone());
    assert_eq!(
        logs.get_logs(ManagementLogQuery {
            limit: Some(0),
            ..ManagementLogQuery::default()
        }),
        Err(ManagementLogError::InvalidLimit)
    );
    assert_eq!(
        logs.get_logs(ManagementLogQuery {
            cursor: Some(" ".to_owned()),
            ..ManagementLogQuery::default()
        }),
        Err(ManagementLogError::InvalidCursor)
    );
    assert!(store.reads.lock().unwrap().is_empty());
}

#[test]
fn default_limit_and_cursor_are_delegated_without_path_disclosure() {
    let store = Arc::new(Store::default());
    let logs = ManagementLogs::new(store.clone());
    let page = logs
        .get_logs(ManagementLogQuery {
            cursor: Some("cursor-v1".to_owned()),
            ..ManagementLogQuery::default()
        })
        .unwrap();
    assert_eq!(page.next_cursor, "opaque-cursor");
    assert_eq!(store.reads.lock().unwrap()[0].limit, Some(200));
}

#[test]
fn request_log_lookups_reject_traversal_before_store_access() {
    let store = Arc::new(Store::default());
    let logs = ManagementLogs::new(store);
    for id in ["", "../secret", "nested/request", "request.log"] {
        assert_eq!(
            logs.request_log_by_id(id),
            Err(ManagementLogError::InvalidRequestId)
        );
    }
    for name in ["error.log", "../error-a.log", "error-.log", "main.log"] {
        assert_eq!(
            logs.request_error_log(name),
            Err(ManagementLogError::InvalidFileName)
        );
    }
}

#[test]
fn error_logs_are_sorted_newest_first_and_delete_is_authority_owned() {
    let store = Arc::new(Store::default());
    let logs = ManagementLogs::new(store.clone());
    let files = logs.request_error_logs().unwrap();
    assert_eq!(files[0].name, "error-new.log");
    assert_eq!(logs.delete_logs().unwrap(), 3);
    assert_eq!(*store.cleared.lock().unwrap(), 1);
}

#[test]
fn attachment_debug_never_renders_log_contents() {
    let rendered = format!("{:?}", attachment("error-a.log"));
    assert!(!rendered.contains("private log contents"));
    assert!(rendered.contains("20 BYTES"));
}
