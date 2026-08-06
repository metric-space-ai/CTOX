// ref: sdk/cliproxy/auth/home_selected_auth_callback_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: selected-auth notification is instance-injected and fires before execution
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::home_execution_paths_test::{request, runtime, TestExecutor, TestHomeTransport};
use super::HomeSelectedAuthPublisher;

#[derive(Default)]
struct Selected(Mutex<Vec<(String, String)>>);
impl HomeSelectedAuthPublisher for Selected {
    fn selected(&self, auth_id: &str, auth_index: &str) {
        self.0
            .lock()
            .unwrap()
            .push((auth_id.to_owned(), auth_index.to_owned()));
    }
}

#[tokio::test]
async fn callback_receives_selected_identity_once_per_attempt() {
    let transport = TestHomeTransport::with_auth_ids(&["auth-selected"]);
    let executor = TestExecutor::failing(0);
    let (runtime, _) = runtime(transport, executor);
    let selected = Arc::new(Selected::default());
    runtime.set_selected_auth_publisher(Some(selected.clone()));
    runtime
        .execute_home(request("gpt"), "", false)
        .await
        .unwrap();
    assert_eq!(
        *selected.0.lock().unwrap(),
        vec![("auth-selected".into(), "auth-selected".into())]
    );
}

#[tokio::test]
async fn callback_can_be_removed_without_global_residue() {
    let transport = TestHomeTransport::with_auth_ids(&["first", "second"]);
    let executor = TestExecutor::failing(0);
    let (runtime, _) = runtime(transport, executor);
    let selected = Arc::new(Selected::default());
    runtime.set_selected_auth_publisher(Some(selected.clone()));
    runtime
        .execute_home(request("gpt"), "", false)
        .await
        .unwrap();
    runtime.set_selected_auth_publisher(None);
    runtime
        .execute_home(request("gpt"), "", false)
        .await
        .unwrap();
    assert_eq!(selected.0.lock().unwrap().len(), 1);
}
