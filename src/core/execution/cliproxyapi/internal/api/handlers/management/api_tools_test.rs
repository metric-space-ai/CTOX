// ref: internal/api/handlers/management/api_tools_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::{
    ManagementApiCallExecutor, ManagementApiToolError, ManagementApiToolRequest,
    ManagementApiToolResponse, ManagementApiTools,
};

#[derive(Default)]
struct Executor(Mutex<Vec<ManagementApiToolRequest>>);

impl ManagementApiCallExecutor for Executor {
    fn execute(
        &self,
        request: &ManagementApiToolRequest,
    ) -> Result<ManagementApiToolResponse, ManagementApiToolError> {
        self.0.lock().unwrap().push(request.clone());
        Ok(ManagementApiToolResponse {
            status_code: 204,
            headers: BTreeMap::new(),
            body: String::new(),
        })
    }
}

#[test]
fn api_call_delegates_proxy_and_token_policy_to_injected_authority() {
    let executor = Arc::new(Executor::default());
    let tools = ManagementApiTools::new(executor.clone());
    let response = tools
        .execute_json(
            br#"{"auth_index":" idx-a ","method":"post","url":"https://api.example.com/v1/ping","header":{"Authorization":"Bearer $TOKEN$"},"data":"{}"}"#,
        )
        .unwrap();
    assert_eq!(response.status_code, 204);
    let requests = executor.0.lock().unwrap();
    assert_eq!(requests[0].auth_index, "idx-a");
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].headers["Authorization"], "Bearer $TOKEN$");
}

#[test]
fn all_upstream_auth_index_spellings_select_the_same_credential() {
    for field in ["auth_index", "authIndex", "AuthIndex"] {
        let executor = Arc::new(Executor::default());
        let tools = ManagementApiTools::new(executor.clone());
        let payload =
            format!(r#"{{"{field}":"idx-b","method":"GET","url":"https://api.example.com"}}"#);
        tools.execute_json(payload.as_bytes()).unwrap();
        assert_eq!(executor.0.lock().unwrap()[0].auth_index, "idx-b");
    }
}

#[test]
fn invalid_or_insecure_targets_are_rejected_before_authority_execution() {
    let executor = Arc::new(Executor::default());
    let tools = ManagementApiTools::new(executor.clone());
    for (url, expected) in [
        ("relative/path", ManagementApiToolError::InvalidUrl),
        (
            "http://127.0.0.1/admin",
            ManagementApiToolError::InsecureUrl,
        ),
        ("file:///etc/passwd", ManagementApiToolError::InvalidUrl),
    ] {
        let request = ManagementApiToolRequest {
            auth_index: String::new(),
            method: "GET".to_owned(),
            url: url.to_owned(),
            headers: BTreeMap::new(),
            body: String::new(),
        };
        assert_eq!(tools.execute(request), Err(expected));
    }
    assert!(executor.0.lock().unwrap().is_empty());
}

#[test]
fn debug_output_redacts_headers_and_body() {
    let request = ManagementApiToolRequest {
        auth_index: "idx".to_owned(),
        method: "POST".to_owned(),
        url: "https://api.example.com".to_owned(),
        headers: BTreeMap::from([("Authorization".to_owned(), "secret-value".to_owned())]),
        body: "private-body".to_owned(),
    };
    let rendered = format!("{request:?}");
    assert!(!rendered.contains("secret-value"));
    assert!(!rendered.contains("private-body"));
}
