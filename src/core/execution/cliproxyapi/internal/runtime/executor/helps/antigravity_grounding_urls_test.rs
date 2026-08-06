// ref: internal/runtime/executor/helps/antigravity_grounding_urls_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Mutex;

use serde_json::Value;
use url::Url;

use super::antigravity_grounding_urls::{
    is_antigravity_vertex_search_redirect, resolve_antigravity_grounding_urls,
    GroundingRedirectError, GroundingRedirectResponse, GroundingRedirectTransport,
};

struct Transport {
    calls: Mutex<Vec<String>>,
    response: Result<GroundingRedirectResponse, GroundingRedirectError>,
}

impl Transport {
    fn new(response: Result<GroundingRedirectResponse, GroundingRedirectError>) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            response,
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl GroundingRedirectTransport for Transport {
    fn head(&self, url: &Url) -> Result<GroundingRedirectResponse, GroundingRedirectError> {
        self.calls.lock().unwrap().push(url.as_str().to_owned());
        self.response.clone()
    }
}

const REDIRECT: &str =
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/example-token";

#[test]
fn resolves_vertex_redirect_with_one_injected_head_request() {
    let transport = Transport::new(Ok(GroundingRedirectResponse {
        status: 302,
        location: Some("https://example.com/weather".to_owned()),
    }));
    let input = format!(
        r#"{{"response":{{"candidates":[{{"groundingMetadata":{{"groundingChunks":[{{"web":{{"uri":"{REDIRECT}","title":"Weather"}}}},{{"web":{{"uri":"https://already.example/source"}}}}]}}}}]}}}}"#
    );
    let output = resolve_antigravity_grounding_urls(&transport, input.as_bytes());
    let value: Value = serde_json::from_slice(&output).unwrap();
    let chunks = value
        .pointer("/response/candidates/0/groundingMetadata/groundingChunks")
        .unwrap();
    assert_eq!(chunks[0]["web"]["uri"], "https://example.com/weather");
    assert_eq!(chunks[1]["web"]["uri"], "https://already.example/source");
    assert_eq!(transport.calls(), vec![REDIRECT]);
}

#[test]
fn alternate_envelope_and_duplicate_urls_share_one_lookup() {
    let transport = Transport::new(Ok(GroundingRedirectResponse {
        status: 307,
        location: Some(" https://target.example/path ".to_owned()),
    }));
    let input = format!(
        r#"{{"candidates":[{{"groundingMetadata":{{"groundingChunks":[{{"web":{{"uri":"{REDIRECT}"}}}},{{"web":{{"uri":"{REDIRECT}"}}}}]}}}}]}}"#
    );
    let output = resolve_antigravity_grounding_urls(&transport, input.as_bytes());
    let value: Value = serde_json::from_slice(&output).unwrap();
    let chunks = value
        .pointer("/candidates/0/groundingMetadata/groundingChunks")
        .unwrap();
    assert_eq!(chunks[0]["web"]["uri"], "https://target.example/path");
    assert_eq!(chunks[1]["web"]["uri"], "https://target.example/path");
    assert_eq!(transport.calls().len(), 1);
}

#[test]
fn host_policy_is_exact_and_rejects_authority_confusion() {
    assert!(is_antigravity_vertex_search_redirect(REDIRECT));
    for url in [
        "http://vertexaisearch.cloud.google.com/grounding-api-redirect/x",
        "https://vertexaisearch.cloud.google.com.evil.test/grounding-api-redirect/x",
        "https://user@vertexaisearch.cloud.google.com/grounding-api-redirect/x",
        "https://vertexaisearch.cloud.google.com:444/grounding-api-redirect/x",
        "https://vertexaisearch.cloud.google.com/not-grounding/x",
    ] {
        assert!(!is_antigravity_vertex_search_redirect(url), "{url}");
    }
}

#[test]
fn invalid_redirect_responses_fail_closed_and_preserve_bytes() {
    for response in [
        Ok(GroundingRedirectResponse {
            status: 200,
            location: Some("https://target.example".to_owned()),
        }),
        Ok(GroundingRedirectResponse {
            status: 302,
            location: Some("http://target.example".to_owned()),
        }),
        Ok(GroundingRedirectResponse {
            status: 302,
            location: Some("https://user:secret@target.example".to_owned()),
        }),
        Err(GroundingRedirectError::Transport),
    ] {
        let transport = Transport::new(response);
        let input = format!(
            r#" {{ "candidates":[{{"groundingMetadata":{{"groundingChunks":[{{"web":{{"uri":"{REDIRECT}"}}}}]}}}}]}} "#
        );
        assert_eq!(
            resolve_antigravity_grounding_urls(&transport, input.as_bytes()),
            input.as_bytes()
        );
    }
}

#[test]
fn unsupported_payloads_are_byte_identical_without_transport_authority() {
    let transport = Transport::new(Err(GroundingRedirectError::Transport));
    for input in [
        b"".as_slice(),
        b"not-json",
        br#" { "candidates": [] } "#,
        br#"{"candidates":[{"groundingMetadata":{"groundingChunks":{}}}]}"#,
    ] {
        assert_eq!(resolve_antigravity_grounding_urls(&transport, input), input);
    }
    assert!(transport.calls().is_empty());
}
