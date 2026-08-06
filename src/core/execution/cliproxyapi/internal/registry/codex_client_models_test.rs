// ref: internal/registry/codex_client_models_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{collections::HashMap, sync::Mutex};

use serde_json::{json, Value};

use super::*;

fn model(slug: &str, priority: i64) -> Value {
    json!({
        "slug": slug,
        "display_name": format!("Test {slug}"),
        "description": "Test model",
        "base_instructions": "Test instructions",
        "minimal_client_version": "0.144.0",
        "visibility": "list",
        "context_window": 372000,
        "max_context_window": 372000,
        "priority": priority,
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [{"effort":"medium","description":"Balanced"}]
    })
}

fn catalog(models: Vec<Value>) -> Vec<u8> {
    serde_json::to_vec(&json!({"models": models})).unwrap()
}

#[test]
fn validates_complete_catalog_and_rejects_every_upstream_invalid_class() {
    let default = model("gpt-5.5", 1);
    let other = model("gpt-5.6-sol", 2);
    assert!(
        validate_codex_client_models_json(&catalog(vec![default.clone(), other.clone()])).is_ok()
    );

    let mut empty_slug = default.clone();
    empty_slug["slug"] = json!("");
    let mut missing = default.clone();
    missing.as_object_mut().unwrap().remove("base_instructions");
    let mut wrong_type = default.clone();
    wrong_type["context_window"] = json!("372000");
    let mut unsupported = default.clone();
    unsupported["default_reasoning_level"] = json!("high");

    let invalid = [
        br#"{"models":"#.to_vec(),
        catalog(vec![]),
        catalog(vec![empty_slug]),
        catalog(vec![default.clone(), default]),
        catalog(vec![other]),
        catalog(vec![missing]),
        catalog(vec![wrong_type]),
        catalog(vec![unsupported]),
    ];
    for raw in invalid {
        assert!(validate_codex_client_models_json(&raw).is_err());
    }
}

#[test]
fn store_copies_snapshots_changes_revision_only_on_content_and_keeps_last_valid() {
    let first = catalog(vec![model("gpt-5.5", 1)]);
    let store = CodexClientModelsStore::new(&first, "seed").unwrap();
    let mut snapshot = store.snapshot();
    assert_eq!(snapshot.revision, 1);
    snapshot.data[0] ^= 0xff;
    assert_eq!(store.snapshot().data, first);
    assert!(!store.load(&first, "same").unwrap());
    assert_eq!(store.snapshot().revision, 1);
    assert!(store.load(br#"{"models":[]}"#, "invalid").is_err());
    assert_eq!(store.snapshot().data, first);

    let second = catalog(vec![model("gpt-5.5", 2)]);
    assert!(store.load(&second, "second").unwrap());
    assert_eq!(store.snapshot().revision, 2);
}

#[derive(Default)]
struct Source {
    responses: Mutex<HashMap<String, Result<Vec<u8>, String>>>,
}

impl CodexClientModelsSource for Source {
    fn fetch<'a>(&'a self, source: &'a str, max_bytes: usize) -> CodexClientModelsFetchFuture<'a> {
        Box::pin(async move {
            assert_eq!(max_bytes, MAX_CODEX_CLIENT_MODELS_SIZE);
            self.responses
                .lock()
                .unwrap()
                .get(source)
                .cloned()
                .unwrap_or_else(|| Err("missing".into()))
        })
    }
}

#[tokio::test]
async fn refresh_falls_back_and_never_replaces_last_valid_snapshot() {
    let seed = catalog(vec![model("gpt-5.5", 1)]);
    let next = catalog(vec![model("gpt-5.5", 2)]);
    let store = CodexClientModelsStore::new(&seed, "seed").unwrap();
    let source = Source::default();
    source.responses.lock().unwrap().extend([
        ("invalid".into(), Ok(br#"{"models":[]}"#.to_vec())),
        ("valid".into(), Ok(next.clone())),
    ]);
    let refreshed =
        refresh_codex_client_models(&store, &source, &["invalid".into(), "valid".into()])
            .await
            .unwrap();
    assert_eq!(refreshed.source, "valid");
    assert!(refreshed.changed);
    assert_eq!(store.snapshot().data, next);

    source.responses.lock().unwrap().clear();
    assert!(
        refresh_codex_client_models(&store, &source, &["missing".into()])
            .await
            .is_err()
    );
    assert_eq!(store.snapshot().data, next);
}

#[cfg(feature = "codex-http-transport")]
#[tokio::test]
async fn concrete_http_source_fetches_a_bounded_catalog_without_environment_proxy() {
    use tokio::{io::AsyncReadExt, io::AsyncWriteExt, net::TcpListener};

    let data = catalog(vec![model("gpt-5.5", 1)]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_data = data.clone();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = vec![0_u8; 4096];
        let read = stream.read(&mut request).await.unwrap();
        assert!(String::from_utf8_lossy(&request[..read]).starts_with("GET /catalog HTTP/1.1"));
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response_data.len()
        );
        stream.write_all(header.as_bytes()).await.unwrap();
        stream.write_all(&response_data).await.unwrap();
        stream.shutdown().await.unwrap();
    });

    let store = CodexClientModelsStore::default();
    let source = WreqCodexClientModelsSource::new(None).unwrap();
    let refreshed =
        refresh_codex_client_models(&store, &source, &[format!("http://{address}/catalog")])
            .await
            .unwrap();
    server.await.unwrap();
    assert!(refreshed.changed);
    assert_eq!(store.snapshot().data, data);
}
