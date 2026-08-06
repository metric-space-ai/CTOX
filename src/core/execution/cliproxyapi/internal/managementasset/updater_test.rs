// ref: internal/managementasset/updater_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use crate::internal::httpfetch::{
    BodyChunkFuture, FetchFuture, FetchResponse, Headers, HttpDoer, ResponseBody,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tempfile::TempDir;

struct Body(VecDeque<Result<Vec<u8>, String>>);

impl ResponseBody for Body {
    fn next_chunk(&mut self) -> BodyChunkFuture<'_> {
        Box::pin(async move { self.0.pop_front().transpose() })
    }
}

#[derive(Default)]
struct MockClient {
    responses: StdMutex<MockResponses>,
    requests: StdMutex<Vec<(String, Headers)>>,
}

type MockResponses = BTreeMap<String, VecDeque<(u16, Vec<u8>)>>;

impl MockClient {
    fn respond(&self, url: &str, status: u16, body: impl Into<Vec<u8>>) {
        self.responses
            .lock()
            .expect("responses")
            .entry(url.to_owned())
            .or_default()
            .push_back((status, body.into()));
    }
}

impl HttpDoer for MockClient {
    fn get<'a>(&'a self, request_url: &'a str, headers: &'a Headers) -> FetchFuture<'a> {
        Box::pin(async move {
            self.requests
                .lock()
                .map_err(|_| "request lock poisoned".to_owned())?
                .push((request_url.to_owned(), headers.clone()));
            let (status, body) = self
                .responses
                .lock()
                .map_err(|_| "response lock poisoned".to_owned())?
                .get_mut(request_url)
                .and_then(VecDeque::pop_front)
                .ok_or_else(|| format!("unexpected request: {request_url}"))?;
            Ok(FetchResponse {
                status,
                body: Box::new(Body(VecDeque::from([Ok(body)]))),
            })
        })
    }
}

fn release_json(download_url: &str, digest: &str) -> Vec<u8> {
    serde_json::json!({
        "assets": [{
            "name": "management.html",
            "browser_download_url": download_url,
            "digest": digest
        }]
    })
    .to_string()
    .into_bytes()
}

fn test_sha256_hex(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

#[test]
fn auto_update_skip_reason_matches_pinned_go_table() {
    assert_eq!(
        auto_update_skip_reason(None),
        ("config not yet available", true)
    );
    let mut config = ManagementConfig {
        home_enabled: true,
        ..ManagementConfig::default()
    };
    assert_eq!(
        auto_update_skip_reason(Some(&config)),
        ("cluster mode enabled", true)
    );
    config.home_enabled = false;
    config.disable_control_panel = true;
    assert_eq!(
        auto_update_skip_reason(Some(&config)),
        ("control panel disabled", true)
    );
    config.disable_control_panel = false;
    config.disable_auto_update_panel = true;
    assert_eq!(
        auto_update_skip_reason(Some(&config)),
        ("disable-auto-update-panel is enabled", true)
    );
    config.disable_auto_update_panel = false;
    assert_eq!(auto_update_skip_reason(Some(&config)), ("", false));
}

#[test]
fn release_url_resolution_matches_upstream_and_rejects_unsafe_forms() {
    assert_eq!(resolve_release_url(""), DEFAULT_MANAGEMENT_RELEASE_URL);
    assert_eq!(
        resolve_release_url("https://github.com/acme/panel.git"),
        "https://api.github.com/repos/acme/panel/releases/latest"
    );
    assert_eq!(
        resolve_release_url("https://api.github.com/repos/acme/panel"),
        "https://api.github.com/repos/acme/panel/releases/latest"
    );
    assert_eq!(
        resolve_release_url("http://github.com/acme/panel"),
        DEFAULT_MANAGEMENT_RELEASE_URL
    );
    assert_eq!(
        resolve_release_url("https://user@github.com/acme/panel"),
        DEFAULT_MANAGEMENT_RELEASE_URL
    );
    assert_eq!(
        resolve_release_url("https://example.com/acme/panel"),
        DEFAULT_MANAGEMENT_RELEASE_URL
    );
}

#[test]
fn typed_paths_preserve_upstream_precedence() {
    let config = Path::new("/srv/config/config.yaml");
    let paths = AssetPaths {
        static_override: Some(PathBuf::from("/override/management.HTML")),
        writable_path: Some(PathBuf::from("/writable")),
    };
    assert_eq!(static_dir(config, &paths), Some(PathBuf::from("/override")));
    assert_eq!(
        file_path(config, &paths),
        Some(PathBuf::from("/override/management.HTML"))
    );
    let paths = AssetPaths {
        static_override: None,
        writable_path: Some(PathBuf::from("/writable")),
    };
    assert_eq!(
        file_path(config, &paths),
        Some(PathBuf::from("/writable/static/management.html"))
    );
    assert_eq!(
        file_path(config, &AssetPaths::default()),
        Some(PathBuf::from("/srv/config/static/management.html"))
    );
}

#[test]
fn digest_parser_matches_upstream() {
    assert_eq!(parse_digest(" SHA256:ABCDef "), "abcdef");
    assert_eq!(parse_digest("ABCDef"), "abcdef");
    assert_eq!(parse_digest(""), "");
}

#[tokio::test]
async fn verified_asset_is_written_and_current_digest_skips_download() {
    let temp = TempDir::new().expect("temp");
    let client = Arc::new(MockClient::default());
    let asset_url = "https://github.com/acme/panel/releases/download/v1/management.html";
    let data = b"<html>verified</html>";
    let hash = test_sha256_hex(data);
    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(asset_url, &format!("sha256:{hash}")),
    );
    client.respond(asset_url, 200, data.to_vec());
    let updater = ManagementAssetUpdater::new(client.clone(), FetchAuth::default())
        .with_intervals(Duration::ZERO, UPDATE_CHECK_INTERVAL);
    assert_eq!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::Updated {
            sha256: hash.clone()
        }
    );
    assert_eq!(
        fs::read(temp.path().join(MANAGEMENT_FILE_NAME)).expect("asset"),
        data
    );

    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(asset_url, &format!("sha256:{hash}")),
    );
    assert_eq!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::AlreadyCurrent
    );
    assert_eq!(client.requests.lock().expect("requests").len(), 3);
}

#[tokio::test]
async fn hot_reloaded_proxy_selects_a_fresh_injected_client_each_attempt() {
    let temp = TempDir::new().expect("temp");
    let first = Arc::new(MockClient::default());
    let second = Arc::new(MockClient::default());
    let first_asset = "https://github.com/acme/panel/releases/download/v1/management.html";
    let second_asset = "https://github.com/acme/panel/releases/download/v2/management.html";
    let first_data = b"first proxy";
    let second_data = b"second proxy";
    first.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(
            first_asset,
            &format!("sha256:{}", test_sha256_hex(first_data)),
        ),
    );
    first.respond(first_asset, 200, first_data.to_vec());
    second.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(
            second_asset,
            &format!("sha256:{}", test_sha256_hex(second_data)),
        ),
    );
    second.respond(second_asset, 200, second_data.to_vec());

    let first_for_selector = Arc::clone(&first);
    let second_for_selector = Arc::clone(&second);
    let updater = ManagementAssetUpdater::new(Arc::clone(&first), FetchAuth::default())
        .with_client_selector(Arc::new(move |proxy_url| match proxy_url {
            "http://proxy-one:8080" => Arc::clone(&first_for_selector),
            "http://proxy-two:8080" => Arc::clone(&second_for_selector),
            unexpected => panic!("unexpected proxy: {unexpected}"),
        }))
        .with_intervals(Duration::ZERO, UPDATE_CHECK_INTERVAL);

    assert!(matches!(
        updater
            .ensure_latest(temp.path(), " http://proxy-one:8080 ", "")
            .await,
        SyncOutcome::Updated { .. }
    ));
    assert!(matches!(
        updater
            .ensure_latest(temp.path(), "http://proxy-two:8080", "")
            .await,
        SyncOutcome::Updated { .. }
    ));
    assert_eq!(first.requests.lock().expect("requests").len(), 2);
    assert_eq!(second.requests.lock().expect("requests").len(), 2);
    assert_eq!(
        fs::read(temp.path().join(MANAGEMENT_FILE_NAME)).expect("asset"),
        second_data
    );
}

#[tokio::test]
async fn digest_mismatch_never_replaces_existing_asset() {
    let temp = TempDir::new().expect("temp");
    let local = temp.path().join(MANAGEMENT_FILE_NAME);
    fs::write(&local, b"old").expect("old asset");
    let client = Arc::new(MockClient::default());
    let asset_url = "https://github.com/acme/panel/releases/download/v1/management.html";
    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(asset_url, "sha256:0000"),
    );
    client.respond(asset_url, 200, b"tampered".to_vec());
    let updater = ManagementAssetUpdater::new(client, FetchAuth::default());
    let result = updater.ensure_latest(temp.path(), "", "").await;
    assert!(
        matches!(result, SyncOutcome::RetainedExisting { reason } if reason.contains("digest mismatch"))
    );
    assert_eq!(fs::read(local).expect("asset"), b"old");
}

#[tokio::test]
async fn missing_asset_uses_explicit_https_fallback() {
    let temp = TempDir::new().expect("temp");
    let client = Arc::new(MockClient::default());
    client.respond(DEFAULT_MANAGEMENT_RELEASE_URL, 500, b"no release".to_vec());
    client.respond(DEFAULT_MANAGEMENT_FALLBACK_URL, 200, b"fallback".to_vec());
    let updater = ManagementAssetUpdater::new(client, FetchAuth::default());
    assert!(matches!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::UpdatedFromFallback { .. }
    ));
    assert_eq!(
        fs::read(temp.path().join(MANAGEMENT_FILE_NAME)).expect("asset"),
        b"fallback"
    );
}

#[tokio::test]
async fn throttle_is_instance_scoped_and_preserves_existence_result() {
    let temp = TempDir::new().expect("temp");
    fs::write(temp.path().join(MANAGEMENT_FILE_NAME), b"old").expect("asset");
    let client = Arc::new(MockClient::default());
    client.respond(DEFAULT_MANAGEMENT_RELEASE_URL, 500, b"failure".to_vec());
    let updater = ManagementAssetUpdater::new(client, FetchAuth::default());
    assert!(matches!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::RetainedExisting { .. }
    ));
    assert_eq!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::Throttled { asset_exists: true }
    );
}

#[tokio::test]
async fn github_token_is_only_sent_at_authorized_api_boundary() {
    let temp = TempDir::new().expect("temp");
    let client = Arc::new(MockClient::default());
    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        br#"{"assets":[]}"#.to_vec(),
    );
    client.respond(DEFAULT_MANAGEMENT_FALLBACK_URL, 500, b"no".to_vec());
    let updater = ManagementAssetUpdater::new(
        client.clone(),
        FetchAuth {
            git_url: "https://github.com/acme/private".to_owned(),
            github_token: Some("secret".to_owned()),
        },
    );
    let _ = updater.ensure_latest(temp.path(), "", "").await;
    let requests = client.requests.lock().expect("requests");
    assert_eq!(
        requests[0].1.get("Authorization").map(String::as_str),
        Some("Bearer secret")
    );
    assert!(!requests[1].1.contains_key("Authorization"));
}

#[tokio::test]
async fn unsafe_download_url_and_symlink_target_are_rejected() {
    let temp = TempDir::new().expect("temp");
    let client = Arc::new(MockClient::default());
    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json("http://127.0.0.1/private", ""),
    );
    let updater = ManagementAssetUpdater::new(client, FetchAuth::default()).with_fallback_url(None);
    assert!(matches!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::Missing { reason } if reason.contains("URL rejected")
    ));

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let second = TempDir::new().expect("temp");
        let victim = second.path().join("victim");
        fs::write(&victim, b"victim").expect("victim");
        symlink(&victim, temp.path().join(MANAGEMENT_FILE_NAME)).expect("symlink");
        let client = Arc::new(MockClient::default());
        let updater =
            ManagementAssetUpdater::new(client, FetchAuth::default()).with_fallback_url(None);
        assert!(matches!(
            updater.ensure_latest(temp.path(), "", "").await,
            SyncOutcome::Missing { reason } if reason.contains("unsafe management asset path")
        ));
        assert_eq!(fs::read(victim).expect("victim"), b"victim");
    }
}

#[tokio::test]
async fn response_larger_than_upstream_limit_is_not_persisted() {
    let temp = TempDir::new().expect("temp");
    let client = Arc::new(MockClient::default());
    let asset_url = "https://github.com/acme/panel/releases/download/v1/management.html";
    client.respond(
        DEFAULT_MANAGEMENT_RELEASE_URL,
        200,
        release_json(asset_url, ""),
    );
    client.respond(asset_url, 200, vec![0; MAX_ASSET_DOWNLOAD_SIZE + 1]);
    let updater = ManagementAssetUpdater::new(client, FetchAuth::default()).with_fallback_url(None);
    assert!(matches!(
        updater.ensure_latest(temp.path(), "", "").await,
        SyncOutcome::Missing { reason } if reason.contains("exceeds maximum")
    ));
    assert!(!temp.path().join(MANAGEMENT_FILE_NAME).exists());
}
