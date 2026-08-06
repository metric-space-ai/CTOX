// ref: internal/pluginstore/github_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use chrono::{TimeZone, Utc};

use crate::sdk::pluginstore::{
    Client, Release, ReleaseAsset, ResolvedAuthConfig, Secret, AUTH_TYPE_BEARER,
    REQUEST_KIND_REGISTRY,
};

use super::checksum::{parse_checksums, verify_checksum};
use super::github::{
    archive_name, select_release_assets, HttpRequest, HttpResponse, PluginStoreTransport,
    SafePluginStoreIo,
};
use super::UrlPolicy;

type FixtureResponse = (u16, BTreeMap<String, String>, Vec<u8>);

struct SequenceTransport {
    responses: Mutex<Vec<FixtureResponse>>,
    authorizations: Mutex<Vec<Option<String>>>,
}

impl PluginStoreTransport for SequenceTransport {
    fn get(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, String> {
        self.authorizations
            .lock()
            .unwrap()
            .push(request.headers.get("Authorization").map(ToOwned::to_owned));
        let (status, headers, body) = self.responses.lock().unwrap().remove(0);
        Ok(HttpResponse {
            status,
            headers,
            body: Box::new(Cursor::new(body)),
        })
    }
}

#[test]
fn release_assets_and_checksum_match_pinned_contract() {
    let name = archive_name("sample", "1.2.3", "linux", "amd64");
    assert_eq!(name, "sample_1.2.3_linux_amd64.zip");
    let release = Release {
        tag_name: "v1.2.3".into(),
        assets: vec![
            ReleaseAsset {
                name: name.clone(),
                browser_download_url: "https://example/archive".into(),
                ..ReleaseAsset::default()
            },
            ReleaseAsset {
                name: "checksums.txt".into(),
                browser_download_url: "https://example/checksums".into(),
                ..ReleaseAsset::default()
            },
        ],
    };
    let (archive, checksums) =
        select_release_assets(&release, "sample", "1.2.3", "linux", "amd64").unwrap();
    assert_eq!(archive.name, name);
    assert_eq!(checksums.name, "checksums.txt");
    let map = parse_checksums(
        b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  sample.zip\n",
    )
    .unwrap();
    verify_checksum("sample.zip", b"abc", &map).unwrap();
    assert!(verify_checksum("sample.zip", b"abd", &map).is_err());
}

#[test]
fn redirect_reapplies_auth_per_origin_and_authenticated_errors_hide_body() {
    let transport = Arc::new(SequenceTransport {
        responses: Mutex::new(vec![
            (
                302,
                BTreeMap::from([(
                    "Location".into(),
                    "https://public.example/registry.json".into(),
                )]),
                Vec::new(),
            ),
            (
                200,
                BTreeMap::new(),
                br#"{"schema_version":1,"plugins":[]}"#.to_vec(),
            ),
        ]),
        authorizations: Mutex::new(Vec::new()),
    });
    let now = Utc.with_ymd_and_hms(2026, 8, 4, 0, 0, 0).unwrap();
    let io = Arc::new(SafePluginStoreIo::new(
        transport.clone(),
        UrlPolicy::default(),
        Arc::new(move || now),
    ));
    let client = Client::with_resolved_auth(
        io,
        "https://private.example/registry.json",
        vec![ResolvedAuthConfig {
            match_url: "https://private.example/".into(),
            apply_to: vec![REQUEST_KIND_REGISTRY.into()],
            auth_type: AUTH_TYPE_BEARER.into(),
            token: Secret::new(b"secret-token".to_vec()),
            ..ResolvedAuthConfig::default()
        }],
    );
    client.fetch_registry().unwrap();
    assert_eq!(
        *transport.authorizations.lock().unwrap(),
        vec![Some("Bearer secret-token".into()), None]
    );

    let failing = Arc::new(SequenceTransport {
        responses: Mutex::new(vec![(
            401,
            BTreeMap::new(),
            b"body contains secret-token".to_vec(),
        )]),
        authorizations: Mutex::new(Vec::new()),
    });
    let io = Arc::new(SafePluginStoreIo::new(
        failing,
        UrlPolicy::default(),
        Arc::new(move || now),
    ));
    let client = Client::with_resolved_auth(
        io,
        "https://private.example/registry.json",
        vec![ResolvedAuthConfig {
            match_url: "https://private.example/".into(),
            auth_type: AUTH_TYPE_BEARER.into(),
            token: Secret::new(b"secret-token".to_vec()),
            ..ResolvedAuthConfig::default()
        }],
    );
    let error = client.fetch_registry().unwrap_err().to_string();
    assert!(error.contains("401"));
    assert!(!error.contains("secret-token"));
}
