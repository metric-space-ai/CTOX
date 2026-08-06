// ref: internal/pluginstore/github.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use url::Url;
use zeroize::Zeroize;

use crate::sdk::pluginstore::{
    github_repository_parts, Client, InstallOptions, InstallResult, Manifest, Plugin,
    PluginStoreError, PluginStoreIo, Registry, Release, ReleaseAsset, Result,
    REQUEST_KIND_ARTIFACT, REQUEST_KIND_METADATA, REQUEST_KIND_REGISTRY,
};

use super::auth::{apply_resolved_auth, request_url_allowed, UrlPolicy};
use super::install::{install_manifest, install_plugin, install_version};
use super::registry::parse_registry;

const MAX_REDIRECTS: usize = 10;

pub(crate) fn store_error(message: impl Into<String>) -> PluginStoreError {
    PluginStoreError::Message(message.into())
}

#[derive(Clone, Eq, PartialEq)]
pub struct HttpRequest {
    pub url: Url,
    pub headers: BTreeMap<String, String>,
}

pub struct HttpResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Box<dyn Read + Send>,
}

/// A no-redirect GET transport. TLS, proxy and socket configuration are owned
/// by the injected implementation; the store owns redirect/auth policy.
pub trait PluginStoreTransport: Send + Sync {
    fn get(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, String>;
}

pub struct SafePluginStoreIo {
    transport: Arc<dyn PluginStoreTransport>,
    policy: UrlPolicy,
    user_agent: String,
    registry_limit: usize,
    metadata_limit: usize,
    artifact_limit: usize,
    now: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
}

impl SafePluginStoreIo {
    pub fn new(
        transport: Arc<dyn PluginStoreTransport>,
        policy: UrlPolicy,
        now: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
    ) -> Self {
        Self {
            transport,
            policy,
            user_agent: "CLIProxyAPI".to_owned(),
            registry_limit: 8 * 1024 * 1024,
            metadata_limit: 2 * 1024 * 1024,
            artifact_limit: 256 * 1024 * 1024,
            now,
        }
    }

    pub fn with_limits(mut self, registry: usize, metadata: usize, artifact: usize) -> Self {
        self.registry_limit = registry;
        self.metadata_limit = metadata;
        self.artifact_limit = artifact;
        self
    }

    fn get(
        &self,
        client: &Client,
        request_url: &str,
        accept: &str,
        kind: &str,
        limit: usize,
    ) -> Result<Vec<u8>> {
        let mut current = request_url.trim().to_owned();
        for redirects in 0..=MAX_REDIRECTS {
            let parsed = request_url_allowed(&self.policy, &current)?;
            let mut headers = BTreeMap::from([
                ("Accept".to_owned(), accept.to_owned()),
                ("User-Agent".to_owned(), self.user_agent.clone()),
            ]);
            let authenticated = apply_resolved_auth(
                &mut headers,
                client.resolved_auth(),
                client.resolved_auth_expires_at(),
                parsed.as_str(),
                kind,
                (self.now)(),
            )?;
            let mut request = HttpRequest {
                url: parsed,
                headers,
            };
            let safe_url = redacted_url(&request.url);
            let response = self.transport.get(&request);
            for value in request.headers.values_mut() {
                value.zeroize();
            }
            let mut response = response
                .map_err(|error| store_error(format!("request {safe_url} failed: {error}")))?;
            if is_redirect(response.status) {
                if redirects == MAX_REDIRECTS {
                    return Err(store_error(format!(
                        "stopped after {MAX_REDIRECTS} redirects"
                    )));
                }
                let location = header(&response.headers, "location")
                    .ok_or_else(|| store_error("redirect missing Location header"))?;
                current = request
                    .url
                    .join(location.trim())
                    .map_err(|_| store_error("invalid redirect location"))?
                    .to_string();
                continue;
            }
            if !(200..300).contains(&response.status) {
                if authenticated {
                    return Err(store_error(format!(
                        "unexpected status {}",
                        response.status
                    )));
                }
                let detail = read_bounded(&mut response.body, 4_096, "error response")?;
                return Err(store_error(format!(
                    "unexpected status {}: {}",
                    response.status,
                    String::from_utf8_lossy(&detail).trim()
                )));
            }
            return read_bounded(&mut response.body, limit, "response");
        }
        unreachable!()
    }

    fn release(&self, client: &Client, plugin: &Plugin, suffix: &str) -> Result<Release> {
        let (owner, repo) = github_repository_parts(&plugin.repository)?;
        let url = format!("https://api.github.com/repos/{owner}/{repo}/releases/{suffix}");
        decode_json(
            &self.get(
                client,
                &url,
                "application/vnd.github+json",
                REQUEST_KIND_METADATA,
                self.metadata_limit,
            )?,
            "release",
        )
    }

    pub(crate) fn download_artifact(
        &self,
        client: &Client,
        url: &str,
        declared_size: i64,
    ) -> Result<Vec<u8>> {
        let size_limit = if declared_size > 0 {
            usize::try_from(declared_size).map_err(|_| store_error("artifact size is invalid"))?
        } else {
            self.artifact_limit
        };
        self.get(
            client,
            url,
            "application/octet-stream",
            REQUEST_KIND_ARTIFACT,
            size_limit,
        )
    }

    pub(crate) fn fetch_registry_at(&self, client: &Client, url: &str) -> Result<Registry> {
        parse_registry(&self.get(
            client,
            url,
            "application/json",
            REQUEST_KIND_REGISTRY,
            self.registry_limit,
        )?)
    }

    pub(crate) fn download_asset(&self, client: &Client, asset: &ReleaseAsset) -> Result<Vec<u8>> {
        let api_authenticated = crate::sdk::pluginstore::resolved_auth_for_request(
            client.resolved_auth(),
            &asset.api_url,
            REQUEST_KIND_ARTIFACT,
        )
        .is_some_and(|item| !matches!(item.auth_type.trim(), "" | "none"));
        let url = if asset.browser_download_url.trim().is_empty() || api_authenticated {
            asset.api_url.trim()
        } else {
            asset.browser_download_url.trim()
        };
        if url.is_empty() {
            return Err(store_error(format!(
                "asset {:?} missing download url",
                asset.name
            )));
        }
        self.download_artifact(client, url, 0)
    }
}

impl PluginStoreIo for SafePluginStoreIo {
    fn fetch_registry(&self, client: &Client) -> Result<Registry> {
        parse_registry(&self.get(
            client,
            client.registry_url(),
            "application/json",
            REQUEST_KIND_REGISTRY,
            self.registry_limit,
        )?)
    }

    fn fetch_latest_release(&self, client: &Client, plugin: &Plugin) -> Result<Release> {
        self.release(client, plugin, "latest")
    }

    fn fetch_release_by_tag(&self, client: &Client, plugin: &Plugin, tag: &str) -> Result<Release> {
        let tag = tag.trim();
        if tag.is_empty() {
            return Err(store_error("release tag is required"));
        }
        self.release(
            client,
            plugin,
            &format!(
                "tags/{}",
                percent_encoding::utf8_percent_encode(tag, percent_encoding::NON_ALPHANUMERIC)
            ),
        )
    }

    fn install(
        &self,
        client: &Client,
        plugin: &Plugin,
        options: &InstallOptions,
    ) -> Result<InstallResult> {
        install_plugin(self, client, plugin, options)
    }

    fn install_version(
        &self,
        client: &Client,
        plugin: &Plugin,
        release_tag: &str,
        version: &str,
        options: &InstallOptions,
    ) -> Result<InstallResult> {
        install_version(self, client, plugin, release_tag, version, options)
    }

    fn install_manifest(
        &self,
        client: &Client,
        manifest: &Manifest,
        options: &InstallOptions,
    ) -> Result<InstallResult> {
        install_manifest(self, client, manifest, options)
    }
}

pub(crate) fn archive_name(id: &str, version: &str, goos: &str, goarch: &str) -> String {
    format!(
        "{}_{}_{}_{}.zip",
        id.trim(),
        version.trim(),
        goos.trim(),
        goarch.trim()
    )
}

pub(crate) fn select_release_assets(
    release: &Release,
    id: &str,
    version: &str,
    goos: &str,
    goarch: &str,
) -> Result<(ReleaseAsset, ReleaseAsset)> {
    let archive_name = archive_name(id, version, goos, goarch);
    let archive = release
        .assets
        .iter()
        .find(|asset| asset.name.trim() == archive_name)
        .cloned()
        .ok_or_else(|| store_error(format!("release asset {archive_name} not found")))?;
    let checksums = release
        .assets
        .iter()
        .find(|asset| asset.name.trim() == "checksums.txt")
        .cloned()
        .ok_or_else(|| store_error("release asset checksums.txt not found"))?;
    Ok((archive, checksums))
}

fn decode_json<T: DeserializeOwned>(data: &[u8], kind: &str) -> Result<T> {
    serde_json::from_slice(data).map_err(|error| store_error(format!("decode {kind}: {error}")))
}

fn read_bounded(reader: &mut dyn Read, limit: usize, label: &str) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| store_error(format!("read {label}: {error}")))?;
        if read == 0 {
            break;
        }
        if read > limit.saturating_sub(output.len()) {
            return Err(store_error(format!(
                "{label} exceeds maximum allowed size of {limit} bytes"
            )));
        }
        output.extend_from_slice(&buffer[..read]);
    }
    Ok(output)
}

fn header<'a>(headers: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn redacted_url(url: &Url) -> String {
    let mut safe = url.clone();
    safe.set_query(None);
    safe.set_fragment(None);
    let _ = safe.set_username("");
    let _ = safe.set_password(None);
    safe.to_string()
}
