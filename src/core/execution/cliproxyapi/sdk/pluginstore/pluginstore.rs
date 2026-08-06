// ref: sdk/pluginstore/pluginstore.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Public plugin-store facade.
//!
//! Go's facade owns an `internal/pluginstore.Client` whose transport and
//! filesystem effects are implicit.  The Rust port keeps the same data and
//! validation semantics, but makes that effect boundary an owned, injected
//! [`PluginStoreIo`].  This prevents the SDK from acquiring a global HTTP
//! client, reading process environment variables, or writing an ambient plugin
//! directory.  The internal plugin-store port can implement this trait when it
//! is activated.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::Url;
use zeroize::Zeroize;

pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/router-for-me/CLIProxyAPI-Plugins-Store/main/registry.json";
pub const DEFAULT_SOURCE_ID: &str = "official";
pub const DEFAULT_SOURCE_NAME: &str = "Official";
pub const SCHEMA_VERSION: i32 = 1;
pub const SCHEMA_VERSION_V2: i32 = 2;
pub const INSTALL_TYPE_GITHUB_RELEASE: &str = "github-release";
pub const INSTALL_TYPE_DIRECT: &str = "direct";
pub const REQUEST_KIND_REGISTRY: &str = "registry";
pub const REQUEST_KIND_METADATA: &str = "metadata";
pub const REQUEST_KIND_ARTIFACT: &str = "artifact";
pub const AUTH_TYPE_NONE: &str = "none";
pub const AUTH_TYPE_BEARER: &str = "bearer";
pub const AUTH_TYPE_BASIC: &str = "basic";
pub const AUTH_TYPE_HEADER: &str = "header";
pub const AUTH_TYPE_GITHUB_TOKEN: &str = "github-token";
pub const PLUGIN_SYNC_SCHEMA_VERSION: i32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PluginStoreError {
    LoadedPluginLocked,
    Message(String),
}

pub const ERR_LOADED_PLUGIN_LOCKED: PluginStoreError = PluginStoreError::LoadedPluginLocked;

impl PluginStoreError {
    fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl Display for PluginStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadedPluginLocked => formatter.write_str(
                "loaded plugin library cannot be overwritten while the server is running",
            ),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl Error for PluginStoreError {}

pub type Result<T> = std::result::Result<T, PluginStoreError>;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Source {
    pub id: String,
    pub name: String,
    pub url: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Registry {
    pub schema_version: i32,
    pub plugins: Vec<Plugin>,
}

impl Registry {
    pub fn plugin_by_id(&self, id: &str) -> Option<&Plugin> {
        let id = id.trim();
        self.plugins.iter().find(|plugin| plugin.id.trim() == id)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Plugin {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub versions: Vec<Version>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub repository: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub logo: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub homepage: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub license: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "InstallPlan::is_empty")]
    pub install: InstallPlan,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub auth_required: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Version {
    pub version: String,
    #[serde(default, skip_serializing_if = "InstallPlan::is_empty")]
    pub install: InstallPlan,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct InstallPlan {
    #[serde(rename = "type", default, skip_serializing_if = "String::is_empty")]
    pub install_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
}

impl InstallPlan {
    fn is_empty(&self) -> bool {
        self.install_type.is_empty() && self.artifacts.is_empty()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Artifact {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub goos: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub goarch: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sha256: String,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub size: i64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Platform {
    pub goos: String,
    pub goarch: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub assets: Vec<ReleaseAsset>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReleaseAsset {
    #[serde(rename = "url")]
    pub api_url: String,
    pub name: String,
    pub browser_download_url: String,
}

pub struct InstallOptions {
    pub plugins_dir: PathBuf,
    pub goos: String,
    pub goarch: String,
    pub plugin_loaded: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub before_write: Option<Arc<dyn Fn() -> Result<()> + Send + Sync>>,
}

impl Default for InstallOptions {
    fn default() -> Self {
        Self {
            plugins_dir: PathBuf::new(),
            goos: String::new(),
            goarch: String::new(),
            plugin_loaded: None,
            before_write: None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct InstallResult {
    pub id: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub release_tag: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub install_type: String,
    pub path: PathBuf,
    pub overwritten: bool,
    pub skipped: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Manifest {
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub schema_version: i32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub author: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub version: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub release_tag: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub repository: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub logo: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub homepage: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub license: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_url: String,
    #[serde(default, skip_serializing_if = "InstallPlan::is_empty")]
    pub install: InstallPlan,
}

impl Manifest {
    pub fn plugin(&self) -> Plugin {
        Plugin {
            id: self.id.trim().to_owned(),
            name: self.name.trim().to_owned(),
            description: self.description.trim().to_owned(),
            author: self.author.trim().to_owned(),
            version: self.version.trim().to_owned(),
            repository: self.repository.trim().to_owned(),
            logo: self.logo.trim().to_owned(),
            homepage: self.homepage.trim().to_owned(),
            license: self.license.trim().to_owned(),
            tags: self.tags.clone(),
            install: normalize_install_plan(self.install.clone()),
            ..Plugin::default()
        }
    }

    pub fn install_type(&self) -> &str {
        let install_type = self.install.install_type.trim();
        if install_type.is_empty() {
            INSTALL_TYPE_GITHUB_RELEASE
        } else {
            install_type
        }
    }

    pub fn validate(&self) -> Result<()> {
        let version = self.version.trim();
        if version.is_empty() {
            return Err(PluginStoreError::new("missing required field version"));
        }
        if !valid_plugin_version(&normalize_version(version)) {
            return Err(PluginStoreError::new(format!(
                "invalid plugin version {:?}",
                self.version
            )));
        }
        match self.install_type() {
            INSTALL_TYPE_DIRECT => {
                if self.schema_version != 0 && self.schema_version != SCHEMA_VERSION_V2 {
                    return Err(PluginStoreError::new(format!(
                        "unsupported schema-version {}",
                        self.schema_version
                    )));
                }
                validate_plugin_id(&self.id)?;
                let mut plan = normalize_install_plan(self.install.clone());
                plan.install_type = INSTALL_TYPE_DIRECT.to_owned();
                if !plan.artifacts.is_empty() {
                    validate_install_plan(&plan)?;
                    return validate_pinned_artifact_urls(&plan.artifacts);
                }
                validate_manifest_source_url(&self.source_url)
            }
            INSTALL_TYPE_GITHUB_RELEASE => {
                let release_tag = self.release_tag.trim();
                if release_tag.is_empty() {
                    return Err(PluginStoreError::new("missing required field release-tag"));
                }
                let mut plugin = self.plugin();
                plugin.install = InstallPlan {
                    install_type: INSTALL_TYPE_GITHUB_RELEASE.to_owned(),
                    artifacts: Vec::new(),
                };
                validate_plugin(&plugin)?;
                let release_version = release_version(&Release {
                    tag_name: release_tag.to_owned(),
                    assets: Vec::new(),
                })?;
                if release_version != normalize_version(version) {
                    return Err(PluginStoreError::new(format!(
                        "release-tag {release_tag:?} resolves version {release_version:?}, want {:?}",
                        normalize_version(version)
                    )));
                }
                Ok(())
            }
            other => Err(PluginStoreError::new(format!(
                "unsupported install type {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Secret(Vec<u8>);

impl Secret {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn expose(&self) -> &[u8] {
        &self.0
    }

    pub fn clear(&mut self) {
        self.0.zeroize();
        self.0.clear();
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Secret([REDACTED])")
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct AuthConfig {
    #[serde(rename = "match", default)]
    pub match_url: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub apply_to: Vec<String>,
    #[serde(rename = "type", default)]
    pub auth_type: String,
    #[serde(default)]
    pub token_env: String,
    #[serde(default)]
    pub username_env: String,
    #[serde(default)]
    pub password_env: String,
    #[serde(default)]
    pub header_name: String,
    #[serde(default)]
    pub header_value_env: String,
    #[serde(default)]
    pub allow_insecure: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolvedAuthConfig {
    #[serde(rename = "match", default)]
    pub match_url: String,
    #[serde(default)]
    pub apply_to: Vec<String>,
    #[serde(rename = "type", default)]
    pub auth_type: String,
    #[serde(default)]
    pub token: Secret,
    #[serde(default)]
    pub username: Secret,
    #[serde(default)]
    pub password: Secret,
    #[serde(default)]
    pub header_name: String,
    #[serde(default)]
    pub header_value: Secret,
}

impl ResolvedAuthConfig {
    pub fn clear(&mut self) {
        self.token.clear();
        self.username.clear();
        self.password.clear();
        self.header_value.clear();
        self.apply_to.clear();
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginSyncRequest {
    pub schema_version: i32,
    pub goos: String,
    pub goarch: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub installed_versions: HashMap<String, String>,
}

impl PluginSyncRequest {
    pub fn clear(&mut self) {
        self.installed_versions.clear();
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginSyncItem {
    pub manifest: Manifest,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth: Vec<ResolvedAuthConfig>,
}

impl PluginSyncItem {
    pub fn clear(&mut self) {
        clear_resolved_auth_configs(&mut self.auth);
        self.auth.clear();
        self.manifest = Manifest::default();
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginSyncResponse {
    pub schema_version: i32,
    pub expires_at: Option<DateTime<Utc>>,
    pub items: Vec<PluginSyncItem>,
}

impl PluginSyncResponse {
    pub fn validate(&self, now: DateTime<Utc>) -> Result<()> {
        if self.schema_version != PLUGIN_SYNC_SCHEMA_VERSION {
            return Err(PluginStoreError::new(format!(
                "unsupported plugin sync schema_version {}",
                self.schema_version
            )));
        }
        let expires_at = self
            .expires_at
            .ok_or_else(|| PluginStoreError::new("plugin sync response missing expires_at"))?;
        if now >= expires_at {
            return Err(PluginStoreError::new("plugin sync response expired"));
        }
        let mut seen = HashSet::new();
        for (index, item) in self.items.iter().enumerate() {
            item.manifest.validate().map_err(|error| {
                PluginStoreError::new(format!("plugin sync item {index}: {error}"))
            })?;
            validate_plugin_sync_manifest_urls(&item.manifest).map_err(|error| {
                PluginStoreError::new(format!("plugin sync item {index}: {error}"))
            })?;
            let id = item.manifest.id.trim();
            if !seen.insert(id) {
                return Err(PluginStoreError::new(format!(
                    "plugin sync response contains duplicate plugin {id:?}"
                )));
            }
            for (auth_index, auth) in item.auth.iter().enumerate() {
                validate_resolved_auth_config(auth).map_err(|error| {
                    PluginStoreError::new(format!(
                        "plugin sync item {index} auth {auth_index}: {error}"
                    ))
                })?;
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        for item in &mut self.items {
            item.clear();
        }
        self.items.clear();
        self.expires_at = None;
        self.schema_version = 0;
    }
}

/// Explicit effect boundary for registry, release and installation I/O.
///
/// Implementations own HTTP redirect/auth policy and atomic filesystem writes.
/// The facade never falls back to a process-global client or plugin directory.
pub trait PluginStoreIo: Send + Sync {
    fn fetch_registry(&self, client: &Client) -> Result<Registry>;
    fn fetch_latest_release(&self, client: &Client, plugin: &Plugin) -> Result<Release>;
    fn fetch_release_by_tag(&self, client: &Client, plugin: &Plugin, tag: &str) -> Result<Release>;
    fn install(
        &self,
        client: &Client,
        plugin: &Plugin,
        options: &InstallOptions,
    ) -> Result<InstallResult>;
    fn install_version(
        &self,
        client: &Client,
        plugin: &Plugin,
        release_tag: &str,
        version: &str,
        options: &InstallOptions,
    ) -> Result<InstallResult>;
    fn install_manifest(
        &self,
        client: &Client,
        manifest: &Manifest,
        options: &InstallOptions,
    ) -> Result<InstallResult>;
}

pub struct Client {
    io: Arc<dyn PluginStoreIo>,
    registry_url: String,
    auth: Vec<AuthConfig>,
    resolved_auth: Vec<ResolvedAuthConfig>,
    resolved_auth_expires_at: Option<DateTime<Utc>>,
}

impl Client {
    pub fn new(io: Arc<dyn PluginStoreIo>, registry_url: impl Into<String>) -> Self {
        Self {
            io,
            registry_url: registry_url.into().trim().to_owned(),
            auth: Vec::new(),
            resolved_auth: Vec::new(),
            resolved_auth_expires_at: None,
        }
    }

    pub fn with_auth(
        io: Arc<dyn PluginStoreIo>,
        registry_url: impl Into<String>,
        auth: Vec<AuthConfig>,
    ) -> Self {
        let mut client = Self::new(io, registry_url);
        client.auth = normalize_auth_configs(auth);
        client
    }

    pub fn with_resolved_auth(
        io: Arc<dyn PluginStoreIo>,
        registry_url: impl Into<String>,
        auth: Vec<ResolvedAuthConfig>,
    ) -> Self {
        Self::with_resolved_auth_expiry(io, registry_url, auth, None)
    }

    pub fn with_resolved_auth_expiry(
        io: Arc<dyn PluginStoreIo>,
        registry_url: impl Into<String>,
        auth: Vec<ResolvedAuthConfig>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            io,
            registry_url: registry_url.into().trim().to_owned(),
            auth: Vec::new(),
            resolved_auth: auth,
            resolved_auth_expires_at: expires_at,
        }
    }

    pub fn registry_url(&self) -> &str {
        if self.registry_url.is_empty() {
            DEFAULT_REGISTRY_URL
        } else {
            &self.registry_url
        }
    }

    pub fn auth(&self) -> &[AuthConfig] {
        &self.auth
    }

    pub fn resolved_auth(&self) -> &[ResolvedAuthConfig] {
        &self.resolved_auth
    }

    pub fn resolved_auth_expires_at(&self) -> Option<DateTime<Utc>> {
        self.resolved_auth_expires_at
    }

    pub fn clear_auth(&mut self) {
        clear_resolved_auth_configs(&mut self.resolved_auth);
        self.resolved_auth.clear();
        self.resolved_auth_expires_at = None;
    }

    pub fn fetch_registry(&self) -> Result<Registry> {
        self.io.fetch_registry(self)
    }

    pub fn fetch_latest_release(&self, plugin: &Plugin) -> Result<Release> {
        self.io.fetch_latest_release(self, plugin)
    }

    pub fn fetch_release_by_tag(&self, plugin: &Plugin, tag: &str) -> Result<Release> {
        self.io.fetch_release_by_tag(self, plugin, tag)
    }

    pub fn install(&self, plugin: &Plugin, options: &InstallOptions) -> Result<InstallResult> {
        self.io.install(self, plugin, options)
    }

    pub fn install_version(
        &self,
        plugin: &Plugin,
        release_tag: &str,
        version: &str,
        options: &InstallOptions,
    ) -> Result<InstallResult> {
        self.io
            .install_version(self, plugin, release_tag, version, options)
    }

    pub fn install_manifest(
        &self,
        manifest: &Manifest,
        options: &InstallOptions,
    ) -> Result<InstallResult> {
        self.io.install_manifest(self, manifest, options)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.clear_auth();
    }
}

pub fn default_source() -> Source {
    Source {
        id: DEFAULT_SOURCE_ID.to_owned(),
        name: DEFAULT_SOURCE_NAME.to_owned(),
        url: DEFAULT_REGISTRY_URL.to_owned(),
    }
}

pub fn normalize_sources(registry_urls: &[String]) -> Result<Vec<Source>> {
    let mut output = vec![default_source()];
    let mut seen_ids = HashMap::from([(DEFAULT_SOURCE_ID.to_owned(), DEFAULT_REGISTRY_URL)]);
    let mut seen_urls = HashSet::from([DEFAULT_REGISTRY_URL]);
    for registry_url in registry_urls {
        let registry_url = registry_url.trim();
        if registry_url.is_empty() || !seen_urls.insert(registry_url) {
            continue;
        }
        let id = source_id(registry_url);
        if let Some(existing) = seen_ids.get(&id) {
            return Err(PluginStoreError::new(format!(
                "plugin store source id collision for {existing:?} and {registry_url:?}"
            )));
        }
        seen_ids.insert(id.clone(), registry_url);
        output.push(Source {
            id,
            name: source_name(registry_url),
            url: registry_url.to_owned(),
        });
    }
    Ok(output)
}

pub fn source_id(registry_url: &str) -> String {
    let digest = Sha256::digest(registry_url.trim().as_bytes());
    format!("source-{}", hex_lower(&digest)[..12].to_owned())
}

fn source_name(registry_url: &str) -> String {
    Url::parse(registry_url.trim())
        .ok()
        .and_then(|url| url.host_str().map(ToOwned::to_owned))
        .filter(|host| !host.trim().is_empty())
        .unwrap_or_else(|| registry_url.trim().to_owned())
}

pub fn validate_plugin(plugin: &Plugin) -> Result<()> {
    for (field, value) in [
        ("id", plugin.id.as_str()),
        ("name", plugin.name.as_str()),
        ("description", plugin.description.as_str()),
        ("author", plugin.author.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(PluginStoreError::new(format!(
                "missing required field {field}"
            )));
        }
    }
    validate_plugin_id(&plugin.id)?;
    if !plugin.version.trim().is_empty() && !valid_plugin_version(plugin.version.trim()) {
        return Err(PluginStoreError::new(format!(
            "invalid plugin version {:?}",
            plugin.version
        )));
    }
    match plugin_install_type(plugin) {
        INSTALL_TYPE_GITHUB_RELEASE => {
            if plugin.repository.trim().is_empty() {
                return Err(PluginStoreError::new("missing required field repository"));
            }
            github_repository_parts(&plugin.repository).map(|_| ())
        }
        INSTALL_TYPE_DIRECT => {
            if plugin.version.trim().is_empty() {
                return Err(PluginStoreError::new("missing required field version"));
            }
            validate_install_plan(&plugin.install)?;
            validate_plugin_versions(plugin)
        }
        other => Err(PluginStoreError::new(format!(
            "unsupported install type {other:?}"
        ))),
    }
}

fn validate_plugin_versions(plugin: &Plugin) -> Result<()> {
    let expected_type = plugin_install_type(plugin);
    let mut seen = HashSet::new();
    for (index, version) in plugin.versions.iter().enumerate() {
        let normalized = normalize_version(&version.version);
        if !valid_plugin_version(&normalized) {
            return Err(PluginStoreError::new(format!(
                "versions[{index}]: invalid plugin version {normalized:?}"
            )));
        }
        if !seen.insert(normalized.clone()) {
            return Err(PluginStoreError::new(format!(
                "versions[{index}]: duplicate plugin version {normalized:?}"
            )));
        }
        let mut plan = normalize_install_plan(version.install.clone());
        if plan.install_type.is_empty() {
            plan.install_type = expected_type.to_owned();
        }
        if plan.install_type != expected_type {
            return Err(PluginStoreError::new(format!(
                "versions[{index}]: install type {:?} does not match plugin install type {expected_type:?}",
                plan.install_type
            )));
        }
        validate_install_plan(&plan)
            .map_err(|error| PluginStoreError::new(format!("versions[{index}]: {error}")))?;
    }
    Ok(())
}

pub fn plugin_install_type(plugin: &Plugin) -> &str {
    let install_type = plugin.install.install_type.trim();
    if install_type.is_empty() {
        INSTALL_TYPE_GITHUB_RELEASE
    } else {
        install_type
    }
}

pub fn plugin_platforms(plugin: &Plugin) -> Vec<Platform> {
    if plugin_install_type(plugin) != INSTALL_TYPE_DIRECT {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    plugin_artifacts(plugin)
        .into_iter()
        .filter_map(|artifact| {
            let platform = Platform {
                goos: artifact.goos,
                goarch: artifact.goarch,
            };
            if platform.goos.is_empty()
                || platform.goarch.is_empty()
                || !seen.insert(platform.clone())
            {
                None
            } else {
                Some(platform)
            }
        })
        .collect()
}

pub fn plugin_artifacts(plugin: &Plugin) -> Vec<Artifact> {
    if plugin_install_type(plugin) != INSTALL_TYPE_DIRECT {
        return Vec::new();
    }
    let mut artifacts = normalize_install_plan(plugin.install.clone()).artifacts;
    for version in &plugin.versions {
        artifacts.extend(normalize_install_plan(version.install.clone()).artifacts);
    }
    artifacts
}

pub fn select_artifact(plan: &InstallPlan, goos: &str, goarch: &str) -> Result<Artifact> {
    let plan = normalize_install_plan(plan.clone());
    let goos = normalize_goos(goos);
    let goarch = normalize_goarch(goarch);
    if plan.install_type != INSTALL_TYPE_DIRECT {
        return Err(PluginStoreError::new(format!(
            "install type {:?} is not direct",
            plan.install_type
        )));
    }
    plan.artifacts
        .into_iter()
        .find(|artifact| artifact.goos == goos && artifact.goarch == goarch)
        .ok_or_else(|| PluginStoreError::new(format!("artifact not found for {goos}/{goarch}")))
}

pub fn github_repository_parts(repository: &str) -> Result<(String, String)> {
    let parsed = Url::parse(repository.trim())
        .map_err(|error| PluginStoreError::new(format!("invalid repository URL: {error}")))?;
    if parsed.scheme() != "https"
        || parsed.host_str() != Some("github.com")
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(github_repository_shape_error());
    }
    let segments: Vec<_> = parsed
        .path_segments()
        .map(|segments| segments.filter(|segment| !segment.is_empty()).collect())
        .unwrap_or_default();
    if segments.len() != 2 || segments[1].ends_with(".git") {
        return Err(github_repository_shape_error());
    }
    Ok((segments[0].to_owned(), segments[1].to_owned()))
}

fn github_repository_shape_error() -> PluginStoreError {
    PluginStoreError::new("repository must be https://github.com/{owner}/{repo}")
}

pub fn normalize_auth_configs(auth: Vec<AuthConfig>) -> Vec<AuthConfig> {
    let mut output = Vec::with_capacity(auth.len());
    for mut item in auth {
        item.match_url = item.match_url.trim().to_owned();
        item.auth_type = item.auth_type.trim().to_lowercase();
        item.token_env = item.token_env.trim().to_owned();
        item.username_env = item.username_env.trim().to_owned();
        item.password_env = item.password_env.trim().to_owned();
        item.header_name = item.header_name.trim().to_owned();
        item.header_value_env = item.header_value_env.trim().to_owned();
        if item.auth_type.is_empty() {
            item.auth_type = AUTH_TYPE_NONE.to_owned();
        }
        if item.match_url.is_empty() {
            continue;
        }
        let mut seen = HashSet::new();
        item.apply_to = item
            .apply_to
            .into_iter()
            .map(|value| value.trim().to_lowercase())
            .filter(|value| !value.is_empty() && seen.insert(value.clone()))
            .collect();
        output.push(item);
    }
    output
}

pub fn clear_resolved_auth_configs(auth: &mut [ResolvedAuthConfig]) {
    for item in auth {
        item.clear();
    }
}

pub fn resolved_auth_for_request(
    auth: &[ResolvedAuthConfig],
    request_url: &str,
    kind: &str,
) -> Option<ResolvedAuthConfig> {
    auth.iter()
        .find(|item| {
            url_matches_auth_rule(request_url, &item.match_url) && applies_to(&item.apply_to, kind)
        })
        .cloned()
}

pub fn validate_resolved_auth_config(item: &ResolvedAuthConfig) -> Result<()> {
    let parsed = Url::parse(item.match_url.trim())
        .map_err(|_| PluginStoreError::new("plugin store resolved auth match is invalid"))?;
    if parsed.scheme() != "https" {
        return Err(PluginStoreError::new(
            "plugin store resolved auth match must use https",
        ));
    }
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(PluginStoreError::new(
            "plugin store resolved auth match must not contain credentials, query, or fragment",
        ));
    }
    for kind in &item.apply_to {
        if !matches!(
            kind.trim().to_lowercase().as_str(),
            REQUEST_KIND_REGISTRY | REQUEST_KIND_METADATA | REQUEST_KIND_ARTIFACT
        ) {
            return Err(PluginStoreError::new(format!(
                "plugin store resolved auth has unsupported apply_to {kind:?}"
            )));
        }
    }
    match item.auth_type.trim().to_lowercase().as_str() {
        "" | AUTH_TYPE_NONE => Ok(()),
        AUTH_TYPE_BEARER | AUTH_TYPE_GITHUB_TOKEN if item.token.expose().is_empty() => Err(
            PluginStoreError::new("plugin store resolved auth token is empty"),
        ),
        AUTH_TYPE_BASIC
            if item.username.expose().is_empty() || item.password.expose().is_empty() =>
        {
            Err(PluginStoreError::new(
                "plugin store resolved basic auth is incomplete",
            ))
        }
        AUTH_TYPE_HEADER
            if item.header_name.trim().is_empty()
                || item.header_name.contains(['\r', '\n', ':']) =>
        {
            Err(PluginStoreError::new(
                "plugin store resolved auth header name is invalid",
            ))
        }
        AUTH_TYPE_HEADER
            if item.header_value.expose().is_empty()
                || item.header_value.expose().contains(&b'\r')
                || item.header_value.expose().contains(&b'\n') =>
        {
            Err(PluginStoreError::new(
                "plugin store resolved auth header value is invalid",
            ))
        }
        AUTH_TYPE_BEARER | AUTH_TYPE_GITHUB_TOKEN | AUTH_TYPE_BASIC | AUTH_TYPE_HEADER => Ok(()),
        other => Err(PluginStoreError::new(format!(
            "unsupported plugin store resolved auth type {other:?}"
        ))),
    }
}

/// Reports whether a matching auth descriptor is structurally configured.
///
/// Unlike upstream, this deliberately does not read named process environment
/// variables. Runtime credentials must arrive as `ResolvedAuthConfig` through
/// CTOX's typed secret-store path.
pub fn auth_configured(auth: &[AuthConfig], request_url: &str, kind: &str) -> bool {
    normalize_auth_configs(auth.to_vec())
        .into_iter()
        .any(|item| {
            url_matches_auth_rule(request_url, &item.match_url)
                && applies_to(&item.apply_to, kind)
                && match item.auth_type.as_str() {
                    AUTH_TYPE_BEARER | AUTH_TYPE_GITHUB_TOKEN => !item.token_env.is_empty(),
                    AUTH_TYPE_BASIC => {
                        !item.username_env.is_empty() && !item.password_env.is_empty()
                    }
                    AUTH_TYPE_HEADER => {
                        !item.header_name.is_empty() && !item.header_value_env.is_empty()
                    }
                    _ => false,
                }
        })
}

pub fn plugin_auth_configured(source: &Source, plugin: &Plugin, auth: &[AuthConfig]) -> bool {
    if auth_configured(auth, &source.url, REQUEST_KIND_REGISTRY) {
        return true;
    }
    match plugin_install_type(plugin) {
        INSTALL_TYPE_DIRECT => plugin_artifacts(plugin)
            .iter()
            .any(|artifact| auth_configured(auth, &artifact.url, REQUEST_KIND_ARTIFACT)),
        INSTALL_TYPE_GITHUB_RELEASE => github_repository_parts(&plugin.repository)
            .ok()
            .map(|(owner, repo)| {
                let base = format!("https://api.github.com/repos/{owner}/{repo}/releases/");
                auth_configured(auth, &(base.clone() + "latest"), REQUEST_KIND_METADATA)
                    || auth_configured(auth, &(base + "tags/"), REQUEST_KIND_METADATA)
            })
            .unwrap_or(false),
        _ => false,
    }
}

pub fn update_available(installed: &str, latest: &str) -> bool {
    let installed = normalize_version(installed);
    let latest = normalize_version(latest);
    if installed.is_empty() || latest.is_empty() || installed == latest {
        return false;
    }
    compare_versions(&installed, &latest)
        .map(|ordering| ordering.is_lt())
        .unwrap_or(true)
}

pub fn release_version(release: &Release) -> Result<String> {
    let version = normalize_version(&release.tag_name);
    if !valid_plugin_version(&version) {
        return Err(PluginStoreError::new(format!(
            "invalid release tag {:?}",
            release.tag_name
        )));
    }
    Ok(version)
}

pub fn manifest_from_release(
    source: &Source,
    plugin: &Plugin,
    release: &Release,
) -> Result<Manifest> {
    let version = release_version(release)?;
    Ok(manifest_from_plugin_fields(
        source,
        plugin,
        Manifest {
            version,
            release_tag: release.tag_name.trim().to_owned(),
            repository: plugin.repository.trim().to_owned(),
            install: InstallPlan {
                install_type: INSTALL_TYPE_GITHUB_RELEASE.to_owned(),
                artifacts: Vec::new(),
            },
            ..Manifest::default()
        },
    ))
}

pub fn manifest_from_plugin(source: &Source, plugin: &Plugin) -> Result<Manifest> {
    validate_plugin(plugin)?;
    match plugin_install_type(plugin) {
        INSTALL_TYPE_DIRECT => {
            let manifest = manifest_from_plugin_fields(
                source,
                plugin,
                Manifest {
                    schema_version: SCHEMA_VERSION_V2,
                    version: plugin.version.trim().to_owned(),
                    install: normalize_install_plan(plugin.install.clone()),
                    ..Manifest::default()
                },
            );
            manifest.validate()?;
            Ok(manifest)
        }
        INSTALL_TYPE_GITHUB_RELEASE => Err(PluginStoreError::new(
            "github-release manifest requires a resolved release",
        )),
        other => Err(PluginStoreError::new(format!(
            "unsupported install type {other:?}"
        ))),
    }
}

fn manifest_from_plugin_fields(source: &Source, plugin: &Plugin, mut base: Manifest) -> Manifest {
    base.id = plugin.id.trim().to_owned();
    base.name = plugin.name.trim().to_owned();
    base.description = plugin.description.trim().to_owned();
    base.author = plugin.author.trim().to_owned();
    base.logo = plugin.logo.trim().to_owned();
    base.homepage = plugin.homepage.trim().to_owned();
    base.license = plugin.license.trim().to_owned();
    base.tags = plugin.tags.clone();
    base.source_id = source.id.trim().to_owned();
    base.source_name = source.name.trim().to_owned();
    base.source_url = source.url.trim().to_owned();
    base
}

fn normalize_install_plan(mut plan: InstallPlan) -> InstallPlan {
    plan.install_type = plan.install_type.trim().to_lowercase();
    for artifact in &mut plan.artifacts {
        artifact.goos = normalize_goos(&artifact.goos);
        artifact.goarch = normalize_goarch(&artifact.goarch);
        artifact.url = artifact.url.trim().to_owned();
        artifact.sha256 = artifact.sha256.trim().to_lowercase();
    }
    plan
}

fn validate_install_plan(plan: &InstallPlan) -> Result<()> {
    let plan = normalize_install_plan(plan.clone());
    if plan.install_type.is_empty() {
        return Err(PluginStoreError::new("missing install type"));
    }
    if !matches!(
        plan.install_type.as_str(),
        INSTALL_TYPE_DIRECT | INSTALL_TYPE_GITHUB_RELEASE
    ) {
        return Err(PluginStoreError::new(format!(
            "unsupported install type {:?}",
            plan.install_type
        )));
    }
    if plan.install_type == INSTALL_TYPE_DIRECT {
        if plan.artifacts.is_empty() {
            return Err(PluginStoreError::new(
                "direct install requires at least one artifact",
            ));
        }
        for (index, artifact) in plan.artifacts.iter().enumerate() {
            validate_artifact(artifact)
                .map_err(|error| PluginStoreError::new(format!("artifacts[{index}]: {error}")))?;
        }
    }
    Ok(())
}

fn validate_artifact(artifact: &Artifact) -> Result<()> {
    let artifact = normalize_install_plan(InstallPlan {
        install_type: INSTALL_TYPE_DIRECT.to_owned(),
        artifacts: vec![artifact.clone()],
    })
    .artifacts
    .remove(0);
    if artifact.goos.is_empty() {
        return Err(PluginStoreError::new("missing goos"));
    }
    if artifact.goarch.is_empty() {
        return Err(PluginStoreError::new("missing goarch"));
    }
    let parsed =
        Url::parse(&artifact.url).map_err(|_| PluginStoreError::new("invalid artifact url"))?;
    if !matches!(parsed.scheme(), "https" | "http") {
        return Err(PluginStoreError::new("artifact url must use http or https"));
    }
    if has_sensitive_query_parameter(&parsed) {
        return Err(PluginStoreError::new(
            "artifact url contains sensitive query parameter",
        ));
    }
    if artifact.sha256.len() != 64 {
        return Err(PluginStoreError::new(if artifact.sha256.is_empty() {
            "missing sha256"
        } else {
            "invalid sha256 length"
        }));
    }
    if !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PluginStoreError::new("invalid sha256"));
    }
    if artifact.size < 0 {
        return Err(PluginStoreError::new("invalid size"));
    }
    Ok(())
}

fn validate_pinned_artifact_urls(artifacts: &[Artifact]) -> Result<()> {
    for (index, artifact) in artifacts.iter().enumerate() {
        let parsed = Url::parse(artifact.url.trim()).map_err(|_| {
            PluginStoreError::new(format!("artifacts[{index}]: invalid artifact url"))
        })?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(PluginStoreError::new(format!(
                "artifacts[{index}]: pinned artifact url must not contain credentials"
            )));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(PluginStoreError::new(format!(
                "artifacts[{index}]: pinned artifact url must not contain query or fragment"
            )));
        }
    }
    Ok(())
}

fn validate_manifest_source_url(source_url: &str) -> Result<()> {
    if source_url.trim().is_empty() {
        return Err(PluginStoreError::new("missing required field source-url"));
    }
    let parsed =
        Url::parse(source_url.trim()).map_err(|_| PluginStoreError::new("invalid source-url"))?;
    if !matches!(parsed.scheme(), "https" | "http") {
        return Err(PluginStoreError::new("source-url must use http or https"));
    }
    if has_sensitive_query_parameter(&parsed) {
        return Err(PluginStoreError::new(
            "source-url contains sensitive query parameter",
        ));
    }
    Ok(())
}

fn validate_plugin_sync_manifest_urls(manifest: &Manifest) -> Result<()> {
    if manifest.install_type() != INSTALL_TYPE_DIRECT {
        return Ok(());
    }
    let plan = normalize_install_plan(manifest.install.clone());
    if plan.artifacts.is_empty() {
        return Err(PluginStoreError::new(
            "direct plugin sync manifest requires pinned artifacts",
        ));
    }
    for (index, artifact) in plan.artifacts.iter().enumerate() {
        if Url::parse(artifact.url.trim()).map(|url| url.scheme().eq_ignore_ascii_case("https"))
            != Ok(true)
        {
            return Err(PluginStoreError::new(format!(
                "direct plugin sync artifact {index} must use https"
            )));
        }
    }
    Ok(())
}

fn normalize_goos(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "mac" | "macos" | "osx" => "darwin".to_owned(),
        value => value.to_owned(),
    }
}

fn normalize_goarch(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "x64" | "x86_64" => "amd64".to_owned(),
        "aarch64" => "arm64".to_owned(),
        value => value.to_owned(),
    }
}

fn normalize_version(version: &str) -> String {
    let version = version.trim();
    version
        .strip_prefix(['v', 'V'])
        .filter(|_| version.len() > 1)
        .unwrap_or(version)
        .to_owned()
}

fn valid_plugin_version(version: &str) -> bool {
    !version.is_empty()
        && !version.starts_with('v')
        && version.as_bytes()[0].is_ascii_digit()
        && version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".+-".contains(&byte))
}

fn validate_plugin_id(id: &str) -> Result<()> {
    let id = id.trim();
    let valid = (1..=128).contains(&id.len())
        && id.as_bytes()[0].is_ascii_alphanumeric()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte));
    if valid {
        Ok(())
    } else if id.is_empty() {
        Err(PluginStoreError::new("missing required field id"))
    } else {
        Err(PluginStoreError::new(format!("invalid plugin id {id:?}")))
    }
}

fn has_sensitive_query_parameter(url: &Url) -> bool {
    url.query_pairs().any(|(key, _)| {
        matches!(
            key.trim().to_lowercase().as_str(),
            "token" | "access_token" | "access_key" | "secret" | "secret_key" | "api_key"
        )
    })
}

fn url_matches_auth_rule(request_url: &str, match_url: &str) -> bool {
    let (Ok(request), Ok(rule)) = (Url::parse(request_url.trim()), Url::parse(match_url.trim()))
    else {
        return false;
    };
    if !request.scheme().eq_ignore_ascii_case(rule.scheme())
        || request.host_str().map(str::to_lowercase) != rule.host_str().map(str::to_lowercase)
    {
        return false;
    }
    let rule_path = rule.path();
    rule_path.is_empty()
        || rule_path == "/"
        || request.path() == rule_path
        || rule_path
            .strip_suffix('/')
            .map(|_| request.path().starts_with(rule_path))
            .unwrap_or_else(|| request.path().starts_with(&(rule_path.to_owned() + "/")))
}

fn applies_to(values: &[String], kind: &str) -> bool {
    values.is_empty()
        || values
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case(kind.trim()))
}

fn compare_versions(a: &str, b: &str) -> Option<std::cmp::Ordering> {
    let a: Option<Vec<i64>> = a.split('.').map(|part| part.parse().ok()).collect();
    let b: Option<Vec<i64>> = b.split('.').map(|part| part.parse().ok()).collect();
    let (a, b) = (a?, b?);
    let length = a.len().max(b.len());
    Some(
        (0..length)
            .map(|index| *a.get(index).unwrap_or(&0))
            .cmp((0..length).map(|index| *b.get(index).unwrap_or(&0))),
    )
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

const fn is_zero_i32(value: &i32) -> bool {
    *value == 0
}

const fn is_zero_i64(value: &i64) -> bool {
    *value == 0
}
