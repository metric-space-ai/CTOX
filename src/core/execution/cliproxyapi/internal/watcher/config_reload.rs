// ref: internal/watcher/config_reload.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::synthesizer::context::ModelAlias;
use super::{WatcherDependencies, WatcherState};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRoute {
    pub name: String,
    #[serde(default)]
    pub alias: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub force_mapping: bool,
    #[serde(default)]
    pub image: bool,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub weight: Option<i64>,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub models: Vec<ModelRoute>,
    #[serde(default)]
    pub excluded_models: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiCompatibility {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub models: Vec<ModelRoute>,
    #[serde(default)]
    pub prompt_cache_key: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatcherConfig {
    #[serde(default)]
    pub providers: BTreeMap<String, Vec<ApiKeyConfig>>,
    #[serde(default)]
    pub openai_compatibility: Vec<OpenAiCompatibility>,
    #[serde(default)]
    pub oauth_excluded_models: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub oauth_model_aliases: BTreeMap<String, Vec<ModelAlias>>,
    #[serde(default)]
    pub settings: BTreeMap<String, serde_json::Value>,
}

pub trait ConfigDecoder: Send + Sync {
    fn decode(&self, bytes: &[u8]) -> io::Result<WatcherConfig>;
}
#[derive(Debug, Default)]
pub struct JsonConfigDecoder;
impl ConfigDecoder for JsonConfigDecoder {
    fn decode(&self, bytes: &[u8]) -> io::Result<WatcherConfig> {
        serde_json::from_slice(bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

pub(super) fn reload_config_if_changed(
    dependencies: &WatcherDependencies,
    path: &Path,
    state: &mut WatcherState,
) -> io::Result<bool> {
    let bytes = dependencies.filesystem.read(path)?;
    if bytes.is_empty() {
        return Ok(false);
    }
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if state.config_hash.as_deref() == Some(&hash) {
        return Ok(false);
    }
    let config = dependencies.config_decoder.decode(&bytes)?;
    state.config = config.clone();
    state.config_hash = Some(hash);
    dependencies.reload_sink.on_reload(&config);
    dependencies.persistence_sink.persist_config()?;
    Ok(true)
}
