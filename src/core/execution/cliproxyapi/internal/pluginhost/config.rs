// ref: internal/pluginhost/config.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: consumes normalized typed CTOX config; never resolves ambient paths
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_yaml::{Mapping, Value};

use crate::internal::config::config_normalization::{PluginInstanceConfig, PluginsConfig};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    pub items: BTreeMap<String, RuntimeItemConfig>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeItemConfig {
    pub id: String,
    pub enabled: bool,
    pub priority: i32,
    pub version: Option<String>,
    pub config_yaml: Vec<u8>,
}

pub fn runtime_config_from_config(config: &PluginsConfig) -> Result<RuntimeConfig, ConfigError> {
    if !config.enabled {
        return Ok(RuntimeConfig {
            enabled: false,
            directory: PathBuf::from("plugins"),
            items: BTreeMap::new(),
        });
    }
    let directory = PathBuf::from(config.dir.trim());
    if !directory.is_absolute() {
        return Err(ConfigError::UnresolvedPluginDirectory);
    }
    let mut items = BTreeMap::new();
    for (id, item) in &config.configs {
        let id = id.trim();
        if id.is_empty() {
            return Err(ConfigError::InvalidPluginId);
        }
        items.insert(id.to_owned(), runtime_item(id, item)?);
    }
    Ok(RuntimeConfig {
        enabled: true,
        directory,
        items,
    })
}

fn runtime_item(id: &str, item: &PluginInstanceConfig) -> Result<RuntimeItemConfig, ConfigError> {
    let enabled = item.enabled.unwrap_or(false);
    let mut raw = match item.raw.clone() {
        Value::Mapping(mapping) => mapping,
        other => return serialize_item(id, enabled, item.priority, None, other),
    };
    raw.entry(Value::String("enabled".to_owned()))
        .or_insert(Value::Bool(enabled));
    raw.entry(Value::String("priority".to_owned()))
        .or_insert(Value::Number(item.priority.into()));
    let version = desired_version(&raw);
    serialize_item(id, enabled, item.priority, version, Value::Mapping(raw))
}

fn serialize_item(
    id: &str,
    enabled: bool,
    priority: i32,
    version: Option<String>,
    raw: Value,
) -> Result<RuntimeItemConfig, ConfigError> {
    let mut config_yaml = serde_yaml::to_string(&raw)
        .map_err(|_| ConfigError::InvalidPluginConfig)?
        .into_bytes();
    if config_yaml.is_empty() {
        config_yaml = format!("enabled: {enabled}\npriority: {priority}\n").into_bytes();
    }
    Ok(RuntimeItemConfig {
        id: id.to_owned(),
        enabled,
        priority,
        version,
        config_yaml,
    })
}

fn desired_version(mapping: &Mapping) -> Option<String> {
    let store = mapping
        .get(Value::String("store".to_owned()))?
        .as_mapping()?;
    ["version", "release-tag"].iter().find_map(|key| {
        let raw = store
            .get(Value::String((*key).to_owned()))?
            .as_str()?
            .trim();
        normalize_version(raw)
    })
}

pub fn normalize_version(raw: &str) -> Option<String> {
    let version = raw.trim().trim_start_matches(['v', 'V']);
    if version.is_empty()
        || !version.starts_with(|character: char| character.is_ascii_digit())
        || !version
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || ".+-".contains(character))
    {
        return None;
    }
    Some(version.to_owned())
}

pub fn desired_plugin_versions(config: &RuntimeConfig) -> BTreeMap<String, String> {
    config
        .items
        .iter()
        .filter_map(|(id, item)| item.version.clone().map(|version| (id.clone(), version)))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    UnresolvedPluginDirectory,
    InvalidPluginId,
    InvalidPluginConfig,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::UnresolvedPluginDirectory => "plugin directory must be resolved by typed config",
            Self::InvalidPluginId => "plugin identifier is invalid",
            Self::InvalidPluginConfig => "plugin configuration cannot be serialized",
        })
    }
}

impl std::error::Error for ConfigError {}
