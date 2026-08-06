// ref: internal/pluginstore/registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashSet;

use crate::sdk::pluginstore::{
    plugin_install_type, validate_plugin, InstallPlan, Plugin, Registry, Result,
    INSTALL_TYPE_DIRECT, SCHEMA_VERSION, SCHEMA_VERSION_V2,
};

use super::github::store_error;

pub(crate) fn parse_registry(data: &[u8]) -> Result<Registry> {
    let mut registry: Registry = serde_json::from_slice(data)
        .map_err(|error| store_error(format!("decode registry: {error}")))?;
    normalize_registry(&mut registry);
    validate_registry(&registry)?;
    Ok(registry)
}

pub(crate) fn validate_registry(registry: &Registry) -> Result<()> {
    if !matches!(registry.schema_version, SCHEMA_VERSION | SCHEMA_VERSION_V2) {
        return Err(store_error(format!(
            "unsupported schema_version {}",
            registry.schema_version
        )));
    }
    let mut seen = HashSet::new();
    for (index, plugin) in registry.plugins.iter().enumerate() {
        if registry.schema_version == SCHEMA_VERSION
            && plugin_install_type(plugin) == INSTALL_TYPE_DIRECT
        {
            return Err(store_error(format!(
                "plugins[{index}]: direct install requires schema_version {SCHEMA_VERSION_V2}"
            )));
        }
        validate_plugin(plugin)
            .map_err(|error| store_error(format!("plugins[{index}]: {error}")))?;
        if !seen.insert(plugin.id.trim()) {
            return Err(store_error(format!(
                "plugins[{index}]: duplicate plugin id {:?}",
                plugin.id.trim()
            )));
        }
    }
    Ok(())
}

fn normalize_registry(registry: &mut Registry) {
    for plugin in &mut registry.plugins {
        plugin.id = plugin.id.trim().to_owned();
        plugin.name = plugin.name.trim().to_owned();
        plugin.description = plugin.description.trim().to_owned();
        plugin.author = plugin.author.trim().to_owned();
        plugin.version = plugin.version.trim().to_owned();
        plugin.repository = plugin.repository.trim().to_owned();
        plugin.logo = plugin.logo.trim().to_owned();
        plugin.homepage = plugin.homepage.trim().to_owned();
        plugin.license = plugin.license.trim().to_owned();
        plugin
            .tags
            .iter_mut()
            .for_each(|tag| *tag = tag.trim().to_owned());
        normalize_plan(&mut plugin.install);
        for version in &mut plugin.versions {
            version.version = normalize_version(&version.version);
            normalize_plan(&mut version.install);
        }
    }
}

fn normalize_plan(plan: &mut InstallPlan) {
    plan.install_type = plan.install_type.trim().to_ascii_lowercase();
    for artifact in &mut plan.artifacts {
        artifact.goos = normalize_goos(&artifact.goos);
        artifact.goarch = normalize_goarch(&artifact.goarch);
        artifact.url = artifact.url.trim().to_owned();
        artifact.sha256 = artifact.sha256.trim().to_ascii_lowercase();
    }
}

pub(crate) fn normalize_goos(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "mac" | "macos" | "osx" => "darwin".to_owned(),
        value => value.to_owned(),
    }
}

pub(crate) fn normalize_goarch(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "x64" | "x86_64" => "amd64".to_owned(),
        "aarch64" => "arm64".to_owned(),
        value => value.to_owned(),
    }
}

pub(crate) fn normalize_version(value: &str) -> String {
    let value = value.trim();
    if value.len() > 1 && matches!(value.as_bytes()[0], b'v' | b'V') {
        value[1..].to_owned()
    } else {
        value.to_owned()
    }
}

pub(crate) fn valid_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    (1..=128).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(byte))
}

pub(crate) fn valid_version(version: &str) -> bool {
    let bytes = version.as_bytes();
    !bytes.is_empty()
        && bytes[0].is_ascii_digit()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || b".+-".contains(byte))
}

pub(crate) fn direct_plugin_version(mut plugin: Plugin, id: &str, version: &str) -> Result<Plugin> {
    let id = id.trim();
    let version = normalize_version(version);
    if normalize_version(&plugin.version) == version {
        plugin.version = version;
        plugin.install.install_type = INSTALL_TYPE_DIRECT.to_owned();
        validate_plugin(&plugin)?;
        return Ok(plugin);
    }
    let selected = plugin
        .versions
        .iter()
        .find(|candidate| normalize_version(&candidate.version) == version)
        .ok_or_else(|| {
            store_error(format!(
                "direct install plugin {id:?} version {version:?} not found in source"
            ))
        })?;
    plugin.version = version;
    plugin.install = selected.install.clone();
    if plugin.install.install_type.trim().is_empty() {
        plugin.install.install_type = INSTALL_TYPE_DIRECT.to_owned();
    }
    validate_plugin(&plugin)?;
    Ok(plugin)
}
