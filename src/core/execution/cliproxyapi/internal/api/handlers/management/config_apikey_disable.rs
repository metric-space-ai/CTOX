// ref: internal/api/handlers/management/config_apikey_disable.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::config_normalization::{CodexKey, ProviderCompatConfig};
use crate::internal::config::vertex_compat::{normalize_excluded_models, VertexCompatKey};
use crate::internal::watcher::synthesizer::helpers::StableIdGenerator;

const CONFIG_API_KEY_DISABLE_PATTERN: &str = "*";

#[must_use]
pub fn set_config_api_key_excluded_all(models: &[String], disable: bool) -> Vec<String> {
    let mut models = models.to_vec();
    if disable {
        if !models
            .iter()
            .any(|item| item.trim() == CONFIG_API_KEY_DISABLE_PATTERN)
        {
            models.push(CONFIG_API_KEY_DISABLE_PATTERN.into());
        }
    } else {
        models.retain(|item| item.trim() != CONFIG_API_KEY_DISABLE_PATTERN);
    }
    normalize_excluded_models(&mut models);
    models
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigApiKeyToggleError {
    EmptyAuthId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagementProviderKeyKind {
    Codex,
    Xai,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ManagementProviderKeyPatch {
    pub priority: Option<i32>,
    pub websockets: Option<bool>,
    pub disable_cooling: Option<bool>,
    pub alpha_search: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagementProviderKeyPatchError {
    IndexOutOfBounds,
    UnsupportedField,
}

pub fn patch_management_provider_key(
    config: &mut ProviderCompatConfig,
    kind: ManagementProviderKeyKind,
    index: usize,
    patch: ManagementProviderKeyPatch,
) -> Result<(), ManagementProviderKeyPatchError> {
    let entries = match kind {
        ManagementProviderKeyKind::Codex => &mut config.codex_api_key,
        ManagementProviderKeyKind::Xai => &mut config.xai_api_key,
    };
    let entry = entries
        .get_mut(index)
        .ok_or(ManagementProviderKeyPatchError::IndexOutOfBounds)?;
    if matches!(kind, ManagementProviderKeyKind::Xai) && patch.alpha_search.is_some() {
        return Err(ManagementProviderKeyPatchError::UnsupportedField);
    }
    if let Some(priority) = patch.priority {
        entry.priority = priority;
    }
    if let Some(websockets) = patch.websockets {
        entry.websockets = websockets;
    }
    if let Some(disable_cooling) = patch.disable_cooling {
        entry.disable_cooling = disable_cooling;
    }
    if let Some(alpha_search) = patch.alpha_search {
        entry.alpha_search = alpha_search;
    }
    Ok(())
}

/// Updates only the credential selected by its synthesized public ID. Secret
/// material is used inside the typed config authority and is never returned.
pub fn toggle_config_api_key_excluded_all(
    config: &mut ProviderCompatConfig,
    auth_id: &str,
    disable: bool,
) -> Result<bool, ConfigApiKeyToggleError> {
    let auth_id = auth_id.trim();
    if auth_id.is_empty() {
        return Err(ConfigApiKeyToggleError::EmptyAuthId);
    }
    let mut ids = StableIdGenerator::default();
    for (kind, entries) in [
        ("gemini:apikey", &mut config.gemini_api_key),
        (
            "gemini-interactions:apikey",
            &mut config.interactions_api_key,
        ),
        ("claude:apikey", &mut config.claude_api_key),
        ("codex:apikey", &mut config.codex_api_key),
        ("xai:apikey", &mut config.xai_api_key),
    ] {
        if toggle_codex_entries(entries, kind, auth_id, disable, &mut ids) {
            return Ok(true);
        }
    }
    Ok(toggle_vertex_entries(
        &mut config.vertex_api_key,
        auth_id,
        disable,
        &mut ids,
    ))
}

fn toggle_codex_entries(
    entries: &mut [CodexKey],
    kind: &str,
    auth_id: &str,
    disable: bool,
    ids: &mut StableIdGenerator,
) -> bool {
    for entry in entries {
        let (id, _) = ids.next(kind, &[&entry.api_key, &entry.base_url]);
        if id == auth_id {
            entry.excluded_models =
                set_config_api_key_excluded_all(&entry.excluded_models, disable);
            return true;
        }
    }
    false
}

fn toggle_vertex_entries(
    entries: &mut [VertexCompatKey],
    auth_id: &str,
    disable: bool,
    ids: &mut StableIdGenerator,
) -> bool {
    for entry in entries {
        let (id, _) = ids.next(
            "vertex:apikey",
            &[&entry.api_key, &entry.base_url, &entry.proxy_url],
        );
        if id == auth_id {
            entry.excluded_models =
                set_config_api_key_excluded_all(&entry.excluded_models, disable);
            return true;
        }
    }
    false
}
