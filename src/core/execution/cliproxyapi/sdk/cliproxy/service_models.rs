// ref: sdk/cliproxy/service_models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Instance-owned model registration and configuration transforms.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::internal::config::ValidatedRuntimeConfig;
use crate::internal::registry::{
    models_for_channel, RegistryModelInfo, RegistryThinkingSupport, StaticModelsCatalog,
    OPENAI_IMAGE_MODEL_TYPE,
};
use crate::sdk::cliproxy::auth::{Auth, AuthKind, AuthSourceKind};
use crate::sdk::cliproxy::model_registry::ModelRegistry;

pub const ATTRIBUTE_CONFIG_INDEX: &str = "config_index";

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConfiguredModel {
    pub name: String,
    pub alias: String,
    pub display_name: String,
    pub max_context_length: usize,
    pub image: bool,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub thinking: Option<RegistryThinkingSupport>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderKeyConfig {
    pub api_key: String,
    pub base_url: String,
    pub models: Vec<ConfiguredModel>,
    pub excluded_models: Vec<String>,
    pub weight: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpenAiCompatibilityConfig {
    pub name: String,
    pub disabled: bool,
    pub models: Vec<ConfiguredModel>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OAuthModelAlias {
    pub name: String,
    pub alias: String,
    pub display_name: String,
    pub fork: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServiceModelConfig {
    pub claude_keys: Vec<ProviderKeyConfig>,
    pub gemini_keys: Vec<ProviderKeyConfig>,
    pub vertex_keys: Vec<ProviderKeyConfig>,
    pub codex_keys: Vec<ProviderKeyConfig>,
    pub xai_keys: Vec<ProviderKeyConfig>,
    pub openai_compatibility: Vec<OpenAiCompatibilityConfig>,
    pub oauth_excluded_models: BTreeMap<String, Vec<String>>,
    pub oauth_model_alias: BTreeMap<String, Vec<OAuthModelAlias>>,
    pub force_model_prefix: bool,
}

impl ServiceModelConfig {
    /// Builds the model-facing snapshot from the already validated host config.
    /// Secret references never cross this boundary.
    pub fn from_runtime(config: &ValidatedRuntimeConfig) -> Self {
        let codex_keys = config
            .codex_accounts()
            .iter()
            .map(|account| ProviderKeyConfig {
                base_url: account.upstream_base_url.clone(),
                models: account
                    .models
                    .iter()
                    .map(|name| ConfiguredModel {
                        name: name.clone(),
                        alias: name.clone(),
                        ..ConfiguredModel::default()
                    })
                    .collect(),
                weight: account.weight,
                ..ProviderKeyConfig::default()
            })
            .collect();
        Self {
            codex_keys,
            ..Self::default()
        }
    }
}

pub struct ServiceModelRuntime {
    config: ServiceModelConfig,
    registry: Arc<dyn ModelRegistry>,
    catalog: Arc<StaticModelsCatalog>,
}

impl ServiceModelRuntime {
    pub fn new(
        config: ServiceModelConfig,
        registry: Arc<dyn ModelRegistry>,
        catalog: Arc<StaticModelsCatalog>,
    ) -> Self {
        Self {
            config,
            registry,
            catalog,
        }
    }

    pub fn config(&self) -> &ServiceModelConfig {
        &self.config
    }

    pub fn register_models_for_auth(&self, auth: &Auth) {
        if auth.id.trim().is_empty() {
            return;
        }
        if auth.disabled {
            self.registry.unregister_client(&auth.id);
            return;
        }
        let provider = auth.provider.trim().to_ascii_lowercase();
        let kind = auth.auth_kind();
        let mut excluded = self.oauth_excluded_models(&provider, kind);
        if let Some(value) = auth
            .attributes
            .get("excluded_models")
            .filter(|value| !value.trim().is_empty())
        {
            excluded = value.split(',').map(ToOwned::to_owned).collect();
        }
        let configured = match provider.as_str() {
            "claude" => self
                .resolve_entry(auth, &self.config.claude_keys)
                .map(|entry| (entry, "anthropic", "claude")),
            "gemini" | "gemini-interactions" => self
                .resolve_entry(auth, &self.config.gemini_keys)
                .map(|entry| (entry, "google", "gemini")),
            "vertex" => self
                .resolve_entry(auth, &self.config.vertex_keys)
                .map(|entry| (entry, "google", "vertex")),
            "codex" if kind == Some(AuthKind::ApiKey) => self
                .resolve_codex_entry(auth)
                .map(|entry| (entry, "openai", "openai")),
            "xai" => self
                .resolve_entry(auth, &self.config.xai_keys)
                .map(|entry| (entry, "xai", "xai")),
            _ => None,
        };
        let mut models = if let Some((entry, owner, model_type)) = configured {
            if provider == "codex" && entry.models.is_empty() {
                models_for_channel(&self.catalog, "codex").unwrap_or_default()
            } else if entry.models.is_empty() {
                self.static_models(&provider, auth)
            } else {
                build_config_models(&entry.models, owner, model_type)
            }
        } else if provider == "codex" && kind == Some(AuthKind::ApiKey) {
            Vec::new()
        } else if let Some(compat) = self.resolve_compat(auth, &provider) {
            build_openai_compatibility_config_models(compat)
        } else {
            self.static_models(&provider, auth)
        };
        if let Some(entry) = configured.map(|item| item.0) {
            if kind == Some(AuthKind::ApiKey) {
                excluded = entry.excluded_models.clone();
            }
        }
        models = apply_excluded_models(models, &excluded);
        models = apply_oauth_model_alias_for_auth(
            &self.config,
            &provider,
            kind,
            &auth.attributes,
            models,
        );
        models = apply_model_prefixes(models, &auth.prefix, self.config.force_model_prefix);
        let registration_provider = if self.resolve_compat(auth, &provider).is_some() {
            auth.attributes
                .get("provider_key")
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| provider.clone())
        } else {
            provider.clone()
        };
        if models.is_empty() {
            self.registry.unregister_client(&auth.id);
        } else {
            self.registry
                .register_client(&auth.id, &registration_provider, &models);
        }
    }

    fn static_models(&self, provider: &str, auth: &Auth) -> Vec<RegistryModelInfo> {
        let channel = if provider == "codex" {
            match auth
                .attributes
                .get("plan_type")
                .map(|v| v.trim().to_ascii_lowercase())
                .as_deref()
            {
                Some("free") => "codex-free",
                Some("plus") => "codex-plus",
                Some("team" | "business" | "go") => "codex-team",
                _ => "codex",
            }
        } else {
            provider
        };
        models_for_channel(&self.catalog, channel).unwrap_or_default()
    }

    fn resolve_entry<'a>(
        &self,
        auth: &Auth,
        entries: &'a [ProviderKeyConfig],
    ) -> Option<&'a ProviderKeyConfig> {
        if let Some(entry) = config_entry_for_auth_index(auth, entries) {
            return Some(entry);
        }
        match_credentials(auth, entries)
    }

    fn resolve_codex_entry<'a>(&'a self, auth: &Auth) -> Option<&'a ProviderKeyConfig> {
        if let Some(entry) = config_entry_for_auth_index(auth, &self.config.codex_keys) {
            if credentials_match(auth, entry) {
                return Some(entry);
            }
        }
        match_credentials(auth, &self.config.codex_keys)
    }

    fn resolve_compat<'a>(
        &'a self,
        auth: &Auth,
        provider: &str,
    ) -> Option<&'a OpenAiCompatibilityConfig> {
        if let Some(entry) = config_entry_for_auth_index(auth, &self.config.openai_compatibility) {
            if !entry.disabled {
                return Some(entry);
            }
        }
        let name = auth
            .attributes
            .get("compat_name")
            .map_or(provider, String::as_str);
        self.config
            .openai_compatibility
            .iter()
            .find(|entry| !entry.disabled && entry.name.eq_ignore_ascii_case(name))
    }

    fn oauth_excluded_models(&self, provider: &str, kind: Option<AuthKind>) -> Vec<String> {
        if kind == Some(AuthKind::ApiKey) {
            return Vec::new();
        }
        self.config
            .oauth_excluded_models
            .get(&provider.to_ascii_lowercase())
            .cloned()
            .unwrap_or_default()
    }
}

pub fn config_entry_for_auth_index<'a, T>(auth: &Auth, entries: &'a [T]) -> Option<&'a T> {
    if auth.auth_source_kind() != Some(AuthSourceKind::Config) {
        return None;
    }
    let index = auth
        .attributes
        .get(ATTRIBUTE_CONFIG_INDEX)?
        .trim()
        .parse::<usize>()
        .ok()?;
    entries.get(index)
}

fn credentials_match(auth: &Auth, entry: &ProviderKeyConfig) -> bool {
    let key = auth.attributes.get("api_key").map_or("", |v| v.trim());
    let base = auth.attributes.get("base_url").map_or("", |v| v.trim());
    if !key.is_empty() {
        entry.api_key.trim().eq_ignore_ascii_case(key)
            && (entry.base_url.trim().is_empty()
                || entry.base_url.trim().eq_ignore_ascii_case(base))
    } else {
        !base.is_empty() && entry.base_url.trim().eq_ignore_ascii_case(base)
    }
}

fn match_credentials<'a>(
    auth: &Auth,
    entries: &'a [ProviderKeyConfig],
) -> Option<&'a ProviderKeyConfig> {
    entries.iter().find(|entry| credentials_match(auth, entry))
}

pub fn apply_excluded_models(
    models: Vec<RegistryModelInfo>,
    excluded: &[String],
) -> Vec<RegistryModelInfo> {
    let patterns = excluded
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        return models;
    }
    models
        .into_iter()
        .filter(|model| {
            !patterns
                .iter()
                .any(|pattern| match_wildcard(pattern, &model.id.trim().to_ascii_lowercase()))
        })
        .collect()
}

pub fn match_wildcard(pattern: &str, value: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    if !pattern.contains('*') {
        return pattern == value;
    }
    let parts = pattern.split('*').collect::<Vec<_>>();
    let mut rest = value;
    if let Some(prefix) = parts.first().filter(|part| !part.is_empty()) {
        let Some(next) = rest.strip_prefix(prefix) else {
            return false;
        };
        rest = next;
    }
    if let Some(suffix) = parts.last().filter(|part| !part.is_empty()) {
        let Some(next) = rest.strip_suffix(suffix) else {
            return false;
        };
        rest = next;
    }
    for segment in parts
        .iter()
        .skip(1)
        .take(parts.len().saturating_sub(2))
        .filter(|part| !part.is_empty())
    {
        let Some(index) = rest.find(segment) else {
            return false;
        };
        rest = &rest[index + segment.len()..];
    }
    true
}

pub fn apply_model_prefixes(
    models: Vec<RegistryModelInfo>,
    prefix: &str,
    force: bool,
) -> Vec<RegistryModelInfo> {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return models;
    }
    let mut out = Vec::with_capacity(models.len() * 2);
    let mut seen = HashSet::new();
    for model in models {
        let id = model.id.trim().to_owned();
        if id.is_empty() {
            continue;
        }
        if (!force || prefix == id) && seen.insert(id.clone()) {
            out.push(model.clone());
        }
        let mut clone = model;
        clone.id = format!("{prefix}/{id}");
        if seen.insert(clone.id.clone()) {
            out.push(clone);
        }
    }
    out
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_secs() as i64)
}

pub fn build_config_models(
    models: &[ConfiguredModel],
    owned_by: &str,
    model_type: &str,
) -> Vec<RegistryModelInfo> {
    let mut seen = HashSet::new();
    models
        .iter()
        .filter_map(|model| {
            let name = model.name.trim();
            let alias = if model.alias.trim().is_empty() {
                name
            } else {
                model.alias.trim()
            };
            if alias.is_empty() || !seen.insert(alias.to_ascii_lowercase()) {
                return None;
            }
            Some(RegistryModelInfo {
                id: alias.to_owned(),
                object: "model".to_owned(),
                created: now_unix(),
                owned_by: owned_by.to_owned(),
                provider_type: model_type.to_owned(),
                display_name: if model.display_name.trim().is_empty() {
                    name.to_owned()
                } else {
                    model.display_name.trim().to_owned()
                },
                context_length: model.max_context_length,
                max_context_length: model.max_context_length,
                thinking: model.thinking.clone(),
                user_defined: true,
                ..RegistryModelInfo::default()
            })
        })
        .collect()
}

pub fn build_codex_config_models(
    models: &[ConfiguredModel],
    catalog: &StaticModelsCatalog,
) -> Vec<RegistryModelInfo> {
    if models.is_empty() {
        models_for_channel(catalog, "codex").unwrap_or_default()
    } else {
        build_config_models(models, "openai", "openai")
    }
}

pub fn build_openai_compatibility_config_models(
    compat: &OpenAiCompatibilityConfig,
) -> Vec<RegistryModelInfo> {
    compat
        .models
        .iter()
        .filter_map(|model| {
            let mut info = build_config_models(
                std::slice::from_ref(model),
                &compat.name,
                if model.image {
                    OPENAI_IMAGE_MODEL_TYPE
                } else {
                    "openai-compatibility"
                },
            )
            .pop()?;
            info.user_defined = false;
            if !model.image && info.thinking.is_none() {
                info.thinking = Some(RegistryThinkingSupport {
                    levels: vec!["low".into(), "medium".into(), "high".into()],
                    ..RegistryThinkingSupport::default()
                });
            }
            info.supported_input_modalities = normalize_modalities(&model.input_modalities);
            info.supported_output_modalities = normalize_modalities(&model.output_modalities);
            Some(info)
        })
        .collect()
}

fn normalize_modalities(raw: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    raw.iter()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty() && seen.insert(v.clone()))
        .collect()
}

pub fn rewrite_model_info_name(name: &str, old_id: &str, new_id: &str) -> String {
    let trimmed = name.trim();
    if trimmed.eq_ignore_ascii_case(old_id) {
        return new_id.to_owned();
    }
    let suffix = format!("/{old_id}");
    if trimmed
        .to_ascii_lowercase()
        .ends_with(&suffix.to_ascii_lowercase())
    {
        return format!("{}{}", &trimmed[..trimmed.len() - old_id.len()], new_id);
    }
    name.to_owned()
}

pub fn apply_oauth_model_alias_for_auth(
    config: &ServiceModelConfig,
    provider: &str,
    kind: Option<AuthKind>,
    attributes: &BTreeMap<String, String>,
    models: Vec<RegistryModelInfo>,
) -> Vec<RegistryModelInfo> {
    if kind == Some(AuthKind::ApiKey) {
        return models;
    }
    let mut aliases = parse_per_auth_aliases(attributes.get("model_aliases"));
    let mut seen = aliases
        .iter()
        .map(|entry| entry.alias.trim().to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if let Some(global) = config
        .oauth_model_alias
        .get(&provider.trim().to_ascii_lowercase())
    {
        aliases.extend(
            global
                .iter()
                .filter(|entry| seen.insert(entry.alias.trim().to_ascii_lowercase()))
                .cloned(),
        );
    }
    apply_oauth_model_alias_entries(&aliases, models)
}

fn parse_per_auth_aliases(raw: Option<&String>) -> Vec<OAuthModelAlias> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let Ok(values) = serde_json::from_str::<Vec<serde_json::Value>>(raw) else {
        return Vec::new();
    };
    values
        .into_iter()
        .map(|value| OAuthModelAlias {
            name: value
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned(),
            alias: value
                .get("alias")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned(),
            display_name: value
                .get("display-name")
                .or_else(|| value.get("display_name"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned(),
            fork: value
                .get("fork")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
        .collect()
}

pub fn apply_oauth_model_alias_entries(
    aliases: &[OAuthModelAlias],
    models: Vec<RegistryModelInfo>,
) -> Vec<RegistryModelInfo> {
    let mut forward: HashMap<String, Vec<&OAuthModelAlias>> = HashMap::new();
    for entry in aliases {
        if !entry.name.trim().is_empty()
            && !entry.alias.trim().is_empty()
            && !entry.name.eq_ignore_ascii_case(&entry.alias)
        {
            forward
                .entry(entry.name.trim().to_ascii_lowercase())
                .or_default()
                .push(entry);
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for model in models {
        let id = model.id.trim().to_owned();
        let entries = forward.get(&id.to_ascii_lowercase());
        let Some(entries) = entries else {
            if seen.insert(id.to_ascii_lowercase()) {
                out.push(model);
            }
            continue;
        };
        let keep = entries.iter().any(|entry| entry.fork);
        if keep && seen.insert(id.to_ascii_lowercase()) {
            out.push(model.clone());
        }
        let mut added = false;
        for entry in entries {
            let alias = entry.alias.trim();
            if alias.is_empty()
                || alias.eq_ignore_ascii_case(&id)
                || !seen.insert(alias.to_ascii_lowercase())
            {
                continue;
            }
            let mut clone = model.clone();
            clone.id = alias.to_owned();
            if !entry.display_name.trim().is_empty() {
                clone.display_name = entry.display_name.trim().to_owned();
            }
            if !clone.name.is_empty() {
                clone.name = rewrite_model_info_name(&clone.name, &id, alias);
            }
            out.push(clone);
            added = true;
        }
        if !keep && !added && seen.insert(id.to_ascii_lowercase()) {
            out.push(model);
        }
    }
    out
}
