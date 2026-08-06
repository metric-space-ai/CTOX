// ref: sdk/cliproxy/auth/oauth_model_alias.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: compiled instance table and per-auth attributes replace manager-global alias authority
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, HashSet};

use crate::internal::config::config_normalization::OAuthModelAlias;
use crate::internal::config::CodexModel;
use crate::internal::thinking::{parse_suffix, SuffixResult};

use super::{Auth, AuthKind};

const OAUTH_MODEL_ALIASES_ATTRIBUTE: &str = "model_aliases";

/// Result of resolving a client-visible model alias for one credential.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OAuthModelAliasResult {
    pub upstream_model: String,
    pub force_mapping: bool,
    pub original_alias: String,
}

/// Instance-owned compiled alias table. Replacing the table is an explicit
/// host operation; request execution never reads process-global alias state.
#[derive(Clone, Debug, Default)]
pub struct OAuthModelAliasTable {
    reverse: BTreeMap<String, BTreeMap<String, OAuthModelAliasEntry>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OAuthModelAliasEntry {
    upstream_model: String,
    config_alias: String,
    force_mapping: bool,
}

pub trait ModelAliasEntry {
    fn model_name(&self) -> &str;
    fn model_alias(&self) -> &str;
    fn force_mapping(&self) -> bool;
}

impl ModelAliasEntry for OAuthModelAlias {
    fn model_name(&self) -> &str {
        &self.name
    }
    fn model_alias(&self) -> &str {
        &self.alias
    }
    fn force_mapping(&self) -> bool {
        self.force_mapping
    }
}

impl ModelAliasEntry for CodexModel {
    fn model_name(&self) -> &str {
        &self.name
    }
    fn model_alias(&self) -> &str {
        &self.alias
    }
    fn force_mapping(&self) -> bool {
        self.force_mapping
    }
}

impl OAuthModelAliasTable {
    #[must_use]
    pub fn compile(aliases: &BTreeMap<String, Vec<OAuthModelAlias>>) -> Self {
        let mut reverse = BTreeMap::new();
        for (channel, aliases) in aliases {
            let channel = channel.trim().to_ascii_lowercase();
            if channel.is_empty() {
                continue;
            }
            let mut entries = BTreeMap::new();
            for alias in sanitize_oauth_model_aliases(aliases) {
                entries
                    .entry(alias.alias.to_ascii_lowercase())
                    .or_insert(OAuthModelAliasEntry {
                        upstream_model: alias.name,
                        config_alias: alias.alias,
                        force_mapping: alias.force_mapping,
                    });
            }
            if !entries.is_empty() {
                reverse.insert(channel, entries);
            }
        }
        Self { reverse }
    }

    #[must_use]
    pub fn resolve(&self, auth: &Auth, requested_model: &str) -> OAuthModelAliasResult {
        if let Some(result) = resolve_upstream_model_from_aliases(
            &oauth_model_aliases_from_attributes(&auth.attributes),
            requested_model,
        )
        .filter_nonempty()
        {
            return result;
        }
        let channel = model_alias_channel(auth);
        let Some(entries) = self.reverse.get(&channel) else {
            return OAuthModelAliasResult::default();
        };
        let (request, candidates) = model_alias_lookup_candidates(requested_model);
        let base = if request.model_name.trim().is_empty() {
            requested_model.trim()
        } else {
            request.model_name.trim()
        };
        for candidate in candidates {
            let Some(entry) = entries.get(&candidate.to_ascii_lowercase()) else {
                continue;
            };
            if entry.upstream_model.eq_ignore_ascii_case(base) && !entry.force_mapping {
                return OAuthModelAliasResult::default();
            }
            return alias_result(
                entry.upstream_model.as_str(),
                entry.config_alias.as_str(),
                entry.force_mapping,
                requested_model,
                &request,
            );
        }
        OAuthModelAliasResult::default()
    }
}

impl OAuthModelAliasResult {
    fn filter_nonempty(self) -> Option<Self> {
        (!self.upstream_model.is_empty()).then_some(self)
    }
}

#[must_use]
pub fn model_alias_lookup_candidates(requested_model: &str) -> (SuffixResult, Vec<String>) {
    let requested_model = requested_model.trim();
    if requested_model.is_empty() {
        return (SuffixResult::default(), Vec::new());
    }
    let result = parse_suffix(requested_model);
    let base = if result.model_name.is_empty() {
        requested_model
    } else {
        result.model_name.trim()
    };
    let mut candidates = vec![requested_model.to_owned()];
    if base != requested_model {
        candidates.push(base.to_owned());
    }
    (result, candidates)
}

#[must_use]
pub fn preserve_resolved_model_suffix(resolved: &str, request: &SuffixResult) -> String {
    let resolved = resolved.trim();
    if resolved.is_empty() {
        return String::new();
    }
    if parse_suffix(resolved).has_suffix {
        return resolved.to_owned();
    }
    if request.has_suffix && !request.raw_suffix.is_empty() {
        format!("{resolved}({})", request.raw_suffix)
    } else {
        resolved.to_owned()
    }
}

#[must_use]
pub fn resolve_model_alias_pool_from_config_models<T: ModelAliasEntry>(
    requested_model: &str,
    models: &[T],
) -> Vec<String> {
    let requested_model = requested_model.trim();
    let (request, candidates) = model_alias_lookup_candidates(requested_model);
    for candidate in &candidates {
        let mut seen = HashSet::new();
        let pool = models
            .iter()
            .filter_map(|entry| {
                let alias = entry.model_alias().trim();
                if alias.is_empty() || !alias.eq_ignore_ascii_case(candidate) {
                    return None;
                }
                let name = entry.model_name().trim();
                let resolved = preserve_resolved_model_suffix(
                    if name.is_empty() { candidate } else { name },
                    &request,
                );
                seen.insert(resolved.to_ascii_lowercase())
                    .then_some(resolved)
            })
            .collect::<Vec<_>>();
        if !pool.is_empty() {
            return pool;
        }
    }
    for candidate in candidates {
        if let Some(entry) = models
            .iter()
            .find(|entry| entry.model_name().trim().eq_ignore_ascii_case(&candidate))
        {
            return vec![preserve_resolved_model_suffix(entry.model_name(), &request)];
        }
    }
    Vec::new()
}

#[must_use]
pub fn resolve_model_alias_result_from_config_models<T: ModelAliasEntry>(
    requested_model: &str,
    models: &[T],
) -> OAuthModelAliasResult {
    let aliases = models
        .iter()
        .map(|entry| OAuthModelAlias {
            name: entry.model_name().to_owned(),
            alias: entry.model_alias().to_owned(),
            force_mapping: entry.force_mapping(),
            ..OAuthModelAlias::default()
        })
        .collect::<Vec<_>>();
    resolve_upstream_model_from_aliases(&aliases, requested_model)
}

#[must_use]
pub fn sanitize_oauth_model_aliases(aliases: &[OAuthModelAlias]) -> Vec<OAuthModelAlias> {
    let mut seen = HashSet::new();
    aliases
        .iter()
        .filter_map(|entry| {
            let mut entry = entry.clone();
            entry.name = entry.name.trim().to_owned();
            entry.alias = entry.alias.trim().to_owned();
            entry.display_name = entry.display_name.trim().to_owned();
            (!entry.name.is_empty()
                && !entry.alias.is_empty()
                && !entry.name.eq_ignore_ascii_case(&entry.alias)
                && seen.insert(entry.alias.to_ascii_lowercase()))
            .then_some(entry)
        })
        .collect()
}

pub fn set_oauth_model_aliases_attribute(auth: &mut Auth, aliases: &[OAuthModelAlias]) {
    let aliases = sanitize_oauth_model_aliases(aliases);
    if aliases.is_empty() {
        return;
    }
    if let Ok(encoded) = serde_json::to_string(&aliases) {
        auth.attributes
            .insert(OAUTH_MODEL_ALIASES_ATTRIBUTE.to_owned(), encoded);
    }
}

#[must_use]
pub fn oauth_model_aliases_from_attributes(
    attributes: &BTreeMap<String, String>,
) -> Vec<OAuthModelAlias> {
    attributes
        .get(OAUTH_MODEL_ALIASES_ATTRIBUTE)
        .and_then(|raw| serde_json::from_str::<Vec<OAuthModelAlias>>(raw.trim()).ok())
        .map_or_else(Vec::new, |aliases| sanitize_oauth_model_aliases(&aliases))
}

#[must_use]
pub fn resolve_upstream_model_from_aliases(
    aliases: &[OAuthModelAlias],
    requested_model: &str,
) -> OAuthModelAliasResult {
    let (request, candidates) = model_alias_lookup_candidates(requested_model);
    let base = if request.model_name.trim().is_empty() {
        requested_model.trim()
    } else {
        request.model_name.trim()
    };
    for candidate in candidates {
        for entry in aliases {
            let original = entry.name.trim();
            let alias = entry.alias.trim();
            if original.is_empty() || alias.is_empty() || !alias.eq_ignore_ascii_case(&candidate) {
                continue;
            }
            if original.eq_ignore_ascii_case(base) && !entry.force_mapping {
                return OAuthModelAliasResult::default();
            }
            return alias_result(
                original,
                alias,
                entry.force_mapping,
                requested_model,
                &request,
            );
        }
    }
    OAuthModelAliasResult::default()
}

fn alias_result(
    upstream: &str,
    configured_alias: &str,
    force_mapping: bool,
    requested_model: &str,
    request: &SuffixResult,
) -> OAuthModelAliasResult {
    OAuthModelAliasResult {
        upstream_model: preserve_resolved_model_suffix(upstream, request),
        force_mapping,
        original_alias: if force_mapping {
            configured_alias.trim().to_owned()
        } else {
            requested_model.trim().to_owned()
        },
    }
}

#[must_use]
pub fn oauth_model_alias_channel(provider: &str, auth_kind: &str) -> String {
    let provider = provider.trim().to_ascii_lowercase();
    let auth_kind = auth_kind
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '-'], "");
    if provider.is_empty() || auth_kind == "apikey" || provider == "gemini" {
        String::new()
    } else {
        provider
    }
}

#[must_use]
pub fn model_alias_channel(auth: &Auth) -> String {
    oauth_model_alias_channel(
        &auth.provider,
        auth.auth_kind().map_or("", AuthKind::as_str),
    )
}
