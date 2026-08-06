// ref: internal/watcher/diff/oauth_model_alias.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::synthesizer::context::ModelAlias;
use std::collections::BTreeMap;
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OAuthModelAliasSummary {
    pub count: usize,
    pub entries: Vec<String>,
}
pub fn summarize_oauth_model_alias(
    entries: &BTreeMap<String, Vec<ModelAlias>>,
) -> BTreeMap<String, OAuthModelAliasSummary> {
    entries
        .iter()
        .filter_map(|(provider, aliases)| {
            let provider = provider.trim().to_ascii_lowercase();
            if provider.is_empty() {
                return None;
            }
            let mut values = aliases
                .iter()
                .filter(|alias| !alias.name.trim().is_empty())
                .map(|alias| {
                    format!(
                        "{}=>{}|{}|{}",
                        alias.name.trim(),
                        alias.alias.trim(),
                        alias.display_name.trim(),
                        alias.fork
                    )
                })
                .collect::<Vec<_>>();
            values.sort();
            values.dedup();
            Some((
                provider,
                OAuthModelAliasSummary {
                    count: values.len(),
                    entries: values,
                },
            ))
        })
        .collect()
}
pub fn diff_oauth_model_alias_changes(
    old: &BTreeMap<String, Vec<ModelAlias>>,
    new: &BTreeMap<String, Vec<ModelAlias>>,
) -> (Vec<String>, Vec<String>) {
    let old = summarize_oauth_model_alias(old);
    let new = summarize_oauth_model_alias(new);
    let mut changes = Vec::new();
    let mut affected = Vec::new();
    for provider in old
        .keys()
        .chain(new.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        if old.get(provider) != new.get(provider) {
            changes.push(format!(
                "oauth-model-alias.{provider}: {} -> {}",
                old.get(provider).map_or(0, |v| v.count),
                new.get(provider).map_or(0, |v| v.count)
            ));
            affected.push(provider.clone());
        }
    }
    (changes, affected)
}
