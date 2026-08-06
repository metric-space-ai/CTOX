// ref: internal/watcher/diff/oauth_excluded.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExcludedModelsSummary {
    pub count: usize,
    pub models: Vec<String>,
    pub hash: String,
}
pub fn summarize_excluded_models(list: &[String]) -> ExcludedModelsSummary {
    let mut models = list
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    let hash = super::model_hash::compute_excluded_models_hash(&models);
    ExcludedModelsSummary {
        count: models.len(),
        models,
        hash,
    }
}
pub fn summarize_oauth_excluded_models(
    entries: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, ExcludedModelsSummary> {
    entries
        .iter()
        .filter_map(|(provider, models)| {
            let provider = provider.trim().to_ascii_lowercase();
            (!provider.is_empty()).then(|| (provider, summarize_excluded_models(models)))
        })
        .collect()
}
pub fn diff_oauth_excluded_model_changes(
    old: &BTreeMap<String, Vec<String>>,
    new: &BTreeMap<String, Vec<String>>,
) -> (Vec<String>, Vec<String>) {
    let old = summarize_oauth_excluded_models(old);
    let new = summarize_oauth_excluded_models(new);
    let mut changed = Vec::new();
    let mut affected = Vec::new();
    for provider in old
        .keys()
        .chain(new.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        if old.get(provider) != new.get(provider) {
            changed.push(format!(
                "oauth-excluded-models.{provider}: {} -> {}",
                old.get(provider).map_or(0, |v| v.count),
                new.get(provider).map_or(0, |v| v.count)
            ));
            affected.push(provider.clone());
        }
    }
    (changed, affected)
}
