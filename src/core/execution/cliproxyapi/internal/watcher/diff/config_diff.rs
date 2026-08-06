// ref: internal/watcher/diff/config_diff.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::config_reload::WatcherConfig;
pub fn build_config_change_details(old: &WatcherConfig, new: &WatcherConfig) -> Vec<String> {
    let mut changes = Vec::new();
    for provider in old
        .providers
        .keys()
        .chain(new.providers.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let before = old.providers.get(provider).map_or(0, Vec::len);
        let after = new.providers.get(provider).map_or(0, Vec::len);
        if old.providers.get(provider) != new.providers.get(provider) {
            changes.push(format!(
                "provider.{provider}: credentials {before} -> {after} (secret values redacted)"
            ));
        }
    }
    changes.extend(super::openai_compat::diff_openai_compatibility(
        &old.openai_compatibility,
        &new.openai_compatibility,
    ));
    let (excluded, _) = super::oauth_excluded::diff_oauth_excluded_model_changes(
        &old.oauth_excluded_models,
        &new.oauth_excluded_models,
    );
    changes.extend(excluded);
    let (aliases, _) = super::oauth_model_alias::diff_oauth_model_alias_changes(
        &old.oauth_model_aliases,
        &new.oauth_model_aliases,
    );
    changes.extend(aliases);
    for setting in old
        .settings
        .keys()
        .chain(new.settings.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        if old.settings.get(setting) != new.settings.get(setting) {
            changes.push(format!("setting.{setting}: updated"));
        }
    }
    changes
}
pub fn trim_strings(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect()
}
pub fn display_optional_value(raw: &str) -> &str {
    if raw.trim().is_empty() {
        "<unset>"
    } else {
        "<set>"
    }
}
