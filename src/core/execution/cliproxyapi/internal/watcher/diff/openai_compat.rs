// ref: internal/watcher/diff/openai_compat.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::config_reload::OpenAiCompatibility;
use std::collections::{BTreeMap, HashSet};

pub fn diff_openai_compatibility(
    old: &[OpenAiCompatibility],
    new: &[OpenAiCompatibility],
) -> Vec<String> {
    let old = keyed(old);
    let new = keyed(new);
    let keys = old
        .keys()
        .chain(new.keys())
        .collect::<std::collections::BTreeSet<_>>();
    keys.into_iter()
        .filter_map(|key| match (old.get(key), new.get(key)) {
            (None, Some(entry)) => Some(format!(
                "openai-compatibility added {key} ({} key(s), {} model(s))",
                count_api_keys(entry),
                count_models(entry)
            )),
            (Some(_), None) => Some(format!("openai-compatibility removed {key}")),
            (Some(before), Some(after)) if before != after => {
                Some(describe_update(key, before, after))
            }
            _ => None,
        })
        .collect()
}

fn keyed(entries: &[OpenAiCompatibility]) -> BTreeMap<String, OpenAiCompatibility> {
    let mut output = BTreeMap::new();
    let mut used = HashSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let base = openai_compat_key(entry, index);
        let mut key = base.clone();
        let mut suffix = 2;
        while !used.insert(key.clone()) {
            key = format!("{base}#{suffix}");
            suffix += 1;
        }
        output.insert(key, entry.clone());
    }
    output
}
pub fn openai_compat_key(entry: &OpenAiCompatibility, index: usize) -> String {
    if !entry.name.trim().is_empty() {
        return entry.name.trim().to_ascii_lowercase();
    }
    if !entry.base_url.trim().is_empty() {
        return redact_url(&entry.base_url);
    }
    let signature = openai_compat_signature(entry);
    if signature.is_empty() {
        format!("entry-{index}")
    } else {
        signature
    }
}
pub fn openai_compat_signature(entry: &OpenAiCompatibility) -> String {
    let mut models = entry
        .models
        .iter()
        .map(|model| {
            if model.alias.trim().is_empty() {
                model.name.trim()
            } else {
                model.alias.trim()
            }
        })
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    if models.is_empty() && entry.api_keys.is_empty() {
        String::new()
    } else {
        format!(
            "{}-key(s)/{}",
            entry
                .api_keys
                .iter()
                .filter(|key| !key.trim().is_empty())
                .count(),
            models.join(",")
        )
    }
}
fn describe_update(key: &str, old: &OpenAiCompatibility, new: &OpenAiCompatibility) -> String {
    format!(
        "openai-compatibility updated {key}: keys {} -> {}, models {} -> {}, endpoint {}",
        count_api_keys(old),
        count_api_keys(new),
        count_models(old),
        count_models(new),
        if old.base_url == new.base_url {
            "unchanged"
        } else {
            "updated"
        }
    )
}
fn count_api_keys(entry: &OpenAiCompatibility) -> usize {
    entry
        .api_keys
        .iter()
        .filter(|key| !key.trim().is_empty())
        .count()
}
fn count_models(entry: &OpenAiCompatibility) -> usize {
    entry
        .models
        .iter()
        .filter(|model| !model.name.trim().is_empty())
        .count()
}
fn redact_url(raw: &str) -> String {
    url::Url::parse(raw)
        .map(|url| {
            format!(
                "{}://{}{}",
                url.scheme(),
                url.host_str().unwrap_or("redacted"),
                url.path()
            )
        })
        .unwrap_or_else(|_| "endpoint".into())
}
