// ref: internal/watcher/diff/model_hash.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::watcher::config_reload::ModelRoute;
use sha2::{Digest, Sha256};

pub fn compute_models_hash(models: &[ModelRoute]) -> String {
    if models.is_empty() {
        return String::new();
    }
    let keys = models
        .iter()
        .map(|model| {
            format!(
                "{}\0{}\0{}\0{}\0{}\0{}\0{}",
                model.name.trim(),
                model.alias.trim(),
                model.display_name.trim(),
                model.force_mapping,
                model.image,
                model
                    .modalities
                    .iter()
                    .map(|v| v.trim())
                    .collect::<Vec<_>>()
                    .join(","),
                model.thinking.as_deref().unwrap_or_default().trim()
            )
        })
        .collect::<Vec<_>>();
    hash_joined(&keys)
}
pub fn compute_excluded_models_hash(excluded: &[String]) -> String {
    let mut values = excluded
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    hash_joined(&values)
}
fn hash_joined(values: &[String]) -> String {
    if values.is_empty() {
        String::new()
    } else {
        format!("{:x}", Sha256::digest(values.join("\n").as_bytes()))
    }
}
