// ref: internal/watcher/synthesizer/helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::SynthesizedAuth;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Default)]
pub struct StableIdGenerator {
    collisions: HashMap<String, usize>,
}
impl StableIdGenerator {
    pub fn next(&mut self, kind: &str, parts: &[&str]) -> (String, String) {
        let canonical = std::iter::once(kind.trim().to_ascii_lowercase())
            .chain(parts.iter().map(|part| part.trim().to_ascii_lowercase()))
            .collect::<Vec<_>>()
            .join("\0");
        let digest = format!("{:x}", Sha256::digest(canonical.as_bytes()));
        let base = format!("{}-{}", sanitize(kind), &digest[..16]);
        let count = self.collisions.entry(base.clone()).or_default();
        *count += 1;
        let id = if *count == 1 {
            base
        } else {
            format!("{base}-{}", *count)
        };
        (id, digest)
    }
}

pub fn apply_auth_excluded_models_meta(
    auth: &mut SynthesizedAuth,
    global: &[String],
    per_key: &[String],
) {
    let mut normalized = global
        .iter()
        .chain(per_key)
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    auth.excluded_models = normalized.clone();
    if !normalized.is_empty() {
        auth.metadata
            .insert("excluded_models".into(), serde_json::json!(normalized));
    }
}

pub fn add_config_headers_to_attrs(
    headers: &BTreeMap<String, String>,
    attrs: &mut BTreeMap<String, String>,
) {
    for (name, value) in headers {
        let name = name.trim();
        if !name.is_empty() {
            attrs.insert(format!("header:{name}"), value.trim().to_owned());
        }
    }
}

fn sanitize(value: &str) -> String {
    let value = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>();
    value.trim_matches('-').to_owned()
}
