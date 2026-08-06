// ref: internal/watcher/synthesizer/config.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::{ModelAlias, SynthesisContext, SynthesizedAuth};
use super::helpers::{
    add_config_headers_to_attrs, apply_auth_excluded_models_meta, StableIdGenerator,
};
use super::interface::AuthSynthesizer;
use crate::internal::credentialweight::normalize;
use std::io;

#[derive(Debug, Default)]
pub struct ConfigSynthesizer;
impl ConfigSynthesizer {
    pub fn new() -> Self {
        Self
    }
}

impl AuthSynthesizer for ConfigSynthesizer {
    fn synthesize(&self, context: &SynthesisContext<'_>) -> io::Result<Vec<SynthesizedAuth>> {
        let mut ids = StableIdGenerator::default();
        let mut output = Vec::new();
        for (provider, entries) in &context.config.providers {
            for (index, entry) in entries.iter().enumerate() {
                let weight = match entry.weight.map(normalize).transpose() {
                    Ok(weight) => weight,
                    Err(_) => continue,
                };
                if entry.api_key.trim().is_empty() {
                    continue;
                }
                let stable = if entry.id.trim().is_empty() {
                    index.to_string()
                } else {
                    entry.id.clone()
                };
                let (id, _) = ids.next(provider, &[&stable, &entry.base_url, &entry.api_key]);
                let mut auth = SynthesizedAuth {
                    id,
                    provider: provider.trim().to_ascii_lowercase(),
                    prefix: entry.prefix.trim_matches('/').to_owned(),
                    label: entry.id.trim().to_owned(),
                    disabled: entry.disabled,
                    priority: entry.priority,
                    weight,
                    proxy_url: entry.proxy_url.trim().to_owned(),
                    ..SynthesizedAuth::default()
                };
                auth.attributes
                    .insert("api_key".into(), entry.api_key.clone());
                auth.attributes
                    .insert("base_url".into(), entry.base_url.trim().to_owned());
                add_config_headers_to_attrs(&entry.headers, &mut auth.attributes);
                auth.model_aliases = entry
                    .models
                    .iter()
                    .filter(|model| !model.name.trim().is_empty())
                    .map(|model| ModelAlias {
                        name: model.name.trim().to_owned(),
                        alias: model.alias.trim().to_owned(),
                        display_name: model.display_name.trim().to_owned(),
                        fork: model.force_mapping,
                    })
                    .collect();
                let global = context
                    .config
                    .oauth_excluded_models
                    .get(provider)
                    .cloned()
                    .unwrap_or_default();
                apply_auth_excluded_models_meta(&mut auth, &global, &entry.excluded_models);
                output.push(auth);
            }
        }
        for (index, compat) in context.config.openai_compatibility.iter().enumerate() {
            for (key_index, key) in compat
                .api_keys
                .iter()
                .enumerate()
                .filter(|(_, key)| !key.trim().is_empty())
            {
                let identity = if compat.name.trim().is_empty() {
                    index.to_string()
                } else {
                    compat.name.clone()
                };
                let (id, _) = ids.next(
                    "openai-compatibility",
                    &[&identity, &key_index.to_string(), key],
                );
                output.push(SynthesizedAuth {
                    id,
                    provider: "openai-compatibility".into(),
                    label: compat.name.trim().to_owned(),
                    proxy_url: compat.proxy_url.trim().to_owned(),
                    attributes: [
                        ("api_key".into(), key.clone()),
                        ("base_url".into(), compat.base_url.trim().to_owned()),
                        (
                            "models_hash".into(),
                            crate::internal::watcher::diff::model_hash::compute_models_hash(
                                &compat.models,
                            ),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    model_aliases: compat
                        .models
                        .iter()
                        .map(|model| ModelAlias {
                            name: model.name.clone(),
                            alias: model.alias.clone(),
                            display_name: model.display_name.clone(),
                            fork: model.force_mapping,
                        })
                        .collect(),
                    ..SynthesizedAuth::default()
                });
            }
        }
        Ok(output)
    }
}
