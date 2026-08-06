// ref: internal/watcher/synthesizer/file.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::{ModelAlias, SynthesisContext, SynthesizedAuth};
use super::helpers::{apply_auth_excluded_models_meta, StableIdGenerator};
use super::interface::AuthSynthesizer;
use crate::internal::credentialweight::normalize;
use serde_json::Value;
use std::io;
use std::path::Path;

#[derive(Debug, Default)]
pub struct FileSynthesizer;
impl FileSynthesizer {
    pub fn new() -> Self {
        Self
    }
}
impl AuthSynthesizer for FileSynthesizer {
    fn synthesize(&self, context: &SynthesisContext<'_>) -> io::Result<Vec<SynthesizedAuth>> {
        let mut output = Vec::new();
        for path in &context.files {
            let data = match context.filesystem.read(path) {
                Ok(data) => data,
                Err(_) => continue,
            };
            match synthesize_auth_file(context, path, &data) {
                Ok(mut auths) => output.append(&mut auths),
                Err(_) => continue,
            }
        }
        Ok(output)
    }
}

pub fn synthesize_auth_file(
    context: &SynthesisContext<'_>,
    path: &Path,
    data: &[u8],
) -> io::Result<Vec<SynthesizedAuth>> {
    if let Some(parser) = &context.parser {
        if let Some(auths) = parser.parse(path, data)? {
            return Ok(compact_plugin_auths(auths));
        }
    }
    let value: Value = serde_json::from_slice(data)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let object = value
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "auth file must be an object"))?;
    let provider = object
        .get("type")
        .or_else(|| object.get("provider"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if provider.is_empty() || provider == "gemini" {
        return Ok(Vec::new());
    }
    let weight = match object
        .get("weight")
        .and_then(Value::as_i64)
        .map(normalize)
        .transpose()
    {
        Ok(weight) => weight,
        Err(_) => return Ok(Vec::new()),
    };
    let relative = path
        .strip_prefix(context.auth_dir)
        .unwrap_or(path)
        .to_string_lossy();
    let mut ids = StableIdGenerator::default();
    let (id, _) = ids.next(&provider, &[&relative]);
    let mut auth = SynthesizedAuth {
        id,
        provider: provider.clone(),
        file_name: relative.into_owned(),
        label: object
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        disabled: object
            .get("disabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        priority: object.get("priority").and_then(Value::as_i64).unwrap_or(0) as i32,
        weight,
        proxy_url: object
            .get("proxy_url")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        ..SynthesizedAuth::default()
    };
    for (key, value) in object {
        if let Some(value) = value.as_str() {
            auth.attributes.insert(key.clone(), value.to_owned());
        }
    }
    auth.model_aliases = extract_model_aliases(object.get("model_aliases"));
    let per_key = extract_strings(object.get("excluded_models"));
    let global = context
        .config
        .oauth_excluded_models
        .get(&provider)
        .cloned()
        .unwrap_or_default();
    apply_auth_excluded_models_meta(&mut auth, &global, &per_key);
    Ok(vec![auth])
}

fn compact_plugin_auths(auths: Vec<SynthesizedAuth>) -> Vec<SynthesizedAuth> {
    let mut seen = std::collections::HashSet::new();
    auths
        .into_iter()
        .filter(|auth| {
            !auth.id.trim().is_empty()
                && !auth.provider.trim().is_empty()
                && seen.insert(auth.id.clone())
        })
        .collect()
}
fn extract_strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}
fn extract_model_aliases(value: Option<&Value>) -> Vec<ModelAlias> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| serde_json::from_value(item.clone()).ok())
        .collect()
}
