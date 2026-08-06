// ref: internal/modelconfig/model_hash.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use sha2::{Digest, Sha256};

use super::model_info::ThinkingSupport;

/// Common configured alias fields shared by Claude, Codex, Gemini, and Vertex.
/// Fields not used by the pinned hashing contract remain outside this private
/// hashing view.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAlias {
    pub name: String,
    pub alias: String,
    pub display_name: String,
    pub force_mapping: bool,
    pub thinking: Option<ThinkingSupport>,
}

pub type VertexCompatModel = ModelAlias;
pub type ClaudeModel = ModelAlias;
pub type CodexModel = ModelAlias;
pub type GeminiModel = ModelAlias;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpenAiCompatibilityModel {
    pub name: String,
    pub alias: String,
    pub display_name: String,
    pub force_mapping: bool,
    pub image: bool,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub thinking: Option<ThinkingSupport>,
}

/// Returns a stable hash for OpenAI-compatible models.
pub fn compute_openai_compat_models_hash(models: &[OpenAiCompatibilityModel]) -> String {
    hash_joined(model_routing_keys(models.iter().filter_map(|model| {
        let name = model.name.trim();
        let alias = model.alias.trim();
        if name.is_empty() && alias.is_empty() {
            return None;
        }
        Some(format!(
            "{}|{}|{}|image={}|force-mapping={}|input={}|output={}{}",
            name.to_ascii_lowercase(),
            alias.to_ascii_lowercase(),
            model.display_name.trim(),
            model.image,
            model.force_mapping,
            normalize_modalities(&model.input_modalities).join(","),
            normalize_modalities(&model.output_modalities).join(","),
            thinking_hash_suffix(model.thinking.as_ref()),
        ))
    })))
}

/// Returns a stable hash for Vertex-compatible models.
pub fn compute_vertex_compat_models_hash(models: &[VertexCompatModel]) -> String {
    compute_alias_models_hash(models)
}

/// Returns a stable hash for Claude model aliases.
pub fn compute_claude_models_hash(models: &[ClaudeModel]) -> String {
    compute_alias_models_hash(models)
}

/// Returns a stable hash for Codex model aliases.
pub fn compute_codex_models_hash(models: &[CodexModel]) -> String {
    compute_alias_models_hash(models)
}

/// Returns a stable hash for Gemini model aliases.
pub fn compute_gemini_models_hash(models: &[GeminiModel]) -> String {
    compute_alias_models_hash(models)
}

fn compute_alias_models_hash(models: &[ModelAlias]) -> String {
    hash_joined(model_routing_keys(models.iter().filter_map(|model| {
        let name = model.name.trim();
        let alias = model.alias.trim();
        if name.is_empty() && alias.is_empty() {
            return None;
        }
        Some(format!(
            "{}|{}|{}|force-mapping={}{}",
            name.to_ascii_lowercase(),
            alias.to_ascii_lowercase(),
            model.display_name.trim(),
            model.force_mapping,
            thinking_hash_suffix(model.thinking.as_ref()),
        ))
    })))
}

fn normalize_modalities(raw: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(raw.len());
    for value in raw {
        let value = value.trim().to_ascii_lowercase();
        if !value.is_empty() && !normalized.contains(&value) {
            normalized.push(value);
        }
    }
    normalized
}

fn thinking_hash_suffix(support: Option<&ThinkingSupport>) -> String {
    let encoded =
        serde_json::to_string(&support).expect("ThinkingSupport serialization is infallible");
    format!("|thinking={encoded}")
}

fn model_routing_keys(keys: impl IntoIterator<Item = String>) -> Vec<String> {
    keys.into_iter().collect()
}

fn hash_joined(keys: Vec<String>) -> String {
    if keys.is_empty() {
        return String::new();
    }
    format!("{:x}", Sha256::digest(keys.join("\n").as_bytes()))
}
