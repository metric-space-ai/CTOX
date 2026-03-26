use anyhow::Context;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

const DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH: &str = "runtime/vllm_serve.env";

pub const DEFAULT_RUNTIME_PID_RELATIVE_PATH: &str = "runtime/vllm_serve.pid";
pub const DEFAULT_RUNTIME_LOG_RELATIVE_PATH: &str = "runtime/vllm_serve.log";

pub fn runtime_config_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH)
}

pub fn runtime_pid_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_RUNTIME_PID_RELATIVE_PATH)
}

pub fn runtime_log_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_RUNTIME_LOG_RELATIVE_PATH)
}

pub fn load_runtime_env_map(root: &Path) -> Result<BTreeMap<String, String>> {
    let path = runtime_config_path(root);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read runtime config {}", path.display()))?;
    Ok(parse_env_map(&raw))
}

pub fn save_runtime_env_map(root: &Path, env_map: &BTreeMap<String, String>) -> Result<()> {
    let path = runtime_config_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create runtime config dir {}", parent.display()))?;
    }
    let mut output = String::new();
    for (key, value) in env_map {
        if key.trim().is_empty() {
            continue;
        }
        output.push_str(key);
        output.push('=');
        output.push_str(&escape_env_value(value));
        output.push('\n');
    }
    std::fs::write(&path, output)
        .with_context(|| format!("failed to write runtime config {}", path.display()))
}

pub fn env_or_config(root: &Path, key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            load_runtime_env_map(root)
                .ok()
                .and_then(|map| map.get(key).cloned())
                .filter(|value| !value.trim().is_empty())
        })
}

pub fn configured_chat_model(root: &Path) -> Option<String> {
    std::env::var("CTOX_CHAT_MODEL_BASE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env_or_config(root, "CTOX_CHAT_MODEL_BASE")
                .or_else(|| env_or_config(root, "CTOX_CHAT_MODEL"))
        })
        .filter(|value| !value.trim().is_empty())
}

pub fn effective_chat_model(root: &Path) -> Option<String> {
    env_or_config(root, "CTOX_ACTIVE_MODEL")
        .or_else(|| configured_chat_model(root))
        .filter(|value| !value.trim().is_empty())
}

pub fn configured_chat_model_from_map(env_map: &BTreeMap<String, String>) -> Option<String> {
    env_map
        .get("CTOX_CHAT_MODEL_BASE")
        .or_else(|| env_map.get("CTOX_CHAT_MODEL"))
        .cloned()
        .filter(|value| !value.trim().is_empty())
}

pub fn effective_chat_model_from_map(env_map: &BTreeMap<String, String>) -> Option<String> {
    env_map
        .get("CTOX_ACTIVE_MODEL")
        .cloned()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| configured_chat_model_from_map(env_map))
}

fn parse_env_map(raw: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let normalized_key = key.trim();
        if normalized_key.is_empty() {
            continue;
        }
        out.insert(normalized_key.to_string(), unescape_env_value(value.trim()));
    }
    out
}

fn escape_env_value(value: &str) -> String {
    if value.is_empty()
        || value.chars().any(|ch| {
            !(ch.is_ascii_alphanumeric()
                || matches!(ch, '_' | '-' | '.' | '/' | ':' | ',' | '@' | '%' | '+'))
        })
    {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_string()
    }
}

fn unescape_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut output = String::new();
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                if let Some(next) = chars.next() {
                    output.push(next);
                }
            } else {
                output.push(ch);
            }
        }
        output
    } else {
        trimmed.to_string()
    }
}
