use anyhow::Context;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

const DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH: &str = "runtime/engine.env";

pub fn runtime_config_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH)
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

pub fn effective_runtime_env_map(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut env_map = load_runtime_env_map(root)?;
    for (key, value) in std::env::vars() {
        if !is_runtime_env_key(&key) || value.trim().is_empty() {
            continue;
        }
        env_map.insert(key, value);
    }
    Ok(env_map)
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

pub fn config_flag(root: &Path, key: &str) -> bool {
    env_or_config(root, key)
        .as_deref()
        .and_then(parse_boolish)
        .unwrap_or(false)
}

pub fn auxiliary_backend_enabled(root: &Path, role_prefix: &str) -> bool {
    if config_flag(root, "CTOX_DISABLE_AUXILIARY_BACKENDS") {
        return false;
    }
    let disable_key = format!("CTOX_DISABLE_{role_prefix}_BACKEND");
    if config_flag(root, &disable_key) {
        return false;
    }
    let enable_key = format!("CTOX_ENABLE_{role_prefix}_BACKEND");
    if let Some(value) = env_or_config(root, &enable_key) {
        return parse_boolish(&value).unwrap_or(true);
    }
    let model_key = format!("CTOX_{role_prefix}_MODEL");
    if let Some(value) = env_or_config(root, &model_key) {
        return !is_disabled_selector(&value);
    }
    true
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

fn is_runtime_env_key(key: &str) -> bool {
    key.starts_with("CTOX_")
        || key.starts_with("CTO_")
        || key.starts_with("OPENAI_")
        || key.starts_with("CODEX_")
}

fn parse_boolish(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn is_disabled_selector(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "none" | "null" | "disabled" | "disable"
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_root() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ctox-runtime-env-test-{unique}"));
        std::fs::create_dir_all(path.join("runtime")).unwrap();
        path
    }

    #[test]
    fn auxiliary_backend_enabled_defaults_to_true() {
        let root = make_temp_root();
        assert!(auxiliary_backend_enabled(&root, "EMBEDDING"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn auxiliary_backend_enabled_honors_global_disable() {
        let root = make_temp_root();
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_DISABLE_AUXILIARY_BACKENDS".to_string(),
            "1".to_string(),
        );
        save_runtime_env_map(&root, &env_map).unwrap();
        assert!(!auxiliary_backend_enabled(&root, "EMBEDDING"));
        assert!(!auxiliary_backend_enabled(&root, "STT"));
        assert!(!auxiliary_backend_enabled(&root, "TTS"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn auxiliary_backend_enabled_honors_disabled_model_selector() {
        let root = make_temp_root();
        let mut env_map = BTreeMap::new();
        env_map.insert("CTOX_STT_MODEL".to_string(), "disabled".to_string());
        save_runtime_env_map(&root, &env_map).unwrap();
        assert!(!auxiliary_backend_enabled(&root, "STT"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
