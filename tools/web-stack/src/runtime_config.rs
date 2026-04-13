use anyhow::Context;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

const DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH: &str = "runtime/engine.env";

pub fn runtime_config_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH)
}

pub fn env_or_config(root: &Path, key: &str) -> Option<String> {
    process_env_value(key).or_else(|| {
        load_runtime_env_map(root)
            .ok()
            .and_then(|map| map.get(key).cloned())
            .filter(|value| !value.trim().is_empty())
    })
}

fn process_env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
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

fn load_runtime_env_map(root: &Path) -> Result<BTreeMap<String, String>> {
    let path = runtime_config_path(root);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read runtime config {}", path.display()))?;
    Ok(parse_env_map(&raw))
}

fn unescape_env_value(raw: &str) -> String {
    let stripped = raw
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            raw.strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(raw);
    let mut out = String::new();
    let mut chars = stripped.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    '\'' => out.push('\''),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}
