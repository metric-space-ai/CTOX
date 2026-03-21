use anyhow::Context;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

pub fn load_json<T>(path: &Path, default: T) -> T
where
    T: DeserializeOwned,
{
    if !path.exists() {
        return default;
    }

    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return default,
    };

    serde_json::from_str(&text).unwrap_or(default)
}

pub fn save_json<T>(path: &Path, payload: &T) -> anyhow::Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory for {}", path.display()))?;
    }

    let text = serde_json::to_string_pretty(payload)?;
    let tmp_path = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&tmp_path, format!("{text}\n"))
        .with_context(|| format!("failed to write temporary file for {}", path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to atomically replace {}", path.display()))?;
    Ok(())
}

pub fn append_jsonl<T>(path: &Path, payload: &T) -> anyhow::Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory for {}", path.display()))?;
    }

    let line = serde_json::to_string(payload)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(line.as_bytes())
        .with_context(|| format!("failed to append line to {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("failed to append newline to {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

pub fn load_jsonl<T>(path: &Path) -> Vec<T>
where
    T: DeserializeOwned,
{
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };

    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}
