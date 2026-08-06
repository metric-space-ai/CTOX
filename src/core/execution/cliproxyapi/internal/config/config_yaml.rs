// ref: internal/config/config_yaml.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde::Serialize;

use super::config_load::TypedConfigSink;

pub fn save_config_preserve_comments<T: Serialize>(
    document: &dyn TypedConfigSink,
    config: &T,
) -> Result<(), ConfigYamlError> {
    let existing = document.read().map_err(|error| ConfigYamlError::Read {
        source: document.description(),
        error,
    })?;
    let comments = String::from_utf8_lossy(&existing)
        .lines()
        .take_while(|line| line.trim().is_empty() || line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let mut encoded = serde_yaml::to_string(config).map_err(ConfigYamlError::Encode)?;
    if !comments.is_empty() {
        encoded = format!("{comments}\n{encoded}");
    }
    document
        .write(&normalize_comment_indentation(encoded.as_bytes()))
        .map_err(|error| ConfigYamlError::Write {
            source: document.description(),
            error,
        })
}

pub fn update_nested_scalar(
    document: &dyn TypedConfigSink,
    key_path: &[&str],
    value: &str,
) -> Result<(), ConfigYamlError> {
    if key_path.is_empty() {
        return Err(ConfigYamlError::InvalidPath);
    }
    let bytes = document.read().map_err(|error| ConfigYamlError::Read {
        source: document.description(),
        error,
    })?;
    let mut root =
        serde_yaml::from_slice::<serde_yaml::Value>(&bytes).map_err(ConfigYamlError::Encode)?;
    let mut current = &mut root;
    for key in &key_path[..key_path.len() - 1] {
        let mapping = current
            .as_mapping_mut()
            .ok_or(ConfigYamlError::InvalidPath)?;
        current = mapping
            .entry(serde_yaml::Value::String((*key).to_owned()))
            .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));
    }
    current
        .as_mapping_mut()
        .ok_or(ConfigYamlError::InvalidPath)?
        .insert(
            serde_yaml::Value::String(key_path[key_path.len() - 1].to_owned()),
            serde_yaml::Value::String(value.to_owned()),
        );
    let encoded = serde_yaml::to_string(&root).map_err(ConfigYamlError::Encode)?;
    document
        .write(encoded.as_bytes())
        .map_err(|error| ConfigYamlError::Write {
            source: document.description(),
            error,
        })
}

#[must_use]
pub fn normalize_comment_indentation(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for line in data.split_inclusive(|byte| *byte == b'\n') {
        let first = line.iter().position(|byte| !matches!(byte, b' ' | b'\t'));
        if first.is_some_and(|index| line[index] == b'#') {
            out.extend_from_slice(&line[first.unwrap_or(0)..]);
        } else {
            out.extend_from_slice(line);
        }
    }
    out
}

#[derive(Debug)]
pub enum ConfigYamlError {
    InvalidPath,
    Encode(serde_yaml::Error),
    Read {
        source: String,
        error: std::io::Error,
    },
    Write {
        source: String,
        error: std::io::Error,
    },
}

impl fmt::Display for ConfigYamlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath => formatter.write_str("invalid nested configuration path"),
            Self::Encode(error) => write!(formatter, "encode configuration YAML: {error}"),
            Self::Read { source, error } => {
                write!(formatter, "read configuration document {source:?}: {error}")
            }
            Self::Write { source, error } => {
                write!(
                    formatter,
                    "write configuration document {source:?}: {error}"
                )
            }
        }
    }
}

impl std::error::Error for ConfigYamlError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::io;
    use std::sync::Mutex;

    use crate::internal::config::{TypedConfigSink, TypedConfigSource};

    struct MemoryDocument(Mutex<Vec<u8>>);

    impl TypedConfigSource for MemoryDocument {
        fn read(&self) -> io::Result<Vec<u8>> {
            Ok(self.0.lock().unwrap().clone())
        }

        fn description(&self) -> String {
            "memory".into()
        }
    }

    impl TypedConfigSink for MemoryDocument {
        fn write(&self, data: &[u8]) -> io::Result<()> {
            let mut buffer = self.0.lock().unwrap();
            buffer.clear();
            buffer.extend_from_slice(data);
            Ok(())
        }
    }

    #[derive(Serialize)]
    struct Example {
        enabled: bool,
    }

    #[test]
    fn typed_save_and_nested_update_do_not_use_external_authority() {
        let document = MemoryDocument(Mutex::new(b"# retained\nremote:\n  value: old\n".to_vec()));
        update_nested_scalar(&document, &["remote", "value"], "new").unwrap();
        let value: serde_yaml::Value = serde_yaml::from_slice(&document.read().unwrap()).unwrap();
        assert_eq!(value["remote"]["value"].as_str(), Some("new"));
        save_config_preserve_comments(&document, &Example { enabled: true }).unwrap();
        assert!(String::from_utf8(document.read().unwrap())
            .unwrap()
            .contains("enabled: true"));
    }
}
