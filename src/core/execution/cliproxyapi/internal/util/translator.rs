// ref: internal/util/translator.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use serde_json::value::RawValue;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::sanitize_function_name;

pub fn walk_json_field_paths(raw: &[u8], field: &str) -> Result<Vec<String>, JsonTransformError> {
    let value = serde_json::from_slice::<Value>(raw).map_err(JsonTransformError::InvalidJson)?;
    let mut paths = Vec::new();
    walk_value(&value, "", field, &mut paths);
    Ok(paths)
}

fn walk_value(value: &Value, path: &str, field: &str, paths: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let child = join_path(path, &escape_gjson_path_key(key));
                if key == field {
                    paths.push(child.clone());
                }
                walk_value(value, &child, field, paths);
            }
        }
        Value::Array(array) => {
            for (index, value) in array.iter().enumerate() {
                let child = join_path(path, &index.to_string());
                walk_value(value, &child, field, paths);
            }
        }
        _ => {}
    }
}

fn join_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_owned()
    } else {
        format!("{parent}.{child}")
    }
}

fn escape_gjson_path_key(key: &str) -> String {
    let mut escaped = String::with_capacity(key.len());
    for character in key.chars() {
        if matches!(character, '.' | '*' | '?' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

pub fn rename_key(
    json: &str,
    old_key_path: &str,
    new_key_path: &str,
) -> Result<String, JsonTransformError> {
    let mut root = serde_json::from_str::<Value>(json).map_err(JsonTransformError::InvalidJson)?;
    let old_path = parse_path(old_key_path)?;
    let new_path = parse_path(new_key_path)?;
    let value = get_path(&root, &old_path)
        .cloned()
        .ok_or_else(|| JsonTransformError::MissingOldKey(old_key_path.to_owned()))?;
    set_path(&mut root, &new_path, value)?;
    delete_path(&mut root, &old_path)?;
    serde_json::to_string(&root).map_err(JsonTransformError::InvalidJson)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PathSegment {
    Raw(String),
}

fn parse_path(path: &str) -> Result<Vec<PathSegment>, JsonTransformError> {
    if path.is_empty() {
        return Err(JsonTransformError::InvalidPath(path.to_owned()));
    }
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for character in path.chars() {
        if escaped {
            current.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '.' {
            push_segment(&mut segments, &mut current, path)?;
        } else {
            current.push(character);
        }
    }
    if escaped {
        return Err(JsonTransformError::InvalidPath(path.to_owned()));
    }
    push_segment(&mut segments, &mut current, path)?;
    Ok(segments)
}

fn push_segment(
    output: &mut Vec<PathSegment>,
    current: &mut String,
    original: &str,
) -> Result<(), JsonTransformError> {
    if current.is_empty() {
        return Err(JsonTransformError::InvalidPath(original.to_owned()));
    }
    let value = std::mem::take(current);
    output.push(PathSegment::Raw(value));
    Ok(())
}

fn get_path<'a>(mut value: &'a Value, path: &[PathSegment]) -> Option<&'a Value> {
    for segment in path {
        value = match (segment, value) {
            (PathSegment::Raw(key), Value::Object(object)) => object.get(key)?,
            (PathSegment::Raw(index), Value::Array(array)) => {
                array.get(index.parse::<usize>().ok()?)?
            }
            _ => return None,
        };
    }
    Some(value)
}

fn set_path(
    value: &mut Value,
    path: &[PathSegment],
    replacement: Value,
) -> Result<(), JsonTransformError> {
    let (last, parents) = path
        .split_last()
        .ok_or_else(|| JsonTransformError::InvalidPath(String::new()))?;
    let mut current = value;
    for segment in parents {
        current = match (segment, current) {
            (PathSegment::Raw(key), Value::Object(object)) => object
                .get_mut(key)
                .ok_or_else(|| JsonTransformError::InvalidPath(key.clone()))?,
            (PathSegment::Raw(index), Value::Array(array)) => array
                .get_mut(
                    index
                        .parse::<usize>()
                        .map_err(|_| JsonTransformError::InvalidPath(index.clone()))?,
                )
                .ok_or_else(|| JsonTransformError::InvalidPath(index.clone()))?,
            (PathSegment::Raw(segment), _) => {
                return Err(JsonTransformError::InvalidPath(segment.clone()));
            }
        };
    }
    match (last, current) {
        (PathSegment::Raw(key), Value::Object(object)) => {
            object.insert(key.clone(), replacement);
        }
        (PathSegment::Raw(index), Value::Array(array)) => {
            let index_value = index
                .parse::<usize>()
                .map_err(|_| JsonTransformError::InvalidPath(index.clone()))?;
            *array
                .get_mut(index_value)
                .ok_or_else(|| JsonTransformError::InvalidPath(index.clone()))? = replacement;
        }
        (PathSegment::Raw(segment), _) => {
            return Err(JsonTransformError::InvalidPath(segment.clone()));
        }
    }
    Ok(())
}

fn delete_path(value: &mut Value, path: &[PathSegment]) -> Result<(), JsonTransformError> {
    let (last, parents) = path
        .split_last()
        .ok_or_else(|| JsonTransformError::InvalidPath(String::new()))?;
    let mut current = value;
    for segment in parents {
        current = match (segment, current) {
            (PathSegment::Raw(key), Value::Object(object)) => object
                .get_mut(key)
                .ok_or_else(|| JsonTransformError::InvalidPath(key.clone()))?,
            (PathSegment::Raw(index), Value::Array(array)) => array
                .get_mut(
                    index
                        .parse::<usize>()
                        .map_err(|_| JsonTransformError::InvalidPath(index.clone()))?,
                )
                .ok_or_else(|| JsonTransformError::InvalidPath(index.clone()))?,
            (PathSegment::Raw(segment), _) => {
                return Err(JsonTransformError::InvalidPath(segment.clone()));
            }
        };
    }
    match (last, current) {
        (PathSegment::Raw(key), Value::Object(object)) => {
            object
                .remove(key)
                .ok_or_else(|| JsonTransformError::MissingOldKey(key.clone()))?;
        }
        (PathSegment::Raw(index), Value::Array(array)) => {
            let index_value = index
                .parse::<usize>()
                .map_err(|_| JsonTransformError::InvalidPath(index.clone()))?;
            if index_value >= array.len() {
                return Err(JsonTransformError::InvalidPath(index.clone()));
            }
            array.remove(index_value);
        }
        (PathSegment::Raw(segment), _) => {
            return Err(JsonTransformError::InvalidPath(segment.clone()));
        }
    }
    Ok(())
}

#[must_use]
pub fn fix_json(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let runes = input.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    while index < runes.len() {
        let character = runes[index];
        if in_double {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_double = false;
            }
        } else if in_single {
            if escaped {
                escaped = false;
                match character {
                    'n' | 'r' | 't' | 'b' | 'f' | '/' | '"' => {
                        output.push('\\');
                        output.push(character);
                    }
                    '\\' => output.push_str("\\\\"),
                    '\'' => output.push('\''),
                    'u' => {
                        output.push_str("\\u");
                        for _ in 0..4 {
                            let Some(next) = runes.get(index + 1).copied() else {
                                break;
                            };
                            if !next.is_ascii_hexdigit() {
                                break;
                            }
                            output.push(next);
                            index += 1;
                        }
                    }
                    _ => {
                        output.push('\\');
                        output.push(character);
                    }
                }
            } else if character == '\\' {
                escaped = true;
            } else if character == '\'' {
                output.push('"');
                in_single = false;
            } else {
                if character == '"' {
                    output.push('\\');
                }
                output.push(character);
            }
        } else if character == '"' {
            in_double = true;
            output.push(character);
        } else if character == '\'' {
            in_single = true;
            output.push('"');
        } else {
            output.push(character);
        }
        index += 1;
    }
    if in_single {
        output.push('"');
    }
    output
}

#[must_use]
pub fn canonical_tool_name(name: &str) -> String {
    name.trim().trim_start_matches('_').to_ascii_lowercase()
}

#[must_use]
pub fn tool_name_map_from_claude_request(raw_json: &[u8]) -> BTreeMap<String, String> {
    let Ok(root) = serde_json::from_slice::<Value>(raw_json) else {
        return BTreeMap::new();
    };
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    let mut output = BTreeMap::new();
    for tool in tools {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| tool.pointer("/function/name").and_then(Value::as_str))
            .map(str::trim)
            .unwrap_or_default();
        let key = canonical_tool_name(name);
        if !key.is_empty() {
            output.entry(key).or_insert_with(|| name.to_owned());
        }
    }
    output
}

#[must_use]
pub fn map_tool_name(name_map: &BTreeMap<String, String>, name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    name_map
        .get(&canonical_tool_name(name))
        .filter(|mapped| !mapped.is_empty())
        .cloned()
        .unwrap_or_else(|| name.to_owned())
}

#[must_use]
pub fn sanitized_function_name_map(raw_json: &[u8]) -> HashMap<String, String> {
    let Ok(root) = serde_json::from_slice::<Value>(raw_json) else {
        return HashMap::new();
    };
    let mut unique = HashSet::new();
    for tool in root
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        collect_names(tool, &mut unique);
    }
    let mut names = unique
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    names.sort();
    let mut counts = HashMap::new();
    for name in &names {
        *counts
            .entry(sanitize_function_name(name))
            .or_insert(0_usize) += 1;
    }
    let mut used = HashMap::<String, String>::new();
    let mut output = HashMap::new();
    for name in names {
        let base = sanitize_function_name(&name);
        let mapped =
            if counts.get(&base).copied().unwrap_or_default() > 1 || used.contains_key(&base) {
                disambiguate(&base, &name, &used)
            } else {
                base
            };
        used.insert(mapped.clone(), name.clone());
        output.insert(name, mapped);
    }
    output
}

#[must_use]
pub fn map_sanitized_function_name(name_map: &HashMap<String, String>, name: &str) -> String {
    name_map
        .get(name)
        .cloned()
        .unwrap_or_else(|| sanitize_function_name(name))
}

#[must_use]
pub fn disambiguated_tool_name_map(raw_json: &[u8]) -> HashMap<String, String> {
    sanitized_function_name_map(raw_json)
        .into_iter()
        .filter(|(original, sanitized)| original != sanitized)
        .map(|(original, sanitized)| (sanitized, original))
        .collect()
}

#[must_use]
pub fn sanitized_tool_name_map(raw_json: &[u8]) -> HashMap<String, String> {
    let Ok(root) = serde_json::from_slice::<Value>(raw_json) else {
        return HashMap::new();
    };
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return HashMap::new();
    };
    let mut output = HashMap::new();
    for tool in tools {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        let sanitized = sanitize_function_name(name);
        if !name.is_empty() && sanitized != name {
            output.entry(sanitized).or_insert_with(|| name.to_owned());
        }
    }
    output
}

fn collect_names(tool: &Value, output: &mut HashSet<String>) {
    if let Some(tools) = tool.get("tools").and_then(Value::as_array) {
        for tool in tools {
            collect_names(tool, output);
        }
        return;
    }
    let mut has_declarations = false;
    for key in ["functionDeclarations", "function_declarations"] {
        if let Some(declarations) = tool.get(key).and_then(Value::as_array) {
            has_declarations = true;
            for declaration in declarations {
                if let Some(name) = declaration.get("name").and_then(Value::as_str) {
                    output.insert(name.to_owned());
                }
            }
        }
    }
    if has_declarations {
        return;
    }
    if let Some(name) = tool
        .pointer("/function/name")
        .or_else(|| tool.get("name"))
        .and_then(Value::as_str)
    {
        output.insert(name.to_owned());
    }
}

fn disambiguate(base: &str, original: &str, used: &HashMap<String, String>) -> String {
    for attempt in 0_u64.. {
        let digest = Sha256::digest(format!("{original}\0{attempt}").as_bytes());
        let suffix = format!("_{}", hex(&digest[..6]));
        let mut prefix = base.to_owned();
        prefix.truncate(prefix.len().min(64 - suffix.len()));
        let candidate = format!("{prefix}{suffix}");
        if !used.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

#[must_use]
pub fn deduplicate_function_declarations(raw: &[u8]) -> Vec<u8> {
    let Ok(declarations) = serde_json::from_slice::<Vec<&RawValue>>(raw) else {
        return raw.to_vec();
    };
    let mut seen = HashSet::new();
    let mut kept = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        let name = serde_json::from_str::<Value>(declaration.get())
            .ok()
            .and_then(|value| value.get("name").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_default();
        if name.is_empty() || seen.insert(name) {
            kept.push(declaration.get());
        }
    }
    format!("[{}]", kept.join(",")).into_bytes()
}

#[must_use]
pub fn restore_sanitized_tool_name(
    tool_name_map: &HashMap<String, String>,
    sanitized_name: &str,
) -> String {
    tool_name_map
        .get(sanitized_name)
        .cloned()
        .unwrap_or_else(|| sanitized_name.to_owned())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Debug)]
pub enum JsonTransformError {
    InvalidJson(serde_json::Error),
    InvalidPath(String),
    MissingOldKey(String),
}

impl fmt::Display for JsonTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(error) => write!(formatter, "invalid JSON: {error}"),
            Self::InvalidPath(path) => write!(formatter, "invalid JSON path {path:?}"),
            Self::MissingOldKey(path) => write!(formatter, "old key '{path}' does not exist"),
        }
    }
}

impl std::error::Error for JsonTransformError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_rename_and_fix_match_pinned_shapes() {
        let raw = br#"{"a.b":{"target":1},"items":[{"target":2}]}"#;
        assert_eq!(
            walk_json_field_paths(raw, "target").unwrap(),
            ["a\\.b.target", "items.0.target"]
        );
        let renamed = rename_key(
            std::str::from_utf8(raw).unwrap(),
            "a\\.b.target",
            "a\\.b.moved",
        )
        .unwrap();
        let value: Value = serde_json::from_str(&renamed).unwrap();
        assert_eq!(value["a.b"]["moved"], 1);
        assert!(value["a.b"].get("target").is_none());
        assert_eq!(
            fix_json("{'a': 'He said \"hi\"'}"),
            r#"{"a": "He said \"hi\""}"#
        );
        assert!(rename_key("{}", "missing", "new")
            .unwrap_err()
            .to_string()
            .contains("old key 'missing' does not exist"));
    }

    #[test]
    fn claude_and_legacy_maps_preserve_first_names() {
        let raw = br#"{"tools":[{"name":"__Read_File"},{"name":"read_file"},{"name":"read file"},{"function":{"name":"Nested Tool"}}]}"#;
        let names = tool_name_map_from_claude_request(raw);
        assert_eq!(map_tool_name(&names, "read_file"), "__Read_File");
        assert_eq!(map_tool_name(&names, "unknown"), "unknown");
        let legacy = sanitized_tool_name_map(raw);
        assert_eq!(legacy["read_file"], "read file");
    }

    #[test]
    fn deduplication_preserves_raw_numeric_lexemes() {
        let output = deduplicate_function_declarations(
            br#"[{"name":"a","x":1.00},{"name":"a","x":2},{"x":3.00}]"#,
        );
        assert_eq!(output, br#"[{"name":"a","x":1.00},{"x":3.00}]"#);
    }
}
