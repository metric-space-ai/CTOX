// ref: internal/runtime/executor/helps/payload_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::internal::config::DisableImageGenerationMode;
use crate::internal::thinking::parse_suffix;
use crate::sdk::cliproxy::executor::Options;

pub type PayloadHeaders = BTreeMap<String, Vec<String>>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadModelRule {
    pub name: String,
    pub protocol: String,
    pub from_protocol: String,
    pub headers: BTreeMap<String, String>,
    pub matches: Vec<BTreeMap<String, Value>>,
    pub not_matches: Vec<BTreeMap<String, Value>>,
    pub exist: Vec<String>,
    pub not_exist: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadRule {
    pub models: Vec<PayloadModelRule>,
    pub params: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadFilterRule {
    pub models: Vec<PayloadModelRule>,
    pub params: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadRules {
    pub default: Vec<PayloadRule>,
    pub default_raw: Vec<PayloadRule>,
    pub override_values: Vec<PayloadRule>,
    pub override_raw: Vec<PayloadRule>,
    pub filter: Vec<PayloadFilterRule>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PayloadApplyConfig {
    pub disable_image_generation: DisableImageGenerationMode,
    pub rules: PayloadRules,
}

pub fn is_images_endpoint_request_path(path: &str) -> bool {
    let path = path.trim();
    if path.is_empty() {
        return false;
    }
    path == "/v1/images/generations"
        || path == "/v1/images/edits"
        || path.ends_with("/v1/images/generations")
        || path.ends_with("/v1/images/edits")
        || path.ends_with("/images/generations")
        || path.ends_with("/images/edits")
}

pub fn should_strip_image_generation(mode: DisableImageGenerationMode, request_path: &str) -> bool {
    match mode {
        DisableImageGenerationMode::All => true,
        DisableImageGenerationMode::Chat => !is_images_endpoint_request_path(request_path),
        DisableImageGenerationMode::Off | DisableImageGenerationMode::Passthrough => false,
    }
}

/// Applies only the accepted image-generation slice of upstream's much larger
/// payload-rule helper. No-op and invalid-JSON paths preserve exact bytes.
pub fn apply_disable_image_generation_with_root(
    payload: &[u8],
    root: &str,
    mode: DisableImageGenerationMode,
    request_path: &str,
) -> Vec<u8> {
    if payload.is_empty() || !should_strip_image_generation(mode, request_path) {
        return payload.to_vec();
    }
    let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let Some(object) = object_at_root_mut(&mut value, root) else {
        return payload.to_vec();
    };
    let mut changed = false;
    if let Some(Value::Array(tools)) = object.get_mut("tools") {
        let original_len = tools.len();
        tools.retain(|tool| tool.get("type").and_then(Value::as_str) != Some("image_generation"));
        changed |= tools.len() != original_len;
    }
    let remove_choice = object
        .get("tool_choice")
        .is_some_and(tool_choice_is_image_generation);
    if remove_choice {
        object.remove("tool_choice");
        changed = true;
    }
    if !changed {
        return payload.to_vec();
    }
    serde_json::to_vec(&value).unwrap_or_else(|_| payload.to_vec())
}

#[allow(clippy::too_many_arguments)]
pub fn apply_payload_config_with_root(
    config: &PayloadApplyConfig,
    model: &str,
    protocol: &str,
    root: &str,
    payload: &[u8],
    original: Option<&[u8]>,
    requested_model: &str,
    request_path: &str,
) -> Vec<u8> {
    apply_payload_config_with_request(
        config,
        model,
        protocol,
        "",
        root,
        payload,
        original,
        requested_model,
        request_path,
        &PayloadHeaders::new(),
    )
}

/// Applies the complete upstream payload rule order. The Go `any` values are
/// represented as `serde_json::Value`, and HTTP headers use the proxy's typed
/// multi-value map rather than `http.Header` globals.
#[allow(clippy::too_many_arguments)]
pub fn apply_payload_config_with_request(
    config: &PayloadApplyConfig,
    model: &str,
    protocol: &str,
    from_protocol: &str,
    root: &str,
    payload: &[u8],
    original: Option<&[u8]>,
    requested_model: &str,
    request_path: &str,
    headers: &PayloadHeaders,
) -> Vec<u8> {
    if payload.is_empty() {
        return payload.to_vec();
    }
    let mut out = apply_disable_image_generation_with_root(
        payload,
        root,
        config.disable_image_generation,
        request_path,
    );
    let candidates = payload_model_candidates(model, requested_model);
    if candidates.is_empty() {
        return out;
    }
    let source = original
        .filter(|value| !value.is_empty())
        .unwrap_or(payload);
    let Ok(source_value) = serde_json::from_slice::<Value>(source) else {
        return out;
    };
    let mut applied_defaults = BTreeSet::new();

    for (rules, raw) in [
        (&config.rules.default, false),
        (&config.rules.default_raw, true),
    ] {
        for rule in rules {
            let Ok(mut document) = serde_json::from_slice::<Value>(&out) else {
                return out;
            };
            if !payload_model_rules_match(
                &rule.models,
                protocol,
                from_protocol,
                headers,
                &document,
                root,
                &candidates,
            ) {
                continue;
            }
            let mut changed = false;
            for (path, configured) in &rule.params {
                let full_path = build_payload_path(root, path);
                for resolved in resolve_payload_rule_paths(&document, &full_path) {
                    if value_at_path(&source_value, &resolved).is_some()
                        || !applied_defaults.insert(resolved.clone())
                    {
                        continue;
                    }
                    let Some(value) = configured_payload_value(configured, raw) else {
                        continue;
                    };
                    changed |= set_value_at_path(&mut document, &resolved, value);
                }
            }
            if changed {
                out = serde_json::to_vec(&document).unwrap_or(out);
            }
        }
    }

    for (rules, raw) in [
        (&config.rules.override_values, false),
        (&config.rules.override_raw, true),
    ] {
        for rule in rules {
            let Ok(mut document) = serde_json::from_slice::<Value>(&out) else {
                return out;
            };
            if !payload_model_rules_match(
                &rule.models,
                protocol,
                from_protocol,
                headers,
                &document,
                root,
                &candidates,
            ) {
                continue;
            }
            let mut changed = false;
            for (path, configured) in &rule.params {
                let Some(value) = configured_payload_value(configured, raw) else {
                    continue;
                };
                let full_path = build_payload_path(root, path);
                for resolved in resolve_payload_rule_paths(&document, &full_path) {
                    if value_at_path(&document, &resolved) != Some(&value) {
                        changed |= set_value_at_path(&mut document, &resolved, value.clone());
                    }
                }
            }
            if changed {
                out = serde_json::to_vec(&document).unwrap_or(out);
            }
        }
    }

    for rule in &config.rules.filter {
        let Ok(mut document) = serde_json::from_slice::<Value>(&out) else {
            return out;
        };
        if !payload_model_rules_match(
            &rule.models,
            protocol,
            from_protocol,
            headers,
            &document,
            root,
            &candidates,
        ) {
            continue;
        }
        let mut changed = false;
        for path in &rule.params {
            let paths = resolve_payload_rule_paths(&document, &build_payload_path(root, path));
            for path in paths.iter().rev() {
                changed |= delete_value_at_path(&mut document, path);
            }
        }
        if changed {
            out = serde_json::to_vec(&document).unwrap_or(out);
        }
    }
    out
}

fn configured_payload_value(value: &Value, raw: bool) -> Option<Value> {
    if !raw {
        return Some(value.clone());
    }
    match value {
        Value::String(raw) => serde_json::from_str(raw).ok(),
        value => Some(value.clone()),
    }
}

fn payload_model_rules_match(
    rules: &[PayloadModelRule],
    protocol: &str,
    from_protocol: &str,
    headers: &PayloadHeaders,
    payload: &Value,
    root: &str,
    models: &[String],
) -> bool {
    models.iter().any(|model| {
        rules.iter().any(|rule| {
            !rule.name.trim().is_empty()
                && (rule.protocol.trim().is_empty()
                    || protocol.trim().is_empty()
                    || rule.protocol.trim().eq_ignore_ascii_case(protocol.trim()))
                && payload_from_protocol_matches(&rule.from_protocol, from_protocol)
                && payload_headers_match(headers, &rule.headers)
                && match_model_pattern(&rule.name, model)
                && payload_conditions_match(payload, root, rule)
        })
    })
}

fn payload_conditions_match(payload: &Value, root: &str, rule: &PayloadModelRule) -> bool {
    rule.matches.iter().all(|condition| {
        condition.iter().all(|(path, expected)| {
            path.trim().is_empty()
                || payload_path_matches_value(payload, &build_payload_path(root, path), expected)
        })
    }) && rule.not_matches.iter().all(|condition| {
        condition.iter().all(|(path, expected)| {
            path.trim().is_empty()
                || !payload_path_matches_value(payload, &build_payload_path(root, path), expected)
        })
    }) && rule.exist.iter().all(|path| {
        path.trim().is_empty() || payload_path_exists(payload, &build_payload_path(root, path))
    }) && rule.not_exist.iter().all(|path| {
        path.trim().is_empty() || !payload_path_exists(payload, &build_payload_path(root, path))
    })
}

fn payload_path_matches_value(payload: &Value, path: &str, expected: &Value) -> bool {
    resolve_payload_rule_paths(payload, path)
        .iter()
        .filter_map(|resolved| value_at_path(payload, resolved))
        .any(|actual| actual == expected)
}

fn payload_path_exists(payload: &Value, path: &str) -> bool {
    resolve_payload_rule_paths(payload, path)
        .iter()
        .filter_map(|resolved| value_at_path(payload, resolved))
        .any(|value| !value.is_null())
}

fn payload_from_protocol_matches(pattern: &str, from_protocol: &str) -> bool {
    let pattern = normalize_payload_from_protocol(pattern);
    pattern.is_empty()
        || (!from_protocol.trim().is_empty()
            && pattern.eq_ignore_ascii_case(&normalize_payload_from_protocol(from_protocol)))
}

fn normalize_payload_from_protocol(protocol: &str) -> String {
    match protocol.trim().to_ascii_lowercase().as_str() {
        "openai-response" | "openai-responses" | "response" => "responses".to_owned(),
        value => value.to_owned(),
    }
}

fn payload_headers_match(headers: &PayloadHeaders, rules: &BTreeMap<String, String>) -> bool {
    rules.iter().all(|(key, pattern)| {
        key.trim().is_empty()
            || headers
                .iter()
                .filter(|(header, _)| header.eq_ignore_ascii_case(key.trim()))
                .flat_map(|(_, values)| values)
                .any(|value| match_model_pattern(pattern, value))
    })
}

fn payload_model_candidates(model: &str, requested_model: &str) -> Vec<String> {
    let mut candidates = Vec::with_capacity(3);
    let mut seen = BTreeSet::new();
    let mut add = |value: &str| {
        let value = value.trim();
        if !value.is_empty() && seen.insert(value.to_ascii_lowercase()) {
            candidates.push(value.to_owned());
        }
    };
    add(model);
    let parsed = parse_suffix(requested_model.trim());
    add(&parsed.model_name);
    if parsed.has_suffix {
        add(requested_model);
    }
    candidates
}

fn build_payload_path(root: &str, path: &str) -> String {
    let root = root.trim();
    let path = path.trim().trim_start_matches('.');
    match (root.is_empty(), path.is_empty()) {
        (true, _) => path.to_owned(),
        (_, true) => root.to_owned(),
        _ => format!("{root}.{path}"),
    }
}

fn resolve_payload_rule_paths(payload: &Value, path: &str) -> Vec<String> {
    let parts = split_payload_rule_path(path.trim());
    if parts.is_empty() {
        return Vec::new();
    }
    let mut paths = vec![String::new()];
    for part in parts {
        if let Some((query, all_matches)) = parse_payload_query_path_part(&part) {
            let mut next = Vec::new();
            for base in &paths {
                let Some(items) = value_at_path(payload, base).and_then(Value::as_array) else {
                    continue;
                };
                for (index, item) in items.iter().enumerate() {
                    if payload_query_matches(item, query) {
                        next.push(append_payload_path_part(base, &index.to_string()));
                        if !all_matches {
                            break;
                        }
                    }
                }
            }
            paths = next;
        } else {
            for base in &mut paths {
                *base = append_payload_path_part(base, &part);
            }
        }
        if paths.is_empty() {
            break;
        }
    }
    paths
}

fn split_payload_rule_path(path: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in path.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if let Some(expected) = quote {
            if character == expected {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => quote = Some(character),
            '(' => depth += 1,
            ')' => depth = (depth - 1).max(0),
            '.' if depth == 0 => {
                parts.push(path[start..index].to_owned());
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(path[start..].to_owned());
    parts
}

fn parse_payload_query_path_part(part: &str) -> Option<(&str, bool)> {
    let body = part.strip_prefix("#(")?;
    let close = body.rfind(')')?;
    let suffix = &body[close + 1..];
    if !matches!(suffix, "" | "#") {
        return None;
    }
    Some((body[..close].trim(), suffix == "#"))
}

fn append_payload_path_part(path: &str, part: &str) -> String {
    match (path.is_empty(), part.is_empty()) {
        (true, _) => part.to_owned(),
        (_, true) => path.to_owned(),
        _ => format!("{path}.{part}"),
    }
}

fn payload_query_matches(item: &Value, query: &str) -> bool {
    split_logical(query, "||").iter().any(|part| {
        split_logical(part, "&&")
            .iter()
            .all(|term| payload_query_term(item, term))
    })
}

fn split_logical<'a>(query: &'a str, operator: &str) -> Vec<&'a str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let bytes = query.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let character = bytes[index] as char;
        if escaped {
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if let Some(expected) = quote {
            if character == expected {
                quote = None;
            }
        } else if matches!(character, '"' | '\'') {
            quote = Some(character);
        } else if query[index..].starts_with(operator) {
            parts.push(query[start..index].trim());
            index += operator.len();
            start = index;
            continue;
        }
        index += 1;
    }
    parts.push(query[start..].trim());
    parts
}

fn payload_query_term(item: &Value, term: &str) -> bool {
    let term = term.trim();
    for operator in ["!=", "==", "="] {
        if let Some(index) = term.find(operator) {
            let path = term[..index].trim();
            let raw = term[index + operator.len()..].trim();
            let expected = parse_query_value(raw);
            let equal = value_at_path(item, path).is_some_and(|actual| actual == &expected);
            return if operator == "!=" { !equal } else { equal };
        }
    }
    value_at_path(item, term).is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
}

fn parse_query_value(raw: &str) -> Value {
    let unquoted = raw
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            raw.strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        });
    unquoted.map_or_else(
        || serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
        |value| Value::String(value.to_owned()),
    )
}

fn value_at_path<'a>(document: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(document);
    }
    path.split('.').try_fold(document, |current, segment| {
        segment
            .parse::<usize>()
            .ok()
            .and_then(|index| current.as_array()?.get(index))
            .or_else(|| current.as_object()?.get(segment))
    })
}

fn set_value_at_path(document: &mut Value, path: &str, value: Value) -> bool {
    let parts = path
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    set_value_parts(document, &parts, value)
}

fn set_value_parts(document: &mut Value, parts: &[&str], value: Value) -> bool {
    let Some((head, tail)) = parts.split_first() else {
        if document != &value {
            *document = value;
            return true;
        }
        return false;
    };
    if let Ok(index) = head.parse::<usize>() {
        let Some(array) = document.as_array_mut() else {
            return false;
        };
        let Some(child) = array.get_mut(index) else {
            return false;
        };
        return set_value_parts(child, tail, value);
    }
    if !document.is_object() {
        *document = Value::Object(Map::new());
    }
    let object = document.as_object_mut().expect("object initialized");
    if tail.is_empty() {
        return object.insert((*head).to_owned(), value).as_ref() != object.get(*head);
    }
    let child = object.entry((*head).to_owned()).or_insert_with(|| {
        if tail[0].parse::<usize>().is_ok() {
            Value::Array(Vec::new())
        } else {
            Value::Object(Map::new())
        }
    });
    set_value_parts(child, tail, value)
}

fn delete_value_at_path(document: &mut Value, path: &str) -> bool {
    let mut parts = path.split('.').collect::<Vec<_>>();
    let Some(last) = parts.pop() else {
        return false;
    };
    let parent_path = parts.join(".");
    let Some(parent) = value_at_path_mut(document, &parent_path) else {
        return false;
    };
    if let Ok(index) = last.parse::<usize>() {
        let Some(array) = parent.as_array_mut() else {
            return false;
        };
        if index < array.len() {
            array.remove(index);
            return true;
        }
        false
    } else {
        parent
            .as_object_mut()
            .and_then(|object| object.remove(last))
            .is_some()
    }
}

fn value_at_path_mut<'a>(document: &'a mut Value, path: &str) -> Option<&'a mut Value> {
    if path.is_empty() {
        return Some(document);
    }
    let mut current = document;
    for segment in path.split('.') {
        current = if let Ok(index) = segment.parse::<usize>() {
            current.as_array_mut()?.get_mut(index)?
        } else {
            current.as_object_mut()?.get_mut(segment)?
        };
    }
    Some(current)
}

#[must_use]
pub fn payload_requested_model(options: &Options, fallback: &str) -> String {
    options
        .metadata
        .requested_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.trim())
        .to_owned()
}

#[must_use]
pub fn payload_request_path(options: &Options) -> String {
    options
        .metadata
        .request_path
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

#[must_use]
pub fn match_model_pattern(pattern: &str, model: &str) -> bool {
    let pattern = pattern.trim().as_bytes();
    let model = model.trim().as_bytes();
    if pattern.is_empty() {
        return false;
    }
    let (mut pattern_index, mut model_index) = (0, 0);
    let mut star = None;
    let mut matched = 0;
    while model_index < model.len() {
        if pattern.get(pattern_index) == model.get(model_index) {
            pattern_index += 1;
            model_index += 1;
        } else if pattern.get(pattern_index) == Some(&b'*') {
            star = Some(pattern_index);
            matched = model_index;
            pattern_index += 1;
        } else if let Some(star_index) = star {
            pattern_index = star_index + 1;
            matched += 1;
            model_index = matched;
        } else {
            return false;
        }
    }
    while pattern.get(pattern_index) == Some(&b'*') {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

fn object_at_root_mut<'a>(
    value: &'a mut Value,
    root: &str,
) -> Option<&'a mut serde_json::Map<String, Value>> {
    let mut current = value;
    for segment in root
        .split('.')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        current = current.get_mut(segment)?;
    }
    current.as_object_mut()
}

fn tool_choice_is_image_generation(choice: &Value) -> bool {
    if let Some(choice) = choice.as_str() {
        return choice.trim().eq_ignore_ascii_case("image_generation");
    }
    let Some(choice) = choice.as_object() else {
        return false;
    };
    let choice_type = choice
        .get("type")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    choice_type.eq_ignore_ascii_case("image_generation")
        || (choice_type.eq_ignore_ascii_case("tool")
            && choice
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .is_some_and(|name| name.eq_ignore_ascii_case("image_generation")))
}
