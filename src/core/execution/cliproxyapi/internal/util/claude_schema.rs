// ref: internal/util/claude_schema.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use serde_json::value::RawValue;

const EMPTY_CLAUDE_TOOL_INPUT_SCHEMA: &[u8] = br#"{"type":"object","properties":{}}"#;
const ROOT_UNIONS: [&str; 3] = ["anyOf", "oneOf", "allOf"];

type RawObject = BTreeMap<String, String>;

/// Makes a JSON Schema compatible with Claude's object-only tool input
/// contract while preserving upstream byte semantics.
///
/// Root and property map keys are sorted like Go's `encoding/json`; untouched
/// raw values keep number spelling and nested key order, insignificant
/// whitespace is removed, and Go's HTML-safe JSON escaping is reproduced.
#[must_use]
pub fn normalize_claude_tool_input_schema(schema: &[u8]) -> Vec<u8> {
    if schema.is_empty() {
        return EMPTY_CLAUDE_TOOL_INPUT_SCHEMA.to_vec();
    }
    let decoded = String::from_utf8_lossy(schema);
    let Some(mut root) = parse_object(&decoded) else {
        return EMPTY_CLAUDE_TOOL_INPUT_SCHEMA.to_vec();
    };

    let mut properties = root
        .get("properties")
        .and_then(|raw| parse_object(raw))
        .unwrap_or_default();

    for union_name in ROOT_UNIONS {
        let Some(union_raw) = root.remove(union_name) else {
            continue;
        };
        let Ok(branches) = serde_json::from_str::<Vec<Box<RawValue>>>(&union_raw) else {
            continue;
        };
        for branch_raw in branches {
            let Some(branch) = parse_object(branch_raw.get()) else {
                continue;
            };
            if !claude_schema_can_be_object(&branch) {
                continue;
            }
            if let Some(branch_properties) =
                branch.get("properties").and_then(|raw| parse_object(raw))
            {
                for (name, property) in branch_properties {
                    properties.entry(name).or_insert(property);
                }
            }
            if union_name == "allOf" {
                merge_required(&mut root, branch.get("required").map(String::as_str));
            }
        }
    }

    root.insert("type".to_owned(), r#""object""#.to_owned());
    root.insert("properties".to_owned(), encode_object(&properties));
    encode_object(&root).into_bytes()
}

fn parse_object(raw: &str) -> Option<RawObject> {
    let parsed = serde_json::from_str::<Option<BTreeMap<String, Box<RawValue>>>>(raw).ok()??;
    Some(
        parsed
            .into_iter()
            .map(|(key, value)| (key, compact_go_json(value.get())))
            .collect(),
    )
}

fn claude_schema_can_be_object(schema: &RawObject) -> bool {
    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    if let Ok(single) = serde_json::from_str::<String>(schema_type) {
        return single == "object";
    }
    serde_json::from_str::<Vec<String>>(schema_type)
        .is_ok_and(|types| types.iter().any(|candidate| candidate == "object"))
}

fn merge_required(root: &mut RawObject, branch_required: Option<&str>) {
    let Some(branch_required) = branch_required else {
        return;
    };
    let Ok(branch_names) = serde_json::from_str::<Option<Vec<String>>>(branch_required) else {
        return;
    };
    let branch_names = branch_names.unwrap_or_default();
    let mut required = root
        .get("required")
        .and_then(|raw| serde_json::from_str::<Option<Vec<String>>>(raw).ok())
        .flatten()
        .unwrap_or_default();
    let mut seen = required
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    for name in branch_names {
        if seen.insert(name.clone()) {
            required.push(name);
        }
    }
    if !required.is_empty() {
        root.insert("required".to_owned(), encode_string_array(&required));
    }
}

fn encode_string_array(values: &[String]) -> String {
    let mut encoded = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            encoded.push(',');
        }
        encoded.push_str(&go_json_string(value));
    }
    encoded.push(']');
    encoded
}

fn encode_object(object: &RawObject) -> String {
    let mut encoded = String::from("{");
    for (index, (key, raw)) in object.iter().enumerate() {
        if index > 0 {
            encoded.push(',');
        }
        encoded.push_str(&go_json_string(key));
        encoded.push(':');
        encoded.push_str(raw);
    }
    encoded.push('}');
    encoded
}

fn go_json_string(value: &str) -> String {
    let encoded = serde_json::to_string(value).expect("strings are serializable");
    escape_go_json_html(&encoded)
}

fn compact_go_json(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut chars = raw.chars();
    while let Some(character) = chars.next() {
        if in_string {
            if character == '\\' {
                output.push('\\');
                if let Some(escaped) = chars.next() {
                    output.push(escaped);
                }
            } else if character == '"' {
                output.push('"');
                in_string = false;
            } else if matches!(character, '<' | '>' | '&') {
                push_html_escape(&mut output, character);
            } else if character == '\u{2028}' {
                output.push_str(r#"\u2028"#);
            } else if character == '\u{2029}' {
                output.push_str(r#"\u2029"#);
            } else {
                output.push(character);
            }
        } else if character == '"' {
            output.push('"');
            in_string = true;
        } else if !character.is_ascii_whitespace() {
            output.push(character);
        }
    }
    output
}

fn escape_go_json_html(encoded: &str) -> String {
    let mut output = String::with_capacity(encoded.len());
    for character in encoded.chars() {
        if matches!(character, '<' | '>' | '&') {
            push_html_escape(&mut output, character);
        } else if character == '\u{2028}' {
            output.push_str(r#"\u2028"#);
        } else if character == '\u{2029}' {
            output.push_str(r#"\u2029"#);
        } else {
            output.push(character);
        }
    }
    output
}

fn push_html_escape(output: &mut String, character: char) {
    output.push_str(match character {
        '<' => r#"\u003c"#,
        '>' => r#"\u003e"#,
        '&' => r#"\u0026"#,
        _ => unreachable!("caller filters HTML-sensitive bytes"),
    });
}
