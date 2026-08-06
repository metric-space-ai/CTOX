// ref: internal/htmlsanitize/htmlsanitize.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

pub fn string(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        output.push_str(match character {
            '&' => "&amp;",
            '\'' => "&#39;",
            '<' => "&lt;",
            '>' => "&gt;",
            '"' => "&#34;",
            _ => {
                output.push(character);
                continue;
            }
        });
    }
    output
}

pub fn strings(values: &[String]) -> Vec<String> {
    values.iter().map(|value| string(value)).collect()
}

pub fn json_body(body: &[u8]) -> (Vec<u8>, bool) {
    if body.iter().all(|byte| byte.is_ascii_whitespace()) {
        return (body.to_vec(), false);
    }
    let value = match serde_json::from_slice::<Value>(body) {
        Ok(value) => value,
        Err(_) => return (body.to_vec(), false),
    };
    match serde_json::to_vec(&json_value(&value)) {
        Ok(body) => (body, true),
        Err(_) => (body.to_vec(), false),
    }
}

pub fn json_body_if_likely(body: &[u8], content_type: &str) -> (Vec<u8>, bool) {
    if is_json_content_type(content_type) || looks_like_json(body) {
        json_body(body)
    } else {
        (body.to_vec(), false)
    }
}

pub fn json_value(value: &Value) -> Value {
    match value {
        Value::String(value) => Value::String(string(value)),
        Value::Array(values) => Value::Array(values.iter().map(json_value).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), json_value(value)))
                .collect(),
        ),
        value => value.clone(),
    }
}

pub fn is_json_content_type(content_type: &str) -> bool {
    let trimmed = content_type.trim();
    let media_type = trimmed
        .parse::<mime::Mime>()
        .map(|value| value.essence_str().to_owned())
        .unwrap_or_else(|_| trimmed.to_ascii_lowercase());
    let media_type = media_type.to_ascii_lowercase();
    media_type == "application/json" || media_type.ends_with("+json")
}

pub fn looks_like_json(body: &[u8]) -> bool {
    let body = trim_unicode_space(body);
    matches!(body.first(), Some(b'{') | Some(b'['))
}

fn trim_unicode_space(mut body: &[u8]) -> &[u8] {
    loop {
        let Ok(text) = std::str::from_utf8(body) else {
            return body.trim_ascii();
        };
        let trimmed = text.trim_matches(char::is_whitespace);
        if trimmed.len() == body.len() {
            return body;
        }
        body = trimmed.as_bytes();
    }
}
