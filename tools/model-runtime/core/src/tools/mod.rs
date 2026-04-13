mod request;
mod response;

use candle_core::Result;
use regex::Regex;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

use crate::Pipeline;
use engine_mcp::CalledFunction;

// Re-export the types so they're accessible as tools::Type
pub use engine_mcp::{ToolCallback, ToolCallbackWithTool};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

fn contains_tool_call_prefix(prefix: &str) -> bool {
    prefix.contains("<tool_call>")
        || prefix.contains("<|tool_call>")
        || prefix.contains("<｜tool▁call▁begin｜>")
        || prefix.contains("<|python_tag|>")
        || prefix.contains("[TOOL_CALLS]")
}

const GEMMA4_STR_DELIM: &str = "<|\"|\x3e";

fn extract_matched_braces(s: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut i = start;
    while i < s.len() {
        if in_string {
            if s[i..].starts_with(GEMMA4_STR_DELIM) {
                in_string = false;
                i += GEMMA4_STR_DELIM.len();
                continue;
            }
            i += 1;
            continue;
        }
        if s[i..].starts_with(GEMMA4_STR_DELIM) {
            in_string = true;
            i += GEMMA4_STR_DELIM.len();
            continue;
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[start + 1..i], i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn escape_inner_quotes(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        let Some(open) = rest.find(GEMMA4_STR_DELIM) else {
            result.push_str(rest);
            break;
        };
        result.push_str(&rest[..open]);
        result.push_str(GEMMA4_STR_DELIM);
        rest = &rest[open + GEMMA4_STR_DELIM.len()..];

        let close = rest.find(GEMMA4_STR_DELIM).unwrap_or(rest.len());
        let inner = &rest[..close];
        for ch in inner.chars() {
            if ch == '"' {
                result.push('\\');
            }
            result.push(ch);
        }
        if close < rest.len() {
            result.push_str(GEMMA4_STR_DELIM);
            rest = &rest[close + GEMMA4_STR_DELIM.len()..];
        } else {
            rest = &rest[close..];
        }
    }
    result
}

fn quote_unquoted_keys(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 32);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_string = false;

    while i < len {
        if in_string {
            result.push(chars[i]);
            if chars[i] == '"' {
                in_string = false;
            } else if chars[i] == '\\' && i + 1 < len {
                i += 1;
                result.push(chars[i]);
            }
            i += 1;
            continue;
        }

        if chars[i] == '"' {
            in_string = true;
            result.push(chars[i]);
            i += 1;
            continue;
        }

        if chars[i].is_alphabetic() || chars[i] == '_' {
            let key_start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let key: String = chars[key_start..i].iter().collect();
            if i < len && chars[i] == ':' {
                result.push('"');
                result.push_str(&key);
                result.push('"');
            } else {
                result.push_str(&key);
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

fn gemma4_args_to_json(raw: &str) -> std::result::Result<Value, candle_core::Error> {
    let with_braces = format!("{{{raw}}}");
    let with_braces = escape_inner_quotes(&with_braces);
    let with_quotes = with_braces.replace(GEMMA4_STR_DELIM, "\"");
    let json_str = quote_unquoted_keys(&with_quotes);
    serde_json::from_str(&json_str).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to parse Gemma 4 tool call arguments: {e}\nConverted JSON: {json_str}"
        ))
    })
}

fn parse_gemma4_tool_calls(message: &str) -> Result<Option<String>> {
    let message = message
        .trim_end()
        .strip_suffix("<|tool_response>")
        .unwrap_or(message);

    let prefix = "<|tool_call>call:";
    let suffix = "<tool_call|>";

    if !message.contains(prefix) {
        return Ok(None);
    }

    #[derive(serde::Serialize)]
    struct ToolCall {
        name: String,
        arguments: Value,
    }

    let mut calls = Vec::new();
    let mut search_start = 0;

    while let Some(rel_pos) = message[search_start..].find(prefix) {
        let abs_start = search_start + rel_pos + prefix.len();
        let Some(brace_rel) = message[abs_start..].find('{') else {
            return Ok(None);
        };
        let name = message[abs_start..abs_start + brace_rel].trim().to_string();
        let brace_abs = abs_start + brace_rel;
        let Some((inner, after_brace)) = extract_matched_braces(message, brace_abs) else {
            return Ok(None);
        };

        let arguments = gemma4_args_to_json(inner)?;
        calls.push(ToolCall { name, arguments });

        let remaining = &message[after_brace..];
        if let Some(suf_pos) = remaining.find(suffix) {
            search_start = after_brace + suf_pos + suffix.len();
        } else {
            search_start = after_brace;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }

    let json = serde_json::to_string(&calls).map_err(candle_core::Error::msg)?;
    Ok(Some(json))
}

fn process_model_specific_message(message: &str) -> Result<String> {
    static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();
    static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

    // These are reasoning models so we need a regex.
    let deepseek_regex = DEEPSEEK_REGEX.get_or_init(|| Regex::new(
        r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
    ).unwrap());
    let qwen_regex = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    if message.contains("<|tool_call>") {
        if let Some(json) = parse_gemma4_tool_calls(message)? {
            return Ok(json);
        }
    }

    if let Some(message) = message.strip_prefix("<|python_tag|>") {
        // Llama case
        Ok(message.to_string())
    } else if qwen_regex.is_match(message) {
        if let Some(caps) = qwen_regex.captures(message) {
            let inner = caps.name("inner").unwrap().as_str();
            return Ok(inner.trim().to_string());
        }
        Ok(message.to_string())
    } else if let Some(message) = message
        .strip_prefix("[TOOL_CALLS][")
        .and_then(|s| s.strip_suffix("]"))
    {
        // Mistral Nemo case
        Ok(message.to_string())
    } else if deepseek_regex.find(message).is_some() {
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }
        let mut calls = Vec::new();
        for caps in deepseek_regex.captures_iter(message) {
            let name = caps
                .name("name")
                .ok_or("Could not capture function name")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim()
                .to_string();
            let json_str = caps
                .name("json")
                .ok_or("Could not capture JSON arguments")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim();
            let arguments: Value =
                serde_json::from_str(json_str).map_err(candle_core::Error::msg)?;
            calls.push(ToolCall { name, arguments });
        }
        Ok(serde_json::to_string(&calls).map_err(candle_core::Error::msg)?)
    } else {
        Ok(message.to_string())
    }
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

// Same as CalledFunction, but has different cases for variations on the names
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    #[serde(alias = "function")]
    pub name: String,
    #[serde(alias = "arguments", deserialize_with = "flexible_args")]
    pub parameters: Value,
}

// Accept either `{...}` **or** a `"stringified { ... }"`
fn flexible_args<'de, D>(d: D) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    struct ArgVisitor;

    impl<'de> Visitor<'de> for ArgVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an object or a JSON-encoded string containing an object")
        }

        // Case 1 – the good case: already a JSON object
        fn visit_map<M>(self, mut m: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = Map::new();
            while let Some((k, v)) = m.next_entry()? {
                map.insert(k, v);
            }
            Ok(Value::Object(map))
        }

        // Case 2 – got a *string*; try parsing it as JSON
        fn visit_str<E>(self, s: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::from_str(s).map_err(|e| E::custom(format!("inner JSON error: {e}")))
        }
    }

    d.deserialize_any(ArgVisitor)
}

/// Fixup potentially broken JSON
/// 1) allow/handle arguments as maps in quotations
fn fix_broken_json(raw: &str) -> anyhow::Result<String> {
    // Only apply the fix if the first pattern matches - otherwise we might corrupt valid JSON
    // where arguments is a properly escaped string containing `}`
    if raw.contains(r#""arguments":"{"#) {
        // 1) Delete the opening quote that shouldn't be there
        let tmp = raw.replacen(r#""arguments":"{"#, r#""arguments":{"#, 1);
        // 2) Delete the closing quote that matches it
        let fixed = tmp.replacen(r#"}"}"#, r#"}}"#, 1);
        Ok(fixed)
    } else {
        Ok(raw.to_string())
    }
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    // Checks if the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(
        &self,
        _pipeline: &dyn Pipeline,
        message_prefix: &str,
    ) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let message_prefix = process_model_specific_message(message_prefix)?;
        let message_prefix = fix_broken_json(&message_prefix).unwrap();

        // Check if the prefix could be a JSON serialization of any of the following types.
        Ok([
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<Vec<CalledFunctionParameters>>,
        ]
        .iter()
        .find_map(|check| {
            let (could_be_tool, is_complete_tool) = check(&message_prefix);
            if could_be_tool || is_complete_tool {
                Some((could_be_tool, is_complete_tool))
            } else {
                None
            }
        })
        .unwrap_or((contains_tool_call_prefix(&message_prefix), false)))
    }

    pub fn get_call(
        &self,
        _pipeline: &dyn Pipeline,
        message: &str,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }
        let message = process_model_specific_message(message)?;
        let message = fix_broken_json(&message).unwrap();

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(&message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                index: 0,
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            Ok(deser
                .into_iter()
                .enumerate()
                .map(|(idx, deser)| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        index: idx,
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a given type, `T`.
///
/// Returns a tuple of `(could_be_tool, is_entire_tool)`.
fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.trim().is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        // EOF show that JSON parsing was successful up to the end of the entire string.
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

/// Takes raw UTf8 text and parses any possible tool calls from it.
pub fn parse_text_tools<'a>(
    pipeline: &dyn Pipeline,
    raw_text: &'a str,
    matcher: Option<Arc<ToolCallingMatcher>>,
) -> anyhow::Result<(Option<&'a str>, Vec<ToolCallResponse>)> {
    let mut tool_calls = Vec::new();
    let mut text_new = Some(raw_text);

    if let Some(ref matcher) = matcher {
        let calls = matcher
            .get_call(pipeline, raw_text)
            .map_err(candle_core::Error::msg)?;
        if !calls.is_empty() {
            text_new = None;
            tool_calls = calls;
        }
    };
    Ok((text_new, tool_calls))
}
