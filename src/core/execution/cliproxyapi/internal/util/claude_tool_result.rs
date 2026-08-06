// ref: internal/util/claude_tool_result.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use gjson::{Kind, Value};

/// Base64 image extracted from a Claude `tool_result` content block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaudeToolResultImage {
    pub mime_type: String,
    pub data: String,
}

/// Normalized Claude `tool_result.content` for a Gemini function response.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClaudeToolResult {
    pub result: String,
    pub result_is_raw: bool,
    pub images: Vec<ClaudeToolResultImage>,
}

/// Separates Claude base64 image blocks while preserving every non-image JSON
/// value as raw text. String content is decoded exactly once.
pub fn convert_claude_tool_result_content(content: &Value<'_>) -> ClaudeToolResult {
    match content.kind() {
        Kind::String => ClaudeToolResult {
            result: content.str().to_owned(),
            ..ClaudeToolResult::default()
        },
        Kind::Array => convert_array(content),
        Kind::Object => {
            if is_claude_base64_image(content) {
                return claude_image_from_block(content)
                    .map(|image| ClaudeToolResult {
                        images: vec![image],
                        ..ClaudeToolResult::default()
                    })
                    .unwrap_or_default();
            }
            raw_result(content.json())
        }
        _ if content.exists() => raw_result(content.json()),
        _ => ClaudeToolResult::default(),
    }
}

fn convert_array(content: &Value<'_>) -> ClaudeToolResult {
    let mut images = Vec::new();
    let mut non_images = Vec::new();
    content.each(|_, block| {
        if is_claude_base64_image(&block) {
            if let Some(image) = claude_image_from_block(&block) {
                images.push(image);
            }
        } else {
            non_images.push(block.json().to_owned());
        }
        true
    });

    match non_images.len() {
        0 => ClaudeToolResult {
            images,
            ..ClaudeToolResult::default()
        },
        1 => ClaudeToolResult {
            result: non_images.pop().expect("length checked"),
            result_is_raw: true,
            images,
        },
        _ => ClaudeToolResult {
            result: format!("[{}]", non_images.join(",")),
            result_is_raw: true,
            images,
        },
    }
}

fn raw_result(raw: &str) -> ClaudeToolResult {
    ClaudeToolResult {
        result: raw.to_owned(),
        result_is_raw: true,
        images: Vec::new(),
    }
}

fn is_claude_base64_image(block: &Value<'_>) -> bool {
    block.get("type").str() == "image" && block.get("source.type").str() == "base64"
}

fn claude_image_from_block(block: &Value<'_>) -> Option<ClaudeToolResultImage> {
    let data_value = block.get("source.data");
    let data = data_value.str();
    if data.is_empty() {
        return None;
    }
    let mime_type = block.get("source.media_type");
    Some(ClaudeToolResultImage {
        mime_type: mime_type.str().to_owned(),
        data: data.to_owned(),
    })
}
