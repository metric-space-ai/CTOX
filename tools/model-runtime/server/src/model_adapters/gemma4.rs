use crate::openai::{FunctionCalled, Message, MessageContent, ToolCall};
use engine_core::ToolType;
use regex::Regex;

use super::AdaptedFunctionCall;
use super::AdaptedResponseItem;
use super::ResponsesTransportKind;

pub fn matches(model_id: &str) -> bool {
    model_id.trim().to_ascii_lowercase().contains("gemma-4")
}

pub fn transport_kind() -> ResponsesTransportKind {
    ResponsesTransportKind::ChatCompletions
}

pub fn requires_buffered_chat_response() -> bool {
    true
}

pub fn parse_response_items(raw_text: &str) -> Vec<AdaptedResponseItem> {
    let tool_call_re =
        Regex::new(r"(?s)<\|tool_call>call:([A-Za-z0-9_.-]+)\s*(\{.*?\})<tool_call\|>")
            .expect("valid Gemma 4 tool call regex");
    let channel_re = Regex::new(r"(?s)<\|channel>thought\n(.*?)<channel\|>")
        .expect("valid Gemma 4 channel regex");

    let mut items = Vec::new();
    let reasoning_parts = channel_re
        .captures_iter(raw_text)
        .filter_map(|captures| {
            captures
                .get(1)
                .map(|capture| capture.as_str().trim().to_string())
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if !reasoning_parts.is_empty() {
        items.push(AdaptedResponseItem::Reasoning(reasoning_parts.join("\n\n")));
    }

    let stripped = channel_re.replace_all(raw_text, "").to_string();
    let mut plain_text = String::new();
    let mut last_end = 0usize;

    for (index, captures) in tool_call_re.captures_iter(&stripped).enumerate() {
        let Some(matched) = captures.get(0) else {
            continue;
        };
        plain_text.push_str(&stripped[last_end..matched.start()]);
        last_end = matched.end();

        let name = captures
            .get(1)
            .map(|capture| capture.as_str().trim().to_string())
            .unwrap_or_default();
        let arguments = captures
            .get(2)
            .map(|capture| capture.as_str().trim().to_string())
            .unwrap_or_else(|| "{}".to_string());
        items.push(AdaptedResponseItem::FunctionCall(AdaptedFunctionCall {
            call_id: format!("call_ctox_local_{index}"),
            name,
            arguments,
        }));
    }

    plain_text.push_str(&stripped[last_end..]);
    let plain_text = plain_text.trim();
    if !plain_text.is_empty() {
        items.insert(0, AdaptedResponseItem::Message(plain_text.to_string()));
    }
    if items.is_empty() && !raw_text.trim().is_empty() {
        items.push(AdaptedResponseItem::Message(raw_text.trim().to_string()));
    }
    items
}

pub fn response_text_to_history_messages(raw_text: &str) -> Vec<Message> {
    parse_response_items(raw_text)
        .into_iter()
        .filter_map(|item| match item {
            AdaptedResponseItem::Reasoning(_) => None,
            AdaptedResponseItem::Message(text) => Some(Message {
                content: Some(MessageContent::from_text(text)),
                role: "assistant".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }),
            AdaptedResponseItem::FunctionCall(call) => Some(Message {
                content: None,
                role: "assistant".to_string(),
                name: None,
                tool_calls: Some(vec![ToolCall {
                    id: Some(call.call_id),
                    tp: ToolType::Function,
                    function: FunctionCalled {
                        name: call.name,
                        arguments: call.arguments,
                    },
                }]),
                tool_call_id: None,
            }),
        })
        .collect()
}
