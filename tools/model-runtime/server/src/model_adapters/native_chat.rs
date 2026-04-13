use super::AdaptedResponseItem;
use super::ResponsesTransportKind;
use crate::openai::{Message, MessageContent};

pub fn transport_kind() -> ResponsesTransportKind {
    ResponsesTransportKind::ChatCompletions
}

pub fn requires_buffered_chat_response() -> bool {
    false
}

pub fn parse_response_items(raw_text: &str) -> Vec<AdaptedResponseItem> {
    if raw_text.trim().is_empty() {
        Vec::new()
    } else {
        vec![AdaptedResponseItem::Message(raw_text.to_string())]
    }
}

pub fn response_text_to_history_messages(raw_text: &str) -> Vec<Message> {
    parse_response_items(raw_text)
        .into_iter()
        .map(|item| match item {
            AdaptedResponseItem::Reasoning(_) => {
                unreachable!("native chat adapter never emits reasoning items")
            }
            AdaptedResponseItem::Message(text) => Message {
                content: Some(MessageContent::from_text(text)),
                role: "assistant".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            AdaptedResponseItem::FunctionCall(_) => {
                unreachable!("native chat adapter never emits function calls")
            }
        })
        .collect()
}
