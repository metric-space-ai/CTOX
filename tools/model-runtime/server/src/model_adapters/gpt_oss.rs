use crate::gpt_oss_harmony;
use crate::openai::{
    CompletionRequest, FunctionCalled, Message, MessageContent, StopTokenIds, ToolCall,
};
use crate::responses::OpenResponsesCreateRequest;
use crate::types::SharedMistralRsState;
use engine_core::{CompletionResponse, Tool, ToolChoice, ToolType};

use super::AdaptedResponseItem;
use super::ResponsesTransportKind;

pub fn matches(model_id: &str) -> bool {
    gpt_oss_harmony::is_gpt_oss_model_id(model_id)
}

pub fn transport_kind() -> ResponsesTransportKind {
    ResponsesTransportKind::ChatCompletions
}

pub fn requires_buffered_chat_response() -> bool {
    false
}

pub fn prefers_chat_completions(_request: &OpenResponsesCreateRequest) -> bool {
    true
}

pub fn chat_stop_token_ids() -> Option<StopTokenIds> {
    Some(gpt_oss_harmony::harmony_stop_token_ids())
}

pub fn prepare_chat_messages(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> (Vec<Message>, Option<Vec<Tool>>) {
    gpt_oss_harmony::prepare_harmony_chat_messages(messages, tools, tool_choice)
}

pub fn effective_model_id(requested_model: &str, state: &SharedMistralRsState) -> Option<String> {
    if requested_model != "default" {
        return Some(requested_model.to_string());
    }
    state.list_models().ok().and_then(|models| {
        models
            .into_iter()
            .find(|candidate| gpt_oss_harmony::is_gpt_oss_model_id(candidate))
    })
}

pub fn completion_exact_text_override(
    messages: &[Message],
    tools: Option<&[engine_core::Tool]>,
) -> Option<String> {
    gpt_oss_harmony::effective_exact_text_override(messages, tools)
}

pub fn build_completion_request(
    request: &OpenResponsesCreateRequest,
    messages: &[Message],
    effective_model_id: &str,
) -> CompletionRequest {
    gpt_oss_harmony::build_completion_request(request, messages, effective_model_id)
}

pub fn build_followup_completion_request(
    initial_request: &CompletionRequest,
    first_completion: &CompletionResponse,
) -> Option<CompletionRequest> {
    gpt_oss_harmony::build_followup_completion_request(initial_request, first_completion)
}

pub fn apply_exact_text_override(
    completion: &mut CompletionResponse,
    exact_text_override: Option<&str>,
) {
    gpt_oss_harmony::apply_exact_text_override(completion, exact_text_override)
}

pub fn parse_response_items(raw_text: &str) -> Vec<AdaptedResponseItem> {
    gpt_oss_harmony::parse_harmony_response_items(raw_text)
        .into_iter()
        .map(|item| match item {
            gpt_oss_harmony::HarmonyResponseItem::Reasoning(text) => {
                AdaptedResponseItem::Reasoning(text)
            }
            gpt_oss_harmony::HarmonyResponseItem::Message(text) => {
                AdaptedResponseItem::Message(text)
            }
            gpt_oss_harmony::HarmonyResponseItem::FunctionCall(call) => {
                AdaptedResponseItem::FunctionCall(super::AdaptedFunctionCall {
                    call_id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                })
            }
        })
        .collect()
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
