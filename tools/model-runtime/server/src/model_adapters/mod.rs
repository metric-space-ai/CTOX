mod gemma4;
mod glm47;
mod gpt_oss;
mod native_chat;
mod nemotron_cascade2;
mod qwen35;

use crate::openai::{CompletionRequest, Message, StopTokenIds};
use crate::responses::OpenResponsesCreateRequest;
use crate::types::SharedMistralRsState;
use engine_core::{CompletionResponse, Tool, ToolChoice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesTransportKind {
    ChatCompletions,
    CompletionTemplate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesModelAdapter {
    NativeChatCompletions,
    GptOssHarmony,
    Qwen35,
    NemotronCascade2,
    Gemma4,
    Glm47,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptedFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptedResponseItem {
    Reasoning(String),
    Message(String),
    FunctionCall(AdaptedFunctionCall),
}

#[derive(Debug, Clone)]
pub struct AdaptedChatRequest {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
}

impl ResponsesModelAdapter {
    pub fn effective_model_id(
        self,
        requested_model: &str,
        state: &SharedMistralRsState,
    ) -> Option<String> {
        match self {
            Self::GptOssHarmony => gpt_oss::effective_model_id(requested_model, state),
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => {
                if requested_model == "default" {
                    default_model_id(state)
                } else {
                    Some(requested_model.to_string())
                }
            }
        }
    }

    pub fn resolve(model: &str, state: &SharedMistralRsState) -> Self {
        let effective_model = if model == "default" {
            default_model_id(state)
        } else {
            Some(model.to_string())
        };
        let model_id = effective_model.as_deref().unwrap_or(model);
        if gpt_oss::matches(model_id) {
            Self::GptOssHarmony
        } else if qwen35::matches(model_id) {
            Self::Qwen35
        } else if nemotron_cascade2::matches(model_id) {
            Self::NemotronCascade2
        } else if gemma4::matches(model_id) {
            Self::Gemma4
        } else if glm47::matches(model_id) {
            Self::Glm47
        } else {
            Self::NativeChatCompletions
        }
    }

    pub fn transport_kind(self) -> ResponsesTransportKind {
        match self {
            Self::NativeChatCompletions => native_chat::transport_kind(),
            Self::GptOssHarmony => gpt_oss::transport_kind(),
            Self::Qwen35 => qwen35::transport_kind(),
            Self::NemotronCascade2 => nemotron_cascade2::transport_kind(),
            Self::Gemma4 => gemma4::transport_kind(),
            Self::Glm47 => glm47::transport_kind(),
        }
    }

    pub fn prefers_chat_completions(self, request: &OpenResponsesCreateRequest) -> bool {
        match self {
            Self::GptOssHarmony => gpt_oss::prefers_chat_completions(request),
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => false,
        }
    }

    pub fn requires_buffered_chat_response(self) -> bool {
        match self {
            Self::NativeChatCompletions => native_chat::requires_buffered_chat_response(),
            Self::GptOssHarmony => gpt_oss::requires_buffered_chat_response(),
            Self::Qwen35 => qwen35::requires_buffered_chat_response(),
            Self::NemotronCascade2 => nemotron_cascade2::requires_buffered_chat_response(),
            Self::Gemma4 => gemma4::requires_buffered_chat_response(),
            Self::Glm47 => glm47::requires_buffered_chat_response(),
        }
    }

    pub fn default_enable_thinking(self) -> Option<bool> {
        match self {
            Self::Qwen35 => Some(false),
            Self::NativeChatCompletions
            | Self::GptOssHarmony
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => None,
        }
    }

    pub fn chat_stop_token_ids(self) -> Option<StopTokenIds> {
        match self {
            Self::GptOssHarmony => gpt_oss::chat_stop_token_ids(),
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => None,
        }
    }

    pub fn prepare_chat_request(
        self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        tool_choice: Option<&ToolChoice>,
    ) -> AdaptedChatRequest {
        match self {
            Self::GptOssHarmony => {
                let (messages, tools) =
                    gpt_oss::prepare_chat_messages(messages, tools, tool_choice);
                AdaptedChatRequest {
                    messages,
                    tools,
                    tool_choice: tool_choice.cloned(),
                }
            }
            Self::Qwen35 => qwen35::prepare_chat_request(messages, tools, tool_choice),
            Self::NemotronCascade2 => {
                nemotron_cascade2::prepare_chat_request(messages, tools, tool_choice)
            }
            Self::Glm47 => glm47::prepare_chat_request(messages, tools, tool_choice),
            Self::NativeChatCompletions | Self::Gemma4 => AdaptedChatRequest {
                messages: messages.to_vec(),
                tools: tools.map(|tools| tools.to_vec()),
                tool_choice: tool_choice.cloned(),
            },
        }
    }

    pub fn completion_exact_text_override(
        self,
        messages: &[Message],
        tools: Option<&[engine_core::Tool]>,
    ) -> Option<String> {
        match self {
            Self::GptOssHarmony => gpt_oss::completion_exact_text_override(messages, tools),
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => None,
        }
    }

    pub fn build_completion_request(
        self,
        request: &OpenResponsesCreateRequest,
        messages: &[Message],
        effective_model_id: &str,
    ) -> Option<CompletionRequest> {
        match self {
            Self::GptOssHarmony => Some(gpt_oss::build_completion_request(
                request,
                messages,
                effective_model_id,
            )),
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => None,
        }
    }

    pub fn build_followup_completion_request(
        self,
        initial_request: &CompletionRequest,
        first_completion: &CompletionResponse,
    ) -> Option<CompletionRequest> {
        match self {
            Self::GptOssHarmony => {
                gpt_oss::build_followup_completion_request(initial_request, first_completion)
            }
            Self::NativeChatCompletions
            | Self::Qwen35
            | Self::NemotronCascade2
            | Self::Gemma4
            | Self::Glm47 => None,
        }
    }

    pub fn apply_exact_text_override(
        self,
        completion: &mut CompletionResponse,
        exact_text_override: Option<&str>,
    ) {
        if let Self::GptOssHarmony = self {
            gpt_oss::apply_exact_text_override(completion, exact_text_override);
        }
    }

    pub fn parse_response_items(self, raw_text: &str) -> Vec<AdaptedResponseItem> {
        match self {
            Self::NativeChatCompletions => native_chat::parse_response_items(raw_text),
            Self::GptOssHarmony => gpt_oss::parse_response_items(raw_text),
            Self::Qwen35 => qwen35::parse_response_items(raw_text),
            Self::NemotronCascade2 => nemotron_cascade2::parse_response_items(raw_text),
            Self::Gemma4 => gemma4::parse_response_items(raw_text),
            Self::Glm47 => glm47::parse_response_items(raw_text),
        }
    }

    pub fn response_text_to_history_messages(self, raw_text: &str) -> Vec<Message> {
        match self {
            Self::NativeChatCompletions => native_chat::response_text_to_history_messages(raw_text),
            Self::GptOssHarmony => gpt_oss::response_text_to_history_messages(raw_text),
            Self::Qwen35 => qwen35::response_text_to_history_messages(raw_text),
            Self::NemotronCascade2 => {
                nemotron_cascade2::response_text_to_history_messages(raw_text)
            }
            Self::Gemma4 => gemma4::response_text_to_history_messages(raw_text),
            Self::Glm47 => glm47::response_text_to_history_messages(raw_text),
        }
    }
}

fn default_model_id(state: &SharedMistralRsState) -> Option<String> {
    state
        .list_models()
        .ok()
        .and_then(|models| models.into_iter().next())
}

#[cfg(test)]
mod tests {
    use super::AdaptedResponseItem;
    use super::ResponsesModelAdapter;
    use crate::openai::{Message, MessageContent, ToolCall};
    use crate::responses::OpenResponsesCreateRequest;
    use engine_core::{Tool, ToolChoice};
    use serde_json::json;

    #[test]
    fn native_chat_adapter_treats_completion_text_as_plain_message() {
        let items =
            ResponsesModelAdapter::NativeChatCompletions.parse_response_items("hello world");
        assert_eq!(
            items,
            vec![AdaptedResponseItem::Message("hello world".to_string())]
        );
    }

    #[test]
    fn gpt_oss_adapter_parses_harmony_function_call() {
        let raw = "<|channel|>commentary to=functions.exec_command 123<|constrain|>json<|message|>{\"cmd\":\"pwd\"}<|call|>";
        let items = ResponsesModelAdapter::GptOssHarmony.parse_response_items(raw);
        assert!(matches!(
            items.first(),
            Some(AdaptedResponseItem::FunctionCall(_))
        ));
    }

    #[test]
    fn gpt_oss_adapter_routes_tool_requests_through_chat_path() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "openai/gpt-oss-20b",
            "input": "call the tool",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_cwd",
                        "description": "returns cwd",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "get_cwd",
                    "description": "returns cwd",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        }))
        .expect("request should parse");

        assert!(
            ResponsesModelAdapter::GptOssHarmony.prefers_chat_completions(&request),
            "GPT-OSS should use the Harmony-aware chat path"
        );
    }

    #[test]
    fn gpt_oss_adapter_supplies_harmony_chat_stop_token_ids() {
        let stop_ids = ResponsesModelAdapter::GptOssHarmony
            .chat_stop_token_ids()
            .expect("GPT-OSS should provide Harmony stop ids");
        assert_eq!(
            serde_json::to_value(stop_ids).unwrap(),
            json!([200002, 200012])
        );
    }

    #[test]
    fn qwen_adapter_parses_xml_tool_call() {
        let raw = "<tool_call>\n<function=get_cwd>\n</function>\n</tool_call>\n";
        let items = ResponsesModelAdapter::Qwen35.parse_response_items(raw);
        assert!(matches!(
            items.first(),
            Some(AdaptedResponseItem::FunctionCall(_))
        ));
    }

    #[test]
    fn qwen_adapter_disables_thinking_by_default() {
        assert_eq!(
            ResponsesModelAdapter::Qwen35.default_enable_thinking(),
            Some(false)
        );
        assert_eq!(
            ResponsesModelAdapter::NativeChatCompletions.default_enable_thinking(),
            None
        );
    }

    #[test]
    fn gemma_adapter_parses_tool_call_and_visible_text() {
        let raw = "<|channel>thought\nNeed cwd.<channel|>Using a tool.<|tool_call>call:get_cwd {}<tool_call|>";
        let items = ResponsesModelAdapter::Gemma4.parse_response_items(raw);
        assert!(matches!(
            items.first(),
            Some(AdaptedResponseItem::Message(text)) if text == "Using a tool."
        ));
        assert!(matches!(
            items.get(1),
            Some(AdaptedResponseItem::FunctionCall(_))
        ));
    }

    #[test]
    fn glm_adapter_parses_xml_tool_call() {
        let raw =
            "<tool_call>get_cwd<arg_key>path</arg_key><arg_value>/tmp</arg_value></tool_call>";
        let items = ResponsesModelAdapter::Glm47.parse_response_items(raw);
        assert!(matches!(
            items.first(),
            Some(AdaptedResponseItem::FunctionCall(_))
        ));
    }

    #[test]
    fn nemotron_cascade2_adapter_disables_native_tools_and_injects_xml_prompt() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Call the get_cwd tool now.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![serde_json::from_value::<Tool>(json!({
            "type": "function",
            "function": {
                "name": "get_cwd",
                "description": "Return cwd.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }))
        .expect("tool parses")];

        let adapted = ResponsesModelAdapter::NemotronCascade2.prepare_chat_request(
            &messages,
            Some(&tools),
            Some(&ToolChoice::Tool(tools[0].clone())),
        );

        assert!(adapted.tools.is_none());
        assert!(adapted.tool_choice.is_none());
        let system = adapted.messages[0]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("system prompt");
        assert!(system.contains("<tool_call>"));
        assert!(system.contains("get_cwd"));
    }

    #[test]
    fn glm_adapter_disables_native_tools_and_injects_xml_prompt() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Call the get_cwd tool now.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![serde_json::from_value::<Tool>(json!({
            "type": "function",
            "function": {
                "name": "get_cwd",
                "description": "Return cwd.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }))
        .expect("tool parses")];

        let adapted = ResponsesModelAdapter::Glm47.prepare_chat_request(
            &messages,
            Some(&tools),
            Some(&ToolChoice::Tool(tools[0].clone())),
        );

        assert!(adapted.tools.is_none());
        assert!(adapted.tool_choice.is_none());
        let system = adapted.messages[0]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("system prompt");
        assert!(system.contains("<tool_call>TOOL_NAME"));
        assert!(system.contains("get_cwd"));
    }

    #[test]
    fn qwen_adapter_disables_native_tools_and_injects_xml_prompt() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Call the get_cwd tool now.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![serde_json::from_value::<Tool>(json!({
            "type": "function",
            "function": {
                "name": "get_cwd",
                "description": "Return cwd.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }))
        .expect("tool parses")];

        let adapted = ResponsesModelAdapter::Qwen35.prepare_chat_request(
            &messages,
            Some(&tools),
            Some(&ToolChoice::Tool(tools[0].clone())),
        );

        assert!(adapted.tools.is_none());
        assert!(adapted.tool_choice.is_none());
        let system = adapted.messages[0]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("system prompt");
        assert!(system.contains("<tool_call>"));
        assert!(system.contains("<function=TOOL_NAME>"));
        assert!(system.contains("get_cwd"));
    }

    #[test]
    fn qwen_adapter_rewrites_tool_history_into_chat_safe_messages() {
        let messages = vec![
            Message {
                content: Some(MessageContent::from_text("Need cwd.".to_string())),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                content: None,
                role: "assistant".to_string(),
                name: None,
                tool_calls: Some(vec![ToolCall {
                    id: Some("call_123".to_string()),
                    tp: engine_core::ToolType::Function,
                    function: crate::openai::FunctionCalled {
                        name: "exec_command".to_string(),
                        arguments: "{\"cmd\":\"pwd\"}".to_string(),
                    },
                }]),
                tool_call_id: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "/home/metricspace/CTOX".to_string(),
                )),
                role: "tool".to_string(),
                name: Some("functions.exec_command".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_123".to_string()),
            },
        ];

        let adapted = ResponsesModelAdapter::Qwen35.prepare_chat_request(&messages, None, None);

        assert_eq!(adapted.messages.len(), 3);
        assert_eq!(adapted.messages[1].role, "assistant");
        let assistant = adapted.messages[1]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("assistant text");
        assert!(assistant.contains("<tool_call>"));
        assert!(assistant.contains("exec_command"));
        assert_eq!(adapted.messages[2].role, "user");
        let tool_output = adapted.messages[2]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("tool output text");
        assert!(tool_output.contains("Tool result from functions.exec_command:"));
        assert!(tool_output.contains("/home/metricspace/CTOX"));
    }

    #[test]
    fn qwen_adapter_maps_developer_messages_to_system() {
        let messages = vec![
            Message {
                content: Some(MessageContent::from_text(
                    "You are a careful coding agent.".to_string(),
                )),
                role: "developer".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "Inspect the workspace.".to_string(),
                )),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let adapted = ResponsesModelAdapter::Qwen35.prepare_chat_request(&messages, None, None);

        assert_eq!(adapted.messages.len(), 2);
        assert_eq!(adapted.messages[0].role, "system");
        assert_eq!(
            adapted.messages[0]
                .content
                .as_ref()
                .and_then(MessageContent::to_text),
            Some("You are a careful coding agent.".to_string())
        );
        assert_eq!(adapted.messages[1].role, "user");
    }

    #[test]
    fn qwen_adapter_merges_system_and_developer_messages_at_front() {
        let messages = vec![
            Message {
                content: Some(MessageContent::from_text("System A".to_string())),
                role: "system".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                content: Some(MessageContent::from_text("User request".to_string())),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                content: Some(MessageContent::from_text("Developer B".to_string())),
                role: "developer".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let adapted = ResponsesModelAdapter::Qwen35.prepare_chat_request(&messages, None, None);

        assert_eq!(adapted.messages.len(), 2);
        assert_eq!(adapted.messages[0].role, "system");
        assert_eq!(
            adapted.messages[0]
                .content
                .as_ref()
                .and_then(MessageContent::to_text),
            Some("System A\n\nDeveloper B".to_string())
        );
        assert_eq!(adapted.messages[1].role, "user");
    }
}
