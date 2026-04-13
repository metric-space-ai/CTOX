use anyhow::Result;
use std::path::PathBuf;

use crate::openai::{CompletionRequest, Message, MessageContent, StopTokenIds, StopTokens};
use crate::responses::OpenResponsesCreateRequest;
use engine_core::{CompletionResponse, Tool};
use openai_harmony::{
    chat::{
        Author as HarmonyAuthor, Conversation as HarmonyConversation,
        DeveloperContent as HarmonyDeveloperContent, Message as HarmonyMessage,
        ReasoningEffort as HarmonyReasoningEffort, Role as HarmonyRole,
        SystemContent as HarmonySystemContent, ToolDescription as HarmonyToolDescription,
    },
    load_harmony_encoding, HarmonyEncodingName,
};
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

const DEFAULT_GPT_OSS_HARMONY_REASONING_CAP: &str = "low";
const DEFAULT_GPT_OSS_RUNTIME_OUTPUT_BUDGET: usize = 131_072;
const DEFAULT_GPT_OSS_HARMONY_MIN_OUTPUT_TOKENS: usize = 128;
const DEFAULT_GPT_OSS_EXACT_TEXT_MAX_OUTPUT_TOKENS: usize = 64;
const DISALLOWED_ENGINE_FUNCTION_TOOLS: &[&str] = &["spawn_agent", "send_input"];
const HARMONY_STOP_MARKERS: &[&str] = &["<|return|>", "<|call|>"];

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HarmonyResponseItem {
    Reasoning(String),
    Message(String),
    FunctionCall(HarmonyFunctionCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct RuntimeStateSnapshot {
    #[serde(default)]
    realized_context_tokens: Option<u32>,
    #[serde(default)]
    gpt_oss: GptOssRuntimeStateSnapshot,
}

#[derive(Debug, Default, Deserialize)]
struct GptOssRuntimeStateSnapshot {
    #[serde(default)]
    harmony_reasoning_cap: Option<String>,
    #[serde(default)]
    harmony_max_output_tokens_cap: Option<u32>,
}

pub fn is_gpt_oss_model_id(model_id: &str) -> bool {
    let lowered = model_id.trim().to_ascii_lowercase();
    lowered == "gpt-oss-20b" || lowered == "openai/gpt-oss-20b" || lowered.contains("gpt-oss")
}

pub fn build_completion_request(
    oairequest: &OpenResponsesCreateRequest,
    messages: &[Message],
    effective_model_id: &str,
) -> CompletionRequest {
    let model = if oairequest.model == "default" {
        effective_model_id.to_string()
    } else {
        oairequest.model.clone()
    };
    let exact_text_override = effective_exact_text_override(messages, oairequest.tools.as_deref());
    let requested_reasoning_effort = oairequest
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort)
        .map(|effort| match effort {
            crate::responses_types::enums::ReasoningEffort::None => "none".to_string(),
            crate::responses_types::enums::ReasoningEffort::Low => "low".to_string(),
            crate::responses_types::enums::ReasoningEffort::Medium => "medium".to_string(),
            crate::responses_types::enums::ReasoningEffort::High => "high".to_string(),
        })
        .unwrap_or_else(|| "medium".to_string());
    let selected_tool_name = selected_tool_name(oairequest.tool_choice.as_ref());
    let tools = if exact_text_override.is_some() {
        Vec::new()
    } else {
        filter_harmony_tools_for_selection(
            harmony_tool_specs(oairequest.tools.as_deref()),
            selected_tool_name.as_deref(),
        )
    };
    let reasoning_effort = if exact_text_override.is_some() {
        "none".to_string()
    } else {
        effective_gpt_oss_harmony_reasoning_effort(
            effective_model_id,
            &requested_reasoning_effort,
            !tools.is_empty(),
        )
    };
    let requested_max_output_tokens = oairequest
        .max_output_tokens
        .unwrap_or_else(|| default_gpt_oss_output_budget(effective_model_id));
    let requested_max_output_tokens =
        apply_exact_text_output_budget(requested_max_output_tokens, exact_text_override.as_deref());
    let capped_max_output_tokens = effective_gpt_oss_harmony_max_output_tokens(
        effective_model_id,
        requested_max_output_tokens,
    );
    let max_output_tokens = if gpt_oss_harmony_thinking_enabled(&reasoning_effort) {
        ensure_gpt_oss_thinking_output_floor(effective_model_id, capped_max_output_tokens)
    } else {
        capped_max_output_tokens
    };
    debug_dump_harmony_tool_state(
        selected_tool_name.as_deref(),
        oairequest.tools.as_deref(),
        &tools,
    );
    let developer_prompt = collect_developer_instructions(messages);
    let (prompt, prompt_tokens) =
        render_gpt_oss_prompt(&developer_prompt, messages, &reasoning_effort, &tools)
            .unwrap_or_else(|_| {
                let conversation = render_harmony_conversation(messages);
                let prompt = build_gpt_oss_harmony_prompt(
                    &developer_prompt,
                    &conversation,
                    &reasoning_effort,
                    &tools,
                );
                let prompt_tokens = engine_core::harmony::encode_harmony_prompt_tokens(&prompt);
                (prompt, prompt_tokens)
            });
    debug_dump_completion_prompt(&prompt);
    debug_dump_prompt_tokens(&prompt_tokens);

    CompletionRequest {
        model,
        prompt,
        prompt_tokens: Some(prompt_tokens),
        best_of: None,
        echo_prompt: false,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        logit_bias: oairequest.logit_bias.clone(),
        logprobs: oairequest.top_logprobs,
        max_tokens: Some(max_output_tokens),
        n_choices: oairequest.n_choices,
        stop_seqs: Some(merge_harmony_stop_tokens(oairequest.stop_seqs.clone())),
        stop_token_ids: Some(harmony_stop_token_ids()),
        // Local OpenResponses over CTOX's Unix socket can adapt a completed GPT-OSS
        // completion back into streamed response events; requesting streamed
        // completions here diverges from the root Harmony path and produced
        // malformed token-by-token output on the model server.
        stream: Some(false),
        temperature: Some(0.0),
        top_p: oairequest.top_p,
        suffix: None,
        _user: None,
        // GPT-OSS Harmony tool use is expressed in the rendered prompt via
        // `namespace functions { ... }` and parsed back from Harmony output.
        // Do not also enable the generic engine-side tool matcher on the
        // completion request, or the model/runtime can get trapped between
        // two incompatible tool-call protocols.
        tools: None,
        tool_choice: None,
        top_k: oairequest.top_k,
        grammar: None,
        min_p: oairequest.min_p,
        repetition_penalty: oairequest.repetition_penalty,
        dry_multiplier: oairequest.dry_multiplier,
        dry_base: oairequest.dry_base,
        dry_allowed_length: oairequest.dry_allowed_length,
        dry_sequence_breakers: oairequest.dry_sequence_breakers.clone(),
        truncate_sequence: oairequest
            .truncation
            .map(|t| matches!(t, crate::responses_types::enums::TruncationStrategy::Auto)),
    }
}

fn merge_harmony_stop_tokens(existing: Option<StopTokens>) -> StopTokens {
    let mut sequences = match existing {
        Some(StopTokens::Single(sequence)) => vec![sequence],
        Some(StopTokens::Multi(sequences)) => sequences,
        None => Vec::new(),
    };

    for marker in HARMONY_STOP_MARKERS {
        if !sequences.iter().any(|candidate| candidate == marker) {
            sequences.push((*marker).to_string());
        }
    }

    StopTokens::Multi(sequences)
}

pub fn harmony_stop_token_ids() -> StopTokenIds {
    let ids = HARMONY_STOP_MARKERS
        .iter()
        .filter_map(|marker| {
            let encoded = engine_core::harmony::encode_harmony_prompt_tokens(marker);
            (encoded.len() == 1).then_some(encoded[0])
        })
        .collect();
    StopTokenIds::Multi(ids)
}

pub fn effective_exact_text_override(
    messages: &[Message],
    tools: Option<&[Tool]>,
) -> Option<String> {
    extract_exact_text_override(messages)
        .filter(|_| !should_preserve_tool_capability_for_exact_text_request(messages, tools))
}

pub fn parse_harmony_response_items(raw_text: &str) -> Vec<HarmonyResponseItem> {
    debug_dump_raw_completion_text(raw_text);
    let mut items = Vec::new();
    if let Some(reasoning) = extract_harmony_reasoning_text(raw_text) {
        items.push(HarmonyResponseItem::Reasoning(reasoning));
    }
    if let Some(call) = parse_harmony_function_call(raw_text) {
        items.push(HarmonyResponseItem::FunctionCall(call));
        return items;
    }
    let text = extract_harmony_message_text(raw_text);
    if !text.trim().is_empty() {
        items.push(HarmonyResponseItem::Message(text));
    }
    items
}

pub fn extract_harmony_reasoning_text(raw_text: &str) -> Option<String> {
    let token = "<|channel|>analysis<|message|>";
    let matches = raw_text.match_indices(token).collect::<Vec<_>>();
    let mut parts = Vec::new();
    for (start, _) in matches {
        let content_start = start + token.len();
        let content_end = raw_text[content_start..]
            .find("<|end|>")
            .map(|offset| content_start + offset)
            .unwrap_or(raw_text.len());
        let text = sanitize_harmony_completion_text(&raw_text[content_start..content_end]);
        if !text.trim().is_empty() {
            parts.push(text);
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

pub fn extract_harmony_message_text(raw_text: &str) -> String {
    if raw_text.contains("<|channel|>analysis<|message|>")
        && !raw_text.contains("<|channel|>final<|message|>")
        && !raw_text.contains("<|channel|>commentary")
    {
        return String::new();
    }
    let final_token = "<|channel|>final<|message|>";
    let commentary_token = "<|channel|>commentary<|message|>";
    for token in [final_token, commentary_token] {
        let matches = raw_text.match_indices(token).collect::<Vec<_>>();
        for (start, _) in matches.into_iter().rev() {
            let content_start = start + token.len();
            let content_end = raw_text[content_start..]
                .find("<|end|>")
                .map(|offset| content_start + offset)
                .unwrap_or(raw_text.len());
            let text = sanitize_harmony_completion_text(&raw_text[content_start..content_end]);
            if !text.trim().is_empty() {
                return text;
            }
        }
    }
    if let Some(text) = extract_plaintext_harmony_final(raw_text) {
        return text;
    }
    sanitize_harmony_completion_text(raw_text)
}

pub fn build_followup_completion_request(
    initial_request: &CompletionRequest,
    first_completion: &CompletionResponse,
) -> Option<CompletionRequest> {
    if !completion_needs_followup(first_completion) {
        return None;
    }

    let first_text = completion_text(first_completion);
    let mut request = initial_request.clone();
    request.prompt = format!("{}{}<|end|><|return|>", request.prompt, first_text);
    request.prompt_tokens = Some(engine_core::harmony::encode_harmony_prompt_tokens(
        &request.prompt,
    ));
    Some(request)
}

pub fn completion_needs_followup(completion: &CompletionResponse) -> bool {
    let raw_text = completion_text(completion);
    let items = parse_harmony_response_items(raw_text);
    items.is_empty() && raw_text.contains("<|channel|>analysis<|message|>")
}

pub fn apply_exact_text_override(
    completion: &mut CompletionResponse,
    exact_text_override: Option<&str>,
) {
    let Some(exact_text) = exact_text_override
        .map(str::trim)
        .filter(|text| !text.is_empty())
    else {
        return;
    };

    if let Some(choice) = completion.choices.first_mut() {
        choice.text = exact_text.to_string();
    }
}

fn completion_text(completion: &CompletionResponse) -> &str {
    completion
        .choices
        .first()
        .map(|choice| choice.text.as_str())
        .unwrap_or_default()
}

fn build_gpt_oss_harmony_prompt(
    developer_prompt: &str,
    conversation: &str,
    reasoning_effort: &str,
    tools: &[HarmonyToolSpec],
) -> String {
    let current_date = "2026-04-07";
    let reasoning_effort = sanitize_reasoning_effort(reasoning_effort);
    let developer_block = build_harmony_developer_block(developer_prompt, tools);
    let system_tool_hint = if tools.is_empty() {
        ""
    } else {
        "\nCalls to these tools must go to the commentary channel: 'functions'."
    };
    let developer_section = if developer_block.is_empty() {
        String::new()
    } else {
        format!("<|start|>developer<|message|>{developer_block}<|end|>")
    };
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
Knowledge cutoff: 2024-06\n\
Current date: {current_date}\n\n\
Reasoning: {reasoning_effort}\n\n\
# Valid channels: analysis, commentary, final. Channel must be included for every message.{system_tool_hint}<|end|>\
{developer_section}\
{conversation}\
<|start|>assistant",
        current_date = current_date,
        reasoning_effort = reasoning_effort,
        system_tool_hint = system_tool_hint,
        developer_section = developer_section,
        conversation = conversation,
    )
}

fn render_gpt_oss_prompt(
    developer_prompt: &str,
    messages: &[Message],
    reasoning_effort: &str,
    tools: &[HarmonyToolSpec],
) -> Result<(String, Vec<u32>)> {
    let Some(reasoning_effort) = official_harmony_reasoning_effort(reasoning_effort) else {
        let conversation = render_harmony_conversation(messages);
        let prompt =
            build_gpt_oss_harmony_prompt(developer_prompt, &conversation, reasoning_effort, tools);
        let prompt_tokens = engine_core::harmony::encode_harmony_prompt_tokens(&prompt);
        return Ok((prompt, prompt_tokens));
    };

    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let mut conversation_messages = vec![HarmonyMessage::from_role_and_content(
        HarmonyRole::System,
        HarmonySystemContent::new()
            .with_reasoning_effort(reasoning_effort)
            .with_conversation_start_date("2026-04-07"),
    )];

    if !developer_prompt.trim().is_empty() || !tools.is_empty() {
        let mut developer = HarmonyDeveloperContent::new();
        if !developer_prompt.trim().is_empty() {
            developer = developer.with_instructions(developer_prompt.trim());
        }
        if !tools.is_empty() {
            developer = developer.with_function_tools(
                tools
                    .iter()
                    .map(|tool| {
                        HarmonyToolDescription::new(
                            tool.name.clone(),
                            tool.description.clone().unwrap_or_default(),
                            normalize_harmony_tool_parameters(tool.parameters.clone()),
                        )
                    })
                    .collect(),
            );
        }
        conversation_messages.push(HarmonyMessage::from_role_and_content(
            HarmonyRole::Developer,
            developer,
        ));
    }

    conversation_messages.extend(convert_messages_to_official_harmony(messages));
    let conversation = HarmonyConversation::from_messages(conversation_messages);
    let prompt_tokens =
        encoding.render_conversation_for_completion(&conversation, HarmonyRole::Assistant, None)?;
    let prompt = encoding.tokenizer().decode_utf8(&prompt_tokens)?;
    Ok((prompt, prompt_tokens))
}

fn official_harmony_reasoning_effort(reasoning_effort: &str) -> Option<HarmonyReasoningEffort> {
    match sanitize_reasoning_effort(reasoning_effort) {
        "low" => Some(HarmonyReasoningEffort::Low),
        "medium" => Some(HarmonyReasoningEffort::Medium),
        "high" => Some(HarmonyReasoningEffort::High),
        _ => None,
    }
}

fn convert_messages_to_official_harmony(messages: &[Message]) -> Vec<HarmonyMessage> {
    let mut converted = Vec::new();
    let mut last_tool_call_name: Option<String> = None;

    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => {}
            "assistant" => {
                if let Some(text) = extract_message_text(message.content.as_ref()) {
                    if !text.trim().is_empty() {
                        converted.push(
                            HarmonyMessage::from_role_and_content(
                                HarmonyRole::Assistant,
                                text.trim().to_string(),
                            )
                            .with_channel("final"),
                        );
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        let recipient = harmony_tool_recipient(&tool_call.function.name);
                        converted.push(
                            HarmonyMessage::from_role_and_content(
                                HarmonyRole::Assistant,
                                tool_call.function.arguments.trim().to_string(),
                            )
                            .with_channel("commentary")
                            .with_recipient(recipient.clone())
                            .with_content_type("<|constrain|>json"),
                        );
                        last_tool_call_name = Some(recipient);
                    }
                }
            }
            "tool" => {
                let text = extract_message_text(message.content.as_ref())
                    .unwrap_or_else(|| extract_function_call_output_text(message.content.as_ref()));
                if !text.trim().is_empty() {
                    let tool_name = message
                        .name
                        .as_deref()
                        .filter(|name| !name.trim().is_empty())
                        .map(ToString::to_string)
                        .or_else(|| last_tool_call_name.clone())
                        .unwrap_or_else(|| "tool".to_string());
                    converted.push(HarmonyMessage::from_author_and_content(
                        HarmonyAuthor::new(HarmonyRole::Tool, tool_name),
                        text.trim().to_string(),
                    ));
                }
            }
            _ => {
                if let Some(text) = extract_message_text(message.content.as_ref()) {
                    if !text.trim().is_empty() {
                        converted.push(HarmonyMessage::from_role_and_content(
                            HarmonyRole::User,
                            text.trim().to_string(),
                        ));
                    }
                }
            }
        }
    }

    converted
}

pub fn extract_exact_text_override(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|message| extract_message_text(message.content.as_ref()))
        .and_then(|text| extract_embedded_exact_text_request(&text))
}

fn should_preserve_tool_capability_for_exact_text_request(
    messages: &[Message],
    tools: Option<&[Tool]>,
) -> bool {
    if !tools.is_some_and(|items| !items.is_empty()) {
        return false;
    }

    let Some(latest_user_text) = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|message| extract_message_text(message.content.as_ref()))
    else {
        return false;
    };

    let normalized = latest_user_text.to_ascii_lowercase();
    let workspace_markers = [
        "work only inside this workspace",
        "arbeite ausschließlich im verzeichnis",
        "workspace:",
        "cwd:",
    ];
    let action_markers = [
        "create ",
        "build ",
        "verify ",
        "run ",
        "edit ",
        "implement ",
        "cmake",
        "cargo",
        "pytest",
        "./build/",
        "do not answer before",
        "on successful run",
        "must print exactly",
        "must output exactly",
        "verify the binary",
        ".cpp",
        ".h",
        ".hpp",
        "cmakelists.txt",
    ];

    workspace_markers
        .iter()
        .any(|marker| normalized.contains(marker))
        && action_markers
            .iter()
            .any(|marker| normalized.contains(marker))
}

fn collect_developer_instructions(messages: &[Message]) -> String {
    let parts = messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "system" | "developer"))
        .filter_map(|message| extract_message_text(message.content.as_ref()))
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    parts.join("\n\n")
}

fn render_harmony_conversation(messages: &[Message]) -> String {
    let mut rendered = String::new();
    let mut last_tool_call_name: Option<String> = None;
    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => {}
            "assistant" => {
                if let Some(text) = extract_message_text(message.content.as_ref()) {
                    if !text.trim().is_empty() {
                        rendered.push_str("<|start|>assistant<|channel|>final<|message|>");
                        rendered.push_str(text.trim());
                        rendered.push_str("<|end|>");
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        let recipient = harmony_tool_recipient(&tool_call.function.name);
                        let encoded_arguments =
                            serde_json::to_string(tool_call.function.arguments.trim())
                                .unwrap_or_else(|_| {
                                    format!("\"{}\"", tool_call.function.arguments.trim())
                                });
                        rendered.push_str("<|start|>assistant<|channel|>commentary to=");
                        rendered.push_str(&recipient);
                        rendered.push_str(" <|constrain|>json<|message|>");
                        rendered.push_str(&encoded_arguments);
                        rendered.push_str("<|call|>");
                        last_tool_call_name = Some(recipient);
                    }
                }
            }
            "tool" => {
                let text = extract_message_text(message.content.as_ref())
                    .unwrap_or_else(|| extract_function_call_output_text(message.content.as_ref()));
                if !text.trim().is_empty() {
                    let tool_name = message
                        .name
                        .as_deref()
                        .filter(|name| !name.trim().is_empty())
                        .map(ToString::to_string)
                        .or_else(|| last_tool_call_name.clone())
                        .unwrap_or_else(|| "tool".to_string());
                    let encoded_text = serde_json::to_string(text.trim())
                        .unwrap_or_else(|_| format!("\"{}\"", text.trim()));
                    rendered.push_str("<|start|>");
                    rendered.push_str(&tool_name);
                    rendered.push_str(" to=assistant<|channel|>commentary<|message|>");
                    rendered.push_str(&encoded_text);
                    rendered.push_str("<|end|>");
                }
            }
            _ => {
                if let Some(text) = extract_message_text(message.content.as_ref()) {
                    if !text.trim().is_empty() {
                        rendered.push_str("<|start|>user<|message|>");
                        rendered.push_str(text.trim());
                        rendered.push_str("<|end|>");
                    }
                }
            }
        }
    }

    if rendered.is_empty() {
        "<|start|>user<|message|><|end|>".to_string()
    } else {
        rendered
    }
}

fn harmony_tool_recipient(tool_name: &str) -> String {
    let trimmed = tool_name.trim();
    if trimmed.starts_with("functions.") || trimmed.starts_with("browser.") || trimmed == "python" {
        trimmed.to_string()
    } else {
        format!("functions.{trimmed}")
    }
}

fn extract_message_text(content: Option<&MessageContent>) -> Option<String> {
    content.and_then(MessageContent::to_text)
}

fn extract_function_call_output_text(content: Option<&MessageContent>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match &**content {
        either::Either::Left(text) => text.clone(),
        either::Either::Right(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(|value| value.as_ref().left().map(ToOwned::to_owned))
                })
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>();
            text.join("\n")
        }
    }
}

fn harmony_tool_specs(tools: Option<&[Tool]>) -> Vec<HarmonyToolSpec> {
    tools
        .into_iter()
        .flatten()
        .filter_map(|tool| serde_json::to_value(tool).ok())
        .flat_map(rewrite_openai_tool)
        .filter_map(|tool| parse_harmony_tool_spec(&tool))
        .collect()
}

fn selected_tool_name(tool_choice: Option<&engine_core::ToolChoice>) -> Option<String> {
    let engine_core::ToolChoice::Tool(tool) = tool_choice? else {
        return None;
    };
    let tool = serde_json::to_value(tool).ok()?;
    tool.get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| tool.get("name").and_then(Value::as_str))
        .map(ToString::to_string)
}

fn filter_harmony_tools_for_selection(
    tools: Vec<HarmonyToolSpec>,
    selected_tool_name: Option<&str>,
) -> Vec<HarmonyToolSpec> {
    let Some(selected_tool_name) = selected_tool_name else {
        return tools;
    };
    let selected = tools
        .into_iter()
        .filter(|tool| tool.name == selected_tool_name)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        Vec::new()
    } else {
        selected
    }
}

pub fn prepare_harmony_chat_messages(
    messages: &[Message],
    tools: Option<&[Tool]>,
    tool_choice: Option<&engine_core::ToolChoice>,
) -> (Vec<Message>, Option<Vec<Tool>>) {
    let selected_tool_name = selected_tool_name(tool_choice);
    let filtered_tools = filter_openai_tools_for_selection(tools, selected_tool_name.as_deref());
    let developer_prompt = collect_developer_instructions(messages);

    let mut rewritten = Vec::new();
    if !developer_prompt.trim().is_empty() {
        rewritten.push(Message {
            content: Some(MessageContent::from_text(developer_prompt)),
            role: "developer".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    rewritten.extend(
        messages
            .iter()
            .filter(|message| !matches!(message.role.as_str(), "system" | "developer"))
            .cloned(),
    );

    (rewritten, filtered_tools)
}

fn rewrite_openai_tool(tool: Value) -> Vec<Value> {
    let Some(object) = tool.as_object() else {
        return Vec::new();
    };
    let Some(tool_type) = object.get("type").and_then(Value::as_str) else {
        return Vec::new();
    };
    match tool_type {
        "web_search" => vec![Value::Object(object.clone())],
        "function" => {
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .unwrap_or(object);
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                return Vec::new();
            };
            if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name) {
                return Vec::new();
            }
            let mut flattened = serde_json::Map::new();
            flattened.insert("type".to_string(), Value::String("function".to_string()));
            for key in ["name", "description", "parameters", "strict"] {
                if let Some(value) = function.get(key) {
                    flattened.insert(key.to_string(), value.clone());
                }
            }
            vec![Value::Object(flattened)]
        }
        "namespace" => object
            .get("tools")
            .and_then(Value::as_array)
            .map(|children| {
                children
                    .iter()
                    .flat_map(|child| rewrite_openai_tool(child.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn filter_openai_tools_for_selection(
    tools: Option<&[Tool]>,
    selected_tool_name: Option<&str>,
) -> Option<Vec<Tool>> {
    let Some(tools) = tools else {
        return None;
    };
    let Some(selected_tool_name) = selected_tool_name else {
        return Some(tools.to_vec());
    };
    let filtered = tools
        .iter()
        .filter(|tool| tool.function.name == selected_tool_name)
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        Some(Vec::new())
    } else {
        Some(filtered)
    }
}

fn parse_harmony_tool_spec(tool: &Value) -> Option<HarmonyToolSpec> {
    let tool_type = tool.get("type").and_then(Value::as_str)?;
    if tool_type != "function" {
        return None;
    }
    let function = tool.get("function").unwrap_or(tool);
    let name = function.get("name").and_then(Value::as_str)?.to_string();
    if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name.as_str()) {
        return None;
    }
    Some(HarmonyToolSpec {
        name,
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        parameters: function.get("parameters").cloned(),
    })
}

fn normalize_harmony_tool_parameters(parameters: Option<Value>) -> Option<Value> {
    let Some(parameters) = parameters else {
        return None;
    };
    let is_empty_object = parameters
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "object")
        && parameters
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(|properties| properties.is_empty())
        && parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|required| required.is_empty())
            .unwrap_or(true)
        && !parameters.as_object().is_some_and(|object| {
            object.keys().any(|key| {
                key != "type"
                    && key != "properties"
                    && key != "required"
                    && key != "additionalProperties"
            })
        });
    if is_empty_object {
        None
    } else {
        Some(parameters)
    }
}

fn sanitize_reasoning_effort(value: &str) -> &str {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => "none",
        "minimal" | "low" => "low",
        "medium" => "medium",
        "high" => "high",
        _ => "medium",
    }
}

fn reasoning_effort_rank(value: &str) -> u8 {
    match sanitize_reasoning_effort(value) {
        "none" => 0,
        "low" => 1,
        "medium" => 2,
        "high" => 3,
        _ => 2,
    }
}

fn gpt_oss_harmony_thinking_enabled(reasoning_effort: &str) -> bool {
    sanitize_reasoning_effort(reasoning_effort) != "none"
}

fn cap_gpt_oss_reasoning_effort(requested: &str, cap: Option<&str>) -> String {
    let requested = sanitize_reasoning_effort(requested);
    let Some(cap) = cap.map(sanitize_reasoning_effort) else {
        return requested.to_string();
    };
    if reasoning_effort_rank(requested) <= reasoning_effort_rank(cap) {
        requested.to_string()
    } else {
        cap.to_string()
    }
}

fn effective_gpt_oss_harmony_reasoning_effort(
    model_id: &str,
    requested: &str,
    _has_function_tools: bool,
) -> String {
    if !is_gpt_oss_model_id(model_id) {
        return sanitize_reasoning_effort(requested).to_string();
    }
    let requested = match sanitize_reasoning_effort(requested) {
        "none" => "low",
        other => other,
    };
    let cap = gpt_oss_harmony_reasoning_cap();
    cap_gpt_oss_reasoning_effort(requested, Some(cap.as_str()))
}

fn cap_gpt_oss_max_output_tokens(requested: usize, cap: Option<usize>) -> usize {
    let Some(cap) = cap.filter(|value| *value > 0) else {
        return requested;
    };
    requested.min(cap)
}

fn effective_gpt_oss_harmony_max_output_tokens(model_id: &str, requested: usize) -> usize {
    if !is_gpt_oss_model_id(model_id) {
        return requested;
    }
    cap_gpt_oss_max_output_tokens(requested, gpt_oss_harmony_max_output_tokens_cap())
}

fn ensure_gpt_oss_thinking_output_floor(model_id: &str, requested: usize) -> usize {
    if !is_gpt_oss_model_id(model_id) {
        return requested;
    }
    requested.max(DEFAULT_GPT_OSS_HARMONY_MIN_OUTPUT_TOKENS)
}

fn apply_exact_text_output_budget(requested: usize, exact_text_override: Option<&str>) -> usize {
    let Some(exact_text) = exact_text_override
        .map(str::trim)
        .filter(|text| !text.is_empty())
    else {
        return requested;
    };

    requested.min(exact_text_output_budget(exact_text))
}

fn exact_text_output_budget(exact_text: &str) -> usize {
    let exact_chars = exact_text.chars().count();
    let padded = exact_chars.saturating_mul(2).saturating_add(16);
    padded.clamp(16, DEFAULT_GPT_OSS_EXACT_TEXT_MAX_OUTPUT_TOKENS)
}

fn current_runtime_root() -> PathBuf {
    std::env::current_dir()
        .ok()
        .filter(|path| path.join("runtime/inference_runtime.json").exists())
        .or_else(|| std::env::var("CTOX_ROOT").ok().map(PathBuf::from))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn should_dump_debug_artifacts() -> bool {
    if std::env::var("CTOX_GPT_OSS_DEBUG_ARTIFACTS")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
    {
        return true;
    }

    gpt_oss_debug_dir().join(".enabled").exists()
}

fn gpt_oss_debug_dir() -> PathBuf {
    current_runtime_root()
        .join("runtime")
        .join("model_adapter_debug")
        .join("gpt_oss")
}

fn write_debug_artifact(file_name: &str, content: &str) {
    if !should_dump_debug_artifacts() {
        return;
    }
    let dir = gpt_oss_debug_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        let _ = std::fs::write(PathBuf::from("/tmp").join(file_name), content);
        return;
    }
    let _ = std::fs::write(dir.join(file_name), content);
    let _ = std::fs::write(PathBuf::from("/tmp").join(file_name), content);
}

pub fn debug_dump_completion_prompt(prompt: &str) {
    write_debug_artifact("latest_prompt.txt", prompt);
}

pub fn debug_dump_prompt_tokens(prompt_tokens: &[u32]) {
    if let Ok(serialized) = serde_json::to_string(prompt_tokens) {
        write_debug_artifact("latest_prompt_tokens.json", &serialized);
    }
}

pub fn debug_dump_raw_completion_text(raw_text: &str) {
    write_debug_artifact("latest_raw_completion.txt", raw_text);
}

fn debug_dump_harmony_tool_state(
    selected_tool_name: Option<&str>,
    original_tools: Option<&[Tool]>,
    filtered_tools: &[HarmonyToolSpec],
) {
    if !should_dump_debug_artifacts() {
        return;
    }

    let original_tools = original_tools
        .into_iter()
        .flatten()
        .filter_map(|tool| serde_json::to_value(tool).ok())
        .collect::<Vec<_>>();
    let filtered_tools = filtered_tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect::<Vec<_>>();

    let payload = json!({
        "selected_tool_name": selected_tool_name,
        "original_tools": original_tools,
        "filtered_tools": filtered_tools,
    });
    if let Ok(pretty) = serde_json::to_string_pretty(&payload) {
        write_debug_artifact("latest_tools.json", &pretty);
    }
}

fn current_runtime_state() -> Option<RuntimeStateSnapshot> {
    let root = current_runtime_root();
    let path = root.join("runtime/inference_runtime.json");
    let raw = std::fs::read(path).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn gpt_oss_harmony_reasoning_cap() -> String {
    current_runtime_state()
        .and_then(|state| state.gpt_oss.harmony_reasoning_cap)
        .filter(|value| !value.trim().is_empty())
        .map(|value| sanitize_reasoning_effort(&value).to_string())
        .unwrap_or_else(|| DEFAULT_GPT_OSS_HARMONY_REASONING_CAP.to_string())
}

fn gpt_oss_harmony_max_output_tokens_cap() -> Option<usize> {
    current_runtime_state()
        .and_then(|state| {
            state
                .gpt_oss
                .harmony_max_output_tokens_cap
                .map(|value| value as usize)
                .or_else(|| state.realized_context_tokens.map(|value| value as usize))
        })
        .filter(|value| *value > 0)
}

fn default_gpt_oss_output_budget(model_id: &str) -> usize {
    if !is_gpt_oss_model_id(model_id) {
        return DEFAULT_GPT_OSS_HARMONY_MIN_OUTPUT_TOKENS;
    }
    current_runtime_state()
        .and_then(|state| state.realized_context_tokens.map(|value| value as usize))
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPT_OSS_RUNTIME_OUTPUT_BUDGET)
}

fn build_harmony_developer_block(system_prompt: &str, tools: &[HarmonyToolSpec]) -> String {
    let mut block = String::new();
    let trimmed = system_prompt.trim();
    if !trimmed.is_empty() {
        block.push_str("# Instructions\n\n");
        block.push_str(trimmed);
        block.push_str("\n\n");
    }
    if !tools.is_empty() {
        block.push_str("# Tools\n\n");
        block.push_str(&render_harmony_tool_namespace("functions", tools));
    }
    block.trim_end().to_string()
}

fn render_harmony_tool_namespace(namespace_name: &str, tools: &[HarmonyToolSpec]) -> String {
    let mut rendered = String::new();
    rendered.push_str("## ");
    rendered.push_str(namespace_name);
    rendered.push_str("\n\nnamespace ");
    rendered.push_str(namespace_name);
    rendered.push_str(" {\n\n");
    for tool in tools {
        rendered.push_str(&render_harmony_tool_signature(tool));
        rendered.push('\n');
    }
    rendered.push_str("} // namespace ");
    rendered.push_str(namespace_name);
    rendered.trim_end().to_string()
}

fn sanitize_harmony_completion_text(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if let Some(idx) = text.rfind("<|message|>") {
        text = text[idx + "<|message|>".len()..].to_string();
    }
    if let Some(idx) = text.find("<|return|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|end|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|start|>") {
        text.truncate(idx);
    }
    sanitize_harmony_channel_leakage(&text)
}

fn json_schema_to_typescript(schema: &Value) -> String {
    if schema.get("type").and_then(Value::as_str) == Some("object")
        && schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.is_empty())
            .unwrap_or(true)
    {
        return "() => any".to_string();
    }

    let object = json_schema_object_to_typescript(schema);
    if object.trim().is_empty() {
        "() => any".to_string()
    } else {
        format!("(_: {object}) => any")
    }
}

fn json_schema_object_to_typescript(schema: &Value) -> String {
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(key, value)| {
                    let optional = if required.iter().any(|item| item == key) {
                        ""
                    } else {
                        "?"
                    };
                    let mut line = String::new();
                    if let Some(description) = value.get("description").and_then(Value::as_str) {
                        line.push_str("// ");
                        line.push_str(description.trim());
                        line.push('\n');
                    }
                    line.push_str(&format!(
                        "{key}{optional}: {}",
                        json_schema_type_to_typescript(value)
                    ));
                    if let Some(default) = value.get("default") {
                        line.push_str(&format!(", // default: {}", default));
                    }
                    line
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    if props.trim().is_empty() {
        "object".to_string()
    } else {
        format!("{{\n{props}\n}}")
    }
}

fn json_schema_type_to_typescript(schema: &Value) -> String {
    match schema.get("enum").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(format!("\"{text}\"")),
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" | "),
        _ => match schema.get("type").and_then(Value::as_str) {
            Some("string") => "string".to_string(),
            Some("integer") | Some("number") => "number".to_string(),
            Some("boolean") => "boolean".to_string(),
            Some("array") => {
                let item_ty = schema
                    .get("items")
                    .map(json_schema_type_to_typescript)
                    .unwrap_or_else(|| "any".to_string());
                format!("{item_ty}[]")
            }
            Some("object") => schema
                .get("properties")
                .map(|_| json_schema_object_to_typescript(schema))
                .unwrap_or_else(|| "object".to_string()),
            _ => "any".to_string(),
        },
    }
}

fn render_harmony_tool_signature(tool: &HarmonyToolSpec) -> String {
    let mut rendered = String::new();
    if let Some(description) = &tool.description {
        rendered.push_str("// ");
        rendered.push_str(description.trim());
        rendered.push('\n');
    }
    rendered.push_str("type ");
    rendered.push_str(&tool.name);
    rendered.push_str(" = ");
    rendered.push_str(
        &tool
            .parameters
            .as_ref()
            .map(json_schema_to_typescript)
            .unwrap_or_else(|| "() => any".to_string()),
    );
    rendered.push_str(";\n");
    rendered
}

fn parse_harmony_function_call(raw_text: &str) -> Option<HarmonyFunctionCall> {
    let message_token = "<|message|>";
    let message_start = raw_text.find(message_token)?;
    let header = &raw_text[..message_start];
    if !header.contains("<|channel|>commentary") {
        return None;
    }

    let recipient_idx = header.find("to=")?;
    let recipient_start = recipient_idx + "to=".len();
    let recipient_end = header[recipient_start..]
        .find(|c: char| c == '<' || c.is_whitespace())
        .map(|offset| recipient_start + offset)
        .unwrap_or(header.len());
    let recipient = header[recipient_start..recipient_end].trim();
    let name = harmony_function_name_from_recipient(recipient);
    if name.is_empty() {
        return None;
    }

    let message_start = message_start + message_token.len();
    let message_end = raw_text[message_start..]
        .find("<|call|>")
        .map(|offset| message_start + offset)
        .or_else(|| {
            raw_text[message_start..]
                .find("<|end|>")
                .map(|offset| message_start + offset)
        })
        .unwrap_or(raw_text.len());
    let arguments = raw_text[message_start..message_end].trim();
    if arguments.is_empty() {
        return None;
    }
    Some(HarmonyFunctionCall {
        call_id: format!("call_{}", Uuid::new_v4().simple()),
        arguments: normalize_function_call_arguments(&name, arguments),
        name,
    })
}

fn harmony_function_name_from_recipient(recipient: &str) -> String {
    recipient
        .strip_prefix("functions.")
        .unwrap_or(recipient)
        .trim()
        .to_string()
}

fn normalize_function_call_arguments(name: &str, arguments: &str) -> String {
    let mut value = match serde_json::from_str::<Value>(arguments) {
        Ok(value) => value,
        Err(_) => match name {
            "exec_command" => parse_relaxed_exec_command_arguments(arguments)
                .unwrap_or_else(|| Value::String(arguments.to_string())),
            "write_stdin" => parse_relaxed_write_stdin_arguments(arguments)
                .unwrap_or_else(|| Value::String(arguments.to_string())),
            _ => Value::String(arguments.to_string()),
        },
    };

    if name == "exec_command" {
        if let Some(object) = value.as_object_mut() {
            if let Some(cmd_items) = object.get("cmd").and_then(Value::as_array) {
                if let Some(rewritten) = normalize_exec_command_array(cmd_items) {
                    for (key, rewritten_value) in rewritten {
                        object.insert(key, rewritten_value);
                    }
                } else {
                    let joined = cmd_items
                        .iter()
                        .map(|item| {
                            item.as_str()
                                .map(shell_escape)
                                .unwrap_or_else(|| shell_escape(&item.to_string()))
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    object.insert("cmd".to_string(), Value::String(joined));
                }
            }
        }
    }

    match value {
        Value::Object(_) => serde_json::to_string(&value).unwrap_or_else(|_| arguments.to_string()),
        Value::String(text) => {
            serde_json::to_string(&json!({ "cmd": text })).unwrap_or_else(|_| arguments.to_string())
        }
        _ => serde_json::to_string(&value).unwrap_or_else(|_| arguments.to_string()),
    }
}

fn shell_escape(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':'))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn normalize_exec_command_array(cmd_items: &[Value]) -> Option<Vec<(String, Value)>> {
    let first = cmd_items.first()?.as_str()?;
    if matches!(first, "bash" | "/bin/bash" | "sh" | "/bin/sh")
        && cmd_items.get(1).and_then(Value::as_str) == Some("-lc")
    {
        if let Some(script) = cmd_items.get(2).and_then(Value::as_str) {
            return Some(vec![
                ("cmd".to_string(), Value::String(script.to_string())),
                (
                    "shell".to_string(),
                    Value::String(if first.contains("bash") { "bash" } else { "sh" }.to_string()),
                ),
                ("login".to_string(), Value::Bool(false)),
            ]);
        }
    }
    None
}

fn parse_relaxed_exec_command_arguments(arguments: &str) -> Option<Value> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return Some(json!({ "cmd": trimmed }));
    }

    let cmd = extract_relaxed_json_string_field(trimmed, "cmd")
        .or_else(|| extract_relaxed_json_array_field(trimmed, "cmd").map(Value::Array))?;

    let mut object = serde_json::Map::new();
    object.insert("cmd".to_string(), cmd);

    for key in [
        "workdir",
        "shell",
        "justification",
        "sandbox_permissions",
        "prefix_rule",
    ] {
        if let Some(value) = extract_relaxed_json_string_field(trimmed, key) {
            object.insert(key.to_string(), value);
        } else if let Some(value) = extract_relaxed_json_array_field(trimmed, key) {
            object.insert(key.to_string(), Value::Array(value));
        }
    }

    for key in ["yield_time_ms", "max_output_tokens"] {
        if let Some(value) = extract_relaxed_json_integer_field(trimmed, key) {
            object.insert(key.to_string(), Value::Number(value.into()));
        }
    }

    for key in ["login", "tty"] {
        if let Some(value) = extract_relaxed_json_bool_field(trimmed, key) {
            object.insert(key.to_string(), Value::Bool(value));
        }
    }

    Some(Value::Object(object))
}

fn parse_relaxed_write_stdin_arguments(arguments: &str) -> Option<Value> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() || !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return None;
    }

    let session_id = extract_relaxed_json_integer_field(trimmed, "session_id")?;
    let mut object = serde_json::Map::new();
    object.insert("session_id".to_string(), Value::Number(session_id.into()));

    if let Some(value) = extract_relaxed_json_string_field(trimmed, "chars") {
        object.insert("chars".to_string(), value);
    }
    if let Some(value) = extract_relaxed_json_integer_field(trimmed, "yield_time_ms") {
        object.insert("yield_time_ms".to_string(), Value::Number(value.into()));
    }
    if let Some(value) = extract_relaxed_json_integer_field(trimmed, "max_output_tokens") {
        object.insert("max_output_tokens".to_string(), Value::Number(value.into()));
    }

    Some(Value::Object(object))
}

fn extract_relaxed_json_string_field(source: &str, key: &str) -> Option<Value> {
    let key_re = Regex::new(&format!(r#""{}"\s*:"#, regex::escape(key))).ok()?;
    let key_match = key_re.find(source)?;
    let remainder = &source[key_match.end()..];
    let value = remainder.trim_start();
    if !value.starts_with('"') {
        return None;
    }
    let value = &value[1..];
    let value_end = find_relaxed_string_end(value);
    let content = value[..value_end].trim_end_matches('"');
    Some(Value::String(unescape_relaxed_string(content)))
}

fn extract_relaxed_json_array_field(source: &str, key: &str) -> Option<Vec<Value>> {
    let key_re = Regex::new(&format!(r#""{}"\s*:"#, regex::escape(key))).ok()?;
    let key_match = key_re.find(source)?;
    let remainder = &source[key_match.end()..];
    let value = remainder.trim_start();
    if !value.starts_with('[') {
        return None;
    }
    let end = find_matching_bracket(value, '[', ']')?;
    let array_text = &value[..=end];
    serde_json::from_str::<Vec<Value>>(array_text).ok()
}

fn extract_relaxed_json_integer_field(source: &str, key: &str) -> Option<i64> {
    let re = Regex::new(&format!(r#""{}"\s*:\s*(-?\d+)"#, regex::escape(key))).ok()?;
    re.captures(source)?.get(1)?.as_str().parse::<i64>().ok()
}

fn extract_relaxed_json_bool_field(source: &str, key: &str) -> Option<bool> {
    let re = Regex::new(&format!(r#""{}"\s*:\s*(true|false)"#, regex::escape(key))).ok()?;
    Some(re.captures(source)?.get(1)?.as_str() == "true")
}

fn find_relaxed_string_end(source: &str) -> usize {
    let mut escaped = false;
    let bytes = source.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        if escaped {
            escaped = false;
            idx += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            idx += 1;
            continue;
        }
        if ch == '"' {
            let tail = source[idx + 1..].trim_start();
            if tail.starts_with(',') || tail.starts_with('}') {
                return idx;
            }
        }
        idx += 1;
    }
    source.len()
}

fn find_matching_bracket(source: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in source.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn unescape_relaxed_string(text: &str) -> String {
    serde_json::from_str::<String>(&format!(
        "\"{}\"",
        text.replace('\\', "\\\\").replace('"', "\\\"")
    ))
    .unwrap_or_else(|_| text.to_string())
}

fn extract_plaintext_harmony_final(raw_text: &str) -> Option<String> {
    for marker in ["assistantfinal", "final"] {
        let matches = raw_text.match_indices(marker).collect::<Vec<_>>();
        for (idx, _) in matches.into_iter().rev() {
            if marker == "final" && idx > 0 {
                let preceding = raw_text[..idx].chars().last();
                if preceding
                    .map(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/'))
                    .unwrap_or(false)
                {
                    continue;
                }
            }
            let payload_start = idx + marker.len();
            let remainder = &raw_text[payload_start..];
            let next_marker = [
                "assistantanalysis",
                "assistantcommentary",
                "assistantfinal",
                "<|start|>",
                "<|end|>",
                "<|return|>",
            ]
            .iter()
            .filter_map(|candidate| remainder.find(candidate))
            .min()
            .unwrap_or(remainder.len());
            let candidate = sanitize_harmony_completion_text(&remainder[..next_marker]);
            if !candidate.trim().is_empty() {
                return Some(candidate);
            }
        }
    }
    None
}

fn sanitize_harmony_channel_leakage(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    let saw_plaintext_harmony = contains_plaintext_harmony_marker(&text);

    loop {
        let mut stripped_any = false;
        for prefix in [
            "assistantfinal",
            "final",
            "assistantcommentary",
            "commentary",
            "assistantanalysis",
            "analysis",
        ] {
            if let Some(stripped) = text.strip_prefix(prefix) {
                text = stripped.trim_start().to_string();
                stripped_any = true;
            }
        }
        if !stripped_any {
            break;
        }
    }

    if let Some(idx) = find_plaintext_harmony_marker(&text) {
        text.truncate(idx);
    }

    if saw_plaintext_harmony
        || text.ends_with("assistant")
        || text.ends_with("analysis")
        || text.ends_with("commentary")
        || text.ends_with("final")
    {
        text = trim_trailing_incomplete_harmony_token(&text).to_string();
    }

    text.trim().to_string()
}

fn find_plaintext_harmony_marker(text: &str) -> Option<usize> {
    let markers = [
        "assistantanalysis",
        "assistantfinal",
        "assistantcommentary",
        "analysis",
        "final",
        "commentary",
    ];
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|idx| (idx, *marker)))
        .filter(|(idx, marker)| {
            if *idx == 0 {
                return false;
            }
            let preceding = text[..*idx].chars().last();
            let following = text[idx + marker.len()..].chars().next();
            let preceding_is_payload = preceding
                .map(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ')' | ']' | '"' | '\''))
                .unwrap_or(false);
            let following_looks_like_channel = following
                .map(|ch| ch.is_ascii_uppercase() || ch == '<' || ch == '{' || ch == '[')
                .unwrap_or(false);
            preceding_is_payload && following_looks_like_channel
        })
        .map(|(idx, _)| idx)
        .min()
}

fn contains_plaintext_harmony_marker(text: &str) -> bool {
    [
        "assistantanalysis",
        "assistantfinal",
        "assistantcommentary",
        "analysis",
        "final",
        "commentary",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn trim_trailing_incomplete_harmony_token(text: &str) -> &str {
    text.strip_suffix("assistant")
        .or_else(|| text.strip_suffix("analysis"))
        .or_else(|| text.strip_suffix("commentary"))
        .or_else(|| text.strip_suffix("final"))
        .unwrap_or(text)
}

fn extract_embedded_exact_text_request(text: &str) -> Option<String> {
    let quoted = Regex::new(r#"(?is)(?:please\s+)?(?:reply|respond|return|output|say|print|emit|antworte(?:\s+genau)?(?:\s+mit)?)\s+(?:with\s+|exactly\s+|genau\s+|mit\s+)?(?:"([^"\r\n]{1,128})"|'([^'\r\n]{1,128})')\s*(?:and\s+nothing\s+else|und\s+nichts\s+ander(?:em|es))?[.!]?\s*$"#)
        .expect("valid embedded exact-text quoted regex");
    let bare = Regex::new(r#"(?is)(?:please\s+)?(?:reply|respond|return|output|say|print|emit|antworte)\s+(?:with\s+)?exactly\s+([^\r\n]{1,128}?)\s*(?:and\s+nothing\s+else)?[.!]?\s*$"#)
        .expect("valid embedded exact-text bare regex");
    let english_bare = Regex::new(r#"(?is)(?:please\s+)?(?:reply|respond|return|output|say|print|emit)\s+(?:with\s+)?([^\r\n]{1,128}?)\s+and\s+nothing\s+else[.!]?\s*$"#)
        .expect("valid embedded exact-text english bare regex");
    let german_bare = Regex::new(r#"(?is)antworte(?:\s+genau)?(?:\s+mit)?\s+([^\r\n]{1,128}?)\s*(?:und\s+nichts\s+ander(?:em|es))?[.!]?\s*$"#)
        .expect("valid embedded exact-text german bare regex");

    let clauses = split_exact_request_clauses(text);
    for clause in clauses.into_iter().rev() {
        let clause = clause.trim();
        if clause.is_empty() {
            continue;
        }
        let explicit_german_exact_clause = is_explicit_german_exact_text_clause(clause);
        if let Some(captures) = quoted.captures(clause) {
            let candidate = captures
                .get(1)
                .or_else(|| captures.get(2))
                .map(|capture| capture.as_str().trim().to_string())
                .filter(|value| !value.is_empty());
            if candidate.is_some() {
                return candidate;
            }
        }
        if let Some(captures) = bare.captures(clause) {
            let candidate = captures
                .get(1)
                .map(|capture| {
                    capture
                        .as_str()
                        .trim()
                        .trim_end_matches(['.', '!', '?'])
                        .trim()
                        .to_string()
                })
                .filter(|value| !value.is_empty());
            if candidate.is_some() {
                return candidate;
            }
        }
        if let Some(captures) = english_bare.captures(clause) {
            let candidate = captures
                .get(1)
                .map(|capture| {
                    capture
                        .as_str()
                        .trim()
                        .trim_end_matches(['.', '!', '?'])
                        .trim()
                        .to_string()
                })
                .filter(|value| !value.is_empty());
            if candidate.is_some() {
                return candidate;
            }
        }
        if let Some(captures) = german_bare.captures(clause) {
            if !explicit_german_exact_clause {
                continue;
            }
            let candidate = captures
                .get(1)
                .map(|capture| {
                    capture
                        .as_str()
                        .trim()
                        .trim_end_matches(['.', '!', '?'])
                        .trim()
                        .to_string()
                })
                .filter(|value| !value.is_empty());
            if candidate.is_some() {
                return candidate;
            }
        }
    }

    None
}

fn is_explicit_german_exact_text_clause(clause: &str) -> bool {
    let normalized = clause.trim().to_ascii_lowercase();
    normalized.contains("genau") || normalized.contains("und nichts ander")
}

fn split_exact_request_clauses(text: &str) -> Vec<&str> {
    let mut clauses = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '\n' | '\r') {
            let end = idx + ch.len_utf8();
            clauses.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        clauses.push(&text[start..]);
    }
    clauses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::OpenResponsesInput;
    use crate::responses::TextFormat;

    #[test]
    fn parses_harmony_final_text() {
        let items = parse_harmony_response_items(
            "<|start|>assistant<|channel|>analysis<|message|>hidden<|end|><|start|>assistant<|channel|>final<|message|>hello world<|end|>",
        );
        assert_eq!(
            items,
            vec![HarmonyResponseItem::Message("hello world".to_string())]
        );
    }

    #[test]
    fn parses_harmony_function_call() {
        let items = parse_harmony_response_items(
            "<|start|>assistant<|channel|>commentary to=functions.exec_command<|constrain|>json<|message|>{\"cmd\":\"pwd\"}<|call|>",
        );
        assert_eq!(items.len(), 1);
        match &items[0] {
            HarmonyResponseItem::FunctionCall(call) => {
                assert_eq!(call.name, "exec_command");
                assert!(call.arguments.contains("\"cmd\":\"pwd\""));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn parses_legacy_harmony_function_call_order() {
        let items = parse_harmony_response_items(
            "<|start|>assistant to=functions.exec_command<|channel|>commentary <|constrain|>json<|message|>{\"cmd\":\"pwd\"}<|call|>",
        );
        assert_eq!(items.len(), 1);
        match &items[0] {
            HarmonyResponseItem::FunctionCall(call) => {
                assert_eq!(call.name, "exec_command");
                assert_eq!(call.arguments, "{\"cmd\":\"pwd\"}");
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn builds_completion_prompt_for_gpt_oss() {
        let request = OpenResponsesCreateRequest {
            model: "openai/gpt-oss-20b".to_string(),
            input: OpenResponsesInput::Text("hello".to_string()),
            instructions: Some("You are Codex.".to_string()),
            previous_response_id: None,
            stream: Some(true),
            stream_options: None,
            background: None,
            store: Some(true),
            metadata: None,
            include: None,
            max_output_tokens: Some(1024),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            max_tool_calls: None,
            reasoning: None,
            text: Some(crate::responses::TextConfig {
                format: Some(TextFormat::Text),
            }),
            truncation: None,
            stop_seqs: None,
            response_format: None,
            logit_bias: None,
            logprobs: false,
            n_choices: 1,
            repetition_penalty: None,
            top_k: None,
            grammar: None,
            min_p: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            web_search_options: None,
        };
        let messages = vec![
            Message {
                content: Some(MessageContent::from_text("You are Codex.".to_string())),
                role: "system".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                content: Some(MessageContent::from_text("hello".to_string())),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");
        assert!(completion.prompt.contains("<|start|>system<|message|>"));
        assert!(completion.prompt.contains("<|start|>developer<|message|>"));
        assert!(completion
            .prompt
            .contains("<|start|>user<|message|>hello<|end|>"));
        assert_eq!(completion.temperature, Some(0.0));
    }

    #[test]
    fn extracts_exact_text_override_from_latest_user_message() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Antworte genau mit CTOX_SOCKET_SMOKE_OK und nichts anderem.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        assert_eq!(
            extract_exact_text_override(&messages),
            Some("CTOX_SOCKET_SMOKE_OK".to_string())
        );
    }

    #[test]
    fn extracts_english_exact_text_override_without_exactly_keyword() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Reply with CTOX_MATRIX_OK and nothing else.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        assert_eq!(
            extract_exact_text_override(&messages),
            Some("CTOX_MATRIX_OK".to_string())
        );
    }

    #[test]
    fn does_not_treat_channel_instruction_as_exact_text_override() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Rufe jetzt das Tool get_cwd auf und antworte nicht im final channel.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        assert_eq!(extract_exact_text_override(&messages), None);
    }

    #[test]
    fn effective_exact_text_override_returns_none_for_workspace_build_request_with_tools() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Work only inside this workspace:\n/home/metricspace/ctox-e2e/workspace/cpp-chat-app\n\nBuild and verify the C++ project.\nReturn exactly CTOX_CPP_SMOKE_OK_1775582681 only after ./build/ctox_cpp_smoke succeeded.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = serde_json::from_value::<Vec<Tool>>(json!([
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "run command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": { "type": "string" }
                        },
                        "required": ["cmd"]
                    }
                }
            }
        ]))
        .expect("valid tools");

        assert_eq!(effective_exact_text_override(&messages, Some(&tools)), None);
    }

    #[test]
    fn exact_text_request_disables_thinking_and_tools() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "default",
            "input": "Antworte genau mit CTOX_SOCKET_SMOKE_OK und nichts anderem.",
            "reasoning": { "effort": "high" },
            "max_output_tokens": 131072,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "run command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cmd": { "type": "string" }
                            },
                            "required": ["cmd"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }))
        .expect("valid request");
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Antworte genau mit CTOX_SOCKET_SMOKE_OK und nichts anderem.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");

        assert_eq!(completion.max_tokens, Some(56));
        assert!(completion.tools.is_none());
        assert!(completion.tool_choice.is_none());
        assert!(completion.prompt.contains("Reasoning: none"));
    }

    #[test]
    fn build_completion_request_appends_harmony_stop_markers() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "default",
            "input": "hello",
            "stop": "DONE"
        }))
        .expect("valid request");
        let messages = vec![Message {
            content: Some(MessageContent::from_text("hello".to_string())),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");
        let stop_sequences = match completion.stop_seqs {
            Some(crate::openai::StopTokens::Multi(sequences)) => sequences,
            other => panic!("unexpected stop tokens: {other:?}"),
        };
        let stop_token_ids = match completion.stop_token_ids {
            Some(crate::openai::StopTokenIds::Multi(ids)) => ids,
            other => panic!("unexpected stop token ids: {other:?}"),
        };

        assert_eq!(
            stop_sequences,
            vec![
                "DONE".to_string(),
                "<|return|>".to_string(),
                "<|call|>".to_string()
            ]
        );
        assert_eq!(
            stop_token_ids,
            match harmony_stop_token_ids() {
                crate::openai::StopTokenIds::Multi(ids) => ids,
                crate::openai::StopTokenIds::Single(id) => vec![id],
            }
        );
    }

    #[test]
    fn workspace_build_request_keeps_thinking_and_harmony_tool_prompt_despite_exact_marker() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "default",
            "input": "Work only inside this workspace:\n/home/metricspace/ctox-e2e/workspace/cpp-chat-app\n\nCreate a bounded C++ verification project in this workspace.\n\nRequirements:\n- Use CMake.\n- Create at least these files: CMakeLists.txt, include/MessageQueue.h, src/MessageQueue.cpp, src/main.cpp.\n- Build it with: cmake -S . -B build && cmake --build build -j\n- Verify the binary with: ./build/ctox_cpp_smoke\n- On successful run, the program must print exactly CTOX_CPP_SMOKE_OK_1775582681\n- Do not answer before the files exist and the binary was executed successfully.\n- Keep the final answer extremely short and return exactly CTOX_CPP_SMOKE_OK_1775582681",
            "reasoning": { "effort": "high" },
            "max_output_tokens": 131072,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "run command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cmd": { "type": "string" }
                            },
                            "required": ["cmd"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }))
        .expect("valid request");
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Work only inside this workspace:\n/home/metricspace/ctox-e2e/workspace/cpp-chat-app\n\nCreate a bounded C++ verification project in this workspace.\n\nRequirements:\n- Use CMake.\n- Create at least these files: CMakeLists.txt, include/MessageQueue.h, src/MessageQueue.cpp, src/main.cpp.\n- Build it with: cmake -S . -B build && cmake --build build -j\n- Verify the binary with: ./build/ctox_cpp_smoke\n- On successful run, the program must print exactly CTOX_CPP_SMOKE_OK_1775582681\n- Do not answer before the files exist and the binary was executed successfully.\n- Keep the final answer extremely short and return exactly CTOX_CPP_SMOKE_OK_1775582681".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");

        assert_eq!(completion.max_tokens, Some(131072));
        assert!(completion.tools.is_none());
        assert!(completion.tool_choice.is_none());
        assert!(completion.prompt.contains("Reasoning: low"));
        assert!(completion.prompt.contains("namespace functions"));
        assert!(completion.prompt.contains("type exec_command = (_: {"));
    }

    #[test]
    fn forced_zero_arg_tool_choice_keeps_selected_tool_namespace_in_prompt() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "default",
            "input": "Rufe jetzt das Tool get_cwd auf.",
            "reasoning": { "effort": "none" },
            "max_output_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_cwd",
                        "description": "Return the current working directory.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "get_cwd",
                    "description": "Return the current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }
        }))
        .expect("valid request");
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Rufe jetzt das Tool get_cwd auf.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");

        assert!(completion.prompt_tokens.is_some());
        assert!(completion.prompt.contains("Reasoning: low"));
        assert!(completion.prompt.contains("namespace functions"));
        assert!(completion.prompt.contains("type get_cwd = () => any;"));
        assert!(!completion.prompt.contains("Call only functions.get_cwd."));
        assert!(!completion
            .prompt
            .contains("Your next reply must be exactly one Harmony function call."));
    }

    #[test]
    fn forced_zero_arg_tool_choice_with_channel_instruction_keeps_tool_namespace_in_prompt() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "model": "default",
            "input": "Rufe jetzt das Tool get_cwd auf und antworte nicht im final channel.",
            "reasoning": { "effort": "none" },
            "max_output_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_cwd",
                        "description": "Return the current working directory.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "get_cwd",
                    "description": "Return the current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }
        }))
        .expect("valid request");
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Rufe jetzt das Tool get_cwd auf und antworte nicht im final channel.".to_string(),
            )),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let completion = build_completion_request(&request, &messages, "openai/gpt-oss-20b");

        assert!(completion.prompt_tokens.is_some());
        assert!(completion.prompt.contains("Reasoning: low"));
        assert!(completion.prompt.contains("namespace functions"));
        assert!(completion.prompt.contains("type get_cwd = () => any;"));
        assert!(!completion.prompt.contains("Call only functions.get_cwd."));
        assert!(!completion
            .prompt
            .contains("Your next reply must be exactly one Harmony function call."));
    }

    #[test]
    fn builds_followup_request_for_analysis_only_completion() {
        let initial = CompletionRequest {
            model: "openai/gpt-oss-20b".to_string(),
            prompt: "<|start|>system<|message|>test".to_string(),
            prompt_tokens: Some(engine_core::harmony::encode_harmony_prompt_tokens(
                "<|start|>system<|message|>test",
            )),
            best_of: None,
            echo_prompt: false,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            max_tokens: Some(64),
            n_choices: 1,
            stop_seqs: None,
            stop_token_ids: None,
            stream: Some(false),
            temperature: Some(0.0),
            top_p: None,
            suffix: None,
            _user: None,
            tools: None,
            tool_choice: None,
            top_k: None,
            grammar: None,
            min_p: None,
            repetition_penalty: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            truncate_sequence: None,
        };
        let completion = CompletionResponse {
            id: "0".to_string(),
            choices: vec![engine_core::CompletionChoice {
                text: "<|channel|>analysis<|message|>thinking".to_string(),
                index: 0,
                finish_reason: "stop".to_string(),
                logprobs: None,
            }],
            created: 0,
            model: "openai/gpt-oss-20b".to_string(),
            system_fingerprint: "local".to_string(),
            object: "text_completion".to_string(),
            usage: engine_core::Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
        };

        let followup = build_followup_completion_request(&initial, &completion)
            .expect("follow-up request should be generated");
        assert!(followup
            .prompt
            .ends_with("<|channel|>analysis<|message|>thinking<|end|><|return|>"));
        assert_eq!(
            followup.prompt_tokens,
            Some(engine_core::harmony::encode_harmony_prompt_tokens(
                &followup.prompt
            ))
        );
    }

    #[test]
    fn render_harmony_conversation_uses_official_tool_call_and_output_shapes() {
        let rendered = render_harmony_conversation(&[
            Message {
                content: None,
                role: "assistant".to_string(),
                name: None,
                tool_calls: Some(vec![crate::openai::ToolCall {
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
        ]);

        assert!(rendered.contains(
            "<|start|>assistant<|channel|>commentary to=functions.exec_command <|constrain|>json<|message|>\"{\\\"cmd\\\":\\\"pwd\\\"}\"<|call|>"
        ));
        assert!(rendered.contains(
            "<|start|>functions.exec_command to=assistant<|channel|>commentary<|message|>\"/home/metricspace/CTOX\"<|end|>"
        ));
    }

    #[test]
    fn prepare_harmony_chat_messages_injects_forced_tool_developer_block() {
        let messages = vec![Message {
            content: Some(MessageContent::from_text(
                "Return only the working directory.".to_string(),
            )),
            role: "system".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![serde_json::from_value::<Tool>(serde_json::json!({
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

        let (rewritten, filtered_tools) = prepare_harmony_chat_messages(
            &messages,
            Some(&tools),
            Some(&engine_core::ToolChoice::Tool(tools[0].clone())),
        );

        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].role, "developer");
        let developer = rewritten[0]
            .content
            .as_ref()
            .and_then(MessageContent::to_text)
            .expect("developer text");
        assert_eq!(developer, "Return only the working directory.");
        let filtered_tools =
            filtered_tools.expect("selected tool should be preserved for template rendering");
        assert_eq!(filtered_tools.len(), 1);
        assert_eq!(filtered_tools[0].function.name, "get_cwd");
    }
}
