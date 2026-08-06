// refs: sdk/api, sdk/translator/types.go, internal/translator @ ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Protocol-neutral contracts used between wire translators and executors.
//! These replace repeated Go `gjson.Result` interpretation with explicit Rust
//! variants while retaining raw JSON for tool arguments and unknown extensions.

use crate::internal::translator::common::RawJson;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRequest {
    pub model: String,
    pub instructions: Option<String>,
    pub input: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub stream: bool,
    pub max_output_tokens: Option<u64>,
    pub reasoning: Option<ReasoningConfig>,
    pub extensions: BTreeMap<String, RawJson>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        media_type: Option<String>,
        data: String,
    },
    File {
        media_type: Option<String>,
        data: String,
        name: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: RawJson,
    },
    ToolResult {
        call_id: String,
        content: Vec<ContentPart>,
        is_error: bool,
    },
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    Unknown {
        kind: String,
        raw: RawJson,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: RawJson,
    pub namespace: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Named {
        name: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReasoningConfig {
    pub effort: ReasoningEffort,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReasoningEffort {
    None,
    Auto,
    Minimal,
    Low,
    Medium,
    High,
    ExtraHigh,
    Max,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamEvent {
    ResponseStarted {
        id: String,
        model: String,
    },
    OutputItemStarted {
        index: usize,
        item: ContentPart,
    },
    TextDelta {
        item_index: usize,
        text: String,
    },
    ReasoningDelta {
        item_index: usize,
        text: String,
    },
    ToolArgumentsDelta {
        item_index: usize,
        delta: Vec<u8>,
    },
    OutputItemFinished {
        index: usize,
        item: ContentPart,
    },
    Usage(Usage),
    ResponseFinished {
        id: String,
    },
    Error {
        code: Option<String>,
        message: String,
    },
}
