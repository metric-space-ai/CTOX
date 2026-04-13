use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;

use anyhow::Result;
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use minijinja::{context, value::Kwargs, Environment, Error, ErrorKind, Value};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{MessageContent, Tool};

const SUPPORTED_ALTERNATE_EOS: &[&str] = &[
    "<|im_end|>",      // Handle ChatML case
    "<end_of_turn>",   // Handle Gemma2 chat case
    "<|end_of_text|>", // Hermes
    "<|end|>",         // Phi-3, Phi-3.5, Harmony
    "<|eot_id|>",      // Llama 3
    "<|message|>",     // Harmony
    "<|start|>",       // Harmony
    "<|channel|>",     // Harmony
];

fn is_valid_alternate_eos(chat_template: &ChatTemplate, token: &str) -> bool {
    if !chat_template.is_harmony_format() {
        return true;
    }

    // Harmony responses structure assistant channels and tool calls with
    // special tokens like `<|start|>`, `<|channel|>`, `<|message|>`,
    // `<|end|>`, `<|return|>`, and `<|call|>`. Treating them as EOS causes GPT-OSS
    // generations to terminate immediately or to drop tool calls/final answers
    // at the point where the structure token appears.
    !matches!(
        token,
        "<|message|>" | "<|start|>" | "<|channel|>" | "<|end|>" | "<|return|>" | "<|call|>"
    )
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkPadTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    pub bos_token: Option<BeginEndUnkPadTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    pub eos_token: Option<BeginEndUnkPadTok>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pub pad_token: Option<BeginEndUnkPadTok>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    pub unk_token: Option<BeginEndUnkPadTok>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub(crate) fn get_template_contents(&self) -> Vec<String> {
        match self.chat_template.as_ref() {
            Some(t) => match &t.0 {
                Either::Left(s) => vec![s.clone()],
                Either::Right(vec) => vec.iter().flat_map(|m| m.values().cloned()).collect(),
            },
            None => vec![],
        }
    }

    /// Check if this chat template uses OpenAI Harmony format.
    pub fn is_harmony_format(&self) -> bool {
        self.get_template_contents()
            .iter()
            .any(|t| crate::harmony::is_harmony_template(t))
    }

    /// Check if this chat template uses `<think>...</think>` tags for reasoning.
    ///
    /// This is mutually exclusive with Harmony format - if the template uses
    /// Harmony format, this returns false even if think tags are present.
    pub fn uses_think_tags(&self) -> bool {
        // Don't enable if Harmony format is detected (mutual exclusivity)
        if self.is_harmony_format() {
            return false;
        }

        self.get_template_contents()
            .iter()
            .any(|t| crate::think_tags::is_think_tag_template(t))
    }

    /// Check if the template uses Gemma 4 channel-based reasoning tags.
    pub fn uses_channel_tags(&self) -> bool {
        if self.is_harmony_format() {
            return false;
        }

        self.get_template_contents()
            .iter()
            .any(|t| crate::channel_tags::is_channel_tag_template(t))
    }

    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

pub fn calculate_eos_tokens(
    chat_template: &ChatTemplate,
    gen_conf: Option<GenerationConfig>,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut eos_tok_ids = chat_template.eos_tok().map(|x| vec![x]).unwrap_or_default();
    let mut bos_tok_ids = chat_template.bos_tok().map(|b| vec![b]).unwrap_or_default();

    let templates = chat_template.get_template_contents();

    for alternate in SUPPORTED_ALTERNATE_EOS {
        if !is_valid_alternate_eos(chat_template, alternate) {
            continue;
        }
        if tokenizer.get_vocab(true).contains_key(*alternate)
            && templates.iter().any(|t| t.contains(*alternate))
        {
            eos_tok_ids.push(alternate.to_string())
        }
    }

    if let Some(gen_conf) = gen_conf {
        if let Some(eos_field) = gen_conf.eos_token_id {
            let ids = match eos_field {
                Either::Left(id) => vec![id],
                Either::Right(ids) => ids,
            };
            for id in ids {
                let s = tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| panic!("Unable to decode id {id})"));
                if !eos_tok_ids.contains(&s) && is_valid_alternate_eos(chat_template, &s) {
                    eos_tok_ids.push(s);
                }
            }
        }

        if let Some(bos_field) = gen_conf.bos_token_id {
            let ids = match bos_field {
                Either::Left(id) => vec![id],
                Either::Right(ids) => ids,
            };
            for id in ids {
                let s = tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| panic!("Unable to decode id {id})"));
                if !bos_tok_ids.contains(&s) {
                    bos_tok_ids.push(s);
                }
            }
        }
    }

    eos_tok_ids = eos_tok_ids
        .into_iter()
        .filter(|tok| is_valid_alternate_eos(chat_template, tok))
        .dedup()
        .collect::<Vec<_>>();
    bos_tok_ids = bos_tok_ids.into_iter().dedup().collect::<Vec<_>>();

    let bos_render = bos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");
    let eos_render = eos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");

    info!(
        "bos_toks = {bos_render}, eos_toks = {eos_render}, unk_tok = {}",
        chat_template.unk_tok().unwrap_or("`None`".to_string()),
    );

    let mut eos_toks = Vec::new();
    for eos_tok in eos_tok_ids {
        eos_toks.push(
            tokenizer
                .get_vocab(true)
                .get(&eos_tok)
                .copied()
                .unwrap_or_else(|| panic!("Unable to extract `{eos_tok}` EOS token.")),
        )
    }
    eos_toks
}

#[cfg(test)]
mod tests {
    use super::*;
    use either::Either;
    use indexmap::IndexMap;

    #[test]
    fn harmony_structural_tokens_are_not_treated_as_eos() {
        let chat_template: ChatTemplate = serde_json::from_str(
            r#"{
                "chat_template": "<|start|>system<|message|>test<|end|><|start|>assistant<|channel|>",
                "eos_token": "<|end|>"
            }"#,
        )
        .expect("chat template should deserialize");

        assert!(chat_template.is_harmony_format());
        assert!(!is_valid_alternate_eos(&chat_template, "<|end|>"));
        assert!(!is_valid_alternate_eos(&chat_template, "<|start|>"));
        assert!(!is_valid_alternate_eos(&chat_template, "<|channel|>"));
        assert!(!is_valid_alternate_eos(&chat_template, "<|return|>"));
        assert!(!is_valid_alternate_eos(&chat_template, "<|call|>"));
        assert!(!is_valid_alternate_eos(&chat_template, "<|message|>"));
    }

    #[test]
    fn harmony_prompt_disables_analysis_when_thinking_is_off() {
        let mut user_message = IndexMap::new();
        user_message.insert("role".to_string(), Either::Left("user".to_string()));
        user_message.insert(
            "content".to_string(),
            Either::Left("Reply with exactly OK.".to_string()),
        );
        let rendered = apply_chat_template_to(
            vec![user_message],
            true,
            Some(false),
            Some(ReasoningEffort::None),
            &ChatTemplateValue(Either::Left(
                "<|start|>system<|message|>Reasoning: {{ reasoning_effort }}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>{% for message in messages %}<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}".to_string(),
            )),
            None,
            None,
            None,
            Vec::new(),
        )
        .expect("harmony prompt should render");

        assert!(rendered.contains("Reasoning: none"));
        assert!(rendered.contains(
            "Do not emit analysis. Use commentary only for tool calls and final for the answer."
        ));
        assert!(rendered.contains(
            "# Valid channels: commentary, final. Channel must be included for every message."
        ));
        assert!(!rendered.contains(
            "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        ));
        assert!(rendered.ends_with("<|start|>assistant"));
    }

    #[test]
    fn harmony_prompt_keeps_assistant_open_for_channel_selection_with_tools() {
        let mut user_message = IndexMap::new();
        user_message.insert("role".to_string(), Either::Left("user".to_string()));
        user_message.insert(
            "content".to_string(),
            Either::Left("Use a tool.".to_string()),
        );
        let rendered = apply_chat_template_to(
            vec![user_message],
            true,
            Some(false),
            Some(ReasoningEffort::None),
            &ChatTemplateValue(Either::Left(
                "<|start|>system<|message|>Reasoning: {{ reasoning_effort }}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>{% for message in messages %}<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}".to_string(),
            )),
            None,
            None,
            None,
            vec![Tool {
                tp: crate::ToolType::Function,
                function: crate::Function {
                    description: Some("demo".to_string()),
                    name: "exec_command".to_string(),
                    parameters: Some(HashMap::new()),
                },
            }],
        )
        .expect("harmony prompt should render with tools");

        assert!(rendered.ends_with("<|start|>assistant"));
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    bos_token_id: Option<Either<u32, Vec<u32>>>,
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    eos_token_id: Option<Either<u32, Vec<u32>>>,
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    if let Ok(indent) = kwargs.get("indent") {
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut ser).unwrap();
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        serde_json::to_string(&value).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        Value::from_safe_string(rv)
    })
}

fn strftime_now(fmt: String) -> Result<String, minijinja::Error> {
    let date = chrono::Utc::now();
    let date_string = date.format(&fmt).to_string();
    Ok(date_string)
}

use crate::request::ReasoningEffort;

fn is_gemma4_tool_template(template: &str) -> bool {
    template.contains("<|tool_call>") && template.contains("<tool_call|>")
}

fn preprocess_gemma4_tool_messages(messages: &mut Vec<IndexMap<String, MessageContent>>) {
    let mut merges: Vec<(usize, usize)> = Vec::new();
    for i in 0..messages.len() {
        let is_tool = messages[i]
            .get("role")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .is_some_and(|r| r == "tool");
        if !is_tool {
            continue;
        }

        if let Some(asst_idx) = (0..i).rev().find(|&j| {
            messages[j]
                .get("role")
                .and_then(|v| match v {
                    Either::Left(s) => Some(s.as_str()),
                    _ => None,
                })
                .is_some_and(|r| r == "assistant")
        }) {
            merges.push((i, asst_idx));
        }
    }

    if merges.is_empty() {
        return;
    }

    let mut asst_responses: HashMap<usize, Vec<IndexMap<String, serde_json::Value>>> =
        HashMap::new();
    for &(tool_idx, asst_idx) in &merges {
        let tool_msg = &messages[tool_idx];
        let name = tool_msg
            .get("name")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".to_string());
        let content = tool_msg
            .get("content")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let response_value: serde_json::Value =
            serde_json::from_str(&content).unwrap_or(serde_json::Value::String(content));

        let mut entry = IndexMap::new();
        entry.insert("name".to_string(), serde_json::Value::String(name));
        entry.insert("response".to_string(), response_value);
        asst_responses.entry(asst_idx).or_default().push(entry);
    }

    for (asst_idx, responses) in asst_responses {
        messages[asst_idx].insert("tool_responses".to_string(), Either::Right(responses));
        if messages[asst_idx].contains_key("tool_calls")
            || !messages[asst_idx].contains_key("content")
        {
            messages[asst_idx].insert("content".to_string(), Either::Left(String::new()));
        }
    }

    let mut to_remove: Vec<usize> = merges.iter().map(|&(tool_idx, _)| tool_idx).collect();
    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        messages.remove(idx);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_chat_template_to(
    mut messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    template: &ChatTemplateValue,
    bos_tok: Option<String>,
    eos_tok: Option<String>,
    unk_tok: Option<String>,
    tools: Vec<Tool>,
) -> Result<String> {
    let mut env = Environment::new();

    // enable python methods such as .strip()
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let resolved_template = match &template.0 {
        Either::Left(x) => x.clone(),
        Either::Right(map) => {
            let mut template = None;
            let has_tool_use = map.iter().any(|t| {
                t.get("name").is_some_and(|name| name == "tool_use") || t.contains_key("tool_use")
            });
            let must_use_tool_template = !tools.is_empty();

            if must_use_tool_template && !has_tool_use {
                anyhow::bail!(
                    "Tools were provided but this chat template does not handle tool usage"
                );
            }

            for t in map {
                let name = t.get("name");
                if let Some(name) = name {
                    template = Some(t["template"].clone());
                    #[allow(clippy::if_same_then_else)]
                    if name == "tool_use" && !tools.is_empty() {
                        break;
                    } else if name == "default" && !must_use_tool_template {
                        break;
                    }
                } else if t.contains_key("tool_use") && !tools.is_empty() {
                    template = Some(t["tool_use"].clone());
                    break;
                } else if t.contains_key("default") && !must_use_tool_template {
                    template = Some(t["default"].clone());
                    break;
                }
            }

            let Some(template) = template else {
                anyhow::bail!("Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools.");
            };
            template
        }
    };

    if is_gemma4_tool_template(&resolved_template) {
        preprocess_gemma4_tool_messages(&mut messages);
    }

    #[derive(Serialize, Deserialize)]
    struct UntaggedContent(#[serde(with = "either::serde_untagged")] MessageContent);
    let mut new_messages = Vec::new();
    for message in messages {
        let mut new_message = IndexMap::new();
        for (k, v) in message {
            new_message.insert(k, UntaggedContent(v));
        }
        new_messages.push(new_message);
    }

    let mut template = resolved_template.replace("[::-1]", "|reverse");
    // Convert Python‑style descending ranges `range(..., -1, -1)` to a forward
    // range followed by Jinja’s `|reverse` filter so it works even when
    // negative‑step ranges aren’t supported.
    let re = Regex::new(r"range\((?P<expr>[^,]+),\s*-1,\s*-1\)").unwrap();
    template = re
        .replace_all(&template, |caps: &regex::Captures| {
            format!("range({})|reverse", &caps["expr"])
        })
        .into_owned();

    if template.contains("{{ meta }}") {
        // Fix for GLM4 models
        template = template.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
        template = template.replace("{{ meta }}", "");
    }
    if template.contains("{% generation %}") && template.contains("{% endgeneration %}") {
        // Strip for smollm3 models
        template = template.replace("{% generation %}", "");
        template = template.replace("{% endgeneration %}", "");
    }

    env.add_template("chat_template", &template)?;
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", tojson);
    env.add_function("strftime_now", strftime_now);
    let tmpl = env.get_template("chat_template").unwrap();

    let date = chrono::Utc::now();
    let date_string = date.format("%d, %B, %Y").to_string();

    // Convert reasoning effort to string for template
    let reasoning_effort_str = reasoning_effort.map(|r| r.as_str()).unwrap_or("medium");

    // Detect builtin tools from the tools list
    // Known builtin tools for GPT-OSS/Harmony format: "browser", "python"
    // Known builtin tools for Llama 3.x: "wolfram_alpha", "web_search", "brave_search", "python", "code_interpreter"
    let builtin_tool_names = [
        "browser",
        "python",
        "code_interpreter",
        "web_search",
        "brave_search",
        "wolfram_alpha",
    ];
    let builtin_tools: Vec<&str> = tools
        .iter()
        .filter_map(|t| {
            let name = t.function.name.as_str();
            if builtin_tool_names.contains(&name) {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    let rendered = if tools.is_empty() {
        tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            date_string => date_string,
            enable_thinking => enable_thinking.unwrap_or(true),
            reasoning_effort => reasoning_effort_str,
        })?
    } else {
        tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            xml_tools => tools.clone(), // SmolLM3
            tools => tools,
            builtin_tools => builtin_tools,
            date_string => date_string,
            enable_thinking => enable_thinking.unwrap_or(true),
            reasoning_effort => reasoning_effort_str,
        })?
    };

    let rendered = if crate::harmony::is_harmony_template(&resolved_template)
        && matches!(enable_thinking, Some(false))
    {
        enforce_harmony_no_analysis_contract(rendered, !tools.is_empty())
    } else {
        rendered
    };

    maybe_log_rendered_prompt(&rendered);

    Ok(rendered)
}

fn enforce_harmony_no_analysis_contract(rendered: String, _prefer_commentary: bool) -> String {
    let rendered = rendered
        .replace(
            "Reasoning: none\n\n",
            "Reasoning: none\nDo not emit analysis. Use commentary only for tool calls and final for the answer.\n\n",
        )
        .replace(
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.",
            "# Valid channels: commentary, final. Channel must be included for every message.",
        );

    for suffix in [
        "<|start|>assistant<|channel|>commentary<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
    ] {
        if let Some(prefix) = rendered.strip_suffix(suffix) {
            return format!("{prefix}<|start|>assistant");
        }
    }

    rendered
}

fn maybe_log_rendered_prompt(rendered: &str) {
    let Some(file) = std::env::var("CTOX_ENGINE_LOG")
        .ok()
        .filter(|v| !v.trim().is_empty())
    else {
        return;
    };
    let Ok(mut f) = OpenOptions::new().append(true).create(true).open(file) else {
        return;
    };
    let time = chrono::offset::Local::now();
    let _ = f.write_all(format!("Rendered prompt at {time}: {rendered}\n\n").as_bytes());
}
