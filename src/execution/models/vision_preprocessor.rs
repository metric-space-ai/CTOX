// Origin: CTOX
// License: Apache-2.0
//
// Vision preprocessor — core of the CTOX vision path.
//
// The tools carried by `view_image` (and any future screenshot / OCR tool)
// can emit `input_image` content blocks. The primary LLM may or may not
// accept those natively — OpenAI's gpt-4o, Anthropic's Claude 3/4, local
// Qwen 3.5-VL and Gemma-4 families can; GPT-OSS, Kimi, Nemotron-Cascade,
// GLM-4.7, MiniMax text-only cannot.
//
// To keep the guarantee "tools can always evaluate images", this module
// intercepts every `/v1/responses` POST payload before adapter dispatch
// and — when the primary model can't natively consume images — describes
// each image using the configured Vision aux model (Qwen3-VL-2B-Instruct
// by default) and replaces the image block with a plain text block:
//
//   {"type":"input_text","text":"[Image description (via aux-vision): ...]"}
//
// When the primary model CAN see images, the preprocessor is a no-op and
// the adapter is responsible for forwarding the image block unchanged.
//
// The aux endpoint is called synchronously via `ureq` (matching the rest
// of the gateway's HTTP style). On aux-failure the image block is replaced
// with a clear error text so the primary model is told "vision aux not
// available", rather than silently receiving a missing image (which would
// encourage hallucination).

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::inference::engine;
use crate::inference::runtime_env;
use crate::inference::runtime_kernel;
use crate::inference::runtime_state;

/// Environment variable that disables vision preprocessing entirely.
/// When set to a truthy value, image blocks are passed through unchanged
/// and no aux call is made.
pub const VISION_AUX_DISABLE_ENV: &str = "CTOX_DISABLE_VISION_BACKEND";

/// Default describer prompt sent alongside each image to the aux model.
const DEFAULT_DESCRIBE_PROMPT: &str =
    "Describe this image in detail. Include visible text (transcribed verbatim), \
    objects and their spatial arrangement, colors, any charts or diagrams with \
    their data, and the overall context. Be thorough — this description replaces \
    the image for a downstream model that cannot see it.";

/// HTTP timeout for the aux describer call. Vision describe for a 2B model
/// on a single GPU typically returns in 2–8 seconds; 60s keeps headroom
/// for cold-starts and larger images.
const AUX_HTTP_TIMEOUT_SECS: u64 = 60;

/// Preprocess the `input` array of a `/v1/responses` POST payload in place.
///
/// Walks every `message.content[]` and every `function_call_output.output[]`
/// looking for `input_image` / `image_url` blocks. For each found image,
/// if `model_supports_vision_primary` is false, calls the aux endpoint,
/// inserts a text block describing the image, and drops the image block.
/// If the primary model supports vision, the image block is left untouched.
///
/// Returns `Ok(true)` if the payload was mutated, `Ok(false)` if untouched.
pub fn preprocess_responses_payload(
    root: &Path,
    primary_model: Option<&str>,
    payload: &mut Value,
) -> Result<bool> {
    // Kill-switch
    if is_vision_preprocessor_disabled(root) {
        return Ok(false);
    }

    let Some(input) = payload.get_mut("input").and_then(Value::as_array_mut) else {
        return Ok(false);
    };

    // Capability check — if the primary model can already see images,
    // no-op. Resolution is delegated to the central registry lookup
    // (`engine::model_supports_vision`) so this path stays in sync with
    // the SUPPORTED_VISION_MODELS / VISION_API_MODELS / ChatFamilyCatalog
    // data sources without duplicating them.
    let primary_supports_vision = primary_model
        .map(engine::model_supports_vision)
        .unwrap_or(false);

    if primary_supports_vision {
        return Ok(false);
    }

    let aux_endpoint = match resolve_vision_aux_endpoint(root) {
        Some(endpoint) => endpoint,
        None => {
            // No aux configured and no primary vision — replace images with
            // structured error so the primary model is told explicitly.
            return replace_images_with_error(
                input,
                "vision aux model not configured (set CTOX_VISION_BASE_URL or \
                enable the Qwen3-VL-2B-Instruct aux in settings)",
            );
        }
    };

    let mut mutated = false;
    for item in input.iter_mut() {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");

        match item_type {
            "message" => {
                if let Some(Value::Array(content)) = object.get_mut("content") {
                    if describe_images_in_content(content, &aux_endpoint)? {
                        mutated = true;
                    }
                }
            }
            "function_call_output" => {
                // Tool outputs can be either a plain string or an array.
                // `view_image` uses the array form with InputImage entries.
                if let Some(Value::Array(output)) = object.get_mut("output") {
                    if describe_images_in_content(output, &aux_endpoint)? {
                        mutated = true;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(mutated)
}

fn is_vision_preprocessor_disabled(root: &Path) -> bool {
    runtime_env::env_or_config(root, VISION_AUX_DISABLE_ENV)
        .map(|value| {
            let trimmed = value.trim();
            matches!(
                trimmed,
                "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

/// Resolve the vision aux endpoint from either an explicit env override or
/// the managed InferenceRuntimeKernel binding for the Vision role.
fn resolve_vision_aux_endpoint(root: &Path) -> Option<VisionAuxEndpoint> {
    if let Some(base_url) = runtime_env::env_or_config(root, "CTOX_VISION_BASE_URL") {
        let base = base_url.trim().trim_end_matches('/');
        if !base.is_empty() {
            let model = runtime_env::env_or_config(root, "CTOX_VISION_MODEL").unwrap_or_else(|| {
                // Fall back to the registry's default-for-role Vision aux
                // selection so the model name never leaks into this module
                // as a hardcoded literal.
                engine::auxiliary_model_selection(engine::AuxiliaryRole::Vision, None)
                    .request_model
                    .to_string()
            });
            return Some(VisionAuxEndpoint {
                chat_completions_url: format!("{base}/v1/chat/completions"),
                model,
            });
        }
    }

    // Fall back to the resolved runtime kernel binding (Phase C plumbing).
    let kernel = runtime_kernel::InferenceRuntimeKernel::resolve(root).ok()?;
    let binding = kernel.binding_for_auxiliary_role(engine::AuxiliaryRole::Vision)?;
    let base_url = binding.base_url.trim().trim_end_matches('/').to_string();
    if base_url.is_empty() {
        return None;
    }
    Some(VisionAuxEndpoint {
        chat_completions_url: format!("{base_url}/v1/chat/completions"),
        model: binding.request_model.clone(),
    })
}

struct VisionAuxEndpoint {
    chat_completions_url: String,
    model: String,
}

/// Walk a content array, describe each image block via the aux, and
/// replace the image block with a text block. Returns true if any
/// replacements were made.
fn describe_images_in_content(content: &mut Vec<Value>, aux: &VisionAuxEndpoint) -> Result<bool> {
    let mut mutated = false;
    let mut i = 0;
    while i < content.len() {
        let is_image = content[i]
            .get("type")
            .and_then(Value::as_str)
            .map(|t| t == "input_image" || t == "image_url")
            .unwrap_or(false);
        if !is_image {
            i += 1;
            continue;
        }
        let image_ref = extract_image_reference(&content[i]);
        let description = match image_ref {
            Some(img_ref) => describe_image_via_aux(aux, &img_ref).unwrap_or_else(|err| {
                format!(
                    "[Vision aux describe failed: {err}. Image omitted from context.]"
                )
            }),
            None => "[Image block had no parseable URL or data payload — omitted.]".to_string(),
        };
        content[i] = json!({
            "type": "input_text",
            "text": format!("[Image description (via aux-vision): {description}]"),
        });
        mutated = true;
        i += 1;
    }
    Ok(mutated)
}

/// Extract the canonical image reference (URL or data-URI) from either the
/// OpenResponses-native shape or the OpenAI chat-compat shape.
fn extract_image_reference(block: &Value) -> Option<String> {
    let block = block.as_object()?;
    // OpenResponses-native: {type:"input_image", image_url:"...", image_data:"<base64>"}
    if let Some(url) = block.get("image_url").and_then(Value::as_str) {
        if !url.trim().is_empty() {
            return Some(url.to_string());
        }
    }
    if let Some(data) = block.get("image_data").and_then(Value::as_str) {
        if !data.trim().is_empty() {
            let mime = block
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            return Some(format!("data:{mime};base64,{data}"));
        }
    }
    // OpenAI chat-compat: {type:"image_url", image_url:{url:"..."}}
    if let Some(inner) = block.get("image_url").and_then(Value::as_object) {
        if let Some(url) = inner.get("url").and_then(Value::as_str) {
            if !url.trim().is_empty() {
                return Some(url.to_string());
            }
        }
    }
    None
}

/// Call the aux describer with a single image + describe prompt.
fn describe_image_via_aux(aux: &VisionAuxEndpoint, image_ref: &str) -> Result<String> {
    let body = json!({
        "model": aux.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_ref}},
                {"type": "text", "text": DEFAULT_DESCRIBE_PROMPT},
            ],
        }],
        "max_tokens": 768,
        "temperature": 0.1,
    });

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(AUX_HTTP_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(10))
        .build();

    let serialized = serde_json::to_string(&body)
        .context("failed to serialize vision aux describe body")?;
    let response = agent
        .post(&aux.chat_completions_url)
        .set("content-type", "application/json")
        .send_string(&serialized)
        .context("vision aux HTTP call failed")?;

    let body_text = response
        .into_string()
        .context("vision aux response body could not be read")?;
    let parsed: Value =
        serde_json::from_str(&body_text).context("vision aux response was not valid JSON")?;

    let choices = parsed
        .get("choices")
        .and_then(Value::as_array)
        .context("vision aux response has no `choices` array")?;
    let first = choices
        .first()
        .context("vision aux response `choices` array is empty")?;
    let content = first
        .get("message")
        .and_then(|msg| msg.get("content"))
        .and_then(Value::as_str)
        .context("vision aux response has no `message.content` string")?;
    let trimmed = content.trim().to_string();
    if trimmed.is_empty() {
        anyhow::bail!("vision aux returned empty content");
    }
    Ok(trimmed)
}

/// When no aux is available, replace every image block with a clear error
/// text so the primary model is explicitly told what happened (rather than
/// silently losing images and potentially hallucinating).
fn replace_images_with_error(input: &mut Vec<Value>, reason: &str) -> Result<bool> {
    let mut mutated = false;
    for item in input.iter_mut() {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        let array_field = match item_type {
            "message" => "content",
            "function_call_output" => "output",
            _ => continue,
        };
        if let Some(Value::Array(content)) = object.get_mut(array_field) {
            let mut i = 0;
            while i < content.len() {
                let is_image = content[i]
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|t| t == "input_image" || t == "image_url")
                    .unwrap_or(false);
                if is_image {
                    content[i] = json!({
                        "type": "input_text",
                        "text": format!(
                            "[Vision aux unavailable: {reason}. Image omitted from context.]"
                        ),
                    });
                    mutated = true;
                }
                i += 1;
            }
        }
    }
    Ok(mutated)
}

// Resolve primary model from runtime state when not supplied by caller.
#[allow(dead_code)]
pub fn resolve_primary_model(root: &Path) -> Option<String> {
    runtime_state::load_or_resolve_runtime_state(root)
        .ok()
        .and_then(|state| {
            state
                .active_or_selected_model()
                .map(ToOwned::to_owned)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_image_reference_handles_native_url() {
        let block = json!({
            "type": "input_image",
            "image_url": "https://example.com/a.png",
        });
        assert_eq!(
            extract_image_reference(&block),
            Some("https://example.com/a.png".to_string())
        );
    }

    #[test]
    fn extract_image_reference_handles_base64_data() {
        let block = json!({
            "type": "input_image",
            "image_data": "AAAA",
            "mime_type": "image/jpeg",
        });
        assert_eq!(
            extract_image_reference(&block),
            Some("data:image/jpeg;base64,AAAA".to_string())
        );
    }

    #[test]
    fn extract_image_reference_handles_openai_compat_shape() {
        let block = json!({
            "type": "image_url",
            "image_url": {"url": "https://example.com/b.webp"},
        });
        assert_eq!(
            extract_image_reference(&block),
            Some("https://example.com/b.webp".to_string())
        );
    }

    #[test]
    fn replace_images_with_error_replaces_message_content_images() {
        let mut input = vec![json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.com/x.png"},
            ],
        })];
        let mutated = replace_images_with_error(&mut input, "testing").unwrap();
        assert!(mutated);
        let content = input[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[1]["type"], "input_text");
        assert!(content[1]["text"]
            .as_str()
            .unwrap()
            .contains("Vision aux unavailable"));
    }

    #[test]
    fn replace_images_with_error_replaces_tool_output_images() {
        let mut input = vec![json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_image", "image_url": "https://example.com/y.png"},
            ],
        })];
        let mutated = replace_images_with_error(&mut input, "no aux").unwrap();
        assert!(mutated);
        let output = input[0]["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "input_text");
    }

    #[test]
    fn known_api_vision_models_are_flagged() {
        assert!(engine::model_supports_vision("anthropic/claude-sonnet-4.6"));
        assert!(engine::model_supports_vision("gpt-5.4"));
        assert!(!engine::model_supports_vision("openai/gpt-oss-20b"));
        assert!(!engine::model_supports_vision("moonshotai/kimi-k2.5"));
    }
}
