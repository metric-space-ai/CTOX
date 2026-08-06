// ref: internal/runtime/executor/codex_openai_images.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{Map, Value};

pub const CODEX_IMAGE_GENERATION_PATH: &str = "/v1/images/generations";
pub const CODEX_IMAGE_EDIT_PATH: &str = "/v1/images/edits";
pub const CODEX_DIRECT_IMAGE_GENERATION_PATH: &str = "/images/generations";
pub const CODEX_DIRECT_IMAGE_EDIT_PATH: &str = "/images/edits";
const MAX_MULTIPART_IMAGE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexImageAction {
    Generate,
    Edit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexImageResponseFormat {
    Base64Json,
    Url,
}

impl CodexImageResponseFormat {
    pub fn parse(value: Option<&str>) -> Self {
        if value.is_some_and(|value| value.trim().eq_ignore_ascii_case("url")) {
            Self::Url
        } else {
            Self::Base64Json
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexImagePreparedRequest {
    pub action: CodexImageAction,
    pub route_model: String,
    pub response_format: CodexImageResponseFormat,
    pub responses_body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexImageResult {
    pub base64_data: String,
    pub output_format: String,
    pub revised_prompt: Option<String>,
    pub size: Option<String>,
    pub quality: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexDirectImagePreparedRequest {
    pub endpoint_path: &'static str,
    pub model: String,
    pub content_type: String,
    pub body: Vec<u8>,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexImageError {
    InvalidJson,
    MissingPrompt,
    MissingImage,
    InvalidImageData,
    InvalidCompletion,
    UnsupportedDirectModel,
    UnsupportedEndpoint,
    InvalidMultipart,
    MultipartTooLarge,
}

impl fmt::Display for CodexImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidJson => "Codex image request is invalid JSON",
            Self::MissingPrompt => "Codex image request is missing a prompt",
            Self::MissingImage => "Codex image edit request is missing image input",
            Self::InvalidImageData => "Codex image result is invalid",
            Self::InvalidCompletion => "Codex image completion is invalid",
            Self::UnsupportedDirectModel => "Codex direct image model is unsupported",
            Self::UnsupportedEndpoint => "Codex image endpoint is unsupported",
            Self::InvalidMultipart => "Codex multipart image request is invalid",
            Self::MultipartTooLarge => "Codex multipart image request is too large",
        })
    }
}

impl std::error::Error for CodexImageError {}

pub fn codex_is_images_endpoint_path(path: &str) -> bool {
    matches!(
        path.trim_end_matches('/'),
        CODEX_IMAGE_GENERATION_PATH | CODEX_IMAGE_EDIT_PATH
    )
}

pub fn codex_openai_image_base_model(model: &str) -> &str {
    match model.trim() {
        "gpt-image-1" | "gpt-image-1-mini" | "gpt-image-1.5" | "gpt-image-2" => model.trim(),
        _ => "gpt-image-2",
    }
}

pub fn codex_direct_image_model(model: &str) -> Option<&str> {
    let model = model.trim().rsplit('/').next()?.trim();
    matches!(model, "gpt-image-1.5" | "gpt-image-2").then_some(model)
}

pub fn prepare_codex_direct_image_request(
    raw: &[u8],
    route_model: &str,
    request_path: &str,
    content_type: Option<&str>,
    stream: bool,
) -> Result<CodexDirectImagePreparedRequest, CodexImageError> {
    let model = serde_json::from_slice::<Value>(raw)
        .ok()
        .and_then(|value| {
            value
                .get("model")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .and_then(|model| codex_direct_image_model(&model).map(ToOwned::to_owned))
        .or_else(|| codex_direct_image_model(route_model).map(ToOwned::to_owned))
        .ok_or(CodexImageError::UnsupportedDirectModel)?;
    let endpoint_path = if request_path
        .trim_end_matches('/')
        .ends_with(CODEX_IMAGE_GENERATION_PATH)
    {
        CODEX_DIRECT_IMAGE_GENERATION_PATH
    } else if request_path
        .trim_end_matches('/')
        .ends_with(CODEX_IMAGE_EDIT_PATH)
    {
        CODEX_DIRECT_IMAGE_EDIT_PATH
    } else {
        return Err(CodexImageError::UnsupportedEndpoint);
    };
    let mut value = match serde_json::from_slice::<Value>(raw) {
        Ok(value) => value,
        Err(_) if endpoint_path == CODEX_DIRECT_IMAGE_EDIT_PATH => {
            multipart_edit_to_json(raw, content_type.ok_or(CodexImageError::InvalidMultipart)?)?
        }
        Err(_) => return Err(CodexImageError::InvalidJson),
    };
    let object = value.as_object_mut().ok_or(CodexImageError::InvalidJson)?;
    object.insert("model".to_owned(), Value::String(model.clone()));
    object.insert("stream".to_owned(), Value::Bool(stream));
    Ok(CodexDirectImagePreparedRequest {
        endpoint_path,
        model,
        content_type: "application/json".to_owned(),
        body: serde_json::to_vec(&value).map_err(|_| CodexImageError::InvalidJson)?,
        stream,
    })
}

fn multipart_edit_to_json(raw: &[u8], content_type: &str) -> Result<Value, CodexImageError> {
    if raw.len() > MAX_MULTIPART_IMAGE_BYTES {
        return Err(CodexImageError::MultipartTooLarge);
    }
    let mime = content_type
        .parse::<mime::Mime>()
        .map_err(|_| CodexImageError::InvalidMultipart)?;
    if mime.type_() != mime::MULTIPART {
        return Err(CodexImageError::InvalidMultipart);
    }
    let boundary = mime
        .get_param(mime::BOUNDARY)
        .map(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or(CodexImageError::InvalidMultipart)?;
    let delimiter = format!("--{boundary}").into_bytes();
    let mut object = Map::new();
    let mut images = Vec::new();
    for part in split_multipart(raw, &delimiter) {
        let (headers, data) = parse_multipart_part(part)?;
        let disposition = headers
            .get("content-disposition")
            .ok_or(CodexImageError::InvalidMultipart)?;
        let name =
            disposition_parameter(disposition, "name").ok_or(CodexImageError::InvalidMultipart)?;
        let filename = disposition_parameter(disposition, "filename");
        if filename.is_some() {
            let media_type = headers
                .get("content-type")
                .map(String::as_str)
                .unwrap_or_else(|| sniff_image_media_type(data));
            let data_url = format!("data:{media_type};base64,{}", STANDARD.encode(data));
            match name.as_str() {
                "image" | "image[]" => images.push(serde_json::json!({"image_url":data_url})),
                "mask" => {
                    object.insert("mask".to_owned(), serde_json::json!({"image_url":data_url}));
                }
                _ => {}
            }
            continue;
        }
        let text = std::str::from_utf8(data)
            .map_err(|_| CodexImageError::InvalidMultipart)?
            .trim();
        if text.is_empty() {
            continue;
        }
        let path = match name.as_str() {
            "mask[file_id]" => Some(("mask", "file_id")),
            "mask[image_url]" => Some(("mask", "image_url")),
            _ => None,
        };
        if let Some((root, child)) = path {
            let nested = object
                .entry(root.to_owned())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(nested) = nested.as_object_mut() {
                nested.insert(child.to_owned(), Value::String(text.to_owned()));
            }
        } else if matches!(name.as_str(), "image" | "image[]" | "images") {
            images.push(Value::String(text.to_owned()));
        } else if matches!(name.as_str(), "n" | "output_compression" | "partial_images") {
            if let Ok(number) = text.parse::<i64>() {
                object.insert(name, Value::from(number));
            }
        } else {
            object.insert(name, Value::String(text.to_owned()));
        }
    }
    if !images.is_empty() {
        object.insert("images".to_owned(), Value::Array(images));
    }
    Ok(Value::Object(object))
}

fn split_multipart<'a>(raw: &'a [u8], delimiter: &[u8]) -> Vec<&'a [u8]> {
    let mut parts = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = raw[cursor..]
        .windows(delimiter.len())
        .position(|window| window == delimiter)
    {
        let start = cursor + offset + delimiter.len();
        if raw.get(start..start + 2) == Some(b"--") {
            break;
        }
        let start = if raw.get(start..start + 2) == Some(b"\r\n") {
            start + 2
        } else {
            start
        };
        let Some(next) = raw[start..]
            .windows(delimiter.len())
            .position(|window| window == delimiter)
        else {
            break;
        };
        let mut end = start + next;
        if raw.get(end.saturating_sub(2)..end) == Some(b"\r\n") {
            end -= 2;
        }
        parts.push(&raw[start..end]);
        cursor = end;
    }
    parts
}

fn parse_multipart_part(
    part: &[u8],
) -> Result<(std::collections::BTreeMap<String, String>, &[u8]), CodexImageError> {
    let separator = part
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or(CodexImageError::InvalidMultipart)?;
    let headers = std::str::from_utf8(&part[..separator])
        .map_err(|_| CodexImageError::InvalidMultipart)?
        .split("\r\n")
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    Ok((headers, &part[separator + 4..]))
}

fn disposition_parameter(value: &str, name: &str) -> Option<String> {
    value.split(';').skip(1).find_map(|parameter| {
        let (key, value) = parameter.trim().split_once('=')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().trim_matches('"').to_owned())
    })
}

fn sniff_image_media_type(data: &[u8]) -> &'static str {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if data.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

pub fn prepare_codex_openai_image_request(
    raw: &[u8],
    route_model: &str,
    action: CodexImageAction,
) -> Result<CodexImagePreparedRequest, CodexImageError> {
    let value: Value = serde_json::from_slice(raw).map_err(|_| CodexImageError::InvalidJson)?;
    let object = value.as_object().ok_or(CodexImageError::InvalidJson)?;
    let prompt = object
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(CodexImageError::MissingPrompt)?;
    let response_format =
        CodexImageResponseFormat::parse(object.get("response_format").and_then(Value::as_str));
    let mut tool = Map::new();
    tool.insert(
        "type".to_owned(),
        Value::String("image_generation".to_owned()),
    );
    tool.insert(
        "action".to_owned(),
        Value::String(
            match action {
                CodexImageAction::Generate => "generate",
                CodexImageAction::Edit => "edit",
            }
            .to_owned(),
        ),
    );
    tool.insert(
        "model".to_owned(),
        Value::String(codex_openai_image_base_model(route_model).to_owned()),
    );
    for key in [
        "size",
        "quality",
        "background",
        "output_format",
        "moderation",
    ] {
        if let Some(value) = object.get(key).cloned() {
            tool.insert(key.to_owned(), value);
        }
    }
    let mut content = vec![serde_json::json!({"type":"input_text","text":prompt})];
    if action == CodexImageAction::Edit {
        let images = extract_image_inputs(object);
        if images.is_empty() {
            return Err(CodexImageError::MissingImage);
        }
        content.extend(
            images
                .into_iter()
                .map(|image_url| serde_json::json!({"type":"input_image","image_url":image_url})),
        );
    }
    let body = serde_json::json!({
        "model": codex_openai_image_base_model(route_model),
        "stream": true,
        "input": [{"role":"user","content":content}],
        "tools": [Value::Object(tool)],
        "tool_choice": {"type":"image_generation"}
    });
    Ok(CodexImagePreparedRequest {
        action,
        route_model: route_model.to_owned(),
        response_format,
        responses_body: serde_json::to_vec(&body).map_err(|_| CodexImageError::InvalidJson)?,
    })
}

pub fn extract_codex_image_results(
    completed: &[u8],
) -> Result<Vec<CodexImageResult>, CodexImageError> {
    let value: Value =
        serde_json::from_slice(completed).map_err(|_| CodexImageError::InvalidCompletion)?;
    let output = value
        .get("output")
        .and_then(Value::as_array)
        .ok_or(CodexImageError::InvalidCompletion)?;
    let mut results = Vec::new();
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("image_generation_call") {
            continue;
        }
        let data = item
            .get("result")
            .or_else(|| item.get("b64_json"))
            .and_then(Value::as_str)
            .map(strip_data_url)
            .filter(|value| !value.is_empty())
            .ok_or(CodexImageError::InvalidImageData)?;
        STANDARD
            .decode(data)
            .map_err(|_| CodexImageError::InvalidImageData)?;
        results.push(CodexImageResult {
            base64_data: data.to_owned(),
            output_format: item
                .get("output_format")
                .and_then(Value::as_str)
                .unwrap_or("png")
                .to_owned(),
            revised_prompt: item
                .get("revised_prompt")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            size: item
                .get("size")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            quality: item
                .get("quality")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        });
    }
    if results.is_empty() {
        Err(CodexImageError::InvalidCompletion)
    } else {
        Ok(results)
    }
}

pub fn build_codex_images_api_response(
    results: &[CodexImageResult],
    created_at: i64,
    response_format: CodexImageResponseFormat,
    usage: Option<Value>,
) -> Result<Vec<u8>, CodexImageError> {
    if results.is_empty() {
        return Err(CodexImageError::InvalidCompletion);
    }
    let data = results
        .iter()
        .map(|image| {
            let mut item = Map::new();
            match response_format {
                CodexImageResponseFormat::Base64Json => {
                    item.insert(
                        "b64_json".to_owned(),
                        Value::String(image.base64_data.clone()),
                    );
                }
                CodexImageResponseFormat::Url => {
                    item.insert(
                        "url".to_owned(),
                        Value::String(format!(
                            "data:{};base64,{}",
                            mime_type(&image.output_format),
                            image.base64_data
                        )),
                    );
                }
            }
            if let Some(prompt) = &image.revised_prompt {
                item.insert("revised_prompt".to_owned(), Value::String(prompt.clone()));
            }
            Value::Object(item)
        })
        .collect::<Vec<_>>();
    let mut response = Map::new();
    response.insert("created".to_owned(), Value::from(created_at));
    response.insert("data".to_owned(), Value::Array(data));
    if let Some(usage) = usage {
        response.insert("usage".to_owned(), usage);
    }
    serde_json::to_vec(&Value::Object(response)).map_err(|_| CodexImageError::InvalidCompletion)
}

pub fn build_codex_image_sse_frame(event: &str, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(event.len() + payload.len() + 16);
    frame.extend_from_slice(b"event: ");
    frame.extend_from_slice(event.as_bytes());
    frame.extend_from_slice(b"\ndata: ");
    frame.extend_from_slice(payload);
    frame.extend_from_slice(b"\n\n");
    frame
}

fn extract_image_inputs(object: &Map<String, Value>) -> Vec<String> {
    let mut values = Vec::new();
    for key in ["image", "images"] {
        match object.get(key) {
            Some(Value::String(value)) if !value.trim().is_empty() => values.push(value.clone()),
            Some(Value::Array(items)) => values.extend(
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .map(ToOwned::to_owned),
            ),
            _ => {}
        }
    }
    values
}

fn strip_data_url(value: &str) -> &str {
    value.split_once(",base64,").map_or(value, |(_, data)| data)
}

fn mime_type(format: &str) -> &'static str {
    match format.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_result_round_trip_preserve_openai_shape() {
        let request = prepare_codex_openai_image_request(
            br#"{"prompt":"draw","response_format":"url","size":"1024x1024"}"#,
            "gpt-image-2",
            CodexImageAction::Generate,
        )
        .unwrap();
        let body: Value = serde_json::from_slice(&request.responses_body).unwrap();
        assert_eq!(body["tools"][0]["type"], "image_generation");
        let encoded = STANDARD.encode(b"png");
        let completed = serde_json::to_vec(&serde_json::json!({"output":[{"type":"image_generation_call","result":encoded,"output_format":"png"}]})).unwrap();
        let results = extract_codex_image_results(&completed).unwrap();
        let response =
            build_codex_images_api_response(&results, 1, request.response_format, None).unwrap();
        let value: Value = serde_json::from_slice(&response).unwrap();
        assert!(value["data"][0]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn direct_json_and_multipart_edit_requests_are_normalized() {
        let direct = prepare_codex_direct_image_request(
            br#"{"model":"gpt-image-2","prompt":"draw"}"#,
            "gpt-image-2",
            CODEX_IMAGE_GENERATION_PATH,
            Some("application/json"),
            true,
        )
        .unwrap();
        assert_eq!(direct.endpoint_path, CODEX_DIRECT_IMAGE_GENERATION_PATH);
        assert!(direct.stream);
        assert_eq!(
            serde_json::from_slice::<Value>(&direct.body).unwrap()["stream"],
            true
        );

        let boundary = "ctox-boundary";
        let multipart = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n2\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\nPNG\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\nContent-Type: image/png\r\n\r\nMASK\r\n\
             --{boundary}--\r\n"
        );
        let edit = prepare_codex_direct_image_request(
            multipart.as_bytes(),
            "gpt-image-1.5",
            CODEX_IMAGE_EDIT_PATH,
            Some(&format!("multipart/form-data; boundary={boundary}")),
            false,
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&edit.body).unwrap();
        assert_eq!(edit.endpoint_path, CODEX_DIRECT_IMAGE_EDIT_PATH);
        assert_eq!(value["model"], "gpt-image-1.5");
        assert_eq!(value["n"], 2);
        assert!(value["images"][0]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        assert!(value["mask"]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }
}
