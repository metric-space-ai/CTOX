// ref: sdk/api/handlers/openai/openai_images_handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct SseFrameAccumulator {
    buffer: Vec<u8>,
}

impl SseFrameAccumulator {
    pub fn add_chunk(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some(end) = self.buffer.windows(2).position(|window| window == b"\n\n") {
            frames.push(self.buffer.drain(..end + 2).collect());
        }
        frames
    }

    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        if self.buffer.is_empty() {
            Vec::new()
        } else {
            vec![std::mem::take(&mut self.buffer)]
        }
    }
}

#[must_use]
pub fn images_model_base(model: &str) -> String {
    model
        .trim()
        .split_once('/')
        .map_or_else(|| model.trim(), |(_, base)| base)
        .to_ascii_lowercase()
}

#[must_use]
pub fn is_supported_images_model(model: &str) -> bool {
    let base = images_model_base(model);
    base.starts_with("gpt-image-")
        || base.starts_with("grok-imagine-image")
        || base.starts_with("dall-e-")
}

#[must_use]
pub fn normalize_images_response_format(format: &str) -> &'static str {
    if format.trim().eq_ignore_ascii_case("url") {
        "url"
    } else {
        "b64_json"
    }
}

#[must_use]
pub fn build_openai_compat_images_json_request(
    raw_json: &[u8],
    image_model: &str,
    stream: bool,
) -> Vec<u8> {
    let Ok(Value::Object(mut document)) = serde_json::from_slice(raw_json) else {
        return raw_json.to_vec();
    };
    document.insert("model".to_owned(), Value::String(image_model.to_owned()));
    if stream {
        document.insert("stream".to_owned(), Value::Bool(true));
    } else {
        document.remove("stream");
    }
    serde_json::to_vec(&document).unwrap_or_else(|_| raw_json.to_vec())
}

#[cfg(test)]
#[path = "openai_images_handlers_test.rs"]
mod openai_images_handlers_test;
