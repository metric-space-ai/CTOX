// ref: sdk/api/handlers/openai/openai_videos_handlers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct VideoBinding {
    auth_id: String,
    model: String,
    expires_at: Instant,
}

#[derive(Debug, Default)]
pub struct VideoAuthBindingStore(Mutex<BTreeMap<String, VideoBinding>>);

impl VideoAuthBindingStore {
    pub fn set(&self, video_id: &str, auth_id: &str, model: &str, ttl: Duration) {
        if video_id.trim().is_empty() || auth_id.trim().is_empty() || ttl.is_zero() {
            return;
        }
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                video_id.trim().to_owned(),
                VideoBinding {
                    auth_id: auth_id.trim().to_owned(),
                    model: model.trim().to_owned(),
                    expires_at: Instant::now() + ttl,
                },
            );
    }

    pub fn get(&self, video_id: &str) -> Option<(String, String)> {
        let mut bindings = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let binding = bindings.get(video_id.trim())?.clone();
        if Instant::now() >= binding.expires_at {
            bindings.remove(video_id.trim());
            return None;
        }
        Some((binding.auth_id, binding.model))
    }
}

#[must_use]
pub fn is_supported_videos_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    let base = model
        .split_once('/')
        .map_or(model.as_str(), |(_, base)| base);
    base.starts_with("grok-imagine-video") || base.starts_with("sora-")
}

pub fn build_xai_videos_create_request(raw_json: &[u8], model: &str) -> Result<Vec<u8>, String> {
    let Value::Object(mut document) =
        serde_json::from_slice(raw_json).map_err(|_| "invalid JSON body")?
    else {
        return Err("request body must be an object".to_owned());
    };
    if !is_supported_videos_model(model) {
        return Err("unsupported video model".to_owned());
    }
    let prompt = document
        .remove("prompt")
        .unwrap_or(Value::String(String::new()));
    let seconds = document.remove("seconds").unwrap_or(json!(5));
    Ok(serde_json::to_vec(&json!({
        "model": "grok-imagine-video",
        "prompt": prompt,
        "duration": seconds
    }))
    .unwrap_or_default())
}

#[must_use]
pub fn openai_video_status(status: &str) -> &'static str {
    match status.trim().to_ascii_lowercase().as_str() {
        "completed" | "succeeded" | "done" => "completed",
        "failed" | "error" => "failed",
        "queued" | "pending" => "queued",
        _ => "in_progress",
    }
}

#[cfg(test)]
#[path = "openai_videos_handlers_test.rs"]
mod openai_videos_handlers_test;
