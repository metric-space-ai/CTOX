// ref: internal/util/claude_tool_id.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};

const GEMINI_CLAUDE_TOOL_USE_ID_PREFIX: &str = "cpa_gemini_";
static CLAUDE_TOOL_USE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn sanitize_claude_tool_id(id: &str) -> String {
    let mut sanitized = id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let sequence = CLAUDE_TOOL_USE_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        sanitized = format!("toolu_{nanos}_{sequence}");
    }
    sanitized
}

pub fn gemini_claude_tool_use_id(call_id: &str, name: &str, args_raw: &str) -> String {
    let call_id = call_id.trim();
    let name = name.trim();
    if call_id.is_empty() || name.is_empty() {
        return String::new();
    }
    let args = if args_raw.trim().is_empty() {
        args_raw.to_owned()
    } else {
        serde_json::from_str::<Value>(args_raw)
            .ok()
            .and_then(|value| serde_json::to_string(&value).ok())
            .unwrap_or_else(|| args_raw.trim().to_owned())
    };
    let digest = Sha256::digest([call_id, name, &args].join("\0").as_bytes());
    format!(
        "{GEMINI_CLAUDE_TOOL_USE_ID_PREFIX}{}",
        digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

pub fn is_gemini_claude_tool_use_id(id: &str) -> bool {
    let Some(digest) = id.trim().strip_prefix(GEMINI_CLAUDE_TOOL_USE_ID_PREFIX) else {
        return false;
    };
    digest.len() == 32 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}
