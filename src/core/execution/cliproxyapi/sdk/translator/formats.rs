// ref: sdk/translator/formats.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::Format;

pub fn openai() -> Format {
    Format::from("openai")
}
pub fn openai_response() -> Format {
    Format::from("openai-response")
}
pub fn claude() -> Format {
    Format::from("claude")
}
pub fn gemini() -> Format {
    Format::from("gemini")
}
pub fn codex() -> Format {
    Format::from("codex")
}
pub fn antigravity() -> Format {
    Format::from("antigravity")
}
pub fn interactions() -> Format {
    Format::from("interactions")
}
