// ref: examples/plugin/claude-web-search-router/go/stream_forward.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
pub fn looks_like_openai_responses_sse(payload: &[u8]) -> bool {
    let s = String::from_utf8_lossy(payload);
    !s.contains("event: message_start")
        && (s.contains("event: response.")
            || s.contains("\"type\":\"response.")
            || s.contains("\"type\": \"response."))
}
pub trait StreamSink {
    fn emit(&mut self, payload: &[u8]) -> Result<(), String>;
    fn close(&mut self, error: Option<&str>);
}
pub fn forward(
    stream_id: &str,
    chunks: &[Vec<u8>],
    sink: &mut dyn StreamSink,
) -> Result<(), String> {
    if stream_id.trim().is_empty() {
        return Err("plugin stream id is required".into());
    }
    for (index, chunk) in chunks.iter().enumerate() {
        if index == 0 && looks_like_openai_responses_sse(chunk) {
            return Err(
                "host model stream returned OpenAI Responses SSE instead of Claude Messages SSE"
                    .into(),
            );
        }
        sink.emit(chunk)?
    }
    sink.close(None);
    Ok(())
}
