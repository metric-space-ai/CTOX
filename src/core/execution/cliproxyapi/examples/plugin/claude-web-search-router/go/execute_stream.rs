// ref: examples/plugin/claude-web-search-router/go/execute_stream.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only
pub fn start(
    stream_id: &str,
    runner: impl FnOnce(&str) -> Result<(), String>,
) -> Result<&'static str, String> {
    if stream_id.trim().is_empty() {
        return Err("stream_id is required for executor.execute_stream".into());
    }
    runner(stream_id)?;
    Ok("text/event-stream")
}
