// ref: internal/runtime/executor/helps/codex_multi_agent_v2.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::Headers;
use crate::sdk::translator::Format;

/// Canonical Codex multi-agent-v2 payload processor supplied by the host/client
/// layer. Eligibility, model catalogs, translation, and JSON mutation stay in
/// that single owner; these executor helpers are deliberately byte-transparent.
pub trait CodexMultiAgentV2Processor {
    fn rewrite_spawn_agent_description(&self, headers: &Headers, payload: &[u8]) -> Vec<u8>;

    fn rewrite_input(&self, headers: &Headers, payload: &[u8]) -> Vec<u8>;

    fn translate_request(
        &self,
        headers: &Headers,
        from: &Format,
        to: &Format,
        model: &str,
        payload: &[u8],
        stream: bool,
    ) -> Vec<u8>;

    fn optimize_request(&self, headers: &Headers, payload: &[u8]) -> (Vec<u8>, bool);

    fn restore_response(&self, payload: &[u8], optimized: bool) -> Vec<u8>;
}

#[must_use]
pub fn rewrite_codex_spawn_agent_description<Processor>(
    processor: &Processor,
    headers: &Headers,
    payload: &[u8],
) -> Vec<u8>
where
    Processor: CodexMultiAgentV2Processor + ?Sized,
{
    processor.rewrite_spawn_agent_description(headers, payload)
}

#[must_use]
pub fn rewrite_codex_multi_agent_v2_input<Processor>(
    processor: &Processor,
    headers: &Headers,
    payload: &[u8],
) -> Vec<u8>
where
    Processor: CodexMultiAgentV2Processor + ?Sized,
{
    processor.rewrite_input(headers, payload)
}

#[must_use]
pub fn translate_request_with_codex_multi_agent_v2<Processor>(
    processor: &Processor,
    headers: &Headers,
    from: &Format,
    to: &Format,
    model: &str,
    payload: &[u8],
    stream: bool,
) -> Vec<u8>
where
    Processor: CodexMultiAgentV2Processor + ?Sized,
{
    processor.translate_request(headers, from, to, model, payload, stream)
}

#[must_use]
pub fn optimize_codex_multi_agent_v2_request<Processor>(
    processor: &Processor,
    headers: &Headers,
    payload: &[u8],
) -> (Vec<u8>, bool)
where
    Processor: CodexMultiAgentV2Processor + ?Sized,
{
    processor.optimize_request(headers, payload)
}

#[must_use]
pub fn restore_codex_multi_agent_v2_response<Processor>(
    processor: &Processor,
    payload: &[u8],
    optimized: bool,
) -> Vec<u8>
where
    Processor: CodexMultiAgentV2Processor + ?Sized,
{
    processor.restore_response(payload, optimized)
}
