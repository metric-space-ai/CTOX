// ref: sdk/cliproxy/auth/conductor_models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::Response;

// Credential-bound candidate/capability resolution from this upstream file is
// implemented by `api_key_model_capabilities`; this module owns the response
// side of the same manager generation.

use super::{
    rewrite_model_in_response, rewrite_sse_payload_lines, OAuthModelAliasResult, StreamRewriter,
};

pub fn rewrite_force_mapped_response(
    response: Option<&mut Response>,
    alias: &OAuthModelAliasResult,
) {
    let Some(response) = response else {
        return;
    };
    if !alias.force_mapping || alias.original_alias.trim().is_empty() {
        return;
    }
    response.payload = rewrite_model_in_response(&response.payload, alias.original_alias.trim());
}

#[must_use]
pub fn rewrite_force_mapped_stream_chunk(
    rewriter: Option<&mut StreamRewriter>,
    payload: &[u8],
) -> Vec<u8> {
    let Some(rewriter) = rewriter else {
        return payload.to_vec();
    };
    if payload.is_empty() {
        return Vec::new();
    }
    let rewritten = rewriter.rewrite_chunk(payload);
    if !rewritten.is_empty() {
        return rewritten;
    }
    if payload.windows(5).any(|window| window == b"data:") {
        let line_wise = rewrite_sse_payload_lines(payload, rewriter.rewrite_model());
        if !line_wise.is_empty() {
            return line_wise;
        }
    }
    Vec::new()
}

#[must_use]
pub fn finish_force_mapped_stream_chunks(rewriter: Option<&mut StreamRewriter>) -> Vec<u8> {
    rewriter.map_or_else(Vec::new, StreamRewriter::finish)
}
