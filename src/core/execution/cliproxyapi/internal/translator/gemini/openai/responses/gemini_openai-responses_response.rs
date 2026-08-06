// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::antigravity::openai::responses::{
    convert_antigravity_response_to_openai_responses_non_stream_with_state,
    convert_antigravity_response_to_openai_responses_stream, AntigravityToResponsesState,
};
use sha2::{Digest, Sha256};

/// Request-local Gemini stream state. Unlike pinned Go's package counters and
/// wall clock, the default path derives a stable opaque identity from the
/// request/first payload. Hosts that own an ID/clock authority can inject both
/// through `with_identity`.
#[derive(Clone, Debug, Default)]
pub struct GeminiToResponsesState {
    inner: Option<AntigravityToResponsesState>,
}

impl GeminiToResponsesState {
    #[must_use]
    pub fn with_identity(response_id: impl Into<String>, created_at: i64) -> Self {
        Self {
            inner: Some(AntigravityToResponsesState::with_identity(
                response_id,
                created_at,
            )),
        }
    }

    fn inner<'a>(
        &'a mut self,
        original_request: &[u8],
        request: &[u8],
        raw_json: &[u8],
    ) -> &'a mut AntigravityToResponsesState {
        self.inner.get_or_insert_with(|| {
            AntigravityToResponsesState::with_identity(
                synthesized_request_identity(original_request, request, raw_json),
                0,
            )
        })
    }
}

/// Converts a complete native Gemini/Vertex response into an OpenAI Responses
/// object. The Antigravity adapter owns the shared implementation of the same
/// upstream Gemini part/output algorithm; this facade deliberately reuses it.
pub fn convert_gemini_response_to_openai_responses_non_stream(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
) -> Vec<u8> {
    let mut state = GeminiToResponsesState::default();
    convert_gemini_response_to_openai_responses_non_stream_with_state(
        original_request,
        request,
        raw_json,
        &mut state,
    )
}

/// Non-stream converter with explicitly injected request identity/clock state.
pub fn convert_gemini_response_to_openai_responses_non_stream_with_state(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    state: &mut GeminiToResponsesState,
) -> Vec<u8> {
    convert_antigravity_response_to_openai_responses_non_stream_with_state(
        original_request,
        request,
        raw_json,
        state.inner(original_request, request, raw_json),
    )
}

/// Converts one native Gemini/Vertex stream payload into Responses SSE events
/// while retaining all per-request aggregation state in the caller-owned value.
pub fn convert_gemini_response_to_openai_responses_stream(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    state: &mut GeminiToResponsesState,
) -> Vec<Vec<u8>> {
    convert_antigravity_response_to_openai_responses_stream(
        original_request,
        request,
        raw_json,
        state.inner(original_request, request, raw_json),
    )
}

fn synthesized_request_identity(
    original_request: &[u8],
    request: &[u8],
    first_payload: &[u8],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ctox-gemini-responses-v1\0");
    for value in [original_request, request, first_payload] {
        digest.update((value.len() as u64).to_be_bytes());
        digest.update(value);
    }
    let digest = digest.finalize();
    let suffix = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("resp_{suffix}")
}
