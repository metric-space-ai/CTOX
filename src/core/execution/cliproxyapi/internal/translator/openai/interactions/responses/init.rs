// ref: internal/translator/openai/interactions/responses/init.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    convert_interactions_request_to_openai_responses,
    convert_interactions_response_to_openai_responses_non_stream,
    convert_interactions_response_to_openai_responses_stream,
    convert_openai_responses_request_to_interactions,
    convert_openai_responses_response_to_interactions_non_stream,
    convert_openai_responses_response_to_interactions_stream,
};
use crate::sdk::translator::{interactions, openai_response, Registry, ResponseTransform};
use std::sync::Arc;

/// Replaces the two upstream Go package `init()` registrations explicitly.
/// Both stateful directions keep their accumulators inside request-local
/// `TranslationState` values.
pub fn register_openai_responses_interactions(registry: &Registry) {
    registry.register_pair(
        openai_response(),
        interactions(),
        Arc::new(convert_openai_responses_request_to_interactions),
        ResponseTransform {
            stream: Some(Arc::new(
                convert_interactions_response_to_openai_responses_stream,
            )),
            non_stream: Some(Arc::new(
                convert_interactions_response_to_openai_responses_non_stream,
            )),
            token_count: None,
        },
    );
    registry.register_pair(
        interactions(),
        openai_response(),
        Arc::new(convert_interactions_request_to_openai_responses),
        ResponseTransform {
            stream: Some(Arc::new(
                convert_openai_responses_response_to_interactions_stream,
            )),
            non_stream: Some(Arc::new(
                convert_openai_responses_response_to_interactions_non_stream,
            )),
            token_count: None,
        },
    );
}
