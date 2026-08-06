// ref: sdk/cliproxy/auth/conductor_force_mapping_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::Response;

use super::force_mapping_live_fixtures_test::LIVE_FIXTURES;
use super::{
    finish_force_mapped_stream_chunks, rewrite_force_mapped_response,
    rewrite_force_mapped_stream_chunk, OAuthModelAliasResult, StreamRewriteOptions, StreamRewriter,
};

fn alias(target: &str) -> OAuthModelAliasResult {
    OAuthModelAliasResult {
        upstream_model: "upstream".into(),
        force_mapping: true,
        original_alias: target.into(),
    }
}

#[test]
fn live_provider_fixtures_rewrite_non_stream_responses() {
    for (target, upstream, fixture) in LIVE_FIXTURES {
        let mut response = Response {
            payload: fixture.as_bytes().to_vec(),
            ..Response::default()
        };
        rewrite_force_mapped_response(Some(&mut response), &alias(target));
        let output = String::from_utf8(response.payload).unwrap();
        assert!(output.contains(target), "{output}");
        if !target.contains(upstream) {
            assert!(!output.contains(upstream), "{output}");
        }
    }
}

#[test]
fn live_provider_fixtures_rewrite_fragmented_streams_once() {
    for (target, upstream, fixture) in LIVE_FIXTURES {
        let mut rewriter = StreamRewriter::new(StreamRewriteOptions {
            rewrite_model: (*target).into(),
        });
        let mut output =
            rewrite_force_mapped_stream_chunk(Some(&mut rewriter), b"event: response.created\n");
        output.extend(rewrite_force_mapped_stream_chunk(
            Some(&mut rewriter),
            format!("data: {fixture}\n\n").as_bytes(),
        ));
        output.extend(finish_force_mapped_stream_chunks(Some(&mut rewriter)));
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains(target), "{output}");
        if !target.contains(upstream) {
            assert!(!output.contains(upstream), "{output}");
        }
    }
}

#[test]
fn force_mapping_is_explicit_and_empty_alias_is_safe() {
    let original = br#"{"model":"upstream"}"#.to_vec();
    let mut response = Response {
        payload: original.clone(),
        ..Response::default()
    };
    rewrite_force_mapped_response(Some(&mut response), &OAuthModelAliasResult::default());
    assert_eq!(response.payload, original);
}
