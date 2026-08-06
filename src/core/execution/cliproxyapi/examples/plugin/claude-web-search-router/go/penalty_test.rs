// ref: examples/plugin/claude-web-search-router/go/penalty_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::{penalty::Penalties, Backend};
#[test]
fn failures_are_deprioritized_and_success_decays() {
    let mut p = Penalties::default();
    p.failure(Backend::CodexWebSearch);
    assert_eq!(
        p.sorted(&[Backend::CodexWebSearch, Backend::Tavily]),
        vec![Backend::Tavily, Backend::CodexWebSearch]
    );
    for _ in 0..5 {
        p.success(Backend::CodexWebSearch)
    }
    assert_eq!(
        p.sorted(&[Backend::CodexWebSearch, Backend::Tavily])[0],
        Backend::CodexWebSearch
    );
}
