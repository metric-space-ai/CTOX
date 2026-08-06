// ref: sdk/cliproxy/auth/home_force_mapping_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: force-mapped retained selections preserve the requested alias
// License: MIT (upstream); modifications AGPL-3.0-only

#[test]
fn force_mapping_is_only_applied_to_retained_selection() {
    let (selection, _registry) = super::home_selection_test::selection();
    let fresh = selection.clone_auth_for_route("alias(high)");
    assert!(!fresh.attributes.contains_key("home_original_alias"));
    selection.retain();
    let retained = selection.clone_auth_for_route("alias(high)");
    assert_eq!(retained.attributes["home_original_alias"], "alias(high)");
    assert_eq!(retained.attributes["home_upstream_model"], "gpt-5(high)");
    selection.end("done");
}
