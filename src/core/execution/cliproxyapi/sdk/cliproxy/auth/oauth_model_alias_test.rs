// ref: sdk/cliproxy/auth/oauth_model_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: per-auth precedence, channel isolation, suffix and force-mapping behavior
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use crate::internal::config::config_normalization::OAuthModelAlias;

use super::{
    model_alias_channel, oauth_model_alias_channel, oauth_model_aliases_from_attributes,
    resolve_upstream_model_from_aliases, set_oauth_model_aliases_attribute, Auth,
    OAuthModelAliasTable,
};

pub(super) fn alias(name: &str, alias: &str, force_mapping: bool) -> OAuthModelAlias {
    OAuthModelAlias {
        name: name.into(),
        alias: alias.into(),
        force_mapping,
        ..Default::default()
    }
}

fn oauth_auth(provider: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = "oauth".into();
    auth.provider = provider.into();
    auth.attributes.insert("auth_kind".into(), "oauth".into());
    auth
}

#[test]
fn suffix_preservation_and_exact_suffixed_alias_priority_match_upstream() {
    let aliases = [
        alias("gemini-pro", "public", false),
        alias("gemini-flash(low)", "public(low)", false),
    ];
    let plain = resolve_upstream_model_from_aliases(&aliases, "public(high)");
    assert_eq!(plain.upstream_model, "gemini-pro(high)");
    assert_eq!(plain.original_alias, "public(high)");
    let exact = resolve_upstream_model_from_aliases(&aliases, "public(low)");
    assert_eq!(exact.upstream_model, "gemini-flash(low)");
}

#[test]
fn force_mapping_uses_config_alias_without_request_suffix() {
    let result = resolve_upstream_model_from_aliases(
        &[alias("glm-5", "claude-sonnet", true)],
        "claude-sonnet(high)",
    );
    assert_eq!(result.upstream_model, "glm-5(high)");
    assert!(result.force_mapping);
    assert_eq!(result.original_alias, "claude-sonnet");
}

#[test]
fn per_auth_aliases_are_sanitized_and_override_compiled_channel_table() {
    let table = OAuthModelAliasTable::compile(&BTreeMap::from([(
        "codex".into(),
        vec![alias("global", "public", false)],
    )]));
    let mut auth = oauth_auth("codex");
    set_oauth_model_aliases_attribute(
        &mut auth,
        &[
            alias(" local ", " public ", true),
            alias("duplicate", "PUBLIC", false),
        ],
    );
    assert_eq!(
        oauth_model_aliases_from_attributes(&auth.attributes).len(),
        1
    );
    let result = table.resolve(&auth, "public(max)");
    assert_eq!(result.upstream_model, "local(max)");
    assert!(result.force_mapping);
}

#[test]
fn channels_reject_api_keys_and_native_gemini_but_allow_plugins() {
    assert_eq!(oauth_model_alias_channel("codex", "api_key"), "");
    assert_eq!(oauth_model_alias_channel("gemini", "oauth"), "");
    assert_eq!(
        oauth_model_alias_channel(" MyPlugin ", "oauth2"),
        "myplugin"
    );
    assert_eq!(model_alias_channel(&oauth_auth("kimi")), "kimi");
}
