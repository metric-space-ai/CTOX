// ref: sdk/cliproxy/auth/openai_compat_pool_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: alias-pool order/dedup and selected force-mapping result
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::config::CodexModel;

use super::{
    resolve_model_alias_pool_from_config_models, resolve_model_alias_result_from_config_models,
};

fn model(name: &str, alias: &str, force_mapping: bool) -> CodexModel {
    CodexModel {
        name: name.into(),
        alias: alias.into(),
        force_mapping,
        ..Default::default()
    }
}

#[test]
fn alias_pool_preserves_config_order_deduplicates_and_prefers_exact_suffix() {
    let models = [
        model("upstream-a", "public", false),
        model("UPSTREAM-A", "public", false),
        model("upstream-b", "public", false),
        model("exact(low)", "public(low)", false),
    ];
    assert_eq!(
        resolve_model_alias_pool_from_config_models("public(high)", &models),
        ["upstream-a(high)", "upstream-b(high)"]
    );
    assert_eq!(
        resolve_model_alias_pool_from_config_models("public(low)", &models),
        ["exact(low)"]
    );
}

#[test]
fn selected_pool_entry_carries_force_mapping_metadata() {
    let models = [model("upstream", "public", true)];
    let result = resolve_model_alias_result_from_config_models("public(max)", &models);
    assert_eq!(result.upstream_model, "upstream(max)");
    assert!(result.force_mapping);
    assert_eq!(result.original_alias, "public");
}
