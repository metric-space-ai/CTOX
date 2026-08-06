// ref: internal/config/oauth_model_alias_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use super::config_normalization::{sanitize_oauth_model_alias, OAuthModelAlias};

#[test]
fn preserves_optional_fields_and_multiple_aliases_for_same_name() {
    let mut aliases = BTreeMap::from([(
        " CoDeX ".into(),
        vec![
            OAuthModelAlias {
                name: " gpt-5 ".into(),
                alias: " g5 ".into(),
                fork: true,
                display_name: " GPT Five ".into(),
                force_mapping: true,
            },
            OAuthModelAlias {
                name: "gpt-5".into(),
                alias: "g5-thinking".into(),
                fork: true,
                ..OAuthModelAlias::default()
            },
        ],
    )]);
    sanitize_oauth_model_alias(&mut aliases);
    let aliases = &aliases["codex"];
    assert_eq!(aliases.len(), 2);
    assert_eq!(aliases[0].name, "gpt-5");
    assert_eq!(aliases[0].alias, "g5");
    assert_eq!(aliases[0].display_name, "GPT Five");
    assert!(aliases[0].fork && aliases[0].force_mapping);
    assert_eq!(aliases[1].alias, "g5-thinking");
}
