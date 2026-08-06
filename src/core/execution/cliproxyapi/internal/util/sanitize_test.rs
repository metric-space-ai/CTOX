// ref: internal/util/sanitize_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::path::{Path, PathBuf};

use super::core::{
    count_auth_files, log_level_decision, resolve_auth_dir, sanitize_function_name, writable_path,
    AuthRecordStore, AuthStoreFailureKind, AuthStoreListError, AuthStoreListFuture, HostLogLevel,
    LogLevelDecision, ResolveAuthDirError, UtilityHostConfig,
};
use super::{
    deduplicate_function_declarations, disambiguated_tool_name_map, restore_sanitized_tool_name,
    sanitized_function_name_map,
};

#[test]
fn sanitize_function_name_matches_all_pinned_upstream_cases() {
    let cases = [
        ("valid_name", "valid_name"),
        ("name.with.dots", "name.with.dots"),
        ("name:with:colons", "name:with:colons"),
        ("name-with-dashes", "name-with-dashes"),
        (
            "name.with_dots:colons-dashes",
            "name.with_dots:colons-dashes",
        ),
        ("name!with@invalid#chars", "name_with_invalid_chars"),
        ("name with spaces", "name_with_spaces"),
        ("name_with_你好_chars", "name_with____chars"),
        ("123name", "_123name"),
        (".name", "_.name"),
        (":name", "_:name"),
        ("-name", "_-name"),
        ("!name", "_name"),
        (
            "this_is_a_very_long_name_that_exactly_reaches_sixty_four_charact",
            "this_is_a_very_long_name_that_exactly_reaches_sixty_four_charact",
        ),
        (
            "this_is_a_very_long_name_that_exactly_reaches_sixty_four_charactX",
            "this_is_a_very_long_name_that_exactly_reaches_sixty_four_charact",
        ),
        (
            "this_is_a_very_long_name_that_exceeds_the_sixty_four_character_limit_for_function_names",
            "this_is_a_very_long_name_that_exceeds_the_sixty_four_character_l",
        ),
        (
            "1234567890123456789012345678901234567890123456789012345678901234",
            "_123456789012345678901234567890123456789012345678901234567890123",
        ),
        (
            "!234567890123456789012345678901234567890123456789012345678901234",
            "_234567890123456789012345678901234567890123456789012345678901234",
        ),
        ("", ""),
        ("@", "_"),
        ("a", "a"),
        ("1", "_1"),
        ("_", "_"),
    ];

    for (input, expected) in cases {
        let actual = sanitize_function_name(input);
        assert_eq!(actual, expected, "input {input:?}");
        assert!(actual.len() <= 64, "input {input:?}");
        assert!(
            actual.is_empty() || {
                let first = actual.as_bytes()[0];
                first.is_ascii_alphabetic() || first == b'_'
            }
        );
    }
}

#[test]
fn sanitizer_replaces_each_unicode_scalar_before_ascii_byte_truncation() {
    let input = format!("a{}{}", "你".repeat(70), "z".repeat(70));
    let actual = sanitize_function_name(&input);
    assert_eq!(actual.len(), 64);
    assert!(actual.is_ascii());
    assert_eq!(actual, format!("a{}", "_".repeat(63)));
}

#[test]
fn supported_tool_shapes_are_sanitized_and_collisions_are_stable() {
    let first = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build";
    let second = "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs";
    let raw = format!(
        r#"{{"tools":[
          {{"name":"{first}"}},{{"name":"{first}"}},{{"name":"{second}"}},
          {{"type":"function","function":{{"name":"nested/name"}}}},
          {{"functionDeclarations":[{{"name":"camel@name"}}],
            "function_declarations":[{{"name":"snake name"}}]}}
        ]}}"#
    );
    let forward = sanitized_function_name_map(raw.as_bytes());
    assert_ne!(forward[first], forward[second]);
    assert!(forward[first].len() <= 64);
    assert!(forward[second].len() <= 64);
    assert_eq!(forward["nested/name"], "nested_name");
    assert_eq!(forward["camel@name"], "camel_name");
    assert_eq!(forward["snake name"], "snake_name");

    let reversed = format!(r#"{{"tools":[{{"name":"{second}"}},{{"name":"{first}"}}]}}"#);
    let reversed_forward = sanitized_function_name_map(reversed.as_bytes());
    assert_eq!(reversed_forward[first], forward[first]);
    assert_eq!(reversed_forward[second], forward[second]);

    let reverse = disambiguated_tool_name_map(raw.as_bytes());
    assert_eq!(reverse[&forward[first]], first);
    assert_eq!(reverse[&forward[second]], second);
}

#[test]
fn missing_tools_deduplication_and_restore_follow_upstream_contracts() {
    assert!(sanitized_function_name_map(b"{}").is_empty());
    assert!(sanitized_function_name_map(&[]).is_empty());

    let deduped = deduplicate_function_declarations(
        br#"[
          {"name":"lookup","description":"first"},
          {"name":"other"},
          {"name":"lookup","description":"second"}
        ]"#,
    );
    let declarations: serde_json::Value = serde_json::from_slice(&deduped).unwrap();
    assert_eq!(declarations.as_array().unwrap().len(), 2);
    assert_eq!(declarations[0]["description"], "first");
    assert_eq!(declarations[1]["name"], "other");

    let names = std::collections::HashMap::from([
        ("mcp_server_read".to_owned(), "mcp/server/read".to_owned()),
        ("tool_v2".to_owned(), "tool@v2".to_owned()),
    ]);
    assert_eq!(
        restore_sanitized_tool_name(&names, "mcp_server_read"),
        "mcp/server/read"
    );
    assert_eq!(restore_sanitized_tool_name(&names, "unknown"), "unknown");
    assert_eq!(
        restore_sanitized_tool_name(&std::collections::HashMap::new(), "name"),
        "name"
    );
    assert_eq!(restore_sanitized_tool_name(&names, ""), "");
}

#[test]
fn log_level_is_a_host_decision_without_global_mutation() {
    let debug = UtilityHostConfig::new(true, None);
    assert_eq!(
        log_level_decision(&debug, HostLogLevel::Info),
        LogLevelDecision::Change {
            from: HostLogLevel::Info,
            to: HostLogLevel::Debug,
            debug: true,
        }
    );
    assert_eq!(
        log_level_decision(&debug, HostLogLevel::Debug),
        LogLevelDecision::Keep {
            level: HostLogLevel::Debug
        }
    );

    let normal = UtilityHostConfig::new(false, None);
    assert_eq!(
        log_level_decision(&normal, HostLogLevel::Warn),
        LogLevelDecision::Change {
            from: HostLogLevel::Warn,
            to: HostLogLevel::Info,
            debug: false,
        }
    );
}

#[test]
fn auth_dir_uses_only_explicit_default_and_home_inputs() {
    let home = Path::new("/srv/operator");
    assert_eq!(
        resolve_auth_dir("", Some(home), "~/.cli-proxy-api").unwrap(),
        PathBuf::from("/srv/operator/.cli-proxy-api")
    );
    assert_eq!(
        resolve_auth_dir("~", Some(home), "ignored").unwrap(),
        PathBuf::from("/srv/operator")
    );
    assert_eq!(
        resolve_auth_dir("~/auth\\nested/../records", Some(home), "ignored").unwrap(),
        PathBuf::from("/srv/operator/auth/records")
    );
    assert_eq!(
        resolve_auth_dir("./runtime/../auth", None, "ignored").unwrap(),
        PathBuf::from("auth")
    );
    assert_eq!(
        resolve_auth_dir("~/.auth", None, "ignored"),
        Err(ResolveAuthDirError::HomeDirectoryUnavailable)
    );
}

#[derive(Clone, Copy)]
enum StoreResult {
    Records(usize),
    Failure(AuthStoreListError),
}

struct FakeStore(StoreResult);

impl AuthRecordStore for FakeStore {
    type Record = ();

    fn list(&self) -> AuthStoreListFuture<'_, Self::Record> {
        Box::pin(async move {
            match self.0 {
                StoreResult::Records(count) => Ok(vec![(); count]),
                StoreResult::Failure(error) => Err(error),
            }
        })
    }
}

#[tokio::test]
async fn auth_count_uses_injected_async_store_and_redacts_failures() {
    assert_eq!(count_auth_files::<FakeStore>(None).await, Ok(0));
    assert_eq!(
        count_auth_files(Some(&FakeStore(StoreResult::Records(3)))).await,
        Ok(3)
    );

    let error = count_auth_files(Some(&FakeStore(StoreResult::Failure(
        AuthStoreListError::new(AuthStoreFailureKind::PermissionDenied),
    ))))
    .await
    .unwrap_err();
    assert_eq!(error.kind(), AuthStoreFailureKind::PermissionDenied);
    assert_eq!(
        error.to_string(),
        "auth record store list failed (permission denied)"
    );
    assert!(!error.to_string().contains('/'));
}

#[test]
fn writable_path_comes_from_typed_config_and_is_cleaned() {
    let config = UtilityHostConfig::new(true, Some("  /srv/runtime/../writable  "));
    assert_eq!(writable_path(&config), Some(PathBuf::from("/srv/writable")));
    assert_eq!(config.writable_path(), Some(Path::new("/srv/writable")));
    assert_eq!(writable_path(&UtilityHostConfig::new(false, None)), None);
    assert_eq!(
        writable_path(&UtilityHostConfig::new(false, Some("   "))),
        None
    );
}
