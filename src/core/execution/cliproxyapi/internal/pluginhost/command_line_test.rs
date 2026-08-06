// ref: internal/pluginhost/command_line_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: flags, output, and auth persistence use typed injected authorities
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::sdk::pluginapi::{
    AuthData, CommandLineExecutionRequest, CommandLineExecutionResponse, CommandLineFlag,
    CommandLinePlugin, CommandLineRegistrationRequest, CommandLineRegistrationResponse,
    HostConfigSummary, Metadata, PluginFuture,
};

use super::command_line::{CommandLineAuthSink, CommandLinePluginRecord, CommandLineRegistry};

struct Plugin {
    flags: Vec<CommandLineFlag>,
    response: CommandLineExecutionResponse,
    received: Mutex<Vec<CommandLineExecutionRequest>>,
}

impl CommandLinePlugin for Plugin {
    fn register_command_line<'a>(
        &'a self,
        _request: CommandLineRegistrationRequest,
    ) -> PluginFuture<'a, CommandLineRegistrationResponse> {
        Box::pin(async move {
            Ok(CommandLineRegistrationResponse {
                flags: self.flags.clone(),
            })
        })
    }

    fn execute_command_line<'a>(
        &'a self,
        request: CommandLineExecutionRequest,
    ) -> PluginFuture<'a, CommandLineExecutionResponse> {
        Box::pin(async move {
            self.received
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request);
            Ok(self.response.clone())
        })
    }
}

#[derive(Default)]
struct Sink(Mutex<Vec<String>>);

impl CommandLineAuthSink for Sink {
    fn save(&self, auth: &AuthData) -> Result<String, String> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(auth.id.clone());
        Ok(format!("/auths/{}.json", auth.id))
    }
}

fn flag(name: &str, kind: &str, default_value: &str) -> CommandLineFlag {
    CommandLineFlag {
        name: name.to_owned(),
        kind: kind.to_owned(),
        default_value: default_value.to_owned(),
        ..CommandLineFlag::default()
    }
}

fn record(id: &str, priority: i32, plugin: Arc<Plugin>) -> CommandLinePluginRecord {
    CommandLinePluginRecord {
        plugin_id: id.to_owned(),
        priority,
        metadata: Metadata {
            name: id.to_owned(),
            ..Metadata::default()
        },
        plugin,
    }
}

#[tokio::test]
async fn registration_skips_native_and_higher_priority_plugin_wins() {
    let low = Arc::new(Plugin {
        flags: vec![flag("shared", "string", "low"), flag("port", "int", "1")],
        response: CommandLineExecutionResponse::default(),
        received: Mutex::new(Vec::new()),
    });
    let high = Arc::new(Plugin {
        flags: vec![flag("shared", "string", "high")],
        response: CommandLineExecutionResponse::default(),
        received: Mutex::new(Vec::new()),
    });
    let mut registry = CommandLineRegistry::default();
    let errors = registry
        .register(
            &[record("low", 1, low), record("high", 10, high)],
            &BTreeSet::from(["port".to_owned()]),
        )
        .await;
    assert_eq!(errors.len(), 1);
    registry.set("shared", "selected").unwrap();
    assert!(registry.has_triggered_flags());
}

#[tokio::test]
async fn execution_passes_all_args_and_only_triggered_plugin_flags() {
    let plugin = Arc::new(Plugin {
        flags: vec![flag("login", "bool", "false"), flag("tenant", "string", "")],
        response: CommandLineExecutionResponse {
            stdout: b"done".to_vec(),
            ..CommandLineExecutionResponse::default()
        },
        received: Mutex::new(Vec::new()),
    });
    let mut registry = CommandLineRegistry::default();
    registry
        .register(&[record("auth", 1, plugin.clone())], &BTreeSet::new())
        .await;
    registry.set("login", "true").unwrap();
    let outcome = registry
        .execute(
            "ctox".to_owned(),
            vec!["--login".to_owned(), "extra".to_owned()],
            "/config.yaml".to_owned(),
            HostConfigSummary::default(),
            BTreeMap::new(),
            &Sink::default(),
        )
        .await;
    assert!(outcome.handled);
    assert_eq!(outcome.stdout, b"done");
    let requests = plugin
        .received
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(requests[0].args, vec!["--login", "extra"]);
    assert!(requests[0].triggered_flags.contains_key("login"));
    assert!(!requests[0].triggered_flags.contains_key("tenant"));
    assert!(requests[0].flags.contains_key("tenant"));
}

#[tokio::test]
async fn successful_execution_persists_auths_and_reports_paths_in_collected_output() {
    let plugin = Arc::new(Plugin {
        flags: vec![flag("login", "bool", "false")],
        response: CommandLineExecutionResponse {
            stdout: b"authenticated".to_vec(),
            auths: vec![AuthData {
                id: "account-a".to_owned(),
                provider: "codex".to_owned(),
                ..AuthData::default()
            }],
            ..CommandLineExecutionResponse::default()
        },
        received: Mutex::new(Vec::new()),
    });
    let mut registry = CommandLineRegistry::default();
    registry
        .register(&[record("auth", 1, plugin)], &BTreeSet::new())
        .await;
    registry.set("login", "true").unwrap();
    let sink = Sink::default();
    let outcome = registry
        .execute(
            "ctox".to_owned(),
            Vec::new(),
            String::new(),
            HostConfigSummary::default(),
            BTreeMap::new(),
            &sink,
        )
        .await;
    assert_eq!(outcome.exit_code, 0);
    assert_eq!(outcome.saved_paths, vec!["/auths/account-a.json"]);
    assert!(String::from_utf8(outcome.stdout)
        .unwrap()
        .contains("Authentication saved to /auths/account-a.json"));
}
