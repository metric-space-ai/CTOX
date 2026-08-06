// Origin: CTOX supplemental tests for sdk/api/management.go.
// License: AGPL-3.0-only

use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde_json::Value;

use super::management::*;

#[derive(Default)]
struct RecordingHandler {
    calls: Mutex<Vec<ManagementEndpoint>>,
}

impl ManagementEndpointHandler for RecordingHandler {
    fn handle(
        &self,
        endpoint: ManagementEndpoint,
        _request: &ManagementRequest,
    ) -> ManagementResponse {
        self.calls.lock().unwrap().push(endpoint);
        ManagementResponse {
            status: 200,
            headers: Default::default(),
            body: endpoint.path().as_bytes().to_vec(),
        }
    }
}

#[derive(Default)]
struct Routes {
    entries: Vec<(&'static str, &'static str, ManagementEndpoint)>,
    fail_at: Option<usize>,
}

impl ManagementRouteRegistry for Routes {
    type Error = &'static str;

    fn register(
        &mut self,
        method: &'static str,
        path: &'static str,
        endpoint: ManagementEndpoint,
    ) -> Result<(), Self::Error> {
        if self.fail_at == Some(self.entries.len()) {
            return Err("registration failed");
        }
        self.entries.push((method, path, endpoint));
        Ok(())
    }
}

#[derive(Debug)]
struct FixedClock(Mutex<SystemTime>);

impl FixedClock {
    fn advance(&self, duration: Duration) {
        let mut now = self.0.lock().unwrap();
        *now += duration;
    }
}

impl OAuthClock for FixedClock {
    fn now(&self) -> SystemTime {
        *self.0.lock().unwrap()
    }
}

#[derive(Default)]
struct MemoryPersistence {
    writes: Mutex<Vec<(String, Vec<u8>)>>,
    fail: bool,
}

impl ManagementPersistence for MemoryPersistence {
    fn write_config(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        if self.fail {
            return Err(io::Error::other("injected failure"));
        }
        self.writes
            .lock()
            .unwrap()
            .push((path.display().to_string(), data.to_vec()));
        Ok(())
    }

    fn write_oauth_callback(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.write_config(path, data)
    }
}

#[test]
fn requester_delegates_exactly_the_six_limited_endpoints_in_call_order() {
    let backend = Arc::new(RecordingHandler::default());
    let requester = new_management_token_requester(new_handler_without_config_file_path(
        ManagementConfig::default(),
        backend.clone(),
    ));
    let request = ManagementRequest::default();
    requester.request_anthropic_token(&request);
    requester.request_codex_token(&request);
    requester.request_antigravity_token(&request);
    requester.request_kimi_token(&request);
    requester.get_auth_status(&request);
    requester.post_oauth_callback(&request);
    assert_eq!(*backend.calls.lock().unwrap(), MANAGEMENT_TOKEN_ENDPOINTS);
}

#[test]
fn route_registration_is_ordered_and_stops_at_first_error() {
    let handler = new_handler_without_config_file_path(
        ManagementConfig::default(),
        Arc::new(RecordingHandler::default()),
    );
    let mut routes = Routes::default();
    handler.register_token_routes(&mut routes).unwrap();
    assert_eq!(routes.entries.len(), 6);
    for (index, endpoint) in MANAGEMENT_TOKEN_ENDPOINTS.into_iter().enumerate() {
        assert_eq!(
            routes.entries[index],
            (endpoint.method(), endpoint.path(), endpoint)
        );
    }

    let mut routes = Routes {
        fail_at: Some(2),
        ..Routes::default()
    };
    assert_eq!(
        handler.register_token_routes(&mut routes),
        Err("registration failed")
    );
    assert_eq!(routes.entries.len(), 2);
}

#[test]
fn constructors_preserve_or_omit_config_path_without_side_effects() {
    let backend = Arc::new(RecordingHandler::default());
    let with_path = new_handler(
        ManagementConfig {
            auth_dir: "/auth".into(),
        },
        "/config/cliproxy.yaml",
        backend.clone(),
    );
    assert_eq!(with_path.config().auth_dir, Path::new("/auth"));
    assert_eq!(
        with_path.config_file_path(),
        Some(Path::new("/config/cliproxy.yaml"))
    );
    let without = new_handler_without_config_file_path(ManagementConfig::default(), backend);
    assert!(without.config_file_path().is_none());
}

#[test]
fn oauth_store_matches_registration_error_completion_and_expiry_semantics() {
    let clock = Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH)));
    let sessions = OAuthSessionStore::new(clock.clone(), Duration::from_secs(120));
    sessions.register(" ", "codex");
    assert!(sessions.get("").is_none());
    sessions.register("state-1", " OpenAI ");
    assert!(sessions.is_pending("state-1", "openai"));
    sessions.set_error("state-1", "  ");
    assert_eq!(
        sessions.get("state-1").unwrap().status,
        "Authentication failed"
    );
    assert!(!sessions.is_pending("state-1", "openai"));

    sessions.register("state-2", "codex");
    sessions.register("state-3", "CODEX");
    assert_eq!(sessions.complete_by_provider("codex"), 2);
    assert!(sessions.get("state-2").is_none());
    clock.advance(Duration::from_secs(61));
    assert!(sessions.get("state-3").is_none());

    sessions.register("expiring", "anthropic");
    clock.advance(Duration::from_secs(121));
    assert!(sessions.get("expiring").is_none());
}

#[test]
fn validation_normalization_and_pending_callback_write_cover_errors() {
    for state in ["", "../escape", "a/b", "a\\b", "state space"] {
        assert_eq!(
            validate_oauth_state(state),
            Err(OAuthManagementError::InvalidState)
        );
    }
    assert!(validate_oauth_state(&"a".repeat(128)).is_ok());
    assert_eq!(
        validate_oauth_state(&"a".repeat(129)),
        Err(OAuthManagementError::InvalidState)
    );
    assert_eq!(normalize_oauth_provider(" Claude "), Ok("anthropic"));
    assert_eq!(normalize_oauth_provider("openai"), Ok("codex"));
    assert_eq!(normalize_oauth_provider("anti-gravity"), Ok("antigravity"));
    assert_eq!(normalize_oauth_provider("grok"), Ok("xai"));
    assert_eq!(
        normalize_oauth_provider("kimi"),
        Err(OAuthManagementError::UnsupportedProvider)
    );

    let persistence = MemoryPersistence::default();
    let sessions = OAuthSessionStore::new(Arc::new(SystemOAuthClock), Duration::from_secs(60));
    assert_eq!(
        write_oauth_callback_file_for_pending_session(
            &persistence,
            &sessions,
            Path::new("/auth"),
            "codex",
            "missing",
            "code",
            ""
        ),
        Err(OAuthManagementError::SessionNotPending)
    );
    sessions.register("safe_state", "codex");
    let path = write_oauth_callback_file_for_pending_session(
        &persistence,
        &sessions,
        Path::new("/auth"),
        "openai",
        "safe_state",
        " code ",
        " err ",
    )
    .unwrap();
    assert_eq!(path, Path::new("/auth/.oauth-codex-safe_state.oauth"));
    let writes = persistence.writes.lock().unwrap();
    let payload: Value = serde_json::from_slice(&writes[0].1).unwrap();
    assert_eq!(
        payload,
        serde_json::json!({"code":"code","state":"safe_state","error":"err"})
    );
    drop(writes);

    let failing = MemoryPersistence {
        fail: true,
        ..Default::default()
    };
    assert_eq!(
        write_oauth_callback_file(&failing, Path::new("/auth"), "codex", "safe", "code", ""),
        Err(OAuthManagementError::Persistence)
    );

    let plugin_path = write_oauth_callback_file(
        &persistence,
        Path::new("/auth"),
        " Vendor-Plugin ",
        "plugin-state",
        "code",
        "",
    )
    .unwrap();
    assert_eq!(
        plugin_path,
        Path::new("/auth/.oauth-vendor-plugin-plugin-state.oauth")
    );
}

#[test]
fn config_write_and_auth_context_use_injected_values() {
    let persistence = MemoryPersistence::default();
    write_config(
        &persistence,
        Path::new("/config.yaml"),
        b"debug: false\n  # standalone\n  value: kept\n",
    )
    .unwrap();
    let writes = persistence.writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].1, b"debug: false\n# standalone\n  value: kept\n");
    drop(writes);

    let failing = MemoryPersistence {
        fail: true,
        ..Default::default()
    };
    assert!(write_config(&failing, Path::new("/config.yaml"), b"value: true\n").is_err());

    let request = ManagementRequest {
        query: [("state".into(), vec!["abc".into()])].into(),
        headers: [("x-request-id".into(), vec!["req-1".into()])].into(),
        body: b"ignored".to_vec(),
    };
    let context = populate_auth_context(&Default::default(), &request);
    let info = context.request_info().unwrap();
    assert_eq!(info.query.get("state").unwrap(), &["abc"]);
    assert_eq!(info.headers.get("x-request-id").unwrap(), &["req-1"]);
}
