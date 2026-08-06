// Origin: CTOX supplemental tests for sdk/api/options.go.
// License: AGPL-3.0-only

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::internal::logging::request_logger::{RequestLogRecord, RequestLogger};

use super::options::*;

#[derive(Debug)]
struct NamedMiddleware(&'static str);

impl ServerMiddleware for NamedMiddleware {
    fn name(&self) -> &str {
        self.0
    }
}

#[derive(Debug)]
struct NullLogger;

impl RequestLogger for NullLogger {
    fn is_enabled(&self) -> bool {
        false
    }

    fn log_request(&self, _record: &RequestLogRecord, _force: bool) -> io::Result<Option<PathBuf>> {
        Ok(None)
    }
}

#[test]
fn options_apply_in_order_and_append_middleware() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let first = Arc::clone(&observed);
    let second = Arc::clone(&observed);
    let config = apply_server_options(vec![
        with_middleware(vec![Arc::new(NamedMiddleware("one"))]),
        with_engine_configurator(Some(Arc::new(move |state| {
            first.lock().unwrap().push("first");
            state.attributes.push(("owner".into(), "first".into()));
        }))),
        with_middleware(vec![Arc::new(NamedMiddleware("two"))]),
        with_engine_configurator(Some(Arc::new(move |state| {
            second.lock().unwrap().push("second");
            state.attributes.push(("owner".into(), "second".into()));
        }))),
    ]);

    assert_eq!(
        config
            .extra_middleware()
            .iter()
            .map(|middleware| middleware.name())
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
    let mut state = EngineBuildState::default();
    config.engine_configurator().unwrap()(&mut state);
    assert_eq!(*observed.lock().unwrap(), ["second"]);
    assert_eq!(state.attributes, [("owner".into(), "second".into())]);
}

#[test]
fn nil_callbacks_and_invalid_keep_alive_are_no_ops() {
    let config = apply_server_options(vec![
        with_engine_configurator(None),
        with_router_configurator(None),
        with_request_logger_factory(None),
        with_keep_alive_endpoint(Duration::ZERO, Some(Arc::new(|| {}))),
        with_keep_alive_endpoint(Duration::from_secs(2), None),
        with_local_management_password("  "),
    ]);
    assert!(config.engine_configurator().is_none());
    assert!(config.router_configurator().is_none());
    assert!(config.request_logger_factory().is_none());
    assert!(config.keep_alive().is_none());
    assert!(config.local_management_password().is_none());
}

#[test]
fn router_keep_alive_password_and_logger_factory_are_typed_and_invocable() {
    let timeout_count = Arc::new(Mutex::new(0));
    let count = Arc::clone(&timeout_count);
    let config = apply_server_options(vec![
        with_local_management_password("  secret  "),
        with_keep_alive_endpoint(
            Duration::from_secs(7),
            Some(Arc::new(move || *count.lock().unwrap() += 1)),
        ),
        with_router_configurator(Some(Arc::new(|routes, handler, _config| {
            routes.register(format!("/{}", handler.name));
        }))),
        with_request_logger_factory(Some(Arc::new(|_| Arc::new(NullLogger)))),
    ]);

    assert_eq!(config.local_management_password(), Some("secret"));
    let keep_alive = config.keep_alive().unwrap();
    assert_eq!(keep_alive.timeout, Duration::from_secs(7));
    (keep_alive.on_timeout)();
    assert_eq!(*timeout_count.lock().unwrap(), 1);

    let runtime: crate::internal::config::CliproxyRuntimeConfig =
        serde_json::from_str("{}").unwrap();
    let mut routes = RouteRegistry::default();
    config.router_configurator().unwrap()(
        &mut routes,
        &BaseApiHandler {
            name: "custom".into(),
        },
        &runtime,
    );
    assert_eq!(routes.routes(), ["/custom"]);

    let logger = config.request_logger_factory().unwrap()(&RequestLoggerBuildContext {
        config: runtime,
        config_path: PathBuf::from("/tmp/ctox/config.yaml"),
    });
    assert!(!logger.is_enabled());
}

#[test]
fn default_config_is_empty_and_debug_redacts_password() {
    let config = apply_server_options(Vec::new());
    assert!(config.extra_middleware().is_empty());
    let configured = apply_server_options(vec![with_local_management_password("never-print")]);
    let debug = format!("{configured:?}");
    assert!(debug.contains("has_local_management_password: true"));
    assert!(!debug.contains("never-print"));
}
