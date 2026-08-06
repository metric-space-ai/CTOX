// ref: sdk/api/options.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Instance-scoped options for embedding the proxy API in a CTOX-owned server.
//!
//! Gin's engine and handler types do not cross the Rust boundary.  The same
//! extension points are represented by typed build state that a host consumes
//! while assembling its supervised listener.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::internal::config::CliproxyRuntimeConfig;
use crate::internal::logging::request_logger::RequestLogger;

pub trait ServerMiddleware: Send + Sync {
    fn name(&self) -> &str;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EngineBuildState {
    pub attributes: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouteRegistry {
    routes: Vec<String>,
}

impl RouteRegistry {
    pub fn register(&mut self, route: impl Into<String>) {
        self.routes.push(route.into());
    }

    #[must_use]
    pub fn routes(&self) -> &[String] {
        &self.routes
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BaseApiHandler {
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestLoggerBuildContext {
    pub config: CliproxyRuntimeConfig,
    pub config_path: PathBuf,
}

impl RequestLoggerBuildContext {
    #[must_use]
    pub fn config_dir(&self) -> &Path {
        self.config_path.parent().unwrap_or_else(|| Path::new(""))
    }
}

pub type EngineConfigurator = Arc<dyn Fn(&mut EngineBuildState) + Send + Sync>;
pub type RouterConfigurator =
    Arc<dyn Fn(&mut RouteRegistry, &BaseApiHandler, &CliproxyRuntimeConfig) + Send + Sync>;
pub type RequestLoggerFactory =
    Arc<dyn Fn(&RequestLoggerBuildContext) -> Arc<dyn RequestLogger> + Send + Sync>;
pub type KeepAliveTimeoutCallback = Arc<dyn Fn() + Send + Sync>;

#[derive(Clone)]
pub struct KeepAliveEndpoint {
    pub timeout: Duration,
    pub on_timeout: KeepAliveTimeoutCallback,
}

impl fmt::Debug for KeepAliveEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KeepAliveEndpoint")
            .field("timeout", &self.timeout)
            .field("has_callback", &true)
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct ServerOptionConfig {
    extra_middleware: Vec<Arc<dyn ServerMiddleware>>,
    engine_configurator: Option<EngineConfigurator>,
    router_configurator: Option<RouterConfigurator>,
    request_logger_factory: Option<RequestLoggerFactory>,
    local_management_password: Option<String>,
    keep_alive: Option<KeepAliveEndpoint>,
}

impl fmt::Debug for ServerOptionConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerOptionConfig")
            .field("middleware_count", &self.extra_middleware.len())
            .field(
                "has_engine_configurator",
                &self.engine_configurator.is_some(),
            )
            .field(
                "has_router_configurator",
                &self.router_configurator.is_some(),
            )
            .field(
                "has_request_logger_factory",
                &self.request_logger_factory.is_some(),
            )
            .field(
                "has_local_management_password",
                &self.local_management_password.is_some(),
            )
            .field("keep_alive", &self.keep_alive)
            .finish()
    }
}

impl ServerOptionConfig {
    #[must_use]
    pub fn extra_middleware(&self) -> &[Arc<dyn ServerMiddleware>] {
        &self.extra_middleware
    }

    #[must_use]
    pub fn engine_configurator(&self) -> Option<&EngineConfigurator> {
        self.engine_configurator.as_ref()
    }

    #[must_use]
    pub fn router_configurator(&self) -> Option<&RouterConfigurator> {
        self.router_configurator.as_ref()
    }

    #[must_use]
    pub fn request_logger_factory(&self) -> Option<&RequestLoggerFactory> {
        self.request_logger_factory.as_ref()
    }

    #[must_use]
    pub fn local_management_password(&self) -> Option<&str> {
        self.local_management_password.as_deref()
    }

    #[must_use]
    pub fn keep_alive(&self) -> Option<&KeepAliveEndpoint> {
        self.keep_alive.as_ref()
    }
}

pub type ServerOption = Box<dyn FnOnce(&mut ServerOptionConfig) + Send>;

/// Applies options in caller order, matching Go's functional-option contract.
#[must_use]
pub fn apply_server_options(options: impl IntoIterator<Item = ServerOption>) -> ServerOptionConfig {
    let mut config = ServerOptionConfig::default();
    for option in options {
        option(&mut config);
    }
    config
}

pub fn with_middleware(middleware: Vec<Arc<dyn ServerMiddleware>>) -> ServerOption {
    Box::new(move |config| config.extra_middleware.extend(middleware))
}

pub fn with_engine_configurator(configurator: Option<EngineConfigurator>) -> ServerOption {
    Box::new(move |config| {
        if let Some(configurator) = configurator {
            config.engine_configurator = Some(configurator);
        }
    })
}

pub fn with_router_configurator(configurator: Option<RouterConfigurator>) -> ServerOption {
    Box::new(move |config| {
        if let Some(configurator) = configurator {
            config.router_configurator = Some(configurator);
        }
    })
}

/// Stores only an instance-local secret supplied by the CTOX secret boundary.
/// Empty values clear the option instead of creating an ambient fallback.
pub fn with_local_management_password(password: impl Into<String>) -> ServerOption {
    let password = password.into();
    Box::new(move |config| {
        let password = password.trim();
        config.local_management_password = (!password.is_empty()).then(|| password.to_owned());
    })
}

pub fn with_keep_alive_endpoint(
    timeout: Duration,
    on_timeout: Option<KeepAliveTimeoutCallback>,
) -> ServerOption {
    Box::new(move |config| {
        if !timeout.is_zero() {
            if let Some(on_timeout) = on_timeout {
                config.keep_alive = Some(KeepAliveEndpoint {
                    timeout,
                    on_timeout,
                });
            }
        }
    })
}

pub fn with_request_logger_factory(factory: Option<RequestLoggerFactory>) -> ServerOption {
    Box::new(move |config| {
        if let Some(factory) = factory {
            config.request_logger_factory = Some(factory);
        }
    })
}
