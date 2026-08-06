// ref: sdk/cliproxy/types.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::internal::config::ValidatedRuntimeConfig;
use crate::sdk::cliproxy::auth::Auth;

pub use super::providers::{ApiKeyClientResult, TokenClientResult};
use super::providers::{ClientProviderError, LoadContext};

pub type ClientLoadFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, ClientProviderError>> + Send + 'a>>;

pub trait TokenClientProvider: Send + Sync {
    fn load<'a>(
        &'a self,
        context: &'a LoadContext,
        config: &'a ValidatedRuntimeConfig,
    ) -> ClientLoadFuture<'a, TokenClientResult>;
}

pub trait ApiKeyClientProvider: Send + Sync {
    fn load<'a>(
        &'a self,
        context: &'a LoadContext,
        config: &'a ValidatedRuntimeConfig,
    ) -> ClientLoadFuture<'a, ApiKeyClientResult>;
}

impl TokenClientProvider for super::providers::FileTokenClientProvider {
    fn load<'a>(
        &'a self,
        context: &'a LoadContext,
        _config: &'a ValidatedRuntimeConfig,
    ) -> ClientLoadFuture<'a, TokenClientResult> {
        Box::pin(async move {
            Ok(super::providers::FileTokenClientProvider::load(
                self, context,
            ))
        })
    }
}

impl ApiKeyClientProvider for super::providers::ConfiguredApiKeyClientProvider {
    fn load<'a>(
        &'a self,
        context: &'a LoadContext,
        _config: &'a ValidatedRuntimeConfig,
    ) -> ClientLoadFuture<'a, ApiKeyClientResult> {
        Box::pin(
            async move { super::providers::ConfiguredApiKeyClientProvider::load(self, context) },
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthUpdateAction {
    Add,
    Modify,
    Delete,
}

#[derive(Clone, Debug, Default)]
pub struct AuthUpdate {
    pub action: Option<AuthUpdateAction>,
    pub id: String,
    pub auth: Option<Auth>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PluginAuthParseRequest {
    pub provider: String,
    pub source_path: PathBuf,
    pub payload: Vec<u8>,
}

pub type PluginAuthParseFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<(T, bool), PluginAuthParseError>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PluginAuthParseError {
    Rejected,
    Invalid,
}

impl fmt::Display for PluginAuthParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rejected => "plugin auth parser rejected the request",
            Self::Invalid => "plugin auth parser returned invalid auth",
        })
    }
}

impl std::error::Error for PluginAuthParseError {}

pub trait PluginAuthParser: Send + Sync {
    fn parse_auth<'a>(
        &'a self,
        request: PluginAuthParseRequest,
    ) -> PluginAuthParseFuture<'a, Option<Auth>>;
}

pub trait PluginMultiAuthParser: Send + Sync {
    fn parse_auths<'a>(
        &'a self,
        request: PluginAuthParseRequest,
    ) -> PluginAuthParseFuture<'a, Vec<Auth>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatcherError {
    Start,
    Stop,
    Factory,
}

impl fmt::Display for WatcherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Start => "watcher start failed",
            Self::Stop => "watcher stop failed",
            Self::Factory => "watcher construction failed",
        })
    }
}

impl std::error::Error for WatcherError {}

pub type WatcherStartFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), WatcherError>> + Send + 'a>>;
pub type ReloadCallback = Arc<dyn Fn(ValidatedRuntimeConfig) + Send + Sync>;
pub type WatcherStart =
    Arc<dyn for<'a> Fn(&'a LoadContext) -> WatcherStartFuture<'a> + Send + Sync>;
pub type WatcherStop = Arc<dyn Fn() -> Result<(), WatcherError> + Send + Sync>;
pub type WatcherConfigSetter = Arc<dyn Fn(ValidatedRuntimeConfig) + Send + Sync>;
pub type WatcherAuthSnapshot = Arc<dyn Fn() -> Vec<Auth> + Send + Sync>;
pub type WatcherQueueSetter = Arc<dyn Fn(mpsc::Sender<AuthUpdate>) + Send + Sync>;
pub type WatcherUpdateDispatcher = Arc<dyn Fn(AuthUpdate) -> bool + Send + Sync>;
pub type WatcherPluginParserSetter = Arc<dyn Fn(Arc<dyn PluginAuthParser>) + Send + Sync>;
pub type WatcherConfigReloader = Arc<dyn Fn() + Send + Sync>;

pub trait WatcherFactory: Send + Sync {
    fn create(
        &self,
        config_path: &Path,
        auth_dir: &Path,
        reload: ReloadCallback,
    ) -> Result<WatcherWrapper, WatcherError>;
}

#[derive(Default)]
pub struct WatcherBindings {
    pub start: Option<WatcherStart>,
    pub stop: Option<WatcherStop>,
    pub set_config: Option<WatcherConfigSetter>,
    pub snapshot_auths: Option<WatcherAuthSnapshot>,
    pub set_update_queue: Option<WatcherQueueSetter>,
    pub dispatch_runtime_update: Option<WatcherUpdateDispatcher>,
    pub dispatch_persisted_auth: Option<WatcherUpdateDispatcher>,
    pub set_plugin_auth_parser: Option<WatcherPluginParserSetter>,
    pub reload_config_if_changed: Option<WatcherConfigReloader>,
}

#[derive(Default)]
pub struct WatcherWrapper {
    bindings: WatcherBindings,
}

impl fmt::Debug for WatcherWrapper {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WatcherWrapper")
            .field("can_start", &self.bindings.start.is_some())
            .field("can_stop", &self.bindings.stop.is_some())
            .field("can_set_config", &self.bindings.set_config.is_some())
            .field(
                "can_snapshot_auths",
                &self.bindings.snapshot_auths.is_some(),
            )
            .field(
                "can_set_update_queue",
                &self.bindings.set_update_queue.is_some(),
            )
            .field(
                "can_dispatch_runtime_update",
                &self.bindings.dispatch_runtime_update.is_some(),
            )
            .field(
                "can_dispatch_persisted_auth",
                &self.bindings.dispatch_persisted_auth.is_some(),
            )
            .field(
                "can_set_plugin_auth_parser",
                &self.bindings.set_plugin_auth_parser.is_some(),
            )
            .field(
                "can_reload_config_if_changed",
                &self.bindings.reload_config_if_changed.is_some(),
            )
            .finish()
    }
}

impl WatcherWrapper {
    #[must_use]
    pub fn new(bindings: WatcherBindings) -> Self {
        Self { bindings }
    }

    pub async fn start(&self, context: &LoadContext) -> Result<(), WatcherError> {
        match &self.bindings.start {
            Some(start) => start(context).await,
            None => Ok(()),
        }
    }

    pub fn stop(&self) -> Result<(), WatcherError> {
        match &self.bindings.stop {
            Some(stop) => stop(),
            None => Ok(()),
        }
    }

    pub fn set_config(&self, config: ValidatedRuntimeConfig) {
        if let Some(set_config) = &self.bindings.set_config {
            set_config(config);
        }
    }

    #[must_use]
    pub fn reload_config_if_changed(&self) -> bool {
        self.bindings
            .reload_config_if_changed
            .as_ref()
            .is_some_and(|reload| {
                reload();
                true
            })
    }

    pub fn set_plugin_auth_parser(&self, parser: Arc<dyn PluginAuthParser>) {
        if let Some(set_parser) = &self.bindings.set_plugin_auth_parser {
            set_parser(parser);
        }
    }

    #[must_use]
    pub fn dispatch_runtime_auth_update(&self, update: AuthUpdate) -> bool {
        self.bindings
            .dispatch_runtime_update
            .as_ref()
            .is_some_and(|dispatch| dispatch(update))
    }

    #[must_use]
    pub fn dispatch_persisted_auth_update(&self, update: AuthUpdate) -> bool {
        self.bindings
            .dispatch_persisted_auth
            .as_ref()
            .is_some_and(|dispatch| dispatch(update))
    }

    #[must_use]
    pub fn snapshot_auths(&self) -> Vec<Auth> {
        self.bindings
            .snapshot_auths
            .as_ref()
            .map_or_else(Vec::new, |snapshot| snapshot())
    }

    pub fn set_auth_update_queue(&self, queue: mpsc::Sender<AuthUpdate>) {
        if let Some(set_queue) = &self.bindings.set_update_queue {
            set_queue(queue);
        }
    }
}
