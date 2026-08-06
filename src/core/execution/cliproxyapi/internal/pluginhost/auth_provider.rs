// ref: internal/pluginhost/auth_provider.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: provider calls cross isolated RPC with an injected host summary
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginabi::{
    METHOD_AUTH_LOGIN_POLL, METHOD_AUTH_LOGIN_START, METHOD_AUTH_PARSE, METHOD_AUTH_REFRESH,
};
use crate::sdk::pluginapi::{
    AuthLoginPollRequest, AuthLoginPollResponse, AuthLoginStartRequest, AuthLoginStartResponse,
    AuthParseRequest, AuthParseResponse, AuthProvider, AuthRefreshRequest, AuthRefreshResponse,
    HostConfigSummary, PluginFuture,
};

use super::adapters::RpcCapabilityClient;

pub trait HostConfigSummarySource: Send + Sync {
    fn snapshot(&self) -> HostConfigSummary;
}

#[derive(Clone)]
pub struct RpcAuthProvider {
    identifier: String,
    client: RpcCapabilityClient,
    host: Arc<dyn HostConfigSummarySource>,
}

impl std::fmt::Debug for RpcAuthProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RpcAuthProvider")
            .field("identifier", &self.identifier)
            .finish_non_exhaustive()
    }
}

impl RpcAuthProvider {
    pub fn new(
        identifier: &str,
        client: RpcCapabilityClient,
        host: Arc<dyn HostConfigSummarySource>,
    ) -> Result<Self, AuthProviderConfigError> {
        let identifier = normalize_provider(identifier);
        if identifier.is_empty() {
            return Err(AuthProviderConfigError::InvalidIdentifier);
        }
        Ok(Self {
            identifier,
            client,
            host,
        })
    }
}

impl AuthProvider for RpcAuthProvider {
    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn parse_auth<'a>(
        &'a self,
        mut request: AuthParseRequest,
    ) -> PluginFuture<'a, AuthParseResponse> {
        Box::pin(async move {
            request.provider = normalize_provider(if request.provider.trim().is_empty() {
                &self.identifier
            } else {
                &request.provider
            });
            if request.host == HostConfigSummary::default() {
                request.host = self.host.snapshot();
            }
            self.client
                .call(METHOD_AUTH_PARSE, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }

    fn start_login<'a>(
        &'a self,
        mut request: AuthLoginStartRequest,
    ) -> PluginFuture<'a, AuthLoginStartResponse> {
        Box::pin(async move {
            request.provider = normalize_provider(if request.provider.trim().is_empty() {
                &self.identifier
            } else {
                &request.provider
            });
            request.base_url = request.base_url.trim().to_owned();
            if request.host == HostConfigSummary::default() {
                request.host = self.host.snapshot();
            }
            self.client
                .call(METHOD_AUTH_LOGIN_START, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }

    fn poll_login<'a>(
        &'a self,
        mut request: AuthLoginPollRequest,
    ) -> PluginFuture<'a, AuthLoginPollResponse> {
        Box::pin(async move {
            request.provider = normalize_provider(if request.provider.trim().is_empty() {
                &self.identifier
            } else {
                &request.provider
            });
            request.state = request.state.trim().to_owned();
            if request.host == HostConfigSummary::default() {
                request.host = self.host.snapshot();
            }
            self.client
                .call(METHOD_AUTH_LOGIN_POLL, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }

    fn refresh_auth<'a>(
        &'a self,
        mut request: AuthRefreshRequest,
    ) -> PluginFuture<'a, AuthRefreshResponse> {
        Box::pin(async move {
            request.auth_provider =
                normalize_provider(if request.auth_provider.trim().is_empty() {
                    &self.identifier
                } else {
                    &request.auth_provider
                });
            if request.host == HostConfigSummary::default() {
                request.host = self.host.snapshot();
            }
            self.client
                .call(METHOD_AUTH_REFRESH, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

fn normalize_provider(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthProviderConfigError {
    InvalidIdentifier,
}

impl std::fmt::Display for AuthProviderConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("auth provider identifier is invalid")
    }
}

impl std::error::Error for AuthProviderConfigError {}
