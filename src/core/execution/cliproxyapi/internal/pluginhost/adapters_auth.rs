// ref: internal/pluginhost/adapters_auth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: auth-adjacent capabilities cross the isolated RPC boundary
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginabi::{
    METHOD_FRONTEND_AUTH_AUTHENTICATE, METHOD_MODEL_FOR_AUTH, METHOD_MODEL_REGISTER,
    METHOD_MODEL_STATIC,
};
use crate::sdk::pluginapi::{
    AuthModelRequest, FrontendAuthProvider, FrontendAuthRequest, FrontendAuthResponse,
    ModelProvider, ModelRegistrar, ModelRegistrationRequest, ModelRegistrationResponse,
    ModelResponse, PluginFuture, StaticModelRequest,
};

use super::adapters::RpcCapabilityClient;

macro_rules! rpc_capability {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            client: RpcCapabilityClient,
        }
        impl $name {
            pub fn new(client: RpcCapabilityClient) -> Self {
                Self { client }
            }
        }
    };
}

rpc_capability!(RpcModelRegistrar);
rpc_capability!(RpcModelProvider);

impl ModelRegistrar for RpcModelRegistrar {
    fn register_models<'a>(
        &'a self,
        request: ModelRegistrationRequest,
    ) -> PluginFuture<'a, ModelRegistrationResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_MODEL_REGISTER, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

impl ModelProvider for RpcModelProvider {
    fn static_models<'a>(&'a self, request: StaticModelRequest) -> PluginFuture<'a, ModelResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_MODEL_STATIC, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }

    fn models_for_auth<'a>(&'a self, request: AuthModelRequest) -> PluginFuture<'a, ModelResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_MODEL_FOR_AUTH, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

#[derive(Clone, Debug)]
pub struct RpcFrontendAuthProvider {
    identifier: String,
    client: RpcCapabilityClient,
}

impl RpcFrontendAuthProvider {
    pub fn new(identifier: &str, client: RpcCapabilityClient) -> Result<Self, InvalidIdentifier> {
        let identifier = identifier.trim();
        if identifier.is_empty() || identifier.len() > 128 {
            return Err(InvalidIdentifier);
        }
        Ok(Self {
            identifier: identifier.to_owned(),
            client,
        })
    }
}

impl FrontendAuthProvider for RpcFrontendAuthProvider {
    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn authenticate<'a>(
        &'a self,
        request: FrontendAuthRequest,
    ) -> PluginFuture<'a, FrontendAuthResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_FRONTEND_AUTH_AUTHENTICATE, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidIdentifier;

impl std::fmt::Display for InvalidIdentifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("plugin capability identifier is invalid")
    }
}

impl std::error::Error for InvalidIdentifier {}
