// ref: internal/pluginhost/adapters_usage_translation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: translation, thinking and usage capabilities use scoped process RPC
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginabi::{
    METHOD_REQUEST_NORMALIZE, METHOD_REQUEST_TRANSLATE, METHOD_RESPONSE_NORMALIZE_AFTER,
    METHOD_RESPONSE_NORMALIZE_BEFORE, METHOD_RESPONSE_TRANSLATE, METHOD_THINKING_APPLY,
    METHOD_USAGE_HANDLE,
};
use crate::sdk::pluginapi::{
    PayloadResponse, PluginFuture, RequestNormalizer, RequestTransformRequest, RequestTranslator,
    ResponseNormalizer, ResponseTransformRequest, ResponseTranslator, ThinkingApplier,
    ThinkingApplyRequest, UsagePlugin, UsageRecord,
};

use super::adapters::RpcCapabilityClient;
use super::rpc_schema::RpcEmptyResponse;

macro_rules! adapter {
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

adapter!(RpcRequestTranslator);
adapter!(RpcRequestNormalizer);
adapter!(RpcResponseTranslator);
adapter!(RpcUsagePlugin);

impl RequestTranslator for RpcRequestTranslator {
    fn translate_request<'a>(
        &'a self,
        request: RequestTransformRequest,
    ) -> PluginFuture<'a, PayloadResponse> {
        call_payload(&self.client, METHOD_REQUEST_TRANSLATE, request)
    }
}

impl RequestNormalizer for RpcRequestNormalizer {
    fn normalize_request<'a>(
        &'a self,
        request: RequestTransformRequest,
    ) -> PluginFuture<'a, PayloadResponse> {
        call_payload(&self.client, METHOD_REQUEST_NORMALIZE, request)
    }
}

impl ResponseTranslator for RpcResponseTranslator {
    fn translate_response<'a>(
        &'a self,
        request: ResponseTransformRequest,
    ) -> PluginFuture<'a, PayloadResponse> {
        call_payload(&self.client, METHOD_RESPONSE_TRANSLATE, request)
    }
}

#[derive(Clone, Debug)]
pub struct RpcResponseNormalizer {
    client: RpcCapabilityClient,
    phase: ResponseNormalizePhase,
}

impl RpcResponseNormalizer {
    pub fn new(client: RpcCapabilityClient, phase: ResponseNormalizePhase) -> Self {
        Self { client, phase }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseNormalizePhase {
    Before,
    After,
}

impl ResponseNormalizer for RpcResponseNormalizer {
    fn normalize_response<'a>(
        &'a self,
        request: ResponseTransformRequest,
    ) -> PluginFuture<'a, PayloadResponse> {
        let method = match self.phase {
            ResponseNormalizePhase::Before => METHOD_RESPONSE_NORMALIZE_BEFORE,
            ResponseNormalizePhase::After => METHOD_RESPONSE_NORMALIZE_AFTER,
        };
        call_payload(&self.client, method, request)
    }
}

#[derive(Clone, Debug)]
pub struct RpcThinkingApplier {
    identifier: String,
    client: RpcCapabilityClient,
}

impl RpcThinkingApplier {
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

impl ThinkingApplier for RpcThinkingApplier {
    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn apply_thinking<'a>(
        &'a self,
        request: ThinkingApplyRequest,
    ) -> PluginFuture<'a, PayloadResponse> {
        call_payload(&self.client, METHOD_THINKING_APPLY, request)
    }
}

impl UsagePlugin for RpcUsagePlugin {
    fn handle_usage<'a>(&'a self, record: UsageRecord) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            self.client
                .call::<_, RpcEmptyResponse>(METHOD_USAGE_HANDLE, &record, None)
                .await
                .map(|_| ())
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

fn call_payload<'a, Request>(
    client: &'a RpcCapabilityClient,
    method: &'static str,
    request: Request,
) -> PluginFuture<'a, PayloadResponse>
where
    Request: serde::Serialize + Send + Sync + 'a,
{
    Box::pin(async move {
        client
            .call(method, &request, None)
            .await
            .map_err(|error| Arc::new(error) as _)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidIdentifier;

impl std::fmt::Display for InvalidIdentifier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("thinking provider identifier is invalid")
    }
}

impl std::error::Error for InvalidIdentifier {}
