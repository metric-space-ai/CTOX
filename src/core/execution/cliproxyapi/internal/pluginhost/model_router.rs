// ref: internal/pluginhost/model_router.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: model routing calls cross the isolated RPC client
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::pluginabi::METHOD_MODEL_ROUTE;
use crate::sdk::pluginapi::{ModelRouteRequest, ModelRouteResponse, ModelRouter, PluginFuture};

use super::rpc_client::RpcPluginClient;

#[derive(Clone, Debug)]
pub struct RpcModelRouter {
    client: RpcPluginClient,
}

impl RpcModelRouter {
    pub fn new(client: RpcPluginClient) -> Self {
        Self { client }
    }
}

impl ModelRouter for RpcModelRouter {
    fn route_model<'a>(
        &'a self,
        request: ModelRouteRequest,
    ) -> PluginFuture<'a, ModelRouteResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_MODEL_ROUTE, &request, None)
                .await
                .map_err(|error| std::sync::Arc::new(error) as _)
        })
    }
}
