// ref: internal/pluginhost/scheduler.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: scheduler calls cross the isolated RPC client
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::pluginabi::METHOD_SCHEDULER_PICK;
use crate::sdk::pluginapi::{PluginFuture, Scheduler, SchedulerPickRequest, SchedulerPickResponse};

use super::rpc_client::RpcPluginClient;

#[derive(Clone, Debug)]
pub struct RpcScheduler {
    client: RpcPluginClient,
}

impl RpcScheduler {
    pub fn new(client: RpcPluginClient) -> Self {
        Self { client }
    }
}

impl Scheduler for RpcScheduler {
    fn pick<'a>(
        &'a self,
        request: SchedulerPickRequest,
    ) -> PluginFuture<'a, SchedulerPickResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_SCHEDULER_PICK, &request, None)
                .await
                .map_err(|error| std::sync::Arc::new(error) as _)
        })
    }
}
