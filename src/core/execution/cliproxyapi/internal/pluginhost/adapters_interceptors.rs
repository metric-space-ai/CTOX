// ref: internal/pluginhost/adapters_interceptors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: request lifecycle and interceptor calls use scoped process RPC
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginabi::{
    METHOD_REQUEST_COMPLETE, METHOD_REQUEST_INTERCEPT_AFTER, METHOD_REQUEST_INTERCEPT_BEFORE,
    METHOD_RESPONSE_INTERCEPT_AFTER, METHOD_RESPONSE_INTERCEPT_STREAM_CHUNK,
};
use crate::sdk::pluginapi::{
    PluginFuture, RequestCompletion, RequestInterceptRequest, RequestInterceptResponse,
    RequestInterceptor, RequestLifecyclePlugin, ResponseInterceptRequest,
    ResponseInterceptResponse, ResponseInterceptor, StreamChunkInterceptRequest,
    StreamChunkInterceptResponse, StreamChunkInterceptor,
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

adapter!(RpcRequestInterceptor);
adapter!(RpcRequestLifecyclePlugin);
adapter!(RpcResponseInterceptor);
adapter!(RpcStreamChunkInterceptor);

impl RequestInterceptor for RpcRequestInterceptor {
    fn intercept_request_before_auth<'a>(
        &'a self,
        request: RequestInterceptRequest,
    ) -> PluginFuture<'a, RequestInterceptResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_REQUEST_INTERCEPT_BEFORE, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }

    fn intercept_request_after_auth<'a>(
        &'a self,
        request: RequestInterceptRequest,
    ) -> PluginFuture<'a, RequestInterceptResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_REQUEST_INTERCEPT_AFTER, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

impl RequestLifecyclePlugin for RpcRequestLifecyclePlugin {
    fn handle_request_complete<'a>(
        &'a self,
        completion: RequestCompletion,
    ) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            self.client
                .call::<_, RpcEmptyResponse>(METHOD_REQUEST_COMPLETE, &completion, None)
                .await
                .map(|_| ())
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

impl ResponseInterceptor for RpcResponseInterceptor {
    fn intercept_response<'a>(
        &'a self,
        request: ResponseInterceptRequest,
    ) -> PluginFuture<'a, ResponseInterceptResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_RESPONSE_INTERCEPT_AFTER, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

impl StreamChunkInterceptor for RpcStreamChunkInterceptor {
    fn intercept_stream_chunk<'a>(
        &'a self,
        request: StreamChunkInterceptRequest,
    ) -> PluginFuture<'a, StreamChunkInterceptResponse> {
        Box::pin(async move {
            self.client
                .call(METHOD_RESPONSE_INTERCEPT_STREAM_CHUNK, &request, None)
                .await
                .map_err(|error| Arc::new(error) as _)
        })
    }
}

#[derive(Clone)]
pub struct RequestInterceptorRecord {
    pub plugin_id: String,
    pub priority: i32,
    pub interceptor: Arc<dyn RequestInterceptor>,
}

#[derive(Clone, Default)]
pub struct RequestInterceptorChain {
    records: Vec<RequestInterceptorRecord>,
}

impl RequestInterceptorChain {
    pub fn new(mut records: Vec<RequestInterceptorRecord>) -> Self {
        records.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.plugin_id.cmp(&right.plugin_id))
        });
        Self { records }
    }

    pub async fn before_auth(&self, request: RequestInterceptRequest) -> RequestInterceptResponse {
        self.run(request, false).await
    }

    pub async fn after_auth(&self, request: RequestInterceptRequest) -> RequestInterceptResponse {
        self.run(request, true).await
    }

    async fn run(
        &self,
        mut request: RequestInterceptRequest,
        after_auth: bool,
    ) -> RequestInterceptResponse {
        let mut combined = RequestInterceptResponse::default();
        for record in &self.records {
            let result = if after_auth {
                record
                    .interceptor
                    .intercept_request_after_auth(request.clone())
                    .await
            } else {
                record
                    .interceptor
                    .intercept_request_before_auth(request.clone())
                    .await
            };
            let Ok(response) = result else {
                continue;
            };
            if !response.headers.is_empty() {
                request.headers.extend(response.headers.clone());
                combined.headers.extend(response.headers.clone());
            }
            if !response.body.is_empty() {
                request.body.clone_from(&response.body);
                combined.body.clone_from(&response.body);
            }
            combined
                .clear_headers
                .extend(response.clear_headers.clone());
            if response.terminate {
                return response;
            }
        }
        combined
    }
}

#[derive(Clone)]
pub struct RequestLifecycleRecord {
    pub plugin_id: String,
    pub priority: i32,
    pub lifecycle: Arc<dyn RequestLifecyclePlugin>,
}

#[derive(Clone, Default)]
pub struct RequestCompletionDispatcher {
    records: Vec<RequestLifecycleRecord>,
}

impl RequestCompletionDispatcher {
    pub fn new(mut records: Vec<RequestLifecycleRecord>) -> Self {
        records.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.plugin_id.cmp(&right.plugin_id))
        });
        Self { records }
    }

    pub fn complete(&self, completion: RequestCompletion) {
        for record in &self.records {
            let lifecycle = record.lifecycle.clone();
            let completion = completion.clone();
            tokio::spawn(async move {
                let _ = lifecycle.handle_request_complete(completion).await;
            });
        }
    }
}
