// Origin: CTOX
// License: AGPL-3.0-only

mod context;
mod lifecycle;
mod types;
mod websocket;

#[cfg(test)]
mod lifecycle_test;
#[cfg(test)]
mod types_test;
#[cfg(test)]
mod websocket_test;

pub use context::{
    downstream_websocket, required_upstream_websocket, with_downstream_websocket,
    with_required_upstream_websocket, ExecutionTransportContext,
};
pub use lifecycle::{
    bind_execution_resource, BindAndCloseError, BoundResourceCloser, ExecutionLifecycle,
    LifecycleError, LifecycleResult, ResourceCloseFn,
};
pub use types::{
    response_format_or_source, ExecutionError, ExecutionMetadata, Headers, JsonMetadata, Options,
    QueryValues, Request, RequestAfterAuthInterceptRequest, RequestAfterAuthInterceptResponse,
    RequestAfterAuthInterceptor, RequestScopedError, RequestTerminatedError, Response,
    SelectedAuthCallback, SelectedAuthIndexCallback, StatusError, StreamChunk, StreamResult,
    AUTH_SELECTION_MODEL_METADATA_KEY, CALLER_SCOPE_METADATA_KEY, DERIVED_SESSION_ID_METADATA_KEY,
    DISALLOW_FREE_AUTH_METADATA_KEY, EXECUTION_SESSION_METADATA_KEY, GENERATE_METADATA_KEY,
    PINNED_AUTH_METADATA_KEY, REASONING_EFFORT_METADATA_KEY, REQUESTED_MODEL_METADATA_KEY,
    REQUEST_PATH_METADATA_KEY, SELECTED_AUTH_CALLBACK_METADATA_KEY,
    SELECTED_AUTH_INDEX_CALLBACK_METADATA_KEY, SELECTED_AUTH_INDEX_METADATA_KEY,
    SELECTED_AUTH_METADATA_KEY, SERVICE_TIER_METADATA_KEY,
};
pub use websocket::{
    is_upstream_websocket_replay_required, new_upstream_websocket_replay_required_error,
    UpstreamWebsocketReplayRequiredError,
};
