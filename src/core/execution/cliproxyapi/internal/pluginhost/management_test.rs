// ref: internal/pluginhost/management_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: typed management route registration and dispatch without an HTTP data bridge
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::sdk::pluginapi::{
    ManagementApi, ManagementHandler, ManagementRegistrationRequest,
    ManagementRegistrationResponse, ManagementRequest, ManagementResponse, ManagementRoute,
    Metadata, PluginFuture, ResourceRoute,
};

use super::management::{ManagementDispatchError, ManagementPluginRecord, ManagementRegistry};

struct Handler {
    response: Vec<u8>,
    seen: Mutex<Vec<ManagementRequest>>,
}

impl ManagementHandler for Handler {
    fn handle_management<'a>(
        &'a self,
        request: ManagementRequest,
    ) -> PluginFuture<'a, ManagementResponse> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request);
            Ok(ManagementResponse {
                status_code: 200,
                body: self.response.clone(),
                ..ManagementResponse::default()
            })
        })
    }
}

struct Api {
    route: String,
    handler: Arc<Handler>,
}

impl ManagementApi for Api {
    fn register_management<'a>(
        &'a self,
        _request: ManagementRegistrationRequest,
    ) -> PluginFuture<'a, ManagementRegistrationResponse> {
        Box::pin(async move {
            Ok(ManagementRegistrationResponse {
                routes: vec![ManagementRoute {
                    method: "GET".to_owned(),
                    path: self.route.clone(),
                    menu: String::new(),
                    description: String::new(),
                    handler: self.handler.clone(),
                }],
                resources: vec![ResourceRoute {
                    path: format!("{}/asset", self.route),
                    menu: String::new(),
                    description: String::new(),
                    handler: self.handler.clone(),
                }],
            })
        })
    }
}

fn record(id: &str, priority: i32, api: Arc<Api>) -> ManagementPluginRecord {
    ManagementPluginRecord {
        plugin_id: id.to_owned(),
        priority,
        metadata: Metadata {
            name: id.to_owned(),
            ..Metadata::default()
        },
        api,
    }
}

#[tokio::test]
async fn higher_priority_route_wins_and_dispatch_preserves_request() {
    let low_handler = Arc::new(Handler {
        response: b"low".to_vec(),
        seen: Mutex::new(Vec::new()),
    });
    let high_handler = Arc::new(Handler {
        response: b"high".to_vec(),
        seen: Mutex::new(Vec::new()),
    });
    let mut registry = ManagementRegistry::default();
    let errors = registry
        .register(
            &[
                record(
                    "low",
                    1,
                    Arc::new(Api {
                        route: "/status".to_owned(),
                        handler: low_handler,
                    }),
                ),
                record(
                    "high",
                    10,
                    Arc::new(Api {
                        route: "/status".to_owned(),
                        handler: high_handler.clone(),
                    }),
                ),
            ],
            "/plugins",
            "/plugin-assets",
        )
        .await;
    assert!(errors.is_empty());
    let response = registry
        .dispatch(ManagementRequest {
            method: "get".to_owned(),
            path: "/status".to_owned(),
            body: b"request".to_vec(),
            ..ManagementRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(response.body, b"high");
    assert_eq!(
        high_handler
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .body,
        b"request"
    );
}

#[tokio::test]
async fn resource_dispatch_forces_get_and_invalid_paths_fail_closed() {
    let handler = Arc::new(Handler {
        response: b"asset".to_vec(),
        seen: Mutex::new(Vec::new()),
    });
    let mut registry = ManagementRegistry::default();
    registry
        .register(
            &[record(
                "plugin",
                1,
                Arc::new(Api {
                    route: "/ui".to_owned(),
                    handler: handler.clone(),
                }),
            )],
            "/plugins",
            "/plugin-assets",
        )
        .await;
    let response = registry
        .dispatch_resource(ManagementRequest {
            method: "DELETE".to_owned(),
            path: "/ui/asset".to_owned(),
            ..ManagementRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(response.body, b"asset");
    assert_eq!(
        handler
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
            .method,
        "GET"
    );
    assert_eq!(
        registry
            .dispatch(ManagementRequest {
                method: "GET".to_owned(),
                path: "/../secret".to_owned(),
                ..ManagementRequest::default()
            })
            .await,
        Err(ManagementDispatchError::InvalidPath)
    );
}
