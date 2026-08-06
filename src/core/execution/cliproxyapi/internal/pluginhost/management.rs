// ref: internal/pluginhost/management.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: typed control-plane route registry; no browser data-plane proxy
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::sdk::pluginapi::{
    ManagementApi, ManagementHandler, ManagementRegistrationRequest, ManagementRequest,
    ManagementResponse, Metadata,
};

#[derive(Clone)]
pub struct ManagementPluginRecord {
    pub plugin_id: String,
    pub priority: i32,
    pub metadata: Metadata,
    pub api: Arc<dyn ManagementApi>,
}

#[derive(Clone)]
struct RouteRecord {
    plugin_id: String,
    priority: i32,
    handler: Arc<dyn ManagementHandler>,
}

#[derive(Default)]
pub struct ManagementRegistry {
    routes: BTreeMap<(String, String), RouteRecord>,
    resources: BTreeMap<String, RouteRecord>,
}

impl ManagementRegistry {
    pub async fn register(
        &mut self,
        records: &[ManagementPluginRecord],
        base_path: &str,
        resource_base_path: &str,
    ) -> Vec<ManagementRegistrationError> {
        let mut errors = Vec::new();
        for record in records {
            let response = record
                .api
                .register_management(ManagementRegistrationRequest {
                    plugin: record.metadata.clone(),
                    base_path: base_path.to_owned(),
                    resource_base_path: resource_base_path.to_owned(),
                })
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    errors.push(ManagementRegistrationError {
                        plugin_id: record.plugin_id.clone(),
                        route: String::new(),
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            for route in response.routes {
                let method = route.method.trim().to_ascii_uppercase();
                let path = normalize_route_path(&route.path);
                if !valid_method(&method) || path.is_none() {
                    errors.push(invalid(record, route.path));
                    continue;
                }
                let path = path.expect("checked above");
                let key = (method, path.clone());
                if self
                    .routes
                    .get(&key)
                    .is_some_and(|current| current.priority >= record.priority)
                {
                    errors.push(conflict(record, path));
                    continue;
                }
                self.routes.insert(
                    key,
                    RouteRecord {
                        plugin_id: record.plugin_id.clone(),
                        priority: record.priority,
                        handler: route.handler,
                    },
                );
            }
            for resource in response.resources {
                let Some(path) = normalize_route_path(&resource.path) else {
                    errors.push(invalid(record, resource.path));
                    continue;
                };
                if self
                    .resources
                    .get(&path)
                    .is_some_and(|current| current.priority >= record.priority)
                {
                    errors.push(conflict(record, path));
                    continue;
                }
                self.resources.insert(
                    path,
                    RouteRecord {
                        plugin_id: record.plugin_id.clone(),
                        priority: record.priority,
                        handler: resource.handler,
                    },
                );
            }
        }
        errors
    }

    pub async fn dispatch(
        &self,
        request: ManagementRequest,
    ) -> Result<ManagementResponse, ManagementDispatchError> {
        let method = request.method.trim().to_ascii_uppercase();
        let path =
            normalize_route_path(&request.path).ok_or(ManagementDispatchError::InvalidPath)?;
        let route = self
            .routes
            .get(&(method, path))
            .ok_or(ManagementDispatchError::NotFound)?;
        route
            .handler
            .handle_management(request)
            .await
            .map_err(|error| ManagementDispatchError::Plugin {
                plugin_id: route.plugin_id.clone(),
                message: error.to_string(),
            })
    }

    pub async fn dispatch_resource(
        &self,
        mut request: ManagementRequest,
    ) -> Result<ManagementResponse, ManagementDispatchError> {
        let path =
            normalize_route_path(&request.path).ok_or(ManagementDispatchError::InvalidPath)?;
        let route = self
            .resources
            .get(&path)
            .ok_or(ManagementDispatchError::NotFound)?;
        request.method = "GET".to_owned();
        route
            .handler
            .handle_management(request)
            .await
            .map_err(|error| ManagementDispatchError::Plugin {
                plugin_id: route.plugin_id.clone(),
                message: error.to_string(),
            })
    }
}

fn normalize_route_path(path: &str) -> Option<String> {
    let path = path.trim();
    if path.is_empty()
        || !path.starts_with('/')
        || path.contains("//")
        || path.split('/').any(|part| matches!(part, "." | ".."))
        || path.contains(['?', '#', '\\'])
    {
        return None;
    }
    Some(path.trim_end_matches('/').to_owned().replace("//", "/"))
}

fn valid_method(method: &str) -> bool {
    matches!(method, "GET" | "POST" | "PUT" | "PATCH" | "DELETE")
}

fn invalid(record: &ManagementPluginRecord, route: String) -> ManagementRegistrationError {
    ManagementRegistrationError {
        plugin_id: record.plugin_id.clone(),
        route,
        reason: "management route is invalid".to_owned(),
    }
}

fn conflict(record: &ManagementPluginRecord, route: String) -> ManagementRegistrationError {
    ManagementRegistrationError {
        plugin_id: record.plugin_id.clone(),
        route,
        reason: "management route conflicts with a higher-priority plugin".to_owned(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagementRegistrationError {
    pub plugin_id: String,
    pub route: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagementDispatchError {
    InvalidPath,
    NotFound,
    Plugin { plugin_id: String, message: String },
}

impl std::fmt::Display for ManagementDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath => formatter.write_str("management path is invalid"),
            Self::NotFound => formatter.write_str("management route was not found"),
            Self::Plugin { plugin_id, message } => {
                write!(formatter, "management plugin {plugin_id} failed: {message}")
            }
        }
    }
}

impl std::error::Error for ManagementDispatchError {}
