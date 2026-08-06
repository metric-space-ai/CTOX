// ref: internal/home/plugin_status.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::time::{Duration, Instant, SystemTime};

use super::client::Client;
use crate::internal::homeplugins::SyncReport;
pub const PLUGIN_STATUS_REPORT_TIMEOUT: Duration = Duration::from_secs(10);

pub trait PluginStatusSink: Send + Sync {
    fn push_plugin_status(
        &self,
        payload: &[u8],
        deadline: Instant,
    ) -> Result<(), PluginStatusError>;
}

impl PluginStatusSink for Client {
    fn push_plugin_status(
        &self,
        payload: &[u8],
        deadline: Instant,
    ) -> Result<(), PluginStatusError> {
        if Instant::now() >= deadline {
            return Err(PluginStatusError(
                "home plugin status deadline exceeded".into(),
            ));
        }
        Client::push_plugin_status(self, payload)
            .map_err(|error| PluginStatusError(error.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginStatusError(pub String);
impl fmt::Display for PluginStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PluginStatusError {}

pub fn report_plugin_status(
    client: &dyn PluginStatusSink,
    node_id: &str,
    mut report: SyncReport,
) -> Result<(), PluginStatusError> {
    let node_id = node_id.trim();
    if node_id.is_empty() {
        return Err(PluginStatusError(
            "home plugin status node id is empty".into(),
        ));
    }
    report.node_id = node_id.into();
    report.updated_at = SystemTime::now().into();
    let raw = serde_json::to_vec(&report).map_err(|e| PluginStatusError(e.to_string()))?;
    client.push_plugin_status(&raw, Instant::now() + PLUGIN_STATUS_REPORT_TIMEOUT)
}
