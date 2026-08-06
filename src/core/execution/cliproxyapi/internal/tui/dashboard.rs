// ref: internal/tui/dashboard.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DashboardSnapshot {
    pub config: Value,
    pub auth_files: Vec<Value>,
    pub api_keys: Vec<String>,
}
#[derive(Debug, Clone, Default)]
pub struct DashboardModel {
    width: u16,
    height: u16,
    snapshot: DashboardSnapshot,
    error: Option<String>,
}
impl DashboardModel {
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
    pub fn apply(&mut self, snapshot: DashboardSnapshot) {
        self.snapshot = snapshot;
        self.error = None;
    }
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }
    pub fn render(&self) -> String {
        if let Some(error) = &self.error {
            return format!("Dashboard error: {error}");
        }
        let debug = self
            .snapshot
            .config
            .get("debug")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let request_log = self
            .snapshot
            .config
            .get("request-log")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let body = format!(
            "Dashboard\n  API keys: {}\n  Auth files: {}\n  Debug: {}\n  Request logs: {}",
            self.snapshot.api_keys.len(),
            self.snapshot.auth_files.len(),
            bool_icon(debug),
            bool_icon(request_log)
        );
        body.lines()
            .take(self.height.max(1) as usize)
            .map(|line| super::styles::clip_width(line, self.width as usize))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
fn bool_icon(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}
pub fn format_large_number(number: i64) -> String {
    let absolute = number.unsigned_abs();
    if absolute >= 1_000_000 {
        format!("{:.1}M", number as f64 / 1_000_000.0)
    } else if absolute >= 1_000 {
        format!("{:.1}K", number as f64 / 1_000.0)
    } else {
        number.to_string()
    }
}
