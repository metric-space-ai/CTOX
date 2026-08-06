// ref: internal/tui/auth_tab.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
#[derive(Debug, Clone, Default)]
pub struct AuthTabModel {
    files: Vec<Value>,
    cursor: usize,
    width: u16,
    height: u16,
    confirm_delete: bool,
    status: String,
}
impl AuthTabModel {
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
    pub fn apply_files(&mut self, files: Vec<Value>) {
        self.files = files;
        self.cursor = self.cursor.min(self.files.len().saturating_sub(1));
        self.confirm_delete = false;
    }
    pub fn move_cursor(&mut self, delta: isize) {
        self.cursor = self
            .cursor
            .saturating_add_signed(delta)
            .min(self.files.len().saturating_sub(1));
    }
    pub fn selected_name(&self) -> Option<&str> {
        self.files
            .get(self.cursor)?
            .get("name")
            .or_else(|| self.files.get(self.cursor)?.get("file"))?
            .as_str()
    }
    pub fn request_delete(&mut self) {
        self.confirm_delete = self.selected_name().is_some();
    }
    pub fn cancel_delete(&mut self) {
        self.confirm_delete = false;
    }
    pub fn confirm_delete(&mut self) -> Option<String> {
        if !self.confirm_delete {
            return None;
        }
        self.confirm_delete = false;
        self.selected_name().map(str::to_owned)
    }
    pub fn toggle_selected(&mut self) -> Option<(String, bool)> {
        let file = self.files.get_mut(self.cursor)?;
        let name = file
            .get("name")
            .or_else(|| file.get("file"))?
            .as_str()?
            .to_owned();
        let disabled = !file
            .get("disabled")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        file["disabled"] = Value::Bool(disabled);
        Some((name, disabled))
    }
    pub fn render(&self) -> String {
        let mut lines = vec!["Auth Files".to_owned()];
        let max = self.height.saturating_sub(2).max(1) as usize;
        for (index, file) in self.files.iter().enumerate().take(max) {
            let marker = if index == self.cursor { ">" } else { " " };
            let name = file
                .get("name")
                .or_else(|| file.get("file"))
                .and_then(Value::as_str)
                .unwrap_or("unnamed");
            let provider = file
                .get("type")
                .or_else(|| file.get("provider"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let disabled = file
                .get("disabled")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            lines.push(super::styles::clip_width(
                &format!(
                    "{marker} {name} [{provider}] {}",
                    if disabled { "disabled" } else { "active" }
                ),
                self.width as usize,
            ));
        }
        if self.confirm_delete {
            lines.push("Delete selected file? [y/N]".into());
        }
        if !self.status.is_empty() {
            lines.push(self.status.clone());
        }
        lines.join("\n")
    }
}
pub fn get_any_string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
        .unwrap_or_default()
}
