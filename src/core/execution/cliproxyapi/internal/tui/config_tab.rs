// ref: internal/tui/config_tab.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigField {
    pub path: String,
    pub value: Value,
    pub editable: bool,
}
#[derive(Debug, Clone, Default)]
pub struct ConfigTabModel {
    fields: Vec<ConfigField>,
    cursor: usize,
    editing: Option<String>,
    width: u16,
    height: u16,
    status: String,
}
impl ConfigTabModel {
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.ensure_visible();
    }
    pub fn apply_config(&mut self, config: &Value) {
        self.fields.clear();
        flatten("", config, &mut self.fields);
        self.cursor = self.cursor.min(self.fields.len().saturating_sub(1));
    }
    pub fn move_cursor(&mut self, delta: isize) {
        self.cursor = self
            .cursor
            .saturating_add_signed(delta)
            .min(self.fields.len().saturating_sub(1));
    }
    pub fn begin_edit(&mut self) {
        if self
            .fields
            .get(self.cursor)
            .is_some_and(|field| field.editable)
        {
            self.editing = self
                .fields
                .get(self.cursor)
                .map(|field| display_value(&field.value));
        }
    }
    pub fn edit_push(&mut self, character: char) {
        if let Some(value) = &mut self.editing {
            value.push(character);
        }
    }
    pub fn edit_backspace(&mut self) {
        if let Some(value) = &mut self.editing {
            value.pop();
        }
    }
    pub fn cancel_edit(&mut self) {
        self.editing = None;
    }
    pub fn commit_edit(&mut self) -> Option<(String, Value)> {
        let raw = self.editing.take()?;
        let field = self.fields.get_mut(self.cursor)?;
        let parsed = match field.value {
            Value::Bool(_) => Value::Bool(matches!(raw.trim(), "true" | "1" | "yes" | "on")),
            Value::Number(_) => serde_json::from_str(&raw).ok()?,
            _ => Value::String(raw),
        };
        field.value = parsed.clone();
        Some((field.path.clone(), parsed))
    }
    pub fn toggle_bool(&mut self) -> Option<(String, Value)> {
        let field = self.fields.get_mut(self.cursor)?;
        let Value::Bool(value) = &mut field.value else {
            return None;
        };
        *value = !*value;
        Some((field.path.clone(), Value::Bool(*value)))
    }
    pub fn render(&self) -> String {
        if self.fields.is_empty() {
            return "Configuration\n  no fields".into();
        }
        let available = self.height.saturating_sub(2).max(1) as usize;
        let start = self.cursor.saturating_sub(available - 1);
        let mut lines = vec!["Configuration".to_owned()];
        for (index, field) in self.fields.iter().enumerate().skip(start).take(available) {
            let marker = if index == self.cursor { ">" } else { " " };
            let stored = display_value(&field.value);
            let displayed = if index == self.cursor {
                self.editing.as_deref().unwrap_or(&stored)
            } else {
                &stored
            };
            lines.push(super::styles::clip_width(
                &format!("{marker} {} = {displayed}", field.path),
                self.width as usize,
            ));
        }
        if !self.status.is_empty() {
            lines.push(self.status.clone());
        }
        lines.join("\n")
    }
    fn ensure_visible(&mut self) {
        self.cursor = self.cursor.min(self.fields.len().saturating_sub(1));
    }
}
fn flatten(prefix: &str, value: &Value, output: &mut Vec<ConfigField>) {
    if let Some(object) = value.as_object() {
        for (key, value) in object {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            if value.is_object() {
                flatten(&path, value, output);
            } else {
                output.push(ConfigField {
                    path,
                    value: value.clone(),
                    editable: matches!(value, Value::Bool(_) | Value::Number(_) | Value::String(_)),
                });
            }
        }
    }
}
fn display_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}
