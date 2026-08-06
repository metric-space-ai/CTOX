// ref: internal/tui/keys_tab.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeysSnapshot {
    pub api_keys: Vec<String>,
    pub provider_keys: BTreeMap<String, Vec<Value>>,
}
#[derive(Debug, Clone, Default)]
pub struct KeysTabModel {
    snapshot: KeysSnapshot,
    cursor: usize,
    width: u16,
    height: u16,
    reveal: bool,
}
impl KeysTabModel {
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
    pub fn apply(&mut self, snapshot: KeysSnapshot) {
        self.snapshot = snapshot;
        self.cursor = self
            .cursor
            .min(self.snapshot.api_keys.len().saturating_sub(1));
    }
    pub fn move_cursor(&mut self, delta: isize) {
        self.cursor = self
            .cursor
            .saturating_add_signed(delta)
            .min(self.snapshot.api_keys.len().saturating_sub(1));
    }
    pub fn toggle_reveal(&mut self) {
        self.reveal = !self.reveal;
    }
    pub fn selected_key(&self) -> Option<&str> {
        self.snapshot.api_keys.get(self.cursor).map(String::as_str)
    }
    pub fn render(&self) -> String {
        let mut lines = vec![format!("API Keys ({})", self.snapshot.api_keys.len())];
        for (index, key) in self
            .snapshot
            .api_keys
            .iter()
            .enumerate()
            .take(self.height.saturating_sub(2) as usize)
        {
            let marker = if index == self.cursor { ">" } else { " " };
            lines.push(super::styles::clip_width(
                &format!(
                    "{marker} {}",
                    if self.reveal {
                        key.clone()
                    } else {
                        mask_key(key)
                    }
                ),
                self.width as usize,
            ));
        }
        for (provider, keys) in &self.snapshot.provider_keys {
            lines.push(format!("{provider}: {}", keys.len()));
        }
        lines.join("\n")
    }
}
pub fn mask_key(key: &str) -> String {
    let chars = key.chars().collect::<Vec<_>>();
    if chars.len() <= 8 {
        "*".repeat(chars.len().max(1))
    } else {
        format!(
            "{}…{}",
            chars[..4].iter().collect::<String>(),
            chars[chars.len() - 4..].iter().collect::<String>()
        )
    }
}
