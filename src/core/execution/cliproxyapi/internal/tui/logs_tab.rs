// ref: internal/tui/logs_tab.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFilter {
    All,
    Info,
    Warn,
    Error,
    Debug,
}
#[derive(Debug, Clone)]
pub struct LogsTabModel {
    lines: VecDeque<String>,
    capacity: usize,
    filter: LogFilter,
    offset: usize,
    width: u16,
    height: u16,
    cursor: i64,
}
impl LogsTabModel {
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            filter: LogFilter::All,
            offset: 0,
            width: 0,
            height: 0,
            cursor: 0,
        }
    }
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
    pub fn append(&mut self, lines: impl IntoIterator<Item = String>, cursor: Option<i64>) {
        for line in lines {
            if self.lines.len() == self.capacity {
                self.lines.pop_front();
            }
            self.lines.push_back(line);
        }
        if let Some(cursor) = cursor {
            self.cursor = self.cursor.max(cursor);
        }
    }
    pub fn set_filter(&mut self, filter: LogFilter) {
        self.filter = filter;
        self.offset = 0;
    }
    pub fn scroll(&mut self, delta: isize) {
        self.offset = self.offset.saturating_add_signed(delta);
    }
    pub fn cursor(&self) -> i64 {
        self.cursor
    }
    pub fn render(&self) -> String {
        let visible = self
            .lines
            .iter()
            .filter(|line| self.matches(line))
            .collect::<Vec<_>>();
        let height = self.height.saturating_sub(1).max(1) as usize;
        let end = visible.len().saturating_sub(self.offset.min(visible.len()));
        let start = end.saturating_sub(height);
        let mut output = vec![format!("Logs [{:?}]", self.filter)];
        output.extend(
            visible[start..end]
                .iter()
                .map(|line| super::styles::clip_width(line.trim_end(), self.width as usize)),
        );
        output.join("\n")
    }
    fn matches(&self, line: &str) -> bool {
        let lower = line.to_ascii_lowercase();
        match self.filter {
            LogFilter::All => true,
            LogFilter::Info => lower.contains("[info ") || lower.contains("[info]"),
            LogFilter::Warn => lower.contains("[warn"),
            LogFilter::Error => lower.contains("[error"),
            LogFilter::Debug => lower.contains("[debug") || lower.contains("[trace"),
        }
    }
}
impl Default for LogsTabModel {
    fn default() -> Self {
        Self::new(2_000)
    }
}
