// ref: internal/tui/styles.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Normal,
    Muted,
    Accent,
    Success,
    Warning,
    Error,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    pub tone: Tone,
    pub bold: bool,
}
impl Style {
    pub const fn new(tone: Tone, bold: bool) -> Self {
        Self { tone, bold }
    }
}
pub const TITLE: Style = Style::new(Tone::Accent, true);
pub const MUTED: Style = Style::new(Tone::Muted, false);
pub const SELECTED: Style = Style::new(Tone::Accent, true);
pub fn log_level_style(level: &str) -> Style {
    match level.trim().to_ascii_lowercase().as_str() {
        "error" | "fatal" => Style::new(Tone::Error, false),
        "warn" | "warning" => Style::new(Tone::Warning, false),
        "debug" | "trace" => MUTED,
        _ => Style::new(Tone::Normal, false),
    }
}
pub fn clip_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= width {
        text.to_owned()
    } else if width == 1 {
        "…".into()
    } else {
        format!("{}…", chars[..width - 1].iter().collect::<String>())
    }
}
