// ref: internal/tui/i18n.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locale {
    English,
    Chinese,
}
#[derive(Debug, Clone)]
pub struct I18n {
    locale: Locale,
    english: BTreeMap<&'static str, &'static str>,
    chinese: BTreeMap<&'static str, &'static str>,
}
impl Default for I18n {
    fn default() -> Self {
        Self {
            locale: Locale::English,
            english: BTreeMap::from([
                ("dashboard", "Dashboard"),
                ("config", "Configuration"),
                ("auth", "Auth Files"),
                ("keys", "API Keys"),
                ("oauth", "OAuth"),
                ("logs", "Logs"),
                ("quit", "Quit"),
            ]),
            chinese: BTreeMap::from([
                ("dashboard", "仪表盘"),
                ("config", "配置"),
                ("auth", "认证文件"),
                ("keys", "API 密钥"),
                ("oauth", "OAuth"),
                ("logs", "日志"),
                ("quit", "退出"),
            ]),
        }
    }
}
impl I18n {
    pub fn locale(&self) -> Locale {
        self.locale
    }
    pub fn set_locale(&mut self, locale: Locale) {
        self.locale = locale;
    }
    pub fn toggle(&mut self) {
        self.locale = if self.locale == Locale::English {
            Locale::Chinese
        } else {
            Locale::English
        };
    }
    pub fn text<'a>(&'a self, key: &'a str) -> &'a str {
        let map = if self.locale == Locale::English {
            &self.english
        } else {
            &self.chinese
        };
        map.get(key).copied().unwrap_or(key)
    }
    pub fn tab_names(&self) -> Vec<&str> {
        ["dashboard", "config", "auth", "keys", "oauth", "logs"]
            .iter()
            .map(|key| self.text(key))
            .collect()
    }
}
