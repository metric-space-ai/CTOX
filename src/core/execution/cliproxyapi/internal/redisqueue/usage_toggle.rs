// ref: internal/redisqueue/usage_toggle.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicBool, Ordering};

/// Instance-owned replacement for upstream's package-global usage switch.
pub struct UsageStatisticsSwitch {
    enabled: AtomicBool,
}

impl Default for UsageStatisticsSwitch {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(true),
        }
    }
}

impl UsageStatisticsSwitch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
}
