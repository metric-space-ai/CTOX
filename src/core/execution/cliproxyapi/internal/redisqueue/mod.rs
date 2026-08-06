// Origin: CTOX
// License: AGPL-3.0-only

mod plugin;
mod queue;
mod usage_toggle;

pub use plugin::UsageQueuePlugin;
pub use queue::{Subscription, UsageQueue};
pub use usage_toggle::UsageStatisticsSwitch;

#[cfg(test)]
mod queue_test;

#[cfg(test)]
mod plugin_test;
