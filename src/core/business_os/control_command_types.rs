// Origin: CTOX
// License: Apache-2.0

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

static ACTIVE_EXTERNAL_SQL_CONTROL_COMMANDS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub(super) struct ActiveExternalSqlControlCommand {
    command_id: String,
}

impl ActiveExternalSqlControlCommand {
    pub(super) fn try_acquire(command_id: &str) -> Option<Self> {
        let mut active = ACTIVE_EXTERNAL_SQL_CONTROL_COMMANDS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        active.insert(command_id.to_string()).then(|| Self {
            command_id: command_id.to_string(),
        })
    }
}

impl Drop for ActiveExternalSqlControlCommand {
    fn drop(&mut self) {
        ACTIVE_EXTERNAL_SQL_CONTROL_COMMANDS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.command_id);
    }
}

pub(super) struct ReportAccepted {
    pub(super) report_id: String,
    pub(super) command_id: String,
    pub(super) task_id: Option<String>,
    pub(super) task_status: Option<String>,
}
