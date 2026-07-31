use crate::core_state::{CoreEvent, CoreState};
use anyhow::{bail, Result};

/// Durable statuses stored in `ticket_event_routing_state.route_status`.
///
/// Parsing is fail-closed so a future or corrupt persisted value cannot be
/// reinterpreted as pending work by either core mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TicketEventRouteStatus {
    Pending,
    Leased,
    Observed,
    Handled,
    Failed,
    Duplicate,
    Blocked,
}

impl TicketEventRouteStatus {
    #[cfg(test)]
    pub(super) const ALL: [Self; 7] = [
        Self::Pending,
        Self::Leased,
        Self::Observed,
        Self::Handled,
        Self::Failed,
        Self::Duplicate,
        Self::Blocked,
    ];

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Leased => "leased",
            Self::Observed => "observed",
            Self::Handled => "handled",
            Self::Failed => "failed",
            Self::Duplicate => "duplicate",
            Self::Blocked => "blocked",
        }
    }

    pub(super) fn parse(raw: &str) -> Result<Self> {
        let status = match raw.trim() {
            "pending" => Self::Pending,
            "leased" => Self::Leased,
            "observed" => Self::Observed,
            "handled" => Self::Handled,
            "failed" => Self::Failed,
            "duplicate" => Self::Duplicate,
            "blocked" => Self::Blocked,
            other => bail!("unsupported ticket event route status: {other}"),
        };
        Ok(status)
    }

    pub(super) const fn core_state(self) -> CoreState {
        match self {
            Self::Pending => CoreState::Pending,
            Self::Leased => CoreState::Leased,
            Self::Observed | Self::Handled => CoreState::Completed,
            Self::Failed => CoreState::Failed,
            Self::Duplicate => CoreState::Superseded,
            Self::Blocked => CoreState::Blocked,
        }
    }

    pub(super) const fn core_event(self) -> CoreEvent {
        match self {
            Self::Pending => CoreEvent::Release,
            Self::Leased => CoreEvent::Lease,
            Self::Observed | Self::Handled => CoreEvent::Complete,
            Self::Failed => CoreEvent::Fail,
            Self::Duplicate => CoreEvent::Supersede,
            Self::Blocked => CoreEvent::Block,
        }
    }
}
