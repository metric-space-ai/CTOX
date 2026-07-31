use crate::service::core_state_machine::{CoreEvent, CoreState};
use anyhow::{bail, Result};

/// Durable states stored in `ticket_cases.state`.
///
/// Writers emit only `as_str()` values. `parse()` also accepts the documented
/// historical/runtime aliases so persisted legacy rows keep their meaning.
///
/// Seven external readers intentionally remain on raw state values for the next
/// vocabulary wave:
/// - `src/core/context/live_context.rs:1709`
/// - `src/core/service/service.rs:4505`
/// - `src/core/mission/tickets/mod.rs:4584`
/// - `src/core/service/service.rs:4288`
/// - `src/apps/desktop/src/views/kanban.rs:294`
/// - `src/apps/desktop/src/db_reader.rs:1014` and `:1988`
/// - `src/core/service/process_mining.rs:5708`
///
/// These readers still consume or compare raw strings instead of this enum. In
/// particular, `state != "closed"` (and equivalent closed-only filters) does
/// not treat the closure aliases `done` and `completed` as closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TicketCaseState {
    Created,
    Open,
    Queued,
    Classified,
    Planned,
    Ready,
    Executable,
    Executing,
    ApprovalPending,
    AwaitingReview,
    ReworkRequired,
    AwaitingVerification,
    Verified,
    WritebackPending,
    Closed,
    Blocked,
    BlockedNeedsClarification,
}

impl TicketCaseState {
    #[cfg(test)]
    pub(super) const ALL: [Self; 17] = [
        Self::Created,
        Self::Open,
        Self::Queued,
        Self::Classified,
        Self::Planned,
        Self::Ready,
        Self::Executable,
        Self::Executing,
        Self::ApprovalPending,
        Self::AwaitingReview,
        Self::ReworkRequired,
        Self::AwaitingVerification,
        Self::Verified,
        Self::WritebackPending,
        Self::Closed,
        Self::Blocked,
        Self::BlockedNeedsClarification,
    ];

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Open => "open",
            Self::Queued => "queued",
            Self::Classified => "classified",
            Self::Planned => "planned",
            Self::Ready => "ready",
            Self::Executable => "executable",
            Self::Executing => "executing",
            Self::ApprovalPending => "approval_pending",
            Self::AwaitingReview => "awaiting_review",
            Self::ReworkRequired => "rework_required",
            Self::AwaitingVerification => "awaiting_verification",
            Self::Verified => "verified",
            Self::WritebackPending => "writeback_pending",
            Self::Closed => "closed",
            Self::Blocked => "blocked",
            Self::BlockedNeedsClarification => "blocked_needs_clarification",
        }
    }

    /// Parse the durable vocabulary and its read-only aliases.
    ///
    /// `in_progress` and `running` denote active execution. `review` and
    /// `reviewing` denote the review checkpoint. `rework` and `verification`
    /// are historical short spellings. `done` and `completed` denote closure.
    /// Unknown values fail closed instead of being reinterpreted as creation.
    pub(super) fn parse(raw: &str) -> Result<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        let state = match normalized.as_str() {
            "created" => Self::Created,
            "open" => Self::Open,
            "queued" => Self::Queued,
            "classified" => Self::Classified,
            "planned" => Self::Planned,
            "ready" => Self::Ready,
            "executable" => Self::Executable,
            "executing" | "in_progress" | "running" => Self::Executing,
            "approval_pending" => Self::ApprovalPending,
            "awaiting_review" | "review" | "reviewing" => Self::AwaitingReview,
            "rework_required" | "rework" => Self::ReworkRequired,
            "awaiting_verification" | "verification" => Self::AwaitingVerification,
            "verified" => Self::Verified,
            "writeback_pending" => Self::WritebackPending,
            "closed" | "done" | "completed" => Self::Closed,
            "blocked" => Self::Blocked,
            "blocked_needs_clarification" => Self::BlockedNeedsClarification,
            other => bail!("ticket case state is not recognized: {other}"),
        };
        Ok(state)
    }

    pub(super) const fn core_state(self) -> CoreState {
        match self {
            Self::Created | Self::Open | Self::Queued => CoreState::Created,
            Self::Classified => CoreState::Classified,
            Self::Planned | Self::Ready | Self::Executable => CoreState::Planned,
            Self::Executing => CoreState::Executing,
            Self::ApprovalPending | Self::AwaitingReview => CoreState::AwaitingReview,
            Self::ReworkRequired => CoreState::ReworkRequired,
            Self::AwaitingVerification => CoreState::AwaitingVerification,
            Self::Verified | Self::WritebackPending => CoreState::Verified,
            Self::Closed => CoreState::Closed,
            Self::Blocked | Self::BlockedNeedsClarification => CoreState::Blocked,
        }
    }

    pub(super) const fn core_event(self) -> CoreEvent {
        match self {
            Self::Created | Self::Open | Self::Queued => CoreEvent::CreateTicket,
            Self::Classified => CoreEvent::Classify,
            Self::Planned | Self::Ready | Self::Executable => CoreEvent::Plan,
            Self::Executing => CoreEvent::Execute,
            Self::ApprovalPending | Self::AwaitingReview => CoreEvent::RequestReview,
            Self::ReworkRequired => CoreEvent::RequireRework,
            Self::AwaitingVerification | Self::Verified | Self::WritebackPending => {
                CoreEvent::Verify
            }
            Self::Closed => CoreEvent::Close,
            Self::Blocked | Self::BlockedNeedsClarification => CoreEvent::Block,
        }
    }

    pub(super) const fn is_executable(self) -> bool {
        matches!(self, Self::Executable | Self::Executing)
    }

    pub(super) const fn is_blocked(self) -> bool {
        matches!(self, Self::Blocked | Self::BlockedNeedsClarification)
    }

    pub(super) const fn is_ready_for_writeback(self) -> bool {
        matches!(self, Self::WritebackPending)
    }

    /// States that no longer count as open work in workflow/status summaries.
    /// Verified cases are terminal-near; writeback-pending cases have passed
    /// verification and are intentionally excluded alongside closed cases.
    pub(super) const fn counts_as_open_work(self) -> bool {
        !matches!(self, Self::Verified | Self::WritebackPending | Self::Closed)
    }
}

/// Fail-closed adapter for ticket-case consumers outside the private tickets
/// module. Database readers validate states before returning views; treating a
/// future/corrupt value as not open also avoids duplicate-work creation.
pub(crate) fn counts_as_open_work(raw: &str) -> bool {
    TicketCaseState::parse(raw).is_ok_and(TicketCaseState::counts_as_open_work)
}
