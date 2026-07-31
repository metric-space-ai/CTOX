use super::case_state::TicketCaseState;
use anyhow::{bail, Result};

/// Control mode governing whether and how a ticket case may execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ControlApprovalMode {
    DryRunOnly,
    HumanApprovalRequired,
    BoundedAutoExecute,
    DirectExecuteAllowed,
}

impl ControlApprovalMode {
    #[cfg(test)]
    pub(super) const ALL: [Self; 4] = [
        Self::DryRunOnly,
        Self::HumanApprovalRequired,
        Self::BoundedAutoExecute,
        Self::DirectExecuteAllowed,
    ];

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::DryRunOnly => "dry_run_only",
            Self::HumanApprovalRequired => "human_approval_required",
            Self::BoundedAutoExecute => "bounded_auto_execute",
            Self::DirectExecuteAllowed => "direct_execute_allowed",
        }
    }

    /// Parse the durable control vocabulary without assigning unknown values a
    /// permissive or implicit approval behavior.
    pub(super) fn parse(raw: &str) -> Result<Self> {
        let mode = match raw.trim() {
            "dry_run_only" => Self::DryRunOnly,
            "human_approval_required" => Self::HumanApprovalRequired,
            "bounded_auto_execute" => Self::BoundedAutoExecute,
            "direct_execute_allowed" => Self::DirectExecuteAllowed,
            other => bail!("unsupported approval mode: {other}"),
        };
        Ok(mode)
    }

    pub(super) const fn rank(self) -> u8 {
        match self {
            Self::DryRunOnly => 0,
            Self::HumanApprovalRequired => 1,
            Self::BoundedAutoExecute => 2,
            Self::DirectExecuteAllowed => 3,
        }
    }

    pub(super) fn missing_approvals(self) -> Vec<String> {
        match self {
            Self::DryRunOnly => vec!["execution is disabled for this bundle".to_string()],
            Self::HumanApprovalRequired => vec!["owner or designated approver".to_string()],
            Self::BoundedAutoExecute | Self::DirectExecuteAllowed => Vec::new(),
        }
    }

    pub(super) const fn initial_case_state(self) -> TicketCaseState {
        match self {
            Self::DryRunOnly => TicketCaseState::Blocked,
            Self::HumanApprovalRequired => TicketCaseState::ApprovalPending,
            Self::BoundedAutoExecute | Self::DirectExecuteAllowed => TicketCaseState::Executable,
        }
    }
}
