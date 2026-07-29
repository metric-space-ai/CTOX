#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueueRouteStatus {
    Pending,
    Leased,
    Running,
    Blocked,
    ReviewRework,
    Failed,
    Handled,
    Cancelled,
}

impl QueueRouteStatus {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 8] = [
        Self::Pending,
        Self::Leased,
        Self::Running,
        Self::Blocked,
        Self::ReviewRework,
        Self::Failed,
        Self::Handled,
        Self::Cancelled,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Leased => "leased",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::ReviewRework => "review_rework",
            Self::Failed => "failed",
            Self::Handled => "handled",
            Self::Cancelled => "cancelled",
        }
    }

    pub(crate) fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "pending" => Some(Self::Pending),
            "leased" => Some(Self::Leased),
            "running" => Some(Self::Running),
            "blocked" => Some(Self::Blocked),
            "review_rework" => Some(Self::ReviewRework),
            "failed" => Some(Self::Failed),
            "handled" => Some(Self::Handled),
            "cancelled" => Some(Self::Cancelled),

            // Historical blank rows were interpreted as pending work.
            "" => Some(Self::Pending),
            // Legacy approval-nag completion remained blocked for routing.
            "approval-nag-handled" => Some(Self::Blocked),
            // Legacy completion spelling is the canonical handled state.
            "completed" => Some(Self::Handled),
            // Legacy replacement spelling is the canonical cancelled state.
            "superseded" => Some(Self::Cancelled),
            // Legacy duplicate rows were readable blocked transition sources.
            "duplicate" => Some(Self::Blocked),
            // Legacy sender-policy rows were readable blocked transition sources.
            "blocked_sender" => Some(Self::Blocked),
            // Legacy meeting rows were readable blocked transition sources.
            "meeting_scheduled" => Some(Self::Blocked),
            _ => None,
        }
    }

    pub(crate) const fn is_terminal(self) -> bool {
        matches!(self, Self::Handled | Self::Cancelled | Self::Failed)
    }

    pub(crate) const fn is_pending(self) -> bool {
        matches!(self, Self::Pending)
    }
}
