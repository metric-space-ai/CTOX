#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkItemStatus {
    Created,
    Open,
    Queued,
    Restored,
    Publishing,
    Published,
    Executing,
    AwaitingReview,
    ReworkRequired,
    AwaitingVerification,
    Verified,
    Blocked,
    Spilled,
    Failed,
    Closed,
    Handled,
    Cancelled,
    Superseded,
    Ready,
    Waiting,
    Satisfied,
}

impl WorkItemStatus {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 21] = [
        Self::Created,
        Self::Open,
        Self::Queued,
        Self::Restored,
        Self::Publishing,
        Self::Published,
        Self::Executing,
        Self::AwaitingReview,
        Self::ReworkRequired,
        Self::AwaitingVerification,
        Self::Verified,
        Self::Blocked,
        Self::Spilled,
        Self::Failed,
        Self::Closed,
        Self::Handled,
        Self::Cancelled,
        Self::Superseded,
        Self::Ready,
        Self::Waiting,
        Self::Satisfied,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Open => "open",
            Self::Queued => "queued",
            Self::Restored => "restored",
            Self::Publishing => "publishing",
            Self::Published => "published",
            Self::Executing => "executing",
            Self::AwaitingReview => "awaiting_review",
            Self::ReworkRequired => "rework_required",
            Self::AwaitingVerification => "awaiting_verification",
            Self::Verified => "verified",
            Self::Blocked => "blocked",
            Self::Spilled => "spilled",
            Self::Failed => "failed",
            Self::Closed => "closed",
            Self::Handled => "handled",
            Self::Cancelled => "cancelled",
            Self::Superseded => "superseded",
            Self::Ready => "ready",
            Self::Waiting => "waiting",
            Self::Satisfied => "satisfied",
        }
    }

    pub(crate) fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "created" => Some(Self::Created),
            "open" => Some(Self::Open),
            "queued" => Some(Self::Queued),
            "restored" => Some(Self::Restored),
            "publishing" => Some(Self::Publishing),
            "published" => Some(Self::Published),
            "executing" => Some(Self::Executing),
            "awaiting_review" => Some(Self::AwaitingReview),
            "rework_required" => Some(Self::ReworkRequired),
            "awaiting_verification" => Some(Self::AwaitingVerification),
            "verified" => Some(Self::Verified),
            "blocked" => Some(Self::Blocked),
            "spilled" => Some(Self::Spilled),
            "failed" => Some(Self::Failed),
            "closed" => Some(Self::Closed),
            "handled" => Some(Self::Handled),
            "cancelled" => Some(Self::Cancelled),
            "superseded" => Some(Self::Superseded),
            "ready" => Some(Self::Ready),
            "waiting" => Some(Self::Waiting),
            "satisfied" => Some(Self::Satisfied),

            // Historical blank state rows were treated as newly created work.
            "" => Some(Self::Created),
            // Runtime spellings all denote active execution.
            "running" | "in_progress" => Some(Self::Executing),
            // Review spellings all denote the review checkpoint.
            "review" | "reviewing" => Some(Self::AwaitingReview),
            // Rework spellings all denote a rejected review checkpoint.
            "review_rework" | "rework" => Some(Self::ReworkRequired),
            // The short noun spelling denotes waiting for verification.
            "verification" => Some(Self::AwaitingVerification),
            // Historical completion spellings denote the closed state. Handled
            // remains distinct because existing workflow predicates treat it
            // differently from closed/done/completed.
            "done" | "completed" => Some(Self::Closed),
            // A passed workflow gate denotes a satisfied step status.
            "passed" => Some(Self::Satisfied),
            _ => None,
        }
    }

    pub(crate) const fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Published | Self::Executing)
    }

    pub(crate) const fn is_workflow_item_satisfied(self) -> bool {
        matches!(self, Self::Closed | Self::Handled | Self::Verified)
    }

    pub(crate) const fn is_workflow_status_satisfied(self) -> bool {
        matches!(self, Self::Verified | Self::Satisfied | Self::Closed)
    }

    pub(crate) const fn is_workflow_runnable(self) -> bool {
        matches!(
            self,
            Self::Created | Self::Open | Self::Blocked | Self::Restored
        )
    }

    pub(crate) const fn is_workflow_case_terminal(self) -> bool {
        matches!(
            self,
            Self::Closed | Self::Handled | Self::Cancelled | Self::Superseded | Self::Failed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::WorkItemStatus;

    const ALIASES: [(&str, WorkItemStatus); 11] = [
        ("", WorkItemStatus::Created),
        ("running", WorkItemStatus::Executing),
        ("in_progress", WorkItemStatus::Executing),
        ("review", WorkItemStatus::AwaitingReview),
        ("reviewing", WorkItemStatus::AwaitingReview),
        ("review_rework", WorkItemStatus::ReworkRequired),
        ("rework", WorkItemStatus::ReworkRequired),
        ("verification", WorkItemStatus::AwaitingVerification),
        ("done", WorkItemStatus::Closed),
        ("completed", WorkItemStatus::Closed),
        ("passed", WorkItemStatus::Satisfied),
    ];

    const ACCEPTED_SPELLINGS: [&str; 32] = [
        "created",
        "",
        "open",
        "queued",
        "restored",
        "publishing",
        "published",
        "executing",
        "running",
        "in_progress",
        "awaiting_review",
        "review",
        "reviewing",
        "rework_required",
        "review_rework",
        "rework",
        "awaiting_verification",
        "verification",
        "verified",
        "blocked",
        "spilled",
        "failed",
        "closed",
        "done",
        "completed",
        "handled",
        "cancelled",
        "superseded",
        "ready",
        "waiting",
        "satisfied",
        "passed",
    ];

    fn matching_spellings(predicate: impl Fn(WorkItemStatus) -> bool) -> Vec<&'static str> {
        ACCEPTED_SPELLINGS
            .iter()
            .copied()
            .filter(|raw| predicate(WorkItemStatus::parse(raw).expect("accepted status")))
            .collect()
    }

    #[test]
    fn canonical_values_roundtrip() {
        for status in WorkItemStatus::ALL {
            assert_eq!(WorkItemStatus::parse(status.as_str()), Some(status));
        }
    }

    #[test]
    fn aliases_map_to_documented_variants() {
        for (raw, expected) in ALIASES {
            assert_eq!(WorkItemStatus::parse(raw), Some(expected), "alias {raw:?}");
        }
    }

    #[test]
    fn parsing_normalizes_case_and_surrounding_whitespace() {
        assert_eq!(
            WorkItemStatus::parse(" Executing "),
            Some(WorkItemStatus::Executing)
        );
    }

    #[test]
    fn unknown_status_fails_closed() {
        assert_eq!(WorkItemStatus::parse("mystery"), None);
        assert_eq!(WorkItemStatus::parse("in-progress"), None);
    }

    #[test]
    fn active_predicate_pins_existing_set() {
        assert_eq!(
            matching_spellings(WorkItemStatus::is_active),
            ["queued", "published", "executing", "running", "in_progress"]
        );
    }

    #[test]
    fn workflow_item_satisfied_predicate_pins_existing_set() {
        assert_eq!(
            matching_spellings(WorkItemStatus::is_workflow_item_satisfied),
            ["verified", "closed", "done", "completed", "handled"]
        );
    }

    #[test]
    fn workflow_status_satisfied_predicate_pins_existing_set() {
        assert_eq!(
            matching_spellings(WorkItemStatus::is_workflow_status_satisfied),
            [
                "verified",
                "closed",
                "done",
                "completed",
                "satisfied",
                "passed"
            ]
        );
    }

    #[test]
    fn workflow_runnable_predicate_pins_existing_set() {
        assert_eq!(
            matching_spellings(WorkItemStatus::is_workflow_runnable),
            ["created", "", "open", "restored", "blocked"]
        );
    }

    #[test]
    fn workflow_case_terminal_predicate_pins_existing_set() {
        assert_eq!(
            matching_spellings(WorkItemStatus::is_workflow_case_terminal),
            [
                "failed",
                "closed",
                "done",
                "completed",
                "handled",
                "cancelled",
                "superseded"
            ]
        );
    }
}
