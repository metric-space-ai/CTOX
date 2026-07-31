// Origin: CTOX
// License: AGPL-3.0-only

/// Durable statuses written to `planned_steps.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlanStepStatus {
    Pending,
    Queued,
    Completed,
    Blocked,
    Failed,
}

impl PlanStepStatus {
    const NEXT_STEP_DISPLAY_VARIANTS: [Self; 3] = [Self::Pending, Self::Queued, Self::Blocked];
    const RUNNABLE_WORK_VARIANTS: [Self; 2] = [Self::Pending, Self::Queued];
    const DUE_WORK_VARIANTS: [Self; 1] = [Self::Pending];
    const LEGACY_ROUTING_MIGRATION_VARIANTS: [Self; 4] =
        [Self::Pending, Self::Completed, Self::Blocked, Self::Failed];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Queued => "queued",
            Self::Completed => "completed",
            Self::Blocked => "blocked",
            Self::Failed => "failed",
        }
    }

    /// Parse only the durable step vocabulary. Unknown values stay unknown so
    /// callers cannot accidentally treat a new or corrupted state as active.
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw {
            "pending" => Some(Self::Pending),
            "queued" => Some(Self::Queued),
            "completed" => Some(Self::Completed),
            "blocked" => Some(Self::Blocked),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }

    /// Statuses shown as the next step in plan list/load projections.
    pub(crate) const fn is_next_step_display(self) -> bool {
        matches!(self, Self::Pending | Self::Queued | Self::Blocked)
    }

    /// Statuses counted by the mission idle watchdog as runnable work.
    pub(crate) const fn is_runnable_work(self) -> bool {
        matches!(self, Self::Pending | Self::Queued)
    }

    /// Statuses eligible for due-work emission after time gates are applied.
    pub(crate) const fn is_due_work(self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Non-queued statuses whose legacy message routing must be settled.
    pub(crate) const fn is_legacy_routing_migration(self) -> bool {
        matches!(
            self,
            Self::Pending | Self::Completed | Self::Blocked | Self::Failed
        )
    }

    pub(crate) const fn next_step_display_variants() -> &'static [Self] {
        &Self::NEXT_STEP_DISPLAY_VARIANTS
    }

    pub(crate) const fn runnable_work_variants() -> &'static [Self] {
        &Self::RUNNABLE_WORK_VARIANTS
    }

    pub(crate) const fn due_work_variants() -> &'static [Self] {
        &Self::DUE_WORK_VARIANTS
    }

    pub(crate) const fn legacy_routing_migration_variants() -> &'static [Self] {
        &Self::LEGACY_ROUTING_MIGRATION_VARIANTS
    }
}

/// Durable statuses written to `planned_goals.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlanGoalStatus {
    Active,
    Completed,
    Blocked,
    Failed,
    Superseded,
}

impl PlanGoalStatus {
    const TERMINAL_VARIANTS: [Self; 3] = [Self::Completed, Self::Failed, Self::Superseded];

    // Include accepted legacy spellings so raw SQL readers classify old rows
    // the same way as `parse`. Writers still emit only `as_str()` values.
    const TERMINAL_READ_VALUES: [&'static str; 5] =
        ["completed", "failed", "superseded", "closed", "cancelled"];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Blocked => "blocked",
            Self::Failed => "failed",
            Self::Superseded => "superseded",
        }
    }

    /// Parse the canonical goal vocabulary plus read-only legacy aliases.
    ///
    /// `closed` means the goal reached a normal terminal closure and therefore
    /// maps to `Completed`. `cancelled` means the goal was withdrawn without a
    /// successful result; `Superseded` is the matching non-failure terminal
    /// variant because it also means that this goal is no longer the live work
    /// slice. Neither alias is ever returned by `as_str()`.
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw {
            "active" => Some(Self::Active),
            "completed" | "closed" => Some(Self::Completed),
            "blocked" => Some(Self::Blocked),
            "failed" => Some(Self::Failed),
            "superseded" | "cancelled" => Some(Self::Superseded),
            _ => None,
        }
    }

    pub(crate) const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Superseded)
    }

    /// Goal states that must not emit another step. `Blocked` is deliberately
    /// included even though it is not terminal.
    pub(crate) const fn prevents_step_emission(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Blocked | Self::Failed | Self::Superseded
        )
    }

    pub(crate) const fn terminal_variants() -> &'static [Self] {
        &Self::TERMINAL_VARIANTS
    }

    /// All raw database spellings recognized as terminal on reads, including
    /// the documented aliases accepted by `parse`.
    pub(crate) fn terminal_read_values() -> &'static [&'static str] {
        debug_assert!(Self::terminal_variants()
            .iter()
            .copied()
            .all(Self::is_terminal));
        &Self::TERMINAL_READ_VALUES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_status_parse_is_fail_closed_and_writes_are_canonical() {
        for status in [
            PlanStepStatus::Pending,
            PlanStepStatus::Queued,
            PlanStepStatus::Completed,
            PlanStepStatus::Blocked,
            PlanStepStatus::Failed,
        ] {
            assert_eq!(PlanStepStatus::parse(status.as_str()), Some(status));
        }
        assert_eq!(PlanStepStatus::parse("cancelled"), None);
        assert_eq!(PlanStepStatus::parse(" pending"), None);
        assert_eq!(PlanStepStatus::parse("future-status"), None);
    }

    #[test]
    fn step_status_sets_preserve_the_four_existing_divergences() {
        let statuses = [
            PlanStepStatus::Pending,
            PlanStepStatus::Queued,
            PlanStepStatus::Completed,
            PlanStepStatus::Blocked,
            PlanStepStatus::Failed,
        ];
        let matching = |predicate: fn(PlanStepStatus) -> bool| {
            statuses
                .into_iter()
                .filter(|status| predicate(*status))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            matching(PlanStepStatus::is_next_step_display),
            PlanStepStatus::next_step_display_variants()
        );
        assert_eq!(
            matching(PlanStepStatus::is_runnable_work),
            PlanStepStatus::runnable_work_variants()
        );
        assert_eq!(
            matching(PlanStepStatus::is_due_work),
            PlanStepStatus::due_work_variants()
        );
        assert_eq!(
            matching(PlanStepStatus::is_legacy_routing_migration),
            PlanStepStatus::legacy_routing_migration_variants()
        );
    }

    #[test]
    fn goal_status_aliases_are_read_only_and_terminal() {
        assert_eq!(
            PlanGoalStatus::parse("closed"),
            Some(PlanGoalStatus::Completed)
        );
        assert_eq!(
            PlanGoalStatus::parse("cancelled"),
            Some(PlanGoalStatus::Superseded)
        );
        assert_eq!(PlanGoalStatus::parse("future-status"), None);
        assert_eq!(PlanGoalStatus::Completed.as_str(), "completed");
        assert_eq!(PlanGoalStatus::Superseded.as_str(), "superseded");
        assert!(PlanGoalStatus::terminal_variants()
            .iter()
            .copied()
            .all(PlanGoalStatus::is_terminal));
        assert!(PlanGoalStatus::terminal_read_values().iter().all(|raw| {
            PlanGoalStatus::parse(raw)
                .map(PlanGoalStatus::is_terminal)
                .unwrap_or(false)
        }));
        assert!(!PlanGoalStatus::Blocked.is_terminal());
        assert!(PlanGoalStatus::Blocked.prevents_step_emission());
    }
}
