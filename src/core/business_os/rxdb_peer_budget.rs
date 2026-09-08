// Origin: CTOX
// License: Apache-2.0

use std::time::Duration;

/// Reconciliation is background work. Count and byte ceilings bound each
/// writer lease; the wall budget ends a record slice between committed pages.
/// Never cancel a write future: a cursor is advanced only after its commit.
#[derive(Clone, Copy, Debug)]
pub(in crate::business_os) struct PeerProjectionBudget {
    pub documents_per_page: usize,
    pub bytes_per_page: usize,
    pub slice_duration: Duration,
    pub source_poll_interval: Duration,
}

impl PeerProjectionBudget {
    pub const DEFAULT: Self = Self {
        documents_per_page: 16,
        bytes_per_page: 256 * 1024,
        slice_duration: Duration::from_millis(500),
        source_poll_interval: Duration::from_millis(250),
    };

    pub fn page_is_full(self, documents: usize, bytes: usize) -> bool {
        documents >= self.documents_per_page || (documents > 0 && bytes >= self.bytes_per_page)
    }

    pub fn slice_expired(self, elapsed: Duration) -> bool {
        elapsed >= self.slice_duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_bounds_count_bytes_and_wall_time_without_rejecting_one_large_document() {
        let budget = PeerProjectionBudget::DEFAULT;
        assert!(!budget.page_is_full(0, budget.bytes_per_page * 2));
        assert!(budget.page_is_full(1, budget.bytes_per_page));
        assert!(budget.page_is_full(budget.documents_per_page, 0));
        assert!(!budget.page_is_full(budget.documents_per_page - 1, 0));
        assert!(!budget.slice_expired(budget.slice_duration - Duration::from_nanos(1)));
        assert!(budget.slice_expired(budget.slice_duration));
    }
}
