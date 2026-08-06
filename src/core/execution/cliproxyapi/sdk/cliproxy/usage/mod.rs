// Origin: CTOX
// License: AGPL-3.0-only

mod accounting;
mod manager;

pub use accounting::{
    ensure_token_breakdown, ensure_token_breakdown_for_provider, new_independent_token_breakdown,
    new_partial_subset_token_breakdown, new_separate_reasoning_token_breakdown,
    new_subset_token_breakdown, new_unclassified_token_breakdown, Detail, TokenAccountingQuality,
    TokenBreakdown, TokenInputBreakdown, TokenOutputBreakdown, TOKEN_ACCOUNTING_SCHEMA_VERSION,
};
pub use manager::{
    generate_enabled, generate_flag, Failure, Manager, Plugin, Record, UsageContext,
    AUTO_SERVICE_TIER, DEFAULT_SERVICE_TIER,
};

#[cfg(test)]
mod accounting_test;

#[cfg(test)]
mod manager_test;
