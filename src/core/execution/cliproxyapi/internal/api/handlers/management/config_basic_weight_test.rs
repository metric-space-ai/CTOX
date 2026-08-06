// ref: internal/api/handlers/management/config_basic_weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::auth::SchedulerStrategy;

use super::normalize_routing_strategy;

#[test]
fn normalize_weighted_round_robin() {
    for input in ["weighted-round-robin", "weightedroundrobin", "wrr"] {
        assert_eq!(
            normalize_routing_strategy(input),
            Ok(SchedulerStrategy::WeightedRoundRobin)
        );
    }
}
