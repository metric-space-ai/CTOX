use ctox_sync::authority::{
    Command, ExecutionSpec, Ownership, Peer, Receipt, Rejection, Request, State,
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn lost_effect_ack_cannot_become_a_second_dispatch_permission() {
    let peers: BTreeMap<_, _> = (1..=3)
        .map(|id| {
            (
                id,
                Peer {
                    identity: format!("peer-{id}"),
                    executor: true,
                    data_replica: true,
                },
            )
        })
        .collect();
    let mut state = State::default();
    let spec = ExecutionSpec {
        job_id: "job".into(),
        session_id: "session".into(),
        scope_id: "scope".into(),
        harness: "codex".into(),
        harness_version: "fixture".into(),
        model_route_id: "route".into(),
        gateway_account_id: "account".into(),
        model_id: "model".into(),
        required_capabilities: BTreeSet::new(),
    };
    state.apply(
        &Request {
            request_id: "create".into(),
            actor: 1,
            command: Command::Create { spec, owner: 1 },
        },
        &peers,
    );
    let start = Request {
        request_id: "begin".into(),
        actor: 1,
        command: Command::BeginEffect {
            job_id: "job".into(),
            ownership: Ownership {
                node_id: 1,
                generation: 1,
            },
            effect_id: "external-publish".into(),
        },
    };
    assert!(matches!(state.apply(&start, &peers), Receipt::Applied(_)));
    // Rehydrating the machine also preserves the distinction after a crash.
    let mut recovered: State =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    assert!(matches!(
        recovered.apply(&start, &peers),
        Receipt::Replayed(_)
    ));
    let retry = Request {
        request_id: "different-request".into(),
        ..start
    };
    assert_eq!(
        recovered.apply(&retry, &peers),
        Receipt::Rejected(Rejection::ReconciliationRequired)
    );
}
