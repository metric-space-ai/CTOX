use ctox_sync::authority::{
    auth::SigningIdentity, Command, Peer, Receipt, Rejection, Request, State, WorkerMembership,
};
use std::collections::BTreeMap;

fn identity() -> String {
    SigningIdentity::from_pkcs8(&SigningIdentity::generate_pkcs8().unwrap())
        .unwrap()
        .public_identity()
}

#[test]
fn enrollment_preserves_voters_and_never_reuses_a_revoked_node_id() {
    let voters: BTreeMap<_, _> = (1..=3)
        .map(|id| {
            (
                id,
                Peer {
                    identity: identity(),
                    executor: id != 3,
                    data_replica: id != 3,
                },
            )
        })
        .collect();
    let mut state = State::default();
    let worker = WorkerMembership {
        node_id: 4,
        identity: identity(),
        data_replica: true,
        revoked: false,
    };
    let apply = |state: &mut State, id: &str, actor, command| {
        state.apply(
            &Request {
                request_id: id.into(),
                actor,
                command,
            },
            &voters,
        )
    };
    let admission = Command::AdmitWorker {
        worker: worker.clone(),
    };
    assert_eq!(
        apply(&mut state, "untrusted", 4, admission.clone()),
        Receipt::Rejected(Rejection::UnknownPeer)
    );
    assert_eq!(
        apply(&mut state, "admit", 3, admission.clone()),
        Receipt::WorkerApplied(worker.clone())
    );
    assert_eq!(
        apply(&mut state, "admit", 3, admission.clone()),
        Receipt::WorkerReplayed(worker.clone())
    );
    for (id, invalid) in [
        (
            "zero",
            WorkerMembership {
                node_id: 0,
                ..worker.clone()
            },
        ),
        (
            "unsafe-js-id",
            WorkerMembership {
                node_id: 9_007_199_254_740_992,
                ..worker.clone()
            },
        ),
        (
            "bad-key",
            WorkerMembership {
                node_id: 5,
                identity: "invalid".into(),
                ..worker.clone()
            },
        ),
        (
            "revoked-on-arrival",
            WorkerMembership {
                node_id: 5,
                revoked: true,
                ..worker.clone()
            },
        ),
    ] {
        assert_eq!(
            apply(&mut state, id, 1, Command::AdmitWorker { worker: invalid }),
            Receipt::Rejected(Rejection::InvalidRequest)
        );
    }
    for (id, duplicate) in [
        ("worker-id", worker.clone()),
        (
            "voter-id",
            WorkerMembership {
                node_id: 1,
                ..worker.clone()
            },
        ),
        (
            "worker-key",
            WorkerMembership {
                node_id: 5,
                ..worker.clone()
            },
        ),
        (
            "voter-key",
            WorkerMembership {
                node_id: 5,
                identity: voters[&1].identity.clone(),
                ..worker.clone()
            },
        ),
    ] {
        assert_eq!(
            apply(
                &mut state,
                id,
                1,
                Command::AdmitWorker { worker: duplicate }
            ),
            Receipt::Rejected(Rejection::AlreadyExists)
        );
    }
    assert_eq!(
        apply(
            &mut state,
            "worker-revoke",
            4,
            Command::RevokeWorker { node_id: 4 }
        ),
        Receipt::Rejected(Rejection::UnknownPeer)
    );
    let revoked = WorkerMembership {
        revoked: true,
        ..worker.clone()
    };
    assert_eq!(
        apply(
            &mut state,
            "revoke",
            1,
            Command::RevokeWorker { node_id: 4 }
        ),
        Receipt::WorkerApplied(revoked.clone())
    );
    assert_eq!(
        apply(&mut state, "admit", 3, admission.clone()),
        Receipt::WorkerReplayed(worker.clone())
    );
    assert_eq!(
        state.workers[&4], revoked,
        "a replay cannot undo revocation"
    );
    assert_eq!(
        apply(&mut state, "reuse", 1, admission),
        Receipt::Rejected(Rejection::AlreadyExists)
    );
    let reenrolled = WorkerMembership {
        node_id: 5,
        ..worker
    };
    assert_eq!(
        apply(
            &mut state,
            "new-enrollment",
            1,
            Command::AdmitWorker {
                worker: reenrolled.clone()
            }
        ),
        Receipt::WorkerApplied(reenrolled)
    );
    assert_eq!(voters.len(), 3);
    assert!(state.workers[&4].revoked);
    assert!(!state.workers[&5].revoked);
}
