use ctox_sync::{
    authority::{auth::SigningIdentity, Peer, WorkerMembership},
    host_config::{self, HostConfiguration, HostMember},
};
use rusqlite::Connection;
use std::{collections::BTreeMap, io};

fn fixture() -> (HostConfiguration, Vec<SigningIdentity>) {
    let keys: Vec<_> = (0..4)
        .map(|_| SigningIdentity::from_pkcs8(&SigningIdentity::generate_pkcs8().unwrap()).unwrap())
        .collect();
    let voters = keys
        .iter()
        .take(3)
        .enumerate()
        .map(|(i, key)| {
            (
                (i + 1) as u64,
                Peer {
                    identity: key.public_identity(),
                    executor: i != 2,
                    data_replica: i != 2,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    (
        HostConfiguration {
            version: 1,
            scope_id: "test-network".into(),
            local: HostMember::Voter { node_id: 1 },
            voters,
            timing: Default::default(),
        },
        keys,
    )
}

#[test]
fn reopen_keeps_identity_and_other_runtime_data() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("ctox-runtime.sqlite3");
    let (config, keys) = fixture();
    {
        let mut connection = Connection::open(&path).unwrap();
        connection.execute_batch("CREATE TABLE runtime_env_kv(key TEXT PRIMARY KEY, value TEXT NOT NULL); INSERT INTO runtime_env_kv VALUES ('existing', 'preserved')").unwrap();
        assert!(host_config::load(&connection).unwrap().is_none());
        host_config::save(&mut connection, &config).unwrap();
    }
    let connection = Connection::open(path).unwrap();
    let loaded = host_config::load(&connection).unwrap().unwrap();
    assert!(config.same_binding(&loaded));
    loaded.validate_key(&keys[0]).unwrap();
    assert_eq!(
        loaded.validate_key(&keys[1]).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT value FROM runtime_env_kv WHERE key='existing'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        "preserved"
    );
}

#[test]
fn rebinding_is_rejected_atomically_and_timing_can_change() {
    let (config, keys) = fixture();
    let mut connection = Connection::open_in_memory().unwrap();
    host_config::save(&mut connection, &config).unwrap();
    let mut changed = config.clone();
    changed.scope_id = "other-network".into();
    assert!(host_config::save(&mut connection, &changed).is_err());
    changed = config.clone();
    changed.local = HostMember::Voter { node_id: 2 };
    assert!(host_config::save(&mut connection, &changed).is_err());
    changed = config.clone();
    changed.voters.get_mut(&3).unwrap().identity = keys[3].public_identity();
    assert!(host_config::save(&mut connection, &changed).is_err());
    changed = config.clone();
    changed.voters.get_mut(&3).unwrap().data_replica = true;
    assert!(host_config::save(&mut connection, &changed).is_err());
    assert!(host_config::load(&connection)
        .unwrap()
        .unwrap()
        .same_binding(&config));
    changed = config.clone();
    changed.timing.heartbeat_ms = 300;
    host_config::save(&mut connection, &changed).unwrap();
    assert_eq!(
        host_config::load(&connection)
            .unwrap()
            .unwrap()
            .timing
            .heartbeat_ms,
        300
    );
}

#[test]
fn worker_pin_cannot_claim_a_vote_or_silently_change_capabilities() {
    let (mut config, keys) = fixture();
    config.local = HostMember::Worker {
        member: WorkerMembership {
            node_id: 4,
            identity: keys[3].public_identity(),
            data_replica: true,
            revoked: false,
        },
    };
    let mut connection = Connection::open_in_memory().unwrap();
    host_config::save(&mut connection, &config).unwrap();
    let mut changed = config.clone();
    let HostMember::Worker { member } = &mut changed.local else {
        unreachable!()
    };
    member.data_replica = false;
    assert!(host_config::save(&mut connection, &changed).is_err());
    for (id, identity, revoked) in [
        (1, keys[3].public_identity(), false),
        (4, keys[0].public_identity(), false),
        (4, keys[3].public_identity(), true),
    ] {
        changed.local = HostMember::Worker {
            member: WorkerMembership {
                node_id: id,
                identity,
                data_replica: true,
                revoked,
            },
        };
        assert!(changed.validate().is_err());
    }
    assert!(host_config::load(&connection)
        .unwrap()
        .unwrap()
        .same_binding(&config));
}

#[test]
fn malformed_configuration_never_replaces_a_valid_record() {
    let (config, _) = fixture();
    let mut connection = Connection::open_in_memory().unwrap();
    host_config::save(&mut connection, &config).unwrap();
    let mut variants = Vec::new();
    for scope in ["", "../escape", "network/../other", "space here"] {
        let mut candidate = config.clone();
        candidate.scope_id = scope.into();
        variants.push(candidate);
    }
    let mut candidate = config.clone();
    candidate.version = 2;
    variants.push(candidate);
    let mut candidate = config.clone();
    candidate.voters.remove(&3);
    variants.push(candidate);
    let mut candidate = config.clone();
    candidate.voters.get_mut(&2).unwrap().identity = candidate.voters[&1].identity.clone();
    variants.push(candidate);
    let mut candidate = config.clone();
    candidate.voters.get_mut(&3).unwrap().identity = "ed25519:invalid".into();
    variants.push(candidate);
    let mut candidate = config.clone();
    candidate.local = HostMember::Voter {
        node_id: 9_007_199_254_740_992,
    };
    variants.push(candidate);
    let mut candidate = config.clone();
    candidate.timing.election_min_ms = 1;
    variants.push(candidate);
    for candidate in variants {
        assert!(host_config::save(&mut connection, &candidate).is_err());
    }
    assert!(host_config::load(&connection)
        .unwrap()
        .unwrap()
        .same_binding(&config));
    connection
        .execute("UPDATE ctox_sync_host SET configuration='{}'", [])
        .unwrap();
    assert!(host_config::load(&connection).is_err());
}

#[cfg(feature = "webrtc")]
#[test]
fn native_options_preserve_binding_without_persisting_live_routes() {
    let (mut config, keys) = fixture();
    let root = tempfile::tempdir().unwrap();
    let voter = config
        .voter_options(root.path(), root.path(), &keys[0])
        .unwrap();
    assert_eq!(voter.room, "ctox-execution:test-network");
    assert_eq!(voter.node_id, 1);
    assert_eq!(voter.peers, config.voters);
    assert!(voter.routes.is_empty());
    assert!(voter.store_path.starts_with(root.path()));
    assert!(config.worker_options(root.path(), &keys[0]).is_err());
    assert!(config
        .voter_options(root.path(), root.path(), &keys[1])
        .is_err());
    assert!(config
        .voter_options(std::path::Path::new("relative"), root.path(), &keys[0])
        .is_err());
    config.local = HostMember::Worker {
        member: WorkerMembership {
            node_id: 4,
            identity: keys[3].public_identity(),
            data_replica: true,
            revoked: false,
        },
    };
    let worker = config.worker_options(root.path(), &keys[3]).unwrap();
    assert_eq!(worker.voters, voter.peers);
    assert_eq!(worker.ipc_directory, voter.ipc_directory);
    assert!(worker.routes.is_empty());
    assert!(config
        .voter_options(root.path(), root.path(), &keys[3])
        .is_err());
    assert!(config
        .worker_options(std::path::Path::new("relative"), &keys[3])
        .is_err());
    let serialized = serde_json::to_string(&config).unwrap();
    for excluded in ["routes", "ipc", "secret", "password", "token", "signaling"] {
        assert!(!serialized.contains(excluded));
    }
}
