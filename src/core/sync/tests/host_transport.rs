#![cfg(feature = "webrtc")]
use ctox_sync::{
    authority::{auth::SigningIdentity, Peer, WorkerMembership},
    host_config::{HostConfiguration, HostMember},
    host_transport::HostTransport,
};
use std::collections::BTreeMap;
fn config(worker: bool) -> HostConfiguration {
    let identities: Vec<_> = (0..4)
        .map(|_| {
            SigningIdentity::from_pkcs8(&SigningIdentity::generate_pkcs8().unwrap())
                .unwrap()
                .public_identity()
        })
        .collect();
    HostConfiguration {
        version: 1,
        scope_id: "network".into(),
        local: if worker {
            HostMember::Worker {
                member: WorkerMembership {
                    node_id: 4,
                    identity: identities[3].clone(),
                    data_replica: true,
                    revoked: false,
                },
            }
        } else {
            HostMember::Voter { node_id: 1 }
        },
        voters: identities
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, identity)| {
                (
                    (i + 1) as u64,
                    Peer {
                        identity: identity.clone(),
                        executor: true,
                        data_replica: true,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
        timing: Default::default(),
    }
}
fn input(url: &str) -> String {
    serde_json::json!({"signalingUrls":[url],"iceServers":[]}).to_string()
}
#[test]
fn transport_keeps_execution_identity_separate_from_business_os_credentials() {
    let config = config(false);
    for url in [
        "wss://signal.example/execution?role=ctox_instance",
        "ws://127.0.0.1:8000?role=ctox_instance",
        "ws://[::1]:8000?role=ctox_instance",
    ] {
        HostTransport::parse(&input(url), &config).unwrap();
    }
    for url in [
        "ws://signal.example?role=ctox_instance",
        "https://signal.example?role=ctox_instance",
        "wss://signal.example?role=workjet_executor",
        "wss://signal.example?role=browser",
        "wss://signal.example?role=ctox_instance&role=browser",
        "wss://signal.example?role=ctox_instance&peer_role=browser",
        "wss://signal.example?role=ctox_instance&auth_version=ctox-role-bound-v1",
        "wss://signal.example?role=ctox_instance&browser_token_hash=secret",
        "wss://signal.example?role=ctox_instance&native_token_hash=secret",
        "wss://signal.example?role=ctox_instance&instance_id=other",
        "wss://user:secret@signal.example?role=ctox_instance",
        "wss://signal.example?role=ctox_instance#secret",
    ] {
        assert!(
            HostTransport::parse(&input(url), &config).is_err(),
            "accepted {url}"
        );
    }
}
#[test]
fn worker_transport_cannot_advertise_a_voter_or_browser_role() {
    let config = config(true);
    HostTransport::parse(
        &input("wss://signal.example?role=workjet_executor&instance_id=network"),
        &config,
    )
    .unwrap();
    assert!(
        HostTransport::parse(&input("wss://signal.example?role=ctox_instance"), &config).is_err()
    );
}
#[test]
fn transport_limits_and_errors_do_not_expose_credentials() {
    let config = config(false);
    let default_transport =
        HostTransport::parse(&input("wss://signal.example?role=ctox_instance"), &config).unwrap();
    assert!(default_transport.native_ice_servers()[0].urls.is_empty());
    let marker = "PRIVATE_CREDENTIAL_MUST_NOT_APPEAR";
    for value in [
        serde_json::json!({"signalingUrls": marker}),
        serde_json::json!({"signalingUrls":[marker]}),
        serde_json::json!({"signalingUrls":[]}),
        serde_json::json!({"signalingUrls":vec!["wss://signal.example?role=ctox_instance"; 9]}),
    ] {
        let error = HostTransport::parse(&value.to_string(), &config)
            .err()
            .unwrap();
        assert!(!error.to_string().contains(marker));
    }
    let value = serde_json::json!({"signalingUrls":["wss://signal.example?role=ctox_instance"], "iceServers":[{"urls":["turn:relay.example:3478"], "username":"user", "credential":marker}]});
    let transport = HostTransport::parse(&value.to_string(), &config).unwrap();
    assert_eq!(transport.native_ice_servers()[0].credential, marker);
}
#[cfg(unix)]
#[test]
fn host_directory_lease_cannot_replace_an_existing_runtime_owner() {
    use ctox_sync::local_host::HostDirectoryLock;
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("host");
    let lease = HostDirectoryLock::acquire(&directory).unwrap();
    assert_eq!(
        HostDirectoryLock::acquire(&directory).err().unwrap().kind(),
        std::io::ErrorKind::AddrInUse
    );
    assert_eq!(
        lease.directory(),
        std::fs::canonicalize(&directory).unwrap()
    );
    drop(lease);
    HostDirectoryLock::acquire(&directory).unwrap();
}
