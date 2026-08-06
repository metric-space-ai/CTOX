// ref: sdk/cliproxy/auth/home_in_flight_publisher_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: deterministic packing, bounded overflow and injected transport/clock behavior
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use crate::internal::home::{HomeError, InFlightFrameKind};
use crate::sdk::cliproxy::executionregistry::{Freeze, Observation, Registry};

use super::{
    encode_home_in_flight_freeze, HomeClock, HomeInFlightPublisher, HomeInFlightPublisherConfig,
    HomeInFlightTransport,
};

#[derive(Default)]
struct SnapshotTransport {
    ready: bool,
    payloads: Mutex<Vec<Vec<u8>>>,
}

impl HomeInFlightTransport for SnapshotTransport {
    fn heartbeat_ok(&self) -> bool {
        self.ready
    }
    fn push_in_flight_snapshot(&self, payload: &[u8]) -> Result<(), HomeError> {
        self.payloads.lock().unwrap().push(payload.to_vec());
        Ok(())
    }
}

struct FixedClock(DateTime<Utc>);
impl HomeClock for FixedClock {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }
}

fn now() -> DateTime<Utc> {
    "2026-08-04T12:00:00Z".parse().unwrap()
}

#[test]
fn publisher_uses_injected_transport_and_skips_disconnected_lifetime() {
    let disconnected = Arc::new(SnapshotTransport::default());
    let transport: Arc<dyn HomeInFlightTransport> = disconnected.clone();
    let publisher = HomeInFlightPublisher::new_with_clock(
        transport,
        Arc::new(Registry::new()),
        HomeInFlightPublisherConfig::default(),
        Arc::new(FixedClock(now())),
    )
    .unwrap();
    assert_eq!(publisher.publish_once(now()).unwrap(), 0);
    assert!(disconnected.payloads.lock().unwrap().is_empty());

    let connected = Arc::new(SnapshotTransport {
        ready: true,
        ..SnapshotTransport::default()
    });
    let transport: Arc<dyn HomeInFlightTransport> = connected.clone();
    let publisher = HomeInFlightPublisher::new_with_clock(
        transport,
        Arc::new(Registry::new()),
        HomeInFlightPublisherConfig::default(),
        Arc::new(FixedClock(now())),
    )
    .unwrap();
    assert_eq!(publisher.publish_once(now()).unwrap(), 1);
    assert_eq!(connected.payloads.lock().unwrap().len(), 1);
}

#[test]
fn frames_are_sorted_and_canonicalize_unaccounted_models() {
    let freeze = Freeze {
        revision: 7,
        barrier_revision: 3,
        executions: vec![
            Observation {
                request_id: "b".into(),
                credential_id: "auth".into(),
                model: "GPT-5(HIGH)".into(),
                request_kind: "request".into(),
                started_at: now(),
                accounted: false,
            },
            Observation {
                request_id: "a".into(),
                credential_id: "auth".into(),
                model: "GPT-5(HIGH)".into(),
                request_kind: "request".into(),
                started_at: now(),
                accounted: false,
            },
        ],
    };
    let frames =
        encode_home_in_flight_freeze(&freeze, now(), HomeInFlightPublisherConfig::default());
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].aggregates[0].model, "gpt-5");
    assert_eq!(frames[0].details[0].request_id, "a");
    assert_eq!(frames[0].details[1].request_id, "b");
}

#[test]
fn oversized_group_set_emits_single_overflow_frame() {
    let freeze = Freeze {
        revision: 1,
        executions: vec![Observation {
            request_id: "r".into(),
            credential_id: "auth".into(),
            model: "gpt".into(),
            request_kind: "request".into(),
            started_at: now(),
            accounted: true,
        }],
        ..Freeze::default()
    };
    let config = HomeInFlightPublisherConfig {
        max_aggregate_groups: 0,
        ..HomeInFlightPublisherConfig::default()
    };
    let frames = encode_home_in_flight_freeze(&freeze, now(), config);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].kind, InFlightFrameKind::Overflow);
}
