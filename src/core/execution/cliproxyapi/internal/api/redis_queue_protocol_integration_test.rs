// ref: internal/api/redis_queue_protocol_integration_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use super::redis_queue_protocol::*;

struct Queue {
    password: String,
    items: Mutex<BTreeMap<String, VecDeque<Vec<u8>>>>,
}

impl Queue {
    fn new() -> Self {
        Self {
            password: "management-secret".into(),
            items: Mutex::new(BTreeMap::from([(
                "usage".into(),
                VecDeque::from([b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]),
            )])),
        }
    }
}

impl RedisQueueAuthority for Queue {
    fn authenticate(&self, password: &str) -> bool {
        password == self.password
    }
    fn pop(&self, channel: &str, count: usize) -> Result<Vec<Vec<u8>>, RedisQueueError> {
        if channel != "usage" {
            return Err(RedisQueueError::UnsupportedChannel);
        }
        let mut items = self.items.lock().map_err(|_| RedisQueueError::Read)?;
        let queue = items.get_mut(channel).ok_or(RedisQueueError::Read)?;
        Ok((0..count).filter_map(|_| queue.pop_front()).collect())
    }
}

fn command(values: &[&str]) -> RespValue {
    RespValue::Array(
        values
            .iter()
            .map(|value| RespValue::Bulk(Some(value.as_bytes().to_vec())))
            .collect(),
    )
}

fn remote_session(policy: RedisQueuePolicy) -> RedisQueueSession {
    RedisQueueSession::new(Arc::new(Queue::new()), policy)
}

#[test]
fn resp_codec_round_trips_nested_arrays_and_rejects_truncation() {
    let value = RespValue::Array(vec![
        RespValue::Bulk(Some(b"AUTH".to_vec())),
        RespValue::Bulk(Some(b"secret".to_vec())),
    ]);
    let encoded = encode_resp(&value);
    assert_eq!(decode_resp(&encoded), Ok((value, encoded.len())));
    assert_eq!(
        decode_resp(&encoded[..encoded.len() - 1]),
        Err(RespError::Incomplete)
    );
}

#[test]
fn management_disabled_and_home_mode_fail_closed() {
    let mut disabled = remote_session(RedisQueuePolicy {
        management_enabled: false,
        home_enabled: false,
        local_client: true,
    });
    assert_eq!(
        disabled.handle(&command(&["PING"])),
        [RespValue::Error("ERR redis usage output disabled".into())]
    );
    let mut home = remote_session(RedisQueuePolicy {
        management_enabled: true,
        home_enabled: true,
        local_client: true,
    });
    assert_eq!(
        home.handle(&command(&["PING"])),
        [RespValue::Error(
            "ERR redis usage output disabled in home mode".into()
        )]
    );
}

#[test]
fn auth_and_pop_contracts_preserve_nil_and_array_shapes() {
    let mut session = remote_session(RedisQueuePolicy {
        management_enabled: true,
        home_enabled: false,
        local_client: false,
    });
    assert!(matches!(
        session.handle(&command(&["RPOP", "usage"]))[0],
        RespValue::Error(_)
    ));
    assert_eq!(
        session.handle(&command(&["AUTH", "wrong"]))[0],
        RespValue::Error("ERR invalid password".into())
    );
    assert_eq!(
        session.handle(&command(&["AUTH", "management-secret"]))[0],
        RespValue::Simple("OK".into())
    );
    assert_eq!(
        session.handle(&command(&["RPOP", "usage"]))[0],
        RespValue::Bulk(Some(b"one".to_vec()))
    );
    assert_eq!(
        session.handle(&command(&["RPOP", "usage", "10"]))[0],
        RespValue::Array(vec![
            RespValue::Bulk(Some(b"two".to_vec())),
            RespValue::Bulk(Some(b"three".to_vec()))
        ])
    );
    assert_eq!(
        session.handle(&command(&["LPOP", "usage"]))[0],
        RespValue::Bulk(None)
    );
    assert!(matches!(
        session.handle(&command(&["RPOP", "errors"]))[0],
        RespValue::Error(_)
    ));
}

#[test]
fn usage_subscribe_emits_ack_then_support_refresh() {
    let mut session = remote_session(RedisQueuePolicy {
        management_enabled: true,
        home_enabled: false,
        local_client: true,
    });
    let responses = session.handle(&command(&["SUBSCRIBE", "usage"]));
    assert_eq!(responses.len(), 2);
    assert_eq!(
        responses[0],
        RespValue::Array(vec![
            RespValue::Bulk(Some(b"subscribe".to_vec())),
            RespValue::Bulk(Some(b"usage".to_vec())),
            RespValue::Integer(1)
        ])
    );
    assert!(encode_resp(&responses[1])
        .windows(b"support_refresh".len())
        .any(|window| window == b"support_refresh"));
}

#[test]
fn debug_never_exposes_management_password() {
    let session = remote_session(RedisQueuePolicy {
        management_enabled: true,
        home_enabled: false,
        local_client: false,
    });
    assert!(!format!("{session:?}").contains("management-secret"));
}
