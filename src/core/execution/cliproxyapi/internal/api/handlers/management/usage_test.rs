// ref: internal/api/handlers/management/usage_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, Mutex};

use super::{
    ManagementAuthenticator, ManagementUsageQueue, ManagementUsageQueueError,
    SystemManagementAuthClock,
};
use crate::internal::api::server_management::ManagementHandler;

#[derive(Default)]
struct InMemoryUsageQueue(Mutex<VecDeque<Vec<u8>>>);

impl InMemoryUsageQueue {
    fn enqueue(&self, item: &[u8]) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back(item.to_vec());
    }

    fn remaining(&self) -> Vec<Vec<u8>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

impl ManagementUsageQueue for InMemoryUsageQueue {
    fn pop_oldest(&self, count: usize) -> Result<Vec<Vec<u8>>, ManagementUsageQueueError> {
        let mut items = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok((0..count).filter_map(|_| items.pop_front()).collect())
    }
}

fn handler(queue: Arc<dyn ManagementUsageQueue>) -> ManagementHandler {
    ManagementHandler::with_usage_queue(
        Arc::new(
            ManagementAuthenticator::new(
                "management-secret",
                false,
                Arc::new(SystemManagementAuthClock),
            )
            .unwrap(),
        ),
        queue,
    )
}

fn headers() -> BTreeMap<String, Vec<String>> {
    BTreeMap::from([(
        "X-Management-Key".to_owned(),
        vec!["management-secret".to_owned()],
    )])
}

#[test]
fn get_usage_queue_pops_requested_records() {
    let queue = Arc::new(InMemoryUsageQueue::default());
    queue.enqueue(br#"{"id":1}"#);
    queue.enqueue(br#"{"id":2}"#);
    queue.enqueue(br#"{"id":3}"#);

    let response = handler(queue.clone()).handle(
        "GET",
        "/v0/management/usage-queue?count=2",
        &headers(),
        &[],
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );

    assert_eq!(response.status(), 200);
    let payload: Vec<serde_json::Value> = serde_json::from_slice(response.body()).unwrap();
    assert_eq!(
        payload,
        vec![serde_json::json!({"id": 1}), serde_json::json!({"id": 2})]
    );
    assert_eq!(queue.remaining(), vec![br#"{"id":3}"#.to_vec()]);
}

#[test]
fn get_usage_queue_invalid_count_does_not_pop() {
    let queue = Arc::new(InMemoryUsageQueue::default());
    queue.enqueue(br#"{"id":1}"#);

    let response = handler(queue.clone()).handle(
        "GET",
        "/v0/management/usage-queue?count=0",
        &headers(),
        &[],
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );

    assert_eq!(response.status(), 400);
    assert_eq!(queue.remaining(), vec![br#"{"id":1}"#.to_vec()]);
}

#[test]
fn usage_queue_preserves_valid_json_and_quotes_invalid_records() {
    let queue = Arc::new(InMemoryUsageQueue::default());
    queue.enqueue(b" true ");
    queue.enqueue(b"not-json");

    let response = handler(queue).handle(
        "GET",
        "/v0/management/usage-queue?count=2",
        &headers(),
        &[],
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    );

    assert_eq!(response.status(), 200);
    assert_eq!(response.body(), br#"[ true ,"not-json"]"#);
}
