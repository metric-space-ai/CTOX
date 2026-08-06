// ref: internal/redisqueue/queue_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{UsageQueue, UsageStatisticsSwitch};

#[tokio::test]
async fn enqueue_broadcasts_to_usage_subscribers_and_skips_queue() {
    let queue = UsageQueue::new();
    queue.set_enabled(true);
    let (mut first, mut first_subscription) = queue.subscribe_usage();
    let (mut second, mut second_subscription) = queue.subscribe_usage();

    assert_eq!(
        first.recv().await.as_deref(),
        Some(br#"{"support_refresh":true}"#.as_slice())
    );
    assert_eq!(
        second.recv().await.as_deref(),
        Some(br#"{"support_refresh":true}"#.as_slice())
    );

    queue.enqueue(b"usage-record");
    assert_eq!(
        first.recv().await.as_deref(),
        Some(b"usage-record".as_slice())
    );
    assert_eq!(
        second.recv().await.as_deref(),
        Some(b"usage-record".as_slice())
    );
    assert!(queue.pop_oldest(1).is_empty());

    first_subscription.unsubscribe();
    second_subscription.unsubscribe();
    queue.enqueue(b"queued-record");
    assert_eq!(queue.pop_oldest(1), vec![b"queued-record".to_vec()]);
}

#[tokio::test]
async fn disabling_closes_usage_and_error_subscribers() {
    let queue = UsageQueue::new();
    queue.set_enabled(true);
    let (mut usage, _usage_subscription) = queue.subscribe_usage();
    let (mut errors, _error_subscription) = queue.subscribe_errors();
    assert!(usage.recv().await.is_some());

    queue.set_enabled(false);
    assert_eq!(usage.recv().await, None);
    assert_eq!(errors.recv().await, None);
}

#[tokio::test]
async fn errors_broadcast_and_are_discarded_without_subscribers() {
    let queue = UsageQueue::new();
    queue.set_enabled(true);
    let (mut errors, mut subscription) = queue.subscribe_errors();
    queue.enqueue_error(b"error-record");
    assert_eq!(
        errors.recv().await.as_deref(),
        Some(b"error-record".as_slice())
    );
    subscription.unsubscribe();
    queue.enqueue_error(b"discarded-error");
    assert!(queue.pop_oldest(1).is_empty());
}

#[tokio::test]
async fn refresh_broadcasts_only_to_usage_subscribers() {
    let queue = UsageQueue::new();
    queue.set_enabled(true);
    let (mut usage, mut usage_subscription) = queue.subscribe_usage();
    let (mut errors, _error_subscription) = queue.subscribe_errors();
    assert!(usage.recv().await.is_some());

    queue.notify_usage_refresh();
    assert_eq!(
        usage.recv().await.as_deref(),
        Some(br#"{"refresh":true}"#.as_slice())
    );
    assert!(errors.try_recv().is_err());
    usage_subscription.unsubscribe();
    queue.notify_usage_refresh();
    assert!(queue.pop_oldest(1).is_empty());
}

#[test]
fn retention_and_usage_switch_normalization_are_instance_scoped() {
    let queue = UsageQueue::new();
    assert_eq!(queue.retention_seconds(), 60);
    queue.set_retention_seconds(0);
    assert_eq!(queue.retention_seconds(), 60);
    queue.set_retention_seconds(9_000);
    assert_eq!(queue.retention_seconds(), 3_600);

    let usage = UsageStatisticsSwitch::new();
    assert!(usage.enabled());
    usage.set_enabled(false);
    assert!(!usage.enabled());
}
