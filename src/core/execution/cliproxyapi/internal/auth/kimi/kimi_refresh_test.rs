// ref: internal/auth/kimi/kimi_refresh_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use tokio::sync::Notify;

use super::{
    DeviceCodeResponse, DeviceFlowClient, KimiClock, KimiDeviceIdentity, KimiHttpFuture,
    KimiHttpRequest, KimiHttpResponse, KimiHttpTransport, KimiRefreshCoordinator, KimiSleepFuture,
    SecretString,
};
use crate::sdk::auth::LoginCancellation;

struct FixedClock {
    now: Mutex<SystemTime>,
}

impl KimiClock for FixedClock {
    fn now(&self) -> SystemTime {
        *self.now.lock().unwrap()
    }

    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> KimiSleepFuture<'a> {
        Box::pin(async move {
            if cancellation.is_cancelled() {
                return Err(super::KimiTransportFailure::Cancelled);
            }
            let mut now = self.now.lock().unwrap();
            *now += duration;
            Ok(())
        })
    }
}

struct RefreshTransport {
    calls: AtomicUsize,
    started: Notify,
    release: Notify,
}

impl KimiHttpTransport for RefreshTransport {
    fn execute<'a>(
        &'a self,
        request: &'a KimiHttpRequest,
        _timeout: Duration,
        _cancellation: &'a LoginCancellation,
    ) -> KimiHttpFuture<'a> {
        Box::pin(async move {
            assert!(String::from_utf8_lossy(&request.body).contains("grant_type=refresh_token"));
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            Ok(KimiHttpResponse::new(
                200,
                br#"{"access_token":"new-access","refresh_token":"new-refresh","token_type":"Bearer","expires_in":3600}"#.to_vec(),
            ))
        })
    }
}

fn client(
    transport: Arc<dyn KimiHttpTransport>,
    clock: Arc<dyn KimiClock>,
    coordinator: Arc<KimiRefreshCoordinator>,
) -> Arc<DeviceFlowClient> {
    Arc::new(DeviceFlowClient::new(
        transport,
        clock,
        KimiDeviceIdentity::new("device", "host", "model", "version").unwrap(),
        coordinator,
    ))
}

#[tokio::test]
async fn refresh_deduplicates_concurrent_calls_across_clients() {
    let transport = Arc::new(RefreshTransport {
        calls: AtomicUsize::new(0),
        started: Notify::new(),
        release: Notify::new(),
    });
    let clock: Arc<dyn KimiClock> = Arc::new(FixedClock {
        now: Mutex::new(SystemTime::UNIX_EPOCH),
    });
    let coordinator = Arc::new(KimiRefreshCoordinator::default());
    let first = client(transport.clone(), clock.clone(), coordinator.clone());
    let second = client(transport.clone(), clock, coordinator);

    let first_task = tokio::spawn(async move {
        first
            .refresh_token(SecretString::new("shared-refresh-token").unwrap())
            .await
    });
    transport.started.notified().await;
    let second_task = tokio::spawn(async move {
        second
            .refresh_token(SecretString::new("shared-refresh-token").unwrap())
            .await
    });
    tokio::task::yield_now().await;
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    transport.release.notify_one();

    for result in [first_task.await.unwrap(), second_task.await.unwrap()] {
        let token = result.unwrap();
        assert_eq!(token.access_token().expose_secret(), "new-access");
        assert_eq!(
            token.refresh_token().unwrap().expose_secret(),
            "new-refresh"
        );
    }
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
}

struct SequenceTransport(Mutex<Vec<KimiHttpResponse>>);

impl KimiHttpTransport for SequenceTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a KimiHttpRequest,
        _timeout: Duration,
        _cancellation: &'a LoginCancellation,
    ) -> KimiHttpFuture<'a> {
        Box::pin(async move { Ok(self.0.lock().unwrap().remove(0)) })
    }
}

#[tokio::test]
async fn poll_waits_before_each_attempt_and_honors_pending() {
    let transport: Arc<dyn KimiHttpTransport> = Arc::new(SequenceTransport(Mutex::new(vec![
        KimiHttpResponse::new(200, br#"{"error":"authorization_pending"}"#.to_vec()),
        KimiHttpResponse::new(
            200,
            br#"{"access_token":"access","token_type":"Bearer","expires_in":60}"#.to_vec(),
        ),
    ])));
    let clock = Arc::new(FixedClock {
        now: Mutex::new(SystemTime::UNIX_EPOCH),
    });
    let client = client(
        transport,
        clock.clone(),
        Arc::new(KimiRefreshCoordinator::default()),
    );
    let token = client
        .poll_for_token(
            &LoginCancellation::default(),
            &DeviceCodeResponse {
                device_code: "device-code".to_owned(),
                user_code: "user-code".to_owned(),
                expires_in: 30,
                interval: 1,
                ..DeviceCodeResponse::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(token.access_token().expose_secret(), "access");
    assert_eq!(
        clock.now(),
        SystemTime::UNIX_EPOCH + Duration::from_secs(10)
    );
}
