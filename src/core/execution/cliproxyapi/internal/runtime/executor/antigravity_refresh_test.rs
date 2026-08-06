// ref: internal/runtime/executor/antigravity_refresh_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::Notify;

use crate::internal::auth::antigravity::{
    AntigravityRefreshCoordinator, AntigravityRefreshHttpResponse, AntigravityRefreshRequest,
    AntigravityRefreshTransport, AntigravityRefreshTransportFailure, AntigravityStoredCredentials,
    SecretString,
};

struct BlockingRefreshTransport {
    calls: AtomicUsize,
    started: Notify,
    release: Notify,
}

impl AntigravityRefreshTransport for BlockingRefreshTransport {
    fn execute<'a>(
        &'a self,
        _: &'a AntigravityRefreshRequest,
        _: Duration,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        AntigravityRefreshHttpResponse,
                        AntigravityRefreshTransportFailure,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            Ok(AntigravityRefreshHttpResponse::new(
                200,
                br#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#
                    .to_vec(),
            ))
        })
    }
}

fn credentials(project_id: &str) -> AntigravityStoredCredentials {
    AntigravityStoredCredentials::new(
        SecretString::new("old-access").unwrap(),
        SecretString::new("shared-refresh-token").unwrap(),
        SystemTime::UNIX_EPOCH + Duration::from_secs(1),
        project_id,
    )
    .unwrap()
}

#[tokio::test]
async fn antigravity_refresh_deduplicates_concurrent_refresh() {
    let coordinator = Arc::new(AntigravityRefreshCoordinator::default());
    let transport = Arc::new(BlockingRefreshTransport {
        calls: AtomicUsize::new(0),
        started: Notify::new(),
        release: Notify::new(),
    });
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);

    let first = {
        let coordinator = Arc::clone(&coordinator);
        let transport = Arc::clone(&transport);
        tokio::spawn(async move {
            coordinator
                .refresh(transport.as_ref(), credentials("project-a"), now)
                .await
        })
    };
    transport.started.notified().await;
    let second = {
        let coordinator = Arc::clone(&coordinator);
        let transport = Arc::clone(&transport);
        tokio::spawn(async move {
            coordinator
                .refresh(transport.as_ref(), credentials("project-b"), now)
                .await
        })
    };
    tokio::task::yield_now().await;

    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    transport.release.notify_one();

    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();
    assert_eq!(first.access_token().expose_secret(), "new-access");
    assert_eq!(second.access_token().expose_secret(), "new-access");
    assert_eq!(first.refresh_token().expose_secret(), "new-refresh");
    assert_eq!(second.refresh_token().expose_secret(), "new-refresh");
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
}
