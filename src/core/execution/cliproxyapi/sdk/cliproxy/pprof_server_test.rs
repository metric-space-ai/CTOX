// ref: sdk/cliproxy/pprof_server_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::pprof_server::*;

struct Listener {
    stops: AtomicUsize,
}

impl Listener {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            stops: AtomicUsize::new(0),
        })
    }
}

impl PprofListener for Listener {
    fn serve(self: Arc<Self>) -> ListenerFuture {
        Box::pin(std::future::pending())
    }
    fn shutdown(&self, _timeout: Duration) -> ListenerFuture {
        self.stops.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { Ok(()) })
    }
}

struct Factory;
impl PprofListenerFactory for Factory {
    fn bind(
        &self,
        _addr: &str,
        _routes: &'static [&'static str],
    ) -> Result<Arc<dyn PprofListener>, PprofError> {
        Ok(Listener::new())
    }
}

fn pprof() -> PprofServer {
    PprofServer::new(Arc::new(Factory), Duration::from_secs(5))
}

#[tokio::test]
async fn pprof_server_stop_owned_server_keeps_replacement() {
    let pprof = pprof();
    let old: Arc<dyn PprofListener> = Listener::new();
    let replacement: Arc<dyn PprofListener> = Listener::new();
    pprof.seed(replacement.clone(), "replacement", true, 2);
    pprof.stop_owned_server(old, 1).await.unwrap();
    assert!(pprof
        .snapshot()
        .0
        .as_ref()
        .is_some_and(|server| Arc::ptr_eq(server, &replacement)));
}

#[tokio::test]
async fn pprof_server_same_pointer_owner_transfer_keeps_current_server() {
    let pprof = pprof();
    let server: Arc<dyn PprofListener> = Listener::new();
    pprof.seed(server.clone(), "127.0.0.1:6060", true, 1);
    assert!(
        pprof
            .apply_context(&PprofConfig {
                enable: true,
                addr: "127.0.0.1:6060".into()
            })
            .await
    );
    assert_ne!(pprof.snapshot().1, 1);
    pprof.stop_owned_server(server.clone(), 1).await.unwrap();
    assert!(pprof
        .snapshot()
        .0
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, &server)));
}

#[test]
fn pprof_server_serve_failure_clears_transferred_owner() {
    let pprof = pprof();
    let server: Arc<dyn PprofListener> = Listener::new();
    pprof.seed(server.clone(), "", true, 2);
    pprof.clear_failed_server(&server);
    assert!(pprof.snapshot().0.is_none());
}
