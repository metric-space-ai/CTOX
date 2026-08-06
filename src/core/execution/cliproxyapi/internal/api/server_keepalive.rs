// ref: internal/api/server_keepalive.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::time::Duration;

use subtle::ConstantTimeEq;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{Instant, Sleep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAliveWatchdogError {
    InvalidTimeout,
}

impl fmt::Display for KeepAliveWatchdogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("keep-alive timeout must be positive")
    }
}

impl std::error::Error for KeepAliveWatchdogError {}

/// Owned replacement for the upstream server's keep-alive goroutine. Dropping
/// the handle requests shutdown; no global task or ambient password is used.
pub struct KeepAliveWatchdog {
    heartbeat: mpsc::Sender<()>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl KeepAliveWatchdog {
    // ref: internal/api/server_keepalive.go:12-29,61-87
    pub fn spawn<F>(timeout: Duration, on_timeout: F) -> Result<Self, KeepAliveWatchdogError>
    where
        F: FnOnce() + Send + 'static,
    {
        if timeout.is_zero() {
            return Err(KeepAliveWatchdogError::InvalidTimeout);
        }
        let (heartbeat, heartbeat_rx) = mpsc::channel(1);
        let (stop, stop_rx) = oneshot::channel();
        let task = tokio::spawn(watch_keep_alive(timeout, heartbeat_rx, stop_rx, on_timeout));
        Ok(Self {
            heartbeat,
            stop: Some(stop),
            task: Some(task),
        })
    }

    pub fn signal(&self) {
        let _ = self.heartbeat.try_send(());
    }

    pub async fn stop(mut self) {
        self.request_stop();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }

    fn request_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

impl Drop for KeepAliveWatchdog {
    fn drop(&mut self) {
        self.request_stop();
    }
}

impl fmt::Debug for KeepAliveWatchdog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KeepAliveWatchdog")
            .field("heartbeat", &"bounded")
            .field("stop", &self.stop.is_some())
            .field("task", &self.task.is_some())
            .finish()
    }
}

async fn watch_keep_alive<F>(
    timeout: Duration,
    mut heartbeat: mpsc::Receiver<()>,
    mut stop: oneshot::Receiver<()>,
    on_timeout: F,
) where
    F: FnOnce() + Send + 'static,
{
    let timer = tokio::time::sleep(timeout);
    tokio::pin!(timer);
    let mut on_timeout = Some(on_timeout);
    loop {
        tokio::select! {
            _ = &mut timer => {
                if let Some(on_timeout) = on_timeout.take() {
                    on_timeout();
                }
                return;
            }
            heartbeat = heartbeat.recv() => match heartbeat {
                Some(()) => reset_timer(&mut timer, timeout),
                None => return,
            },
            _ = &mut stop => return,
        }
    }
}

fn reset_timer(timer: &mut std::pin::Pin<&mut Sleep>, timeout: Duration) {
    timer.as_mut().reset(Instant::now() + timeout);
}

// ref: internal/api/server_keepalive.go:31-50
pub fn authorize_keep_alive(
    expected_password: &str,
    authorization: Option<&str>,
    local_password: Option<&str>,
) -> bool {
    if expected_password.is_empty() {
        return true;
    }
    let authorization = authorization.unwrap_or_default().trim();
    let provided = authorization
        .split_once(' ')
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, value)| value.trim())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            local_password
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_default();
    bool::from(provided.as_bytes().ct_eq(expected_password.as_bytes()))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn authentication_accepts_bearer_or_local_header_in_constant_time_path() {
        assert!(authorize_keep_alive("", None, None));
        assert!(authorize_keep_alive(
            "local-secret",
            Some("Bearer local-secret"),
            None
        ));
        assert!(authorize_keep_alive(
            "local-secret",
            Some(""),
            Some(" local-secret ")
        ));
        assert!(!authorize_keep_alive(
            "local-secret",
            Some("Basic local-secret"),
            None
        ));
        assert!(!authorize_keep_alive("local-secret", None, Some("wrong")));
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_invokes_callback_once() {
        let (fired, fired_rx) = oneshot::channel();
        let watchdog = KeepAliveWatchdog::spawn(Duration::from_millis(10), move || {
            let _ = fired.send(());
        })
        .unwrap();
        tokio::time::advance(Duration::from_millis(10)).await;
        fired_rx.await.unwrap();
        watchdog.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn signal_resets_timer_and_stop_suppresses_callback() {
        let fired = Arc::new(AtomicBool::new(false));
        let callback_fired = Arc::clone(&fired);
        let watchdog = KeepAliveWatchdog::spawn(Duration::from_millis(100), move || {
            callback_fired.store(true, Ordering::SeqCst);
        })
        .unwrap();
        tokio::time::advance(Duration::from_millis(60)).await;
        watchdog.signal();
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(60)).await;
        assert!(!fired.load(Ordering::SeqCst));
        watchdog.stop().await;
        assert!(!fired.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn zero_timeout_is_rejected_without_spawning() {
        assert_eq!(
            KeepAliveWatchdog::spawn(Duration::ZERO, || {}).unwrap_err(),
            KeepAliveWatchdogError::InvalidTimeout
        );
    }
}
