// ref: internal/watcher/dispatcher.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::runtime::AuthUpdate;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

pub trait AuthUpdateSink: Send + Sync {
    fn send_batch(&self, updates: &[AuthUpdate]) -> io::Result<()>;
}

enum DispatchMessage {
    Updates(Vec<AuthUpdate>),
    Stop,
}

pub struct AuthUpdateDispatcher {
    sender: mpsc::SyncSender<DispatchMessage>,
    worker: Mutex<Option<JoinHandle<()>>>,
    stopped: AtomicBool,
    dropped: AtomicU64,
}

impl AuthUpdateDispatcher {
    pub fn start(capacity: usize, sink: Arc<dyn AuthUpdateSink>) -> Self {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        let worker = thread::Builder::new()
            .name("cliproxy-auth-dispatch".into())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    match message {
                        DispatchMessage::Updates(updates) => {
                            let _ = sink.send_batch(&updates);
                        }
                        DispatchMessage::Stop => break,
                    }
                }
            })
            .expect("auth dispatch worker must start");
        Self {
            sender,
            worker: Mutex::new(Some(worker)),
            stopped: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
        }
    }

    pub fn dispatch(&self, updates: Vec<AuthUpdate>) -> bool {
        if updates.is_empty() {
            return true;
        }
        if self.stopped.load(Ordering::Acquire)
            || self
                .sender
                .try_send(DispatchMessage::Updates(updates))
                .is_err()
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    pub fn dropped_batches(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self.sender.send(DispatchMessage::Stop);
        if let Some(worker) = self.worker.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = worker.join();
        }
    }
}

impl Drop for AuthUpdateDispatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn normalize_auth(
    mut auth: super::synthesizer::context::SynthesizedAuth,
) -> super::synthesizer::context::SynthesizedAuth {
    auth.attributes.remove("updated_at");
    auth.metadata.remove("last_error");
    auth
}

pub fn auth_equal(
    a: &super::synthesizer::context::SynthesizedAuth,
    b: &super::synthesizer::context::SynthesizedAuth,
) -> bool {
    normalize_auth(a.clone()) == normalize_auth(b.clone())
}
