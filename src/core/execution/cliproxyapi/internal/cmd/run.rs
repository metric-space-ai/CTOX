// ref: internal/cmd/run.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_manager::CommandConfig;
use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretRef {
    pub scope: String,
    pub name: String,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServicePlan {
    pub config: CommandConfig,
    pub local_management_secret: Option<SecretRef>,
    pub plugin_host_id: Option<String>,
    pub options: BTreeMap<String, String>,
}
pub trait ProxyService: Send {
    fn run(&mut self, cancellation: &ServiceCancellation) -> io::Result<()>;
    fn shutdown(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub trait ServiceFactory: Send + Sync {
    fn build(&self, plan: &ServicePlan) -> io::Result<Box<dyn ProxyService>>;
}

#[derive(Debug, Default)]
pub struct ServiceCancellation {
    cancelled: AtomicBool,
    lock: Mutex<()>,
    changed: Condvar,
}
impl ServiceCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.changed.notify_all();
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    pub fn wait(&self) {
        let mut guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        while !self.is_cancelled() {
            guard = self.changed.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
    }
}

pub fn start_service(
    factory: &dyn ServiceFactory,
    plan: &ServicePlan,
    cancellation: &ServiceCancellation,
) -> io::Result<()> {
    let mut service = factory.build(plan)?;
    let result = service.run(cancellation);
    let shutdown = service.shutdown();
    match (result, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

pub struct BackgroundService {
    cancellation: Arc<ServiceCancellation>,
    worker: Option<JoinHandle<io::Result<()>>>,
}
impl BackgroundService {
    pub fn start(factory: Arc<dyn ServiceFactory>, plan: ServicePlan) -> io::Result<Self> {
        let cancellation = Arc::new(ServiceCancellation::default());
        let worker_cancel = Arc::clone(&cancellation);
        let worker = thread::Builder::new()
            .name("cliproxy-service-command".into())
            .spawn(move || start_service(factory.as_ref(), &plan, worker_cancel.as_ref()))?;
        Ok(Self {
            cancellation,
            worker: Some(worker),
        })
    }
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
    pub fn join(mut self) -> io::Result<()> {
        self.cancel();
        self.join_inner()
    }
    fn join_inner(&mut self) -> io::Result<()> {
        self.worker.take().map_or(Ok(()), |worker| {
            worker
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("service command worker panicked")))
        })
    }
}
impl Drop for BackgroundService {
    fn drop(&mut self) {
        self.cancel();
        let _ = self.join_inner();
    }
}

pub fn wait_for_cloud_deploy(cancellation: &ServiceCancellation) {
    cancellation.wait();
}
