// ref: stalwart/src/lib.rs:1-25
pub mod config;
pub mod util;

pub mod calcard;
pub mod caldav;
pub mod carddav;
pub mod imap;
pub mod smtp;
pub mod store;

pub use config::{MailserverRuntimeSettings, StalwartConfig};
pub use imap::ImapServer;
pub use util::errors::{StalwartError, StalwartResult};

use serde::Serialize;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

enum RuntimeCommand {
    Apply(MailserverRuntimeSettings),
    Stop,
}

static RUNTIME_CONTROL: OnceLock<mpsc::UnboundedSender<RuntimeCommand>> = OnceLock::new();
static RUNTIME_STATUS: OnceLock<Mutex<MailserverRuntimeStatus>> = OnceLock::new();

#[derive(Clone, Debug, Serialize)]
pub struct MailserverRuntimeStatus {
    pub state: String,
    pub enabled: bool,
    pub generation: u64,
    pub started_at: Option<i64>,
    pub last_error: Option<String>,
}

impl Default for MailserverRuntimeStatus {
    fn default() -> Self {
        Self {
            state: "not_started".to_string(),
            enabled: false,
            generation: 0,
            started_at: None,
            last_error: None,
        }
    }
}

fn status_store() -> &'static Mutex<MailserverRuntimeStatus> {
    RUNTIME_STATUS.get_or_init(|| Mutex::new(MailserverRuntimeStatus::default()))
}

pub fn runtime_status() -> MailserverRuntimeStatus {
    status_store()
        .lock()
        .map(|status| status.clone())
        .unwrap_or_default()
}

pub fn apply_runtime_settings(settings: MailserverRuntimeSettings) -> StalwartResult<()> {
    let sender = RUNTIME_CONTROL.get().ok_or_else(|| {
        StalwartError::General("mailserver runtime supervisor is not initialized".to_string())
    })?;
    sender
        .send(RuntimeCommand::Apply(settings))
        .map_err(|_| StalwartError::General("mailserver runtime supervisor stopped".to_string()))
}

pub fn stop_runtime() -> StalwartResult<()> {
    let sender = RUNTIME_CONTROL.get().ok_or_else(|| {
        StalwartError::General("mailserver runtime supervisor is not initialized".to_string())
    })?;
    sender
        .send(RuntimeCommand::Stop)
        .map_err(|_| StalwartError::General("mailserver runtime supervisor stopped".to_string()))
}

pub fn start_services_thread(db_path: String) {
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    if RUNTIME_CONTROL.set(command_tx).is_err() {
        tracing::warn!("[ctox-mailserver] runtime supervisor already started");
        return;
    }

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for mailserver");
        rt.block_on(run_runtime_supervisor(db_path, command_rx));
    });
}

async fn run_runtime_supervisor(
    db_path: String,
    mut command_rx: mpsc::UnboundedReceiver<RuntimeCommand>,
) {
    let store = store::SqliteStore::new(&db_path);
    if let Err(error) = store.init() {
        set_runtime_error(format!("failed to initialize mailserver store: {error}"));
        return;
    }

    let initial_settings = match store.load_runtime_settings() {
        Ok(settings) => settings,
        Err(error) => {
            set_runtime_error(format!("failed to load mailserver configuration: {error}"));
            MailserverRuntimeSettings::default()
        }
    };

    // Calendar/address-book services remain collaboration services with their
    // own lifecycle. The runtime controls below deliberately govern only SMTP,
    // IMAP and the outbound queue shown in the Mail app.
    start_collaboration_services(store.clone(), &db_path).await;

    let mut email_tasks = Vec::new();
    apply_email_runtime(&store, &db_path, &initial_settings, &mut email_tasks).await;

    while let Some(command) = command_rx.recv().await {
        match command {
            RuntimeCommand::Apply(settings) => {
                apply_email_runtime(&store, &db_path, &settings, &mut email_tasks).await;
            }
            RuntimeCommand::Stop => {
                stop_email_tasks(&mut email_tasks);
                let mut status = status_store().lock().unwrap_or_else(|err| err.into_inner());
                status.state = "stopped".to_string();
                status.enabled = false;
                status.generation += 1;
                status.started_at = None;
                status.last_error = None;
            }
        }
    }
}

async fn start_collaboration_services(store: store::SqliteStore, db_path: &str) {
    let mut config = StalwartConfig::default();
    config.server.db_path = db_path.to_string();
    let caldav = Arc::new(caldav::CalDavServer::new(store.clone(), config.caldav));
    tokio::spawn(async move {
        if let Err(error) = caldav.start().await {
            tracing::error!("[ctox-mailserver] CalDAV failed: {error}");
        }
    });
    let carddav = Arc::new(carddav::CardDavServer::new(store, config.carddav));
    tokio::spawn(async move {
        if let Err(error) = carddav.start().await {
            tracing::error!("[ctox-mailserver] CardDAV failed: {error}");
        }
    });
}

async fn apply_email_runtime(
    store: &store::SqliteStore,
    db_path: &str,
    settings: &MailserverRuntimeSettings,
    tasks: &mut Vec<tokio::task::JoinHandle<()>>,
) {
    stop_email_tasks(tasks);
    if !settings.enabled {
        let mut status = status_store().lock().unwrap_or_else(|err| err.into_inner());
        status.state = "stopped".to_string();
        status.enabled = false;
        status.generation += 1;
        status.started_at = None;
        status.last_error = None;
        return;
    }

    let config = match settings.stalwart_config(db_path.to_string()) {
        Ok(config) => config,
        Err(error) => {
            set_runtime_error(error);
            return;
        }
    };
    tracing::info!(
        "[ctox-mailserver] applying SMTP {} and IMAP {}",
        config.smtp.bind_address,
        config.imap.bind_address
    );

    let smtp = Arc::new(smtp::server::SmtpInboundServer::new(
        store.clone(),
        config.smtp.clone(),
    ));
    tasks.push(tokio::spawn(async move {
        if let Err(error) = smtp.start().await {
            set_runtime_error(format!("SMTP listener failed: {error}"));
        }
    }));

    let imap = Arc::new(imap::ImapServer::new(store.clone(), config.imap));
    tasks.push(tokio::spawn(async move {
        if let Err(error) = imap.start().await {
            set_runtime_error(format!("IMAP listener failed: {error}"));
        }
    }));

    let queue = Arc::new(smtp::client_queue::SmtpOutboundQueue::new(
        store.clone(),
        config.smtp,
    ));
    tasks.push(tokio::spawn(async move { queue.start().await }));

    let mut status = status_store().lock().unwrap_or_else(|err| err.into_inner());
    status.state = "running".to_string();
    status.enabled = true;
    status.generation += 1;
    status.started_at = Some(chrono::Utc::now().timestamp_millis());
    status.last_error = None;
}

fn stop_email_tasks(tasks: &mut Vec<tokio::task::JoinHandle<()>>) {
    for task in tasks.drain(..) {
        task.abort();
    }
}

fn set_runtime_error(error: String) {
    tracing::error!("[ctox-mailserver] {error}");
    let mut status = status_store().lock().unwrap_or_else(|err| err.into_inner());
    status.state = "degraded".to_string();
    status.last_error = Some(error);
}
