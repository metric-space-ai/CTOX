// ref: stalwart/src/lib.rs:1-25
pub mod config;
pub mod util;

pub mod calcard;
pub mod caldav;
pub mod carddav;
pub mod imap;
pub mod smtp;
pub mod store;

pub use config::StalwartConfig;
pub use imap::ImapServer;
pub use util::errors::{StalwartError, StalwartResult};

use std::sync::{Arc, OnceLock, RwLock};

pub use store::sqlite::DeliveryOutcome;

pub type DeliveryOutcomeSink =
    Arc<dyn Fn(&DeliveryOutcome) -> Result<(), String> + Send + Sync + 'static>;

static DELIVERY_OUTCOME_SINK: OnceLock<RwLock<Option<DeliveryOutcomeSink>>> = OnceLock::new();

/// Register the process-local consumer for terminal SMTP outcomes. Business OS
/// installs this before the mailserver thread starts; replacing it is allowed so
/// tests and restarted service roots do not retain a stale path.
pub fn register_delivery_outcome_sink(sink: DeliveryOutcomeSink) {
    let slot = DELIVERY_OUTCOME_SINK.get_or_init(|| RwLock::new(None));
    let mut guard = slot
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = Some(sink);
}

fn registered_delivery_outcome_sink() -> Option<DeliveryOutcomeSink> {
    DELIVERY_OUTCOME_SINK
        .get()
        .and_then(|slot| slot.read().ok().and_then(|guard| guard.clone()))
}

pub fn start_services_thread(db_path: String) {
    let delivery_outcome_sink = registered_delivery_outcome_sink();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for mailserver");

        rt.block_on(async {
            tracing::info!("[ctox-mailserver] Starting mailserver thread with DB: {}", db_path);
            println!("[ctox-mailserver] Starting mailserver thread with DB: {}", db_path);
            let store = store::SqliteStore::new(&db_path);
            if let Err(e) = store.init() {
                tracing::error!("[ctox-mailserver] ERROR: Failed to initialize mailserver SQLite store: {:?}", e);
                eprintln!("[ctox-mailserver] ERROR: Failed to initialize mailserver SQLite store: {:?}", e);
                return;
            }
            tracing::info!("[ctox-mailserver] SQLite store initialized successfully");
            println!("[ctox-mailserver] SQLite store initialized successfully");

            let mut config = StalwartConfig::default();
            config.server.db_path = db_path;

            // Optional port configurations from env variables
            if let Ok(smtp_port) = std::env::var("CTOX_SMTP_PORT") {
                if let Ok(port) = smtp_port.parse::<u16>() {
                    config.smtp.bind_address = std::net::SocketAddr::new(
                        "127.0.0.1".parse().unwrap(),
                        port
                    );
                }
            } else {
                config.smtp.bind_address = "127.0.0.1:2525".parse().unwrap();
            }

            if let Ok(imap_port) = std::env::var("CTOX_IMAP_PORT") {
                if let Ok(port) = imap_port.parse::<u16>() {
                    config.imap.bind_address = std::net::SocketAddr::new(
                        "127.0.0.1".parse().unwrap(),
                        port
                    );
                }
            } else {
                config.imap.bind_address = "127.0.0.1:1143".parse().unwrap();
            }

            if let Ok(caldav_port) = std::env::var("CTOX_CALDAV_PORT") {
                if let Ok(port) = caldav_port.parse::<u16>() {
                    config.caldav.bind_address = std::net::SocketAddr::new(
                        "127.0.0.1".parse().unwrap(),
                        port
                    );
                }
            } else {
                config.caldav.bind_address = "127.0.0.1:8080".parse().unwrap();
            }

            if let Ok(carddav_port) = std::env::var("CTOX_CARDDAV_PORT") {
                if let Ok(port) = carddav_port.parse::<u16>() {
                    config.carddav.bind_address = std::net::SocketAddr::new(
                        "127.0.0.1".parse().unwrap(),
                        port
                    );
                }
            } else {
                config.carddav.bind_address = "127.0.0.1:8081".parse().unwrap();
            }

            tracing::info!(
                "[ctox-mailserver] Starting services: SMTP on {}, IMAP on {}, CalDAV on {}, CardDAV on {}",
                config.smtp.bind_address, config.imap.bind_address, config.caldav.bind_address, config.carddav.bind_address
            );
            println!(
                "[ctox-mailserver] Starting services: SMTP on {}, IMAP on {}, CalDAV on {}, CardDAV on {}",
                config.smtp.bind_address, config.imap.bind_address, config.caldav.bind_address, config.carddav.bind_address
            );

            // Start Inbound SMTP Server
            let smtp_server = Arc::new(smtp::server::SmtpInboundServer::new(
                store.clone(),
                config.smtp.clone(),
            ));
            tokio::spawn(async move {
                tracing::info!("[ctox-mailserver] SMTP Inbound Server thread running");
                if let Err(e) = smtp_server.start().await {
                    tracing::error!("[ctox-mailserver] SMTP Inbound Server failed to start: {:?}", e);
                    eprintln!("SMTP Inbound Server failed to start: {:?}", e);
                }
            });

            // Start IMAP Server
            let imap_server = Arc::new(imap::ImapServer::new(
                store.clone(),
                config.imap.clone(),
            ));
            tokio::spawn(async move {
                tracing::info!("[ctox-mailserver] IMAP Server thread running");
                if let Err(e) = imap_server.start().await {
                    tracing::error!("[ctox-mailserver] IMAP Server failed to start: {:?}", e);
                    eprintln!("IMAP Server failed to start: {:?}", e);
                }
            });

            // Start CalDAV Server
            let caldav_server = Arc::new(caldav::CalDavServer::new(
                store.clone(),
                config.caldav.clone(),
            ));
            tokio::spawn(async move {
                tracing::info!("[ctox-mailserver] CalDAV Server thread running");
                if let Err(e) = caldav_server.start().await {
                    tracing::error!("[ctox-mailserver] CalDAV Server failed to start: {:?}", e);
                    eprintln!("CalDAV Server failed to start: {:?}", e);
                }
            });

            // Start CardDAV Server
            let carddav_server = Arc::new(carddav::CardDavServer::new(
                store.clone(),
                config.carddav.clone(),
            ));
            tokio::spawn(async move {
                tracing::info!("[ctox-mailserver] CardDAV Server thread running");
                if let Err(e) = carddav_server.start().await {
                    tracing::error!("[ctox-mailserver] CardDAV Server failed to start: {:?}", e);
                    eprintln!("CardDAV Server failed to start: {:?}", e);
                }
            });

            // Start SMTP Outbound Queue Runner
            let queue_runner = Arc::new(
                smtp::client_queue::SmtpOutboundQueue::new_with_delivery_outcome_sink(
                    store.clone(),
                    config.smtp.clone(),
                    delivery_outcome_sink,
                ),
            );
            tokio::spawn(async move {
                tracing::info!("[ctox-mailserver] SMTP Outbound Queue Runner thread running");
                queue_runner.start().await;
            });

            // Keep the tokio runtime alive
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
            }
        });
    });
}
