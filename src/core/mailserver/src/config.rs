// ref: stalwart/src/config/mod.rs:1-40
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MailserverRuntimeSettings {
    pub enabled: bool,
    pub hostname: String,
    pub bind_host: String,
    pub smtp_port: u16,
    pub imap_port: u16,
    pub outbound_throttle_per_min: usize,
    pub max_connections: usize,
    pub tracking_base_url: String,
}

impl Default for MailserverRuntimeSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            hostname: "localhost".to_string(),
            bind_host: "127.0.0.1".to_string(),
            smtp_port: 2525,
            imap_port: 1143,
            outbound_throttle_per_min: 120,
            max_connections: 10,
            tracking_base_url: String::new(),
        }
    }
}

impl MailserverRuntimeSettings {
    pub fn stalwart_config(&self, db_path: String) -> Result<StalwartConfig, String> {
        let ip = self
            .bind_host
            .parse()
            .map_err(|_| "bind_host must be an IPv4 or IPv6 address".to_string())?;
        Ok(StalwartConfig {
            server: ServerConfig {
                host: self.hostname.clone(),
                db_path,
            },
            smtp: SmtpConfig {
                bind_address: SocketAddr::new(ip, self.smtp_port),
                outbound_throttle_per_min: self.outbound_throttle_per_min.max(1),
                max_connections: self.max_connections.max(1),
            },
            imap: ImapConfig {
                bind_address: SocketAddr::new(ip, self.imap_port),
            },
            ..StalwartConfig::default()
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StalwartConfig {
    pub server: ServerConfig,
    pub smtp: SmtpConfig,
    pub imap: ImapConfig,
    pub caldav: CalDavConfig,
    pub carddav: CardDavConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub db_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub bind_address: SocketAddr,
    pub outbound_throttle_per_min: usize,
    pub max_connections: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImapConfig {
    pub bind_address: SocketAddr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalDavConfig {
    pub bind_address: SocketAddr,
    pub enable_scheduling: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardDavConfig {
    pub bind_address: SocketAddr,
}

impl Default for StalwartConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "localhost".to_string(),
                db_path: "runtime/ctox.sqlite3".to_string(),
            },
            smtp: SmtpConfig {
                bind_address: "127.0.0.1:25".parse().unwrap(),
                outbound_throttle_per_min: 120,
                max_connections: 10,
            },
            imap: ImapConfig {
                bind_address: "127.0.0.1:1143".parse().unwrap(),
            },
            caldav: CalDavConfig {
                bind_address: "127.0.0.1:8080".parse().unwrap(),
                enable_scheduling: true,
            },
            carddav: CardDavConfig {
                bind_address: "127.0.0.1:8081".parse().unwrap(),
            },
        }
    }
}
