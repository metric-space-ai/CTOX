//! Synchrones TLS fuer die nativen Mail-Pfade (IMAP/SMTP/STARTTLS).
//!
//! Ersetzt `native_tls` mit derselben Aufrufflaeche. Der Grund ist kein
//! Geschmack, sondern der Binder: `native-tls` zieht auf Linux OpenSSL, und
//! OpenSSL kollidiert mit dem BoringSSL aus `wreq` (beide beanspruchen
//! `-lssl`/`-lcrypto`, Cargo entdoppelt, und dem Gewinner fehlen die Symbole
//! des Verlierers). Vertrauensanker bleiben die Systemzertifikate, damit sich
//! am Vertrauensverhalten nichts aendert.
//!
//! Unterschied zu `native_tls`, bewusst in Kauf genommen: der Handschlag
//! passiert hier verzoegert beim ersten Lesen oder Schreiben, nicht schon in
//! `connect`. Alle Aufrufer lesen unmittelbar danach (IMAP-Begruessung,
//! SMTP-Banner), der Fehler taucht also weiterhin auf — nur eine Zeile spaeter.

use std::io::{Read, Write};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};

/// Verschluesselter Strom ueber einem beliebigen Transport.
pub type TlsStream<S> = StreamOwned<ClientConnection, S>;

/// Wiederverwendbare Klientenkonfiguration mit den Systemzertifikaten.
pub struct TlsConnector {
    config: Arc<ClientConfig>,
}

impl TlsConnector {
    pub fn new() -> Result<Self> {
        let mut roots = RootCertStore::empty();
        let loaded = rustls_native_certs::load_native_certs();
        for cert in loaded.certs {
            // Einzelne unlesbare Anker sind kein Grund aufzugeben; ein leerer
            // Speicher dagegen schon (siehe unten).
            let _ = roots.add(cert);
        }
        if roots.is_empty() {
            let details = loaded
                .errors
                .iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            bail!("no system trust anchors available for TLS: {details}");
        }

        // Anbieter explizit mitgeben, statt sich auf einen prozessweit
        // installierten zu verlassen — sonst haengt der Mail-Pfad davon ab,
        // ob irgendwo sonst im Programm `install_default()` lief.
        let config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .context("failed to select TLS protocol versions")?
                .with_root_certificates(roots)
                .with_no_client_auth();

        Ok(Self {
            config: Arc::new(config),
        })
    }

    pub fn connect<S: Read + Write>(&self, domain: &str, stream: S) -> Result<TlsStream<S>> {
        let server_name = ServerName::try_from(domain.to_owned())
            .map_err(|error| anyhow!("invalid TLS server name {domain}: {error}"))?;
        let connection = ClientConnection::new(self.config.clone(), server_name)
            .context("failed to start TLS session")?;
        Ok(StreamOwned::new(connection, stream))
    }
}
