// ref: internal/home/certificate.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Home enrollment protocol with injected secret-file and transport owners.
//! CTOX chooses where credentials are persisted; this module never consults a
//! home directory or opens a network socket.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use aws_lc_rs::encoding::{AsDer, Pkcs8V1Der};
use aws_lc_rs::rsa::{KeyPair as RsaKeyPair, KeySize};
use aws_lc_rs::signature::KeyPair as _;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::client::{HomeConfig, HomeTlsConfig};

pub const CLIENT_CERT_NAME: &str = "client-crt.pem";
pub const CLIENT_KEY_NAME: &str = "client-key.pem";
pub const CA_CERT_NAME: &str = "home-ca-crt.pem";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CertificateError {
    InvalidJwt,
    MissingClaim(&'static str),
    InvalidTarget,
    InvalidPem,
    FingerprintMismatch,
    Store(String),
    Enrollment(String),
    Crypto(String),
}
impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJwt => f.write_str("home jwt is invalid"),
            Self::MissingClaim(name) => write!(f, "home jwt {name} is required"),
            Self::InvalidTarget => f.write_str("home jwt target address is invalid"),
            Self::InvalidPem => f.write_str("home ca certificate pem is invalid"),
            Self::FingerprintMismatch => f.write_str("home ca fingerprint mismatch"),
            Self::Store(e) | Self::Enrollment(e) | Self::Crypto(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for CertificateError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HomeJwtClaims {
    pub certificate_id: String,
    pub cluster_id: String,
    pub ca_fingerprint: String,
    pub enrollment_secret: String,
    pub ip: String,
    pub port: u16,
    pub iat: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrollmentRequest {
    pub certificate_id: String,
    pub cluster_id: String,
    pub enrollment_secret: String,
    pub public_key_pem: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertificateResponse {
    pub certificate: Vec<u8>,
    pub ca: Vec<u8>,
}

pub trait CertificateStore: Send + Sync {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>, CertificateError>;
    fn write_private(&self, name: &str, value: &[u8]) -> Result<(), CertificateError>;
}
pub trait CertificateEnrollment: Send + Sync {
    fn enroll(&self, request: EnrollmentRequest) -> Result<CertificateResponse, CertificateError>;
}

pub struct CertificateProvisioner {
    store: Arc<dyn CertificateStore>,
    enrollment: Arc<dyn CertificateEnrollment>,
}
impl CertificateProvisioner {
    pub fn new(
        store: Arc<dyn CertificateStore>,
        enrollment: Arc<dyn CertificateEnrollment>,
    ) -> Self {
        Self { store, enrollment }
    }
    pub fn config_from_jwt(&self, raw_jwt: &str) -> Result<HomeConfig, CertificateError> {
        let claims = parse_home_jwt_claims(raw_jwt)?;
        self.ensure_files(&claims)?;
        Ok(HomeConfig {
            enabled: true,
            node_id: claims.certificate_id.trim().into(),
            host: claims.ip.trim().into(),
            port: claims.port,
            tls: HomeTlsConfig {
                enabled: true,
                ca_cert: CA_CERT_NAME.into(),
                client_cert: CLIENT_CERT_NAME.into(),
                client_key: CLIENT_KEY_NAME.into(),
                use_target_server_name: true,
                ..HomeTlsConfig::default()
            },
        })
    }
    fn ensure_files(&self, claims: &HomeJwtClaims) -> Result<(), CertificateError> {
        let cert = self.store.read(CLIENT_CERT_NAME)?;
        let key = self.store.read(CLIENT_KEY_NAME)?;
        let ca = self.store.read(CA_CERT_NAME)?;
        if cert.is_some() && key.is_some() {
            return verify_ca_certificate_pem(
                ca.as_deref().ok_or(CertificateError::InvalidPem)?,
                &claims.ca_fingerprint,
            );
        }
        let (key, key_pem) = match key {
            Some(raw) => {
                let text = std::str::from_utf8(&raw)
                    .map_err(|e| CertificateError::Crypto(e.to_string()))?;
                let (label, der) = private_key_pem_parts(text)?;
                let key = match label {
                    "RSA PRIVATE KEY" => RsaKeyPair::from_der(&der),
                    "PRIVATE KEY" => RsaKeyPair::from_pkcs8(&der),
                    _ => {
                        return Err(CertificateError::Crypto(
                            "unsupported RSA key PEM label".into(),
                        ))
                    }
                }
                .map_err(|e| CertificateError::Crypto(format!("invalid RSA private key: {e}")))?;
                (key, encode_pem(label, &der))
            }
            None => {
                let key = RsaKeyPair::generate(KeySize::Rsa2048).map_err(|e| {
                    CertificateError::Crypto(format!("RSA key generation failed: {e}"))
                })?;
                let der = AsDer::<Pkcs8V1Der>::as_der(&key).map_err(|e| {
                    CertificateError::Crypto(format!("RSA key encoding failed: {e}"))
                })?;
                let pem = encode_pem("PRIVATE KEY", der.as_ref());
                (key, pem)
            }
        };
        let public_key_pem = encode_pem("RSA PUBLIC KEY", key.public_key().as_ref());
        let response = self.enrollment.enroll(EnrollmentRequest {
            certificate_id: claims.certificate_id.clone(),
            cluster_id: claims.cluster_id.clone(),
            enrollment_secret: claims.enrollment_secret.clone(),
            public_key_pem,
        })?;
        if response.certificate.is_empty() || response.ca.is_empty() {
            return Err(CertificateError::Enrollment(
                "home certificate response is incomplete".into(),
            ));
        }
        verify_ca_certificate_pem(&response.ca, &claims.ca_fingerprint)?;
        self.store
            .write_private(CLIENT_KEY_NAME, key_pem.as_bytes())?;
        self.store
            .write_private(CLIENT_CERT_NAME, &response.certificate)?;
        self.store.write_private(CA_CERT_NAME, &response.ca)
    }
}

fn private_key_pem_parts(value: &str) -> Result<(&str, Vec<u8>), CertificateError> {
    let begin = value
        .find("-----BEGIN ")
        .map(|index| index + "-----BEGIN ".len())
        .ok_or_else(|| CertificateError::Crypto("private key PEM header is missing".into()))?;
    let label_end = value[begin..]
        .find("-----")
        .map(|offset| begin + offset)
        .ok_or_else(|| CertificateError::Crypto("private key PEM label is invalid".into()))?;
    let label = &value[begin..label_end];
    let body_start = label_end + "-----".len();
    let footer = format!("-----END {label}-----");
    let body_end = value[body_start..]
        .find(&footer)
        .map(|offset| body_start + offset)
        .ok_or_else(|| CertificateError::Crypto("private key PEM footer is missing".into()))?;
    let payload = value[body_start..body_end]
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '/' | '=')
        })
        .collect::<String>();
    let der = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| CertificateError::Crypto("private key PEM payload is invalid".into()))?;
    Ok((label, der))
}

fn encode_pem(label: &str, der: &[u8]) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(der);
    let mut output = format!("-----BEGIN {label}-----\n");
    for chunk in encoded.as_bytes().chunks(64) {
        output.push_str(std::str::from_utf8(chunk).expect("base64 is valid ASCII"));
        output.push('\n');
    }
    output.push_str(&format!("-----END {label}-----\n"));
    output
}

pub fn parse_home_jwt_claims(raw: &str) -> Result<HomeJwtClaims, CertificateError> {
    let parts = raw.trim().split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(CertificateError::InvalidJwt);
    }
    let payload = decode_jwt_part(parts[1])?;
    let claims: HomeJwtClaims =
        serde_json::from_slice(&payload).map_err(|_| CertificateError::InvalidJwt)?;
    for (name, value) in [
        ("certificate_id", &claims.certificate_id),
        ("cluster_id", &claims.cluster_id),
        ("ca_fingerprint", &claims.ca_fingerprint),
        ("enrollment_secret", &claims.enrollment_secret),
    ] {
        if value.trim().is_empty() {
            return Err(CertificateError::MissingClaim(name));
        }
    }
    if claims.ip.trim().is_empty() || claims.port == 0 {
        return Err(CertificateError::InvalidTarget);
    }
    Ok(claims)
}
pub fn decode_jwt_part(part: &str) -> Result<Vec<u8>, CertificateError> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(part)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(part))
        .map_err(|_| CertificateError::InvalidJwt)
}
pub fn normalize_fingerprint(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace([':', ' '], "")
}
pub fn certificate_fingerprint_pem(raw: &[u8]) -> Result<String, CertificateError> {
    let text = std::str::from_utf8(raw).map_err(|_| CertificateError::InvalidPem)?;
    let body = text
        .lines()
        .skip_while(|line| !line.contains("BEGIN CERTIFICATE"))
        .skip(1)
        .take_while(|line| !line.contains("END CERTIFICATE"))
        .collect::<String>();
    if body.is_empty() {
        return Err(CertificateError::InvalidPem);
    }
    let der = base64::engine::general_purpose::STANDARD
        .decode(body)
        .map_err(|_| CertificateError::InvalidPem)?;
    Ok(format!("{:x}", Sha256::digest(der)))
}
pub fn verify_ca_certificate_pem(raw: &[u8], expected: &str) -> Result<(), CertificateError> {
    let expected = normalize_fingerprint(expected);
    if expected.is_empty() {
        return Err(CertificateError::MissingClaim("ca_fingerprint"));
    }
    if certificate_fingerprint_pem(raw)? == expected {
        Ok(())
    } else {
        Err(CertificateError::FingerprintMismatch)
    }
}

pub fn encode_resp_array(args: &[String]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        out.extend(format!("${}\r\n", arg.len()).as_bytes());
        out.extend(arg.as_bytes());
        out.extend(b"\r\n");
    }
    out
}

#[derive(Default)]
pub struct MemoryCertificateStore(pub std::sync::Mutex<BTreeMap<String, Vec<u8>>>);
impl CertificateStore for MemoryCertificateStore {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>, CertificateError> {
        Ok(self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(name)
            .cloned())
    }
    fn write_private(&self, name: &str, value: &[u8]) -> Result<(), CertificateError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.into(), value.to_vec());
        Ok(())
    }
}
