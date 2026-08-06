// ref: internal/cmd/vertex_import.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{Map, Value};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

pub trait ImportFilesystem: Send + Sync {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
}
#[derive(Debug, Default)]
pub struct NativeImportFilesystem;
impl ImportFilesystem for NativeImportFilesystem {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }
}
pub trait VertexCredentialSink: Send + Sync {
    fn save(&self, auth_dir: &Path, record: &VertexCredentialRecord) -> io::Result<PathBuf>;
}

#[derive(Clone, PartialEq)]
pub struct VertexCredentialRecord {
    pub id: String,
    pub prefix: String,
    pub project_id: String,
    pub email: String,
    pub location: String,
    pub service_account: Map<String, Value>,
}
impl fmt::Debug for VertexCredentialRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexCredentialRecord")
            .field("id", &self.id)
            .field("prefix", &self.prefix)
            .field("project_id", &self.project_id)
            .field("email", &self.email)
            .field("location", &self.location)
            .field(
                "service_account_keys",
                &self.service_account.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexImportPlan {
    pub key_path: PathBuf,
    pub auth_dir: PathBuf,
    pub prefix: String,
    pub location: String,
}

pub fn execute_vertex_import(
    plan: &VertexImportPlan,
    filesystem: &dyn ImportFilesystem,
    sink: &dyn VertexCredentialSink,
) -> io::Result<PathBuf> {
    if plan.key_path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "service account key path is required",
        ));
    }
    let prefix = normalize_prefix(&plan.prefix)?;
    let data = filesystem.read(&plan.key_path)?;
    let service_account = serde_json::from_slice::<Value>(&data)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "service account must be a JSON object",
            )
        })?;
    let project_id = required_string(&service_account, "project_id")?;
    let email = service_account
        .get("client_email")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    required_string(&service_account, "private_key")?;
    let base = sanitize_file_part(&project_id);
    let id = if prefix.is_empty() {
        format!("vertex-{base}.json")
    } else {
        format!("vertex-{}-{base}.json", sanitize_file_part(&prefix))
    };
    let record = VertexCredentialRecord {
        id,
        prefix,
        project_id,
        email,
        location: if plan.location.trim().is_empty() {
            "us-central1".into()
        } else {
            plan.location.trim().to_owned()
        },
        service_account,
    };
    sink.save(&plan.auth_dir, &record)
}
pub fn sanitize_file_part(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}
pub fn label_for_vertex(project_id: &str, email: &str) -> String {
    match (project_id.trim(), email.trim()) {
        ("", "") => "vertex".into(),
        (project, "") => project.into(),
        ("", email) => email.into(),
        (project, email) => format!("{project} ({email})"),
    }
}
fn normalize_prefix(prefix: &str) -> io::Result<String> {
    let prefix = prefix.trim().trim_matches('/');
    if prefix.contains('/') {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vertex prefix must be one segment",
        ))
    } else {
        Ok(prefix.to_owned())
    }
}
fn required_string(object: &Map<String, Value>, key: &str) -> io::Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{key} missing in service account"),
            )
        })
}
