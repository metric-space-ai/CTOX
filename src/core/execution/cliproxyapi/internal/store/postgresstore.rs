// ref: internal/store/postgresstore.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! PostgreSQL-shaped store with an injected persistence authority.
//!
//! Upstream opens a DSN directly and owns a PostgreSQL pool. CTOX keeps that
//! authority in the host: this module owns the spool, serialization, path and
//! lifecycle semantics while [`PostgresBackend`] performs atomic database
//! operations. A host adapter can use PostgreSQL or map these records into the
//! existing CTOX runtime store without giving this portable crate credentials.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::internal::credentialweight::{parse_string, parse_value};
use crate::sdk::cliproxy::auth::{
    apply_custom_headers_from_metadata, Auth, AuthStatus, AuthStore, AuthStoreError,
};

pub const DEFAULT_CONFIG_TABLE: &str = "config_store";
pub const DEFAULT_AUTH_TABLE: &str = "auth_store";
pub const DEFAULT_COOLDOWN_TABLE: &str = "cooldown_store";
pub const DEFAULT_CONFIG_KEY: &str = "config";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostgresStoreConfig {
    pub schema: String,
    pub config_table: String,
    pub auth_table: String,
    pub cooldown_table: String,
    pub spool_dir: PathBuf,
}

impl PostgresStoreConfig {
    pub fn normalized(mut self) -> Result<Self, StoreError> {
        if self.config_table.trim().is_empty() {
            self.config_table = DEFAULT_CONFIG_TABLE.to_owned();
        }
        if self.auth_table.trim().is_empty() {
            self.auth_table = DEFAULT_AUTH_TABLE.to_owned();
        }
        if self.cooldown_table.trim().is_empty() {
            self.cooldown_table = DEFAULT_COOLDOWN_TABLE.to_owned();
        }
        for identifier in [
            self.schema.as_str(),
            self.config_table.as_str(),
            self.auth_table.as_str(),
            self.cooldown_table.as_str(),
        ] {
            validate_identifier(identifier)?;
        }
        if self.spool_dir.as_os_str().is_empty() || !self.spool_dir.is_absolute() {
            return Err(StoreError::InvalidConfig(
                "postgres store spool_dir must be an absolute CTOX-owned path",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredAuthRecord {
    pub id: String,
    pub content: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CooldownRow {
    pub auth_id: String,
    pub model: String,
    pub content: Vec<u8>,
    pub deleted: bool,
    pub updated_at_ms: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CooldownMutation {
    pub row: CooldownRow,
    /// For tombstones, only delete the version observed by this store. This is
    /// the upstream compare-and-set guard that prevents a stale instance from
    /// erasing a newer update.
    pub delete_if_not_newer_than_ms: Option<i64>,
}

/// Database authority supplied by CTOX. Each method is one logical database
/// transaction; `apply_cooldown_mutations` must apply its complete slice
/// atomically and enforce the version guards.
pub trait PostgresBackend: Send + Sync {
    fn ensure_schema(&self, config: &PostgresStoreConfig) -> Result<(), StoreError>;
    fn load_config(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError>;
    fn put_config(&self, key: &str, content: &[u8]) -> Result<(), StoreError>;
    fn delete_config(&self, key: &str) -> Result<(), StoreError>;
    fn list_auth(&self) -> Result<Vec<StoredAuthRecord>, StoreError>;
    fn put_auth(&self, id: &str, content: &[u8]) -> Result<(), StoreError>;
    fn delete_auth(&self, id: &str) -> Result<(), StoreError>;
    fn load_cooldowns(&self) -> Result<Vec<CooldownRow>, StoreError>;
    fn apply_cooldown_mutations(&self, mutations: &[CooldownMutation]) -> Result<(), StoreError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoreError {
    InvalidConfig(&'static str),
    InvalidIdentifier(String),
    InvalidRecord(String),
    PathOutsideStore(PathBuf),
    Io(String),
    Backend(String),
    Serialization(String),
    TokenStorage(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => formatter.write_str(message),
            Self::InvalidIdentifier(value) => write!(formatter, "invalid SQL identifier {value:?}"),
            Self::InvalidRecord(message) => write!(formatter, "invalid store record: {message}"),
            Self::PathOutsideStore(path) => {
                write!(formatter, "path outside managed store: {}", path.display())
            }
            Self::Io(message) => write!(formatter, "store I/O failed: {message}"),
            Self::Backend(message) => write!(formatter, "store backend failed: {message}"),
            Self::Serialization(message) => {
                write!(formatter, "store serialization failed: {message}")
            }
            Self::TokenStorage(message) => write!(formatter, "token storage failed: {message}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<io::Error> for StoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

pub struct PostgresStore {
    backend: Arc<dyn PostgresBackend>,
    cfg: PostgresStoreConfig,
    spool_root: PathBuf,
    config_path: PathBuf,
    auth_dir: PathBuf,
    operation_lock: Mutex<()>,
}

impl fmt::Debug for PostgresStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PostgresStore")
            .field("cfg", &self.cfg)
            .field("spool_root", &self.spool_root)
            .finish_non_exhaustive()
    }
}

impl PostgresStore {
    pub fn new(
        cfg: PostgresStoreConfig,
        backend: Arc<dyn PostgresBackend>,
    ) -> Result<Self, StoreError> {
        let cfg = cfg.normalized()?;
        let spool_root = cfg.spool_dir.clone();
        let config_path = spool_root.join("config/config.yaml");
        let auth_dir = spool_root.join("auths");
        create_private_dir(config_path.parent().expect("config path has parent"))?;
        create_private_dir(&auth_dir)?;
        Ok(Self {
            backend,
            cfg,
            spool_root,
            config_path,
            auth_dir,
            operation_lock: Mutex::new(()),
        })
    }

    pub fn ensure_schema(&self) -> Result<(), StoreError> {
        self.backend.ensure_schema(&self.cfg)
    }

    pub fn bootstrap(&self, example_config_path: Option<&Path>) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_schema()?;
        self.sync_config_from_database(example_config_path)?;
        self.sync_auth_from_database()
    }

    #[must_use]
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    #[must_use]
    pub fn auth_dir(&self) -> &Path {
        &self.auth_dir
    }

    #[must_use]
    pub fn work_dir(&self) -> &Path {
        &self.spool_root
    }

    pub fn save(&self, auth: &mut Auth) -> Result<PathBuf, StoreError> {
        validate_auth(auth)?;
        let path = self.resolve_auth_path(auth)?;
        if auth.disabled && !path.exists() {
            return Ok(PathBuf::new());
        }
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        write_auth_file(&path, auth)?;
        auth.attributes
            .insert("path".to_owned(), path.to_string_lossy().into_owned());
        auth.attributes
            .insert("source_backend".to_owned(), "postgres".to_owned());
        if auth.file_name.trim().is_empty() {
            auth.file_name.clone_from(&auth.id);
        }
        let rel_id = self.relative_auth_id(&path)?;
        let data = fs::read(&path)?;
        if data.is_empty() {
            self.backend.delete_auth(&rel_id)?;
        } else {
            self.backend.put_auth(&rel_id, &data)?;
        }
        Ok(path)
    }

    pub fn list(&self) -> Result<Vec<Auth>, StoreError> {
        let mut rows = self.backend.list_auth()?;
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        let mut auths = Vec::with_capacity(rows.len());
        for row in rows {
            let Ok(path) = self.absolute_auth_path(&row.id) else {
                continue;
            };
            let Ok(mut metadata) = serde_json::from_slice::<BTreeMap<String, Value>>(&row.content)
            else {
                continue;
            };
            if validate_metadata_weight(&metadata).is_err() {
                continue;
            }
            let provider = string_value(metadata.get("type")).unwrap_or("unknown");
            let disabled = metadata
                .get("disabled")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let mut attributes = BTreeMap::from([
                ("path".to_owned(), path.to_string_lossy().into_owned()),
                ("source_backend".to_owned(), "postgres".to_owned()),
            ]);
            if let Some(email) = string_value(metadata.get("email")) {
                attributes.insert("email".to_owned(), email.to_owned());
            }
            let mut auth = Auth::default();
            auth.id = normalize_auth_id(&row.id);
            auth.provider = provider.to_owned();
            auth.file_name = normalize_auth_id(&row.id);
            auth.label = label_for(&metadata);
            auth.status = if disabled {
                AuthStatus::Disabled
            } else {
                AuthStatus::Active
            };
            auth.disabled = disabled;
            auth.attributes = attributes;
            auth.metadata = std::mem::take(&mut metadata);
            auth.created_at = row.created_at;
            auth.updated_at = row.updated_at;
            apply_custom_headers_from_metadata(&mut auth);
            auths.push(auth);
        }
        Ok(auths)
    }

    pub fn delete(&self, id: &str) -> Result<(), StoreError> {
        let path = self.resolve_delete_path(id)?;
        let rel_id = self.relative_auth_id(&path)?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.backend.delete_auth(&rel_id)
    }

    pub fn persist_auth_files<I, P>(&self, paths: I) -> Result<(), StoreError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for path in paths {
            let input = path.as_ref();
            if input.as_os_str().is_empty() {
                continue;
            }
            let absolute = if input.is_absolute() {
                input.to_path_buf()
            } else {
                self.auth_dir.join(input)
            };
            let rel_id = self.relative_auth_id(&absolute)?;
            match fs::read(&absolute) {
                Ok(data) if !data.is_empty() => self.backend.put_auth(&rel_id, &data)?,
                Ok(_) => self.backend.delete_auth(&rel_id)?,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    self.backend.delete_auth(&rel_id)?;
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    pub fn persist_config(&self) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match fs::read(&self.config_path) {
            Ok(data) => self
                .backend
                .put_config(DEFAULT_CONFIG_KEY, &normalize_line_endings_bytes(&data)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.backend.delete_config(DEFAULT_CONFIG_KEY)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn sync_config_from_database(&self, example: Option<&Path>) -> Result<(), StoreError> {
        if let Some(content) = self.backend.load_config(DEFAULT_CONFIG_KEY)? {
            return atomic_write(&self.config_path, &normalize_line_endings_bytes(&content));
        }
        if !self.config_path.exists() {
            let content = match example {
                Some(path) => normalize_line_endings_bytes(&fs::read(path)?),
                None => Vec::new(),
            };
            atomic_write(&self.config_path, &content)?;
        }
        let content = fs::read(&self.config_path)?;
        self.backend
            .put_config(DEFAULT_CONFIG_KEY, &normalize_line_endings_bytes(&content))
    }

    fn sync_auth_from_database(&self) -> Result<(), StoreError> {
        let rows = self.backend.list_auth()?;
        let staging = self.spool_root.join(".auths.sync");
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        create_private_dir(&staging)?;
        for row in rows {
            let rel = validate_relative_id(&row.id)?;
            atomic_write(&staging.join(rel), &row.content)?;
        }
        let backup = self.spool_root.join(".auths.previous");
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        if self.auth_dir.exists() {
            fs::rename(&self.auth_dir, &backup)?;
        }
        if let Err(error) = fs::rename(&staging, &self.auth_dir) {
            if backup.exists() {
                let _ = fs::rename(&backup, &self.auth_dir);
            }
            return Err(error.into());
        }
        if backup.exists() {
            fs::remove_dir_all(backup)?;
        }
        Ok(())
    }

    fn resolve_auth_path(&self, auth: &Auth) -> Result<PathBuf, StoreError> {
        let candidate = auth
            .attributes
            .get("path")
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .or_else(|| (!auth.file_name.trim().is_empty()).then(|| PathBuf::from(&auth.file_name)))
            .unwrap_or_else(|| PathBuf::from(&auth.id));
        self.managed_auth_path(&candidate, false)
    }

    fn resolve_delete_path(&self, id: &str) -> Result<PathBuf, StoreError> {
        let id = id.trim();
        if id.is_empty() {
            return Err(StoreError::InvalidRecord("auth id is empty".to_owned()));
        }
        self.managed_auth_path(Path::new(id), false)
    }

    fn managed_auth_path(&self, input: &Path, add_json: bool) -> Result<PathBuf, StoreError> {
        let mut path = if input.is_absolute() {
            input.to_path_buf()
        } else {
            self.auth_dir.join(input)
        };
        if add_json && path.extension().is_none() {
            path.set_extension("json");
        }
        ensure_lexically_below(&path, &self.auth_dir)?;
        Ok(path)
    }

    fn relative_auth_id(&self, path: &Path) -> Result<String, StoreError> {
        ensure_lexically_below(path, &self.auth_dir)?;
        let rel = path
            .strip_prefix(&self.auth_dir)
            .map_err(|_| StoreError::PathOutsideStore(path.to_path_buf()))?;
        Ok(normalize_auth_id(&rel.to_string_lossy()))
    }

    fn absolute_auth_path(&self, id: &str) -> Result<PathBuf, StoreError> {
        self.managed_auth_path(validate_relative_id(id)?, false)
    }

    #[must_use]
    pub fn config(&self) -> &PostgresStoreConfig {
        &self.cfg
    }

    #[must_use]
    pub fn backend(&self) -> &Arc<dyn PostgresBackend> {
        &self.backend
    }
}

impl AuthStore for PostgresStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        PostgresStore::list(self).map_err(|_| AuthStoreError::Read)
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        let mut auth = auth.clone();
        PostgresStore::save(self, &mut auth)
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(|error| match error {
                StoreError::InvalidConfig(_)
                | StoreError::InvalidIdentifier(_)
                | StoreError::InvalidRecord(_)
                | StoreError::PathOutsideStore(_) => AuthStoreError::InvalidRecord,
                _ => AuthStoreError::Write,
            })
    }

    fn delete(&self, id: &str) -> Result<(), AuthStoreError> {
        PostgresStore::delete(self, id).map_err(|_| AuthStoreError::Delete)
    }
}

pub(crate) fn validate_auth(auth: &Auth) -> Result<(), StoreError> {
    if auth.id.trim().is_empty() && auth.file_name.trim().is_empty() {
        return Err(StoreError::InvalidRecord("auth id is empty".to_owned()));
    }
    validate_metadata_weight(&auth.metadata)
}

fn validate_metadata_weight(metadata: &BTreeMap<String, Value>) -> Result<(), StoreError> {
    if let Some(value) = metadata.get("weight") {
        parse_value(value).map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
    }
    Ok(())
}

pub(crate) fn write_auth_file(path: &Path, auth: &mut Auth) -> Result<(), StoreError> {
    create_private_dir(
        path.parent()
            .ok_or_else(|| StoreError::InvalidRecord("auth path has no parent".to_owned()))?,
    )?;
    let had_metadata = !auth.metadata.is_empty();
    auth.metadata
        .insert("disabled".to_owned(), Value::Bool(auth.disabled));
    if let Some(storage) = &auth.storage {
        storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .save_token_to_file(path)
            .map_err(|error| StoreError::TokenStorage(error.to_string()))?;
        return Ok(());
    }
    if !had_metadata {
        return Err(StoreError::InvalidRecord(format!(
            "nothing to persist for {}",
            auth.id
        )));
    }
    let raw = serde_json::to_vec(&auth.metadata)
        .map_err(|error| StoreError::Serialization(error.to_string()))?;
    if fs::read(path).is_ok_and(|existing| json_equal(&existing, &raw)) {
        return Ok(());
    }
    atomic_write(path, &raw)
}

pub(crate) fn read_auth_file(
    path: &Path,
    base_dir: &Path,
    source_backend: &str,
) -> Result<Option<Auth>, StoreError> {
    let data = fs::read(path)?;
    if data.is_empty() {
        return Ok(None);
    }
    let metadata: BTreeMap<String, Value> = serde_json::from_slice(&data)
        .map_err(|error| StoreError::Serialization(error.to_string()))?;
    validate_metadata_weight(&metadata)?;
    let provider = string_value(metadata.get("type")).unwrap_or("unknown");
    let info = fs::metadata(path)?;
    let modified = info
        .modified()
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(|_| DateTime::<Utc>::from(std::time::UNIX_EPOCH));
    let rel = path.strip_prefix(base_dir).unwrap_or(path);
    let rel = normalize_auth_id(&rel.to_string_lossy());
    let disabled = metadata
        .get("disabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut attributes = BTreeMap::from([
        ("path".to_owned(), path.to_string_lossy().into_owned()),
        ("source_backend".to_owned(), source_backend.to_owned()),
    ]);
    if let Some(email) = string_value(metadata.get("email")) {
        attributes.insert("email".to_owned(), email.to_owned());
    }
    let mut auth = Auth::default();
    auth.id = rel.clone();
    auth.provider = provider.to_owned();
    auth.file_name = rel;
    auth.label = label_for(&metadata);
    auth.status = if disabled {
        AuthStatus::Disabled
    } else {
        AuthStatus::Active
    };
    auth.disabled = disabled;
    auth.attributes = attributes;
    auth.metadata = metadata;
    auth.created_at = modified;
    auth.updated_at = modified;
    apply_custom_headers_from_metadata(&mut auth);
    Ok(Some(auth))
}

pub(crate) fn atomic_write(path: &Path, data: &[u8]) -> Result<(), StoreError> {
    let parent = path
        .parent()
        .ok_or_else(|| StoreError::InvalidRecord("path has no parent".to_owned()))?;
    create_private_dir(parent)?;
    let mut entropy = [0_u8; 12];
    getrandom::fill(&mut entropy)
        .map_err(|error| StoreError::Io(format!("temporary name entropy: {error}")))?;
    let temp = parent.join(format!(".store-{}.tmp", hex_lower(&entropy)));
    let result = (|| -> Result<(), StoreError> {
        let mut options = fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(data)?;
        file.sync_all()?;
        replace_file_atomically(&temp, path)?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[cfg(not(windows))]
fn replace_file_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both owned buffers are NUL-terminated UTF-16 paths and remain
    // alive for the duration of the Win32 call.
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        value.push(char::from(HEX[usize::from(byte >> 4)]));
        value.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    value
}

pub(crate) fn create_private_dir(path: &Path) -> Result<(), StoreError> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), StoreError> {
    #[cfg(unix)]
    {
        fs::File::open(path)?.sync_all()?;
    }
    Ok(())
}

pub(crate) fn ensure_lexically_below(path: &Path, root: &Path) -> Result<(), StoreError> {
    if !path.is_absolute() || !root.is_absolute() || !path.starts_with(root) {
        return Err(StoreError::PathOutsideStore(path.to_path_buf()));
    }
    let rel = path
        .strip_prefix(root)
        .map_err(|_| StoreError::PathOutsideStore(path.to_path_buf()))?;
    if rel.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(StoreError::PathOutsideStore(path.to_path_buf()));
    }
    let mut cursor = root.to_path_buf();
    for component in rel.components() {
        cursor.push(component.as_os_str());
        if cursor.exists()
            && fs::symlink_metadata(&cursor)
                .map(|metadata| metadata.file_type().is_symlink())
                .unwrap_or(false)
        {
            return Err(StoreError::PathOutsideStore(path.to_path_buf()));
        }
    }
    Ok(())
}

pub(crate) fn validate_relative_id(id: &str) -> Result<&Path, StoreError> {
    let path = Path::new(id);
    if id.trim().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(StoreError::InvalidRecord(format!("invalid auth id {id:?}")));
    }
    Ok(path)
}

pub(crate) fn normalize_auth_id(id: &str) -> String {
    id.replace('\\', "/")
}

pub(crate) fn normalize_line_endings_bytes(data: &[u8]) -> Vec<u8> {
    let mut normalized = Vec::with_capacity(data.len());
    let mut index = 0;
    while index < data.len() {
        if data[index] == b'\r' {
            normalized.push(b'\n');
            index += usize::from(data.get(index + 1) == Some(&b'\n')) + 1;
        } else {
            normalized.push(data[index]);
            index += 1;
        }
    }
    normalized
}

pub(crate) fn label_for(metadata: &BTreeMap<String, Value>) -> String {
    ["label", "email", "project_id"]
        .into_iter()
        .find_map(|key| string_value(metadata.get(key)).map(str::to_owned))
        .unwrap_or_default()
}

fn string_value(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn json_equal(left: &[u8], right: &[u8]) -> bool {
    match (
        serde_json::from_slice::<Value>(left),
        serde_json::from_slice::<Value>(right),
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn validate_identifier(identifier: &str) -> Result<(), StoreError> {
    if identifier.is_empty() {
        return Ok(());
    }
    let valid = identifier
        .chars()
        .all(|character| character == '_' || character.is_ascii_alphanumeric());
    if valid {
        Ok(())
    } else {
        Err(StoreError::InvalidIdentifier(identifier.to_owned()))
    }
}

#[must_use]
pub fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

pub fn parse_weight_attribute(raw: &str) -> Result<i64, StoreError> {
    parse_string(raw).map_err(|error| StoreError::InvalidRecord(error.to_string()))
}
