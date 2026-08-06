// ref: internal/store/objectstore.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! S3-compatible object-store semantics over a typed CTOX backend.
//!
//! Endpoint credentials deliberately do not appear in this crate's config or
//! Debug surface. The host creates an authenticated [`ObjectBackend`] from the
//! CTOX secret store and injects it here.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

use super::postgresstore::{
    atomic_write, create_private_dir, ensure_lexically_below, normalize_line_endings_bytes,
    read_auth_file, validate_auth, validate_relative_id, write_auth_file, StoreError,
};

const CONFIG_KEY: &str = "config/config.yaml";
const AUTH_PREFIX: &str = "auths";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectStoreConfig {
    pub bucket: String,
    pub region: String,
    pub prefix: String,
    pub local_root: PathBuf,
}

impl ObjectStoreConfig {
    fn normalized(mut self) -> Result<Self, StoreError> {
        self.bucket = self.bucket.trim().to_owned();
        self.region = self.region.trim().to_owned();
        self.prefix = self.prefix.trim_matches('/').to_owned();
        if self.bucket.is_empty() {
            return Err(StoreError::InvalidConfig("object store bucket is required"));
        }
        if self.local_root.as_os_str().is_empty() || !self.local_root.is_absolute() {
            return Err(StoreError::InvalidConfig(
                "object store local_root must be an absolute CTOX-owned path",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectEntry {
    pub key: String,
    pub content: Vec<u8>,
}

pub trait ObjectBackend: Send + Sync {
    fn ensure_bucket(&self, bucket: &str, region: &str) -> Result<(), StoreError>;
    fn get(&self, bucket: &str, key: &str) -> Result<Option<Vec<u8>>, StoreError>;
    fn list(&self, bucket: &str, prefix: &str) -> Result<Vec<ObjectEntry>, StoreError>;
    fn put(
        &self,
        bucket: &str,
        key: &str,
        content: &[u8],
        content_type: &str,
    ) -> Result<(), StoreError>;
    fn delete(&self, bucket: &str, key: &str) -> Result<(), StoreError>;
}

pub struct ObjectTokenStore {
    backend: Arc<dyn ObjectBackend>,
    cfg: ObjectStoreConfig,
    spool_root: PathBuf,
    config_path: PathBuf,
    auth_dir: PathBuf,
    operation_lock: Mutex<()>,
}

impl fmt::Debug for ObjectTokenStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ObjectTokenStore")
            .field("bucket", &self.cfg.bucket)
            .field("region", &self.cfg.region)
            .field("prefix", &self.cfg.prefix)
            .field("spool_root", &self.spool_root)
            .finish_non_exhaustive()
    }
}

impl ObjectTokenStore {
    pub fn new(
        cfg: ObjectStoreConfig,
        backend: Arc<dyn ObjectBackend>,
    ) -> Result<Self, StoreError> {
        let cfg = cfg.normalized()?;
        let spool_root = cfg.local_root.clone();
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

    pub fn bootstrap(&self, example_config: Option<&Path>) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.backend
            .ensure_bucket(&self.cfg.bucket, &self.cfg.region)?;
        self.sync_config_from_bucket(example_config)?;
        self.sync_auth_from_bucket()
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
            .insert("source_backend".to_owned(), "object-store".to_owned());
        if auth.file_name.trim().is_empty() {
            auth.file_name.clone_from(&auth.id);
        }
        self.upload_auth(&path)?;
        Ok(path)
    }

    pub fn list(&self) -> Result<Vec<Auth>, StoreError> {
        let mut files = Vec::new();
        collect_json_files(&self.auth_dir, &mut files)?;
        files.sort();
        let mut auths = Vec::with_capacity(files.len());
        for path in files {
            // Upstream skips malformed individual entries while preserving a
            // hard error for failure to walk the store itself.
            if let Ok(Some(auth)) = read_auth_file(&path, &self.auth_dir, "object-store") {
                auths.push(auth);
            }
        }
        Ok(auths)
    }

    pub fn delete(&self, id: &str) -> Result<(), StoreError> {
        let path = self.resolve_delete_path(id)?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.delete_auth_object(&path)
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
        for input in paths {
            let input = input.as_ref();
            if input.as_os_str().is_empty() {
                continue;
            }
            let path = if input.is_absolute() {
                input.to_path_buf()
            } else {
                self.auth_dir.join(input)
            };
            ensure_lexically_below(&path, &self.auth_dir)?;
            self.upload_auth(&path)?;
        }
        Ok(())
    }

    pub fn persist_config(&self) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match fs::read(&self.config_path) {
            Ok(data) if data.is_empty() => self.delete_object(CONFIG_KEY),
            Ok(data) => self.put_object(CONFIG_KEY, &data, "application/x-yaml"),
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.delete_object(CONFIG_KEY),
            Err(error) => Err(error.into()),
        }
    }

    fn sync_config_from_bucket(&self, example: Option<&Path>) -> Result<(), StoreError> {
        let key = self.prefixed_key(CONFIG_KEY);
        if let Some(data) = self.backend.get(&self.cfg.bucket, &key)? {
            return atomic_write(&self.config_path, &normalize_line_endings_bytes(&data));
        }
        if !self.config_path.exists() {
            let content = match example {
                Some(path) => normalize_line_endings_bytes(&fs::read(path)?),
                None => Vec::new(),
            };
            atomic_write(&self.config_path, &content)?;
        }
        let data = fs::read(&self.config_path)?;
        if !data.is_empty() {
            self.put_object(CONFIG_KEY, &data, "application/x-yaml")?;
        }
        Ok(())
    }

    fn sync_auth_from_bucket(&self) -> Result<(), StoreError> {
        // Deliberately incremental: removing the directory can race a watcher
        // and propagate spurious remote deletions, exactly as upstream notes.
        create_private_dir(&self.auth_dir)?;
        let prefix = self.prefixed_key(&format!("{AUTH_PREFIX}/"));
        for object in self.backend.list(&self.cfg.bucket, &prefix)? {
            let Some(rel) = object.key.strip_prefix(&prefix) else {
                continue;
            };
            if rel.is_empty() || rel.ends_with('/') {
                continue;
            }
            let rel = match validate_relative_id(rel) {
                Ok(path) => path,
                Err(_) => continue,
            };
            let local = self.auth_dir.join(rel);
            ensure_lexically_below(&local, &self.auth_dir)?;
            atomic_write(&local, &object.content)?;
        }
        Ok(())
    }

    fn upload_auth(&self, path: &Path) -> Result<(), StoreError> {
        ensure_lexically_below(path, &self.auth_dir)?;
        let rel = path
            .strip_prefix(&self.auth_dir)
            .map_err(|_| StoreError::PathOutsideStore(path.to_path_buf()))?;
        let key = format!("{AUTH_PREFIX}/{}", rel.to_string_lossy().replace('\\', "/"));
        match fs::read(path) {
            Ok(data) if !data.is_empty() => self.put_object(&key, &data, "application/json"),
            Ok(_) => self.delete_object(&key),
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.delete_object(&key),
            Err(error) => Err(error.into()),
        }
    }

    fn delete_auth_object(&self, path: &Path) -> Result<(), StoreError> {
        ensure_lexically_below(path, &self.auth_dir)?;
        let rel = path
            .strip_prefix(&self.auth_dir)
            .map_err(|_| StoreError::PathOutsideStore(path.to_path_buf()))?;
        self.delete_object(&format!(
            "{AUTH_PREFIX}/{}",
            rel.to_string_lossy().replace('\\', "/")
        ))
    }

    fn put_object(&self, key: &str, data: &[u8], content_type: &str) -> Result<(), StoreError> {
        if data.is_empty() {
            return self.delete_object(key);
        }
        self.backend.put(
            &self.cfg.bucket,
            &self.prefixed_key(key),
            data,
            content_type,
        )
    }

    fn delete_object(&self, key: &str) -> Result<(), StoreError> {
        self.backend
            .delete(&self.cfg.bucket, &self.prefixed_key(key))
    }

    fn prefixed_key(&self, key: &str) -> String {
        let key = key.trim_start_matches('/');
        if self.cfg.prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{}/{key}", self.cfg.prefix)
        }
    }

    fn resolve_auth_path(&self, auth: &Auth) -> Result<PathBuf, StoreError> {
        let raw = auth
            .attributes
            .get("path")
            .filter(|value| !value.trim().is_empty())
            .map(String::as_str)
            .or_else(|| (!auth.file_name.trim().is_empty()).then_some(auth.file_name.as_str()))
            .unwrap_or(auth.id.as_str());
        let mut path = PathBuf::from(raw.trim());
        if !path.is_absolute() {
            path = self.auth_dir.join(path);
        }
        if path.extension().is_none() {
            path.set_extension("json");
        }
        ensure_lexically_below(&path, &self.auth_dir)?;
        Ok(path)
    }

    fn resolve_delete_path(&self, id: &str) -> Result<PathBuf, StoreError> {
        let id = id.trim();
        if id.is_empty() {
            return Err(StoreError::InvalidRecord("auth id is empty".to_owned()));
        }
        let mut path = PathBuf::from(id);
        if !path.is_absolute() {
            validate_relative_id(id)?;
            path = self.auth_dir.join(path);
        }
        if path.extension().is_none() {
            path.set_extension("json");
        }
        ensure_lexically_below(&path, &self.auth_dir)?;
        Ok(path)
    }
}

impl AuthStore for ObjectTokenStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        ObjectTokenStore::list(self).map_err(|_| AuthStoreError::Read)
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        let mut auth = auth.clone();
        ObjectTokenStore::save(self, &mut auth)
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
        ObjectTokenStore::delete(self, id).map_err(|_| AuthStoreError::Delete)
    }
}

fn collect_json_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), StoreError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            collect_json_files(&entry.path(), files)?;
        } else if entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        {
            files.push(entry.path());
        }
    }
    Ok(())
}
