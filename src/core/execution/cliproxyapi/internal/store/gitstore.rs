// ref: internal/store/gitstore.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: network Git and credentials cross an injected typed transport authority
// License: MIT (upstream); modifications AGPL-3.0-only

//! Git-backed auth/config store.
//!
//! The port uses libgit2 for local remotes and a typed, instance-bound
//! [`GitTransportAuthority`] for authenticated HTTPS/SSH. The store never
//! starts `git`, reads ambient credential helpers or environment variables, or
//! places credentials in argv. Network authorities own credential resolution
//! and must enforce the supplied optimistic lease atomically.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use git2::build::{CheckoutBuilder, RepoBuilder};
use git2::{
    Cred, CredentialType, Direction, ErrorCode, FetchOptions, Oid, PushOptions, RemoteCallbacks,
    Repository, ResetType, Signature, Status, StatusOptions,
};

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

use super::postgresstore::{
    create_private_dir, ensure_lexically_below, read_auth_file, validate_auth,
    validate_relative_id, write_auth_file, StoreError,
};

const AUTH_DIR_NAME: &str = "auths";
const CONFIG_PATH: &str = "config/config.yaml";
const GC_INTERVAL: Duration = Duration::from_secs(5 * 60);
const GC_PRUNE_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Eq, PartialEq)]
pub struct GitCredentialRef(String);

impl GitCredentialRef {
    pub fn new(value: impl Into<String>) -> Result<Self, StoreError> {
        let value = value.into();
        let value = value.trim();
        if value.is_empty()
            || value.len() > 256
            || value.chars().any(char::is_control)
            || value.contains(['/', '\\'])
        {
            return Err(StoreError::InvalidConfig(
                "git credential reference is invalid",
            ));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GitCredentialRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GitCredentialRef([REDACTED])")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct GitRemoteRequest {
    pub remote: String,
    pub repository: PathBuf,
    pub branch: Option<String>,
    pub credential: GitCredentialRef,
}

impl fmt::Debug for GitRemoteRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitRemoteRequest")
            .field("remote_kind", &remote_kind(&self.remote))
            .field("repository", &self.repository)
            .field("branch", &self.branch)
            .field("credential", &self.credential)
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct GitPushRequest {
    pub remote: String,
    pub repository: PathBuf,
    pub branch: String,
    pub credential: GitCredentialRef,
    /// Expected remote branch tip. The authority must compare and update the
    /// remote reference atomically (force-with-lease semantics).
    pub expected_remote_oid: Option<String>,
    pub allow_create: bool,
}

impl fmt::Debug for GitPushRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitPushRequest")
            .field("remote_kind", &remote_kind(&self.remote))
            .field("repository", &self.repository)
            .field("branch", &self.branch)
            .field("credential", &self.credential)
            .field(
                "has_expected_remote_oid",
                &self.expected_remote_oid.is_some(),
            )
            .field("allow_create", &self.allow_create)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitTransportError {
    Authentication,
    EmptyRemote,
    LeaseRejected,
    Unavailable,
    InvalidResponse,
}

impl fmt::Display for GitTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Authentication => "git transport authentication failed",
            Self::EmptyRemote => "git transport remote is empty",
            Self::LeaseRejected => "git transport lease was rejected",
            Self::Unavailable => "git transport is unavailable",
            Self::InvalidResponse => "git transport response is invalid",
        })
    }
}

impl std::error::Error for GitTransportError {}

/// Typed boundary implemented by CTOX's isolated network transport owner.
/// Implementations retain credentials internally; only an opaque store handle
/// crosses this interface.
pub trait GitTransportAuthority: Send + Sync {
    fn clone_repository(&self, request: &GitRemoteRequest) -> Result<(), GitTransportError>;
    fn fetch(&self, request: &GitRemoteRequest) -> Result<(), GitTransportError>;
    fn default_branch(
        &self,
        request: &GitRemoteRequest,
    ) -> Result<Option<String>, GitTransportError>;
    fn push(&self, request: &GitPushRequest) -> Result<(), GitTransportError>;
    fn maintenance(&self, request: &GitRemoteRequest) -> Result<(), GitTransportError>;
}

#[derive(Clone)]
struct NetworkTransport {
    authority: Arc<dyn GitTransportAuthority>,
    credential: GitCredentialRef,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitStoreConfig {
    pub remote: String,
    pub branch: Option<String>,
    /// Absolute path to the managed `auths` directory. Its parent is the
    /// working-tree root.
    pub auth_dir: PathBuf,
}

impl GitStoreConfig {
    fn normalized(mut self, network_allowed: bool) -> Result<Self, StoreError> {
        self.remote = self.remote.trim().to_owned();
        self.branch = self
            .branch
            .take()
            .map(|branch| branch.trim().to_owned())
            .filter(|branch| !branch.is_empty());
        if self.remote.is_empty() {
            return Err(StoreError::InvalidConfig(
                "git token store remote is required",
            ));
        }
        if is_network_remote(&self.remote) && remote_has_inline_credentials(&self.remote) {
            return Err(StoreError::InvalidConfig(
                "git remote must not contain inline credentials",
            ));
        }
        let local_remote =
            Path::new(&self.remote).is_absolute() || self.remote.starts_with("file://");
        let authorized_network = network_allowed && is_network_remote(&self.remote);
        if !local_remote && !authorized_network {
            return Err(StoreError::InvalidConfig(
                "network git remotes require the isolated CTOX git transport",
            ));
        }
        if !self.auth_dir.is_absolute() || self.auth_dir.file_name() != Some(AUTH_DIR_NAME.as_ref())
        {
            return Err(StoreError::InvalidConfig(
                "git auth_dir must be an absolute path ending in auths",
            ));
        }
        if let Some(branch) = &self.branch {
            validate_branch_name(branch)?;
        }
        Ok(self)
    }
}

pub struct GitTokenStore {
    cfg: GitStoreConfig,
    repo_dir: PathBuf,
    config_dir: PathBuf,
    operation_lock: Mutex<()>,
    explicit_deletions: Mutex<BTreeSet<String>>,
    network: Option<NetworkTransport>,
    last_gc: Mutex<Option<SystemTime>>,
}

impl fmt::Debug for GitTokenStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitTokenStore")
            .field("remote_kind", &remote_kind(&self.cfg.remote))
            .field("branch", &self.cfg.branch)
            .field("auth_dir", &self.cfg.auth_dir)
            .field("has_network_authority", &self.network.is_some())
            .finish()
    }
}

impl GitTokenStore {
    pub fn new(cfg: GitStoreConfig) -> Result<Self, StoreError> {
        Self::build(cfg, None)
    }

    pub fn new_with_transport(
        cfg: GitStoreConfig,
        credential: GitCredentialRef,
        authority: Arc<dyn GitTransportAuthority>,
    ) -> Result<Self, StoreError> {
        Self::build(
            cfg,
            Some(NetworkTransport {
                authority,
                credential,
            }),
        )
    }

    fn build(cfg: GitStoreConfig, network: Option<NetworkTransport>) -> Result<Self, StoreError> {
        let cfg = cfg.normalized(network.is_some())?;
        if network.is_some() && !is_network_remote(&cfg.remote) {
            return Err(StoreError::InvalidConfig(
                "git network authority requires an HTTPS or SSH remote",
            ));
        }
        let repo_dir = cfg
            .auth_dir
            .parent()
            .expect("absolute auths path has parent")
            .to_path_buf();
        let config_dir = repo_dir.join("config");
        Ok(Self {
            cfg,
            repo_dir,
            config_dir,
            operation_lock: Mutex::new(()),
            explicit_deletions: Mutex::new(BTreeSet::new()),
            network,
            last_gc: Mutex::new(None),
        })
    }

    #[must_use]
    pub fn auth_dir(&self) -> &Path {
        &self.cfg.auth_dir
    }

    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.yaml")
    }

    #[must_use]
    pub fn work_dir(&self) -> &Path {
        &self.repo_dir
    }

    pub fn ensure_repository(&self) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_repository_locked()
    }

    fn ensure_repository_locked(&self) -> Result<(), StoreError> {
        if self.repo_dir.join(".git").exists() {
            let repository = Repository::open(&self.repo_dir).map_err(git_error);
            let baseline = repository
                .as_ref()
                .ok()
                .and_then(|repo| recovery_baseline(repo, &self.repo_dir).ok());
            match repository.and_then(|repo| self.update_existing(&repo)) {
                Ok(()) => {}
                Err(error) if is_repository_corruption(&error) => {
                    self.recover_repository(&error, baseline)?;
                }
                Err(error) => return Err(error),
            }
        } else {
            self.clone_or_initialize()?;
        }
        create_private_dir(&self.cfg.auth_dir)?;
        create_private_dir(&self.config_dir)?;
        Ok(())
    }

    /// Replaces corrupt object storage while preserving only dirty managed
    /// paths proven not to overlap remote changes. Without a trustworthy
    /// baseline, recovery remains fail-closed.
    fn recover_repository(
        &self,
        cause: &StoreError,
        baseline: Option<RecoveryBaseline>,
    ) -> Result<(), StoreError> {
        let recovery_dir = reserve_sibling_dir(&self.repo_dir, "ctox-git-recovery")?;
        let recovery = (|| {
            let repo = self.clone_repository(&recovery_dir).map_err(git_error)?;
            if self.cfg.branch.is_none() {
                self.checkout_remote_default_at(&repo)?;
            }
            match baseline {
                Some(baseline) => {
                    let remote_head = repo
                        .head()
                        .and_then(|head| head.peel_to_commit())
                        .map_err(git_error)?
                        .id();
                    let remote_tree = managed_commit_snapshot(&repo, remote_head)?;
                    let changed = changed_snapshot_paths(&baseline.committed, &remote_tree);
                    if let Some((remote, dirty)) = changed.iter().find_map(|remote| {
                        baseline
                            .dirty
                            .keys()
                            .find(|dirty| paths_overlap(remote, dirty))
                            .map(|dirty| (remote, dirty))
                    }) {
                        return Err(StoreError::Backend(format!(
                            "remote path {} conflicts with local change {} during repository recovery",
                            remote.display(),
                            dirty.display()
                        )));
                    }
                    restore_dirty_paths(&recovery_dir, baseline.dirty)?;
                }
                None => ensure_managed_trees_equal(&self.repo_dir, &recovery_dir)?,
            }

            let backup_dir = reserve_sibling_dir(&self.repo_dir, "ctox-git-backup")?;
            fs::remove_dir(&backup_dir)?;
            fs::rename(&self.repo_dir, &backup_dir)?;
            if let Err(install_error) = fs::rename(&recovery_dir, &self.repo_dir) {
                let _ = fs::rename(&backup_dir, &self.repo_dir);
                return Err(StoreError::Io(format!(
                    "install recovered git repository failed: {install_error}"
                )));
            }
            if let Err(cleanup_error) = fs::remove_dir_all(&backup_dir) {
                return Err(StoreError::Io(format!(
                    "recovered repository installed but backup cleanup failed: {cleanup_error}"
                )));
            }
            Ok(())
        })();
        if recovery_dir.exists() {
            let _ = fs::remove_dir_all(&recovery_dir);
        }
        recovery.map_err(|error| {
            StoreError::Backend(format!(
                "repository recovery after {cause} failed closed: {error}"
            ))
        })
    }

    fn clone_or_initialize(&self) -> Result<(), StoreError> {
        if self.repo_dir.exists() && fs::read_dir(&self.repo_dir)?.next().is_some() {
            return Err(StoreError::InvalidConfig(
                "git repository directory exists without .git",
            ));
        }
        if let Some(parent) = self.repo_dir.parent() {
            create_private_dir(parent)?;
        }
        match self.clone_repository(&self.repo_dir) {
            Ok(repo) => {
                if self.cfg.branch.is_none() {
                    self.checkout_remote_default(&repo)?;
                }
                Ok(())
            }
            Err(error) if is_empty_remote_error(&error) => self.initialize_empty_remote(),
            Err(error) => Err(git_error(error)),
        }
    }

    fn initialize_empty_remote(&self) -> Result<(), StoreError> {
        if self.repo_dir.exists() {
            fs::remove_dir_all(&self.repo_dir)?;
        }
        create_private_dir(&self.repo_dir)?;
        let repo = Repository::init(&self.repo_dir).map_err(git_error)?;
        repo.remote("origin", &self.cfg.remote).map_err(git_error)?;
        let branch = self.cfg.branch.as_deref().unwrap_or("master");
        repo.set_head(&format!("refs/heads/{branch}"))
            .map_err(git_error)?;
        create_private_dir(&self.cfg.auth_dir)?;
        create_private_dir(&self.config_dir)?;
        fs::write(self.cfg.auth_dir.join(".gitkeep"), b"")?;
        fs::write(self.config_dir.join(".gitkeep"), b"")?;
        self.commit_and_push_with_options(
            &repo,
            "Initialize CLIProxyAPI store",
            true,
            &[
                PathBuf::from("auths/.gitkeep"),
                PathBuf::from("config/.gitkeep"),
            ],
        )
    }

    fn update_existing(&self, repo: &Repository) -> Result<(), StoreError> {
        verify_repository(repo)?;
        let dirty = capture_dirty_paths(repo, &self.repo_dir)?;
        self.fetch(repo)?;
        let branch = match &self.cfg.branch {
            Some(branch) => branch.clone(),
            None => self
                .resolve_remote_default(repo)?
                .unwrap_or_else(|| current_branch(repo).unwrap_or_else(|| "master".to_owned())),
        };
        let remote_ref = format!("refs/remotes/origin/{branch}");
        let remote_oid = repo
            .refname_to_id(&remote_ref)
            .map_err(|error| StoreError::Backend(format!("configured branch {branch}: {error}")))?;
        let head_oid = repo.head().ok().and_then(|head| head.target());
        if let Some(head_oid) = head_oid {
            let changed = changed_paths(repo, head_oid, remote_oid)?;
            if let Some(conflict) = dirty.keys().find(|path| {
                changed
                    .iter()
                    .any(|remote| paths_overlap(path.as_path(), remote.as_path()))
            }) {
                return Err(StoreError::Backend(format!(
                    "git pull conflicts with dirty managed path {}",
                    conflict.display()
                )));
            }
        }
        checkout_oid(repo, &branch, remote_oid)?;
        restore_dirty_paths(&self.repo_dir, dirty)?;
        Ok(())
    }

    fn fetch(&self, repo: &Repository) -> Result<(), StoreError> {
        if let Some(network) = &self.network {
            network
                .authority
                .fetch(&self.remote_request(network, self.repo_dir.clone()))
                .map_err(transport_error)?;
            return Ok(());
        }
        let mut remote = repo.find_remote("origin").map_err(git_error)?;
        let mut options = FetchOptions::new();
        options.remote_callbacks(self.callbacks());
        remote
            .fetch(&[] as &[&str], Some(&mut options), None)
            .map_err(git_error)
    }

    fn resolve_remote_default(&self, repo: &Repository) -> Result<Option<String>, StoreError> {
        if let Some(network) = &self.network {
            return network
                .authority
                .default_branch(&self.remote_request(network, self.repo_dir.clone()))
                .map_err(transport_error)
                .and_then(validate_default_branch);
        }
        let mut remote = repo.find_remote("origin").map_err(git_error)?;
        remote
            .connect_auth(Direction::Fetch, Some(self.callbacks()), None)
            .map_err(git_error)?;
        let default = remote.default_branch().ok();
        remote.disconnect().map_err(git_error)?;
        Ok(default
            .as_ref()
            .and_then(|name| name.as_str())
            .and_then(|name| name.strip_prefix("refs/heads/"))
            .map(str::to_owned))
    }

    fn checkout_remote_default(&self, repo: &Repository) -> Result<(), StoreError> {
        self.checkout_remote_default_at(repo)
    }

    fn checkout_remote_default_at(&self, repo: &Repository) -> Result<(), StoreError> {
        self.fetch(repo)?;
        let Some(branch) = self.resolve_remote_default(repo)? else {
            return Ok(());
        };
        let oid = repo
            .refname_to_id(&format!("refs/remotes/origin/{branch}"))
            .map_err(git_error)?;
        checkout_oid(repo, &branch, oid)
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
        self.ensure_repository_locked()?;
        write_auth_file(&path, auth)?;
        auth.attributes
            .insert("path".to_owned(), path.to_string_lossy().into_owned());
        auth.attributes
            .insert("source_backend".to_owned(), "git".to_owned());
        if auth.file_name.trim().is_empty() {
            auth.file_name.clone_from(&auth.id);
        }
        let rel = self.relative_to_repo(&path)?;
        let repo = Repository::open(&self.repo_dir).map_err(git_error)?;
        self.commit_and_push_with_options(
            &repo,
            &format!("Update auth {}", auth.id),
            false,
            &[rel],
        )?;
        Ok(path)
    }

    pub fn list(&self) -> Result<Vec<Auth>, StoreError> {
        let mut files = Vec::new();
        collect_json_files(&self.cfg.auth_dir, &mut files)?;
        files.sort();
        let mut auths = Vec::with_capacity(files.len());
        for path in files {
            if let Ok(Some(auth)) = read_auth_file(&path, &self.cfg.auth_dir, "git") {
                auths.push(auth);
            }
        }
        Ok(auths)
    }

    pub fn delete(&self, id: &str) -> Result<(), StoreError> {
        let path = self.resolve_delete_path(id)?;
        let rel = self.relative_to_repo(&path)?;
        let rel_string = git_path(&rel)?;
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_repository_locked()?;
        self.explicit_deletions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(rel_string.clone());
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let repo = Repository::open(&self.repo_dir).map_err(git_error)?;
        let result = self.commit_and_push_with_options(
            &repo,
            &format!("Remove auth {id}"),
            false,
            std::slice::from_ref(&rel),
        );
        if result.is_err() {
            self.explicit_deletions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&rel_string);
        }
        result
    }

    pub fn persist_auth_files<I, P>(&self, message: &str, paths: I) -> Result<(), StoreError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut rel_paths = Vec::new();
        for input in paths {
            let input = input.as_ref();
            if input.as_os_str().is_empty() {
                continue;
            }
            let path = if input.is_absolute() {
                input.to_path_buf()
            } else {
                self.cfg.auth_dir.join(input)
            };
            ensure_lexically_below(&path, &self.cfg.auth_dir)?;
            rel_paths.push(self.relative_to_repo(&path)?);
        }
        if rel_paths.is_empty() {
            return Ok(());
        }
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_repository_locked()?;
        if self.guard_watcher_removal(message, &rel_paths)? {
            return Ok(());
        }
        let repo = Repository::open(&self.repo_dir).map_err(git_error)?;
        self.commit_and_push_with_options(&repo, message, false, &rel_paths)
    }

    pub fn persist_config(&self) -> Result<(), StoreError> {
        let _guard = self
            .operation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_repository_locked()?;
        let repo = Repository::open(&self.repo_dir).map_err(git_error)?;
        self.commit_and_push_with_options(
            &repo,
            "Update configuration",
            false,
            &[PathBuf::from(CONFIG_PATH)],
        )
    }

    fn guard_watcher_removal(
        &self,
        message: &str,
        rel_paths: &[PathBuf],
    ) -> Result<bool, StoreError> {
        let mut deletions = self
            .explicit_deletions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut all_explicit = true;
        for rel in rel_paths {
            let absolute = self.repo_dir.join(rel);
            if absolute.exists() {
                continue;
            }
            let rel = git_path(rel)?;
            if deletions.remove(&rel) {
                continue;
            }
            all_explicit = false;
            if message.trim_start().starts_with("Remove auth ") {
                return Err(StoreError::Backend(format!(
                    "refusing watcher-originated removal of {rel}"
                )));
            }
        }
        Ok(all_explicit
            && rel_paths
                .iter()
                .all(|path| !self.repo_dir.join(path).exists()))
    }

    pub(super) fn commit_and_push_with_options(
        &self,
        repo: &Repository,
        message: &str,
        allow_missing_remote: bool,
        rel_paths: &[PathBuf],
    ) -> Result<(), StoreError> {
        let managed = normalize_managed_paths(rel_paths)?;
        if managed.is_empty() {
            return Ok(());
        }
        let mut index = repo.index().map_err(git_error)?;
        reset_index_to_head(repo, &mut index)?;
        for path in &managed {
            let full = self.repo_dir.join(path);
            if full.exists() {
                index.add_path(path).map_err(git_error)?;
            } else {
                match index.remove_path(path) {
                    Ok(()) => {}
                    Err(error) if error.code() == ErrorCode::NotFound => {}
                    Err(error) => return Err(git_error(error)),
                }
            }
        }
        index.write().map_err(git_error)?;
        let tree_oid = index.write_tree().map_err(git_error)?;
        let tree = repo.find_tree(tree_oid).map_err(git_error)?;
        let parent = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
        if parent
            .as_ref()
            .is_some_and(|parent| parent.tree_id() == tree_oid)
        {
            return Ok(());
        }
        let signature =
            Signature::now("CLIProxyAPI Rust Port", "noreply@ctox.local").map_err(git_error)?;
        let oid = match parent.as_ref() {
            Some(parent) => repo
                .commit(
                    Some("HEAD"),
                    &signature,
                    &signature,
                    message,
                    &tree,
                    &[parent],
                )
                .map_err(git_error)?,
            None => repo
                .commit(Some("HEAD"), &signature, &signature, message, &tree, &[])
                .map_err(git_error)?,
        };
        if let Err(error) =
            validate_commit_paths(repo, parent.as_ref().map(git2::Commit::id), oid, &managed)
        {
            rollback_local_commit(repo, parent.as_ref())?;
            return Err(error);
        }
        if let Err(error) = self.push(repo, allow_missing_remote) {
            rollback_local_commit(repo, parent.as_ref())?;
            return Err(error);
        }
        Ok(())
    }

    fn push(&self, repo: &Repository, allow_missing_remote: bool) -> Result<(), StoreError> {
        let branch = current_branch(repo)
            .or_else(|| self.cfg.branch.clone())
            .ok_or_else(|| {
                StoreError::Backend("git repository has no current branch".to_owned())
            })?;
        let tracking_ref = format!("refs/remotes/origin/{branch}");
        let expected = repo
            .refname_to_id(&tracking_ref)
            .ok()
            .map(|oid| oid.to_string());
        let head = repo.head().ok().and_then(|head| head.target());
        if let Some(network) = &self.network {
            network
                .authority
                .push(&GitPushRequest {
                    remote: self.cfg.remote.clone(),
                    repository: self.repo_dir.clone(),
                    branch: branch.clone(),
                    credential: network.credential.clone(),
                    expected_remote_oid: expected,
                    allow_create: allow_missing_remote,
                })
                .map_err(transport_error)?;
            if let Some(head) = head {
                repo.reference(&tracking_ref, head, true, "network push lease accepted")
                    .map_err(git_error)?;
            }
            self.maybe_run_gc(repo);
            return Ok(());
        }
        let mut remote = repo.find_remote("origin").map_err(git_error)?;
        let mut options = PushOptions::new();
        options.remote_callbacks(self.callbacks());
        match remote.push(
            &[format!("refs/heads/{branch}:refs/heads/{branch}")],
            Some(&mut options),
        ) {
            Ok(()) => {
                if let Some(head) = head {
                    repo.reference(&tracking_ref, head, true, "local push lease accepted")
                        .map_err(git_error)?;
                }
                self.maybe_run_gc(repo);
                Ok(())
            }
            Err(error) if allow_missing_remote && is_missing_remote_error(&error) => Ok(()),
            Err(error) if matches!(error.code(), ErrorCode::NotFastForward) => {
                Err(transport_error(GitTransportError::LeaseRejected))
            }
            Err(error) => Err(git_error(error)),
        }
    }

    fn clone_repository(&self, destination: &Path) -> Result<Repository, git2::Error> {
        if let Some(network) = &self.network {
            return match network
                .authority
                .clone_repository(&self.remote_request(network, destination.to_path_buf()))
            {
                Ok(()) => Repository::open(destination),
                Err(GitTransportError::EmptyRemote) => {
                    Err(git2::Error::from_str("empty remote repository"))
                }
                Err(error) => Err(git2::Error::from_str(error.to_string().as_str())),
            };
        }
        let mut fetch = FetchOptions::new();
        fetch.remote_callbacks(self.callbacks());
        let mut builder = RepoBuilder::new();
        builder.fetch_options(fetch);
        if let Some(branch) = &self.cfg.branch {
            builder.branch(branch);
        }
        builder.clone(&self.cfg.remote, destination)
    }

    fn remote_request(&self, network: &NetworkTransport, repository: PathBuf) -> GitRemoteRequest {
        GitRemoteRequest {
            remote: self.cfg.remote.clone(),
            repository,
            branch: self.cfg.branch.clone(),
            credential: network.credential.clone(),
        }
    }

    fn maybe_run_gc(&self, repo: &Repository) {
        let now = SystemTime::now();
        let mut last = self
            .last_gc
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if last
            .is_some_and(|previous| now.duration_since(previous).unwrap_or_default() < GC_INTERVAL)
        {
            return;
        }
        *last = Some(now);
        drop(last);
        if let Some(network) = &self.network {
            let _ = network
                .authority
                .maintenance(&self.remote_request(network, self.repo_dir.clone()))
                .map_err(transport_error);
            return;
        }
        let _ = prune_unreachable_loose_objects(repo, now - GC_PRUNE_GRACE);
    }

    fn callbacks(&self) -> RemoteCallbacks<'static> {
        let mut callbacks = RemoteCallbacks::new();
        callbacks.credentials(move |_, username, allowed| {
            if allowed.contains(CredentialType::DEFAULT) {
                Cred::default()
            } else if allowed.contains(CredentialType::USERNAME) {
                Cred::username(username.unwrap_or("git"))
            } else {
                Err(git2::Error::from_str(
                    "remote requires credentials but no CTOX credential provider was injected",
                ))
            }
        });
        callbacks
    }

    fn resolve_auth_path(&self, auth: &Auth) -> Result<PathBuf, StoreError> {
        let raw = auth
            .attributes
            .get("path")
            .filter(|value| !value.trim().is_empty())
            .map(String::as_str)
            .or_else(|| (!auth.file_name.trim().is_empty()).then_some(auth.file_name.as_str()))
            .unwrap_or(auth.id.as_str());
        let mut path = PathBuf::from(raw);
        if !path.is_absolute() {
            path = self.cfg.auth_dir.join(path);
        }
        if path.extension().is_none() {
            path.set_extension("json");
        }
        ensure_lexically_below(&path, &self.cfg.auth_dir)?;
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
            path = self.cfg.auth_dir.join(path);
        }
        if path.extension().is_none() {
            path.set_extension("json");
        }
        ensure_lexically_below(&path, &self.cfg.auth_dir)?;
        Ok(path)
    }

    fn relative_to_repo(&self, path: &Path) -> Result<PathBuf, StoreError> {
        ensure_lexically_below(path, &self.repo_dir)?;
        path.strip_prefix(&self.repo_dir)
            .map(Path::to_path_buf)
            .map_err(|_| StoreError::PathOutsideStore(path.to_path_buf()))
    }
}

struct RecoveryBaseline {
    committed: BTreeMap<PathBuf, Vec<u8>>,
    dirty: BTreeMap<PathBuf, Option<Vec<u8>>>,
}

fn recovery_baseline(repo: &Repository, repo_dir: &Path) -> Result<RecoveryBaseline, StoreError> {
    let head = repo
        .head()
        .and_then(|head| head.peel_to_commit())
        .map_err(git_error)?
        .id();
    Ok(RecoveryBaseline {
        committed: managed_commit_snapshot(repo, head)?,
        dirty: capture_dirty_paths(repo, repo_dir)?,
    })
}

fn managed_commit_snapshot(
    repo: &Repository,
    commit: Oid,
) -> Result<BTreeMap<PathBuf, Vec<u8>>, StoreError> {
    let tree = repo
        .find_commit(commit)
        .and_then(|commit| commit.tree())
        .map_err(git_error)?;
    let mut snapshot = BTreeMap::new();
    tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
        let Some(name) = entry.name() else {
            return git2::TreeWalkResult::Ok;
        };
        let path = PathBuf::from(format!("{root}{name}"));
        if !is_managed_path(
            &path,
            &[PathBuf::from(AUTH_DIR_NAME), PathBuf::from(CONFIG_PATH)],
        ) {
            return if entry.kind() == Some(git2::ObjectType::Tree) {
                git2::TreeWalkResult::Skip
            } else {
                git2::TreeWalkResult::Ok
            };
        }
        if let Ok(blob) = repo.find_blob(entry.id()) {
            snapshot.insert(path, blob.content().to_vec());
        }
        git2::TreeWalkResult::Ok
    })
    .map_err(git_error)?;
    Ok(snapshot)
}

fn changed_snapshot_paths(
    baseline: &BTreeMap<PathBuf, Vec<u8>>,
    remote: &BTreeMap<PathBuf, Vec<u8>>,
) -> BTreeSet<PathBuf> {
    baseline
        .keys()
        .chain(remote.keys())
        .filter(|path| baseline.get(*path) != remote.get(*path))
        .cloned()
        .collect()
}

impl AuthStore for GitTokenStore {
    fn list(&self) -> Result<Vec<Auth>, AuthStoreError> {
        GitTokenStore::list(self).map_err(|_| AuthStoreError::Read)
    }

    fn save(&self, auth: &Auth) -> Result<String, AuthStoreError> {
        let mut auth = auth.clone();
        GitTokenStore::save(self, &mut auth)
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
        GitTokenStore::delete(self, id).map_err(|_| AuthStoreError::Delete)
    }
}

fn validate_branch_name(branch: &str) -> Result<(), StoreError> {
    if branch.starts_with('-')
        || branch.ends_with('/')
        || branch.contains("..")
        || branch.contains("@{")
        || branch.chars().any(|character| {
            character.is_control()
                || matches!(character, ' ' | '~' | '^' | ':' | '?' | '*' | '[' | '\\')
        })
    {
        return Err(StoreError::InvalidConfig("invalid git branch name"));
    }
    Ok(())
}

fn verify_repository(repo: &Repository) -> Result<(), StoreError> {
    let head = repo.head().map_err(git_error)?;
    if head.is_branch() || head.target().is_some() {
        head.peel_to_commit().map_err(git_error)?;
    }
    Ok(())
}

fn checkout_oid(repo: &Repository, branch: &str, oid: Oid) -> Result<(), StoreError> {
    let local_ref = format!("refs/heads/{branch}");
    repo.reference(&local_ref, oid, true, "sync remote branch")
        .map_err(git_error)?;
    repo.set_head(&local_ref).map_err(git_error)?;
    let object = repo.find_object(oid, None).map_err(git_error)?;
    let mut checkout = CheckoutBuilder::new();
    checkout.force().remove_untracked(false);
    repo.reset(&object, ResetType::Hard, Some(&mut checkout))
        .map_err(git_error)
}

fn capture_dirty_paths(
    repo: &Repository,
    repo_dir: &Path,
) -> Result<BTreeMap<PathBuf, Option<Vec<u8>>>, StoreError> {
    let mut options = StatusOptions::new();
    options
        .include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_ignored(false);
    let statuses = repo.statuses(Some(&mut options)).map_err(git_error)?;
    let mut dirty = BTreeMap::new();
    for status in statuses.iter() {
        if status.status() == Status::CURRENT {
            continue;
        }
        let Some(path) = status.path() else {
            continue;
        };
        let path = PathBuf::from(path);
        if !is_managed_path(
            &path,
            &[PathBuf::from(AUTH_DIR_NAME), PathBuf::from(CONFIG_PATH)],
        ) {
            continue;
        }
        let absolute = repo_dir.join(&path);
        let content = match fs::read(&absolute) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        dirty.insert(path, content);
    }
    Ok(dirty)
}

fn restore_dirty_paths(
    repo_dir: &Path,
    dirty: BTreeMap<PathBuf, Option<Vec<u8>>>,
) -> Result<(), StoreError> {
    for (path, content) in dirty {
        let absolute = repo_dir.join(path);
        match content {
            Some(content) => super::postgresstore::atomic_write(&absolute, &content)?,
            None => match fs::remove_file(absolute) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            },
        }
    }
    Ok(())
}

fn changed_paths(repo: &Repository, old: Oid, new: Oid) -> Result<Vec<PathBuf>, StoreError> {
    let old_tree = repo
        .find_commit(old)
        .and_then(|commit| commit.tree())
        .map_err(git_error)?;
    let new_tree = repo
        .find_commit(new)
        .and_then(|commit| commit.tree())
        .map_err(git_error)?;
    let diff = repo
        .diff_tree_to_tree(Some(&old_tree), Some(&new_tree), None)
        .map_err(git_error)?;
    let mut changed = Vec::new();
    diff.foreach(
        &mut |delta, _| {
            if let Some(path) = delta.new_file().path().or_else(|| delta.old_file().path()) {
                changed.push(path.to_path_buf());
            }
            true
        },
        None,
        None,
        None,
    )
    .map_err(git_error)?;
    Ok(changed)
}

fn current_branch(repo: &Repository) -> Option<String> {
    repo.head()
        .ok()
        .and_then(|head| head.shorthand().map(str::to_owned))
}

fn normalize_managed_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>, StoreError> {
    let mut normalized = BTreeSet::new();
    for path in paths {
        if path.is_absolute() {
            return Err(StoreError::PathOutsideStore(path.clone()));
        }
        validate_relative_id(&path.to_string_lossy())?;
        if !is_managed_path(
            path,
            &[PathBuf::from(AUTH_DIR_NAME), PathBuf::from(CONFIG_PATH)],
        ) {
            return Err(StoreError::PathOutsideStore(path.clone()));
        }
        normalized.insert(path.clone());
    }
    Ok(normalized.into_iter().collect())
}

fn validate_commit_paths(
    repo: &Repository,
    parent: Option<Oid>,
    commit: Oid,
    managed: &[PathBuf],
) -> Result<(), StoreError> {
    let commit_tree = repo
        .find_commit(commit)
        .and_then(|commit| commit.tree())
        .map_err(git_error)?;
    let parent_tree = parent
        .map(|oid| repo.find_commit(oid).and_then(|commit| commit.tree()))
        .transpose()
        .map_err(git_error)?;
    let diff = repo
        .diff_tree_to_tree(parent_tree.as_ref(), Some(&commit_tree), None)
        .map_err(git_error)?;
    let mut invalid = None;
    diff.foreach(
        &mut |delta, _| {
            let path = delta.new_file().path().or_else(|| delta.old_file().path());
            if path.is_some_and(|path| !is_managed_path(path, managed)) {
                invalid = path.map(Path::to_path_buf);
                return false;
            }
            true
        },
        None,
        None,
        None,
    )
    .map_err(git_error)?;
    if let Some(path) = invalid {
        Err(StoreError::Backend(format!(
            "generated commit modified unmanaged path {}",
            path.display()
        )))
    } else {
        Ok(())
    }
}

fn rollback_local_commit(
    repo: &Repository,
    parent: Option<&git2::Commit<'_>>,
) -> Result<(), StoreError> {
    if let Some(parent) = parent {
        let object = repo.find_object(parent.id(), None).map_err(git_error)?;
        repo.reset(&object, ResetType::Mixed, None)
            .map_err(git_error)?;
    } else {
        let reference_name = repo
            .head()
            .ok()
            .and_then(|head| head.name().map(str::to_owned));
        if let Some(reference_name) = reference_name {
            repo.find_reference(&reference_name)
                .and_then(|mut reference| reference.delete())
                .map_err(git_error)?;
        }
        let mut index = repo.index().map_err(git_error)?;
        index.clear().map_err(git_error)?;
        index.write().map_err(git_error)?;
    }
    Ok(())
}

fn reset_index_to_head(repo: &Repository, index: &mut git2::Index) -> Result<(), StoreError> {
    let Ok(head) = repo.head() else {
        index.clear().map_err(git_error)?;
        return Ok(());
    };
    let tree = head.peel_to_tree().map_err(git_error)?;
    index.read_tree(&tree).map_err(git_error)
}

fn reserve_sibling_dir(repo_dir: &Path, purpose: &str) -> Result<PathBuf, StoreError> {
    let parent = repo_dir
        .parent()
        .ok_or(StoreError::InvalidConfig("git repository has no parent"))?;
    let name =
        repo_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(StoreError::InvalidConfig(
                "git repository name must be UTF-8",
            ))?;
    for attempt in 0..32 {
        let candidate = parent.join(format!(".{name}.{purpose}.{attempt}"));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(StoreError::Backend(format!(
        "cannot reserve isolated {purpose} directory"
    )))
}

fn ensure_managed_trees_equal(current: &Path, recovered: &Path) -> Result<(), StoreError> {
    let current = managed_snapshot(current)?;
    let recovered = managed_snapshot(recovered)?;
    if current == recovered {
        Ok(())
    } else {
        Err(StoreError::Backend(
            "corrupt repository has dirty or stale managed paths without a trustworthy baseline"
                .to_owned(),
        ))
    }
}

fn managed_snapshot(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, StoreError> {
    let mut snapshot = BTreeMap::new();
    collect_managed_files(root, &root.join(AUTH_DIR_NAME), &mut snapshot)?;
    let config = root.join(CONFIG_PATH);
    match fs::read(&config) {
        Ok(content) => {
            snapshot.insert(PathBuf::from(CONFIG_PATH), content);
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(snapshot)
}

fn collect_managed_files(
    root: &Path,
    directory: &Path,
    snapshot: &mut BTreeMap<PathBuf, Vec<u8>>,
) -> Result<(), StoreError> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(StoreError::Backend(format!(
                "managed recovery path is a symlink: {}",
                entry.path().display()
            )));
        }
        if kind.is_dir() {
            collect_managed_files(root, &entry.path(), snapshot)?;
        } else if kind.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .map(Path::to_path_buf)
                .map_err(|_| StoreError::PathOutsideStore(entry.path()))?;
            snapshot.insert(relative, fs::read(entry.path())?);
        }
    }
    Ok(())
}

fn is_repository_corruption(error: &StoreError) -> bool {
    let StoreError::Backend(message) = error else {
        return false;
    };
    let message = message.to_ascii_lowercase();
    [
        "corrupt",
        "invalid data in index",
        "object not found",
        "failed to parse",
        "reference 'head' not found",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn is_managed_path(path: &Path, managed: &[PathBuf]) -> bool {
    managed.iter().any(|candidate| {
        path == candidate || path.starts_with(candidate) || candidate.starts_with(path)
    })
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn collect_json_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), StoreError> {
    if !root.exists() {
        return Ok(());
    }
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

fn git_path(path: &Path) -> Result<String, StoreError> {
    path.to_str()
        .map(|path| path.replace('\\', "/"))
        .ok_or_else(|| StoreError::InvalidRecord("git path is not UTF-8".to_owned()))
}

fn is_empty_remote_error(error: &git2::Error) -> bool {
    matches!(error.code(), ErrorCode::UnbornBranch)
        || error.message().to_ascii_lowercase().contains("empty")
}

fn is_network_remote(remote: &str) -> bool {
    let remote = remote.to_ascii_lowercase();
    remote.starts_with("https://")
        || remote.starts_with("ssh://")
        || (remote.contains('@')
            && remote.contains(':')
            && !Path::new(remote.as_str()).is_absolute())
}

fn remote_has_inline_credentials(remote: &str) -> bool {
    let Some((scheme, authority)) = remote.split_once("://") else {
        // SCP-style SSH permits a public username (`git@host:path`); the
        // private credential still stays behind GitCredentialRef.
        return false;
    };
    let authority = authority.split('/').next().unwrap_or_default();
    match scheme.to_ascii_lowercase().as_str() {
        "http" | "https" => authority.contains('@'),
        // SSH usernames are routing metadata. Password-bearing userinfo is
        // still rejected; the private key remains behind GitCredentialRef.
        "ssh" => authority
            .split_once('@')
            .is_some_and(|(userinfo, _)| userinfo.contains(':')),
        _ => false,
    }
}

fn remote_kind(remote: &str) -> &'static str {
    if remote.to_ascii_lowercase().starts_with("https://") {
        "https"
    } else if is_network_remote(remote) {
        "ssh"
    } else {
        "local"
    }
}

fn validate_default_branch(branch: Option<String>) -> Result<Option<String>, StoreError> {
    branch
        .map(|branch| {
            let branch = branch.trim().to_owned();
            validate_branch_name(&branch)?;
            Ok(branch)
        })
        .transpose()
}

fn transport_error(error: GitTransportError) -> StoreError {
    StoreError::Backend(error.to_string())
}

/// Prunes only old, unreachable loose objects. Packed objects are retained;
/// this deliberately favors recovery grace over aggressive disk reclamation.
fn prune_unreachable_loose_objects(
    repo: &Repository,
    older_than: SystemTime,
) -> Result<(), StoreError> {
    let reachable = reachable_objects(repo)?;
    let objects = repo.path().join("objects");
    let directories = match fs::read_dir(objects) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for directory in directories {
        let directory = directory?;
        let prefix = directory.file_name().to_string_lossy().into_owned();
        if prefix.len() != 2 || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        if !directory.file_type()?.is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory.path())? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let suffix = entry.file_name().to_string_lossy().into_owned();
            if suffix.len() != 38 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                continue;
            }
            let Ok(oid) = Oid::from_str(format!("{prefix}{suffix}").as_str()) else {
                continue;
            };
            if reachable.contains(&oid) {
                continue;
            }
            let modified = entry.metadata()?.modified()?;
            if modified <= older_than {
                fs::remove_file(entry.path())?;
            }
        }
    }
    Ok(())
}

fn reachable_objects(repo: &Repository) -> Result<BTreeSet<Oid>, StoreError> {
    let mut reachable = BTreeSet::new();
    let mut revwalk = repo.revwalk().map_err(git_error)?;
    for reference in repo.references().map_err(git_error)? {
        let reference = reference.map_err(git_error)?;
        let Some(oid) = reference.target() else {
            continue;
        };
        reachable.insert(oid);
        if let Ok(commit) = repo.find_commit(oid) {
            revwalk.push(commit.id()).map_err(git_error)?;
        } else if let Ok(object) = repo.find_object(oid, None) {
            if let Ok(commit) = object.peel_to_commit() {
                revwalk.push(commit.id()).map_err(git_error)?;
            }
        }
    }
    for oid in revwalk {
        let oid = oid.map_err(git_error)?;
        reachable.insert(oid);
        let commit = repo.find_commit(oid).map_err(git_error)?;
        let tree = commit.tree().map_err(git_error)?;
        reachable.insert(tree.id());
        tree.walk(git2::TreeWalkMode::PreOrder, |_, entry| {
            reachable.insert(entry.id());
            git2::TreeWalkResult::Ok
        })
        .map_err(git_error)?;
    }
    Ok(reachable)
}

fn is_missing_remote_error(error: &git2::Error) -> bool {
    matches!(error.code(), ErrorCode::NotFound)
}

fn git_error(error: git2::Error) -> StoreError {
    StoreError::Backend(error.message().to_owned())
}
