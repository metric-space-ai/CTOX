// ref: internal/store/gitstore_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: includes typed network authority, lease, recovery, and maintenance coverage
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use git2::build::CheckoutBuilder;
use git2::{IndexAddOption, Oid, Repository, Signature};
use serde_json::json;
use tempfile::TempDir;

use crate::sdk::cliproxy::auth::{Auth, AuthStore, AuthStoreError};

use super::gitstore::{
    GitCredentialRef, GitPushRequest, GitRemoteRequest, GitStoreConfig, GitTokenStore,
    GitTransportAuthority, GitTransportError,
};

fn commit_all(repo: &Repository, message: &str) -> Oid {
    let mut index = repo.index().unwrap();
    index
        .add_all(["*"].iter(), IndexAddOption::DEFAULT, None)
        .unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    let signature = Signature::now("Store Test", "store@example.test").unwrap();
    let parent = repo.head().ok().and_then(|head| head.peel_to_commit().ok());
    match parent.as_ref() {
        Some(parent) => repo
            .commit(
                Some("HEAD"),
                &signature,
                &signature,
                message,
                &tree,
                &[parent],
            )
            .unwrap(),
        None => repo
            .commit(Some("HEAD"), &signature, &signature, message, &tree, &[])
            .unwrap(),
    }
}

fn push_branch(repo: &Repository, branch: &str) {
    repo.find_remote("origin")
        .unwrap()
        .push(&[format!("refs/heads/{branch}:refs/heads/{branch}")], None)
        .unwrap();
}

fn checkout_branch(repo: &Repository, branch: &str, start: Option<Oid>) {
    let reference = format!("refs/heads/{branch}");
    let has_commit = if let Some(start) = start {
        repo.reference(&reference, start, true, "test branch")
            .unwrap();
        true
    } else {
        false
    };
    repo.set_head(&reference).unwrap();
    if has_commit {
        let mut checkout = CheckoutBuilder::new();
        checkout.force();
        repo.checkout_head(Some(&mut checkout)).unwrap();
    }
}

fn setup_remote(root: &Path) -> (PathBuf, PathBuf) {
    let remote_path = root.join("remote.git");
    let remote = Repository::init_bare(&remote_path).unwrap();
    let seed_path = root.join("seed");
    let seed = Repository::init(&seed_path).unwrap();
    seed.remote("origin", remote_path.to_str().unwrap())
        .unwrap();

    checkout_branch(&seed, "trunk", None);
    fs::write(seed_path.join("branch.txt"), b"remote default branch\n").unwrap();
    fs::create_dir_all(seed_path.join("auths")).unwrap();
    fs::create_dir_all(seed_path.join("config")).unwrap();
    fs::write(seed_path.join("auths/.gitkeep"), b"").unwrap();
    fs::write(seed_path.join("config/.gitkeep"), b"").unwrap();
    let trunk = commit_all(&seed, "trunk");
    push_branch(&seed, "trunk");

    checkout_branch(&seed, "release/2026", Some(trunk));
    fs::write(seed_path.join("branch.txt"), b"release branch\n").unwrap();
    commit_all(&seed, "release");
    push_branch(&seed, "release/2026");
    remote.set_head("refs/heads/trunk").unwrap();
    (remote_path, seed_path)
}

fn store(remote: &Path, workspace: &Path, branch: Option<&str>) -> GitTokenStore {
    GitTokenStore::new(GitStoreConfig {
        remote: remote.to_string_lossy().into_owned(),
        branch: branch.map(str::to_owned),
        auth_dir: workspace.join("auths"),
    })
    .unwrap()
}

fn branch_name(repo_path: &Path) -> String {
    Repository::open(repo_path)
        .unwrap()
        .head()
        .unwrap()
        .shorthand()
        .unwrap()
        .to_owned()
}

fn advance(seed_path: &Path, branch: &str, content: &[u8]) {
    let seed = Repository::open(seed_path).unwrap();
    let oid = seed.refname_to_id(&format!("refs/heads/{branch}")).unwrap();
    checkout_branch(&seed, branch, Some(oid));
    fs::write(seed_path.join("branch.txt"), content).unwrap();
    commit_all(&seed, "advance");
    push_branch(&seed, branch);
}

#[test]
fn ensure_repository_tracks_remote_default_and_configured_branch() {
    let root = TempDir::new().unwrap();
    let (remote, seed) = setup_remote(root.path());
    let default_workspace = root.path().join("workspace-default");
    let default = store(&remote, &default_workspace, None);
    default.ensure_repository().unwrap();
    assert_eq!(branch_name(&default_workspace), "trunk");
    assert_eq!(
        fs::read(default_workspace.join("branch.txt")).unwrap(),
        b"remote default branch\n"
    );

    let release_workspace = root.path().join("workspace-release");
    let release = store(&remote, &release_workspace, Some("release/2026"));
    release.ensure_repository().unwrap();
    assert_eq!(branch_name(&release_workspace), "release/2026");
    assert_eq!(
        fs::read(release_workspace.join("branch.txt")).unwrap(),
        b"release branch\n"
    );

    advance(&seed, "trunk", b"trunk updated\n");
    advance(&seed, "release/2026", b"release updated\n");
    default.ensure_repository().unwrap();
    release.ensure_repository().unwrap();
    assert_eq!(
        fs::read(default_workspace.join("branch.txt")).unwrap(),
        b"trunk updated\n"
    );
    assert_eq!(
        fs::read(release_workspace.join("branch.txt")).unwrap(),
        b"release updated\n"
    );
}

#[test]
fn missing_configured_branch_fails_without_changing_remote_head() {
    let root = TempDir::new().unwrap();
    let (remote_path, _) = setup_remote(root.path());
    let missing = store(
        &remote_path,
        &root.path().join("workspace"),
        Some("missing"),
    );
    assert!(missing.ensure_repository().is_err());
    let remote = Repository::open_bare(remote_path).unwrap();
    assert_eq!(remote.head().unwrap().shorthand(), Some("trunk"));
}

#[test]
fn save_list_delete_and_watcher_guard_preserve_remote_authority() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();
    let mut auth = Auth::default();
    auth.id = "team/protected.json".to_owned();
    auth.metadata = BTreeMap::from([
        ("type".to_owned(), json!("codex")),
        ("access_token".to_owned(), json!("secret")),
    ]);
    let path = store.save(&mut auth).unwrap();
    assert_eq!(store.list().unwrap().len(), 1);
    assert!(!format!("{store:?}").contains("secret"));

    fs::remove_file(&path).unwrap();
    let error = store
        .persist_auth_files("Remove auth protected.json", [&path])
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("refusing watcher-originated removal"));
    assert!(remote_has_path(
        &remote,
        "trunk",
        "auths/team/protected.json"
    ));

    store.delete("team/protected.json").unwrap();
    assert!(!remote_has_path(
        &remote,
        "trunk",
        "auths/team/protected.json"
    ));
    store.delete("team/protected.json").unwrap();
    store
        .persist_auth_files("Remove auth protected.json", [&path])
        .unwrap();
    assert!(!remote_has_path(
        &remote,
        "trunk",
        "auths/team/protected.json"
    ));
}

#[test]
fn dirty_nonconflicting_managed_path_survives_pull_but_overlap_fails_closed() {
    let root = TempDir::new().unwrap();
    let (remote, seed_path) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();
    fs::write(store.config_path(), b"local config\n").unwrap();
    advance(&seed_path, "trunk", b"remote unrelated\n");
    store.ensure_repository().unwrap();
    assert_eq!(fs::read(store.config_path()).unwrap(), b"local config\n");

    fs::write(workspace.join("auths/conflict.json"), b"local\n").unwrap();
    let seed = Repository::open(&seed_path).unwrap();
    checkout_branch(
        &seed,
        "trunk",
        Some(seed.refname_to_id("refs/heads/trunk").unwrap()),
    );
    fs::write(seed_path.join("auths/conflict.json"), b"remote\n").unwrap();
    commit_all(&seed, "remote conflict");
    push_branch(&seed, "trunk");
    assert!(store.ensure_repository().is_err());
    assert_eq!(
        fs::read(workspace.join("auths/conflict.json")).unwrap(),
        b"local\n"
    );
}

#[test]
fn paths_outside_repository_are_rejected_before_mutation() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();
    let outside = root.path().join("outside.json");
    fs::write(&outside, b"outside").unwrap();
    assert!(store.persist_auth_files("bad path", [&outside]).is_err());
    assert_eq!(fs::read(outside).unwrap(), b"outside");
    assert!(store.delete("../../outside.json").is_err());

    let network = GitTokenStore::new(GitStoreConfig {
        remote: "https://example.test/private.git".to_owned(),
        branch: None,
        auth_dir: root.path().join("network/auths"),
    });
    assert!(network
        .unwrap_err()
        .to_string()
        .contains("isolated CTOX git transport"));
}

#[test]
fn persist_config_resets_unrelated_index_changes_and_whitelists_commit_paths() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();

    let repo = Repository::open(&workspace).unwrap();
    let mut index = repo.index().unwrap();
    index.remove_path(Path::new("branch.txt")).unwrap();
    index.write().unwrap();
    fs::write(store.config_path(), b"typed: true\n").unwrap();

    store.persist_config().unwrap();
    assert!(remote_has_path(&remote, "trunk", "branch.txt"));
    assert!(remote_has_path(&remote, "trunk", "config/config.yaml"));
}

#[test]
fn failed_push_rolls_back_head_and_preserves_retryable_worktree_content() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();
    let repo = Repository::open(&workspace).unwrap();
    let original_head = repo.head().unwrap().target().unwrap();
    let config_path = store.config_path();
    fs::write(&config_path, b"retryable: true\n").unwrap();

    let unavailable = root.path().join("remote-unavailable.git");
    fs::rename(&remote, &unavailable).unwrap();
    let error = store
        .commit_and_push_with_options(
            &repo,
            "Update configuration",
            false,
            &[PathBuf::from("config/config.yaml")],
        )
        .unwrap_err();
    assert!(error.to_string().contains("backend"));
    assert_eq!(repo.head().unwrap().target(), Some(original_head));
    assert_eq!(fs::read(&config_path).unwrap(), b"retryable: true\n");

    fs::rename(&unavailable, &remote).unwrap();
    store.persist_config().unwrap();
    assert!(remote_has_path(&remote, "trunk", "config/config.yaml"));
}

#[test]
fn stale_lease_rejects_push_and_retry_preserves_remote_only_paths() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace_a = root.path().join("workspace-a");
    let workspace_b = root.path().join("workspace-b");
    let store_a = store(&remote, &workspace_a, None);
    let store_b = store(&remote, &workspace_b, None);
    store_a.ensure_repository().unwrap();
    store_b.ensure_repository().unwrap();

    let mut remote_auth = Auth::default();
    remote_auth.id = "remote-only.json".to_owned();
    remote_auth.metadata = BTreeMap::from([("type".to_owned(), json!("codex"))]);
    store_b.save(&mut remote_auth).unwrap();

    fs::write(store_a.config_path(), b"source: stale-a\n").unwrap();
    let repo_a = Repository::open(&workspace_a).unwrap();
    let original_head = repo_a.head().unwrap().target();
    let error = store_a
        .commit_and_push_with_options(
            &repo_a,
            "Update stale config",
            false,
            &[PathBuf::from("config/config.yaml")],
        )
        .unwrap_err();
    assert!(error.to_string().contains("lease") || error.to_string().contains("backend"));
    assert_eq!(repo_a.head().unwrap().target(), original_head);
    assert!(remote_has_path(&remote, "trunk", "auths/remote-only.json"));
    assert!(!remote_has_path(&remote, "trunk", "config/config.yaml"));

    store_a.persist_config().unwrap();
    assert!(remote_has_path(&remote, "trunk", "auths/remote-only.json"));
    assert!(remote_has_path(&remote, "trunk", "config/config.yaml"));
}

#[test]
fn corrupt_clean_clone_recovers_but_dirty_managed_bytes_fail_closed() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());

    let clean_workspace = root.path().join("clean-workspace");
    let clean = store(&remote, &clean_workspace, None);
    clean.ensure_repository().unwrap();
    fs::write(clean_workspace.join(".git/HEAD"), b"broken head\n").unwrap();
    clean.ensure_repository().unwrap();
    assert_eq!(branch_name(&clean_workspace), "trunk");

    let dirty_workspace = root.path().join("dirty-workspace");
    let dirty = store(&remote, &dirty_workspace, None);
    dirty.ensure_repository().unwrap();
    fs::write(dirty.config_path(), b"uncommitted: secret\n").unwrap();
    fs::write(dirty_workspace.join(".git/HEAD"), b"broken head\n").unwrap();
    let error = dirty.ensure_repository().unwrap_err();
    assert!(error.to_string().contains("failed closed"));
    assert_eq!(
        fs::read(dirty.config_path()).unwrap(),
        b"uncommitted: secret\n"
    );
}

#[test]
fn auth_store_trait_preserves_error_classes_and_provider_neutral_records() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let workspace = root.path().join("workspace");
    let store = store(&remote, &workspace, None);
    store.ensure_repository().unwrap();

    let mut auth = Auth::default();
    auth.id = "provider-neutral.json".to_owned();
    auth.metadata = BTreeMap::from([("type".to_owned(), json!("kimi"))]);
    let path = <GitTokenStore as AuthStore>::save(&store, &auth).unwrap();
    assert!(Path::new(&path).starts_with(store.auth_dir()));
    let records = <GitTokenStore as AuthStore>::list(&store).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "provider-neutral.json");
    <GitTokenStore as AuthStore>::delete(&store, &auth.id).unwrap();
    assert!(<GitTokenStore as AuthStore>::list(&store)
        .unwrap()
        .is_empty());

    let invalid = Auth::default();
    assert_eq!(
        <GitTokenStore as AuthStore>::save(&store, &invalid),
        Err(AuthStoreError::InvalidRecord)
    );
}

#[derive(Default)]
struct TransportEvents {
    cloned: usize,
    fetched: usize,
    pushed: Vec<GitPushRequest>,
    maintained: usize,
}

struct RecordingNetworkAuthority {
    backing: PathBuf,
    events: Mutex<TransportEvents>,
}

impl RecordingNetworkAuthority {
    fn new(backing: PathBuf) -> Self {
        Self {
            backing,
            events: Mutex::new(TransportEvents::default()),
        }
    }
}

impl GitTransportAuthority for RecordingNetworkAuthority {
    fn clone_repository(&self, request: &GitRemoteRequest) -> Result<(), GitTransportError> {
        git2::build::RepoBuilder::new()
            .clone(self.backing.to_str().unwrap(), &request.repository)
            .map_err(|_| GitTransportError::Unavailable)?;
        let repo = Repository::open(&request.repository)
            .map_err(|_| GitTransportError::InvalidResponse)?;
        repo.remote_set_url("origin", &request.remote)
            .map_err(|_| GitTransportError::InvalidResponse)?;
        self.events.lock().unwrap().cloned += 1;
        Ok(())
    }

    fn fetch(&self, _request: &GitRemoteRequest) -> Result<(), GitTransportError> {
        self.events.lock().unwrap().fetched += 1;
        Ok(())
    }

    fn default_branch(
        &self,
        _request: &GitRemoteRequest,
    ) -> Result<Option<String>, GitTransportError> {
        Ok(Some("trunk".to_owned()))
    }

    fn push(&self, request: &GitPushRequest) -> Result<(), GitTransportError> {
        self.events.lock().unwrap().pushed.push(request.clone());
        Ok(())
    }

    fn maintenance(&self, _request: &GitRemoteRequest) -> Result<(), GitTransportError> {
        self.events.lock().unwrap().maintained += 1;
        Ok(())
    }
}

#[test]
fn authenticated_network_transport_is_typed_leased_and_redacted() {
    let root = TempDir::new().unwrap();
    let (remote, _) = setup_remote(root.path());
    let authority = Arc::new(RecordingNetworkAuthority::new(remote));
    let credential = GitCredentialRef::new("ctox-secret-handle-7").unwrap();
    let workspace = root.path().join("network-workspace");
    let store = GitTokenStore::new_with_transport(
        GitStoreConfig {
            remote: "https://git.example.test/team/private.git".to_owned(),
            branch: None,
            auth_dir: workspace.join("auths"),
        },
        credential.clone(),
        authority.clone(),
    )
    .unwrap();

    store.ensure_repository().unwrap();
    let mut auth = Auth::default();
    auth.id = "network.json".to_owned();
    auth.metadata = BTreeMap::from([("type".to_owned(), json!("codex"))]);
    store.save(&mut auth).unwrap();

    let events = authority.events.lock().unwrap();
    assert_eq!(events.cloned, 1);
    assert!(events.fetched >= 2);
    assert_eq!(events.pushed.len(), 1);
    assert_eq!(events.pushed[0].branch, "trunk");
    assert!(events.pushed[0].expected_remote_oid.is_some());
    assert_eq!(events.pushed[0].credential.as_str(), credential.as_str());
    assert_eq!(events.maintained, 1);
    let debug = format!("{store:?} {credential:?} {:?}", events.pushed[0]);
    assert!(!debug.contains("ctox-secret-handle-7"));
    assert!(!debug.contains("private.git"));
}

#[test]
fn network_remotes_fail_closed_without_authority_or_with_inline_credentials() {
    let root = TempDir::new().unwrap();
    let cfg = |remote: &str| GitStoreConfig {
        remote: remote.to_owned(),
        branch: None,
        auth_dir: root.path().join("workspace/auths"),
    };
    assert!(GitTokenStore::new(cfg("ssh://git.example.test/team/repo.git")).is_err());
    let authority = Arc::new(RecordingNetworkAuthority::new(root.path().join("unused")));
    assert!(GitTokenStore::new_with_transport(
        cfg("https://user:password@git.example.test/team/repo.git"),
        GitCredentialRef::new("secret-handle").unwrap(),
        authority,
    )
    .is_err());
    let ssh_authority = Arc::new(RecordingNetworkAuthority::new(root.path().join("unused")));
    assert!(GitTokenStore::new_with_transport(
        cfg("ssh://git@git.example.test/team/repo.git"),
        GitCredentialRef::new("ssh-key-handle").unwrap(),
        ssh_authority,
    )
    .is_ok());
}

fn remote_has_path(remote_path: &Path, branch: &str, path: &str) -> bool {
    let repo = Repository::open_bare(remote_path).unwrap();
    let oid = repo.refname_to_id(&format!("refs/heads/{branch}")).unwrap();
    let exists = repo
        .find_commit(oid)
        .unwrap()
        .tree()
        .unwrap()
        .get_path(Path::new(path))
        .is_ok();
    exists
}
