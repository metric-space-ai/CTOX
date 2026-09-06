use super::*;

#[tokio::test]
async fn failed_listener_join_can_be_followed_by_shutdown() {
    let task = tokio::spawn(std::future::pending::<io::Result<()>>());
    task.abort();
    let mut host = LocalAuthorityHost {
        endpoint: PathBuf::new(),
        stop: None,
        task: Some(task),
    };
    assert!(host.wait_stopped().await.is_err());
    // A completed failed JoinHandle must be consumed, not polled a second time.
    host.shutdown().await.unwrap();
}

#[test]
fn private_host_recovers_only_dead_sockets_and_never_replaces_a_live_listener() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let path = root.path().join("authority.sock");
    let live = StdListener::bind(&path).unwrap();
    assert_eq!(
        BoundSocket::bind(root.path()).err().unwrap().kind(),
        io::ErrorKind::AddrInUse
    );
    assert!(path.exists());
    drop(live);
    let host = BoundSocket::bind(root.path()).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
    assert_eq!(
        BoundSocket::bind(root.path()).err().unwrap().kind(),
        io::ErrorKind::AddrInUse
    );
    drop(host);
    assert!(!path.exists());
    assert!(BoundSocket::bind(root.path()).is_ok());
}

#[test]
fn host_preserves_foreign_files_and_rejects_shared_or_symlinked_directories() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let path = root.path().join("authority.sock");
    fs::write(&path, b"unrelated file").unwrap();
    assert!(BoundSocket::bind(root.path()).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"unrelated file");
    fs::remove_file(&path).unwrap();
    let target = root.path().join("target");
    fs::write(&target, b"keep me").unwrap();
    std::os::unix::fs::symlink(&target, &path).unwrap();
    assert!(BoundSocket::bind(root.path()).is_err());
    assert_eq!(fs::read(&target).unwrap(), b"keep me");
    fs::remove_file(&path).unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(BoundSocket::bind(root.path()).is_err());
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let alias = root.path().join("alias");
    std::os::unix::fs::symlink(root.path(), &alias).unwrap();
    assert!(BoundSocket::bind(&alias).is_err());
}

#[test]
fn late_host_cleanup_cannot_remove_a_replacement_path() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let host = BoundSocket::bind(root.path()).unwrap();
    let path = host.path.clone();
    fs::remove_file(&path).unwrap();
    fs::write(&path, b"replacement").unwrap();
    drop(host);
    assert_eq!(fs::read(&path).unwrap(), b"replacement");
}
