// Origin: CTOX
// License: Apache-2.0
// Included in rxdb_peer::tests to exercise the production peer entry points.

#[tokio::test]
async fn peer_startup_runtime_projection_does_not_wait_for_unrelated_source_lock(
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    // Initialize the source before the liveness measurement, just as bring-up
    // does. The cross-loop lock below represents a slow, unrelated source.
    store::runtime_settings_for_rxdb(root.path())?;
    let path = store::rxdb_store_path(root.path());
    fs::create_dir_all(path.parent().unwrap())?;
    let database = open_test_database(path).await?;
    let creators = collection_creators()
        .into_iter()
        .filter(|(name, _)| name == "business_runtime_settings")
        .collect();
    database
        .add_collections(creators)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let lock = Arc::new(AsyncMutex::new(()));
    let held = lock.lock().await;
    let task = tokio::spawn(sync_runtime_settings_background_loop(
        root.path().to_path_buf(),
        Arc::clone(&database),
        Arc::clone(&lock),
    ));
    let collection = database.collection("business_runtime_settings").unwrap();
    let observed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let documents = collection
                .storage_instance
                .find_documents_by_id(&["runtime-settings".to_string()], false)
                .await
                .unwrap();
            if !documents.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    task.abort();
    let _ = task.await;
    drop(held);
    assert!(
        observed.is_ok(),
        "an unrelated first projection must not block runtime readiness"
    );
    Ok(())
}

#[tokio::test]
async fn peer_startup_desktop_scan_releases_writer_after_each_file() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let path = store::rxdb_store_path(root.path());
    fs::create_dir_all(path.parent().unwrap())?;
    let database = open_test_database(path).await?;
    let creators = collection_creators()
        .into_iter()
        .filter(|(name, _)| {
            matches!(
                name.as_str(),
                "desktop_files" | "desktop_folders" | "desktop_file_chunks"
            )
        })
        .collect();
    database
        .add_collections(creators)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let workspace = tempfile::tempdir()?;
    let paths = (0..2)
        .map(|n| workspace.path().join(format!("fixture-{n}.txt")))
        .collect::<Vec<_>>();
    for path in &paths {
        fs::write(path, "small file")?;
    }
    let scan_root = DesktopFileScanRoot {
        path: workspace.path().to_path_buf(),
        label: "Fixture".to_string(),
    };
    let scan = DesktopFileIndexScan {
        scan_roots: vec![scan_root.clone()],
        candidates: paths
            .iter()
            .map(|path| DesktopFileIndexCandidate {
                path: path.clone(),
                scan_root: scan_root.clone(),
            })
            .collect(),
        stamp: DesktopFileIndexProjectionStamp {
            scan_root_count: 1,
            candidate_count: 2,
            truncated: true,
            content_hash: "fixture".to_string(),
        },
    };
    let ids = paths
        .iter()
        .map(|path| desktop_file_id(path))
        .collect::<Vec<_>>();
    let held = NATIVE_RXDB_WRITE_LOCK.lock().await;
    let mut scan = Box::pin(sync_desktop_file_scan_with_database(
        root.path(),
        &database,
        scan,
    ));
    assert!(futures_util::poll!(&mut scan).is_pending());
    let mut foreground = Box::pin(NATIVE_RXDB_WRITE_LOCK.lock());
    assert!(futures_util::poll!(&mut foreground).is_pending());
    drop(held);
    let inspect = async {
        let _held = foreground.await;
        database
            .collection("desktop_files")
            .unwrap()
            .storage_instance
            .find_documents_by_id(&ids, false)
            .await
            .unwrap()
            .len()
    };
    let (indexed, visible_before_foreground) = tokio::join!(scan, inspect);
    assert_eq!(indexed?, 2);
    assert_eq!(
        visible_before_foreground, 1,
        "foreground work must run before indexing the second file"
    );
    Ok(())
}
