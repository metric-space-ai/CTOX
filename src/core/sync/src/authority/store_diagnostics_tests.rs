use super::*;

#[tokio::test]
async fn cancelled_waiter_does_not_report_unfinished_storage_as_cancelled() {
    let store = SqliteStore::open(Path::new(":memory:")).unwrap();
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let writer_store = store.clone();
    let writer = tokio::spawn(async move {
        writer_store
            .run("held_write", move |conn| {
                entered_tx.send(()).unwrap();
                release_rx.recv().map_err(other)?;
                put(conn, "diagnostic_fixture", &true)
            })
            .await
    });
    entered_rx.await.unwrap();
    writer.abort();
    assert!(writer.await.unwrap_err().is_cancelled());
    // The blocking closure still holds the connection and can commit the write.
    assert_eq!(store.diagnostics()["held_write"].running.active, 1);
    assert_eq!(store.diagnostics()["held_write"].interrupted, 0);
    release_tx.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while store.diagnostics()["held_write"].succeeded != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let value: Option<bool> = store
        .run("read_fixture", |conn| get(conn, "diagnostic_fixture"))
        .await
        .unwrap();
    assert_eq!(value, Some(true));
    let observation = store.diagnostics()["held_write"].clone();
    assert_eq!(observation.running.active, 0);
    assert_eq!(observation.interrupted, 0);
}
