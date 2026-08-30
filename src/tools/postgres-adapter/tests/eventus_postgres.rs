use ctox_postgres_adapter::{PostgresAdapter, PostgresConfig, PostgresSslMode, SqlParameter};
use serde_json::Value;

fn eventus_config() -> PostgresConfig {
    PostgresConfig {
        server: "127.0.0.1".into(),
        port: 55_432,
        database: "outbound_rfq_test".into(),
        user: "rfqdev".into(),
        password: None,
        sslmode: PostgresSslMode::Disable,
        request_timeout_ms: 30_000,
        max_rows: 5_000,
        allow_writes: true,
        application_name: "ctox-postgres-adapter-integration-test".into(),
    }
}

#[tokio::test]
#[ignore = "requires the local Eventus PostgreSQL test database"]
async fn eventus_query_binding_and_transaction_rollback() -> anyhow::Result<()> {
    let mut adapter = PostgresAdapter::new(eventus_config())?;

    let rows = adapter.query("SELECT 1 AS value", &[]).await?;
    assert_eq!(rows, vec![serde_json::json!({"value": 1})]);

    let first_job = adapter
        .query(
            "SELECT id FROM ctox_jobs ORDER BY requested_at DESC LIMIT 1",
            &[],
        )
        .await?
        .into_iter()
        .next()
        .expect("ctox_jobs contains a test row");
    let job_id = first_job
        .get("id")
        .and_then(Value::as_str)
        .expect("ctox_jobs.id is text")
        .to_owned();
    let rows = adapter
        .query(
            "SELECT id, status, requested_at, payload FROM ctox_jobs WHERE id=@P1",
            &[SqlParameter::String(job_id.clone())],
        )
        .await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("id").and_then(Value::as_str),
        Some(job_id.as_str())
    );
    assert!(rows[0]
        .get("requested_at")
        .and_then(Value::as_str)
        .is_some_and(|value| value.contains('T')));

    adapter
        .execute(
            "CREATE TEMP TABLE IF NOT EXISTS ctox_postgres_adapter_tx_test (id BIGINT NOT NULL)",
            &[],
        )
        .await?;
    adapter.begin_transaction().await?;
    adapter
        .execute(
            "INSERT INTO ctox_postgres_adapter_tx_test(id) VALUES(@P1)",
            &[SqlParameter::I32(42)],
        )
        .await?;
    adapter.rollback_transaction().await?;
    let rows = adapter
        .query(
            "SELECT COUNT(*) AS row_count FROM ctox_postgres_adapter_tx_test",
            &[],
        )
        .await?;
    assert_eq!(rows, vec![serde_json::json!({"row_count": 0})]);

    Ok(())
}
