use super::*;
use tempfile::TempDir;

fn setup() -> Result<(TempDir, Connection)> {
    let root = tempfile::tempdir()?;
    std::fs::create_dir_all(root.path().join("runtime"))?;
    let conn = Connection::open(crate::paths::core_db(root.path()))?;
    conn.execute_batch("CREATE TABLE communication_routing_state(message_key TEXT PRIMARY KEY,route_status TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE communication_messages(message_key TEXT PRIMARY KEY,channel TEXT,direction TEXT);
        CREATE TRIGGER fixture_queue_identity AFTER INSERT ON communication_routing_state BEGIN INSERT INTO communication_messages VALUES(new.message_key,'queue','inbound'); END;
        CREATE TABLE ctox_harness_flow_events(event_id TEXT PRIMARY KEY,event_kind TEXT,title TEXT,body_text TEXT,message_key TEXT,work_id TEXT,attempt_index INTEGER,metadata_json TEXT,created_at TEXT);
        CREATE TABLE worker_attempt_finalizations(attempt_id TEXT PRIMARY KEY,work_key TEXT,status TEXT,agent_outcome TEXT,created_at TEXT,updated_at TEXT,terminal_at TEXT,error_text TEXT,resumable INTEGER);
        CREATE TABLE api_model_cost_events(provider TEXT,model TEXT,turn_id TEXT,input_tokens INTEGER,cached_input_tokens INTEGER,output_tokens INTEGER,reasoning_output_tokens INTEGER,day TEXT);
        CREATE TABLE api_model_price_rates(provider TEXT,model TEXT,input_usd_per_million REAL,cached_input_usd_per_million REAL,output_usd_per_million REAL,effective_from_day TEXT);")?;
    let rxdb = Connection::open(store::rxdb_store_path(root.path()))?;
    for (name, version) in [
        ("ctox_queue_tasks", 2),
        ("ctox_harness_events", 0),
        ("ctox_harness_status", 0),
        ("ctox_runs", 1),
    ] {
        rxdb.execute_batch(&format!("CREATE TABLE ctox_business_os__{name}__v{version}(id TEXT PRIMARY KEY,revision TEXT,deleted INTEGER DEFAULT 0,lastWriteTime REAL DEFAULT 0,data TEXT NOT NULL);"))?;
    }
    Ok((root, conn))
}

fn record(root: &Path, collection: &str, id: &str) -> Result<Value> {
    let raw: String = store::open_store(root)?.query_row(
        "SELECT payload_json FROM business_records WHERE collection=?1 AND record_id=?2",
        params![collection, id],
        |r| r.get(0),
    )?;
    Ok(serde_json::from_str(&raw)?)
}

#[test]
fn malformed_pause_is_visible_without_stopping_admission_and_recovers() -> Result<()> {
    let (root, conn) = setup()?;
    crate::inference::runtime_env::set_runtime_env_value(root.path(), "queue.pause", "{broken")?;
    assert!(!super::super::queue_is_paused(root.path()));
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    project_status(root.path(), &conn, &mut writer, &WorkerSnapshot::default())?;
    let status = record(root.path(), "ctox_harness_status", "harness")?;
    assert_eq!(status["paused"], false);
    assert!(status["last_error"]
        .as_str()
        .unwrap()
        .contains("Invalid queue.pause"));
    crate::inference::runtime_env::set_runtime_env_value(
        root.path(),
        "queue.pause",
        r#"{"paused":false,"reason":null}"#,
    )?;
    // Also clear a diagnostic restored from the previously persisted singleton.
    let snapshot = persisted_snapshot(root.path())?;
    project_status(root.path(), &conn, &mut writer, &snapshot)?;
    assert!(record(root.path(), "ctox_harness_status", "harness")?["last_error"].is_null());
    Ok(())
}

#[test]
fn event_step_uses_the_plan_at_emission_and_unknown_eligibility_replays() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('task','leased','2026-01-01T00:00:00Z')",
        [],
    )?;
    for (id, at, steps) in [
        (
            "plan-1",
            "2026-01-01T00:00:01Z",
            json!([{"step":"one","status":"in_progress"},{"step":"two","status":"pending"}]),
        ),
        (
            "plan-2",
            "2026-01-01T00:00:03Z",
            json!([{"step":"one","status":"completed"},{"step":"two","status":"in_progress"}]),
        ),
    ] {
        conn.execute("INSERT INTO ctox_harness_flow_events VALUES(?1,'worker.plan_updated','Plan','','task',NULL,NULL,?2,?3)",
            params![id, json!({"attempt_id":"attempt","attempt":7,"plan":{"plan":steps}}).to_string(), at])?;
    }
    for (id, at) in [
        ("tool-1", "2026-01-01T00:00:02Z"),
        ("tool-2", "2026-01-01T00:00:04Z"),
    ] {
        conn.execute("INSERT INTO ctox_harness_flow_events VALUES(?1,'worker.tool_started','Tool','','task',NULL,NULL,?2,?3)",
            params![id, json!({"attempt_id":"attempt"}).to_string(), at])?;
    }
    project_events(
        root.path(),
        &conn,
        &mut BusinessProjectionWriter::open(root.path())?,
    )?;
    assert_eq!(
        record(root.path(), "ctox_harness_events", "tool-1")?["step_position"],
        1
    );
    assert_eq!(
        record(root.path(), "ctox_harness_events", "tool-2")?["step_position"],
        2
    );
    assert_eq!(
        record(root.path(), "ctox_harness_events", "tool-1")?["attempt"],
        7
    );
    Ok(())
}

#[test]
fn invalid_required_projection_times_skip_documents_and_replay_after_repair() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('task','leased','2026-01-01T00:00:00Z')",
        [],
    )?;
    conn.execute("INSERT INTO ctox_harness_flow_events VALUES('bad-event','worker.phase','Phase','','task',NULL,NULL,'{}','bad')", [])?;
    conn.execute("INSERT INTO worker_attempt_finalizations VALUES('bad-run','work','failed','failed','bad','bad','bad',NULL,0)", [])?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    for _ in 0..2 {
        project_events(root.path(), &conn, &mut writer)?;
        project_runs(root.path(), &conn, &mut writer)?;
        let count: i64 = store::open_store(root.path())?.query_row(
            "SELECT COUNT(*) FROM business_records WHERE collection IN ('ctox_harness_events','ctox_runs') AND deleted=0", [], |row| row.get(0))?;
        assert_eq!(
            count, 0,
            "invalid required/index timestamps must not reach replication"
        );
    }
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM ctox_harness_flow_events", [], |row| {
            row.get::<_, i64>(0)
        })?,
        1
    );
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM worker_attempt_finalizations",
            [],
            |row| row.get::<_, i64>(0)
        )?,
        1
    );
    conn.execute(
        "UPDATE ctox_harness_flow_events SET created_at='2026-01-01T00:00:00Z'",
        [],
    )?;
    conn.execute("UPDATE worker_attempt_finalizations SET terminal_at='1767225603000', updated_at='1767225603000'", [])?;
    let before_replay = Utc::now().timestamp_millis();
    project_events(root.path(), &conn, &mut writer)?;
    project_runs(root.path(), &conn, &mut writer)?;
    let event = record(root.path(), "ctox_harness_events", "bad-event")?;
    assert_eq!(event["created_at_ms"], 1767225600000_i64);
    // The writer stamps observation time, independently of source creation time.
    assert!(event["updated_at_ms"]
        .as_i64()
        .is_some_and(|stamp| (before_replay..=Utc::now().timestamp_millis()).contains(&stamp)));
    let run = record(root.path(), "ctox_runs", "bad-run")?;
    assert_eq!(run["finished_at_ms"], 1767225603000_i64);
    assert!(run["updated_at_ms"]
        .as_i64()
        .is_some_and(|stamp| (before_replay..=Utc::now().timestamp_millis()).contains(&stamp)));
    assert!(run["started_at_ms"].is_null());
    assert!(run["metrics"]["elapsed_ms"].is_null());
    Ok(())
}

#[test]
fn invalid_run_times_remain_unknown_instead_of_epoch_sized_elapsed() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute("INSERT INTO worker_attempt_finalizations VALUES('invalid','work','failed','failed','bad','1767225603000','1767225603000',NULL,0)", [])?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    project_runs(root.path(), &conn, &mut writer)?;
    let run = record(root.path(), "ctox_runs", "invalid")?;
    assert!(run["started_at_ms"].is_null());
    assert!(run["metrics"]["elapsed_ms"].is_null());
    assert_eq!(run["finished_at_ms"], 1767225603000_i64);
    assert!(
        run.get("attempt_id").is_none(),
        "id alone is the attempt identity"
    );
    assert_eq!(millis("bad"), None);
    assert_eq!(millis("0"), Some(0));
    Ok(())
}

#[test]
fn native_cockpit_queries_use_the_declared_time_task_and_crew_indexes() -> Result<()> {
    let (root, _) = setup()?;
    let _writer = BusinessProjectionWriter::open(root.path())?;
    let conn = store::open_store(root.path())?;
    for (query, index) in [
        ("SELECT record_id FROM business_records WHERE collection='ctox_runs' AND deleted=0 AND json_extract(payload_json,'$.task_id')='task' ORDER BY json_extract(payload_json,'$.finished_at_ms') DESC", "idx_cockpit_run_task_finished"),
        ("SELECT record_id FROM business_records WHERE collection='ctox_runs' AND deleted=0 AND json_extract(payload_json,'$.crew_member_id')='member'", "idx_cockpit_run_crew"),
        ("SELECT record_id FROM business_records WHERE collection='ctox_harness_events' AND deleted=0 AND json_extract(payload_json,'$.created_at_ms')>=1 ORDER BY json_extract(payload_json,'$.created_at_ms')", "idx_cockpit_event_created"),
    ] {
        let details = conn.prepare(&format!("EXPLAIN QUERY PLAN {query}"))?
            .query_map([], |r| r.get::<_, String>(3))?
            .collect::<rusqlite::Result<Vec<_>>>()?.join("\n");
        assert!(details.contains(index), "{index}: {details}");
    }
    Ok(())
}

#[test]
fn keyset_pages_project_every_active_task_and_older_active_run() -> Result<()> {
    let (root, conn) = setup()?;
    for index in 0..PROJECTION_PAGE_SIZE + 1 {
        let id = format!("active-{index:04}");
        conn.execute(
            "INSERT INTO communication_routing_state VALUES(?1,'pending','2020-01-01T00:00:00Z')",
            [&id],
        )?;
        conn.execute("INSERT INTO worker_attempt_finalizations VALUES(?1,'work','failed','failed','1','2','2',NULL,1)", [&id])?;
        conn.execute("INSERT INTO ctox_harness_flow_events VALUES(?1,'worker.phase','Phase','',?1,NULL,1,?2,'2020-01-01T00:00:00Z')",
            params![id, json!({"attempt_id":id,"step_position":null}).to_string()])?;
    }
    for index in 0..501 {
        conn.execute("INSERT INTO worker_attempt_finalizations VALUES(?1,'work','succeeded','success','3','4','4',NULL,0)", [format!("terminal-{index:04}")])?;
    }
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    project_events(root.path(), &conn, &mut writer)?;
    project_runs(root.path(), &conn, &mut writer)?;
    let business = store::open_store(root.path())?;
    let events: i64 = business.query_row("SELECT COUNT(*) FROM business_records WHERE collection='ctox_harness_events' AND deleted=0", [], |r| r.get(0))?;
    let runs: i64 = business.query_row(
        "SELECT COUNT(*) FROM business_records WHERE collection='ctox_runs' AND deleted=0",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(events, PROJECTION_PAGE_SIZE + 1);
    assert_eq!(runs, 500 + PROJECTION_PAGE_SIZE + 1);
    assert!(record(root.path(), "ctox_runs", "terminal-0000").is_err());
    assert!(record(root.path(), "ctox_runs", "active-0000").is_ok());
    Ok(())
}

#[test]
fn retention_keeps_active_tasks_and_tombstones_all_three_projection_surfaces() -> Result<()> {
    let (root, conn) = setup()?;
    crate::inference::runtime_env::set_runtime_env_value(
        root.path(),
        super::super::QUEUE_RETENTION_KEY,
        "3",
    )?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    for index in 0..7 {
        let id = format!("task-{index}");
        let status = if index < 5 { "handled" } else { "pending" };
        conn.execute(
            "INSERT INTO communication_routing_state VALUES(?1,?2,?3)",
            params![id, status, format!("2020-01-0{}T00:00:00Z", index + 1)],
        )?;
        writer.upsert_source_projection(
            "ctox_queue_tasks",
            &id,
            index,
            json!({"id":id,"route_status":status}),
        )?;
    }
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('recent','handled',?1)",
        [Utc::now().to_rfc3339()],
    )?;
    writer.upsert_source_projection(
        "ctox_queue_tasks",
        "recent",
        0,
        json!({"id":"recent","route_status":"handled"}),
    )?;
    for index in 0..210 {
        let id = format!("event-{index:03}");
        writer.upsert_source_projection(
            "ctox_harness_events",
            &id,
            index,
            json!({"id":id,"task_id":"task-5","created_at_ms":index}),
        )?;
    }
    for (id, task) in [("expired", "task-0"), ("recent-event", "recent")] {
        writer.upsert_source_projection(
            "ctox_harness_events",
            id,
            0,
            json!({"id":id,"task_id":task,"created_at_ms":0}),
        )?;
    }
    for index in 0..503 {
        let id = format!("run-{index:03}");
        let task = if index == 0 { "task-5" } else { "task-0" };
        writer.upsert_source_projection(
            "ctox_runs",
            &id,
            index,
            json!({"id":id,"task_id":task,"finished_at_ms":index}),
        )?;
    }
    retain(
        root.path(),
        &conn,
        &mut writer,
        Utc::now().timestamp_millis(),
    )?;
    let business = store::open_store(root.path())?;
    for (collection, expected) in [
        ("ctox_queue_tasks", 5),
        ("ctox_harness_events", 201),
        ("ctox_runs", 501),
    ] {
        let actual: i64 = business.query_row(
            "SELECT COUNT(*) FROM business_records WHERE collection=?1 AND deleted=0",
            [collection],
            |r| r.get(0),
        )?;
        assert_eq!(actual, expected, "{collection}");
    }
    assert_eq!(
        record(root.path(), "ctox_harness_events", "expired")?["_deleted"],
        true
    );
    assert_eq!(
        record(root.path(), "ctox_runs", "run-000")?["_deleted"],
        false
    );
    let rxdb = Connection::open(store::rxdb_store_path(root.path()))?;
    let deleted: bool = rxdb.query_row(
        "SELECT deleted FROM ctox_business_os__ctox_harness_events__v0 WHERE id='expired'",
        [],
        |r| r.get(0),
    )?;
    assert!(deleted, "retention tombstones replicate");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM communication_routing_state",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        8,
        "durable history is untouched"
    );
    Ok(())
}

#[test]
fn events_are_bounded_redacted_and_idempotent_and_terminal_tasks_do_not_grow() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('active','leased','2026-01-01T00:00:00Z')",
        [],
    )?;
    for index in 0..210 {
        conn.execute("INSERT INTO ctox_harness_flow_events VALUES(?1,'worker.tool_started','Terminal','private tool output','active',NULL,2,?2,?3)",params![format!("event-{index:03}"),json!({"attempt_id":"attempt","attempt":2,"command_id":"command","tool":{"name":"Terminal","type":"exec_command","call_id":format!("call-{index}"),"arguments":{"secret":"must not replicate"}},"step_position":1}).to_string(),format!("2026-01-01T00:{:02}:{:02}Z",index/60,index%60)])?;
    }
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    project_events(root.path(), &conn, &mut writer)?;
    let before = record(root.path(), "ctox_harness_events", "event-209")?;
    assert_eq!(before["kind"], "tool_started");
    assert_eq!(before["attempt"], 2);
    assert_eq!(before["step_position"], 1);
    assert!(!before.to_string().contains("must not replicate"));
    assert!(!before.to_string().contains("private tool output"));
    project_events(root.path(), &conn, &mut writer)?;
    assert_eq!(
        before,
        record(root.path(), "ctox_harness_events", "event-209")?,
        "unchanged replay creates no new revision"
    );
    assert_eq!(store::open_store(root.path())?.query_row("SELECT COUNT(*) FROM business_records WHERE collection='ctox_harness_events' AND deleted=0",[],|r|r.get::<_,i64>(0))?,200);
    conn.execute(
        "UPDATE communication_routing_state SET route_status='handled'",
        [],
    )?;
    conn.execute("INSERT INTO ctox_harness_flow_events SELECT 'after-terminal',event_kind,title,body_text,message_key,work_id,attempt_index,json_set(metadata_json,'$.cockpit_eligible',json('false')),'2026-01-02T00:00:00Z' FROM ctox_harness_flow_events LIMIT 1",[])?;
    project_events(root.path(), &conn, &mut writer)?;
    assert!(record(root.path(), "ctox_harness_events", "after-terminal").is_err());
    conn.execute(
        "UPDATE communication_routing_state SET route_status='pending'",
        [],
    )?;
    project_events(root.path(), &conn, &mut writer)?;
    assert!(
        record(root.path(), "ctox_harness_events", "after-terminal").is_err(),
        "retry must not publish events emitted after terminalization"
    );
    Ok(())
}

#[test]
fn short_completed_turn_replays_only_events_eligible_at_emission() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('quick','handled',?1)",
        [Utc::now().to_rfc3339()],
    )?;
    for (id, eligible) in [("before-finish", true), ("after-finish", false)] {
        conn.execute("INSERT INTO ctox_harness_flow_events VALUES(?1,'worker.turn_completed','Done','','quick',NULL,1,?2,?3)",params![id,json!({"cockpit_eligible":eligible}).to_string(),Utc::now().to_rfc3339()])?;
    }
    project_events(
        root.path(),
        &conn,
        &mut BusinessProjectionWriter::open(root.path())?,
    )?;
    assert!(record(root.path(), "ctox_harness_events", "before-finish").is_ok());
    assert!(record(root.path(), "ctox_harness_events", "after-finish").is_err());
    Ok(())
}

#[test]
fn runs_join_real_turn_ids_and_refresh_late_costs_without_double_counting() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('task','handled','2026-01-01T00:00:03Z')",
        [],
    )?;
    conn.execute("INSERT INTO worker_attempt_finalizations VALUES('attempt','work','succeeded','success','1767225602000','1767225603000','1767225603000',NULL,0)",[])?;
    conn.execute("INSERT INTO ctox_harness_flow_events VALUES('link','worker.tool_started','Terminal','','task',NULL,1,?1,'2026-01-01T00:00:00Z')",[json!({"attempt_id":"attempt","command_id":"command","turn_id":"actual-turn","tool":{"call_id":"call-1"}}).to_string()])?;
    conn.execute("INSERT INTO api_model_cost_events VALUES('provider','model','actual-turn',100,20,10,4,'2026-01-01')",[])?;
    conn.execute("INSERT INTO api_model_cost_events VALUES('provider','model','attempt',9999,0,9999,0,'2026-01-01')",[])?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    project_runs(root.path(), &conn, &mut writer)?;
    let first = record(root.path(), "ctox_runs", "attempt")?;
    assert_eq!(first["metrics"]["input_tokens"], 100);
    assert_eq!(first["metrics"]["elapsed_ms"], 3000);
    assert_eq!(first["metrics"]["tool_calls"], 1);
    assert!(
        first["metrics"]["cost_usd"].is_null(),
        "unpriced usage is unknown, not free"
    );
    conn.execute(
        "INSERT INTO api_model_price_rates VALUES('provider','model',2,1,4,'2025-01-01')",
        [],
    )?;
    conn.execute("INSERT INTO api_model_cost_events VALUES('provider','model','actual-turn',50,0,5,2,'2026-01-01')",[])?;
    project_runs(root.path(), &conn, &mut writer)?;
    let late = record(root.path(), "ctox_runs", "attempt")?;
    assert_eq!(late["metrics"]["input_tokens"], 150);
    assert_eq!(late["metrics"]["reasoning_tokens"], 6);
    assert!((late["metrics"]["cost_usd"].as_f64().unwrap() - 0.00034).abs() < 1e-10);
    project_runs(root.path(), &conn, &mut writer)?;
    assert_eq!(late, record(root.path(), "ctox_runs", "attempt")?);
    Ok(())
}

#[test]
fn projection_delivery_recovers_when_rxdb_collection_appears_after_writer_open() -> Result<()> {
    let (root, _) = setup()?;
    let rxdb = Connection::open(store::rxdb_store_path(root.path()))?;
    rxdb.execute_batch("DROP TABLE ctox_business_os__ctox_harness_events__v0;")?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    let payload =
        json!({"id":"late","kind":"phase","task_id":"task","created_at_ms":1,"updated_at_ms":1});
    writer.upsert_source_projection("ctox_harness_events", "late", 1, payload.clone())?;
    assert!(writer.payloads.is_empty());
    rxdb.execute_batch("CREATE TABLE ctox_business_os__ctox_harness_events__v0(id TEXT PRIMARY KEY,revision TEXT,deleted INTEGER DEFAULT 0,lastWriteTime REAL DEFAULT 0,data TEXT NOT NULL);")?;
    writer.upsert_source_projection("ctox_harness_events", "late", 1, payload)?;
    assert_eq!(
        rxdb.query_row(
            "SELECT COUNT(*) FROM ctox_business_os__ctox_harness_events__v0 WHERE id='late'",
            [],
            |r| r.get::<_, i64>(0)
        )?,
        1
    );
    Ok(())
}

#[test]
fn worker_snapshot_hooks_publish_start_and_graceful_stop() -> Result<()> {
    let (root, _) = setup()?;
    publish_worker_snapshot(
        root.path(),
        WorkerSnapshot {
            service_running: true,
            busy: true,
            worker_active_count: 1,
            active_task_ids: vec!["hook-task".into()],
            boot_id: "hook-boot".into(),
            ..Default::default()
        },
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if record(root.path(), "ctox_harness_status", "harness")
            .ok()
            .is_some_and(|r| r["busy"] == true && r["active_task_ids"] == json!(["hook-task"]))
        {
            break;
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "worker start snapshot was not projected"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    publish_service_stopped(root.path(), "hook-boot".into());
    assert_eq!(
        record(root.path(), "ctox_harness_status", "harness")?["service_running"],
        false
    );
    Ok(())
}

#[test]
fn status_tracks_start_stop_and_persistent_pause_with_no_change_churn() -> Result<()> {
    let (root, conn) = setup()?;
    conn.execute(
        "INSERT INTO communication_routing_state VALUES('task','leased',?1)",
        [Utc::now().to_rfc3339()],
    )?;
    let mut writer = BusinessProjectionWriter::open(root.path())?;
    let active = WorkerSnapshot {
        service_running: true,
        busy: true,
        worker_active_count: 1,
        worker_phase: Some("model".into()),
        active_task_ids: vec!["task".into()],
        boot_id: "boot".into(),
        ..Default::default()
    };
    project_status(root.path(), &conn, &mut writer, &active)?;
    let before = record(root.path(), "ctox_harness_status", "harness")?;
    project_status(root.path(), &conn, &mut writer, &active)?;
    assert_eq!(
        before,
        record(root.path(), "ctox_harness_status", "harness")?
    );
    crate::inference::runtime_env::set_runtime_env_value(
        root.path(),
        super::super::PAUSE_KEY,
        r#"{"paused":true,"reason":"operator"}"#,
    )?;
    project_status(
        root.path(),
        &conn,
        &mut writer,
        &WorkerSnapshot {
            boot_id: "boot".into(),
            ..Default::default()
        },
    )?;
    let stopped = record(root.path(), "ctox_harness_status", "harness")?;
    assert_eq!(stopped["worker_active_count"], 0);
    assert_eq!(stopped["service_running"], false);
    assert_eq!(stopped["paused"], true);
    assert_eq!(stopped["pause_reason"], "operator");
    assert!(stopped["active_task_ids"].as_array().unwrap().is_empty());
    assert!(stopped["active_crew_member_id"].is_null());
    Ok(())
}
