// Read-side API for scrape data: registry summary, latest run views and
// record queries. Nothing here writes.

use super::registry::{count_rows, open_db, resolve_db_path, show_api};
use super::{
    build_target_api_contract, load_all_latest_active_records, load_last_successful_run,
    load_target_view, record_matches_filters, resolve_workspace_dir, LatestRecordView,
    RecentRunView,
};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::path::Path;

pub(super) fn summary_payload(root: &Path) -> Result<Value> {
    let conn = open_db(root)?;
    let recent_runs = {
        let mut statement = conn.prepare(
            r#"
            SELECT r.run_id, t.target_key, r.status, r.trigger_kind, r.finished_at
            FROM scrape_run r
            JOIN scrape_target t ON t.target_id = r.target_id
            ORDER BY COALESCE(r.finished_at, r.started_at) DESC
            LIMIT 10
            "#,
        )?;
        let rows = statement.query_map([], |row| {
            Ok(RecentRunView {
                run_id: row.get(0)?,
                target_key: row.get(1)?,
                status: row.get(2)?,
                trigger_kind: row.get(3)?,
                finished_at: row.get(4)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    Ok(json!({
        "ok": true,
        "targets_total": count_rows(&conn, "scrape_target")?,
        "targets_active": count_filtered_rows(&conn, "scrape_target", "status = 'active'")?,
        "script_revisions_total": count_rows(&conn, "scrape_script_revision")?,
        "source_revisions_total": count_rows(&conn, "scrape_source_revision")?,
        "template_examples_total": count_rows(&conn, "scrape_template_example")?,
        "templates_promoted_total": count_filtered_rows(&conn, "scrape_template_promoted", "is_active = 1")?,
        "runs_total": count_rows(&conn, "scrape_run")?,
        "materialized_active_records_total": count_filtered_rows(&conn, "scrape_record_latest", "deleted_at IS NULL")?,
        "recent_runs": recent_runs,
    }))
}

pub(crate) fn show_latest(root: &Path, target_key: &str, limit: usize) -> Result<Option<Value>> {
    let conn = open_db(root)?;
    let Some(target) = load_target_view(&conn, target_key)? else {
        return Ok(None);
    };
    let limit = limit.max(1) as i64;
    let active_count = conn.query_row(
        "SELECT COUNT(*) FROM scrape_record_latest WHERE target_id = ?1 AND deleted_at IS NULL",
        params![target.target_id],
        |row| row.get::<_, i64>(0),
    )?;
    let deleted_count = conn.query_row(
        "SELECT COUNT(*) FROM scrape_record_latest WHERE target_id = ?1 AND deleted_at IS NOT NULL",
        params![target.target_id],
        |row| row.get::<_, i64>(0),
    )?;
    let latest_records = {
        let mut statement = conn.prepare(
            r#"
            SELECT record_key, last_seen_at, record_json
            FROM scrape_record_latest
            WHERE target_id = ?1 AND deleted_at IS NULL
            ORDER BY last_seen_at DESC, record_key ASC
            LIMIT ?2
            "#,
        )?;
        let rows = statement.query_map(params![target.target_id, limit], |row| {
            let record_json: String = row.get(2)?;
            Ok(LatestRecordView {
                record_key: row.get(0)?,
                last_seen_at: row.get(1)?,
                record: serde_json::from_str(&record_json).unwrap_or_else(|_| json!({})),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let last_successful_run = load_last_successful_run(&conn, &target.target_id)?;
    let state_dir = resolve_workspace_dir(root, &target.workspace_dir).join("state");
    Ok(Some(json!({
        "target_key": target.target_key,
        "workspace_dir": target.workspace_dir,
        "active_record_count": active_count,
        "deleted_record_count": deleted_count,
        "state_paths": {
            "latest_records": state_dir.join("latest_records.json"),
            "latest_summary": state_dir.join("latest_summary.json"),
        },
        "last_successful_run": last_successful_run,
        "records": latest_records,
    })))
}

pub(super) fn query_records(
    root: &Path,
    target_key: &str,
    filters: &[(String, String)],
    limit: usize,
) -> Result<Option<Value>> {
    let conn = open_db(root)?;
    let Some(target) = load_target_view(&conn, target_key)? else {
        return Ok(None);
    };
    let items = load_all_latest_active_records(&conn, &target.target_id)?;
    let filtered = items
        .into_iter()
        .filter(|item| record_matches_filters(&item.record, filters))
        .take(limit.max(1))
        .map(|item| {
            json!({
                "record_key": item.record_key,
                "last_seen_at": item.last_seen_at,
                "record": item.record,
            })
        })
        .collect::<Vec<_>>();
    Ok(Some(json!({
        "target_key": target.target_key,
        "filters": filters.iter().map(|(field, value)| json!({"field": field, "value": value})).collect::<Vec<_>>(),
        "limit": limit.max(1),
        "count": filtered.len(),
        "items": filtered,
        "api": build_target_api_contract(root, &target),
    })))
}

pub(crate) fn service_show_api(root: &Path, target_key: &str) -> Result<Option<Value>> {
    show_api(root, target_key)
}

pub(crate) fn service_query_records(
    root: &Path,
    target_key: &str,
    filters: &[(String, String)],
    limit: usize,
) -> Result<Option<Value>> {
    query_records(root, target_key, filters, limit)
}

pub(super) fn count_filtered_rows(conn: &Connection, table: &str, condition: &str) -> Result<i64> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {condition}");
    conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
        .map_err(anyhow::Error::from)
}
