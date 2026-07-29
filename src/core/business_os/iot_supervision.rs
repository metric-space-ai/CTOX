// Origin: CTOX
// License: Apache-2.0

use super::store;
use anyhow::Context;
use serde_json::json;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug)]
struct IotAgentBootRow {
    id: String,
    realm: String,
    kind: String,
    data: Value,
}

fn load_enabled_iot_agents(root: &Path) -> anyhow::Result<Vec<IotAgentBootRow>> {
    let conn = crate::iot::store::open_iot_store(root)?;
    crate::iot::commands::ensure_stub_schema(&conn)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, realm, kind, data
             FROM iot_agents
             WHERE enabled != 0
             ORDER BY id ASC",
        )
        .context("prepare enabled iot agent query")?;
    let rows = stmt
        .query_map([], |row| {
            let data_raw: String = row.get(3)?;
            let data = serde_json::from_str::<Value>(&data_raw).unwrap_or(Value::Null);
            Ok(IotAgentBootRow {
                id: row.get(0)?,
                realm: row.get(1)?,
                kind: row.get(2)?,
                data,
            })
        })
        .context("query enabled iot agents")?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("read enabled iot agent")?);
    }
    Ok(out)
}

fn configured_iot_agent_links(
    data: &Value,
) -> anyhow::Result<Vec<crate::iot::adapters::AgentLink>> {
    let Some(raw) = data
        .get("links")
        .or_else(|| data.get("agentLinks"))
        .or_else(|| data.get("linkedAttributes"))
    else {
        return Ok(Vec::new());
    };
    if raw.is_null() {
        return Ok(Vec::new());
    }
    let items = raw
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("iot agent links must be an array"))?;
    let mut links = Vec::with_capacity(items.len());
    for item in items {
        let link: crate::iot::adapters::AgentLink = serde_json::from_value(item.clone())
            .context("parse iot agent link from iot_agents.data.links")?;
        anyhow::ensure!(
            !link.asset_id.trim().is_empty() && !link.attribute_name.trim().is_empty(),
            "iot agent link requires asset_id/assetId and attribute_name/attributeName"
        );
        links.push(link);
    }
    Ok(links)
}

fn record_iot_agent_runtime_status(
    root: &Path,
    agent_id: &str,
    realm: &str,
    link_state: &str,
    last_event_ms: Option<i64>,
    data: Value,
) -> anyhow::Result<()> {
    let conn = crate::iot::store::open_iot_store(root)?;
    crate::iot::commands::ensure_stub_schema(&conn)?;
    let now = crate::iot::now_iso();
    conn.execute(
        "INSERT INTO iot_agent_status
            (id, agent_id, realm, link_state, last_event_ms, error, data, created_at, updated_at)
         VALUES (?1, ?1, ?2, ?3, ?4, NULL, ?5, ?6, ?6)
         ON CONFLICT(id) DO UPDATE SET
            realm = excluded.realm,
            link_state = excluded.link_state,
            last_event_ms = COALESCE(excluded.last_event_ms, iot_agent_status.last_event_ms),
            error = NULL,
            data = excluded.data,
            updated_at = excluded.updated_at",
        rusqlite::params![
            agent_id,
            realm,
            link_state,
            last_event_ms,
            serde_json::to_string(&data)?,
            now
        ],
    )
    .context("upsert iot agent runtime status")?;
    let rows = crate::iot::projector::project_agent_status(&conn, agent_id)?;
    store::project_iot_projection_rows(root, rows)?;
    Ok(())
}

fn connection_status_label(status: crate::iot::adapters::ConnectionStatus) -> &'static str {
    match status {
        crate::iot::adapters::ConnectionStatus::Disconnected => "disconnected",
        crate::iot::adapters::ConnectionStatus::Connecting => "connecting",
        crate::iot::adapters::ConnectionStatus::Waiting => "waiting",
        crate::iot::adapters::ConnectionStatus::Connected => "connected",
        crate::iot::adapters::ConnectionStatus::Disconnecting => "disconnecting",
    }
}

pub(crate) fn spawn_iot_agent_supervisors(root: PathBuf) -> Vec<Arc<AtomicBool>> {
    let agents = match load_enabled_iot_agents(&root) {
        Ok(agents) => agents,
        Err(err) => {
            eprintln!("[business-os] IoT agent supervisor bootstrap skipped: {err:#}");
            return Vec::new();
        }
    };
    let mut stops = Vec::new();
    for agent_row in agents {
        let links = match configured_iot_agent_links(&agent_row.data) {
            Ok(links) => links,
            Err(err) => {
                let _ = record_iot_agent_runtime_status(
                    &root,
                    &agent_row.id,
                    &agent_row.realm,
                    "misconfigured",
                    None,
                    json!({"runtime": "supervisor", "reason": "invalid-links"}),
                );
                eprintln!(
                    "[business-os] IoT agent `{}` not started: invalid link configuration ({err:#})",
                    agent_row.id
                );
                continue;
            }
        };
        if links.is_empty() {
            let _ = record_iot_agent_runtime_status(
                &root,
                &agent_row.id,
                &agent_row.realm,
                "unconfigured",
                None,
                json!({"runtime": "supervisor", "reason": "no-links"}),
            );
            continue;
        }
        let Some(kind) = crate::iot::adapters::IotAgentKind::from_str(&agent_row.kind) else {
            let _ = record_iot_agent_runtime_status(
                &root,
                &agent_row.id,
                &agent_row.realm,
                "misconfigured",
                None,
                json!({"runtime": "supervisor", "reason": "unknown-kind"}),
            );
            eprintln!(
                "[business-os] IoT agent `{}` not started: unknown kind `{}`",
                agent_row.id, agent_row.kind
            );
            continue;
        };
        let ctx = crate::iot::adapters::AgentContext {
            root: &root,
            agent_id: agent_row.id.clone(),
            realm: agent_row.realm.clone(),
            config: agent_row.data.clone(),
        };
        let agent = match crate::iot::gateway::build_agent(kind, ctx) {
            Ok(agent) => agent,
            Err(err) => {
                let _ = record_iot_agent_runtime_status(
                    &root,
                    &agent_row.id,
                    &agent_row.realm,
                    "misconfigured",
                    None,
                    json!({"runtime": "supervisor", "reason": "agent-build-failed"}),
                );
                eprintln!(
                    "[business-os] IoT agent `{}` not started: failed to construct native agent ({err:#})",
                    agent_row.id
                );
                continue;
            }
        };
        let mut runtime = crate::iot::runtime::AgentRuntime::new(agent, agent_row.realm.clone());
        let mut link_failed = false;
        for link in links {
            if let Err(err) = runtime.link(link) {
                link_failed = true;
                let _ = record_iot_agent_runtime_status(
                    &root,
                    &agent_row.id,
                    &agent_row.realm,
                    "misconfigured",
                    None,
                    json!({"runtime": "supervisor", "reason": "link-failed"}),
                );
                eprintln!(
                    "[business-os] IoT agent `{}` not started: failed to link attribute ({err:#})",
                    agent_row.id
                );
                break;
            }
        }
        if link_failed {
            continue;
        }
        let _ = record_iot_agent_runtime_status(
            &root,
            &agent_row.id,
            &agent_row.realm,
            "starting",
            None,
            json!({"runtime": "supervisor", "kind": agent_row.kind}),
        );
        let projection_root = root.clone();
        let status_root = root.clone();
        let status_agent_id = agent_row.id.clone();
        let status_realm = agent_row.realm.clone();
        let status_kind = agent_row.kind.clone();
        let stop = crate::iot::runtime::spawn_supervisor(
            runtime,
            root.clone(),
            agent_row.id.clone(),
            agent_row.data.clone(),
            move |rows, status| {
                if !rows.is_empty() {
                    store::project_iot_projection_rows(&projection_root, rows)?;
                }
                record_iot_agent_runtime_status(
                    &status_root,
                    &status_agent_id,
                    &status_realm,
                    connection_status_label(status),
                    Some(crate::iot::now_ms()),
                    json!({"runtime": "supervisor", "kind": status_kind.clone()}),
                )?;
                Ok(())
            },
        );
        eprintln!(
            "[business-os] IoT agent supervisor started for `{}` ({})",
            agent_row.id, agent_row.kind
        );
        stops.push(stop);
    }
    stops
}
