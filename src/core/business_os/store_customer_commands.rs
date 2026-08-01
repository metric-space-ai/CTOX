// Origin: CTOX
// License: Apache-2.0

use super::store::{
    now_ms, open_store, outbound_first_string, outbound_id_from_command, outbound_load_record,
    outbound_load_required, outbound_merge_fields, outbound_object_payload,
    outbound_put_default_i64, outbound_put_default_object, outbound_put_default_string,
    outbound_put_i64, outbound_put_string, outbound_required_from_payload_or_record,
    outbound_required_string, outbound_session_actor_id, outbound_string, BusinessCommand,
    BusinessOsSession, CommandOrigin,
};
use super::store_projections::upsert_business_record;
use crate::mission::channels;
use anyhow::Context;
use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;
use url::Url;

pub(super) fn handle_customers_active_command(
    root: &Path,
    session: &BusinessOsSession,
    command: &BusinessCommand,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        command.module == "customers",
        "active customers commands require module=customers"
    );
    let now = now_ms() as i64;
    let conn = open_store(root)?;
    match command.command_type.as_str() {
        "customers.account.create" => customers_account_create(&conn, session, command, now),
        "customers.account.update" => customers_account_update(&conn, session, command, now),
        "customers.account.archive" => customers_account_archive(&conn, session, command, now),
        "customers.contact.create" => customers_contact_create(&conn, session, command, now),
        "customers.contact.update" => customers_contact_update(&conn, session, command, now),
        "customers.contact.archive" => customers_contact_archive(&conn, session, command, now),
        "customers.opportunity.create" => {
            customers_opportunity_create(&conn, session, command, now)
        }
        "customers.opportunity.update" => {
            customers_opportunity_update(&conn, session, command, now)
        }
        "customers.opportunity.move_stage" => {
            customers_opportunity_move_stage(&conn, session, command, now)
        }
        "customers.opportunity.close_won" => {
            customers_opportunity_close(&conn, session, command, now, "closed_won")
        }
        "customers.opportunity.close_lost" => {
            customers_opportunity_close(&conn, session, command, now, "closed_lost")
        }
        "customers.task.create" => customers_task_create(&conn, session, command, now),
        "customers.task.update" => customers_task_update(&conn, session, command, now),
        "customers.task.complete" => customers_task_complete(&conn, session, command, now),
        "customers.note.create" => customers_note_create(&conn, session, command, now),
        "customers.note.update" => customers_note_update(&conn, session, command, now),
        "customers.activity.record" => customers_activity_record(&conn, session, command, now),
        "customers.view.save" => customers_view_save(&conn, session, command, now),
        "customers.import.from_outbound" => {
            customers_import_from_outbound(&conn, session, command, now)
        }
        "customers.dedupe.resolve" => customers_dedupe_resolve(&conn, session, command, now),
        other => anyhow::bail!("unsupported customers command type: {other}"),
    }
}

fn customers_account_create(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let account_id = customers_id_from_command(command, &["account_id", "id"], "acct")?;
    anyhow::ensure!(
        outbound_load_record(conn, "customer_accounts", &account_id)?.is_none(),
        "customer account already exists"
    );
    let mut account = outbound_object_payload(&command.payload);
    customers_put_string(&mut account, "id", account_id.clone());
    customers_require_field(&account, &["name"])?;
    customers_put_default_string(&mut account, "account_status", "active");
    customers_put_default_string(&mut account, "customer_stage", "active");
    customers_put_default_string(&mut account, "health_status", "unknown");
    customers_put_default_string(&mut account, "currency", "EUR");
    customers_put_default_bool(&mut account, "ideal_customer_profile", false);
    customers_put_default_bool(&mut account, "is_deleted", false);
    customers_put_default_object(&mut account, "payload");
    customers_put_default_i64(&mut account, "created_at_ms", now);
    customers_validate_account(&account)?;
    customers_refresh_search_text(&mut account, &["name", "domain", "industry"]);
    upsert_business_record(conn, "customer_accounts", &account_id, now, account.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "account_created",
            name: "Kunde erstellt",
            account_id: Some(account_id.clone()),
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("account"),
            linked_record_id: Some(account_id.clone()),
            linked_record_name: outbound_string(&account, &["name"]),
            properties: serde_json::json!({ "account": account.clone() }),
        },
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": "customer_accounts",
        "account": account,
        "activity": activity,
    }))
}

fn customers_account_update(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let account_id = customers_required_from_payload_or_record(
        command,
        &["account_id", "id"],
        "account_id is required",
    )?;
    let mut account = outbound_load_required(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    customers_merge_fields(
        &mut account,
        &command.payload,
        &[
            "name",
            "domain",
            "website_url",
            "linkedin_url",
            "x_url",
            "account_status",
            "customer_stage",
            "account_owner_id",
            "annual_recurring_revenue_cents",
            "currency",
            "employee_count",
            "industry",
            "address",
            "ideal_customer_profile",
            "source",
            "source_record_id",
            "last_activity_at_ms",
            "next_action_at_ms",
            "health_status",
            "payload",
        ],
    );
    customers_validate_account(&account)?;
    customers_refresh_search_text(&mut account, &["name", "domain", "industry"]);
    upsert_business_record(conn, "customer_accounts", &account_id, now, account.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "account_updated",
            name: "Kunde aktualisiert",
            account_id: Some(account_id.clone()),
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("account"),
            linked_record_id: Some(account_id.clone()),
            linked_record_name: outbound_string(&account, &["name"]),
            properties: serde_json::json!({ "patch": command.payload.clone() }),
        },
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": "customer_accounts",
        "account": account,
        "activity": activity,
    }))
}

fn customers_account_archive(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let account_id = customers_required_from_payload_or_record(
        command,
        &["account_id", "id"],
        "account_id is required",
    )?;
    let mut account = outbound_load_required(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    customers_put_string(&mut account, "account_status", "archived");
    customers_put_string(&mut account, "customer_stage", "archived");
    customers_put_bool(&mut account, "is_deleted", true);
    customers_put_i64(&mut account, "deleted_at_ms", now);
    customers_refresh_search_text(&mut account, &["name", "domain", "industry"]);
    upsert_business_record(conn, "customer_accounts", &account_id, now, account.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "account_archived",
            name: "Kunde archiviert",
            account_id: Some(account_id.clone()),
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("account"),
            linked_record_id: Some(account_id.clone()),
            linked_record_name: outbound_string(&account, &["name"]),
            properties: serde_json::json!({ "account_id": account_id }),
        },
    )?;
    Ok(serde_json::json!({ "ok": true, "account": account, "activity": activity }))
}

fn customers_contact_create(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let contact_id = customers_id_from_command(command, &["contact_id", "id"], "contact")?;
    anyhow::ensure!(
        outbound_load_record(conn, "customer_contacts", &contact_id)?.is_none(),
        "customer contact already exists"
    );
    let account_id = outbound_required_string(&command.payload, &["account_id"])?;
    customers_require_existing(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    let mut contact = outbound_object_payload(&command.payload);
    customers_put_string(&mut contact, "id", contact_id.clone());
    customers_put_default_string(&mut contact, "first_name", "");
    customers_put_default_string(&mut contact, "last_name", "");
    customers_put_default_string(&mut contact, "email", "");
    let has_name = outbound_string(&contact, &["first_name"]).is_some()
        || outbound_string(&contact, &["last_name"]).is_some();
    let has_email = outbound_string(&contact, &["email"]).is_some();
    anyhow::ensure!(has_name || has_email, "contact name or email is required");
    customers_put_default_bool(&mut contact, "is_primary_contact", false);
    customers_put_default_bool(&mut contact, "is_deleted", false);
    customers_put_default_object(&mut contact, "payload");
    customers_put_default_i64(&mut contact, "created_at_ms", now);
    customers_refresh_search_text(
        &mut contact,
        &["first_name", "last_name", "email", "job_title", "city"],
    );
    upsert_business_record(conn, "customer_contacts", &contact_id, now, contact.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "contact_created",
            name: "Kontakt erstellt",
            account_id: Some(account_id.clone()),
            contact_id: Some(contact_id.clone()),
            opportunity_id: None,
            linked_record_type: Some("contact"),
            linked_record_id: Some(contact_id.clone()),
            linked_record_name: Some(customers_contact_display_name(&contact)),
            properties: serde_json::json!({ "contact": contact.clone() }),
        },
    )?;
    customers_touch_account_last_activity(conn, &account_id, now)?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": "customer_contacts",
        "contact": contact,
        "activity": activity,
    }))
}

fn customers_contact_update(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let contact_id = customers_required_from_payload_or_record(
        command,
        &["contact_id", "id"],
        "contact_id is required",
    )?;
    let mut contact = outbound_load_required(
        conn,
        "customer_contacts",
        &contact_id,
        "customer contact not found",
    )?;
    customers_merge_fields(
        &mut contact,
        &command.payload,
        &[
            "account_id",
            "first_name",
            "last_name",
            "email",
            "phone",
            "job_title",
            "city",
            "linkedin_url",
            "x_url",
            "is_primary_contact",
            "contact_owner_id",
            "last_activity_at_ms",
            "source",
            "source_record_id",
            "payload",
        ],
    );
    let account_id = outbound_required_string(&contact, &["account_id"])?;
    customers_require_existing(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    customers_refresh_search_text(
        &mut contact,
        &["first_name", "last_name", "email", "job_title", "city"],
    );
    upsert_business_record(conn, "customer_contacts", &contact_id, now, contact.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "contact_updated",
            name: "Kontakt aktualisiert",
            account_id: Some(account_id.clone()),
            contact_id: Some(contact_id.clone()),
            opportunity_id: None,
            linked_record_type: Some("contact"),
            linked_record_id: Some(contact_id.clone()),
            linked_record_name: Some(customers_contact_display_name(&contact)),
            properties: serde_json::json!({ "patch": command.payload.clone() }),
        },
    )?;
    customers_touch_account_last_activity(conn, &account_id, now)?;
    Ok(serde_json::json!({ "ok": true, "contact": contact, "activity": activity }))
}

fn customers_contact_archive(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let contact_id = customers_required_from_payload_or_record(
        command,
        &["contact_id", "id"],
        "contact_id is required",
    )?;
    let mut contact = outbound_load_required(
        conn,
        "customer_contacts",
        &contact_id,
        "customer contact not found",
    )?;
    let account_id = outbound_string(&contact, &["account_id"]);
    customers_put_bool(&mut contact, "is_deleted", true);
    customers_put_i64(&mut contact, "deleted_at_ms", now);
    upsert_business_record(conn, "customer_contacts", &contact_id, now, contact.clone())?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "contact_archived",
            name: "Kontakt archiviert",
            account_id: account_id.clone(),
            contact_id: Some(contact_id.clone()),
            opportunity_id: None,
            linked_record_type: Some("contact"),
            linked_record_id: Some(contact_id.clone()),
            linked_record_name: Some(customers_contact_display_name(&contact)),
            properties: serde_json::json!({ "contact_id": contact_id }),
        },
    )?;
    if let Some(account_id) = account_id {
        customers_touch_account_last_activity(conn, &account_id, now)?;
    }
    Ok(serde_json::json!({ "ok": true, "contact": contact, "activity": activity }))
}

fn customers_opportunity_create(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let opportunity_id = customers_id_from_command(command, &["opportunity_id", "id"], "opp")?;
    anyhow::ensure!(
        outbound_load_record(conn, "customer_opportunities", &opportunity_id)?.is_none(),
        "customer opportunity already exists"
    );
    let account_id = outbound_required_string(&command.payload, &["account_id"])?;
    customers_require_existing(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    if let Some(contact_id) = outbound_string(&command.payload, &["primary_contact_id"]) {
        customers_require_existing(
            conn,
            "customer_contacts",
            &contact_id,
            "primary contact not found",
        )?;
    }
    let mut opportunity = outbound_object_payload(&command.payload);
    customers_put_string(&mut opportunity, "id", opportunity_id.clone());
    customers_require_field(&opportunity, &["name"])?;
    customers_put_default_string(&mut opportunity, "opportunity_type", "new_business");
    customers_put_default_string(&mut opportunity, "stage", "qualification");
    customers_put_default_i64(&mut opportunity, "amount_cents", 0);
    customers_put_default_string(&mut opportunity, "currency", "EUR");
    customers_put_default_i64(&mut opportunity, "position", now);
    customers_put_default_bool(&mut opportunity, "is_deleted", false);
    customers_put_default_i64(&mut opportunity, "created_at_ms", now);
    customers_put_default_i64(&mut opportunity, "last_stage_changed_at_ms", now);
    customers_put_default_object(&mut opportunity, "payload");
    customers_validate_opportunity(&opportunity)?;
    customers_refresh_search_text(&mut opportunity, &["name", "stage", "opportunity_type"]);
    upsert_business_record(
        conn,
        "customer_opportunities",
        &opportunity_id,
        now,
        opportunity.clone(),
    )?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "opportunity_created",
            name: "Opportunity erstellt",
            account_id: Some(account_id.clone()),
            contact_id: outbound_string(&opportunity, &["primary_contact_id"]),
            opportunity_id: Some(opportunity_id.clone()),
            linked_record_type: Some("opportunity"),
            linked_record_id: Some(opportunity_id.clone()),
            linked_record_name: outbound_string(&opportunity, &["name"]),
            properties: serde_json::json!({ "opportunity": opportunity.clone() }),
        },
    )?;
    customers_touch_account_last_activity(conn, &account_id, now)?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": "customer_opportunities",
        "opportunity": opportunity,
        "activity": activity,
    }))
}

fn customers_opportunity_update(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let opportunity_id = customers_required_from_payload_or_record(
        command,
        &["opportunity_id", "id"],
        "opportunity_id is required",
    )?;
    let mut opportunity = outbound_load_required(
        conn,
        "customer_opportunities",
        &opportunity_id,
        "customer opportunity not found",
    )?;
    customers_merge_fields(
        &mut opportunity,
        &command.payload,
        &[
            "name",
            "account_id",
            "primary_contact_id",
            "owner_id",
            "opportunity_type",
            "amount_cents",
            "currency",
            "close_date_ms",
            "probability",
            "position",
            "source",
            "source_record_id",
            "payload",
        ],
    );
    customers_validate_opportunity(&opportunity)?;
    let account_id = outbound_required_string(&opportunity, &["account_id"])?;
    customers_require_existing(
        conn,
        "customer_accounts",
        &account_id,
        "customer account not found",
    )?;
    customers_refresh_search_text(&mut opportunity, &["name", "stage", "opportunity_type"]);
    upsert_business_record(
        conn,
        "customer_opportunities",
        &opportunity_id,
        now,
        opportunity.clone(),
    )?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "opportunity_updated",
            name: "Opportunity aktualisiert",
            account_id: Some(account_id.clone()),
            contact_id: outbound_string(&opportunity, &["primary_contact_id"]),
            opportunity_id: Some(opportunity_id.clone()),
            linked_record_type: Some("opportunity"),
            linked_record_id: Some(opportunity_id.clone()),
            linked_record_name: outbound_string(&opportunity, &["name"]),
            properties: serde_json::json!({ "patch": command.payload.clone() }),
        },
    )?;
    customers_touch_account_last_activity(conn, &account_id, now)?;
    Ok(serde_json::json!({ "ok": true, "opportunity": opportunity, "activity": activity }))
}

fn customers_opportunity_move_stage(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let opportunity_id = customers_required_from_payload_or_record(
        command,
        &["opportunity_id", "id"],
        "opportunity_id is required",
    )?;
    let stage = outbound_required_string(&command.payload, &["stage"])?;
    customers_validate_allowed("opportunity stage", &stage, CUSTOMER_OPPORTUNITY_STAGES)?;
    let mut opportunity = outbound_load_required(
        conn,
        "customer_opportunities",
        &opportunity_id,
        "customer opportunity not found",
    )?;
    let from_stage = outbound_string(&opportunity, &["stage"]).unwrap_or_default();
    customers_validate_stage_transition(&from_stage, &stage)?;
    customers_put_string(&mut opportunity, "stage", stage.clone());
    customers_put_i64(&mut opportunity, "last_stage_changed_at_ms", now);
    if let Some(position) = customers_i64(&command.payload, &["position"]) {
        customers_put_i64(&mut opportunity, "position", position);
    }
    customers_refresh_search_text(&mut opportunity, &["name", "stage", "opportunity_type"]);
    upsert_business_record(
        conn,
        "customer_opportunities",
        &opportunity_id,
        now,
        opportunity.clone(),
    )?;
    let account_id = outbound_required_string(&opportunity, &["account_id"])?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "opportunity_stage_changed",
            name: "Opportunity verschoben",
            account_id: Some(account_id.clone()),
            contact_id: outbound_string(&opportunity, &["primary_contact_id"]),
            opportunity_id: Some(opportunity_id.clone()),
            linked_record_type: Some("opportunity"),
            linked_record_id: Some(opportunity_id.clone()),
            linked_record_name: outbound_string(&opportunity, &["name"]),
            properties: serde_json::json!({ "from_stage": from_stage, "to_stage": stage }),
        },
    )?;
    customers_touch_account_last_activity(conn, &account_id, now)?;
    Ok(serde_json::json!({ "ok": true, "opportunity": opportunity, "activity": activity }))
}

fn customers_opportunity_close(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
    stage: &str,
) -> anyhow::Result<Value> {
    let mut payload = command.payload.clone();
    if let Some(object) = payload.as_object_mut() {
        object.insert("stage".to_string(), Value::String(stage.to_string()));
        if stage == "closed_won" {
            object.insert("probability".to_string(), Value::from(100));
        }
        if stage == "closed_lost" {
            let lost_reason = outbound_required_string(&command.payload, &["lost_reason"])?;
            object.insert("lost_reason".to_string(), Value::String(lost_reason));
            object.insert("probability".to_string(), Value::from(0));
        }
    }
    let command = BusinessCommand {
        origin: CommandOrigin::TrustedLocal,
        payload,
        ..command.clone()
    };
    customers_opportunity_move_stage(conn, session, &command, now)
}

fn customers_task_create(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let task_id = customers_id_from_command(command, &["task_id", "id"], "task")?;
    anyhow::ensure!(
        outbound_load_record(conn, "customer_tasks", &task_id)?.is_none(),
        "customer task already exists"
    );
    let mut task = outbound_object_payload(&command.payload);
    customers_put_string(&mut task, "id", task_id.clone());
    customers_require_field(&task, &["title"])?;
    customers_validate_links(conn, &task, true)?;
    customers_put_default_string(&mut task, "status", "open");
    customers_put_default_i64(&mut task, "position", now);
    customers_put_default_bool(&mut task, "is_deleted", false);
    customers_put_default_i64(&mut task, "created_at_ms", now);
    customers_put_default_object(&mut task, "payload");
    customers_validate_allowed(
        "task status",
        &outbound_required_string(&task, &["status"])?,
        CUSTOMER_TASK_STATUSES,
    )?;
    customers_refresh_search_text(&mut task, &["title", "body", "status"]);
    upsert_business_record(conn, "customer_tasks", &task_id, now, task.clone())?;
    let activity = customers_write_task_activity(
        conn,
        session,
        command,
        now,
        &task,
        "task_created",
        "Aufgabe erstellt",
    )?;
    Ok(serde_json::json!({ "ok": true, "task": task, "activity": activity }))
}

fn customers_task_update(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let task_id = customers_required_from_payload_or_record(
        command,
        &["task_id", "id"],
        "task_id is required",
    )?;
    let mut task =
        outbound_load_required(conn, "customer_tasks", &task_id, "customer task not found")?;
    customers_merge_fields(
        &mut task,
        &command.payload,
        &[
            "title",
            "body",
            "status",
            "due_at_ms",
            "assignee_id",
            "account_id",
            "contact_id",
            "opportunity_id",
            "position",
            "payload",
        ],
    );
    customers_validate_links(conn, &task, true)?;
    customers_validate_allowed(
        "task status",
        &outbound_required_string(&task, &["status"])?,
        CUSTOMER_TASK_STATUSES,
    )?;
    customers_refresh_search_text(&mut task, &["title", "body", "status"]);
    upsert_business_record(conn, "customer_tasks", &task_id, now, task.clone())?;
    let activity = customers_write_task_activity(
        conn,
        session,
        command,
        now,
        &task,
        "task_updated",
        "Aufgabe aktualisiert",
    )?;
    Ok(serde_json::json!({ "ok": true, "task": task, "activity": activity }))
}

fn customers_task_complete(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let task_id = customers_required_from_payload_or_record(
        command,
        &["task_id", "id"],
        "task_id is required",
    )?;
    let mut task =
        outbound_load_required(conn, "customer_tasks", &task_id, "customer task not found")?;
    customers_put_string(&mut task, "status", "completed");
    customers_put_i64(&mut task, "completed_at_ms", now);
    customers_refresh_search_text(&mut task, &["title", "body", "status"]);
    upsert_business_record(conn, "customer_tasks", &task_id, now, task.clone())?;
    let activity = customers_write_task_activity(
        conn,
        session,
        command,
        now,
        &task,
        "task_completed",
        "Aufgabe erledigt",
    )?;
    Ok(serde_json::json!({ "ok": true, "task": task, "activity": activity }))
}

fn customers_note_create(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let note_id = customers_id_from_command(command, &["note_id", "id"], "note")?;
    anyhow::ensure!(
        outbound_load_record(conn, "customer_notes", &note_id)?.is_none(),
        "customer note already exists"
    );
    let mut note = outbound_object_payload(&command.payload);
    customers_put_string(&mut note, "id", note_id.clone());
    customers_validate_links(conn, &note, true)?;
    customers_put_default_string(&mut note, "title", "Notiz");
    customers_put_default_string(&mut note, "body", "");
    customers_put_default_string(&mut note, "body_format", "markdown");
    customers_put_default_bool(&mut note, "is_deleted", false);
    customers_put_default_i64(&mut note, "created_at_ms", now);
    customers_put_default_object(&mut note, "payload");
    customers_refresh_search_text(&mut note, &["title", "body"]);
    upsert_business_record(conn, "customer_notes", &note_id, now, note.clone())?;
    let activity = customers_write_note_activity(
        conn,
        session,
        command,
        now,
        &note,
        "note_created",
        "Notiz erstellt",
    )?;
    Ok(serde_json::json!({ "ok": true, "note": note, "activity": activity }))
}

fn customers_note_update(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let note_id = customers_required_from_payload_or_record(
        command,
        &["note_id", "id"],
        "note_id is required",
    )?;
    let mut note =
        outbound_load_required(conn, "customer_notes", &note_id, "customer note not found")?;
    customers_merge_fields(
        &mut note,
        &command.payload,
        &[
            "title",
            "body",
            "body_format",
            "author_id",
            "account_id",
            "contact_id",
            "opportunity_id",
            "linked_note_id",
            "payload",
        ],
    );
    customers_validate_links(conn, &note, true)?;
    customers_refresh_search_text(&mut note, &["title", "body"]);
    upsert_business_record(conn, "customer_notes", &note_id, now, note.clone())?;
    let activity = customers_write_note_activity(
        conn,
        session,
        command,
        now,
        &note,
        "note_updated",
        "Notiz aktualisiert",
    )?;
    Ok(serde_json::json!({ "ok": true, "note": note, "activity": activity }))
}

fn customers_activity_record(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    customers_validate_links(conn, &command.payload, false)?;
    let activity_type = outbound_required_string(&command.payload, &["activity_type"])?;
    let name = outbound_string(&command.payload, &["name"]).unwrap_or(activity_type.clone());
    let linked_record_type = outbound_string(&command.payload, &["linked_record_type"]);
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: activity_type.as_str(),
            name: name.as_str(),
            account_id: outbound_string(&command.payload, &["account_id"]),
            contact_id: outbound_string(&command.payload, &["contact_id"]),
            opportunity_id: outbound_string(&command.payload, &["opportunity_id"]),
            linked_record_type: linked_record_type.as_deref(),
            linked_record_id: outbound_string(&command.payload, &["linked_record_id"]),
            linked_record_name: outbound_string(&command.payload, &["linked_record_name"]),
            properties: command
                .payload
                .get("properties")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
        },
    )?;
    Ok(serde_json::json!({ "ok": true, "activity": activity }))
}

fn customers_view_save(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let view_id = customers_id_from_command(command, &["view_id", "id"], "view")?;
    let mut view = outbound_object_payload(command.payload.get("view").unwrap_or(&command.payload));
    customers_put_string(&mut view, "id", view_id.clone());
    customers_require_field(&view, &["name"])?;
    let object_type = outbound_required_string(&view, &["object_type"])?;
    customers_validate_allowed(
        "view object_type",
        &object_type,
        &["account", "contact", "opportunity"],
    )?;
    customers_put_default_string(&mut view, "view_type", "table");
    customers_put_default_string(&mut view, "visibility", "private");
    customers_put_default_bool(&mut view, "is_deleted", false);
    customers_put_default_i64(&mut view, "created_at_ms", now);
    customers_put_default_object(&mut view, "payload");
    upsert_business_record(conn, "customer_views", &view_id, now, view.clone())?;
    let filters = customers_save_view_children(
        conn,
        &view_id,
        command.payload.get("filters"),
        "customer_view_filters",
        now,
    )?;
    let sorts = customers_save_view_children(
        conn,
        &view_id,
        command.payload.get("sorts"),
        "customer_view_sorts",
        now,
    )?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "view_saved",
            name: "Ansicht gespeichert",
            account_id: None,
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("view"),
            linked_record_id: Some(view_id.clone()),
            linked_record_name: outbound_string(&view, &["name"]),
            properties: serde_json::json!({ "view": view.clone(), "filters": filters, "sorts": sorts }),
        },
    )?;
    Ok(serde_json::json!({ "ok": true, "view": view, "activity": activity }))
}

fn customers_import_from_outbound(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let source_record_id = outbound_first_string(&[
        outbound_string(&command.payload, &["source_record_id"]),
        outbound_string(&command.payload, &["outbound_company_id"]),
        outbound_string(&command.payload, &["company_id"]),
        command.record_id.clone(),
    ])
    .context("source_record_id or outbound company_id is required")?;
    let company = outbound_load_required(
        conn,
        "outbound_companies",
        &source_record_id,
        "outbound company not found",
    )?;
    let pipeline_id = outbound_string(&command.payload, &["pipeline_id"]).or_else(|| {
        customers_find_outbound_pipeline_for_company(conn, &source_record_id)
            .ok()
            .flatten()
    });
    let pipeline = pipeline_id.as_ref().and_then(|id| {
        outbound_load_record(conn, "outbound_pipeline_items", id)
            .ok()
            .flatten()
    });
    let domain = outbound_string(&company, &["domain"]).or_else(|| {
        outbound_string(&company, &["website"]).and_then(|url| customers_domain_from_website(&url))
    });
    let existing_account = domain.as_ref().and_then(|value| {
        customers_find_by_string_field(conn, "customer_accounts", "domain", value)
            .ok()
            .flatten()
    });
    let batch_id = customers_id_from_command(command, &["import_batch_id"], "import")?;
    let mut batch = serde_json::json!({
        "id": batch_id,
        "source": "outbound",
        "source_record_id": source_record_id,
        "source_filename": "",
        "status": "completed",
        "object_type": "account",
        "imported_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "dedupe_count": 0,
        "payload": command.payload.clone(),
        "is_deleted": false,
        "created_at_ms": now
    });
    let mut contacts = Vec::new();
    if let Some(existing) = existing_account {
        let existing_id = outbound_required_string(&existing, &["id"])?;
        let candidate_id = format!(
            "dedupe_{}",
            channels::stable_digest(&format!("outbound:{source_record_id}:{existing_id}"))
        );
        let candidate = serde_json::json!({
            "id": candidate_id,
            "object_type": "account",
            "match_key": domain.unwrap_or(source_record_id.clone()),
            "match_type": "domain",
            "source_record_id": source_record_id,
            "existing_record_id": existing_id,
            "import_batch_id": batch.get("id").and_then(Value::as_str).unwrap_or_default(),
            "status": "open",
            "confidence": 0.95,
            "payload": { "outbound_company": company },
            "is_deleted": false,
            "created_at_ms": now
        });
        upsert_business_record(
            conn,
            "customer_dedupe_candidates",
            candidate
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            now,
            candidate.clone(),
        )?;
        customers_put_i64(&mut batch, "dedupe_count", 1);
        customers_put_string(&mut batch, "status", "needs_review");
        upsert_business_record(
            conn,
            "customer_import_batches",
            batch.get("id").and_then(Value::as_str).unwrap_or_default(),
            now,
            batch.clone(),
        )?;
        return Ok(serde_json::json!({
            "ok": true,
            "status": "needs_review",
            "import_batch": batch,
            "dedupe_candidate": candidate,
        }));
    }
    let account_id = customers_id_from_command(command, &["account_id"], "acct")?;
    let account_name = outbound_string(&company, &["name"])
        .or_else(|| outbound_string(&pipeline.clone().unwrap_or(Value::Null), &["company_name"]))
        .context("outbound company name is required")?;
    let mut account = serde_json::json!({
        "id": account_id,
        "name": account_name,
        "domain": domain.unwrap_or_default(),
        "website_url": outbound_string(&company, &["website"]).unwrap_or_default(),
        "account_status": "active",
        "customer_stage": "active",
        "health_status": "unknown",
        "source": "outbound",
        "source_record_id": source_record_id,
        "payload": { "outbound_company": company, "outbound_pipeline": pipeline },
        "is_deleted": false,
        "created_at_ms": now
    });
    customers_refresh_search_text(&mut account, &["name", "domain"]);
    upsert_business_record(
        conn,
        "customer_accounts",
        account
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        now,
        account.clone(),
    )?;
    if let Some(Value::Array(raw_contacts)) = account.pointer("/payload/outbound_pipeline/contacts")
    {
        for (index, raw_contact) in raw_contacts.iter().enumerate() {
            let contact_id = format!(
                "contact_{}",
                channels::stable_digest(&format!(
                    "{}:{}:{}",
                    account
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    index,
                    raw_contact
                ))
            );
            let full_name = outbound_string(raw_contact, &["name"]).unwrap_or_default();
            let (first_name, last_name) = customers_split_name(&full_name);
            let mut contact = serde_json::json!({
                "id": contact_id,
                "account_id": account.get("id").and_then(Value::as_str).unwrap_or_default(),
                "first_name": first_name,
                "last_name": last_name,
                "email": outbound_string(raw_contact, &["email"]).unwrap_or_default(),
                "phone": outbound_string(raw_contact, &["phone"]).unwrap_or_default(),
                "job_title": outbound_string(raw_contact, &["job_title"]).or_else(|| outbound_string(raw_contact, &["title"])).unwrap_or_default(),
                "is_primary_contact": index == 0,
                "source": "outbound",
                "source_record_id": outbound_string(raw_contact, &["id"]).unwrap_or_default(),
                "payload": raw_contact.clone(),
                "is_deleted": false,
                "created_at_ms": now
            });
            customers_refresh_search_text(
                &mut contact,
                &["first_name", "last_name", "email", "job_title"],
            );
            upsert_business_record(
                conn,
                "customer_contacts",
                contact
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                now,
                contact.clone(),
            )?;
            contacts.push(contact);
        }
    }
    customers_put_i64(&mut batch, "imported_count", 1 + contacts.len() as i64);
    upsert_business_record(
        conn,
        "customer_import_batches",
        batch.get("id").and_then(Value::as_str).unwrap_or_default(),
        now,
        batch.clone(),
    )?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "outbound_imported",
            name: "Outbound-Uebergabe importiert",
            account_id: outbound_string(&account, &["id"]),
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("account"),
            linked_record_id: outbound_string(&account, &["id"]),
            linked_record_name: outbound_string(&account, &["name"]),
            properties: serde_json::json!({ "import_batch": batch.clone(), "contact_count": contacts.len() }),
        },
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "status": "imported",
        "account": account,
        "contacts": contacts,
        "import_batch": batch,
        "activity": activity,
    }))
}

fn customers_dedupe_resolve(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
) -> anyhow::Result<Value> {
    let candidate_id = customers_required_from_payload_or_record(
        command,
        &["candidate_id", "dedupe_candidate_id", "id"],
        "candidate_id is required",
    )?;
    let mut candidate = outbound_load_required(
        conn,
        "customer_dedupe_candidates",
        &candidate_id,
        "dedupe candidate not found",
    )?;
    let decision = outbound_required_string(&command.payload, &["decision"])?;
    customers_validate_allowed(
        "dedupe decision",
        &decision,
        &["merge", "keep_existing", "create_new", "skip"],
    )?;
    if decision == "merge" {
        customers_require_field(&candidate, &["existing_record_id"])?;
    }
    customers_put_string(&mut candidate, "status", "resolved");
    customers_put_string(&mut candidate, "decision", decision.clone());
    customers_put_string(
        &mut candidate,
        "decided_by_id",
        outbound_session_actor_id(session),
    );
    customers_put_i64(&mut candidate, "decided_at_ms", now);
    upsert_business_record(
        conn,
        "customer_dedupe_candidates",
        &candidate_id,
        now,
        candidate.clone(),
    )?;
    let activity = customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type: "dedupe_resolved",
            name: "Duplikat entschieden",
            account_id: outbound_string(&candidate, &["existing_record_id"]),
            contact_id: None,
            opportunity_id: None,
            linked_record_type: Some("dedupe_candidate"),
            linked_record_id: Some(candidate_id.clone()),
            linked_record_name: outbound_string(&candidate, &["match_key"]),
            properties: serde_json::json!({ "candidate": candidate.clone(), "decision": decision }),
        },
    )?;
    Ok(serde_json::json!({ "ok": true, "dedupe_candidate": candidate, "activity": activity }))
}

const CUSTOMER_ACCOUNT_STATUSES: &[&str] = &["active", "inactive", "archived"];
const CUSTOMER_STAGES: &[&str] = &[
    "prospect",
    "onboarding",
    "active",
    "renewal",
    "expansion",
    "at_risk",
    "churned",
    "archived",
];
const CUSTOMER_HEALTH_STATUSES: &[&str] = &["unknown", "healthy", "neutral", "at_risk", "critical"];
const CUSTOMER_OPPORTUNITY_STAGES: &[&str] = &[
    "qualification",
    "proposal",
    "negotiation",
    "committed",
    "closed_won",
    "closed_lost",
];
const CUSTOMER_OPPORTUNITY_TYPES: &[&str] = &["new_business", "expansion", "renewal"];
const CUSTOMER_TASK_STATUSES: &[&str] = &["open", "in_progress", "completed", "cancelled"];

struct CustomersActivity<'a> {
    activity_type: &'a str,
    name: &'a str,
    account_id: Option<String>,
    contact_id: Option<String>,
    opportunity_id: Option<String>,
    linked_record_type: Option<&'a str>,
    linked_record_id: Option<String>,
    linked_record_name: Option<String>,
    properties: Value,
}

fn customers_write_activity(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
    activity: CustomersActivity<'_>,
) -> anyhow::Result<Value> {
    let command_id = command.id.as_deref().unwrap_or("customers-command");
    let activity_id = format!(
        "act_{}",
        channels::stable_digest(&format!(
            "{}:{}:{}",
            command_id,
            activity.activity_type,
            activity.linked_record_id.as_deref().unwrap_or_default()
        ))
    );
    let mut properties = activity.properties;
    if !properties.is_object() {
        properties = serde_json::json!({ "value": properties });
    }
    if let Some(object) = properties.as_object_mut() {
        object.insert(
            "command_id".to_string(),
            Value::String(command_id.to_string()),
        );
        object.insert(
            "command_type".to_string(),
            Value::String(command.command_type.clone()),
        );
    }
    let record = serde_json::json!({
        "id": activity_id,
        "happens_at_ms": now,
        "activity_type": activity.activity_type,
        "name": activity.name,
        "properties": properties,
        "actor_id": outbound_session_actor_id(session),
        "account_id": activity.account_id.unwrap_or_default(),
        "contact_id": activity.contact_id.unwrap_or_default(),
        "opportunity_id": activity.opportunity_id.unwrap_or_default(),
        "linked_record_type": activity.linked_record_type.unwrap_or_default(),
        "linked_record_id": activity.linked_record_id.unwrap_or_default(),
        "linked_record_name": activity.linked_record_name.unwrap_or_default(),
        "source": "customers",
        "source_record_id": command_id,
        "is_deleted": false,
        "created_at_ms": now,
    });
    upsert_business_record(
        conn,
        "customer_activities",
        &activity_id,
        now,
        record.clone(),
    )?;
    Ok(record)
}

fn customers_write_task_activity(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
    task: &Value,
    activity_type: &'static str,
    name: &'static str,
) -> anyhow::Result<Value> {
    let account_id = outbound_string(task, &["account_id"]);
    customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type,
            name,
            account_id: account_id.clone(),
            contact_id: outbound_string(task, &["contact_id"]),
            opportunity_id: outbound_string(task, &["opportunity_id"]),
            linked_record_type: Some("task"),
            linked_record_id: outbound_string(task, &["id"]),
            linked_record_name: outbound_string(task, &["title"]),
            properties: serde_json::json!({ "task": task.clone() }),
        },
    )
}

fn customers_write_note_activity(
    conn: &Connection,
    session: &BusinessOsSession,
    command: &BusinessCommand,
    now: i64,
    note: &Value,
    activity_type: &'static str,
    name: &'static str,
) -> anyhow::Result<Value> {
    customers_write_activity(
        conn,
        session,
        command,
        now,
        CustomersActivity {
            activity_type,
            name,
            account_id: outbound_string(note, &["account_id"]),
            contact_id: outbound_string(note, &["contact_id"]),
            opportunity_id: outbound_string(note, &["opportunity_id"]),
            linked_record_type: Some("note"),
            linked_record_id: outbound_string(note, &["id"]),
            linked_record_name: outbound_string(note, &["title"]),
            properties: serde_json::json!({ "note": note.clone() }),
        },
    )
}

fn customers_validate_account(account: &Value) -> anyhow::Result<()> {
    customers_validate_allowed(
        "account_status",
        &outbound_required_string(account, &["account_status"])?,
        CUSTOMER_ACCOUNT_STATUSES,
    )?;
    customers_validate_allowed(
        "customer_stage",
        &outbound_required_string(account, &["customer_stage"])?,
        CUSTOMER_STAGES,
    )?;
    customers_validate_allowed(
        "health_status",
        &outbound_required_string(account, &["health_status"])?,
        CUSTOMER_HEALTH_STATUSES,
    )?;
    Ok(())
}

fn customers_validate_opportunity(opportunity: &Value) -> anyhow::Result<()> {
    customers_validate_allowed(
        "opportunity_type",
        &outbound_required_string(opportunity, &["opportunity_type"])?,
        CUSTOMER_OPPORTUNITY_TYPES,
    )?;
    customers_validate_allowed(
        "opportunity stage",
        &outbound_required_string(opportunity, &["stage"])?,
        CUSTOMER_OPPORTUNITY_STAGES,
    )?;
    if let Some(probability) = customers_i64(opportunity, &["probability"]) {
        anyhow::ensure!(
            (0..=100).contains(&probability),
            "opportunity probability must be between 0 and 100"
        );
    }
    Ok(())
}

fn customers_validate_stage_transition(from_stage: &str, to_stage: &str) -> anyhow::Result<()> {
    if matches!(from_stage, "closed_won" | "closed_lost") && from_stage != to_stage {
        anyhow::bail!("closed opportunities cannot move stage");
    }
    Ok(())
}

fn customers_validate_allowed(label: &str, value: &str, allowed: &[&str]) -> anyhow::Result<()> {
    anyhow::ensure!(
        allowed.contains(&value),
        "{label} `{value}` is not supported"
    );
    Ok(())
}

fn customers_validate_links(
    conn: &Connection,
    record: &Value,
    require_any: bool,
) -> anyhow::Result<()> {
    let account_id = outbound_string(record, &["account_id"]);
    let contact_id = outbound_string(record, &["contact_id"]);
    let opportunity_id = outbound_string(record, &["opportunity_id"]);
    if require_any {
        anyhow::ensure!(
            account_id.is_some() || contact_id.is_some() || opportunity_id.is_some(),
            "account_id, contact_id or opportunity_id is required"
        );
    }
    if let Some(id) = account_id {
        customers_require_existing(conn, "customer_accounts", &id, "customer account not found")?;
    }
    if let Some(id) = contact_id {
        customers_require_existing(conn, "customer_contacts", &id, "customer contact not found")?;
    }
    if let Some(id) = opportunity_id {
        customers_require_existing(
            conn,
            "customer_opportunities",
            &id,
            "customer opportunity not found",
        )?;
    }
    Ok(())
}

fn customers_require_existing(
    conn: &Connection,
    collection: &str,
    record_id: &str,
    message: &str,
) -> anyhow::Result<()> {
    outbound_load_required(conn, collection, record_id, message)?;
    Ok(())
}

fn customers_require_field(value: &Value, path: &[&str]) -> anyhow::Result<String> {
    outbound_required_string(value, path)
}

fn customers_id_from_command(
    command: &BusinessCommand,
    payload_keys: &[&str],
    prefix: &str,
) -> anyhow::Result<String> {
    outbound_id_from_command(command, payload_keys, prefix)
}

fn customers_required_from_payload_or_record(
    command: &BusinessCommand,
    payload_keys: &[&str],
    message: &str,
) -> anyhow::Result<String> {
    outbound_required_from_payload_or_record(command, payload_keys, message)
}

fn customers_merge_fields(record: &mut Value, patch: &Value, keys: &[&str]) {
    outbound_merge_fields(record, patch, keys);
}

fn customers_put_string(record: &mut Value, key: &str, value: impl Into<String>) {
    outbound_put_string(record, key, value);
}

fn customers_put_i64(record: &mut Value, key: &str, value: i64) {
    outbound_put_i64(record, key, value);
}

fn customers_put_bool(record: &mut Value, key: &str, value: bool) {
    if let Some(object) = record.as_object_mut() {
        object.insert(key.to_string(), Value::Bool(value));
    }
}

fn customers_put_default_string(record: &mut Value, key: &str, default: &str) {
    outbound_put_default_string(record, key, default);
}

fn customers_put_default_i64(record: &mut Value, key: &str, default: i64) {
    outbound_put_default_i64(record, key, default);
}

fn customers_put_default_bool(record: &mut Value, key: &str, default: bool) {
    let should_insert = record.get(key).and_then(Value::as_bool).is_none();
    if should_insert {
        customers_put_bool(record, key, default);
    }
}

fn customers_put_default_object(record: &mut Value, key: &str) {
    outbound_put_default_object(record, key);
}

fn customers_i64(value: &Value, path: &[&str]) -> Option<i64> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor
        .as_i64()
        .or_else(|| cursor.as_u64().map(|value| value as i64))
}

fn customers_refresh_search_text(record: &mut Value, fields: &[&str]) {
    let text = fields
        .iter()
        .filter_map(|field| outbound_string(record, &[*field]))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    customers_put_string(record, "search_text", text);
}

fn customers_contact_display_name(contact: &Value) -> String {
    let first = outbound_string(contact, &["first_name"]).unwrap_or_default();
    let last = outbound_string(contact, &["last_name"]).unwrap_or_default();
    let name = format!("{first} {last}").trim().to_string();
    if name.is_empty() {
        outbound_string(contact, &["email"]).unwrap_or_else(|| "Kontakt".to_string())
    } else {
        name
    }
}

fn customers_touch_account_last_activity(
    conn: &Connection,
    account_id: &str,
    now: i64,
) -> anyhow::Result<()> {
    if account_id.trim().is_empty() {
        return Ok(());
    }
    let Some(mut account) = outbound_load_record(conn, "customer_accounts", account_id)? else {
        return Ok(());
    };
    customers_put_i64(&mut account, "last_activity_at_ms", now);
    upsert_business_record(conn, "customer_accounts", account_id, now, account)
}

fn customers_save_view_children(
    conn: &Connection,
    view_id: &str,
    value: Option<&Value>,
    collection: &str,
    now: i64,
) -> anyhow::Result<Vec<Value>> {
    let Some(Value::Array(items)) = value else {
        return Ok(Vec::new());
    };
    let mut saved = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let mut record = outbound_object_payload(item);
        customers_put_string(&mut record, "view_id", view_id.to_string());
        customers_put_default_i64(&mut record, "position", index as i64);
        customers_put_default_bool(&mut record, "is_deleted", false);
        customers_put_default_i64(&mut record, "created_at_ms", now);
        let id = outbound_string(&record, &["id"]).unwrap_or_else(|| {
            format!(
                "{}_{}",
                if collection == "customer_view_filters" {
                    "filter"
                } else {
                    "sort"
                },
                channels::stable_digest(&format!("{view_id}:{collection}:{index}:{record}"))
            )
        });
        customers_put_string(&mut record, "id", id.clone());
        upsert_business_record(conn, collection, &id, now, record.clone())?;
        saved.push(record);
    }
    Ok(saved)
}

fn customers_find_by_string_field(
    conn: &Connection,
    collection: &str,
    field: &str,
    value: &str,
) -> anyhow::Result<Option<Value>> {
    let mut stmt = conn.prepare(
        "SELECT payload_json
         FROM business_records
         WHERE collection = ?1 AND deleted = 0",
    )?;
    let rows = stmt.query_map(params![collection], |row| row.get::<_, String>(0))?;
    let expected = value.trim().to_ascii_lowercase();
    for row in rows {
        let record: Value = serde_json::from_str(&row?)?;
        let actual = outbound_string(&record, &[field])
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !expected.is_empty() && actual == expected {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

fn customers_find_outbound_pipeline_for_company(
    conn: &Connection,
    company_id: &str,
) -> anyhow::Result<Option<String>> {
    let mut stmt = conn.prepare(
        "SELECT record_id, payload_json
         FROM business_records
         WHERE collection = 'outbound_pipeline_items' AND deleted = 0",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (record_id, raw) = row?;
        let record: Value = serde_json::from_str(&raw)?;
        if outbound_string(&record, &["company_id"]).as_deref() == Some(company_id) {
            return Ok(Some(record_id));
        }
    }
    Ok(None)
}

fn customers_domain_from_website(value: &str) -> Option<String> {
    let raw = value.trim();
    if raw.is_empty() {
        return None;
    }
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("https://{raw}")
    };
    Url::parse(&candidate)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .map(|host| host.trim_start_matches("www.").to_ascii_lowercase())
}

fn customers_split_name(value: &str) -> (String, String) {
    let mut parts = value.split_whitespace().collect::<Vec<_>>();
    if parts.is_empty() {
        return (String::new(), String::new());
    }
    let first = parts.remove(0).to_string();
    let last = parts.join(" ");
    (first, last)
}

#[cfg(test)]
mod tests {
    use super::super::store::{
        accept_rxdb_business_command, open_store, outbound_load_record,
        outbound_load_records_by_string_field, outbound_load_required, outbound_string,
    };
    use super::super::store_projections::upsert_business_record;
    use anyhow::Context;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn customers_commands_create_pipeline_task_and_activities() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let root = temp.path();
        let actor = serde_json::json!({
            "actor": { "id": "tester", "role": "admin", "display_name": "Tester" }
        });

        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_account_create",
                "command_id": "cmd_customer_account_create",
                "module": "customers",
                "command_type": "customers.account.create",
                "record_id": "acct_customers",
                "status": "pending_sync",
                "payload": {
                    "account_id": "acct_customers",
                    "name": "Metric Space",
                    "domain": "metric-space.ai",
                    "health_status": "healthy"
                },
                "client_context": actor.clone()
            }),
        )?;
        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_contact_create",
                "command_id": "cmd_customer_contact_create",
                "module": "customers",
                "command_type": "customers.contact.create",
                "record_id": "contact_customers",
                "status": "pending_sync",
                "payload": {
                    "contact_id": "contact_customers",
                    "account_id": "acct_customers",
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                    "is_primary_contact": true
                },
                "client_context": actor.clone()
            }),
        )?;
        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_opportunity_create",
                "command_id": "cmd_customer_opportunity_create",
                "module": "customers",
                "command_type": "customers.opportunity.create",
                "record_id": "opp_customers",
                "status": "pending_sync",
                "payload": {
                    "opportunity_id": "opp_customers",
                    "name": "Renewal 2026",
                    "account_id": "acct_customers",
                    "primary_contact_id": "contact_customers",
                    "amount_cents": 1200000,
                    "opportunity_type": "renewal"
                },
                "client_context": actor.clone()
            }),
        )?;
        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_opportunity_stage",
                "command_id": "cmd_customer_opportunity_stage",
                "module": "customers",
                "command_type": "customers.opportunity.move_stage",
                "record_id": "opp_customers",
                "status": "pending_sync",
                "payload": {
                    "opportunity_id": "opp_customers",
                    "stage": "proposal"
                },
                "client_context": actor.clone()
            }),
        )?;
        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_task_create",
                "command_id": "cmd_customer_task_create",
                "module": "customers",
                "command_type": "customers.task.create",
                "record_id": "task_customers",
                "status": "pending_sync",
                "payload": {
                    "task_id": "task_customers",
                    "account_id": "acct_customers",
                    "opportunity_id": "opp_customers",
                    "title": "Renewal Call vorbereiten"
                },
                "client_context": actor.clone()
            }),
        )?;
        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_task_complete",
                "command_id": "cmd_customer_task_complete",
                "module": "customers",
                "command_type": "customers.task.complete",
                "record_id": "task_customers",
                "status": "pending_sync",
                "payload": { "task_id": "task_customers" },
                "client_context": actor
            }),
        )?;

        let conn = open_store(root)?;
        let opportunity = outbound_load_required(
            &conn,
            "customer_opportunities",
            "opp_customers",
            "opportunity",
        )?;
        assert_eq!(
            outbound_string(&opportunity, &["stage"]).as_deref(),
            Some("proposal")
        );
        let task = outbound_load_required(&conn, "customer_tasks", "task_customers", "task")?;
        assert_eq!(
            outbound_string(&task, &["status"]).as_deref(),
            Some("completed")
        );
        let activity_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM business_records WHERE collection = 'customer_activities' AND deleted = 0",
            [],
            |row| row.get(0),
        )?;
        assert!(
            activity_count >= 6,
            "expected activities for every Customers transition"
        );
        Ok(())
    }

    #[test]
    fn customers_invalid_command_writes_failed_projection_without_partial_record(
    ) -> anyhow::Result<()> {
        let temp = tempdir()?;
        let root = temp.path();
        let actor = serde_json::json!({
            "actor": { "id": "tester", "role": "admin", "display_name": "Tester" }
        });

        let err = accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_invalid_account",
                "command_id": "cmd_customer_invalid_account",
                "module": "customers",
                "command_type": "customers.account.create",
                "record_id": "acct_invalid",
                "status": "pending_sync",
                "payload": {
                    "account_id": "acct_invalid",
                    "name": "Invalid GmbH",
                    "health_status": "glowing"
                },
                "client_context": actor
            }),
        )
        .expect_err("unsupported health status must fail");
        assert!(err.to_string().contains("health_status"), "{err}");

        let conn = open_store(root)?;
        assert!(outbound_load_record(&conn, "customer_accounts", "acct_invalid")?.is_none());
        let command = outbound_load_required(
            &conn,
            "business_commands",
            "cmd_customer_invalid_account",
            "command",
        )?;
        assert_eq!(
            outbound_string(&command, &["status"]).as_deref(),
            Some("failed")
        );
        assert!(command
            .pointer("/result/error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("health_status"));
        Ok(())
    }

    #[test]
    fn customers_import_from_outbound_creates_account_contacts_and_dedupe() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let root = temp.path();
        let actor = serde_json::json!({
            "actor": { "id": "tester", "role": "admin", "display_name": "Tester" }
        });
        let conn = open_store(root)?;
        upsert_business_record(
            &conn,
            "outbound_companies",
            "company_one",
            1,
            serde_json::json!({
                "id": "company_one",
                "campaign_id": "camp",
                "name": "Acme GmbH",
                "domain": "acme.example",
                "website": "https://acme.example",
                "qualification_status": "qualified",
                "research_status": "done",
                "pipeline_status": "ready",
                "payload": {},
                "created_at_ms": 1
            }),
        )?;
        upsert_business_record(
            &conn,
            "outbound_pipeline_items",
            "pipeline_one",
            1,
            serde_json::json!({
                "id": "pipeline_one",
                "campaign_id": "camp",
                "company_id": "company_one",
                "company_name": "Acme GmbH",
                "stage": "qualified",
                "contact_research_status": "done",
                "outreach_status": "ready",
                "contacts": [
                    { "name": "Grace Hopper", "email": "grace@acme.example", "job_title": "CTO" }
                ],
                "payload": {},
                "created_at_ms": 1
            }),
        )?;
        drop(conn);

        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_import",
                "command_id": "cmd_customer_import",
                "module": "customers",
                "command_type": "customers.import.from_outbound",
                "record_id": "company_one",
                "status": "pending_sync",
                "payload": {
                    "source_record_id": "company_one",
                    "pipeline_id": "pipeline_one",
                    "account_id": "acct_acme"
                },
                "client_context": actor.clone()
            }),
        )?;

        let conn = open_store(root)?;
        let account = outbound_load_required(&conn, "customer_accounts", "acct_acme", "account")?;
        assert_eq!(
            outbound_string(&account, &["name"]).as_deref(),
            Some("Acme GmbH")
        );
        let contacts = outbound_load_records_by_string_field(
            &conn,
            "customer_contacts",
            "account_id",
            "acct_acme",
        )?;
        assert_eq!(contacts.len(), 1);
        drop(conn);

        let conn = open_store(root)?;
        upsert_business_record(
            &conn,
            "outbound_companies",
            "company_duplicate",
            2,
            serde_json::json!({
                "id": "company_duplicate",
                "campaign_id": "camp",
                "name": "Acme Duplicate",
                "domain": "acme.example",
                "qualification_status": "qualified",
                "research_status": "done",
                "pipeline_status": "ready",
                "payload": {},
                "created_at_ms": 2
            }),
        )?;
        drop(conn);

        let duplicate = accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_import_duplicate",
                "command_id": "cmd_customer_import_duplicate",
                "module": "customers",
                "command_type": "customers.import.from_outbound",
                "record_id": "company_duplicate",
                "status": "pending_sync",
                "payload": { "source_record_id": "company_duplicate" },
                "client_context": actor.clone()
            }),
        )?;
        assert_eq!(
            duplicate.pointer("/result/status").and_then(Value::as_str),
            Some("needs_review")
        );
        let candidate_id = duplicate
            .pointer("/result/dedupe_candidate/id")
            .and_then(Value::as_str)
            .context("dedupe candidate id")?
            .to_string();

        accept_rxdb_business_command(
            root,
            serde_json::json!({
                "id": "cmd_customer_dedupe_resolve",
                "command_id": "cmd_customer_dedupe_resolve",
                "module": "customers",
                "command_type": "customers.dedupe.resolve",
                "record_id": candidate_id,
                "status": "pending_sync",
                "payload": {
                    "candidate_id": candidate_id,
                    "decision": "keep_existing"
                },
                "client_context": actor
            }),
        )?;
        let conn = open_store(root)?;
        let resolved = outbound_load_records_by_string_field(
            &conn,
            "customer_dedupe_candidates",
            "status",
            "resolved",
        )?;
        assert_eq!(resolved.len(), 1);
        Ok(())
    }
}
