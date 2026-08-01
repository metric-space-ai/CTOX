// Release-Review und Data-Access-Audit.
//
// Diese Familie beantwortet eine einzige Frage: ist ein Release fachlich
// geprueft, und welchen Datenzugriff hat es dabei zugesagt. Sie lag verstreut
// im Store zwischen Versionskern und Grant-Helfern; die Reihenfolge dort war
// gewachsen, nicht gemeint.
//
// Move-only aus store.rs: Koerper unveraendert.

use anyhow::Context;
use rusqlite::Connection;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::policy::BusinessOsPermission;
use super::store::active_team_data_grant_scope;

pub(super) fn module_release_review_completed(review: &Value) -> bool {
    review.get("completed").and_then(Value::as_bool) == Some(true)
        || matches!(
            review
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "completed" | "approved" | "reviewed"
        )
}

pub(super) fn module_release_review_collection_ids(review: &Value) -> Vec<String> {
    let Some(values) = review.get("collections").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut collections = values
        .iter()
        .filter_map(|value| {
            value.as_str().or_else(|| {
                value
                    .get("collection")
                    .or_else(|| value.get("collection_id"))
                    .or_else(|| value.get("id"))
                    .or_else(|| value.get("name"))
                    .and_then(Value::as_str)
            })
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    collections.sort();
    collections.dedup();
    collections
}

pub(super) fn module_release_data_access_review_summary(
    conn: &Connection,
    module_id: &str,
    review: &Value,
    expected_collections: &[String],
    reviewed_at_ms: i64,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        review.is_object(),
        "data_access_review is required before publishing a Team release"
    );
    anyhow::ensure!(
        module_release_review_completed(review),
        "data_access_review must be completed before publishing a Team release"
    );
    let reviewed_collections = module_release_review_collection_ids(review);
    anyhow::ensure!(
        reviewed_collections == expected_collections,
        "data_access_review collections must match the module manifest collections"
    );
    let expected = expected_collections
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let read_collections = release_review_access_collection_ids(review, "read_collections");
    let write_collections = release_review_access_collection_ids(review, "write_collections");
    let locked_read_collections =
        release_review_access_collection_ids(review, "locked_read_collections");
    let locked_write_collections =
        release_review_access_collection_ids(review, "locked_write_collections");
    let unknown_read = read_collections
        .iter()
        .find(|collection| !expected.contains(collection.as_str()));
    anyhow::ensure!(
        unknown_read.is_none(),
        "data_access_review read_collections must be declared in the module manifest"
    );
    let unknown_write = write_collections
        .iter()
        .find(|collection| !expected.contains(collection.as_str()));
    anyhow::ensure!(
        unknown_write.is_none(),
        "data_access_review write_collections must be declared in the module manifest"
    );
    let unknown_locked_read = locked_read_collections
        .iter()
        .find(|collection| !expected.contains(collection.as_str()));
    anyhow::ensure!(
        unknown_locked_read.is_none(),
        "data_access_review locked_read_collections must be declared in the module manifest"
    );
    let unknown_locked_write = locked_write_collections
        .iter()
        .find(|collection| !expected.contains(collection.as_str()));
    anyhow::ensure!(
        unknown_locked_write.is_none(),
        "data_access_review locked_write_collections must be declared in the module manifest"
    );
    let locked_state_behavior = release_review_locked_state_behavior(review);
    let read_reconciliation = release_review_access_reconciliation(
        conn,
        module_id,
        BusinessOsPermission::DataRead,
        &read_collections,
        &locked_read_collections,
        &locked_state_behavior,
    )?;
    let write_reconciliation = release_review_access_reconciliation(
        conn,
        module_id,
        BusinessOsPermission::DataWrite,
        &write_collections,
        &locked_write_collections,
        &locked_state_behavior,
    )?;
    Ok(serde_json::json!({
        "completed": true,
        "status": review
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("completed"),
        "reviewed_by": review
            .get("reviewed_by")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        "reviewed_at_ms": review
            .get("reviewed_at_ms")
            .and_then(Value::as_i64)
            .unwrap_or(reviewed_at_ms),
        "collections": reviewed_collections,
        "read_collections": read_collections,
        "write_collections": write_collections,
        "locked_read_collections": locked_read_collections,
        "locked_write_collections": locked_write_collections,
        "locked_state_behavior": locked_state_behavior,
        "grant_reconciliation": {
            "audience": "team",
            "role": "user",
            "read": read_reconciliation,
            "write": write_reconciliation
        },
        "review_is_evidence_only": true,
        "grants_implied": false,
        "notes": review
            .get("notes")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }))
}

pub(super) fn release_review_locked_state_behavior(review: &Value) -> String {
    review
        .get("locked_state_behavior")
        .or_else(|| review.pointer("/locked_state/behavior"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

pub(super) fn release_review_access_reconciliation(
    conn: &Connection,
    module_id: &str,
    permission: BusinessOsPermission,
    collections: &[String],
    locked_collections: &[String],
    locked_state_behavior: &str,
) -> anyhow::Result<Vec<Value>> {
    let locked = locked_collections
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut reconciled = Vec::new();
    for collection in collections {
        if let Some(grant_scope) =
            active_team_data_grant_scope(conn, module_id, collection, permission)?
        {
            reconciled.push(serde_json::json!({
                "collection": collection,
                "permission": permission.as_str(),
                "status": "granted",
                "grant_subject_type": "role",
                "grant_subject_id": "user",
                "grant_scope": grant_scope,
            }));
            continue;
        }
        anyhow::ensure!(
            locked.contains(collection.as_str()) && !locked_state_behavior.is_empty(),
            "data_access_review {} collection '{}' must have an explicit Team grant or locked-state behavior",
            permission.as_str(),
            collection
        );
        reconciled.push(serde_json::json!({
            "collection": collection,
            "permission": permission.as_str(),
            "status": "locked",
            "locked_state_behavior": locked_state_behavior,
        }));
    }
    Ok(reconciled)
}

pub(super) fn release_review_access_collection_ids(review: &Value, key: &str) -> Vec<String> {
    let mut collections = review
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    collections.sort();
    collections.dedup();
    collections
}

pub(super) fn data_access_review_from_release_snapshot(snapshot: &Value) -> Value {
    snapshot
        .get("data_access_review")
        .cloned()
        .unwrap_or(Value::Null)
}

pub(super) fn release_review_reconciliation_statuses(
    review: &Value,
    side: &str,
) -> BTreeMap<String, String> {
    let Some(rows) = review
        .get("grant_reconciliation")
        .and_then(|reconciliation| reconciliation.get(side))
        .and_then(Value::as_array)
    else {
        return BTreeMap::new();
    };
    let mut statuses = BTreeMap::new();
    for row in rows {
        let Some(collection) = row
            .get("collection")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let status = row
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .trim()
            .to_owned();
        statuses.insert(collection.to_owned(), status);
    }
    statuses
}

pub(super) fn release_review_data_access_projection(review: &Value) -> Value {
    if !review.is_object() {
        return serde_json::json!({
            "status": "not_reviewed",
            "completed": false,
            "areas": [],
            "granted_collection_ids": [],
            "locked_collection_ids": [],
            "locked_state_behavior": ""
        });
    }

    let read_statuses = release_review_reconciliation_statuses(review, "read");
    let write_statuses = release_review_reconciliation_statuses(review, "write");
    let mut collections = BTreeSet::new();
    for collection in module_release_review_collection_ids(review) {
        collections.insert(collection);
    }
    for collection in read_statuses.keys() {
        collections.insert(collection.clone());
    }
    for collection in write_statuses.keys() {
        collections.insert(collection.clone());
    }

    let mut areas = Vec::new();
    let mut granted_collection_ids = BTreeSet::new();
    let mut locked_collection_ids = BTreeSet::new();
    for collection in collections {
        let read = read_statuses
            .get(&collection)
            .map(String::as_str)
            .unwrap_or("not_requested");
        let write = write_statuses
            .get(&collection)
            .map(String::as_str)
            .unwrap_or("not_requested");
        if read == "granted" || write == "granted" {
            granted_collection_ids.insert(collection.clone());
        }
        if read == "locked" || write == "locked" {
            locked_collection_ids.insert(collection.clone());
        }
        areas.push(serde_json::json!({
            "collection": collection,
            "read": read,
            "write": write,
            "locked": read == "locked" || write == "locked",
            "granted": read == "granted" || write == "granted",
        }));
    }

    serde_json::json!({
        "status": if module_release_review_completed(review) { "reviewed" } else { "incomplete" },
        "completed": module_release_review_completed(review),
        "reviewed_by": review
            .get("reviewed_by")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        "reviewed_at_ms": review
            .get("reviewed_at_ms")
            .cloned()
            .unwrap_or(Value::Null),
        "areas": areas,
        "granted_collection_ids": granted_collection_ids.into_iter().map(Value::String).collect::<Vec<_>>(),
        "locked_collection_ids": locked_collection_ids.into_iter().map(Value::String).collect::<Vec<_>>(),
        "locked_state_behavior": release_review_locked_state_behavior(review),
        "review_is_evidence_only": review
            .get("review_is_evidence_only")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "grants_implied": review
            .get("grants_implied")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

pub(super) fn release_audit_review_from_result(
    result: &Value,
    module_id: &str,
    version_id: &str,
) -> Value {
    result
        .get("releases")
        .and_then(|releases| releases.get(module_id))
        .and_then(Value::as_array)
        .and_then(|releases| {
            releases.iter().find(|release| {
                release
                    .get("version_id")
                    .and_then(Value::as_str)
                    .is_some_and(|candidate| candidate == version_id)
            })
        })
        .and_then(|release| release.get("data_access_review"))
        .cloned()
        .unwrap_or(Value::Null)
}
