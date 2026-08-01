// Origin: CTOX
// License: Apache-2.0

use super::policy::{self, BusinessOsPermission};
use super::store::{
    business_command_client_context_value, business_os_app_command_target_metadata,
    rxdb_authenticated_session, BusinessCommand,
};
use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;

pub(super) fn normalize_business_role(role: &str) -> String {
    policy::normalize_role(role)
}

pub(super) fn founder_owns_module(
    conn: &Connection,
    module_id: &str,
    user_id: &str,
) -> anyhow::Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*)
         FROM business_module_acl acl
         LEFT JOIN business_users user ON user.user_id = acl.user_id
         WHERE acl.module_id = ?1
           AND acl.user_id = ?2
           AND acl.role = 'founder'
           AND acl.active = 1
           AND COALESCE(user.active, 1) = 1",
        params![module_id.trim(), user_id.trim()],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub(super) fn app_build_command_policy_target(
    command: &BusinessCommand,
) -> Option<(BusinessOsPermission, String)> {
    let (module_id, _install_target, _module_dir) =
        business_os_app_command_target_metadata(command)?;
    let permission = if command.command_type == "ctox.business_os.app.modify" {
        BusinessOsPermission::AppsModify
    } else {
        BusinessOsPermission::AppsInstall
    };
    Some((permission, module_id))
}

pub(super) fn support_actor_summary(value: Option<&Value>) -> Value {
    let value = value.unwrap_or(&Value::Null);
    serde_json::json!({
        "id": support_string(value.get("id")),
        "display_name": support_string(value.get("display_name")),
        "role": support_string(value.get("role")),
        "trusted": value.get("trusted").cloned().unwrap_or(Value::Null),
    })
}

pub(super) fn support_policy_decision_summary(value: Option<&Value>) -> Value {
    let value = value.unwrap_or(&Value::Null);
    serde_json::json!({
        "allowed": value.get("allowed").cloned().unwrap_or(Value::Bool(false)),
        "permission": support_string(value.get("permission")),
        "scope_type": support_string(value.get("scope_type")),
        "scope_id": value.get("scope_id").cloned().unwrap_or(Value::Null),
        "reason_code": support_string(value.get("reason_code")),
        "display_reason": support_string(value.get("display_reason")),
        "requires_approval": value
            .get("requires_approval")
            .cloned()
            .unwrap_or(Value::Bool(false)),
        "audit_level": support_string(value.get("audit_level")),
        "source": support_string(value.get("source")),
    })
}

pub(super) fn support_client_scope_summary(value: Option<&Value>) -> Value {
    let value = value.unwrap_or(&Value::Null);
    let visible_scope = value.get("visible_scope").unwrap_or(value);
    serde_json::json!({
        "source": support_string(value.get("source")),
        "module_id": support_string(
            visible_scope
                .get("module_id")
                .or_else(|| visible_scope.get("module"))
                .or_else(|| value.get("module_id"))
        ),
        "app_id": support_string(
            visible_scope
                .get("app_id")
                .or_else(|| visible_scope.get("app"))
                .or_else(|| value.get("app_id"))
        ),
        "record_id": support_string(
            visible_scope
                .get("record_id")
                .or_else(|| value.get("record_id"))
        ),
        "collection": support_string(
            visible_scope
                .get("collection")
                .or_else(|| value.get("collection"))
        ),
        "action": support_string(
            visible_scope
                .get("action")
                .or_else(|| value.get("action"))
        ),
    })
}

pub(super) fn support_string(value: Option<&Value>) -> Value {
    value
        .and_then(Value::as_str)
        .map(|text| Value::String(text.trim().chars().take(240).collect()))
        .unwrap_or(Value::Null)
}

pub(super) fn policy_audit_actor_context(root: &Path, command: &BusinessCommand) -> Value {
    if let Ok(session) = rxdb_authenticated_session(root, command) {
        if let Some(user) = session.user {
            return serde_json::json!({
                "id": user.id,
                "display_name": user.display_name,
                "role": user.role,
                "trusted": true
            });
        }
    }
    let client_context = business_command_client_context_value(command);
    let actor = client_context
        .get("actor")
        .or_else(|| client_context.get("user"));
    let id = actor
        .and_then(|value| value.get("id"))
        .or_else(|| client_context.get("user_id"))
        .and_then(Value::as_str)
        .unwrap_or("rxdb-command");
    let display_name = actor
        .and_then(|value| value.get("display_name"))
        .or_else(|| actor.and_then(|value| value.get("name")))
        .or_else(|| client_context.get("display_name"))
        .and_then(Value::as_str)
        .unwrap_or(id);
    serde_json::json!({
        "id": id,
        "display_name": display_name,
        "trusted": false
    })
}

pub(super) fn policy_audit_actor_context_from_client_context(
    client_context: Option<&Value>,
) -> Value {
    let actor = client_context
        .and_then(|context| context.get("actor"))
        .or_else(|| client_context.and_then(|context| context.get("user")));
    let id = actor
        .and_then(|value| value.get("id"))
        .or_else(|| client_context.and_then(|context| context.get("user_id")))
        .and_then(Value::as_str)
        .unwrap_or("rxdb-command");
    let display_name = actor
        .and_then(|value| value.get("display_name"))
        .or_else(|| actor.and_then(|value| value.get("name")))
        .or_else(|| client_context.and_then(|context| context.get("display_name")))
        .and_then(Value::as_str)
        .unwrap_or(id);
    serde_json::json!({
        "id": id,
        "display_name": display_name,
        "trusted": false
    })
}

pub(super) fn policy_audit_client_context(command: &BusinessCommand) -> Value {
    let client_context = business_command_client_context_value(command);
    let Some(object) = client_context.as_object() else {
        return Value::Null;
    };
    let mut audited = serde_json::Map::new();
    for key in [
        "source",
        "surface",
        "action",
        "mode",
        "target",
        "module",
        "module_id",
        "app_id",
        "source_module",
        "record_id",
        "record_type",
        "collection",
        "column",
        "active_scope",
        "version",
        "visibility",
    ] {
        if let Some(value) = object.get(key).and_then(policy_audit_safe_scalar) {
            audited.insert(key.to_string(), value);
        }
    }
    if let Some(scope) = object
        .get("visible_scope")
        .or_else(|| client_context.pointer("/scope/visible_scope"))
        .map(policy_audit_visible_scope_value)
    {
        audited.insert("visible_scope".to_string(), scope);
    }
    Value::Object(audited)
}

fn policy_audit_safe_scalar(value: &Value) -> Option<Value> {
    match value {
        Value::String(text) => Some(Value::String(policy_audit_truncate(text, 160))),
        Value::Number(_) | Value::Bool(_) | Value::Null => Some(value.clone()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn policy_audit_visible_scope_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut audited = serde_json::Map::new();
            for (key, nested) in object {
                if !policy_audit_visible_scope_key_allowed(key) {
                    continue;
                }
                let audited_value = policy_audit_visible_scope_value(nested);
                if !audited_value.is_null() {
                    audited.insert(key.clone(), audited_value);
                }
            }
            Value::Object(audited)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .take(24)
                .map(policy_audit_visible_scope_value)
                .filter(|item| !item.is_null())
                .collect(),
        ),
        Value::String(text) => Value::String(policy_audit_truncate(text, 240)),
        Value::Number(_) | Value::Bool(_) => value.clone(),
        Value::Null => Value::Null,
    }
}

fn policy_audit_visible_scope_key_allowed(key: &str) -> bool {
    matches!(
        key,
        "access"
            | "actor"
            | "app"
            | "app_id"
            | "badge"
            | "can_modify"
            | "can_read"
            | "can_write"
            | "collection"
            | "data"
            | "display_name"
            | "external_actions"
            | "id"
            | "key"
            | "label"
            | "lifecycle"
            | "mode"
            | "module_id"
            | "permission"
            | "permission_label"
            | "record_id"
            | "record_type"
            | "role"
            | "rows"
            | "scope"
            | "scope_id"
            | "scope_type"
            | "selection"
            | "source"
            | "status"
            | "target"
            | "title"
            | "trusted"
            | "value"
            | "version"
            | "visibility"
    )
}

fn policy_audit_truncate(text: &str, max_chars: usize) -> String {
    let mut truncated = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            truncated.push_str("...");
            return truncated;
        }
        truncated.push(ch);
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::normalize_business_role;

    #[test]
    fn business_role_normalization_preserves_phase0_aliases() {
        assert_eq!(normalize_business_role("chef"), "chef");
        assert_eq!(normalize_business_role("owner"), "chef");
        assert_eq!(normalize_business_role("business_os_admin"), "admin");
        assert_eq!(normalize_business_role("founder"), "founder");
        assert_eq!(normalize_business_role("user"), "user");
        assert_eq!(normalize_business_role("business_os_user"), "user");
        assert_eq!(normalize_business_role("team"), "user");
        assert_eq!(normalize_business_role("business_os_team"), "user");
        assert_eq!(normalize_business_role("unknown"), "user");
    }
}
