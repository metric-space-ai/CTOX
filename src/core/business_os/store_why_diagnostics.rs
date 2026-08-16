fn business_os_why_diagnostics(
    root: &Path,
    session: &BusinessOsSession,
    request: &BusinessOsWhyDiagnosticsRequest,
) -> anyhow::Result<Value> {
    let module_id = source_sanitize_slug(&request.module_id);
    anyhow::ensure!(!module_id.is_empty(), "module_id is required");

    let catalog = module_catalog_for_rxdb(root)?;
    let module = catalog
        .get("modules")
        .and_then(Value::as_array)
        .and_then(|modules| {
            modules
                .iter()
                .find(|module| {
                    module
                        .get("id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| id == module_id)
                })
                .cloned()
        })
        .with_context(|| format!("module '{module_id}' was not found"))?;
    let lifecycle = module.get("lifecycle").cloned().unwrap_or(Value::Null);
    let conn = open_store(root)?;
    let actor_id = session_user_id(session)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("rxdb-command")
        .to_owned();
    let actor_role = session_role(session).to_owned();
    let actor_display_name = session
        .user
        .as_ref()
        .map(|user| user.display_name.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or(actor_id.as_str())
        .to_owned();

    let visibility = business_os_why_visibility_decision(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        module_id.as_str(),
        &lifecycle,
    )?;
    let open = business_os_why_open_decision(&visibility);
    let modify = business_os_why_module_policy_decision(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        BusinessOsPermission::AppsModify,
        module_id.as_str(),
        "policy_engine",
    )?;
    let source = business_os_why_module_policy_decision(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        BusinessOsPermission::AppsSourceView,
        module_id.as_str(),
        "policy_engine",
    )?;
    let release = business_os_why_module_policy_decision(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        BusinessOsPermission::AppsRelease,
        module_id.as_str(),
        "policy_engine",
    )?;
    let rollback = business_os_why_module_policy_decision(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        BusinessOsPermission::AppsRollback,
        module_id.as_str(),
        "policy_engine",
    )?;
    let data_areas = business_os_why_data_area_diagnostics(
        &conn,
        actor_id.as_str(),
        actor_role.as_str(),
        module_id.as_str(),
        &module,
        &lifecycle,
    )?;

    Ok(serde_json::json!({
        "ok": true,
        "schema_version": 1,
        "kind": "business_os_why_diagnostics",
        "actor": {
            "id": actor_id,
            "display_name": actor_display_name,
            "role": actor_role,
        },
        "module": {
            "id": module_id,
            "title": module.get("title").and_then(Value::as_str).unwrap_or_default(),
            "version": lifecycle
                .get("current_semver")
                .and_then(Value::as_str)
                .or_else(|| module.get("version").and_then(Value::as_str))
                .unwrap_or_default(),
            "runtime_installed": lifecycle
                .get("runtime_installed")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        "lifecycle": {
            "visibility_state": lifecycle
                .get("visibility_state")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            "audience": lifecycle
                .get("audience")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            "release_channel": lifecycle
                .get("release_channel")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            "current_semver": lifecycle.get("current_semver").cloned().unwrap_or(Value::Null),
            "public": lifecycle.get("public").and_then(Value::as_bool).unwrap_or(false),
            "warning_code": lifecycle.get("warning_code").cloned().unwrap_or(Value::Null),
            "release_state": lifecycle.get("release_state").cloned().unwrap_or(Value::Null),
        },
        "decisions": {
            "visibility": visibility,
            "open": open,
            "modify": modify,
            "source": source,
            "release": release,
            "rollback": rollback,
        },
        "data_access": {
            "status": lifecycle
                .pointer("/data_access/status")
                .and_then(Value::as_str)
                .unwrap_or("not_reviewed"),
            "completed": lifecycle
                .pointer("/data_access/completed")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "areas": data_areas,
        },
    }))
}

fn business_os_why_safe_command(
    command: &BusinessCommand,
    request: &BusinessOsWhyDiagnosticsRequest,
    session: &BusinessOsSession,
) -> BusinessCommand {
    BusinessCommand {
        origin: CommandOrigin::TrustedLocal,
        id: command.id.clone(),
        module: command.module.clone(),
        command_type: command.command_type.clone(),
        record_id: command.record_id.clone(),
        payload: serde_json::json!({
            "module_id": source_sanitize_slug(&request.module_id),
        }),
        client_context: serde_json::json!({
            "actor": {
                "id": session_user_id(session).unwrap_or("rxdb-command"),
                "display_name": session
                    .user
                    .as_ref()
                    .map(|user| user.display_name.as_str())
                    .unwrap_or("rxdb-command"),
                "role": session_role(session),
            }
        }),
    }
}

fn business_os_why_visibility_decision(
    conn: &Connection,
    actor_id: &str,
    actor_role: &str,
    module_id: &str,
    lifecycle: &Value,
) -> anyhow::Result<Value> {
    let visibility_state = lifecycle
        .get("visibility_state")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let public = lifecycle.get("public").and_then(Value::as_bool) == Some(true);
    let local_module = lifecycle.get("local_module").and_then(Value::as_bool) == Some(true);
    let packaged = !local_module
        && (visibility_state == "packaged"
            || lifecycle.get("runtime_installed").and_then(Value::as_bool) == Some(false));
    let explicit_or_assigned = business_os_explicit_or_assigned_module_permission_allows(
        conn,
        actor_id,
        actor_role,
        BusinessOsPermission::AppsView,
        module_id,
    )?;
    let (allowed, reason_code, display_reason, source) = if local_module {
        (
            true,
            "local_instance_app",
            "Diese App ist ausschließlich auf dieser CTOX-Instanz installiert.",
            "lifecycle_local_instance",
        )
    } else if packaged {
        (
            true,
            "packaged_app",
            "System-Apps sind außerhalb des Runtime-App-Lifecycle sichtbar.",
            "lifecycle_packaged",
        )
    } else if public {
        (
            true,
            "team_visible_version",
            "Apps ab Version 1.0.0 sind standardmäßig für das Team sichtbar.",
            "lifecycle_public_team_version",
        )
    } else if explicit_or_assigned {
        (
            true,
            "explicit_or_responsible_app_view",
            "Diese nicht öffentliche App ist durch App-Verantwortung oder explizite App-Freigabe sichtbar.",
            "explicit_app_view_or_app_responsibility",
        )
    } else {
        (
            false,
            "non_public_without_app_view",
            "Diese App ist nicht team-sichtbar; es fehlt App-Verantwortung oder eine explizite App-Freigabe.",
            "lifecycle_non_public",
        )
    };

    Ok(serde_json::json!({
        "allowed": allowed,
        "permission": BusinessOsPermission::AppsView.as_str(),
        "scope_type": BusinessOsScopeType::Module.as_str(),
        "scope_id": module_id,
        "reason_code": reason_code,
        "display_reason": display_reason,
        "requires_approval": false,
        "audit_level": "decision",
        "source": source,
        "lifecycle_state": visibility_state,
    }))
}

fn business_os_why_open_decision(visibility: &Value) -> Value {
    serde_json::json!({
        "allowed": visibility.get("allowed").and_then(Value::as_bool).unwrap_or(false),
        "permission": BusinessOsPermission::AppsView.as_str(),
        "scope_type": BusinessOsScopeType::Module.as_str(),
        "scope_id": visibility.get("scope_id").cloned().unwrap_or(Value::Null),
        "reason_code": visibility
            .get("reason_code")
            .and_then(Value::as_str)
            .unwrap_or("visibility_decision"),
        "display_reason": visibility
            .get("display_reason")
            .and_then(Value::as_str)
            .unwrap_or("App öffnen folgt der App-Sichtbarkeit."),
        "requires_approval": false,
        "audit_level": "decision",
        "source": "visibility_decision",
    })
}

fn business_os_why_module_policy_decision(
    conn: &Connection,
    actor_id: &str,
    actor_role: &str,
    permission: BusinessOsPermission,
    module_id: &str,
    source: &str,
) -> anyhow::Result<Value> {
    let decision = trusted_actor_policy_decision_with_conn(
        conn,
        actor_id,
        actor_role,
        permission,
        BusinessOsScopeType::Module,
        Some(module_id),
    )?;
    Ok(business_os_why_policy_payload(&decision, source))
}

fn business_os_why_data_area_diagnostics(
    conn: &Connection,
    actor_id: &str,
    actor_role: &str,
    module_id: &str,
    module: &Value,
    lifecycle: &Value,
) -> anyhow::Result<Vec<Value>> {
    let data_access = lifecycle.get("data_access").unwrap_or(&Value::Null);
    let mut collections = BTreeSet::new();
    business_os_collect_json_string_array(module.get("collections"), &mut collections);
    business_os_collect_json_string_array(
        data_access.get("granted_collection_ids"),
        &mut collections,
    );
    business_os_collect_json_string_array(
        data_access.get("locked_collection_ids"),
        &mut collections,
    );
    if let Some(areas) = data_access.get("areas").and_then(Value::as_array) {
        for area in areas {
            if let Some(collection) = area.get("collection").and_then(Value::as_str) {
                let collection = collection.trim();
                if !collection.is_empty() {
                    collections.insert(collection.to_owned());
                }
            }
        }
    }

    let mut areas = Vec::new();
    for collection in collections {
        areas.push(serde_json::json!({
            "collection": collection,
            "read_review_state": business_os_data_review_state(data_access, collection.as_str(), "read"),
            "write_review_state": business_os_data_review_state(data_access, collection.as_str(), "write"),
            "read_decision": business_os_why_data_permission_decision(
                conn,
                actor_id,
                actor_role,
                module_id,
                collection.as_str(),
                BusinessOsPermission::DataRead,
            )?,
            "write_decision": business_os_why_data_permission_decision(
                conn,
                actor_id,
                actor_role,
                module_id,
                collection.as_str(),
                BusinessOsPermission::DataWrite,
            )?,
        }));
    }
    Ok(areas)
}

fn business_os_why_data_permission_decision(
    conn: &Connection,
    actor_id: &str,
    actor_role: &str,
    module_id: &str,
    collection: &str,
    permission: BusinessOsPermission,
) -> anyhow::Result<Value> {
    let collection_decision = trusted_actor_policy_decision_with_conn(
        conn,
        actor_id,
        actor_role,
        permission,
        BusinessOsScopeType::Collection,
        Some(collection),
    )?;
    let module_decision = trusted_actor_policy_decision_with_conn(
        conn,
        actor_id,
        actor_role,
        permission,
        BusinessOsScopeType::Module,
        Some(module_id),
    )?;
    let (effective, source) = if collection_decision.allowed {
        (&collection_decision, "collection_policy")
    } else if module_decision.allowed {
        (&module_decision, "declared_module_data_policy")
    } else {
        (&module_decision, "denied_collection_and_module_policy")
    };
    let mut payload = business_os_why_policy_payload(effective, source);
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "collection_decision".to_owned(),
            policy_decision_payload(&collection_decision),
        );
        object.insert(
            "module_decision".to_owned(),
            policy_decision_payload(&module_decision),
        );
    }
    Ok(payload)
}

fn business_os_explicit_or_assigned_module_permission_allows(
    conn: &Connection,
    actor_id: &str,
    actor_role: &str,
    permission: BusinessOsPermission,
    module_id: &str,
) -> anyhow::Result<bool> {
    let actor = BusinessOsActor::new(Some(actor_id.to_owned()), actor_role);
    let unassigned_scope = BusinessOsScope::module(module_id, false);
    if active_permission_grant_allows(conn, &actor, permission, &unassigned_scope)? {
        return Ok(true);
    }
    if founder_owns_module(conn, module_id, actor_id)? {
        let assigned_actor = BusinessOsActor::new(Some(actor_id.to_owned()), "founder");
        let assigned_scope = BusinessOsScope::module(module_id, true);
        return Ok(policy::evaluate(&assigned_actor, permission, &assigned_scope).allowed);
    }
    Ok(false)
}

fn business_os_collect_json_string_array(value: Option<&Value>, output: &mut BTreeSet<String>) {
    if let Some(items) = value.and_then(Value::as_array) {
        for item in items {
            if let Some(text) = item.as_str() {
                let text = text.trim();
                if !text.is_empty() {
                    output.insert(text.to_owned());
                }
            }
        }
    }
}

fn business_os_data_review_state(data_access: &Value, collection: &str, key: &str) -> String {
    data_access
        .get("areas")
        .and_then(Value::as_array)
        .and_then(|areas| {
            areas.iter().find(|area| {
                area.get("collection")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value == collection)
            })
        })
        .and_then(|area| area.get(key))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn business_os_why_policy_payload(decision: &PolicyDecision, source: &str) -> Value {
    let mut payload = policy_decision_payload(decision);
    if let Some(object) = payload.as_object_mut() {
        object.insert("source".to_owned(), Value::String(source.to_owned()));
    }
    payload
}
