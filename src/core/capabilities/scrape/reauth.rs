// Session-expiry reauthorization: protected-capture config resolution,
// login-landing detection, and the typed auth-assist handoff (source id,
// safe login URL, allowed domains, credential REFERENCES only).

use super::classify::{Classification, ScrapeRunStatus};
use super::execute::ProbeResult;
use super::{host_within_domains, url_host_lower, RegisteredTarget};
use anyhow::Result;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub(super) struct ProtectedCaptureConfig {
    pub(super) login_url: Option<String>,
    pub(super) allowed_domains: Vec<String>,
    pub(super) credential_ref: Option<String>,
}

pub(super) fn js_object_string_field(body: &str, key: &str) -> Option<String> {
    let marker = format!("{key}: \"");
    let start = body.find(&marker)? + marker.len();
    let end = body[start..].find('"')? + start;
    Some(body[start..end].to_string())
}

pub(super) fn js_object_string_list(body: &str, key: &str) -> Vec<String> {
    let marker = format!("{key}: [");
    let Some(start) = body.find(&marker).map(|i| i + marker.len()) else {
        return Vec::new();
    };
    let Some(end) = body[start..].find(']').map(|i| i + start) else {
        return Vec::new();
    };
    body[start..end]
        .split(',')
        .filter_map(|item| {
            let item = item.trim().trim_matches('"');
            (!item.is_empty()).then(|| item.to_string())
        })
        .collect()
}

/// Parse the `PROTECTED_SOURCE_CONFIG` entry for `source_id` out of the
/// registered adapter script body — the same static contract the unlock
/// acceptance report validates. Entry bodies contain arrays but no nested
/// braces, so a brace-delimited slice is sufficient.
pub(super) fn protected_config_from_script(
    script: &str,
    source_id: &str,
) -> Option<ProtectedCaptureConfig> {
    let marker = "const PROTECTED_SOURCE_CONFIG = Object.freeze({";
    let start = script.find(marker)? + marker.len();
    let rest = &script[start..];
    let end = rest.find("\n});").unwrap_or(rest.len());
    let section = &rest[..end];
    let entry_marker = format!("\"{source_id}\": {{");
    let entry_start = section.find(&entry_marker)? + entry_marker.len();
    let entry_end = section[entry_start..].find('}')? + entry_start;
    let body = &section[entry_start..entry_end];
    let config = ProtectedCaptureConfig {
        login_url: js_object_string_field(body, "login_url"),
        allowed_domains: js_object_string_list(body, "allowed_domains"),
        credential_ref: js_object_string_field(body, "credential_ref"),
    };
    (config.login_url.is_some() || !config.allowed_domains.is_empty()).then_some(config)
}

/// A credential reference is only ever `ctox-secret://<scope>/<NAME>` — never
/// a raw value, never userinfo/query/fragment.
pub(super) fn valid_credential_reference(value: &str) -> bool {
    let prefix = format!("ctox-secret://{}/", crate::secrets::credential_scope());
    let Some(name) = value.strip_prefix(prefix.as_str()) else {
        return false;
    };
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

/// Resolve the protected-capture config for a target's `expected_provider`.
/// The compiled-in Rust `browser_recipe()` is authoritative when the source
/// module exists; the registered adapter script's PROTECTED_SOURCE_CONFIG
/// entry fills the gaps (and is the only source for script-only adapters).
/// The result is validated: https login URL inside the allow-list, and a
/// secret-store credential reference — runner scripts are hot-revisable and
/// therefore untrusted, so nothing is taken from the run payload.
pub(super) fn resolve_protected_capture_config(
    target: &RegisteredTarget,
) -> Option<(String, ProtectedCaptureConfig)> {
    let provider = target
        .view
        .config
        .get("expected_provider")
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    if provider.is_empty() {
        return None;
    }
    let recipe =
        ctox_web_stack::sources::find(&provider).and_then(|module| module.browser_recipe());
    let script_config = fs::read_to_string(&target.script.script_path)
        .ok()
        .and_then(|body| protected_config_from_script(&body, &provider));
    if recipe.is_none() && script_config.is_none() {
        return None;
    }
    let mut config = ProtectedCaptureConfig::default();
    if let Some(script_config) = script_config {
        config = script_config;
    }
    if let Some(recipe) = recipe {
        // The compiled-in recipe wins on every field it carries.
        config.login_url = Some(recipe.login_url.clone());
        if !recipe.allowed_domains.is_empty() {
            config.allowed_domains = recipe.allowed_domains.clone();
        }
        if let Some(secret_name) = recipe.required_secret_name {
            config.credential_ref = Some(format!(
                "ctox-secret://{}/{}",
                crate::secrets::credential_scope(),
                secret_name
            ));
        }
    }
    let login_url = config.login_url.clone()?;
    if !login_url.starts_with("https://") {
        return None;
    }
    if let Some(host) = url_host_lower(&login_url) {
        if !config.allowed_domains.is_empty()
            && !host_within_domains(&host, &config.allowed_domains)
        {
            return None;
        }
    }
    if let Some(reference) = &config.credential_ref {
        if !valid_credential_reference(reference) {
            config.credential_ref = None;
        }
    }
    Some((provider, config))
}

/// True when `final_url` is the source's own login page: same host as the
/// configured login URL with a matching (or root) login path, or a URL on an
/// allowed domain whose host/path is unambiguously a login endpoint
/// (`login.<domain>`, `/login`, `/sign/in`, `/anmeldung`, ...).
pub(super) fn url_is_login_landing(
    final_url: &str,
    login_url: &str,
    allowed_domains: &[String],
) -> bool {
    let Some(host) = url_host_lower(final_url) else {
        return false;
    };
    let path = url::Url::parse(final_url)
        .map(|parsed| parsed.path().to_ascii_lowercase())
        .unwrap_or_default();
    let login_host = url_host_lower(login_url);
    let login_path = url::Url::parse(login_url)
        .map(|parsed| parsed.path().trim_end_matches('/').to_ascii_lowercase())
        .unwrap_or_default();
    if login_host.as_deref() == Some(host.as_str())
        && (login_path.is_empty() || path.starts_with(login_path.as_str()))
    {
        return true;
    }
    host_within_domains(&host, allowed_domains)
        && (host.starts_with("login.")
            || path.contains("login")
            || path.contains("sign/in")
            || path.contains("sign-in")
            || path.contains("anmeld"))
}

/// Build the typed reauthorization action for a run when the evidence shows
/// an expired/invalid session on a credential-protected source: either the
/// adapter script classified `authorization_required` explicitly, or the
/// portal probe landed on the source's own login page while the run drifted
/// (which is how an expired stored session presents — the login redirect is
/// not layout drift). Returns the action payload to persist and to hand off.
pub(super) fn session_expiry_reauthorization(
    target: &RegisteredTarget,
    probe: &ProbeResult,
    payload: &Value,
    classification: &Classification,
) -> Option<Value> {
    let explicit =
        payload.get("failure_mode").and_then(Value::as_str) == Some("authorization_required");
    let login_landing_upgrade = classification.status == ScrapeRunStatus::PortalDrift
        && probe.reachable
        && !probe.human_verification;
    if !explicit && !login_landing_upgrade {
        return None;
    }
    let (provider, config) = resolve_protected_capture_config(target)?;
    let login_url = config.login_url.clone()?;
    if !explicit && !url_is_login_landing(&probe.final_url, &login_url, &config.allowed_domains) {
        return None;
    }
    Some(json!({
        "kind": "auth-assist-request",
        "source_id": provider,
        "login_url": login_url,
        "allowed_domains": config.allowed_domains,
        "credential_ref": config.credential_ref,
        "reason": "session_expired_or_invalid",
        "secret_value_in_payload": false,
    }))
}

/// Emit the typed `auth-assist-request` handoff through the existing
/// Business OS web-stack mechanism (the same enqueue the adapter scripts and
/// the `ctox business-os web-stack auth-assist-request` CLI use). Lossy by
/// design: a handoff failure is reported in the run error, never fatal.
pub(super) fn emit_reauthorization_handoff(
    root: &Path,
    run_id: &str,
    thread_key: Option<&str>,
    reauthorization: &Value,
) -> Option<Value> {
    let source_id = reauthorization.get("source_id")?.as_str()?.to_string();
    let login_url = reauthorization.get("login_url")?.as_str()?.to_string();
    let credential_ref = reauthorization
        .get("credential_ref")
        .and_then(Value::as_str)
        .map(str::to_string);
    let task_id = thread_key.unwrap_or(run_id).to_string();
    match crate::service::business_os::enqueue_web_stack_auth_assist_request(
        root,
        &source_id,
        Some(&login_url),
        credential_ref.as_deref(),
        Some("stored session expired or invalid; reauthorization required"),
        Some(reauthorization),
        &task_id,
        "scrape_executor",
        "ctox scrape execute",
        None,
        !task_id.trim().is_empty(),
        true,
    ) {
        Ok(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert("kind".to_string(), json!("auth-assist-request"));
                object.insert("run_id".to_string(), json!(run_id));
            }
            Some(value)
        }
        Err(err) => {
            eprintln!("scrape execute: auth-assist handoff failed: {err:#}");
            None
        }
    }
}
