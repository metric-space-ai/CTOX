//! P2: native owner of the pi-code coding sidecar
//! (`src/core/coding_agents/pi-sidecar`). Spawns the LocalTransport daemon and
//! drives one bounded turn over a Unix socket, then reaps it.
//!
//! This is the transport client the higher-level owner uses: it projects a
//! module's app source into a `CtoxTurnRequest.files` snapshot, runs one bounded
//! turn, and reads back the `CtoxTurnResponse` snapshot to record as P0 commits.
//! The sidecar is a bounded leaf executor — a fresh daemon per turn, killed on
//! drop; it never shares the daemon's process authority with the CTOX daemon.
use anyhow::Context;
use serde_json::Value;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[path = "anthropic_coding_bridge.rs"]
pub(crate) mod anthropic_coding_bridge;

use anthropic_coding_bridge::{AnthropicCodingBridge, BRIDGE_TOKEN_HEADER};

struct PreparedCodingTurnModel {
    model: Value,
    coding_plan_bridge: Option<AnthropicCodingBridge>,
    provider: String,
    model_id: String,
    account_id: Option<String>,
}

/// Path to the built sidecar bundle relative to the repo root (dev / tests).
pub fn sidecar_dist_path(repo_root: &Path) -> PathBuf {
    repo_root.join("src/core/coding_agents/pi-sidecar/dist/ctox-pi-sidecar.mjs")
}

/// The pi-sidecar bundle is embedded into the ctox binary at build time so a
/// deployed CTOX ships as one artifact (no source tree). Build order: the
/// sidecar bundle (`npm run build` in pi-sidecar) must exist before `cargo
/// build`; a CI/build step guarantees this.
const SIDECAR_BUNDLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/core/coding_agents/pi-sidecar/dist/ctox-pi-sidecar.mjs"
));

/// Resolve a runnable sidecar bundle path for `root`, extracting the embedded
/// bytes to `<root>/coding-agents/ctox-pi-sidecar.mjs` when missing or
/// size-mismatched. An explicit `CTOX_PI_SIDECAR_DIST` override wins (dev /
/// custom deployments). This is the runtime resolver the owner uses.
pub fn resolve_sidecar_dist(root: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(override_path) = std::env::var("CTOX_PI_SIDECAR_DIST") {
        let path = PathBuf::from(override_path);
        anyhow::ensure!(
            path.exists(),
            "CTOX_PI_SIDECAR_DIST does not exist: {}",
            path.display()
        );
        return Ok(path);
    }
    let dir = root.join("coding-agents");
    std::fs::create_dir_all(&dir).context("create sidecar runtime dir")?;
    let path = dir.join("ctox-pi-sidecar.mjs");
    let needs_write = match std::fs::metadata(&path) {
        Ok(meta) => meta.len() != SIDECAR_BUNDLE.len() as u64,
        Err(_) => true,
    };
    if needs_write {
        std::fs::write(&path, SIDECAR_BUNDLE).context("extract embedded sidecar bundle")?;
    }
    Ok(path)
}

/// The Business OS app skill: the system prompt that teaches the coding agent
/// how Business OS app modules are structured (module.json, `mount(ctx)`, the
/// shared kit, RxDB/WebRTC data boundary, command dispatch). This is
/// CTOX-specific knowledge, so it lives in the owner and is injected per turn —
/// the sidecar port stays a generic pi engine.
const BUSINESS_OS_APP_SKILL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/core/coding_agents/business-os-app-skill.md"
));

/// The coding agent's system prompt for a Business OS module turn: the app skill
/// plus the minimal tool-usage footer (the workspace is an in-memory projection
/// of the module's synced source, edited through the pi tools — no host FS).
pub fn business_os_system_prompt() -> String {
    format!(
        "{skill}\n\n## Tools and workspace\n\nUse the pi coding tools (read, \
edit, write, grep, find, ls) to inspect and change files. The filesystem is an \
isolated in-memory projection of this module's synced source — not the host \
filesystem. Make changes through write/edit; your edits are applied back as \
versioned commits to the module source.",
        skill = BUSINESS_OS_APP_SKILL
    )
}

/// Build the pi model config that points the sidecar's stream at the local CTOX
/// model gateway — a loopback Responses HTTP server. The provider api is pi-ai's
/// OpenAI Responses provider (`openai-responses`); the exact protocol + auth
/// match is confirmed by a real turn against the running gateway (decision-1).
pub fn gateway_model(root: &Path) -> Value {
    let gateway = crate::execution::responses::gateway::GatewayConfig::resolve_with_root(root);
    let base_url = format!("http://{}:{}/v1", gateway.listen_host, gateway.listen_port);
    let model_id = gateway
        .active_model
        .unwrap_or_else(|| "ctox-gateway".to_string());
    serde_json::json!({
        "id": model_id,
        "name": "CTOX Model Gateway",
        "api": "openai-responses",
        "provider": "ctox-gateway",
        "baseUrl": base_url,
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 0,
        "maxTokens": 0
    })
}

/// The coding default inherits CTOX's active main model through the gateway.
/// Provider credentials and routing remain server-side; callers may still
/// supply an explicit typed pi-ai model override.
pub fn coding_default_model(root: &Path) -> Value {
    gateway_model(root)
}

/// Server-authoritative public capability document. The browser receives no
/// credentials and may select the subscription route only after the owning
/// same-root listener is actually ready.
pub fn coding_model_capabilities(root: &Path) -> Value {
    let model = gateway_model(root);
    let active_model = model
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("ctox-gateway");
    let mut presets = vec![serde_json::json!({
        "id": "ctox",
        "label": format!("CTOX (Standard · {active_model})"),
        "default": true,
        "model": Value::Null,
    })];
    for route in crate::execution::cliproxyapi_host::instance_proxy_route_capabilities(root) {
        let provider = route.provider;
        let route_model = route.model;
        presets.push(serde_json::json!({
            "id": format!("{provider}-subscription-{route_model}"),
            "label": format!("{provider} subscription · {route_model}"),
            "default": false,
            "model": {
                "id": route_model,
                "name": format!("{route_model} via {provider} subscription"),
                "api": "openai-responses",
                "provider": "ctox-gateway",
                "baseUrl": crate::execution::cliproxyapi_host::instance_codex_proxy_base_url(),
                "headers": { "X-CTOX-Provider": provider },
                "reasoning": false,
                "input": ["text"],
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                "contextWindow": 0,
                "maxTokens": 0
            }
        }));
    }
    // MiniMax Coding Plan is an independent direct account route. It does not
    // inherit or mutate CTOX's main provider and is never represented as
    // `ctox_proxy`. Only accounts whose encrypted secret handle resolves are
    // advertised.
    for account in
        crate::execution::models::minimax_coding::ready_accounts(root).unwrap_or_default()
    {
        for route_model in account.effective_models() {
            presets.push(serde_json::json!({
                // Length-prefixing makes the opaque id injective without
                // parsing provider/model identity back out of user input.
                "id": format!("minimax-coding-{}-{}-{route_model}", account.id.len(), account.id),
                "label": format!("MiniMax Coding Plan · {} · {route_model}", account.id),
                "default": false,
                "model": {
                    "id": route_model,
                    "name": format!("{route_model} via MiniMax Coding Plan ({})", account.id),
                    "api": "anthropic-messages",
                    "provider": "ctox-minimax-coding",
                    // Replaced with a fresh turn-scoped loopback bridge after
                    // server-authoritative preset resolution.
                    "baseUrl": "http://127.0.0.1:1",
                    "reasoning": true,
                    "input": ["text"],
                    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                    "contextWindow": 204800,
                    "maxTokens": 8192,
                    "ctoxRoute": {
                        "kind": "minimax_coding_plan",
                        "accountId": account.id,
                    }
                }
            }));
        }
    }
    // Kimi Coding Plan is likewise account-scoped and independent from both
    // the Kimi subscription route and CTOX's main model.
    for account in crate::execution::models::kimi_coding::ready_accounts(root).unwrap_or_default() {
        for route_model in account.effective_models() {
            presets.push(serde_json::json!({
                "id": format!("kimi-coding-{}-{}-{route_model}", account.id.len(), account.id),
                "label": format!("Kimi Coding Plan · {} · {route_model}", account.id),
                "default": false,
                "model": {
                    "id": route_model,
                    "name": format!("{route_model} via Kimi Coding Plan ({})", account.id),
                    "api": "anthropic-messages",
                    "provider": "ctox-kimi-coding",
                    "baseUrl": "http://127.0.0.1:1",
                    "reasoning": true,
                    "input": ["text"],
                    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                    "contextWindow": 1048576,
                    "maxTokens": 8192,
                    "ctoxRoute": {
                        "kind": "kimi_coding_plan",
                        "accountId": account.id,
                    }
                }
            }));
        }
    }
    serde_json::json!({
        "schema": "ctox.coding.models.v1",
        "default": {
            "mode": "inherit_ctox",
            "provider": "ctox-gateway",
            "model": active_model,
        },
        "presets": presets,
    })
}

/// Resolve one server-authored preset immediately before a turn. Business OS
/// sends only the opaque preset ID; provider URLs, routing headers and model
/// objects are never accepted from the browser command payload.
pub fn resolve_coding_model_preset(root: &Path, preset_id: &str) -> anyhow::Result<Option<Value>> {
    let preset_id = preset_id.trim();
    anyhow::ensure!(!preset_id.is_empty(), "coding model preset_id is required");
    let capabilities = coding_model_capabilities(root);
    let presets = capabilities
        .get("presets")
        .and_then(Value::as_array)
        .context("coding model capabilities are malformed")?;
    let mut matching = presets
        .iter()
        .filter(|preset| preset.get("id").and_then(Value::as_str) == Some(preset_id));
    let preset = matching
        .next()
        .with_context(|| format!("coding model preset is unavailable: {preset_id}"))?;
    anyhow::ensure!(
        matching.next().is_none(),
        "coding model preset is ambiguous"
    );
    match preset.get("model") {
        None | Some(Value::Null) => Ok(None),
        Some(model @ Value::Object(_)) => Ok(Some(model.clone())),
        Some(_) => anyhow::bail!("coding model preset is malformed"),
    }
}

/// A spawned sidecar daemon listening on a Unix socket. Killed + cleaned on drop
/// so a turn can never leak a live agent process.
struct SidecarDaemon {
    child: Child,
    socket_path: PathBuf,
}

impl Drop for SidecarDaemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

fn spawn_sidecar(dist: &Path, socket_path: &Path, faux: bool) -> anyhow::Result<SidecarDaemon> {
    anyhow::ensure!(
        dist.exists(),
        "pi-sidecar bundle is not built: {} (run `npm run build` in pi-sidecar)",
        dist.display()
    );
    let mut command = Command::new("node");
    command
        .arg(dist)
        .arg(socket_path)
        // Sandbox invariant: the sidecar is a bounded leaf executor whose rights
        // must be a strict SUBSET of the CTOX daemon's. It must NOT inherit the
        // daemon's environment (secret store, tokens, state-root paths). Start
        // from an empty env and grant only PATH (needed to resolve `node`) plus
        // the flags the turn needs; a real turn adds ONLY the gateway auth here.
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if faux {
        command.env("CTOX_PI_SIDECAR_FAUX", "1");
    }
    let child = command
        .spawn()
        .context("spawn pi-sidecar daemon (is `node` on PATH?)")?;
    Ok(SidecarDaemon {
        child,
        socket_path: socket_path.to_path_buf(),
    })
}

fn connect_with_retry(socket_path: &Path, timeout: Duration) -> anyhow::Result<UnixStream> {
    let deadline = Instant::now() + timeout;
    loop {
        match UnixStream::connect(socket_path) {
            Ok(stream) => return Ok(stream),
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                return Err(error).context("connect to pi-sidecar socket");
            }
        }
    }
}

const MAX_PI_TURN_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_PI_TURN_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const PI_TURN_TIMEOUT: Duration = Duration::from_secs(600);

fn read_line(stream: &mut UnixStream, max_bytes: usize) -> anyhow::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let read = stream.read(&mut byte).context("read turn response")?;
        if read == 0 || byte[0] == b'\n' {
            break;
        }
        buffer.push(byte[0]);
        anyhow::ensure!(
            buffer.len() <= max_bytes,
            "sidecar turn response is too large"
        );
    }
    Ok(buffer)
}

/// Run one bounded turn through a freshly spawned sidecar daemon: send `request`
/// (a `CtoxTurnRequest` JSON), return the `CtoxTurnResponse` JSON. `faux` runs
/// the sidecar's offline no-model mode (owner integration tests).
pub fn run_pi_turn(dist: &Path, request: &Value, faux: bool) -> anyhow::Result<Value> {
    let socket_path = std::env::temp_dir().join(format!("ctox-pi-{}.sock", Uuid::new_v4()));
    let _daemon = spawn_sidecar(dist, &socket_path, faux)?;
    let mut stream = connect_with_retry(&socket_path, Duration::from_secs(10))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .context("set pi-sidecar write timeout")?;
    stream
        .set_read_timeout(Some(PI_TURN_TIMEOUT))
        .context("set pi-sidecar turn timeout")?;

    let mut line = serde_json::to_string(request).context("serialize turn request")?;
    anyhow::ensure!(
        line.len() < MAX_PI_TURN_REQUEST_BYTES,
        "sidecar turn request is too large"
    );
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .context("write turn request")?;
    stream.flush().ok();

    let response_bytes = read_line(&mut stream, MAX_PI_TURN_RESPONSE_BYTES)?;
    anyhow::ensure!(
        !response_bytes.is_empty(),
        "sidecar closed without a response"
    );
    let response: Value =
        serde_json::from_slice(&response_bytes).context("parse turn response JSON")?;
    Ok(response)
}

/// Project a module's synced app source (`business_module_source_files` records)
/// into a `{path -> content}` map for a `CtoxTurnRequest.files` snapshot. This is
/// the app-source-projection workspace model: the sidecar edits a materialized
/// view of the source records; its writes come back as P0 commits. No host FS.
pub fn project_module_source(
    root: &Path,
    module_id: &str,
) -> anyhow::Result<serde_json::Map<String, Value>> {
    let records = crate::business_os::store::pull_collection_records(
        root,
        "business_module_source_files",
        None,
        None,
    )?;
    let mut files = serde_json::Map::new();
    if let Some(documents) = records.get("documents").and_then(Value::as_array) {
        for document in documents {
            if document.get("module_id").and_then(Value::as_str) != Some(module_id) {
                continue;
            }
            if document.get("_deleted").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            let (Some(path), Some(content)) = (
                document.get("path").and_then(Value::as_str),
                document.get("content").and_then(Value::as_str),
            ) else {
                continue;
            };
            files.insert(path.to_string(), Value::String(content.to_string()));
        }
    }
    Ok(files)
}

/// Apply a turn's returned snapshot back into the module's app source. Each file
/// is written through the same policy-gated source path that records P0
/// versions/commits — the agent proposed, the trusted owner disposes. The
/// sidecar env cwd prefix (`/workspace/`) is stripped to the module-relative
/// path. Returns the paths written.
pub fn apply_turn_snapshot(
    root: &Path,
    module_id: &str,
    snapshot: &[Value],
) -> anyhow::Result<Vec<String>> {
    let mut applied = Vec::new();
    for entry in snapshot {
        if entry.get("kind").and_then(Value::as_str) != Some("file") {
            continue;
        }
        let Some(raw_path) = entry.get("path").and_then(Value::as_str) else {
            continue;
        };
        let path = raw_path
            .strip_prefix("/workspace/")
            .unwrap_or_else(|| raw_path.trim_start_matches('/'));
        let Some(content) = entry.get("content").and_then(Value::as_str) else {
            continue;
        };
        crate::business_os::store::save_module_source_record(
            root,
            crate::business_os::store::ModuleSourceSaveMutation {
                module_id: module_id.to_string(),
                path: path.to_string(),
                content: content.to_string(),
            },
        )?;
        applied.push(path.to_string());
    }
    Ok(applied)
}

/// The owner's core delegation primitive: one bounded coding turn against a
/// module's app source. Project the source into the request, run the pi turn
/// through the sidecar (`faux` = offline no-model), then apply the resulting
/// snapshot back into the source (recording P0 versions). Returns a summary.
pub fn run_module_coding_turn(
    root: &Path,
    dist: &Path,
    module_id: &str,
    prompt: &str,
    faux: bool,
    model_override: Option<Value>,
) -> anyhow::Result<Value> {
    run_module_coding_turn_inner(root, dist, module_id, prompt, faux, model_override, None)
}

fn prepare_coding_turn_model(
    root: &Path,
    model_override: Option<Value>,
    coding_plan_upstream_override: Option<&str>,
    require_unique_subscription_account: bool,
) -> anyhow::Result<PreparedCodingTurnModel> {
    let mut model = model_override.unwrap_or_else(|| coding_default_model(root));
    let model_id = model
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("coding route is missing model id")?
        .to_owned();
    let route_kind = model
        .pointer("/ctoxRoute/kind")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut coding_plan_bridge = None;
    let (provider, account_id) = if matches!(
        route_kind.as_deref(),
        Some("minimax_coding_plan" | "kimi_coding_plan")
    ) {
        let account_id = model
            .pointer("/ctoxRoute/accountId")
            .and_then(Value::as_str)
            .context("coding-plan route is missing accountId")?
            .to_owned();
        let (provider, api_key, configured_upstream_base_url) = match route_kind.as_deref() {
            Some("minimax_coding_plan") => {
                anyhow::ensure!(
                    model.get("provider").and_then(Value::as_str) == Some("ctox-minimax-coding")
                        && model.get("api").and_then(Value::as_str) == Some("anthropic-messages"),
                    "MiniMax coding route is malformed"
                );
                let account = crate::execution::models::minimax_coding::resolve_ready_account(
                    root,
                    &account_id,
                )?;
                anyhow::ensure!(
                    account
                        .effective_models()
                        .iter()
                        .any(|allowed| allowed == &model_id),
                    "MiniMax coding model is not allowed for account {account_id}"
                );
                (
                    "minimax_coding_plan".to_owned(),
                    crate::execution::models::minimax_coding::read_api_key(root, &account)?,
                    account.endpoint_profile.base_url(),
                )
            }
            Some("kimi_coding_plan") => {
                anyhow::ensure!(
                    model.get("provider").and_then(Value::as_str) == Some("ctox-kimi-coding")
                        && model.get("api").and_then(Value::as_str) == Some("anthropic-messages"),
                    "Kimi coding route is malformed"
                );
                let account = crate::execution::models::kimi_coding::resolve_ready_account(
                    root,
                    &account_id,
                )?;
                anyhow::ensure!(
                    account
                        .effective_models()
                        .iter()
                        .any(|allowed| allowed == &model_id),
                    "Kimi coding model is not allowed for account {account_id}"
                );
                (
                    "kimi_coding_plan".to_owned(),
                    crate::execution::models::kimi_coding::read_api_key(root, &account)?,
                    account.endpoint_profile.base_url(),
                )
            }
            _ => unreachable!("route kind was checked above"),
        };
        let upstream_base_url =
            coding_plan_upstream_override.unwrap_or(configured_upstream_base_url);
        let bridge = AnthropicCodingBridge::spawn(api_key, upstream_base_url)?;
        model["baseUrl"] = Value::String(bridge.base_url().to_owned());
        model["headers"] = serde_json::json!({
            (BRIDGE_TOKEN_HEADER): bridge.capability_token(),
        });
        // The account selector has already been consumed by the native owner.
        // Do not project it into the less-privileged sidecar request.
        if let Some(object) = model.as_object_mut() {
            object.remove("ctoxRoute");
        }
        coding_plan_bridge = Some(bridge);
        (provider, Some(account_id))
    } else if let Some(provider) = model
        .pointer("/headers/X-CTOX-Provider")
        .and_then(Value::as_str)
    {
        let provider = provider.trim().to_ascii_lowercase();
        let account_id = require_unique_subscription_account
            .then(|| {
                crate::execution::cliproxyapi_host::unique_instance_proxy_account_for_route(
                    root, &provider, &model_id,
                )
            })
            .transpose()?;
        (provider, account_id)
    } else {
        (
            model
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("ctox-gateway")
                .to_owned(),
            None,
        )
    };

    Ok(PreparedCodingTurnModel {
        model,
        coding_plan_bridge,
        provider,
        model_id,
        account_id,
    })
}

/// Execute an operator-requested, bounded live smoke without reading or
/// mutating Business OS source. The opaque preset is resolved immediately
/// before the turn; the returned evidence contains no URL, account identifier,
/// credential, bridge capability or model response text.
pub fn run_coding_preset_smoke(
    root: &Path,
    dist: &Path,
    preset_id: &str,
    prompt: Option<&str>,
) -> anyhow::Result<Value> {
    run_coding_preset_smoke_inner(root, dist, preset_id, prompt, None, None)
}

fn run_coding_preset_smoke_inner(
    root: &Path,
    dist: &Path,
    preset_id: &str,
    prompt: Option<&str>,
    coding_plan_upstream_override: Option<&str>,
    subscription_proxy_override: Option<&str>,
) -> anyhow::Result<Value> {
    let model = resolve_coding_model_preset(root, preset_id)?
        .context("the inherited CTOX preset is not an independent provider smoke target")?;
    let main_model_before = crate::inference::runtime_env::effective_chat_model(root);
    let mut prepared =
        prepare_coding_turn_model(root, Some(model), coding_plan_upstream_override, true)?;
    if let Some(proxy_base_url) = subscription_proxy_override {
        anyhow::ensure!(
            prepared
                .model
                .pointer("/headers/X-CTOX-Provider")
                .and_then(Value::as_str)
                .is_some(),
            "subscription proxy override requires a native subscription preset"
        );
        prepared.model["baseUrl"] = Value::String(proxy_base_url.to_owned());
    }
    let account_id = prepared
        .account_id
        .as_deref()
        .context("coding preset is not bound to a provider account")?;
    let account_digest = ring::digest::digest(&ring::digest::SHA256, account_id.as_bytes());
    let account_id_sha256 = account_digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let request = serde_json::json!({
        "id": "operator-provider-smoke",
        "prompt": prompt.unwrap_or(
            "Edit index.js and change only the exported value from 1 to 2. Use the edit tool."
        ),
        "files": { "index.js": "export const v = 1;\n" },
        "tools": ["read", "edit"],
        "maxAssistantTurns": 4,
        "systemPrompt": "This is a bounded provider-route smoke. Edit only index.js in the in-memory workspace.",
        "model": prepared.model,
    });
    let response = run_pi_turn(dist, &request, false)?;
    #[cfg(test)]
    drop(prepared.coding_plan_bridge);
    anyhow::ensure!(
        response.get("ok").and_then(Value::as_bool) == Some(true),
        "provider smoke Pi turn failed"
    );
    let edited = response
        .get("snapshot")
        .and_then(Value::as_array)
        .and_then(|entries| {
            entries.iter().find(|entry| {
                entry.get("kind").and_then(Value::as_str) == Some("file")
                    && entry
                        .get("path")
                        .and_then(Value::as_str)
                        .is_some_and(|path| path.ends_with("/index.js"))
            })
        })
        .and_then(|entry| entry.get("content"))
        .and_then(Value::as_str)
        .context("provider smoke did not return index.js")?;
    let compact = edited
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    anyhow::ensure!(
        compact.contains("v=2"),
        "provider smoke did not apply the expected bounded edit"
    );
    let main_model_after = crate::inference::runtime_env::effective_chat_model(root);
    anyhow::ensure!(
        main_model_after == main_model_before,
        "provider smoke changed the CTOX main model"
    );
    Ok(serde_json::json!({
        "ok": true,
        "schema": "ctox.coding.provider-smoke.v1",
        "provider": prepared.provider,
        "model": prepared.model_id,
        "account_id_sha256": account_id_sha256,
        "main_model_unchanged": true,
        "bounded_edit_verified": true,
    }))
}

/// Test-only listener seam for proving that an opaque subscription preset
/// crosses the real Pi process and the native provider router. Production
/// always resolves the fixed instance-loopback listener.
#[cfg(test)]
pub(crate) fn run_coding_preset_smoke_with_subscription_proxy_for_test(
    root: &Path,
    dist: &Path,
    preset_id: &str,
    proxy_base_url: &str,
) -> anyhow::Result<Value> {
    crate::execution::cliproxyapi_host::mark_instance_codex_proxy_ready_for_test(root);
    run_coding_preset_smoke_inner(root, dist, preset_id, None, None, Some(proxy_base_url))
}

/// Internal owner seam. Production always leaves
/// `coding_plan_upstream_override` unset. Tests use a controlled loopback
/// upstream so the complete preset -> account -> bridge -> Pi route can be
/// proven without adding an ambient runtime toggle or contacting a provider.
fn run_module_coding_turn_inner(
    root: &Path,
    dist: &Path,
    module_id: &str,
    prompt: &str,
    faux: bool,
    model_override: Option<Value>,
    coding_plan_upstream_override: Option<&str>,
) -> anyhow::Result<Value> {
    let files = project_module_source(root, module_id)?;
    let mut request = serde_json::json!({
        "id": module_id,
        "prompt": prompt,
        "files": files,
        "maxAssistantTurns": 8,
        // The agent gets the Business OS app skill so it edits modules the way
        // the shell/kit/data-boundary contract requires (not as a generic web page).
        "systemPrompt": business_os_system_prompt(),
    });
    // Omission inherits CTOX's active provider/model through the main gateway;
    // an explicit server-authored model may choose another provider route.
    let mut coding_plan_bridge = None;
    if !faux {
        let prepared =
            prepare_coding_turn_model(root, model_override, coding_plan_upstream_override, false)?;
        request["model"] = prepared.model;
        coding_plan_bridge = prepared.coding_plan_bridge;
    }
    let response = run_pi_turn(dist, &request, faux)?;
    drop(coding_plan_bridge);
    anyhow::ensure!(
        response.get("ok").and_then(Value::as_bool) == Some(true),
        "pi-sidecar turn failed: {}",
        response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    );
    let empty = Vec::new();
    let snapshot = response
        .get("snapshot")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let applied = apply_turn_snapshot(root, module_id, snapshot)?;
    let message_count = response
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    // Record the turn under the module's coding session (one session per app) so
    // the workbench can show a per-app history. Best-effort: a session-log hiccup
    // must not discard an edit that already landed in the source.
    if let Err(error) = crate::business_os::store::record_coding_agent_session_turn(
        root,
        module_id,
        prompt,
        &applied,
        message_count,
    ) {
        eprintln!("coding session log failed for {module_id}: {error}");
    }
    Ok(serde_json::json!({
        "ok": true,
        "module_id": module_id,
        "applied_files": applied,
        "message_count": message_count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tiny_http::{Header, Response, Server};

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn node_available() -> bool {
        Command::new("node")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn spawn_anthropic_edit_upstream(
        expected_model: &'static str,
        expected_api_key: &'static str,
    ) -> anyhow::Result<(String, std::thread::JoinHandle<()>)> {
        let server =
            Server::http("127.0.0.1:0").map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let address = server.server_addr().to_ip().context("fake upstream IP")?;
        let worker = std::thread::spawn(move || {
            for turn in 0..2 {
                let mut request = server
                    .recv_timeout(Duration::from_secs(30))
                    .expect("fake upstream receive")
                    .expect("Pi request reaches fake upstream");
                assert_eq!(request.url(), "/v1/messages");
                let api_key_header = request
                    .headers()
                    .iter()
                    .find(|header| {
                        header
                            .field
                            .as_str()
                            .as_str()
                            .eq_ignore_ascii_case("x-api-key")
                    })
                    .map(|header| header.value.as_str().to_owned());
                let bridge_header = request.headers().iter().find(|header| {
                    header
                        .field
                        .as_str()
                        .as_str()
                        .eq_ignore_ascii_case(BRIDGE_TOKEN_HEADER)
                });
                assert_eq!(api_key_header.as_deref(), Some(expected_api_key));
                assert!(bridge_header.is_none());
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap();
                let body: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(body["model"], expected_model);
                let sse = if turn == 0 {
                    concat!(
                        "event: message_start\n",
                        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-smoke-1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
                        "event: content_block_start\n",
                        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-smoke-1\",\"name\":\"edit\",\"input\":{}}}\n\n",
                        "event: content_block_delta\n",
                        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"index.js\\\",\\\"edits\\\":[{\\\"oldText\\\":\\\"v = 1\\\",\\\"newText\\\":\\\"v = 2\\\"}]}\"}}\n\n",
                        "event: content_block_stop\n",
                        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                        "event: message_delta\n",
                        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":8}}\n\n",
                        "event: message_stop\n",
                        "data: {\"type\":\"message_stop\"}\n\n",
                    )
                } else {
                    concat!(
                        "event: message_start\n",
                        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-smoke-2\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
                        "event: content_block_start\n",
                        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                        "event: content_block_delta\n",
                        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Done.\"}}\n\n",
                        "event: content_block_stop\n",
                        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                        "event: message_delta\n",
                        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
                        "event: message_stop\n",
                        "data: {\"type\":\"message_stop\"}\n\n",
                    )
                };
                request
                    .respond(Response::from_string(sse).with_header(
                        Header::from_bytes("content-type", "text/event-stream").unwrap(),
                    ))
                    .unwrap();
            }
        });
        Ok((format!("http://{address}"), worker))
    }

    fn seed_unrelated_main_model(root: &Path) -> anyhow::Result<()> {
        let mut env = BTreeMap::new();
        env.insert("CTOX_API_PROVIDER".to_owned(), "openai".to_owned());
        env.insert(
            "CTOX_CHAT_MODEL_BASE".to_owned(),
            "main-model-must-stay-selected".to_owned(),
        );
        env.insert(
            "CTOX_CHAT_MODEL".to_owned(),
            "main-model-must-stay-selected".to_owned(),
        );
        crate::inference::runtime_env::save_runtime_env_map(root, &env)
    }

    #[test]
    fn faux_sidecar_serves_a_turn_over_the_socket() -> anyhow::Result<()> {
        let dist = sidecar_dist_path(&repo_root());
        if !dist.exists() {
            eprintln!("SKIP: pi-sidecar bundle not built ({})", dist.display());
            return Ok(());
        }
        if !node_available() {
            eprintln!("SKIP: `node` not on PATH");
            return Ok(());
        }

        let request = serde_json::json!({
            "id": "rust-1",
            "prompt": "add a marker",
            "files": { "index.js": "export const v = 1;\n" },
            "maxAssistantTurns": 4
        });
        let response = run_pi_turn(&dist, &request, true)?;

        assert_eq!(response["ok"], Value::Bool(true), "turn ok");
        assert_eq!(response["id"], "rust-1", "response echoes id");
        let has_marker = response["snapshot"]
            .as_array()
            .map(|entries| {
                entries.iter().any(|entry| {
                    entry["path"]
                        .as_str()
                        .map(|path| path.ends_with("faux-marker.js"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        assert!(has_marker, "faux write should round-trip over the socket");
        Ok(())
    }

    #[test]
    fn projects_module_source_records_into_a_files_map() -> anyhow::Result<()> {
        use crate::business_os::store::{load_module_source_records, ModuleSourceLoadMutation};

        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let app_root = root.join("src").join("apps").join("business-os");
        std::fs::create_dir_all(app_root.join("modules").join("widget"))?;
        std::fs::write(app_root.join("index.html"), b"<!doctype html>")?;
        std::fs::write(
            app_root.join("modules").join("widget").join("module.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "id": "widget",
                "title": "Widget",
                "entry": "modules/widget/index.html"
            }))?,
        )?;
        std::fs::write(
            app_root.join("modules").join("widget").join("index.js"),
            "export const v = 1;\n",
        )?;

        load_module_source_records(
            root,
            &ModuleSourceLoadMutation {
                module_id: "widget".to_string(),
            },
        )?;

        let files = project_module_source(root, "widget")?;
        assert!(!files.is_empty(), "projected some source files");
        let has_content = files
            .values()
            .any(|value| value.as_str() == Some("export const v = 1;\n"));
        assert!(
            has_content,
            "widget source content projected into the files map"
        );
        Ok(())
    }

    #[test]
    fn run_module_coding_turn_records_the_faux_edit() -> anyhow::Result<()> {
        use crate::business_os::store::{load_module_source_records, ModuleSourceLoadMutation};

        let dist = sidecar_dist_path(&repo_root());
        if !dist.exists() {
            eprintln!("SKIP: pi-sidecar bundle not built ({})", dist.display());
            return Ok(());
        }
        if !node_available() {
            eprintln!("SKIP: `node` not on PATH");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let app_root = root.join("src").join("apps").join("business-os");
        std::fs::create_dir_all(app_root.join("modules").join("widget"))?;
        std::fs::write(app_root.join("index.html"), b"<!doctype html>")?;
        std::fs::write(
            app_root.join("modules").join("widget").join("module.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "id": "widget",
                "title": "Widget",
                "entry": "modules/widget/index.html"
            }))?,
        )?;
        std::fs::write(
            app_root.join("modules").join("widget").join("index.js"),
            "export const v = 1;\n",
        )?;
        load_module_source_records(
            root,
            &ModuleSourceLoadMutation {
                module_id: "widget".to_string(),
            },
        )?;

        let summary = run_module_coding_turn(root, &dist, "widget", "add a marker", true, None)?;
        assert_eq!(summary["ok"], Value::Bool(true), "owner turn ok");

        // The faux edit must now be part of the module's source records — proving
        // the full owner loop project -> pi turn -> apply -> P0 source records.
        let files = project_module_source(root, "widget")?;
        assert!(
            files.keys().any(|path| path.ends_with("faux-marker.js")),
            "faux edit recorded into module source via the owner loop"
        );
        Ok(())
    }

    #[test]
    fn apply_snapshot_round_trips_a_seeded_file_edit() -> anyhow::Result<()> {
        use crate::business_os::store::{load_module_source_records, ModuleSourceLoadMutation};

        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let app_root = root.join("src").join("apps").join("business-os");
        std::fs::create_dir_all(app_root.join("modules").join("widget"))?;
        std::fs::write(app_root.join("index.html"), b"<!doctype html>")?;
        std::fs::write(
            app_root.join("modules").join("widget").join("module.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "id": "widget",
                "title": "Widget",
                "entry": "modules/widget/index.html"
            }))?,
        )?;
        std::fs::write(
            app_root.join("modules").join("widget").join("index.js"),
            "export const v = 1;\n",
        )?;
        load_module_source_records(
            root,
            &ModuleSourceLoadMutation {
                module_id: "widget".to_string(),
            },
        )?;

        // Learn the projected path key for index.js, then simulate a turn snapshot
        // that edited exactly that file (with the sidecar env cwd prefix).
        let before = project_module_source(root, "widget")?;
        let key = before
            .keys()
            .find(|path| path.ends_with("index.js"))
            .cloned()
            .expect("index.js is projected");
        let snapshot = vec![serde_json::json!({
            "path": format!("/workspace/{key}"),
            "kind": "file",
            "content": "export const v = 2;\n"
        })];
        apply_turn_snapshot(root, "widget", &snapshot)?;

        // The SAME path must now carry the edit — not a nested duplicate.
        let after = project_module_source(root, "widget")?;
        assert_eq!(
            after.get(&key).and_then(Value::as_str),
            Some("export const v = 2;\n"),
            "a real edit round-trips project -> apply to the same module path ({key})"
        );
        Ok(())
    }

    #[test]
    fn embedded_sidecar_extracts_and_runs() -> anyhow::Result<()> {
        // The override must not leak in from the environment for this test.
        std::env::remove_var("CTOX_PI_SIDECAR_DIST");
        let temp = tempfile::tempdir()?;
        let root = temp.path();

        let dist = resolve_sidecar_dist(root)?;
        assert!(
            dist.exists(),
            "embedded sidecar extracted to {}",
            dist.display()
        );
        assert_eq!(
            std::fs::metadata(&dist)?.len(),
            SIDECAR_BUNDLE.len() as u64,
            "extracted bundle size matches the embedded bytes"
        );
        // Idempotent: a second resolve does not rewrite / returns the same path.
        assert_eq!(resolve_sidecar_dist(root)?, dist);

        if !node_available() {
            eprintln!("SKIP: `node` not on PATH (extraction verified, run skipped)");
            return Ok(());
        }
        // The extracted bundle must actually be runnable end-to-end.
        let request = serde_json::json!({
            "id": "embed-1",
            "prompt": "x",
            "files": { "index.js": "1\n" },
            "maxAssistantTurns": 4
        });
        let response = run_pi_turn(&dist, &request, true)?;
        assert_eq!(
            response["ok"],
            Value::Bool(true),
            "the extracted embedded sidecar serves a turn"
        );
        Ok(())
    }

    #[test]
    fn business_os_system_prompt_carries_the_app_skill() {
        let prompt = business_os_system_prompt();
        // The agent must be taught the load-bearing Business OS app conventions,
        // not just generic file tools.
        for marker in [
            "Business OS app",
            "mount(ctx)",
            "WebRTC",
            "kit",
            "in-memory projection",
        ] {
            assert!(
                prompt.contains(marker),
                "system prompt should mention `{marker}`"
            );
        }
    }

    #[test]
    fn coding_default_inherits_the_active_ctox_model() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let model = coding_default_model(temp.path());
        assert_eq!(model["id"], gateway_model(temp.path())["id"]);
        assert_eq!(
            model["api"].as_str(),
            Some("openai-responses"),
            "the coding default speaks the gateway's Responses shape"
        );
        let base_url = model["baseUrl"].as_str().unwrap_or_default();
        assert!(
            base_url.starts_with("http://") && base_url.ends_with(":12434/v1"),
            "coding default routes through the loopback gateway on :12434 (got {base_url})"
        );
        Ok(())
    }

    #[test]
    fn coding_model_preset_is_resolved_server_side_and_unknown_ids_fail() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        assert_eq!(resolve_coding_model_preset(temp.path(), "ctox")?, None);
        let error = resolve_coding_model_preset(temp.path(), "browser-forged")
            .unwrap_err()
            .to_string();
        assert!(error.contains("unavailable"));
        Ok(())
    }

    #[test]
    fn minimax_coding_plan_is_an_independent_account_preset() -> anyhow::Result<()> {
        use crate::execution::models::minimax_coding::{
            store_accounts, MiniMaxCodingAccount, MiniMaxCodingAccountsConfig,
            MiniMaxCodingEndpointProfile, MiniMaxSecretRef,
        };

        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let config = MiniMaxCodingAccountsConfig {
            accounts: vec![MiniMaxCodingAccount {
                id: "bulk-primary".to_owned(),
                disabled: false,
                models: vec!["MiniMax-M3".to_owned()],
                api_key_secret: MiniMaxSecretRef {
                    scope: "provider-subscriptions".to_owned(),
                    name: "bulk-primary-api-key".to_owned(),
                },
                endpoint_profile: MiniMaxCodingEndpointProfile::GlobalAnthropic,
            }],
            ..MiniMaxCodingAccountsConfig::default()
        };
        store_accounts(root, &config)?;

        // Config alone must not advertise an unusable account.
        assert_eq!(
            coding_model_capabilities(root)["presets"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        crate::secrets::write_secret_record(
            root,
            "provider-subscriptions",
            "bulk-primary-api-key",
            "test-secret",
            None,
            serde_json::json!({"provider":"minimax","access_mode":"coding_plan"}),
        )?;

        let capabilities = coding_model_capabilities(root);
        let preset = capabilities["presets"]
            .as_array()
            .and_then(|presets| {
                presets.iter().find(|preset| {
                    preset["model"]["provider"].as_str() == Some("ctox-minimax-coding")
                })
            })
            .context("MiniMax preset missing")?;
        let preset_id = preset["id"].as_str().context("preset id missing")?;
        let resolved = resolve_coding_model_preset(root, preset_id)?.context("route model")?;
        assert_eq!(resolved["id"], "MiniMax-M3");
        assert_eq!(resolved["ctoxRoute"]["accountId"], "bulk-primary");
        assert_ne!(resolved["provider"], "ctox_proxy");
        assert!(
            serde_json::to_string(&capabilities)?
                .find("test-secret")
                .is_none(),
            "public capability document must not expose secret material"
        );
        assert!(
            serde_json::to_string(&capabilities)?
                .find(BRIDGE_TOKEN_HEADER)
                .is_none(),
            "turn-local bridge authority must not exist in capability projection"
        );
        Ok(())
    }

    #[test]
    fn kimi_coding_plan_is_an_independent_account_preset() -> anyhow::Result<()> {
        use crate::execution::models::kimi_coding::{
            store_accounts, KimiCodingAccount, KimiCodingAccountsConfig, KimiCodingEndpointProfile,
            KimiSecretRef,
        };

        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let config = KimiCodingAccountsConfig {
            accounts: vec![KimiCodingAccount {
                id: "kimi-primary".to_owned(),
                disabled: false,
                models: vec!["k3[1m]".to_owned()],
                api_key_secret: KimiSecretRef {
                    scope: "provider-subscriptions".to_owned(),
                    name: "kimi-primary-api-key".to_owned(),
                },
                endpoint_profile: KimiCodingEndpointProfile::KimiCoding,
            }],
            ..KimiCodingAccountsConfig::default()
        };
        store_accounts(root, &config)?;

        assert_eq!(
            coding_model_capabilities(root)["presets"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        crate::secrets::write_secret_record(
            root,
            "provider-subscriptions",
            "kimi-primary-api-key",
            "test-secret",
            None,
            serde_json::json!({"provider":"kimi","access_mode":"coding_plan"}),
        )?;

        let capabilities = coding_model_capabilities(root);
        let preset = capabilities["presets"]
            .as_array()
            .and_then(|presets| {
                presets
                    .iter()
                    .find(|preset| preset["model"]["provider"].as_str() == Some("ctox-kimi-coding"))
            })
            .context("Kimi coding preset missing")?;
        let preset_id = preset["id"].as_str().context("preset id missing")?;
        let resolved = resolve_coding_model_preset(root, preset_id)?.context("route model")?;
        assert_eq!(resolved["id"], "k3[1m]");
        assert_eq!(resolved["contextWindow"], 1_048_576);
        assert_eq!(resolved["ctoxRoute"]["accountId"], "kimi-primary");
        assert_ne!(resolved["provider"], "ctox_proxy");
        let public = serde_json::to_string(&capabilities)?;
        assert!(!public.contains("test-secret"));
        assert!(!public.contains(BRIDGE_TOKEN_HEADER));
        Ok(())
    }

    #[test]
    fn minimax_preset_drives_a_real_pi_edit_through_the_selected_account() -> anyhow::Result<()> {
        use crate::execution::models::minimax_coding::{
            store_accounts, MiniMaxCodingAccount, MiniMaxCodingAccountsConfig,
            MiniMaxCodingEndpointProfile, MiniMaxSecretRef,
        };

        if !node_available() {
            eprintln!("SKIP: `node` not on PATH");
            return Ok(());
        }
        let dist = sidecar_dist_path(&repo_root());
        if !dist.exists() {
            eprintln!("SKIP: pi-sidecar bundle not built ({})", dist.display());
            return Ok(());
        }
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        seed_unrelated_main_model(root)?;
        store_accounts(
            root,
            &MiniMaxCodingAccountsConfig {
                accounts: vec![MiniMaxCodingAccount {
                    id: "minimax-smoke-account".to_owned(),
                    disabled: false,
                    models: vec!["MiniMax-M3".to_owned()],
                    api_key_secret: MiniMaxSecretRef {
                        scope: "provider-subscriptions".to_owned(),
                        name: "minimax-smoke-key".to_owned(),
                    },
                    endpoint_profile: MiniMaxCodingEndpointProfile::GlobalAnthropic,
                }],
                ..MiniMaxCodingAccountsConfig::default()
            },
        )?;
        crate::secrets::write_secret_record(
            root,
            "provider-subscriptions",
            "minimax-smoke-key",
            "minimax-smoke-secret-must-not-leak",
            None,
            serde_json::json!({"test":true}),
        )?;
        let preset_id = coding_model_capabilities(root)["presets"]
            .as_array()
            .and_then(|presets| {
                presets.iter().find(|preset| {
                    preset["model"]["provider"].as_str() == Some("ctox-minimax-coding")
                })
            })
            .and_then(|preset| preset["id"].as_str())
            .context("MiniMax smoke preset")?
            .to_owned();
        let (upstream, worker) =
            spawn_anthropic_edit_upstream("MiniMax-M3", "minimax-smoke-secret-must-not-leak")?;
        let evidence =
            run_coding_preset_smoke_inner(root, &dist, &preset_id, None, Some(&upstream), None)?;
        worker.join().expect("fake MiniMax upstream");

        assert_eq!(evidence["provider"], "minimax_coding_plan");
        assert_eq!(evidence["model"], "MiniMax-M3");
        assert_eq!(evidence["main_model_unchanged"], true);
        assert_eq!(
            crate::inference::runtime_env::effective_chat_model(root).as_deref(),
            Some("main-model-must-stay-selected")
        );
        let rendered = evidence.to_string();
        assert!(!rendered.contains("minimax-smoke-account"));
        assert!(!rendered.contains("minimax-smoke-secret-must-not-leak"));
        assert!(!rendered.contains(&upstream));
        Ok(())
    }

    #[test]
    fn kimi_preset_drives_a_real_pi_edit_through_the_selected_account() -> anyhow::Result<()> {
        use crate::execution::models::kimi_coding::{
            store_accounts, KimiCodingAccount, KimiCodingAccountsConfig, KimiCodingEndpointProfile,
            KimiSecretRef,
        };

        if !node_available() {
            eprintln!("SKIP: `node` not on PATH");
            return Ok(());
        }
        let dist = sidecar_dist_path(&repo_root());
        if !dist.exists() {
            eprintln!("SKIP: pi-sidecar bundle not built ({})", dist.display());
            return Ok(());
        }
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        seed_unrelated_main_model(root)?;
        store_accounts(
            root,
            &KimiCodingAccountsConfig {
                accounts: vec![KimiCodingAccount {
                    id: "kimi-smoke-account".to_owned(),
                    disabled: false,
                    models: vec!["k3[1m]".to_owned()],
                    api_key_secret: KimiSecretRef {
                        scope: "provider-subscriptions".to_owned(),
                        name: "kimi-smoke-key".to_owned(),
                    },
                    endpoint_profile: KimiCodingEndpointProfile::KimiCoding,
                }],
                ..KimiCodingAccountsConfig::default()
            },
        )?;
        crate::secrets::write_secret_record(
            root,
            "provider-subscriptions",
            "kimi-smoke-key",
            "kimi-smoke-secret-must-not-leak",
            None,
            serde_json::json!({"test":true}),
        )?;
        let preset_id = coding_model_capabilities(root)["presets"]
            .as_array()
            .and_then(|presets| {
                presets
                    .iter()
                    .find(|preset| preset["model"]["provider"].as_str() == Some("ctox-kimi-coding"))
            })
            .and_then(|preset| preset["id"].as_str())
            .context("Kimi smoke preset")?
            .to_owned();
        let (upstream, worker) =
            spawn_anthropic_edit_upstream("k3[1m]", "kimi-smoke-secret-must-not-leak")?;
        let evidence =
            run_coding_preset_smoke_inner(root, &dist, &preset_id, None, Some(&upstream), None)?;
        worker.join().expect("fake Kimi upstream");

        assert_eq!(evidence["provider"], "kimi_coding_plan");
        assert_eq!(evidence["model"], "k3[1m]");
        assert_eq!(evidence["main_model_unchanged"], true);
        assert_eq!(
            crate::inference::runtime_env::effective_chat_model(root).as_deref(),
            Some("main-model-must-stay-selected")
        );
        let rendered = evidence.to_string();
        assert!(!rendered.contains("kimi-smoke-account"));
        assert!(!rendered.contains("kimi-smoke-secret-must-not-leak"));
        assert!(!rendered.contains(&upstream));
        Ok(())
    }

    #[test]
    fn gateway_model_points_at_the_loopback_responses_gateway() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let model = gateway_model(temp.path());
        let base_url = model["baseUrl"].as_str().unwrap_or_default();
        assert!(
            base_url.starts_with("http://") && base_url.ends_with(":12434/v1"),
            "gateway model targets the loopback Responses gateway on :12434 (got {base_url})"
        );
        assert_eq!(
            model["api"].as_str(),
            Some("openai-responses"),
            "uses pi-ai's OpenAI Responses provider"
        );
        assert!(
            model["id"]
                .as_str()
                .map(|id| !id.is_empty())
                .unwrap_or(false),
            "an active model id is resolved from the gateway config"
        );
        Ok(())
    }
}
