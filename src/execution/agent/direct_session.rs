// Origin: CTOX
// License: Apache-2.0
//
// Direct session: in-process ctox-core integration via InProcessAppServerClient.
// One persistent client per CTOX mission-turn-loop. Multiple sequential turns
// (main turn + continuity refreshes) reuse the same client and thread.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ctox_app_server_client::{
    InProcessAppServerClient, InProcessClientStartArgs, InProcessServerEvent,
    DEFAULT_IN_PROCESS_CHANNEL_CAPACITY,
};
use ctox_app_server_protocol::{
    ClientRequest, JSONRPCNotification, RequestId, ThreadCompactStartParams,
    ThreadCompactStartResponse, ThreadStartParams, ThreadStartResponse, ThreadUnsubscribeParams,
    ThreadUnsubscribeResponse, TurnStartParams, TurnStartResponse,
};
use ctox_arg0::Arg0DispatchPaths;
use ctox_cloud_requirements::cloud_requirements_loader;
use ctox_core::config::{
    find_codex_home, load_config_as_toml_with_cli_overrides, ConfigBuilder, ConfigOverrides,
};
use ctox_core::AuthManager;
use ctox_feedback::CodexFeedback;
use ctox_protocol::config_types::SandboxMode;
use ctox_protocol::protocol::{AskForApproval, EventMsg, SandboxPolicy, SessionSource};
use ctox_protocol::user_input::UserInput;
use ctox_utils_absolute_path::AbsolutePathBuf;

use crate::context::compact::{CompactDecision, CompactMode, CompactPolicy};
use crate::inference::runtime_kernel;
use crate::lcm::CompactionResult;

// ---------------------------------------------------------------------------
// PersistentSession — lives across turns within a mission-turn-loop iteration
// ---------------------------------------------------------------------------

/// Holds a running InProcessAppServerClient + thread. Created once per
/// mission-turn-loop iteration, reused for the main turn AND all continuity
/// refresh calls. Solves the resource-exhaustion hang that occurred when
/// spawning a new client per call.
pub(crate) struct PersistentSession {
    runtime: tokio::runtime::Runtime,
    // These are wrapped in Option so we can take() them in shutdown.
    client: Option<InProcessAppServerClient>,
    thread_id: String,
    seq: RequestIdSeq,
    cwd: PathBuf,
    policy: CompactPolicy,
    configured_context_window: Option<i64>,
    ctx_log: ContextLogger,
    root: PathBuf,
}

impl PersistentSession {
    /// Start a persistent session: creates tokio runtime, app-server client,
    /// and thread. Call this ONCE per mission-turn-loop iteration.
    pub fn start(root: &Path, settings: &BTreeMap<String, String>) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("failed to start tokio runtime")?;

        let (client, thread_id, cwd, seq, configured_context_window) =
            rt.block_on(async { Self::start_client_and_thread(root, settings).await })?;

        let policy = CompactPolicy::from_settings(
            settings.get("CTOX_COMPACT_TRIGGER").map(String::as_str),
            settings.get("CTOX_COMPACT_MODE").map(String::as_str),
            settings
                .get("CTOX_COMPACT_FIXED_INTERVAL")
                .map(String::as_str),
            settings
                .get("CTOX_COMPACT_ADAPTIVE_THRESHOLD")
                .map(String::as_str),
            settings
                .get("CTOX_COMPACT_EMERGENCY_RATIO")
                .map(String::as_str),
            settings
                .get("CTOX_CHAT_MODEL_MAX_CONTEXT")
                .map(String::as_str),
        );
        let ctx_log = ContextLogger::open(root);

        eprintln!(
            "[ctox direct-session] persistent session started thread_id={}",
            thread_id
        );

        Ok(Self {
            runtime: rt,
            client: Some(client),
            thread_id,
            seq,
            cwd,
            policy,
            configured_context_window,
            ctx_log,
            root: root.to_path_buf(),
        })
    }

    /// Run a single turn on the persistent session. Can be called multiple
    /// times (main turn, then refreshes). All share the same client+thread.
    pub fn run_turn(
        &mut self,
        prompt: &str,
        timeout: Option<Duration>,
        _base_instructions: Option<&str>,
        _include_apply_patch_tool: Option<bool>,
        conversation_id: i64,
    ) -> Result<String> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("session already shut down"))?;
        let cwd = self.cwd.clone();
        let prompt = prompt.to_string();
        let root = self.root.clone();

        self.ctx_log.log(
            "turn_request",
            &format!("\"prompt_len\":{},\"timeout\":{:?}", prompt.len(), timeout),
        );

        let result = self.runtime.block_on(async {
            Self::run_turn_async(
                client,
                &mut self.thread_id,
                &cwd,
                &root,
                &prompt,
                timeout,
                &mut self.seq,
                &mut self.policy,
                self.configured_context_window,
                &mut self.ctx_log,
                conversation_id,
            )
            .await
        });

        result
    }

    /// Shut down the client and runtime cleanly.
    pub fn shutdown(mut self) {
        if let Some(client) = self.client.take() {
            let tid = self.thread_id.clone();
            let _ = self.runtime.block_on(async {
                eprintln!(
                    "[ctox direct-session] shutting down persistent session thread_id={}",
                    tid
                );
                let _ = client.shutdown().await;
            });
        }
    }

    // --- Internal async helpers ---

    async fn start_client_and_thread(
        root: &Path,
        settings: &BTreeMap<String, String>,
    ) -> Result<(
        InProcessAppServerClient,
        String,
        PathBuf,
        RequestIdSeq,
        Option<i64>,
    )> {
        let model = settings
            .get("CTOX_CHAT_MODEL")
            .or_else(|| settings.get("CODEX_MODEL"))
            .cloned()
            .unwrap_or_else(|| "gpt-5.4-mini".to_string());
        let cwd: PathBuf = root.to_path_buf();

        let codex_home =
            find_codex_home().map_err(|err| anyhow::anyhow!("find_codex_home: {err}"))?;

        // Write API key from CTOX settings map into auth.json.
        // No process env — key flows: TUI/SQLite settings → settings map → auth.json → AuthManager.
        let api_key = settings
            .get("OPENAI_API_KEY")
            .or_else(|| settings.get("OPENROUTER_API_KEY"))
            .or_else(|| settings.get("ANTHROPIC_API_KEY"))
            .or_else(|| settings.get("MINIMAX_API_KEY"))
            .filter(|v| !v.trim().is_empty());
        if let Some(key) = api_key {
            let _ = ctox_core::auth::login_with_api_key(&codex_home, key, Default::default());
        }
        let config_cwd =
            AbsolutePathBuf::from_absolute_path(cwd.canonicalize().unwrap_or(cwd.clone()))
                .map_err(|err| anyhow::anyhow!("cwd resolve: {err}"))?;
        let config_toml = load_config_as_toml_with_cli_overrides(&codex_home, &config_cwd, vec![])
            .await
            .map_err(|err| anyhow::anyhow!("load config.toml: {err}"))?;

        let auth_manager = AuthManager::shared(
            codex_home.clone(),
            true,
            config_toml.cli_auth_credentials_store.unwrap_or_default(),
        );
        let cloud_requirements = cloud_requirements_loader(
            auth_manager,
            config_toml
                .chatgpt_base_url
                .clone()
                .unwrap_or_else(|| "https://chatgpt.com/backend-api/".to_string()),
            codex_home.clone(),
        );

        // Resolve model-provider BEFORE building overrides
        let resolved_runtime = runtime_kernel::InferenceRuntimeKernel::resolve(root).ok();
        let api_provider = super::turn_loop::resolve_api_model_provider_spec(
            &model,
            settings,
            resolved_runtime.as_ref(),
        );
        let local_provider =
            super::turn_loop::resolve_local_model_provider_spec(resolved_runtime.as_ref());
        if resolved_runtime
            .as_ref()
            .is_some_and(|runtime| runtime.state.source.is_local())
            && local_provider.is_none()
        {
            anyhow::bail!(
                "CTOX local runtime requires socket-based Responses transport; no managed socket path is available"
            );
        }
        let selected_provider_id = local_provider
            .as_ref()
            .map(|provider| provider.provider_id.to_string())
            .or_else(|| {
                api_provider
                    .as_ref()
                    .map(|provider| provider.provider_id.to_string())
            });
        let configured_context_window = settings
            .get("CTOX_CHAT_MODEL_MAX_CONTEXT")
            .and_then(|value| value.trim().parse::<i64>().ok())
            .filter(|value| *value > 0);
        let overrides = ConfigOverrides {
            model: Some(model.clone()),
            model_context_window: configured_context_window,
            model_provider: selected_provider_id,
            cwd: Some(cwd.clone()),
            approval_policy: Some(AskForApproval::Never),
            sandbox_mode: Some(SandboxMode::DangerFullAccess),
            ephemeral: Some(true),
            ..Default::default()
        };
        // Hand ctox-core one of the two explicit CTOX provider modes:
        // `ctox_core_local` for managed socket-backed local runtimes or
        // `ctox_core_api` for remote providers. Both stay Responses-facing
        // from CTOX's perspective; any provider-specific wire adaptation
        // happens only at the outer edge.
        let mut cli_overrides: Vec<(String, toml::Value)> = vec![];
        if let Some(context_window) = configured_context_window {
            cli_overrides.push((
                "model_context_window".to_string(),
                toml::Value::Integer(context_window),
            ));
        }
        if let Some(ref provider) = local_provider {
            cli_overrides.extend(provider.ctox_core_cli_overrides());
            eprintln!(
                "[ctox direct-session] provider mode=ctox_core_local id={} endpoint={} wire_api={}",
                provider.provider_id, provider.transport_endpoint, provider.wire_api
            );
        }
        if let Some(ref provider) = api_provider {
            cli_overrides.extend(provider.ctox_core_cli_overrides());
            eprintln!(
                "[ctox direct-session] provider mode=ctox_core_api id={} base_url={} wire_api={}",
                provider.provider_id, provider.base_url, provider.wire_api
            );
        }
        let config = ConfigBuilder::default()
            .cli_overrides(cli_overrides)
            .harness_overrides(overrides)
            .cloud_requirements(cloud_requirements.clone())
            .build()
            .await
            .map_err(|err| anyhow::anyhow!("config build: {err}"))?;

        let start_args = InProcessClientStartArgs {
            arg0_paths: Arg0DispatchPaths::default(),
            config: Arc::new(config),
            cli_overrides: vec![],
            loader_overrides: Default::default(),
            cloud_requirements,
            feedback: CodexFeedback::new(),
            config_warnings: vec![],
            session_source: SessionSource::Exec,
            enable_ctox_api_key_env: true,
            client_name: "ctox-direct".to_string(),
            client_version: env!("CTOX_BUILD_VERSION").to_string(),
            experimental_api: true,
            opt_out_notification_methods: vec![],
            channel_capacity: DEFAULT_IN_PROCESS_CHANNEL_CAPACITY,
        };

        eprintln!("[ctox direct-session] starting InProcessAppServerClient...");
        let client = InProcessAppServerClient::start(start_args)
            .await
            .map_err(|err| anyhow::anyhow!("client start: {err}"))?;
        eprintln!("[ctox direct-session] client started");

        let mut seq = RequestIdSeq::new();
        let thread_id = Self::start_thread(
            &client,
            &mut seq,
            &cwd,
            Some(model.as_str()),
            configured_context_window,
        )
        .await?;
        eprintln!("[ctox direct-session] thread started: {}", thread_id);

        Ok((client, thread_id, cwd, seq, configured_context_window))
    }

    async fn start_thread(
        client: &InProcessAppServerClient,
        seq: &mut RequestIdSeq,
        cwd: &Path,
        model: Option<&str>,
        configured_context_window: Option<i64>,
    ) -> Result<String> {
        let config = configured_context_window.map(|context_window| {
            let mut values = std::collections::HashMap::new();
            values.insert(
                "model_context_window".to_string(),
                serde_json::Value::from(context_window),
            );
            values
        });
        let thread_resp: ThreadStartResponse = client
            .request_typed(ClientRequest::ThreadStart {
                request_id: seq.next(),
                params: ThreadStartParams {
                    model: model.map(str::to_string),
                    cwd: Some(cwd.to_string_lossy().to_string()),
                    approval_policy: Some(AskForApproval::Never.into()),
                    sandbox: Some(ctox_app_server_protocol::SandboxMode::DangerFullAccess),
                    config,
                    ephemeral: Some(true),
                    ..ThreadStartParams::default()
                },
            })
            .await
            .map_err(|err| anyhow::anyhow!("thread/start: {err}"))?;
        Ok(thread_resp.thread.id)
    }

    async fn run_turn_async(
        client: &mut InProcessAppServerClient,
        thread_id: &mut String,
        cwd: &Path,
        root: &Path,
        prompt: &str,
        timeout: Option<Duration>,
        seq: &mut RequestIdSeq,
        policy: &mut CompactPolicy,
        configured_context_window: Option<i64>,
        ctx_log: &mut ContextLogger,
        conversation_id: i64,
    ) -> Result<String> {
        let mut current_thread_id = thread_id.clone();
        let turn_resp: TurnStartResponse = match client
            .request_typed(ClientRequest::TurnStart {
                request_id: seq.next(),
                params: TurnStartParams {
                    thread_id: current_thread_id.clone(),
                    input: vec![UserInput::Text {
                        text: prompt.to_string(),
                        text_elements: Vec::new(),
                    }
                    .into()],
                    cwd: Some(cwd.to_path_buf()),
                    approval_policy: Some(AskForApproval::Never.into()),
                    approvals_reviewer: None,
                    sandbox_policy: Some(SandboxPolicy::DangerFullAccess.into()),
                    model: None,
                    service_tier: None,
                    effort: None,
                    summary: None,
                    personality: None,
                    output_schema: None,
                    collaboration_mode: None,
                },
            })
            .await
        {
            Ok(response) => response,
            Err(err) => {
                eprintln!(
                    "[ctox direct-session] turn/start failed on thread {}: {}. starting a fresh thread",
                    current_thread_id, err
                );
                current_thread_id =
                    Self::start_thread(client, seq, cwd, None, configured_context_window).await?;
                *thread_id = current_thread_id.clone();
                client
                    .request_typed(ClientRequest::TurnStart {
                        request_id: seq.next(),
                        params: TurnStartParams {
                            thread_id: current_thread_id.clone(),
                            input: vec![UserInput::Text {
                                text: prompt.to_string(),
                                text_elements: Vec::new(),
                            }
                            .into()],
                            cwd: Some(cwd.to_path_buf()),
                            approval_policy: Some(AskForApproval::Never.into()),
                            approvals_reviewer: None,
                            sandbox_policy: Some(SandboxPolicy::DangerFullAccess.into()),
                            model: None,
                            service_tier: None,
                            effort: None,
                            summary: None,
                            personality: None,
                            output_schema: None,
                            collaboration_mode: None,
                        },
                    })
                    .await
                    .map_err(|retry_err| anyhow::anyhow!("turn/start retry: {retry_err}"))?
            }
        };
        eprintln!(
            "[ctox direct-session] active thread for turn: {}",
            current_thread_id
        );
        let turn_id = turn_resp.turn.id;

        // Event loop
        let mut final_message: Option<String> = None;
        let mut forced_followup_fired = false;
        let deadline = timeout.map(|d| tokio::time::Instant::now() + d);

        loop {
            let event = match deadline {
                Some(d) => tokio::select! {
                    ev = client.next_event() => ev,
                    _ = tokio::time::sleep_until(d) => {
                        ctx_log.log(
                            "turn_timeout",
                            &format!(
                                "\"turn_id\":\"{}\",\"thread_id\":\"{}\"",
                                turn_id, current_thread_id
                            ),
                        );
                        let _ = client.request_typed::<ThreadUnsubscribeResponse>(ClientRequest::ThreadUnsubscribe {
                            request_id: seq.next(),
                            params: ThreadUnsubscribeParams {
                                thread_id: current_thread_id.clone(),
                            },
                        }).await;
                        if let Ok(replacement_thread_id) =
                            Self::start_thread(client, seq, cwd, None, configured_context_window)
                                .await
                        {
                            *thread_id = replacement_thread_id;
                        }
                        anyhow::bail!("direct session timeout after {:?}", timeout.unwrap());
                    }
                },
                None => client.next_event().await,
            };
            let Some(event) = event else { break };
            match event {
                InProcessServerEvent::ServerRequest(_) => {}
                InProcessServerEvent::ServerNotification(_) => {}
                InProcessServerEvent::LegacyNotification(notif) => {
                    if let Some(msg) = try_extract_event_msg(&notif) {
                        if !matches!(&msg, EventMsg::TurnComplete(tc) if tc.turn_id != turn_id) {
                            ctx_log.observe(&msg);
                        }

                        if let CompactDecision::Compact { reason } = policy.evaluate(&msg) {
                            eprintln!(
                                "[ctox direct-session] compact mode={:?} reason={}",
                                policy.mode,
                                reason.log_summary()
                            );
                            match policy.mode {
                                CompactMode::MidTask => {
                                    ctx_log.log_compact_decision("decision", &reason, policy);
                                    let compact_req = ClientRequest::ThreadCompactStart {
                                        request_id: seq.next(),
                                        params: ThreadCompactStartParams {
                                            thread_id: current_thread_id.clone(),
                                        },
                                    };
                                    match client
                                        .request_typed::<ThreadCompactStartResponse>(compact_req)
                                        .await
                                    {
                                        Ok(_) => {
                                            ctx_log.log_compact_decision(
                                                "compact_ok",
                                                &reason,
                                                policy,
                                            );
                                            policy.note_compacted();
                                        }
                                        Err(err) => {
                                            eprintln!(
                                                "[ctox direct-session] compact failed: {err}"
                                            );
                                            ctx_log.log_compact_decision(
                                                &format!("compact_fail:{err}"),
                                                &reason,
                                                policy,
                                            );
                                            policy.note_compacted();
                                        }
                                    }
                                    let durable_compaction = log_durable_lcm_compaction(
                                        root,
                                        conversation_id,
                                        policy.context_window,
                                        ctx_log,
                                    );
                                    if should_abort_for_compaction_followup(
                                        policy.mode,
                                        &reason,
                                        durable_compaction.as_ref(),
                                    ) {
                                        ctx_log.log_compact_decision(
                                            "forced_followup",
                                            &reason,
                                            policy,
                                        );
                                        let unsub_req = ClientRequest::ThreadUnsubscribe {
                                            request_id: seq.next(),
                                            params: ThreadUnsubscribeParams {
                                                thread_id: current_thread_id.clone(),
                                            },
                                        };
                                        let _ = client
                                            .request_typed::<ThreadUnsubscribeResponse>(unsub_req)
                                            .await;
                                        if let Ok(replacement_thread_id) = Self::start_thread(
                                            client,
                                            seq,
                                            cwd,
                                            None,
                                            configured_context_window,
                                        )
                                        .await
                                        {
                                            *thread_id = replacement_thread_id;
                                        }
                                        anyhow::bail!(
                                            "runtime time budget guard: compaction follow-up required after {}",
                                            reason.log_summary()
                                        );
                                    }
                                }
                                CompactMode::ForcedFollowup => {
                                    let unsub_req = ClientRequest::ThreadUnsubscribe {
                                        request_id: seq.next(),
                                        params: ThreadUnsubscribeParams {
                                            thread_id: current_thread_id.clone(),
                                        },
                                    };
                                    let _ = client
                                        .request_typed::<ThreadUnsubscribeResponse>(unsub_req)
                                        .await;
                                    let signal = root.join("runtime/compact-followup-requested");
                                    let _ =
                                        std::fs::create_dir_all(signal.parent().unwrap_or(root));
                                    if let Ok(mut f) = std::fs::OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open(&signal)
                                    {
                                        let _ = writeln!(
                                            f,
                                            "{}\t{}\t{}",
                                            0,
                                            thread_id,
                                            reason.log_summary()
                                        );
                                    }
                                    policy.note_compacted();
                                    log_durable_lcm_compaction(
                                        root,
                                        conversation_id,
                                        policy.context_window,
                                        ctx_log,
                                    );
                                    forced_followup_fired = true;
                                    break;
                                }
                            }
                        }
                        match msg {
                            EventMsg::AgentMessage(am) => {
                                final_message = Some(am.message.clone());
                            }
                            EventMsg::TurnComplete(tc) if tc.turn_id == turn_id => break,
                            EventMsg::Error(ref err) => {
                                let msg_str = format!("{:?}", err);
                                // Compaction errors are non-fatal — the model
                                // may not produce the exact JSON format ctox-core
                                // expects for compaction. Log and continue.
                                if msg_str.contains("compaction")
                                    || msg_str.contains("revisedTitle")
                                {
                                    eprintln!(
                                        "[ctox direct-session] compaction error (non-fatal): {}",
                                        msg_str
                                    );
                                    ctx_log.log(
                                        "compaction_error",
                                        &format!(
                                            "\"message\":\"{}\"",
                                            msg_str
                                                .replace('"', "'")
                                                .chars()
                                                .take(200)
                                                .collect::<String>()
                                        ),
                                    );
                                } else {
                                    anyhow::bail!("direct session error: {}", msg_str);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                InProcessServerEvent::Lagged { skipped } => {
                    eprintln!("[ctox direct-session] lagged: dropped {skipped} events");
                }
            }
        }

        ctx_log.log(
            "turn_end",
            &format!(
                "\"forced_followup\":{forced_followup_fired},\"reply_chars\":{}",
                final_message.as_ref().map(|m| m.len()).unwrap_or(0)
            ),
        );

        if forced_followup_fired {
            Ok(final_message.unwrap_or_default())
        } else {
            final_message.ok_or_else(|| anyhow::anyhow!("turn completed without assistant message"))
        }
    }
}

fn log_durable_lcm_compaction(
    root: &Path,
    conversation_id: i64,
    token_budget: i64,
    ctx_log: &mut ContextLogger,
) -> Option<CompactionResult> {
    if conversation_id <= 0 {
        return None;
    }
    match crate::lcm::LcmEngine::open(
        &crate::persistence::sqlite_path(root),
        crate::lcm::LcmConfig::default(),
    ) {
        Ok(engine) => match engine.compact(
            conversation_id,
            token_budget.max(1),
            &crate::lcm::HeuristicSummarizer,
            true,
        ) {
            Ok(result) => {
                ctx_log.log(
                    "compact_persist_ok",
                    &format!(
                        "\"conversation_id\":{},\"action_taken\":{},\"rounds\":{},\"created\":{},\"tokens_before\":{},\"tokens_after\":{}",
                        conversation_id,
                        result.action_taken,
                        result.rounds,
                        result.created_summary_ids.len(),
                        result.tokens_before,
                        result.tokens_after
                    ),
                );
                Some(result)
            }
            Err(err) => {
                ctx_log.log(
                    "compact_persist_fail",
                    &format!(
                        "\"conversation_id\":{},\"message\":\"{}\"",
                        conversation_id,
                        err.to_string().replace('"', "'")
                    ),
                );
                None
            }
        },
        Err(err) => {
            ctx_log.log(
                "compact_persist_fail",
                &format!(
                    "\"conversation_id\":{},\"message\":\"{}\"",
                    conversation_id,
                    err.to_string().replace('"', "'")
                ),
            );
            None
        }
    }
}

fn should_abort_for_compaction_followup(
    mode: CompactMode,
    reason: &crate::context::compact::CompactReason,
    durable_result: Option<&CompactionResult>,
) -> bool {
    match mode {
        CompactMode::ForcedFollowup => true,
        CompactMode::MidTask => {
            matches!(
                reason,
                crate::context::compact::CompactReason::Emergency { .. }
            ) && durable_result.is_none_or(|result| {
                !result.action_taken || result.tokens_after >= result.tokens_before
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

struct RequestIdSeq {
    next: i64,
}
impl RequestIdSeq {
    fn new() -> Self {
        Self { next: 1 }
    }
    fn next(&mut self) -> RequestId {
        let id = self.next;
        self.next += 1;
        RequestId::Integer(id)
    }
}

fn try_extract_event_msg(notif: &JSONRPCNotification) -> Option<EventMsg> {
    let method = notif
        .method
        .strip_prefix("codex/event/")
        .unwrap_or(&notif.method)
        .to_string();
    let value = notif.params.clone()?;
    let serde_json::Value::Object(mut obj) = value else {
        return None;
    };
    let mut payload = if let Some(serde_json::Value::Object(msg_obj)) = obj.get("msg") {
        serde_json::Value::Object(msg_obj.clone())
    } else {
        obj.remove("conversationId");
        serde_json::Value::Object(obj)
    };
    if let serde_json::Value::Object(ref mut map) = payload {
        map.insert("type".to_string(), serde_json::Value::String(method));
    }
    serde_json::from_value(payload).ok()
}

// ---------------------------------------------------------------------------
// Context forensics logger
// ---------------------------------------------------------------------------

struct ContextLogger {
    file: Option<std::fs::File>,
    session_start: Instant,
    items_this_turn: u32,
    last_total_tokens: i64,
    last_context_window: i64,
}

impl ContextLogger {
    fn open(root: &Path) -> Self {
        let path = root.join("runtime/context-log.jsonl");
        let _ = std::fs::create_dir_all(root.join("runtime"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok();
        Self {
            file,
            session_start: Instant::now(),
            items_this_turn: 0,
            last_total_tokens: 0,
            last_context_window: 0,
        }
    }

    fn log(&mut self, event: &str, extra: &str) {
        let Some(f) = self.file.as_mut() else { return };
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let elapsed = self.session_start.elapsed().as_secs();
        let _ = writeln!(
            f,
            "{{\"ts\":{ts},\"elapsed_s\":{elapsed},\"event\":\"{event}\",\
             \"total_tokens\":{},\"context_window\":{},\"items_this_turn\":{},{extra}}}",
            self.last_total_tokens, self.last_context_window, self.items_this_turn
        );
    }

    fn observe(&mut self, msg: &EventMsg) {
        match msg {
            EventMsg::TurnStarted(ts) => {
                self.items_this_turn = 0;
                if let Some(w) = ts.model_context_window {
                    self.last_context_window = w;
                }
                self.log(
                    "turn_started",
                    &format!(
                        "\"turn_id\":\"{}\",\"model_context_window\":{}",
                        ts.turn_id,
                        ts.model_context_window.unwrap_or(-1)
                    ),
                );
            }
            EventMsg::TokenCount(tc) => {
                if let Some(info) = tc.info.as_ref() {
                    self.last_total_tokens = info.total_token_usage.total_tokens;
                    let last_input = info.last_token_usage.input_tokens;
                    let last_output = info.last_token_usage.output_tokens;
                    if let Some(w) = info.model_context_window {
                        self.last_context_window = w;
                    }
                    let fill_pct = if self.last_context_window > 0 {
                        (last_input as f64) / (self.last_context_window as f64)
                    } else {
                        0.0
                    };
                    self.log(
                        "token_count",
                        &format!(
                            "\"call_input\":{last_input},\"call_output\":{last_output},\
                         \"cum_output\":{},\"cum_input\":{},\
                         \"fill_pct\":{fill_pct:.3},\"window\":{}",
                            info.total_token_usage.output_tokens,
                            info.total_token_usage.input_tokens,
                            self.last_context_window,
                        ),
                    );
                }
            }
            EventMsg::TurnComplete(tc) => {
                self.log(
                    "turn_complete",
                    &format!(
                        "\"turn_id\":\"{}\",\"has_last_message\":{}",
                        tc.turn_id,
                        tc.last_agent_message.is_some()
                    ),
                );
            }
            EventMsg::AgentMessage(am) => {
                self.log("agent_message", &format!("\"chars\":{}", am.message.len()));
            }
            _ => {
                self.items_this_turn += 1;
            }
        }
    }

    fn log_compact_decision(
        &mut self,
        phase: &str,
        reason: &crate::context::compact::CompactReason,
        policy: &CompactPolicy,
    ) {
        self.log(
            &format!("compact_{phase}"),
            &format!(
                "\"reason\":\"{}\",\"trigger\":\"{:?}\",\"mode\":\"{:?}\"",
                reason.log_summary(),
                policy.trigger,
                policy.mode
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::should_abort_for_compaction_followup;
    use crate::context::compact::{CompactMode, CompactReason};
    use crate::lcm::CompactionResult;

    #[test]
    fn forced_followup_mode_always_aborts_after_compaction() {
        let reason = CompactReason::SelfOutputRatio {
            output_tokens: 1200,
            total_context_tokens: 4000,
            actual_pct: 30,
            threshold_pct: 15,
        };
        assert!(should_abort_for_compaction_followup(
            CompactMode::ForcedFollowup,
            &reason,
            None,
        ));
    }

    #[test]
    fn emergency_midtask_aborts_when_durable_compaction_makes_no_progress() {
        let reason = CompactReason::Emergency {
            input_tokens: 99_959,
            context_window: 131_072,
            fill_ratio: 0.763,
        };
        let result = CompactionResult {
            action_taken: false,
            tokens_before: 1467,
            tokens_after: 1467,
            created_summary_ids: Vec::new(),
            rounds: 2,
        };
        assert!(should_abort_for_compaction_followup(
            CompactMode::MidTask,
            &reason,
            Some(&result),
        ));
    }

    #[test]
    fn emergency_midtask_continues_when_durable_compaction_reduces_context() {
        let reason = CompactReason::Emergency {
            input_tokens: 99_959,
            context_window: 131_072,
            fill_ratio: 0.763,
        };
        let result = CompactionResult {
            action_taken: true,
            tokens_before: 1800,
            tokens_after: 900,
            created_summary_ids: vec!["sum_local".to_string()],
            rounds: 1,
        };
        assert!(!should_abort_for_compaction_followup(
            CompactMode::MidTask,
            &reason,
            Some(&result),
        ));
    }
}
