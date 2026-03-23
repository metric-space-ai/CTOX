use anyhow::Context;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const DEFAULT_CODEX_REPO_REF: &str = "c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982";
const DEFAULT_VLLM_SERVE_REPO_REF: &str = "master";

const DISALLOWED_VLLM_SERVE_FUNCTION_TOOLS: &[&str] = &["spawn_agent", "send_input"];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalModelFamily {
    GptOss,
    Qwen35Vision,
}

impl FromStr for LocalModelFamily {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpt_oss" | "gpt-oss" | "gptoss" => Ok(Self::GptOss),
            "qwen3_5" | "qwen3.5" | "qwen3-5" | "qwen35" | "qwen3_5_vision" => {
                Ok(Self::Qwen35Vision)
            }
            other => anyhow::bail!("unsupported clean-room model family: {other}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VendoredDependencyPaths {
    pub references_root: PathBuf,
    pub codex_repo_root: PathBuf,
    pub codex_rs_root: PathBuf,
    pub codex_exec_binary: PathBuf,
    pub ctox_vllm_serve_root: PathBuf,
    pub ctox_vllm_serve_binary: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyBootstrapSpec {
    pub name: String,
    #[serde(rename = "repoUrl")]
    pub repo_url: String,
    #[serde(rename = "targetDir")]
    pub target_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanRoomBootstrapManifest {
    pub version: u32,
    pub goal: String,
    pub dependencies: Vec<DependencyBootstrapSpec>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependencyBootstrapResult {
    pub name: String,
    pub target_dir: PathBuf,
    pub repo_url: String,
    pub git_ref: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependencyBootstrapOutcome {
    pub manifest_path: PathBuf,
    pub results: Vec<DependencyBootstrapResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VllmServeRuntimeConfig {
    pub family: LocalModelFamily,
    pub model: String,
    pub port: u16,
    pub proxy_port: Option<u16>,
    pub max_seq_len: Option<u32>,
    pub max_seqs: u32,
    pub max_batch_size: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VllmServeFamilyProfile {
    pub family: LocalModelFamily,
    pub launcher_mode: String,
    pub arch: Option<String>,
    pub paged_attn: String,
    pub pa_cache_type: Option<String>,
    pub pa_memory_fraction: Option<String>,
    pub pa_context_len: Option<u32>,
    pub max_seq_len: u32,
    pub max_batch_size: u32,
    pub max_seqs: u32,
    pub isq: Option<String>,
    pub tensor_parallel_backend: Option<String>,
    pub disable_nccl: bool,
    pub target_world_size: Option<u32>,
    pub preferred_gpu_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodexExecInvocation {
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyProxyRequest {
    pub model: String,
    pub system_prompt: String,
    pub conversation_items: Vec<Value>,
    pub reasoning_effort: String,
    pub max_output_tokens: usize,
    pub stream: bool,
    pub tools: Vec<HarmonyToolSpec>,
    pub tool_payloads: Vec<Value>,
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HarmonyResponseItem {
    Message(String),
    FunctionCall(HarmonyFunctionCall),
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanRoomBaselinePlan {
    pub dependencies: VendoredDependencyPaths,
    pub runtime: VllmServeRuntimeConfig,
    pub family_profile: VllmServeFamilyProfile,
    pub vllm_serve_command: Vec<String>,
    pub codex_exec_command: Option<Vec<String>>,
    pub bridge_mode: String,
}

pub fn load_clean_room_bootstrap_manifest(
    root: &Path,
) -> anyhow::Result<CleanRoomBootstrapManifest> {
    let path = root.join("contracts/clean_room_bootstrap_manifest.json");
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read clean-room bootstrap manifest at {path:?}"))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse clean-room bootstrap manifest at {path:?}"))
}

pub fn discover_vendored_dependency_paths(root: &Path) -> VendoredDependencyPaths {
    let references_root = root.join("references");
    let codex_repo_root = references_root.join("openai-codex");
    let codex_rs_root = codex_repo_root.join("codex-rs");
    let ctox_vllm_serve_root = root.join("ctox-vllm-serve");
    VendoredDependencyPaths {
        references_root,
        codex_exec_binary: codex_rs_root.join("target/release/codex-exec"),
        codex_repo_root,
        codex_rs_root,
        ctox_vllm_serve_binary: ctox_vllm_serve_root.join("target/release/mistralrs"),
        ctox_vllm_serve_root,
    }
}

pub fn default_runtime_config(family: LocalModelFamily) -> VllmServeRuntimeConfig {
    match family {
        LocalModelFamily::GptOss => VllmServeRuntimeConfig {
            family,
            model: "openai/gpt-oss-20b".to_string(),
            port: 1234,
            proxy_port: Some(12434),
            max_seq_len: Some(131_072),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::Qwen35Vision => VllmServeRuntimeConfig {
            family,
            model: "Qwen/Qwen3.5-27B".to_string(),
            port: 1235,
            proxy_port: None,
            max_seq_len: Some(32_768),
            max_seqs: 1,
            max_batch_size: 1,
        },
    }
}

pub fn default_family_profile(family: LocalModelFamily) -> VllmServeFamilyProfile {
    match family {
        LocalModelFamily::GptOss => VllmServeFamilyProfile {
            family,
            launcher_mode: "text".to_string(),
            arch: Some("gpt_oss".to_string()),
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("f8e4m3".to_string()),
            pa_memory_fraction: Some("0.80".to_string()),
            pa_context_len: None,
            max_seq_len: 131_072,
            max_batch_size: 1,
            max_seqs: 1,
            isq: None,
            tensor_parallel_backend: Some("disabled".to_string()),
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: None,
        },
        LocalModelFamily::Qwen35Vision => VllmServeFamilyProfile {
            family,
            launcher_mode: "vision".to_string(),
            arch: None,
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("f8e4m3".to_string()),
            pa_memory_fraction: Some("0.80".to_string()),
            pa_context_len: None,
            max_seq_len: 32_768,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: Some("disabled".to_string()),
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(3),
        },
    }
}

pub fn build_vllm_serve_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &VllmServeRuntimeConfig,
) -> Vec<String> {
    let family_profile = default_family_profile(runtime.family);
    let mut command = vec![dependencies.ctox_vllm_serve_binary.display().to_string()];
    match runtime.family {
        LocalModelFamily::GptOss => {
            command.extend([
                "serve".to_string(),
                "--port".to_string(),
                runtime.port.to_string(),
                "--max-seqs".to_string(),
                runtime.max_seqs.to_string(),
                "--max-batch-size".to_string(),
                runtime.max_batch_size.to_string(),
                "--paged-attn".to_string(),
                family_profile.paged_attn,
                "-m".to_string(),
                runtime.model.clone(),
                "-a".to_string(),
                family_profile
                    .arch
                    .expect("gpt-oss family profile must provide an arch"),
            ]);
            if let Some(max_seq_len) = runtime.max_seq_len {
                command.extend(["--max-seq-len".to_string(), max_seq_len.to_string()]);
            }
        }
        LocalModelFamily::Qwen35Vision => {
            command.extend([
                "serve".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
                "vision".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
        }
    }
    command
}

pub fn build_codex_exec_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &VllmServeRuntimeConfig,
    invocation: &CodexExecInvocation,
) -> Option<Vec<String>> {
    if runtime.family != LocalModelFamily::GptOss {
        return None;
    }

    let proxy_port = runtime.proxy_port.unwrap_or(runtime.port);
    let provider_config = format!(
        "model_providers.cto_local={{name=\"cto-local\",base_url=\"http://127.0.0.1:{proxy_port}/v1\",wire_api=\"responses\",requires_openai_auth=false}}"
    );
    Some(vec![
        dependencies.codex_exec_binary.display().to_string(),
        "-m".to_string(),
        runtime.model.clone(),
        "--skip-git-repo-check".to_string(),
        "--json".to_string(),
        "-c".to_string(),
        "model_provider=\"cto_local\"".to_string(),
        "-c".to_string(),
        provider_config,
        "-c".to_string(),
        "include_apply_patch_tool=false".to_string(),
        "-c".to_string(),
        "web_search=\"disabled\"".to_string(),
        invocation.prompt.clone(),
    ])
}

pub fn build_clean_room_baseline_plan(
    root: &Path,
    family: LocalModelFamily,
    prompt: String,
) -> CleanRoomBaselinePlan {
    let dependencies = discover_vendored_dependency_paths(root);
    let runtime = default_runtime_config(family);
    let family_profile = default_family_profile(family);
    let vllm_serve_command = build_vllm_serve_command(&dependencies, &runtime);
    let codex_exec_command = build_codex_exec_command(
        &dependencies,
        &runtime,
        &CodexExecInvocation { prompt },
    );
    let bridge_mode = match family {
        LocalModelFamily::GptOss => "codex_responses_proxy".to_string(),
        LocalModelFamily::Qwen35Vision => "qwen_custom_execution".to_string(),
    };
    CleanRoomBaselinePlan {
        dependencies,
        runtime,
        family_profile,
        vllm_serve_command,
        codex_exec_command,
        bridge_mode,
    }
}

// TODO(ctox-adapter): Keep family-specific adapter behavior centralized here.
// GPT-OSS already needs Harmony-specific prompt construction, parsing, and
// turn-shaping rules. Qwen3.5 should get its own family prompt assets and
// response parser instead of being forced through GPT-OSS assumptions.
// The proxy should dispatch by model family and own:
// - family-specific system/developer prompts
// - family-specific request shaping
// - family-specific response parsing / tool-call extraction
// - family-specific multi-turn continuation rules
// This keeps codex-cli pointed at one stable local responses endpoint while
// the proxy manages backend compatibility per family.

pub fn bootstrap_clean_room_dependencies(root: &Path) -> anyhow::Result<DependencyBootstrapOutcome> {
    let manifest = load_clean_room_bootstrap_manifest(root)?;
    let mut results = Vec::new();
    for dependency in manifest.dependencies {
        let git_ref = default_git_ref(&dependency.name).to_string();
        let target_dir = root.join(&dependency.target_dir);
        let action = ensure_git_checkout(&dependency.repo_url, &git_ref, &target_dir)?;
        results.push(DependencyBootstrapResult {
            name: dependency.name,
            target_dir,
            repo_url: dependency.repo_url,
            git_ref,
            action,
        });
    }
    Ok(DependencyBootstrapOutcome {
        manifest_path: root.join("contracts/clean_room_bootstrap_manifest.json"),
        results,
    })
}

fn default_git_ref(name: &str) -> &'static str {
    match name {
        "openai-codex" => DEFAULT_CODEX_REPO_REF,
        "vllm-serve" | "mistral.rs" => DEFAULT_VLLM_SERVE_REPO_REF,
        _ => DEFAULT_CODEX_REPO_REF,
    }
}

fn ensure_git_checkout(repo_url: &str, git_ref: &str, target_dir: &Path) -> anyhow::Result<String> {
    if target_dir.join(".git").exists() {
        ensure_git_worktree_clean(target_dir)?;
        run_git(["-C", path_arg(target_dir), "fetch", "--depth", "1", "origin", git_ref])?;
        run_git(["-C", path_arg(target_dir), "checkout", "--detach", "FETCH_HEAD"])?;
        return Ok("updated".to_string());
    }

    if target_dir.exists() {
        anyhow::bail!(
            "refusing to overwrite non-git clean-room dependency path: {}",
            target_dir.display()
        );
    }

    if let Some(parent) = target_dir.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create dependency parent dir {}", parent.display()))?;
    }

    run_git(["clone", "--no-checkout", repo_url, path_arg(target_dir)])?;
    run_git(["-C", path_arg(target_dir), "fetch", "--depth", "1", "origin", git_ref])?;
    run_git(["-C", path_arg(target_dir), "checkout", "--detach", "FETCH_HEAD"])?;
    Ok("cloned".to_string())
}

fn ensure_git_worktree_clean(target_dir: &Path) -> anyhow::Result<()> {
    let output = Command::new("git")
        .args(["-C", path_arg(target_dir), "status", "--porcelain"])
        .output()
        .context("failed to inspect clean-room dependency git status")?;
    if !output.status.success() {
        anyhow::bail!(
            "git status failed while checking clean-room dependency {}",
            target_dir.display()
        );
    }

    if !String::from_utf8_lossy(&output.stdout).trim().is_empty() {
        anyhow::bail!(
            "refusing to update dirty clean-room dependency {}; commit or stash local vendor changes first",
            target_dir.display()
        );
    }

    Ok(())
}

fn run_git<const N: usize>(args: [&str; N]) -> anyhow::Result<()> {
    let status = Command::new("git")
        .args(args)
        .status()
        .context("failed to launch git for clean-room dependency bootstrap")?;
    if !status.success() {
        anyhow::bail!("git command failed during clean-room dependency bootstrap");
    }
    Ok(())
}

fn path_arg(path: &Path) -> &str {
    path.to_str()
        .expect("clean-room dependency path should remain valid UTF-8")
}

pub fn rewrite_vllm_serve_responses_request(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut payload: Value = serde_json::from_slice(raw).context("failed to parse responses request")?;

    if let Some(tools) = payload.get_mut("tools").and_then(Value::as_array_mut) {
        let mut rewritten = Vec::new();
        for tool in tools.drain(..) {
            if let Some(tool) = rewrite_tool(tool) {
                rewritten.push(tool);
            }
        }
        *tools = rewritten;
    }

    if let Some(input) = payload.get("input").cloned() {
        if let Some(flattened) = flatten_input_items(&input) {
            payload["input"] = Value::String(flattened);
        }
    }

    if let Some(instructions) = payload
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
    {
        let prefix = format!("[instructions]\n{instructions}");
        let merged = match payload.get("input").and_then(Value::as_str) {
            Some(existing) if !existing.trim().is_empty() => format!("{prefix}\n\n{existing}"),
            _ => prefix,
        };
        payload["input"] = Value::String(merged);
        if let Some(object) = payload.as_object_mut() {
            object.remove("instructions");
        }
    }

    if payload.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        payload["parallel_tool_calls"] = Value::Bool(true);
    }
    if let Some(object) = payload.as_object_mut() {
        object.remove("max_tool_calls");
    }

    serde_json::to_vec(&payload).context("failed to encode rewritten responses request")
}

pub fn should_use_gpt_oss_harmony_proxy(raw: &[u8]) -> anyhow::Result<bool> {
    Ok(parse_harmony_proxy_request(raw)
        .map(|request| is_gpt_oss_model_id(&request.model))
        .unwrap_or(false))
}

pub fn responses_request_streams(raw: &[u8]) -> anyhow::Result<bool> {
    let payload: Value = serde_json::from_slice(raw).context("failed to parse responses request")?;
    Ok(payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

pub fn rewrite_responses_to_gpt_oss_completion(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let request = parse_harmony_proxy_request(raw)?;
    let completion_payload = serde_json::json!({
        "model": request.model,
        "prompt": build_gpt_oss_harmony_prompt(
            &request.system_prompt,
            &request.conversation_items,
            &request.reasoning_effort,
            &request.tools
        ),
        "max_tokens": request.max_output_tokens,
        "temperature": 0.0,
        "stream": false,
        "tools": request.tool_payloads,
        "tool_choice": request.tool_choice.unwrap_or_else(|| json!("auto"))
    });
    serde_json::to_vec(&completion_payload).context("failed to encode GPT-OSS completion payload")
}

pub fn rewrite_gpt_oss_completion_to_responses(
    raw: &[u8],
    fallback_model: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let payload: Value = serde_json::from_slice(raw).context("failed to parse completion response")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(fallback_model)
        .unwrap_or("openai/gpt-oss-20b")
        .to_string();
    let response_id = payload
        .get("id")
        .and_then(Value::as_str)
        .map(|value| format!("resp_{value}"))
        .unwrap_or_else(|| "resp_ctox_proxy".to_string());
    let created_at = payload
        .get("created")
        .and_then(Value::as_u64)
        .unwrap_or_else(current_unix_ts);
    let raw_text = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let items = parse_harmony_response_items(raw_text);
    let output_text = items.iter().find_map(|item| match item {
        HarmonyResponseItem::Message(text) if !text.trim().is_empty() => Some(text.clone()),
        _ => None,
    });
    let output = items
        .into_iter()
        .map(harmony_item_to_responses_output)
        .collect::<Vec<_>>();
    let usage = payload.get("usage").cloned().unwrap_or_else(|| {
        serde_json::json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    });
    let response_payload = serde_json::json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": current_unix_ts(),
        "model": model,
        "status": "completed",
        "output": output,
        "output_text": output_text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0)
        }
    });
    serde_json::to_vec(&response_payload).context("failed to encode responses-compatible payload")
}

pub fn rewrite_gpt_oss_completion_to_sse(
    raw: &[u8],
    fallback_model: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let payload: Value = serde_json::from_slice(raw).context("failed to parse completion response")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(fallback_model)
        .unwrap_or("openai/gpt-oss-20b");
    let response_id = payload
        .get("id")
        .and_then(Value::as_str)
        .map(|value| format!("resp_{value}"))
        .unwrap_or_else(|| "resp_ctox_proxy".to_string());
    let raw_text = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let items = parse_harmony_response_items(raw_text);
    let usage = payload.get("usage").cloned().unwrap_or_else(|| {
        json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    });
    let mut frames = Vec::new();
    frames.push((
        "response.created",
        json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "model": model
            }
        })
        .to_string(),
    ));
    for item in items {
        frames.push((
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "item": harmony_item_to_responses_output(item)
            })
            .to_string(),
        ));
    }
    frames.push((
        "response.completed",
        json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "model": model,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)),
                    "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)),
                    "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0)
                }
            }
        })
        .to_string(),
    ));
    Ok(frames
        .into_iter()
        .map(|(event, frame)| format!("event: {event}\ndata: {frame}\n\n"))
        .chain(std::iter::once("data: [DONE]\n\n".to_string()))
        .collect::<String>()
        .into_bytes())
}

pub fn gpt_oss_completion_needs_followup(raw: &[u8]) -> anyhow::Result<bool> {
    let payload: Value = serde_json::from_slice(raw).context("failed to parse completion response")?;
    let raw_text = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let items = parse_harmony_response_items(raw_text);
    Ok(items.is_empty() && raw_text.contains("<|channel|>analysis<|message|>"))
}

pub fn build_gpt_oss_followup_completion_request(
    initial_request_raw: &[u8],
    first_completion_raw: &[u8],
) -> anyhow::Result<Option<Vec<u8>>> {
    if !gpt_oss_completion_needs_followup(first_completion_raw)? {
        return Ok(None);
    }

    let mut request: Value =
        serde_json::from_slice(initial_request_raw).context("failed to parse initial completion request")?;
    let first_payload: Value =
        serde_json::from_slice(first_completion_raw).context("failed to parse first completion response")?;
    let first_text = first_payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let prompt = request
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    request["prompt"] = Value::String(format!("{prompt}{first_text}<|end|><|return|>"));
    Ok(Some(serde_json::to_vec(&request)?))
}

fn rewrite_tool(tool: Value) -> Option<Value> {
    let object = tool.as_object()?;
    let tool_type = object.get("type")?.as_str()?;
    match tool_type {
        "function" => {
            if let Some(function) = object.get("function").and_then(Value::as_object) {
                let name = function.get("name").and_then(Value::as_str)?;
                if DISALLOWED_VLLM_SERVE_FUNCTION_TOOLS.contains(&name) {
                    return None;
                }
                return Some(Value::Object(object.clone()));
            }

            let name = object.get("name").and_then(Value::as_str)?;
            if DISALLOWED_VLLM_SERVE_FUNCTION_TOOLS.contains(&name) {
                return None;
            }
            let mut function_payload = serde_json::Map::new();
            for (key, value) in object {
                if key == "type" || key == "function" {
                    continue;
                }
                function_payload.insert(key.clone(), value.clone());
            }
            Some(serde_json::json!({
                "type": "function",
                "function": function_payload,
            }))
        }
        "namespace" => {
            let children = object.get("tools")?.as_array()?;
            let rewritten_children: Vec<Value> = children
                .iter()
                .filter_map(|child| rewrite_tool(child.clone()))
                .collect();
            if rewritten_children.is_empty() {
                return None;
            }
            let mut rewritten = object.clone();
            rewritten.insert("tools".to_string(), Value::Array(rewritten_children));
            Some(Value::Object(rewritten))
        }
        _ => None,
    }
}

fn parse_harmony_proxy_request(raw: &[u8]) -> anyhow::Result<HarmonyProxyRequest> {
    let payload: Value = serde_json::from_slice(raw).context("failed to parse responses request")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("openai/gpt-oss-20b")
        .to_string();
    let system_prompt = payload
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let conversation_items = normalize_responses_input(payload.get("input"));
    let reasoning_effort = payload
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str)
        .unwrap_or("medium")
        .to_string();
    let max_output_tokens = payload
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(1024);
    let stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let tool_payloads = payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(|tool| rewrite_tool(tool.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tools = tool_payloads
        .iter()
        .filter_map(parse_harmony_tool_spec)
        .collect::<Vec<_>>();
    let tool_choice = payload.get("tool_choice").cloned();
    Ok(HarmonyProxyRequest {
        model,
        system_prompt,
        conversation_items,
        reasoning_effort,
        max_output_tokens,
        stream,
        tools,
        tool_payloads,
        tool_choice,
    })
}

fn parse_harmony_tool_spec(tool: &Value) -> Option<HarmonyToolSpec> {
    let tool_type = tool.get("type").and_then(Value::as_str)?;
    if tool_type != "function" {
        return None;
    }
    let function = tool.get("function").unwrap_or(tool);
    let name = function.get("name").and_then(Value::as_str)?.to_string();
    if DISALLOWED_VLLM_SERVE_FUNCTION_TOOLS.contains(&name.as_str()) {
        return None;
    }
    Some(HarmonyToolSpec {
        name,
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        parameters: function.get("parameters").cloned(),
    })
}

fn is_gpt_oss_model_id(model_id: &str) -> bool {
    let lowered = model_id.trim().to_ascii_lowercase();
    lowered == "gpt-oss-20b" || lowered == "openai/gpt-oss-20b" || lowered.contains("gpt-oss")
}

fn sanitize_reasoning_effort(value: &str) -> &str {
    match value.trim().to_ascii_lowercase().as_str() {
        "minimal" | "low" => "low",
        "medium" => "medium",
        "high" => "high",
        _ => "medium",
    }
}

fn build_gpt_oss_harmony_prompt(
    system_prompt: &str,
    conversation_items: &[Value],
    reasoning_effort: &str,
    tools: &[HarmonyToolSpec],
) -> String {
    let current_date = "2026-03-22";
    let reasoning_effort = sanitize_reasoning_effort(reasoning_effort);
    let developer_block = build_harmony_developer_block(system_prompt, tools);
    let system_tool_hint = if tools.is_empty() {
        ""
    } else {
        "\nCalls to these tools must go to the commentary channel: 'functions'."
    };
    let assistant_prefix = if tools.is_empty() {
        "<|start|>assistant<|channel|>final<|message|>"
    } else {
        "<|start|>assistant"
    };
    let conversation = render_harmony_conversation(conversation_items);
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
Knowledge cutoff: 2024-06\n\
Current date: {current_date}\n\n\
Reasoning: {reasoning_effort}\n\n\
# Valid channels: analysis, commentary, final. Channel must be included for every message.{system_tool_hint}<|end|>\
<|start|>developer<|message|>{developer_block}<|end|>\
{conversation}\
{assistant_prefix}",
        current_date = current_date,
        reasoning_effort = reasoning_effort,
        system_tool_hint = system_tool_hint,
        developer_block = developer_block,
        conversation = conversation,
        assistant_prefix = assistant_prefix,
    )
}

fn render_harmony_conversation(conversation_items: &[Value]) -> String {
    if conversation_items.is_empty() {
        return "<|start|>user<|message|><|end|>".to_string();
    }

    let mut rendered = String::new();
    for item in conversation_items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match item_type {
            "message" => {
                let role = object
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user");
                let text = extract_message_content_text(object.get("content"));
                if text.trim().is_empty() {
                    continue;
                }
                match role {
                    "assistant" => {
                        rendered.push_str("<|start|>assistant<|channel|>final<|message|>");
                        rendered.push_str(text.trim());
                        rendered.push_str("<|end|>");
                    }
                    "tool" => {
                        rendered.push_str("<|start|>tool<|message|>");
                        rendered.push_str(text.trim());
                        rendered.push_str("<|end|>");
                    }
                    _ => {
                        rendered.push_str("<|start|>user<|message|>");
                        rendered.push_str(text.trim());
                        rendered.push_str("<|end|>");
                    }
                }
            }
            "function_call" => {
                let name = object.get("name").and_then(Value::as_str).unwrap_or_default();
                let arguments = object
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                if name.is_empty() {
                    continue;
                }
                rendered.push_str("<|start|>assistant<|channel|>commentary to=functions.");
                rendered.push_str(name);
                rendered.push_str("<|constrain|>json<|message|>");
                rendered.push_str(arguments.trim());
                rendered.push_str("<|call|>");
            }
            "function_call_output" => {
                let text = extract_function_call_output_text(object.get("output"));
                if text.trim().is_empty() {
                    continue;
                }
                rendered.push_str("<|start|>tool<|message|>");
                rendered.push_str(text.trim());
                rendered.push_str("<|end|>");
            }
            _ => {}
        }
    }

    if rendered.is_empty() {
        "<|start|>user<|message|><|end|>".to_string()
    } else {
        rendered
    }
}

fn extract_message_content_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    if let Some(entries) = content.as_array() {
        let mut parts = Vec::new();
        for entry in entries {
            if let Some(text) = entry.get("text").and_then(Value::as_str) {
                if !text.trim().is_empty() {
                    parts.push(text.to_string());
                }
            }
        }
        return parts.join("\n");
    }
    String::new()
}

fn extract_function_call_output_text(output: Option<&Value>) -> String {
    fn render(value: &Value) -> Option<String> {
        match value {
            Value::String(text) => Some(text.clone()),
            Value::Array(items) => {
                let parts = items
                    .iter()
                    .filter_map(render)
                    .filter(|text| !text.trim().is_empty())
                    .collect::<Vec<_>>();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join("\n"))
                }
            }
            Value::Object(map) => {
                if let Some(text) = map.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
                if let Some(content) = map.get("content") {
                    return render(content);
                }
                serde_json::to_string(value).ok()
            }
            _ => serde_json::to_string(value).ok(),
        }
    }
    output.and_then(render).unwrap_or_default()
}

fn normalize_responses_input(input: Option<&Value>) -> Vec<Value> {
    match input {
        None => Vec::new(),
        Some(Value::String(text)) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": text }]
        })],
        Some(Value::Array(items)) => items.clone(),
        Some(other) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": other.to_string() }]
        })],
    }
}

pub fn materialize_responses_request(
    raw: &[u8],
    previous_conversation: Option<&Value>,
) -> anyhow::Result<Value> {
    let mut payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request for materialization")?;
    let current_items = normalize_responses_input(payload.get("input"));
    let merged_items = if let Some(previous) = previous_conversation.and_then(Value::as_array) {
        let mut merged = previous.clone();
        merged.extend(current_items);
        merged
    } else {
        current_items
    };
    if let Some(object) = payload.as_object_mut() {
        object.insert("input".to_string(), Value::Array(merged_items));
        object.remove("previous_response_id");
    }
    Ok(payload)
}

pub fn extend_conversation_with_response(
    request_payload: &Value,
    response_payload: &Value,
) -> anyhow::Result<Value> {
    let mut conversation = normalize_responses_input(request_payload.get("input"));
    if let Some(output_items) = response_payload.get("output").and_then(Value::as_array) {
        conversation.extend(output_items.iter().cloned());
    }
    Ok(Value::Array(conversation))
}

fn build_harmony_developer_block(system_prompt: &str, tools: &[HarmonyToolSpec]) -> String {
    let mut block = String::from("# Instructions\n\n");
    let trimmed = system_prompt.trim();
    if !trimmed.is_empty() {
        block.push_str(trimmed);
        block.push_str("\n\n");
    }
    if !tools.is_empty() {
        block.push_str("# Tools\n\n");
        block.push_str("When tools are available, emit exactly one next step per completion.\n");
        block.push_str("Either emit one tool call on the commentary channel or one final answer on the final channel.\n");
        block.push_str("Do not emit multiple tool calls in a single completion.\n");
        block.push_str("After emitting a tool call, stop immediately.\n\n");
        block.push_str("## functions\n\n");
        block.push_str("namespace functions {\n");
        for tool in tools {
            block.push_str(&render_harmony_tool_signature(tool));
            block.push('\n');
        }
        block.push_str("} // namespace functions\n");
    }
    block.trim_end().to_string()
}

fn json_schema_to_typescript(schema: &Value) -> String {
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(key, value)| {
                    let optional = if required.iter().any(|item| item == key) {
                        ""
                    } else {
                        "?"
                    };
                    let mut line = String::new();
                    if let Some(description) = value.get("description").and_then(Value::as_str) {
                        line.push_str("// ");
                        line.push_str(description.trim());
                        line.push('\n');
                    }
                    line.push_str(&format!("{key}{optional}: {}", json_schema_type_to_typescript(value)));
                    if let Some(default) = value.get("default") {
                        line.push_str(&format!(", // default: {}", default));
                    }
                    line
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    if props.trim().is_empty() {
        "() => any".to_string()
    } else {
        format!("(_: {{\n{props}\n}}) => any")
    }
}

fn json_schema_type_to_typescript(schema: &Value) -> String {
    match schema.get("enum").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(format!("\"{text}\"")),
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" | "),
        _ => match schema.get("type").and_then(Value::as_str) {
            Some("string") => "string".to_string(),
            Some("integer") | Some("number") => "number".to_string(),
            Some("boolean") => "boolean".to_string(),
            Some("array") => {
                let item_ty = schema
                    .get("items")
                    .map(json_schema_type_to_typescript)
                    .unwrap_or_else(|| "any".to_string());
                format!("{item_ty}[]")
            }
            Some("object") => "Record<string, any>".to_string(),
            _ => "any".to_string(),
        },
    }
}

fn render_harmony_tool_signature(tool: &HarmonyToolSpec) -> String {
    let mut rendered = String::new();
    if let Some(description) = &tool.description {
        rendered.push_str("// ");
        rendered.push_str(description.trim());
        rendered.push('\n');
    }
    rendered.push_str("type ");
    rendered.push_str(&tool.name);
    rendered.push_str(" = ");
    rendered.push_str(
        &tool
            .parameters
            .as_ref()
            .map(json_schema_to_typescript)
            .unwrap_or_else(|| "() => any".to_string()),
    );
    rendered.push_str(";\n");
    rendered
}

fn sanitize_harmony_completion_text(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if let Some(idx) = text.rfind("<|message|>") {
        text = text[idx + "<|message|>".len()..].to_string();
    }
    if let Some(idx) = text.find("<|return|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|end|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|start|>") {
        text.truncate(idx);
    }
    text.trim().to_string()
}

fn parse_harmony_response_items(raw_text: &str) -> Vec<HarmonyResponseItem> {
    let mut items = Vec::new();
    if let Some(call) = parse_harmony_function_call(raw_text) {
        items.push(HarmonyResponseItem::FunctionCall(call));
        return items;
    }
    let text = extract_harmony_message_text(raw_text);
    if !text.trim().is_empty() {
        items.push(HarmonyResponseItem::Message(text));
    }
    items
}

fn parse_harmony_function_call(raw_text: &str) -> Option<HarmonyFunctionCall> {
    let marker = "<|channel|>commentary to=functions.";
    let channel_idx = raw_text.find(marker)?;
    let name_start = channel_idx + marker.len();
    let name_end = raw_text[name_start..]
        .find(|c: char| c == '<' || c.is_whitespace())
        .map(|offset| name_start + offset)
        .unwrap_or(raw_text.len());
    let name = raw_text[name_start..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let message_token = "<|message|>";
    let message_start = raw_text[name_end..]
        .find(message_token)
        .map(|offset| name_end + offset + message_token.len())?;
    let message_end = raw_text[message_start..]
        .find("<|call|>")
        .map(|offset| message_start + offset)
        .or_else(|| {
            raw_text[message_start..]
                .find("<|end|>")
                .map(|offset| message_start + offset)
        })
        .unwrap_or(raw_text.len());
    let arguments = raw_text[message_start..message_end].trim();
    if arguments.is_empty() {
        return None;
    }
    Some(HarmonyFunctionCall {
        call_id: format!("call_ctox_{}", current_unix_ts()),
        name: name.to_string(),
        arguments: normalize_function_call_arguments(name, arguments),
    })
}

fn normalize_function_call_arguments(name: &str, arguments: &str) -> String {
    let mut value = match serde_json::from_str::<Value>(arguments) {
        Ok(value) => value,
        Err(_) => match name {
            "exec_command" => parse_relaxed_exec_command_arguments(arguments)
                .unwrap_or_else(|| Value::String(arguments.to_string())),
            "write_stdin" => parse_relaxed_write_stdin_arguments(arguments)
                .unwrap_or_else(|| Value::String(arguments.to_string())),
            _ => Value::String(arguments.to_string()),
        },
    };

    if name == "exec_command" {
        if let Some(object) = value.as_object_mut() {
            if let Some(cmd_items) = object.get("cmd").and_then(Value::as_array) {
                if let Some(rewritten) = normalize_exec_command_array(cmd_items) {
                    for (key, rewritten_value) in rewritten {
                        object.insert(key, rewritten_value);
                    }
                } else {
                    let joined = cmd_items
                        .iter()
                        .map(|item| {
                            item.as_str()
                                .map(shell_escape)
                                .unwrap_or_else(|| shell_escape(&item.to_string()))
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    object.insert("cmd".to_string(), Value::String(joined));
                }
            }
        }
    }

    match value {
        Value::Object(_) => serde_json::to_string(&value).unwrap_or_else(|_| arguments.to_string()),
        Value::String(text) => serde_json::to_string(&json!({ "cmd": text }))
            .unwrap_or_else(|_| arguments.to_string()),
        _ => serde_json::to_string(&value).unwrap_or_else(|_| arguments.to_string()),
    }
}

fn shell_escape(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':'))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn normalize_exec_command_array(cmd_items: &[Value]) -> Option<Vec<(String, Value)>> {
    let first = cmd_items.first()?.as_str()?;
    if matches!(first, "bash" | "/bin/bash" | "sh" | "/bin/sh")
        && cmd_items.get(1).and_then(Value::as_str) == Some("-lc")
    {
        if let Some(script) = cmd_items.get(2).and_then(Value::as_str) {
            return Some(vec![
                ("cmd".to_string(), Value::String(script.to_string())),
                (
                    "shell".to_string(),
                    Value::String(if first.contains("bash") { "bash" } else { "sh" }.to_string()),
                ),
                ("login".to_string(), Value::Bool(false)),
            ]);
        }
    }
    if first == "apply_patch" {
        if let Some(patch) = cmd_items.get(1).and_then(Value::as_str) {
            return Some(vec![(
                "cmd".to_string(),
                Value::String(format!("apply_patch <<'PATCH'\n{patch}\nPATCH")),
            )]);
        }
    }
    None
}

fn parse_relaxed_exec_command_arguments(arguments: &str) -> Option<Value> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return Some(json!({ "cmd": trimmed }));
    }

    let cmd = extract_relaxed_json_string_field(trimmed, "cmd")
        .or_else(|| extract_relaxed_json_array_field(trimmed, "cmd").map(Value::Array))?;

    let mut object = serde_json::Map::new();
    object.insert("cmd".to_string(), cmd);

    for key in [
        "workdir",
        "shell",
        "justification",
        "sandbox_permissions",
        "prefix_rule",
    ] {
        if let Some(value) = extract_relaxed_json_string_field(trimmed, key) {
            object.insert(key.to_string(), value);
        } else if let Some(value) = extract_relaxed_json_array_field(trimmed, key) {
            object.insert(key.to_string(), Value::Array(value));
        }
    }

    for key in ["yield_time_ms", "max_output_tokens"] {
        if let Some(value) = extract_relaxed_json_integer_field(trimmed, key) {
            object.insert(key.to_string(), Value::Number(value.into()));
        }
    }

    for key in ["login", "tty"] {
        if let Some(value) = extract_relaxed_json_bool_field(trimmed, key) {
            object.insert(key.to_string(), Value::Bool(value));
        }
    }

    Some(Value::Object(object))
}

fn parse_relaxed_write_stdin_arguments(arguments: &str) -> Option<Value> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() || !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return None;
    }

    let session_id = extract_relaxed_json_integer_field(trimmed, "session_id")?;
    let mut object = serde_json::Map::new();
    object.insert("session_id".to_string(), Value::Number(session_id.into()));

    if let Some(value) = extract_relaxed_json_string_field(trimmed, "chars") {
        object.insert("chars".to_string(), value);
    }
    if let Some(value) = extract_relaxed_json_integer_field(trimmed, "yield_time_ms") {
        object.insert("yield_time_ms".to_string(), Value::Number(value.into()));
    }
    if let Some(value) = extract_relaxed_json_integer_field(trimmed, "max_output_tokens") {
        object.insert("max_output_tokens".to_string(), Value::Number(value.into()));
    }

    Some(Value::Object(object))
}

fn extract_relaxed_json_string_field(source: &str, key: &str) -> Option<Value> {
    let key_re = Regex::new(&format!(r#""{}"\s*:"# , regex::escape(key))).ok()?;
    let key_match = key_re.find(source)?;
    let remainder = &source[key_match.end()..];
    let value = remainder.trim_start();
    if !value.starts_with('"') {
        return None;
    }
    let value = &value[1..];
    let value_end = find_relaxed_string_end(value);
    let content = value[..value_end].trim_end_matches('"');
    Some(Value::String(unescape_relaxed_string(content)))
}

fn extract_relaxed_json_array_field(source: &str, key: &str) -> Option<Vec<Value>> {
    let key_re = Regex::new(&format!(r#""{}"\s*:"# , regex::escape(key))).ok()?;
    let key_match = key_re.find(source)?;
    let remainder = &source[key_match.end()..];
    let value = remainder.trim_start();
    if !value.starts_with('[') {
        return None;
    }
    let end = find_matching_bracket(value, '[', ']')?;
    let array_text = &value[..=end];
    serde_json::from_str::<Vec<Value>>(array_text).ok()
}

fn extract_relaxed_json_integer_field(source: &str, key: &str) -> Option<i64> {
    let re = Regex::new(&format!(r#""{}"\s*:\s*(-?\d+)"#, regex::escape(key))).ok()?;
    re.captures(source)?
        .get(1)?
        .as_str()
        .parse::<i64>()
        .ok()
}

fn extract_relaxed_json_bool_field(source: &str, key: &str) -> Option<bool> {
    let re = Regex::new(&format!(r#""{}"\s*:\s*(true|false)"#, regex::escape(key))).ok()?;
    Some(re.captures(source)?.get(1)?.as_str() == "true")
}

fn find_relaxed_string_end(source: &str) -> usize {
    let mut escaped = false;
    let bytes = source.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        if escaped {
            escaped = false;
            idx += 1;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            idx += 1;
            continue;
        }
        if ch == '"' {
            let tail = source[idx + 1..].trim_start();
            if tail.starts_with(',') || tail.starts_with('}') {
                return idx;
            }
        }
        idx += 1;
    }
    source.len()
}

fn find_matching_bracket(source: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in source.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn unescape_relaxed_string(text: &str) -> String {
    serde_json::from_str::<String>(&format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\"")))
        .unwrap_or_else(|_| text.to_string())
}

fn extract_harmony_message_text(raw_text: &str) -> String {
    let final_token = "<|channel|>final<|message|>";
    let commentary_token = "<|channel|>commentary<|message|>";
    for token in [final_token, commentary_token] {
        if let Some(start) = raw_text.find(token) {
            let content_start = start + token.len();
            let content_end = raw_text[content_start..]
                .find("<|end|>")
                .map(|offset| content_start + offset)
                .unwrap_or(raw_text.len());
            let text = sanitize_harmony_completion_text(&raw_text[content_start..content_end]);
            if !text.trim().is_empty() {
                return text;
            }
        }
    }
    sanitize_harmony_completion_text(raw_text)
}

fn harmony_item_to_responses_output(item: HarmonyResponseItem) -> Value {
    match item {
        HarmonyResponseItem::Message(text) => json!({
            "type": "message",
            "id": "msg_ctox_proxy",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                }
            ]
        }),
        HarmonyResponseItem::FunctionCall(call) => json!({
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments
        }),
    }
}

fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn flatten_input_items(input: &Value) -> Option<String> {
    let items = input.as_array()?;
    let mut parts = Vec::new();
    for item in items {
        let object = item.as_object()?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .or_else(|| object.get("type").and_then(Value::as_str))
            .unwrap_or("message");
        let content = object.get("content")?;
        let mut chunks = Vec::new();
        if let Some(text) = content.as_str() {
            chunks.push(text.to_string());
        } else if let Some(entries) = content.as_array() {
            for part in entries {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        chunks.push(text.to_string());
                    }
                }
            }
        }
        if !chunks.is_empty() {
            parts.push(format!("[{role}]\n{}", chunks.join("\n")));
        }
    }
    Some(parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_runtime_uses_vllm_serve_gpt_oss_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::GptOss);
        let command = build_vllm_serve_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert!(command.iter().any(|part| part == "gpt_oss"));
        assert!(command.iter().any(|part| part == "--max-seq-len"));
    }

    #[test]
    fn qwen_runtime_uses_vllm_serve_vision_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::Qwen35Vision);
        let command = build_vllm_serve_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert_eq!(command[2], "-p");
        assert_eq!(command[4], "vision");
        assert!(command.iter().any(|part| part == "Qwen/Qwen3.5-27B"));
    }

    #[test]
    fn family_profiles_drive_nccl_policy() {
        let gpt_oss = default_family_profile(LocalModelFamily::GptOss);
        let qwen = default_family_profile(LocalModelFamily::Qwen35Vision);
        assert!(gpt_oss.disable_nccl);
        assert_eq!(gpt_oss.tensor_parallel_backend.as_deref(), Some("disabled"));
        assert!(qwen.disable_nccl);
        assert_eq!(qwen.tensor_parallel_backend.as_deref(), Some("disabled"));
        assert_eq!(qwen.target_world_size, None);
        assert_eq!(qwen.preferred_gpu_count, Some(3));
    }

    #[test]
    fn clean_room_baseline_plan_exposes_family_profile() {
        let plan = build_clean_room_baseline_plan(
            Path::new("/tmp/ctox"),
            LocalModelFamily::Qwen35Vision,
            "ignored".to_string(),
        );
        assert_eq!(plan.family_profile.family, LocalModelFamily::Qwen35Vision);
        assert_eq!(plan.family_profile.launcher_mode, "vision");
        assert_eq!(plan.family_profile.tensor_parallel_backend.as_deref(), Some("disabled"));
    }

    #[test]
    fn gpt_oss_codex_exec_plan_uses_responses_proxy() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::GptOss);
        let invocation = CodexExecInvocation {
            prompt: "Reply with OK".to_string(),
        };
        let command = build_codex_exec_command(&deps, &runtime, &invocation).unwrap();
        assert!(command
            .iter()
            .any(|part| part.contains("wire_api=\"responses\"")));
        assert!(command
            .iter()
            .any(|part| part == "include_apply_patch_tool=false"));
        assert!(command.iter().any(|part| part == "web_search=\"disabled\""));
    }

    #[test]
    fn qwen_plan_has_no_direct_codex_exec_baseline() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::Qwen35Vision);
        let invocation = CodexExecInvocation {
            prompt: "ignored".to_string(),
        };
        assert!(build_codex_exec_command(&deps, &runtime, &invocation).is_none());
    }

    #[test]
    fn responses_rewrite_wraps_and_filters_tools() {
        let payload = serde_json::json!({
            "tools": [
                {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
                {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}},
                {"type": "web_search"},
            ],
            "parallel_tool_calls": false,
            "max_tool_calls": 1,
        });
        let rewritten = rewrite_vllm_serve_responses_request(
            serde_json::to_vec(&payload).unwrap().as_slice(),
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            value["tools"],
            serde_json::json!([
                {"type":"function","function":{"name":"exec_command","parameters":{"type":"object"}}}
            ])
        );
        assert_eq!(value["parallel_tool_calls"], Value::Bool(true));
        assert!(value.get("max_tool_calls").is_none());
    }

    #[test]
    fn responses_rewrite_flattens_input_and_merges_instructions() {
        let payload = serde_json::json!({
            "instructions": "System rule",
            "input": [
                {"role": "developer", "content": [{"text": "Dev text"}]},
                {"role": "user", "content": [{"text": "User text"}]}
            ]
        });
        let rewritten = rewrite_vllm_serve_responses_request(
            serde_json::to_vec(&payload).unwrap().as_slice(),
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            value["input"],
            Value::String(
                "[instructions]\nSystem rule\n\n[developer]\nDev text\n\n[user]\nUser text"
                    .to_string()
            )
        );
        assert!(value.get("instructions").is_none());
    }

    #[test]
    fn detects_gpt_oss_harmony_proxy_need() {
        let payload = serde_json::json!({"model":"openai/gpt-oss-20b"});
        assert!(should_use_gpt_oss_harmony_proxy(&serde_json::to_vec(&payload).unwrap()).unwrap());
    }

    #[test]
    fn detects_streaming_request_flag() {
        let payload = serde_json::json!({"model":"openai/gpt-oss-20b","stream":true});
        assert!(responses_request_streams(&serde_json::to_vec(&payload).unwrap()).unwrap());
    }

    #[test]
    fn translates_responses_request_to_gpt_oss_completion() {
        let payload = serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "instructions": "System rules",
            "input": [
                {"role":"user","content":[{"text":"Do the thing"}]}
            ],
            "max_output_tokens": 333,
            "reasoning": {"effort":"high"}
        });
        let rewritten = rewrite_responses_to_gpt_oss_completion(&serde_json::to_vec(&payload).unwrap()).unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "openai/gpt-oss-20b");
        assert_eq!(value["max_tokens"], 333);
        assert!(value["prompt"].as_str().unwrap().contains("System rules"));
        assert!(value["prompt"].as_str().unwrap().contains("Do the thing"));
        assert!(value["prompt"].as_str().unwrap().contains("<|start|>assistant"));
    }

    #[test]
    fn translates_harmony_completion_back_to_responses() {
        let payload = serde_json::json!({
            "id":"123",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"<|start|>assistant<|channel|>final<|message|>CTOX_OK<|end|>"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten = rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None).unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "CTOX_OK");
        assert_eq!(value["output"][0]["content"][0]["text"], "CTOX_OK");
        assert_eq!(value["usage"]["input_tokens"], 11);
        assert_eq!(value["usage"]["output_tokens"], 7);
    }

    #[test]
    fn translates_harmony_tool_call_back_to_responses() {
        let payload = serde_json::json!({
            "id":"456",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"<|channel|>commentary to=functions.shell_command<|constrain|>json<|message|>{\"command\":\"printf CTOX_TOOL\"}<|call|>"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten = rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None).unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert!(value["output_text"].is_null());
        assert_eq!(value["output"][0]["type"], "function_call");
        assert_eq!(value["output"][0]["name"], "shell_command");
    }

    #[test]
    fn normalizes_relaxed_exec_command_multiline_string_arguments() {
        let raw = "{\n\"cmd\": \"apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch\nPATCH\"\n}";
        let normalized = normalize_function_call_arguments("exec_command", raw);
        let value: Value = serde_json::from_str(&normalized).unwrap();
        assert_eq!(value["cmd"].as_str().unwrap(), "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch\nPATCH");
    }

    #[test]
    fn normalizes_relaxed_exec_command_array_arguments() {
        let raw = r#"{"cmd":["bash","-lc","printf CTOX_OK"],"yield_time_ms":1000}"#;
        let normalized = normalize_function_call_arguments("exec_command", raw);
        let value: Value = serde_json::from_str(&normalized).unwrap();
        assert_eq!(value["cmd"], "printf CTOX_OK");
        assert_eq!(value["shell"], "bash");
        assert_eq!(value["login"], false);
        assert_eq!(value["yield_time_ms"], 1000);
    }

    #[test]
    fn normalizes_apply_patch_array_to_heredoc_command() {
        let raw = r#"{"cmd":["apply_patch","*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch"]}"#;
        let normalized = normalize_function_call_arguments("exec_command", raw);
        let value: Value = serde_json::from_str(&normalized).unwrap();
        assert_eq!(
            value["cmd"].as_str().unwrap(),
            "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch\nPATCH"
        );
    }

    #[test]
    fn translates_harmony_completion_to_sse() {
        let payload = serde_json::json!({
            "id":"789",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"<|channel|>final<|message|>CTOX_STREAM_OK<|end|>"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten = rewrite_gpt_oss_completion_to_sse(&serde_json::to_vec(&payload).unwrap(), None).unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.contains("\"type\":\"response.created\""));
        assert!(text.contains("\"type\":\"response.output_item.done\""));
        assert!(text.contains("\"CTOX_STREAM_OK\""));
        assert!(text.contains("\"type\":\"response.completed\""));
    }
}
