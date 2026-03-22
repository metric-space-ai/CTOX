use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;

const DEFAULT_CODEX_REPO_REF: &str = "main";
const DEFAULT_MISTRALRS_REPO_REF: &str = "master";

const DISALLOWED_MISTRALRS_FUNCTION_TOOLS: &[&str] = &["spawn_agent", "send_input"];

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
    pub mistralrs_repo_root: PathBuf,
    pub mistralrs_binary: PathBuf,
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
pub struct MistralRsRuntimeConfig {
    pub family: LocalModelFamily,
    pub model: String,
    pub port: u16,
    pub proxy_port: Option<u16>,
    pub max_seq_len: Option<u32>,
    pub max_seqs: u32,
    pub max_batch_size: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodexExecInvocation {
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanRoomBaselinePlan {
    pub dependencies: VendoredDependencyPaths,
    pub runtime: MistralRsRuntimeConfig,
    pub mistralrs_command: Vec<String>,
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
    let mistralrs_repo_root = references_root.join("mistral.rs");
    VendoredDependencyPaths {
        references_root,
        codex_exec_binary: codex_rs_root.join("target/release/codex-exec"),
        codex_repo_root,
        codex_rs_root,
        mistralrs_binary: mistralrs_repo_root.join("target/release/mistralrs"),
        mistralrs_repo_root,
    }
}

pub fn default_runtime_config(family: LocalModelFamily) -> MistralRsRuntimeConfig {
    match family {
        LocalModelFamily::GptOss => MistralRsRuntimeConfig {
            family,
            model: "openai/gpt-oss-20b".to_string(),
            port: 1234,
            proxy_port: Some(12434),
            max_seq_len: Some(131_072),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::Qwen35Vision => MistralRsRuntimeConfig {
            family,
            model: "Qwen/Qwen3.5-35B-A3B".to_string(),
            port: 1235,
            proxy_port: None,
            max_seq_len: None,
            max_seqs: 1,
            max_batch_size: 1,
        },
    }
}

pub fn build_mistralrs_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &MistralRsRuntimeConfig,
) -> Vec<String> {
    let mut command = vec![dependencies.mistralrs_binary.display().to_string()];
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
                "off".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
                "-a".to_string(),
                "gpt_oss".to_string(),
            ]);
            if let Some(max_seq_len) = runtime.max_seq_len {
                command.extend(["--max-seq-len".to_string(), max_seq_len.to_string()]);
            }
        }
        LocalModelFamily::Qwen35Vision => {
            command.extend([
                "serve".to_string(),
                "vision".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
        }
    }
    command
}

pub fn build_codex_exec_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &MistralRsRuntimeConfig,
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
    let mistralrs_command = build_mistralrs_command(&dependencies, &runtime);
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
        mistralrs_command,
        codex_exec_command,
        bridge_mode,
    }
}

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
        "mistral.rs" => DEFAULT_MISTRALRS_REPO_REF,
        _ => DEFAULT_CODEX_REPO_REF,
    }
}

fn ensure_git_checkout(repo_url: &str, git_ref: &str, target_dir: &Path) -> anyhow::Result<String> {
    if target_dir.join(".git").exists() {
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

    run_git([
        "clone",
        "--depth",
        "1",
        "--branch",
        git_ref,
        repo_url,
        path_arg(target_dir),
    ])?;
    Ok("cloned".to_string())
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

pub fn rewrite_mistralrs_responses_request(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
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

fn rewrite_tool(tool: Value) -> Option<Value> {
    let object = tool.as_object()?;
    let tool_type = object.get("type")?.as_str()?;
    match tool_type {
        "function" => {
            if let Some(function) = object.get("function").and_then(Value::as_object) {
                let name = function.get("name").and_then(Value::as_str)?;
                if DISALLOWED_MISTRALRS_FUNCTION_TOOLS.contains(&name) {
                    return None;
                }
                return Some(Value::Object(object.clone()));
            }

            let name = object.get("name").and_then(Value::as_str)?;
            if DISALLOWED_MISTRALRS_FUNCTION_TOOLS.contains(&name) {
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
    fn gpt_oss_runtime_uses_mistralrs_gpt_oss_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::GptOss);
        let command = build_mistralrs_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert!(command.iter().any(|part| part == "gpt_oss"));
        assert!(command.iter().any(|part| part == "--max-seq-len"));
    }

    #[test]
    fn qwen_runtime_uses_mistralrs_vision_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::Qwen35Vision);
        let command = build_mistralrs_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert_eq!(command[2], "vision");
        assert!(command.iter().any(|part| part == "Qwen/Qwen3.5-35B-A3B"));
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
        let rewritten = rewrite_mistralrs_responses_request(
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
        let rewritten = rewrite_mistralrs_responses_request(
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
}
