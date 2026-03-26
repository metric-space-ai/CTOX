use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::execution_baseline;

const MIN_POLICY_CONTEXT: u32 = 16_384;
const DEFAULT_MIN_COMPACTION_TOKENS: u32 = 12_288;
const DEFAULT_GPU0_DESKTOP_RESERVE_MB: u64 = 1024;
const CHAT_PLAN_RELATIVE_PATH: &str = "runtime/chat_runtime_plan.json";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatPreset {
    Quality,
    MaxContext,
    Performance,
}

impl ChatPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Quality => "Quality",
            Self::MaxContext => "Max Context",
            Self::Performance => "Performance",
        }
    }

    pub fn from_label(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "quality" => Self::Quality,
            "max context" | "max_context" | "context" => Self::MaxContext,
            "performance" | "perf" => Self::Performance,
            _ => Self::Quality,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareGpu {
    pub index: usize,
    pub name: String,
    pub total_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareProfile {
    pub gpus: Vec<HardwareGpu>,
    pub gpu0_desktop_reserve_mb: u64,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannedGpuAllocation {
    pub gpu_index: usize,
    pub name: String,
    pub total_mb: u64,
    pub desktop_reserve_mb: u64,
    pub aux_reserve_mb: u64,
    pub chat_budget_mb: u64,
    pub weight_mb: u64,
    pub kv_cache_mb: u64,
    pub free_headroom_mb: u64,
    pub chat_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatRuntimePlan {
    pub model: String,
    pub preset: ChatPreset,
    pub quantization: String,
    pub runtime_isq: Option<String>,
    pub max_seq_len: u32,
    pub compaction_threshold_percent: u8,
    pub compaction_min_tokens: u32,
    pub min_context_floor_applied: bool,
    pub paged_attn: String,
    pub pa_cache_type: Option<String>,
    pub pa_memory_fraction: Option<String>,
    pub disable_nccl: bool,
    pub tensor_parallel_backend: Option<String>,
    pub mn_local_world_size: Option<u32>,
    pub max_batch_size: u32,
    pub max_seqs: u32,
    pub cuda_visible_devices: String,
    pub device_layers: Option<String>,
    pub topology: Option<String>,
    pub allow_device_layers_with_topology: bool,
    pub nm_device_ordinal: Option<u32>,
    pub base_device_ordinal: Option<u32>,
    pub moe_experts_backend: Option<String>,
    pub disable_flash_attn: bool,
    pub force_no_mmap: bool,
    pub force_language_model_only: bool,
    pub isq_singlethread: bool,
    pub isq_cpu_threads: Option<u32>,
    pub expected_tok_s: f64,
    pub hardware_fingerprint: String,
    pub rationale: Vec<String>,
    pub gpu_allocations: Vec<PlannedGpuAllocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatPresetBundle {
    pub model: String,
    pub hardware: HardwareProfile,
    pub selected_preset: ChatPreset,
    pub selected_plan: ChatRuntimePlan,
    pub plans: Vec<ChatRuntimePlan>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompactionPolicy {
    pub threshold_percent: u8,
    pub min_tokens: u32,
}

pub fn compaction_policy_for_preset(preset: ChatPreset) -> CompactionPolicy {
    let threshold_percent = match preset {
        ChatPreset::Quality => 75,
        ChatPreset::MaxContext => 85,
        ChatPreset::Performance => 70,
    };
    CompactionPolicy {
        threshold_percent,
        min_tokens: DEFAULT_MIN_COMPACTION_TOKENS,
    }
}

#[derive(Debug, Clone, Copy)]
struct QuantOption {
    label: &'static str,
    runtime_isq: Option<&'static str>,
    weight_factor_milli: u32,
    speed_factor_milli: u32,
}

#[derive(Debug, Clone, Copy)]
struct ModelHarness {
    model: &'static str,
    base_weight_mb_q4: u64,
    kv_mb_per_1k_tokens_q4: u64,
    base_toks_per_sec_q4: f64,
    repeating_layers: u32,
    context_cap: u32,
    paged_attn: &'static str,
    pa_cache_type: Option<&'static str>,
    pa_memory_fraction: Option<&'static str>,
    force_no_mmap: bool,
    force_language_model_only: bool,
    disable_flash_attn: bool,
    isq_singlethread: bool,
    isq_cpu_threads: Option<u32>,
    fixed_device_layers_3gpu: Option<&'static str>,
    fixed_cuda_visible_devices_3gpu: Option<&'static str>,
    topology_rel_path: Option<&'static str>,
    allow_device_layers_with_topology: bool,
    nm_device_ordinal: Option<u32>,
    base_device_ordinal: Option<u32>,
    moe_experts_backend: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedHarnessRuntime {
    fixed_device_layers: Option<&'static str>,
    fixed_cuda_visible_devices: Option<&'static str>,
    topology_rel_path: Option<&'static str>,
    allow_device_layers_with_topology: bool,
    nm_device_ordinal: Option<u32>,
    base_device_ordinal: Option<u32>,
    moe_experts_backend: Option<&'static str>,
    force_no_mmap: bool,
}

const MXFP4_NATIVE: QuantOption = QuantOption {
    label: "native_mxfp4",
    runtime_isq: None,
    weight_factor_milli: 1000,
    speed_factor_milli: 1000,
};
const Q4K: QuantOption = QuantOption {
    label: "Q4K",
    runtime_isq: Some("Q4K"),
    weight_factor_milli: 1000,
    speed_factor_milli: 1000,
};
#[derive(Debug, Clone, Copy)]
struct PlanSpec {
    quant: QuantOption,
    backend: BackendMode,
    context_target: Option<u32>,
    max_batch_size: u32,
    max_seqs: u32,
}

pub fn chat_preset_choices() -> Vec<&'static str> {
    vec![
        ChatPreset::Quality.label(),
        ChatPreset::MaxContext.label(),
        ChatPreset::Performance.label(),
    ]
}

pub fn preview_chat_preset_bundle(
    _root: &Path,
    env_map: &BTreeMap<String, String>,
) -> Result<Option<ChatPresetBundle>> {
    if infer_chat_source(env_map).eq_ignore_ascii_case("api") {
        return Ok(None);
    }
    let model = env_map
        .get("CTOX_CHAT_MODEL")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("openai/gpt-oss-20b");
    let hardware = match inspect_hardware_profile() {
        Ok(profile) if !profile.gpus.is_empty() => profile,
        _ => return Ok(None),
    };
    let selected_preset = ChatPreset::from_label(
        env_map
            .get("CTOX_CHAT_LOCAL_PRESET")
            .map(String::as_str)
            .unwrap_or(ChatPreset::Quality.label()),
    );
    let bundle = build_bundle_for_model(model, selected_preset, &hardware, env_map)?;
    Ok(Some(bundle))
}

pub fn apply_chat_runtime_plan(
    root: &Path,
    env_map: &mut BTreeMap<String, String>,
) -> Result<Option<ChatRuntimePlan>> {
    clear_chat_plan_env(env_map);
    if infer_chat_source(env_map).eq_ignore_ascii_case("api") {
        persist_chat_runtime_plan(root, None)?;
        return Ok(None);
    }
    let Some(bundle) = preview_chat_preset_bundle(root, env_map)? else {
        persist_chat_runtime_plan(root, None)?;
        return Ok(None);
    };
    let plan = bundle.selected_plan.clone();
    let plan_json =
        serde_json::to_vec_pretty(&plan).context("failed to encode chat runtime plan")?;
    let digest = format!("{:x}", Sha256::digest(&plan_json));
    env_map.insert(
        "CTOX_CHAT_LOCAL_PRESET".to_string(),
        plan.preset.label().to_string(),
    );
    env_map.insert(
        "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT".to_string(),
        plan.compaction_threshold_percent.to_string(),
    );
    env_map.insert(
        "CTOX_CHAT_COMPACTION_MIN_TOKENS".to_string(),
        plan.compaction_min_tokens.to_string(),
    );
    env_map.insert("CTOX_VLLM_SERVE_MODEL".to_string(), plan.model.clone());
    env_map.insert(
        "CTOX_VLLM_SERVE_MAX_SEQ_LEN".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert(
        "CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert(
        "CTOX_CHAT_MODEL_REALIZED_CONTEXT".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert(
        "CTOX_VLLM_SERVE_REALIZED_MODEL".to_string(),
        plan.model.clone(),
    );
    if let Some(runtime_isq) = &plan.runtime_isq {
        env_map.insert("CTOX_VLLM_SERVE_ISQ".to_string(), runtime_isq.clone());
    }
    env_map.insert(
        "CTOX_VLLM_SERVE_PAGED_ATTN".to_string(),
        plan.paged_attn.clone(),
    );
    if let Some(cache_type) = &plan.pa_cache_type {
        env_map.insert(
            "CTOX_VLLM_SERVE_PA_CACHE_TYPE".to_string(),
            cache_type.clone(),
        );
    }
    if let Some(memory_fraction) = &plan.pa_memory_fraction {
        env_map.insert(
            "CTOX_VLLM_SERVE_PA_MEMORY_FRACTION".to_string(),
            memory_fraction.clone(),
        );
    }
    env_map.insert(
        "CTOX_VLLM_SERVE_DISABLE_NCCL".to_string(),
        if plan.disable_nccl { "1" } else { "0" }.to_string(),
    );
    if let Some(backend) = &plan.tensor_parallel_backend {
        env_map.insert(
            "CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND".to_string(),
            backend.clone(),
        );
    }
    if let Some(world_size) = plan.mn_local_world_size {
        env_map.insert(
            "CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE".to_string(),
            world_size.to_string(),
        );
    }
    env_map.insert(
        "CTOX_VLLM_SERVE_MAX_BATCH_SIZE".to_string(),
        plan.max_batch_size.to_string(),
    );
    env_map.insert(
        "CTOX_VLLM_SERVE_MAX_SEQS".to_string(),
        plan.max_seqs.to_string(),
    );
    env_map.insert(
        "CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES".to_string(),
        plan.cuda_visible_devices.clone(),
    );
    if let Some(device_layers) = &plan.device_layers {
        env_map.insert(
            "CTOX_VLLM_SERVE_DEVICE_LAYERS".to_string(),
            device_layers.clone(),
        );
    }
    if let Some(topology) = &plan.topology {
        let topology_path = if Path::new(topology).is_absolute() {
            topology.clone()
        } else {
            root.join(topology).display().to_string()
        };
        env_map.insert("CTOX_VLLM_SERVE_TOPOLOGY".to_string(), topology_path);
    }
    if plan.allow_device_layers_with_topology {
        env_map.insert(
            "CTOX_VLLM_SERVE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY".to_string(),
            "1".to_string(),
        );
    }
    if let Some(ordinal) = plan.nm_device_ordinal {
        env_map.insert(
            "CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL".to_string(),
            ordinal.to_string(),
        );
    }
    if let Some(ordinal) = plan.base_device_ordinal {
        env_map.insert(
            "CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL".to_string(),
            ordinal.to_string(),
        );
    }
    if let Some(backend) = &plan.moe_experts_backend {
        env_map.insert(
            "CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND".to_string(),
            backend.clone(),
        );
    }
    if plan.disable_flash_attn {
        env_map.insert(
            "CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN".to_string(),
            "1".to_string(),
        );
    }
    if plan.force_no_mmap {
        env_map.insert("CTOX_VLLM_SERVE_NO_MMAP".to_string(), "1".to_string());
    }
    if plan.force_language_model_only {
        env_map.insert(
            "CTOX_VLLM_SERVE_LANGUAGE_MODEL_ONLY".to_string(),
            "1".to_string(),
        );
    }
    if plan.isq_singlethread {
        env_map.insert(
            "CTOX_VLLM_SERVE_ISQ_SINGLETHREAD".to_string(),
            "1".to_string(),
        );
    }
    if let Some(cpu_threads) = plan.isq_cpu_threads {
        env_map.insert(
            "CTOX_VLLM_SERVE_ISQ_CPU_THREADS".to_string(),
            cpu_threads.to_string(),
        );
    }
    env_map.insert("CTOX_CHAT_RUNTIME_PLAN_DIGEST".to_string(), digest);
    env_map.insert("CTOX_CHAT_RUNTIME_PLAN_ACTIVE".to_string(), "1".to_string());
    env_map.insert(
        "CTOX_CHAT_RUNTIME_PLAN_PATH".to_string(),
        root.join(CHAT_PLAN_RELATIVE_PATH).display().to_string(),
    );
    persist_chat_runtime_plan(root, Some(&plan))?;
    Ok(Some(plan))
}

pub fn clear_chat_plan_env(env_map: &mut BTreeMap<String, String>) {
    for key in [
        "CTOX_CHAT_RUNTIME_PLAN_DIGEST",
        "CTOX_CHAT_RUNTIME_PLAN_ACTIVE",
        "CTOX_CHAT_RUNTIME_PLAN_PATH",
        "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
        "CTOX_CHAT_COMPACTION_MIN_TOKENS",
        "CTOX_VLLM_SERVE_ISQ",
        "CTOX_VLLM_SERVE_PAGED_ATTN",
        "CTOX_VLLM_SERVE_PA_CACHE_TYPE",
        "CTOX_VLLM_SERVE_PA_MEMORY_FRACTION",
        "CTOX_VLLM_SERVE_PA_CONTEXT_LEN",
        "CTOX_VLLM_SERVE_DISABLE_NCCL",
        "CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND",
        "CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE",
        "CTOX_VLLM_SERVE_MAX_BATCH_SIZE",
        "CTOX_VLLM_SERVE_MAX_SEQS",
        "CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN",
        "CTOX_VLLM_SERVE_NO_MMAP",
        "CTOX_VLLM_SERVE_LANGUAGE_MODEL_ONLY",
        "CTOX_VLLM_SERVE_ISQ_SINGLETHREAD",
        "CTOX_VLLM_SERVE_ISQ_CPU_THREADS",
        "CTOX_VLLM_SERVE_MAX_SEQ_LEN",
        "CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS",
        "CTOX_VLLM_SERVE_DEVICE_LAYERS",
        "CTOX_VLLM_SERVE_TOPOLOGY",
        "CTOX_VLLM_SERVE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY",
        "CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL",
        "CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL",
        "CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND",
        "CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES",
        "CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN",
        "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
        "CTOX_VLLM_SERVE_REALIZED_MODEL",
    ] {
        env_map.remove(key);
    }
}

pub fn clear_persisted_chat_plan(
    root: &Path,
    env_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    clear_chat_plan_env(env_map);
    persist_chat_runtime_plan(root, None)
}

fn persist_chat_runtime_plan(root: &Path, plan: Option<&ChatRuntimePlan>) -> Result<()> {
    let path = root.join(CHAT_PLAN_RELATIVE_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create planner dir {}", parent.display()))?;
    }
    match plan {
        Some(plan) => {
            let bytes = serde_json::to_vec_pretty(plan).context("failed to encode planner json")?;
            std::fs::write(&path, bytes)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        None => {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

fn build_bundle_for_model(
    model: &str,
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> Result<ChatPresetBundle> {
    match model.trim() {
        "openai/gpt-oss-20b" => Ok(build_gpt_oss_20b_bundle(selected_preset, hardware, env_map)),
        "Qwen/Qwen3.5-4B" => Ok(build_qwen35_4b_bundle(selected_preset, hardware, env_map)),
        "Qwen/Qwen3.5-9B" => Ok(build_qwen35_9b_bundle(selected_preset, hardware, env_map)),
        "Qwen/Qwen3.5-27B" => Ok(build_qwen35_27b_bundle(selected_preset, hardware, env_map)),
        "Qwen/Qwen3.5-35B-A3B" => Ok(build_qwen35_35b_a3b_bundle(
            selected_preset,
            hardware,
            env_map,
        )),
        "zai-org/GLM-4.7-Flash" => Ok(build_glm47_flash_bundle(selected_preset, hardware, env_map)),
        other => anyhow::bail!("unsupported runtime planner model: {other}"),
    }
}

fn bundle_from_plans(
    model: &str,
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    plans: Vec<ChatRuntimePlan>,
) -> ChatPresetBundle {
    let selected_plan = plans
        .iter()
        .find(|plan| plan.preset == selected_preset)
        .cloned()
        .unwrap_or_else(|| plans[0].clone());
    ChatPresetBundle {
        model: model.to_string(),
        hardware: hardware.clone(),
        selected_preset,
        selected_plan,
        plans,
    }
}

fn plan_from_specs(
    harness: ModelHarness,
    preset: ChatPreset,
    specs: &[PlanSpec],
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatRuntimePlan {
    let mut candidates = Vec::new();
    for spec in specs {
        if let Some(candidate) = build_candidate(harness, preset, *spec, hardware, env_map) {
            candidates.push(candidate);
        }
    }
    if let Some(best) = choose_best_candidate(preset, &candidates) {
        return best;
    }
    let fallback_spec = specs.first().copied().unwrap_or(PlanSpec {
        quant: Q4K,
        backend: BackendMode::DeviceLayers,
        context_target: Some(MIN_POLICY_CONTEXT),
        max_batch_size: 1,
        max_seqs: 1,
    });
    build_floor_fallback_plan(harness, preset, fallback_spec, hardware, env_map)
}

fn build_gpt_oss_20b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_gpt_oss_20b();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[PlanSpec {
            quant: MXFP4_NATIVE,
            backend: BackendMode::DeviceLayers,
            context_target: Some(131_072),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[PlanSpec {
            quant: MXFP4_NATIVE,
            backend: BackendMode::DeviceLayers,
            context_target: Some(131_072),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[PlanSpec {
            quant: MXFP4_NATIVE,
            backend: BackendMode::Nccl,
            context_target: Some(65_536),
            max_batch_size: 2,
            max_seqs: 2,
        }],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn build_qwen35_4b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_qwen35_4b();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(65_536),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(262_144),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(32_768),
            max_batch_size: 4,
            max_seqs: 4,
        }],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn build_qwen35_9b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_qwen35_9b();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(65_536),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(262_144),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(32_768),
            max_batch_size: 2,
            max_seqs: 2,
        }],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn build_qwen35_27b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_qwen35_27b();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(49_152),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(131_072),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(24_576),
            max_batch_size: 2,
            max_seqs: 2,
        }],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn build_qwen35_35b_a3b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_qwen35_35b_a3b();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(20_480),
                max_batch_size: 1,
                max_seqs: 1,
            },
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(16_384),
                max_batch_size: 1,
                max_seqs: 1,
            },
        ],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(24_576),
                max_batch_size: 1,
                max_seqs: 1,
            },
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(20_480),
                max_batch_size: 1,
                max_seqs: 1,
            },
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(16_384),
                max_batch_size: 1,
                max_seqs: 1,
            },
        ],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(16_384),
                max_batch_size: 2,
                max_seqs: 2,
            },
            PlanSpec {
                quant: Q4K,
                backend: BackendMode::DeviceLayers,
                context_target: Some(16_384),
                max_batch_size: 1,
                max_seqs: 1,
            },
        ],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn build_glm47_flash_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    let harness = plan_glm47_flash();
    let quality = plan_from_specs(
        harness,
        ChatPreset::Quality,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(32_768),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        harness,
        ChatPreset::MaxContext,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(65_536),
            max_batch_size: 1,
            max_seqs: 1,
        }],
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        harness,
        ChatPreset::Performance,
        &[PlanSpec {
            quant: Q4K,
            backend: BackendMode::DeviceLayers,
            context_target: Some(16_384),
            max_batch_size: 2,
            max_seqs: 2,
        }],
        hardware,
        env_map,
    );
    bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )
}

fn choose_best_candidate(
    preset: ChatPreset,
    candidates: &[ChatRuntimePlan],
) -> Option<ChatRuntimePlan> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|left, right| {
        let rank = |plan: &ChatRuntimePlan| -> (i64, i64, i64) {
            let quant_rank_high = match plan.quantization.as_str() {
                "native_mxfp4" => 4,
                "Q6K" => 3,
                "Q5K" => 2,
                _ => 1,
            };
            let quant_rank_low = match plan.quantization.as_str() {
                "native_mxfp4" => 4,
                "Q4K" => 3,
                "Q5K" => 2,
                _ => 1,
            };
            match preset {
                ChatPreset::Quality => (
                    quant_rank_high,
                    plan.max_seq_len as i64,
                    (plan.expected_tok_s * 100.0).round() as i64,
                ),
                ChatPreset::MaxContext => (
                    plan.max_seq_len as i64,
                    quant_rank_low,
                    (plan.expected_tok_s * 100.0).round() as i64,
                ),
                ChatPreset::Performance => (
                    (plan.expected_tok_s * 100.0).round() as i64,
                    plan.max_seq_len as i64,
                    quant_rank_low,
                ),
            }
        };
        rank(right).cmp(&rank(left))
    });
    sorted.into_iter().next()
}

#[derive(Debug, Clone, Copy)]
enum BackendMode {
    Nccl,
    DeviceLayers,
}

fn build_candidate(
    harness: ModelHarness,
    preset: ChatPreset,
    spec: PlanSpec,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> Option<ChatRuntimePlan> {
    let runtime = resolve_harness_runtime(harness, hardware);
    let compaction_policy = compaction_policy_for_preset(preset);
    let quant = spec.quant;
    let backend = spec.backend;
    let mut gpu_indices = select_chat_gpu_indices(preset, backend, &hardware.gpus);
    if matches!(backend, BackendMode::DeviceLayers) {
        if let Some(visible) = runtime.fixed_cuda_visible_devices {
            let fixed = visible
                .split(',')
                .filter_map(|chunk| chunk.trim().parse::<usize>().ok())
                .collect::<Vec<_>>();
            if !fixed.is_empty() {
                gpu_indices = fixed;
            }
        }
    }
    if gpu_indices.is_empty() {
        return None;
    }
    let fixed_device_layers_override = match backend {
        BackendMode::DeviceLayers => runtime.fixed_device_layers,
        _ => None,
    };

    let aux_reserves = compute_aux_reserves_mb(hardware, env_map, preset, &gpu_indices);
    let mut per_gpu_budgets = Vec::new();
    for gpu in &hardware.gpus {
        let desktop_reserve = if gpu.index == 0 {
            hardware.gpu0_desktop_reserve_mb
        } else {
            0
        };
        let aux_reserve = *aux_reserves.get(&gpu.index).unwrap_or(&0);
        let usable = gpu
            .total_mb
            .saturating_sub(desktop_reserve)
            .saturating_sub(aux_reserve);
        per_gpu_budgets.push((gpu.index, usable));
    }

    let selected_budgets = per_gpu_budgets
        .iter()
        .filter(|(index, _)| gpu_indices.contains(index))
        .map(|(_, usable)| *usable)
        .collect::<Vec<_>>();
    if selected_budgets.is_empty() {
        return None;
    }

    let weight_mb = scale_mb(harness.base_weight_mb_q4, quant.weight_factor_milli);
    let fixed_overhead_mb = fixed_overhead_mb(backend, gpu_indices.len());
    let effective_total_budget_mb = match backend {
        BackendMode::Nccl => {
            let min_budget = selected_budgets.iter().copied().min().unwrap_or(0);
            min_budget.saturating_mul(gpu_indices.len() as u64)
        }
        BackendMode::DeviceLayers => selected_budgets.iter().copied().sum(),
    };
    let kv_budget_cap_mb = effective_total_budget_mb
        .saturating_sub(weight_mb)
        .saturating_sub(fixed_overhead_mb);
    if kv_budget_cap_mb == 0 {
        return None;
    }

    let effective_concurrency = spec.max_seqs.max(spec.max_batch_size).max(1) as u64;
    let kv_mb_per_1k = scale_mb(harness.kv_mb_per_1k_tokens_q4, quant.weight_factor_milli)
        .saturating_mul(effective_concurrency);
    let raw_context =
        (((kv_budget_cap_mb as f64) / (kv_mb_per_1k.max(1) as f64)) * 1024.0).floor() as u32;
    let mut plan_context = align_context(raw_context.min(harness.context_cap));
    if let Some(target) = spec.context_target {
        plan_context = plan_context.min(align_context(target));
    }
    let policy_floor_ok = plan_context >= MIN_POLICY_CONTEXT;
    if !policy_floor_ok {
        return None;
    }
    let kv_budget_mb = (((plan_context as u64) * kv_mb_per_1k) + 1023) / 1024;
    if kv_budget_mb > kv_budget_cap_mb {
        return None;
    }

    let expected_tok_s = estimate_tok_s(
        harness,
        quant,
        backend,
        gpu_indices.len(),
        plan_context,
        spec.max_batch_size,
        spec.max_seqs,
        hardware,
        &gpu_indices,
    );
    let allocations = distribute_allocations(
        backend,
        hardware,
        &gpu_indices,
        &aux_reserves,
        harness.repeating_layers,
        fixed_device_layers_override,
        weight_mb,
        kv_budget_mb,
    );
    let overcommitted = allocations.iter().any(|allocation| {
        allocation.chat_enabled
            && allocation
                .desktop_reserve_mb
                .saturating_add(allocation.aux_reserve_mb)
                .saturating_add(allocation.chat_budget_mb)
                > allocation.total_mb
    });
    if overcommitted {
        return None;
    }
    let cuda_visible_devices = gpu_indices
        .iter()
        .map(|gpu| gpu.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let device_layers = match backend {
        BackendMode::Nccl => None,
        BackendMode::DeviceLayers if hardware.gpus.len() == 3 => fixed_device_layers_override
            .map(str::to_string)
            .or_else(|| Some(device_layers_cli(&allocations, harness.repeating_layers))),
        BackendMode::DeviceLayers => {
            Some(device_layers_cli(&allocations, harness.repeating_layers))
        }
    };
    let topology = runtime
        .topology_rel_path
        .map(|rel| Path::new(rel).display().to_string());
    let disable_nccl = !matches!(backend, BackendMode::Nccl);
    let tensor_parallel_backend = (!disable_nccl).then(|| "nccl".to_string());
    let mn_local_world_size = (!disable_nccl).then(|| gpu_indices.len() as u32);
    let mut rationale = vec![
        format!("preset {}", preset.label()),
        format!("quant {}", quant.label),
        format!("context {}", plan_context),
        format!("max_seqs {}", spec.max_seqs),
        format!("max_batch_size {}", spec.max_batch_size),
        match backend {
            BackendMode::Nccl => format!("backend nccl x{}", gpu_indices.len()),
            BackendMode::DeviceLayers => format!("backend device-layers x{}", gpu_indices.len()),
        },
    ];
    if matches!(backend, BackendMode::Nccl) && hardware.gpus.len() == 3 && gpu_indices.len() == 2 {
        rationale.push("3-GPU performance path keeps one GPU for auxiliary models".to_string());
    }
    if harness.disable_flash_attn {
        rationale.push("flash-attn disabled for this model/runtime path".to_string());
    }
    if runtime.force_no_mmap {
        rationale.push("mmap disabled for this model/runtime path".to_string());
    }
    if harness.force_language_model_only {
        rationale.push("vision tower disabled for text-only runtime path".to_string());
    }
    if harness.isq_singlethread && quant.runtime_isq.is_some() {
        rationale.push("ISQ is serialized to avoid model-load VRAM spikes".to_string());
    }
    if let Some(backend_override) = runtime.moe_experts_backend {
        rationale.push(format!("moe experts backend {}", backend_override));
    }
    Some(ChatRuntimePlan {
        model: harness.model.to_string(),
        preset,
        quantization: quant.label.to_string(),
        runtime_isq: quant.runtime_isq.map(str::to_string),
        max_seq_len: plan_context,
        compaction_threshold_percent: compaction_policy.threshold_percent,
        compaction_min_tokens: compaction_policy.min_tokens,
        min_context_floor_applied: true,
        paged_attn: harness.paged_attn.to_string(),
        pa_cache_type: harness.pa_cache_type.map(str::to_string),
        pa_memory_fraction: harness.pa_memory_fraction.map(str::to_string),
        disable_nccl,
        tensor_parallel_backend,
        mn_local_world_size,
        max_batch_size: spec.max_batch_size,
        max_seqs: spec.max_seqs,
        cuda_visible_devices,
        device_layers,
        topology,
        allow_device_layers_with_topology: runtime.allow_device_layers_with_topology,
        nm_device_ordinal: runtime.nm_device_ordinal,
        base_device_ordinal: runtime.base_device_ordinal,
        moe_experts_backend: runtime.moe_experts_backend.map(str::to_string),
        disable_flash_attn: harness.disable_flash_attn,
        force_no_mmap: runtime.force_no_mmap,
        force_language_model_only: harness.force_language_model_only,
        isq_singlethread: harness.isq_singlethread && quant.runtime_isq.is_some(),
        isq_cpu_threads: harness
            .isq_cpu_threads
            .filter(|_| quant.runtime_isq.is_some()),
        expected_tok_s,
        hardware_fingerprint: hardware.fingerprint.clone(),
        rationale,
        gpu_allocations: allocations,
    })
}

fn build_floor_fallback_plan(
    harness: ModelHarness,
    preset: ChatPreset,
    fallback_spec: PlanSpec,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatRuntimePlan {
    let runtime = resolve_harness_runtime(harness, hardware);
    let plan =
        build_candidate(harness, preset, fallback_spec, hardware, env_map).unwrap_or_else(|| {
            let compaction_policy = compaction_policy_for_preset(preset);
            ChatRuntimePlan {
                model: harness.model.to_string(),
                preset,
                quantization: fallback_spec.quant.label.to_string(),
                runtime_isq: fallback_spec.quant.runtime_isq.map(str::to_string),
                max_seq_len: MIN_POLICY_CONTEXT,
                compaction_threshold_percent: compaction_policy.threshold_percent,
                compaction_min_tokens: compaction_policy.min_tokens,
                min_context_floor_applied: false,
                paged_attn: harness.paged_attn.to_string(),
                pa_cache_type: harness.pa_cache_type.map(str::to_string),
                pa_memory_fraction: harness.pa_memory_fraction.map(str::to_string),
                disable_nccl: true,
                tensor_parallel_backend: None,
                mn_local_world_size: None,
                max_batch_size: fallback_spec.max_batch_size,
                max_seqs: fallback_spec.max_seqs,
                cuda_visible_devices: hardware
                    .gpus
                    .iter()
                    .map(|gpu| gpu.index.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                device_layers: None,
                topology: runtime
                    .topology_rel_path
                    .map(|rel| Path::new(rel).display().to_string()),
                allow_device_layers_with_topology: runtime.allow_device_layers_with_topology,
                nm_device_ordinal: runtime.nm_device_ordinal,
                base_device_ordinal: runtime.base_device_ordinal,
                moe_experts_backend: runtime.moe_experts_backend.map(str::to_string),
                disable_flash_attn: harness.disable_flash_attn,
                force_no_mmap: runtime.force_no_mmap,
                force_language_model_only: harness.force_language_model_only,
                isq_singlethread: harness.isq_singlethread
                    && fallback_spec.quant.runtime_isq.is_some(),
                isq_cpu_threads: harness
                    .isq_cpu_threads
                    .filter(|_| fallback_spec.quant.runtime_isq.is_some()),
                expected_tok_s: harness.base_toks_per_sec_q4 * 0.65,
                hardware_fingerprint: hardware.fingerprint.clone(),
                rationale: vec!["policy floor fallback".to_string()],
                gpu_allocations: hardware
                    .gpus
                    .iter()
                    .map(|gpu| PlannedGpuAllocation {
                        gpu_index: gpu.index,
                        name: gpu.name.clone(),
                        total_mb: gpu.total_mb,
                        desktop_reserve_mb: if gpu.index == 0 {
                            hardware.gpu0_desktop_reserve_mb
                        } else {
                            0
                        },
                        aux_reserve_mb: 0,
                        chat_budget_mb: 0,
                        weight_mb: 0,
                        kv_cache_mb: 0,
                        free_headroom_mb: gpu.total_mb,
                        chat_enabled: true,
                    })
                    .collect(),
            }
        });
    let mut fallback = plan;
    fallback.rationale.push(
        "hardware could not satisfy the full preset policy; kept Q4 and the 16k floor".to_string(),
    );
    fallback
}

fn resolve_harness_runtime(
    harness: ModelHarness,
    hardware: &HardwareProfile,
) -> ResolvedHarnessRuntime {
    let mut runtime = ResolvedHarnessRuntime {
        fixed_device_layers: (hardware.gpus.len() == 3).then_some(harness.fixed_device_layers_3gpu).flatten(),
        fixed_cuda_visible_devices: (hardware.gpus.len() == 3)
            .then_some(harness.fixed_cuda_visible_devices_3gpu)
            .flatten(),
        topology_rel_path: harness.topology_rel_path,
        allow_device_layers_with_topology: harness.allow_device_layers_with_topology,
        nm_device_ordinal: harness.nm_device_ordinal,
        base_device_ordinal: harness.base_device_ordinal,
        moe_experts_backend: harness.moe_experts_backend,
        force_no_mmap: harness.force_no_mmap,
    };

    // The 35B Qwen legacy path is a real 3-GPU-specific harness. On larger hosts,
    // drop the topology and mmap overrides and let the planner derive a fresh map.
    if harness.model == "Qwen/Qwen3.5-35B-A3B" && hardware.gpus.len() >= 4 {
        runtime.fixed_device_layers = None;
        runtime.fixed_cuda_visible_devices = None;
        runtime.topology_rel_path = None;
        runtime.allow_device_layers_with_topology = false;
        runtime.nm_device_ordinal = None;
        runtime.base_device_ordinal = None;
        runtime.moe_experts_backend = None;
        runtime.force_no_mmap = false;
    }

    runtime
}

fn distribute_allocations(
    backend: BackendMode,
    hardware: &HardwareProfile,
    gpu_indices: &[usize],
    aux_reserves: &BTreeMap<usize, u64>,
    total_layers: u32,
    fixed_device_layers: Option<&str>,
    weight_mb: u64,
    kv_budget_mb: u64,
) -> Vec<PlannedGpuAllocation> {
    let selected = hardware
        .gpus
        .iter()
        .filter(|gpu| gpu_indices.contains(&gpu.index))
        .collect::<Vec<_>>();
    let fixed_layer_weights =
        fixed_device_layer_weights(fixed_device_layers, gpu_indices, total_layers);
    let weight_shares = match backend {
        BackendMode::Nccl => even_shares(weight_mb, selected.len()),
        BackendMode::DeviceLayers => match fixed_layer_weights.as_ref() {
            Some(weights) => proportional_shares(weight_mb, weights),
            None => proportional_shares(
                weight_mb,
                &selected
                    .iter()
                    .map(|gpu| {
                        gpu.total_mb
                            .saturating_sub(if gpu.index == 0 {
                                hardware.gpu0_desktop_reserve_mb
                            } else {
                                0
                            })
                            .saturating_sub(*aux_reserves.get(&gpu.index).unwrap_or(&0))
                    })
                    .collect::<Vec<_>>(),
            ),
        },
    };
    let kv_shares = match backend {
        BackendMode::Nccl => even_shares(kv_budget_mb, selected.len()),
        BackendMode::DeviceLayers => match fixed_layer_weights.as_ref() {
            Some(weights) => proportional_shares(kv_budget_mb, weights),
            None => proportional_shares(
                kv_budget_mb,
                &selected
                    .iter()
                    .map(|gpu| {
                        gpu.total_mb
                            .saturating_sub(if gpu.index == 0 {
                                hardware.gpu0_desktop_reserve_mb
                            } else {
                                0
                            })
                            .saturating_sub(*aux_reserves.get(&gpu.index).unwrap_or(&0))
                    })
                    .collect::<Vec<_>>(),
            ),
        },
    };

    let mut selected_index = 0usize;
    hardware
        .gpus
        .iter()
        .map(|gpu| {
            let desktop_reserve = if gpu.index == 0 {
                hardware.gpu0_desktop_reserve_mb
            } else {
                0
            };
            let aux_reserve = *aux_reserves.get(&gpu.index).unwrap_or(&0);
            let chat_enabled = gpu_indices.contains(&gpu.index);
            let (weight_share, kv_share) = if chat_enabled {
                let share = (
                    *weight_shares.get(selected_index).unwrap_or(&0),
                    *kv_shares.get(selected_index).unwrap_or(&0),
                );
                selected_index += 1;
                share
            } else {
                (0, 0)
            };
            let chat_budget_mb = weight_share.saturating_add(kv_share);
            let consumed = desktop_reserve
                .saturating_add(aux_reserve)
                .saturating_add(chat_budget_mb);
            PlannedGpuAllocation {
                gpu_index: gpu.index,
                name: gpu.name.clone(),
                total_mb: gpu.total_mb,
                desktop_reserve_mb: desktop_reserve,
                aux_reserve_mb: aux_reserve,
                chat_budget_mb,
                weight_mb: weight_share,
                kv_cache_mb: kv_share,
                free_headroom_mb: gpu.total_mb.saturating_sub(consumed),
                chat_enabled,
            }
        })
        .collect::<Vec<_>>()
}

fn fixed_device_layer_weights(
    fixed_device_layers: Option<&str>,
    gpu_indices: &[usize],
    total_layers: u32,
) -> Option<Vec<u64>> {
    let map = fixed_device_layers?;
    let mut counts = BTreeMap::new();
    for chunk in map.split(';') {
        let (gpu, layers) = chunk.split_once(':')?;
        let gpu_index = gpu.trim().parse::<usize>().ok()?;
        let layer_count = layers.trim().parse::<u64>().ok()?;
        counts.insert(gpu_index, layer_count);
    }
    let selected = gpu_indices
        .iter()
        .map(|gpu_index| counts.get(gpu_index).copied())
        .collect::<Option<Vec<_>>>()?;
    let sum = selected.iter().copied().sum::<u64>();
    if sum != total_layers as u64 {
        return None;
    }
    Some(selected)
}

fn device_layers_cli(allocations: &[PlannedGpuAllocation], total_layers: u32) -> String {
    let selected = allocations
        .iter()
        .filter(|allocation| allocation.chat_enabled && allocation.chat_budget_mb > 0)
        .collect::<Vec<_>>();
    let total_budget = selected
        .iter()
        .map(|allocation| allocation.chat_budget_mb)
        .sum::<u64>()
        .max(1);
    let total_layers = total_layers as u64;
    let mut raw = selected
        .iter()
        .map(|allocation| {
            (
                allocation.gpu_index,
                ((allocation.chat_budget_mb as f64 / total_budget as f64) * total_layers as f64)
                    .round() as u64,
            )
        })
        .collect::<Vec<_>>();
    let sum_layers = raw.iter().map(|(_, layers)| *layers).sum::<u64>();
    if sum_layers != total_layers {
        let delta = total_layers as i64 - sum_layers as i64;
        if let Some(first) = raw.first_mut() {
            first.1 = ((first.1 as i64) + delta).max(1) as u64;
        }
    }
    raw.into_iter()
        .map(|(gpu, layers)| format!("{gpu}:{layers}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn compute_aux_reserves_mb(
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
    preset: ChatPreset,
    chat_gpu_indices: &[usize],
) -> BTreeMap<usize, u64> {
    let aux_models = [
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Embedding,
            env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Stt,
            env_map.get("CTOX_STT_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Tts,
            env_map.get("CTOX_TTS_MODEL").map(String::as_str),
        ),
    ];

    let explicit_devices = parse_csv_indices(env_map.get("CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES"));
    let default_distribution = default_aux_distribution(hardware, preset, chat_gpu_indices);
    let target_devices = if explicit_devices.is_empty() {
        default_distribution
    } else {
        explicit_devices
    };

    let mut reserves = BTreeMap::new();
    if target_devices.is_empty() {
        return reserves;
    }
    for selection in aux_models {
        let reserve_mb = selection.gpu_reserve_mb();
        if reserve_mb == 0 {
            continue;
        }
        let shares = even_shares(reserve_mb, target_devices.len());
        for (idx, gpu_index) in target_devices.iter().enumerate() {
            let entry = reserves.entry(*gpu_index).or_insert(0);
            *entry = entry.saturating_add(*shares.get(idx).unwrap_or(&0));
        }
    }
    reserves
}

fn default_aux_distribution(
    hardware: &HardwareProfile,
    preset: ChatPreset,
    chat_gpu_indices: &[usize],
) -> Vec<usize> {
    let total_gpu_count = hardware.gpus.len();
    if total_gpu_count == 0 {
        return Vec::new();
    }
    if matches!(preset, ChatPreset::Performance) && total_gpu_count == 3 {
        let aux_only = hardware
            .gpus
            .iter()
            .map(|gpu| gpu.index)
            .filter(|index| !chat_gpu_indices.contains(index))
            .collect::<Vec<_>>();
        if !aux_only.is_empty() {
            return aux_only;
        }
    }
    if total_gpu_count >= 4 {
        return hardware.gpus.iter().map(|gpu| gpu.index).collect();
    }
    vec![0]
}

fn estimate_tok_s(
    harness: ModelHarness,
    quant: QuantOption,
    backend: BackendMode,
    gpu_count: usize,
    context: u32,
    max_batch_size: u32,
    max_seqs: u32,
    hardware: &HardwareProfile,
    gpu_indices: &[usize],
) -> f64 {
    let mut tps = harness.base_toks_per_sec_q4 * (quant.speed_factor_milli as f64 / 1000.0);
    match backend {
        BackendMode::Nccl => {
            tps *= 1.0 + ((gpu_count.saturating_sub(1) as f64) * 0.22);
        }
        BackendMode::DeviceLayers => {
            tps *= 1.0 + ((gpu_count.saturating_sub(1) as f64) * 0.11);
        }
    }
    let mixed_penalty = if gpu_indices
        .iter()
        .filter_map(|index| hardware.gpus.iter().find(|gpu| gpu.index == *index))
        .map(|gpu| gpu.total_mb)
        .collect::<Vec<_>>()
        .windows(2)
        .any(|window| window[0] != window[1])
    {
        0.92
    } else {
        1.0
    };
    let context_penalty = if context > 65_536 {
        0.90
    } else if context > 32_768 {
        0.95
    } else {
        1.0
    };
    let concurrency_bonus = (1.0 + (max_seqs.saturating_sub(1) as f64 * 0.08))
        * (1.0 + (max_batch_size.saturating_sub(1) as f64 * 0.05));
    tps * mixed_penalty * context_penalty * concurrency_bonus.min(1.35)
}

fn select_chat_gpu_indices(
    preset: ChatPreset,
    backend: BackendMode,
    gpus: &[HardwareGpu],
) -> Vec<usize> {
    if gpus.is_empty() {
        return Vec::new();
    }
    match backend {
        BackendMode::Nccl => {
            let target = largest_power_of_two(gpus.len());
            if target < 2 {
                return vec![gpus[0].index];
            }
            let mut ordered = gpus.to_vec();
            ordered.sort_by(|left, right| {
                right
                    .total_mb
                    .cmp(&left.total_mb)
                    .then_with(|| left.index.cmp(&right.index))
            });
            let mut selected = ordered
                .into_iter()
                .filter(|gpu| gpu.index != 0 || gpus.len() <= target)
                .take(target)
                .map(|gpu| gpu.index)
                .collect::<Vec<_>>();
            selected.sort_unstable();
            selected
        }
        BackendMode::DeviceLayers => {
            let mut selected = gpus.iter().map(|gpu| gpu.index).collect::<Vec<_>>();
            if matches!(preset, ChatPreset::Performance) && selected.len() > 1 {
                selected.sort_unstable();
            }
            selected
        }
    }
}

fn largest_power_of_two(value: usize) -> usize {
    let mut power = 1usize;
    while power.saturating_mul(2) <= value {
        power *= 2;
    }
    power
}

fn scale_mb(base_mb: u64, factor_milli: u32) -> u64 {
    ((base_mb as u128 * factor_milli as u128) / 1000u128) as u64
}

fn fixed_overhead_mb(backend: BackendMode, gpu_count: usize) -> u64 {
    match backend {
        BackendMode::Nccl => 1400u64.saturating_mul(gpu_count as u64),
        BackendMode::DeviceLayers => 900u64.saturating_add((gpu_count as u64) * 180),
    }
}

fn align_context(context: u32) -> u32 {
    if context < 1024 {
        return context;
    }
    (context / 1024) * 1024
}

fn proportional_shares(total: u64, weights: &[u64]) -> Vec<u64> {
    if weights.is_empty() {
        return Vec::new();
    }
    let weight_sum = weights.iter().copied().sum::<u64>().max(1);
    let mut shares = weights
        .iter()
        .map(|weight| total.saturating_mul(*weight) / weight_sum)
        .collect::<Vec<_>>();
    let mut distributed = shares.iter().copied().sum::<u64>();
    let mut idx = 0usize;
    while distributed < total {
        let target = idx % shares.len();
        shares[target] = shares[target].saturating_add(1);
        distributed += 1;
        idx += 1;
    }
    shares
}

fn even_shares(total: u64, count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let base = total / count as u64;
    let mut shares = vec![base; count];
    let mut remaining = total - (base * count as u64);
    let mut idx = 0usize;
    while remaining > 0 {
        shares[idx % count] += 1;
        remaining -= 1;
        idx += 1;
    }
    shares
}

fn infer_chat_source(env_map: &BTreeMap<String, String>) -> String {
    env_map
        .get("CTOX_CHAT_SOURCE")
        .cloned()
        .or_else(|| {
            env_map
                .get("CTOX_CHAT_MODEL")
                .filter(|value| execution_baseline::is_openai_api_chat_model(value))
                .map(|_| "api".to_string())
        })
        .unwrap_or_else(|| "local".to_string())
}

fn parse_csv_indices(raw: Option<&String>) -> Vec<usize> {
    raw.map(|value| {
        value
            .split(',')
            .filter_map(|chunk| chunk.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    })
    .unwrap_or_default()
}

fn inspect_hardware_profile() -> Result<HardwareProfile> {
    if let Ok(spec) = std::env::var("CTOX_TEST_GPU_TOTALS_MB") {
        let gpus = spec
            .split(';')
            .filter_map(|chunk| {
                let (index, total) = chunk.split_once(':')?;
                Some(HardwareGpu {
                    index: index.trim().parse().ok()?,
                    name: format!("Test GPU {}", index.trim()),
                    total_mb: total.trim().parse().ok()?,
                })
            })
            .collect::<Vec<_>>();
        if !gpus.is_empty() {
            return Ok(HardwareProfile {
                fingerprint: hardware_fingerprint(&gpus, DEFAULT_GPU0_DESKTOP_RESERVE_MB),
                gpus,
                gpu0_desktop_reserve_mb: gpu0_desktop_reserve_mb(),
            });
        }
    }

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to run nvidia-smi for hardware planner")?;
    if !output.status.success() {
        anyhow::bail!("nvidia-smi hardware planner query failed");
    }

    let mut gpus = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
        if parts.len() < 3 {
            continue;
        }
        let Ok(index) = parts[0].parse::<usize>() else {
            continue;
        };
        let Ok(total_mb) = parts[2].parse::<u64>() else {
            continue;
        };
        gpus.push(HardwareGpu {
            index,
            name: parts[1].to_string(),
            total_mb,
        });
    }
    gpus.sort_by_key(|gpu| gpu.index);
    let gpu0_reserve = gpu0_desktop_reserve_mb();
    Ok(HardwareProfile {
        fingerprint: hardware_fingerprint(&gpus, gpu0_reserve),
        gpus,
        gpu0_desktop_reserve_mb: gpu0_reserve,
    })
}

fn gpu0_desktop_reserve_mb() -> u64 {
    if std::env::var("CTOX_HEADLESS")
        .ok()
        .map(|value| value == "1")
        .unwrap_or(false)
    {
        return 0;
    }
    std::env::var("CTOX_GPU0_DESKTOP_RESERVE_MB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_GPU0_DESKTOP_RESERVE_MB)
}

fn hardware_fingerprint(gpus: &[HardwareGpu], gpu0_desktop_reserve_mb: u64) -> String {
    let raw = format!(
        "gpu0_reserve={gpu0_desktop_reserve_mb};{}",
        gpus.iter()
            .map(|gpu| format!("{}:{}:{}", gpu.index, gpu.name, gpu.total_mb))
            .collect::<Vec<_>>()
            .join("|")
    );
    format!("{:x}", Sha256::digest(raw.as_bytes()))
}

fn plan_gpt_oss_20b() -> ModelHarness {
    ModelHarness {
        model: "openai/gpt-oss-20b",
        base_weight_mb_q4: 15_800,
        kv_mb_per_1k_tokens_q4: 220,
        base_toks_per_sec_q4: 90.0,
        repeating_layers: 24,
        context_cap: 131_072,
        paged_attn: "auto",
        pa_cache_type: Some("f8e4m3"),
        pa_memory_fraction: Some("0.80"),
        force_no_mmap: true,
        force_language_model_only: false,
        disable_flash_attn: false,
        isq_singlethread: false,
        isq_cpu_threads: None,
        fixed_device_layers_3gpu: None,
        fixed_cuda_visible_devices_3gpu: None,
        topology_rel_path: Some("scripts/qwen35b_3gpu_striped.topology.yaml"),
        allow_device_layers_with_topology: true,
        nm_device_ordinal: Some(2),
        base_device_ordinal: Some(2),
        moe_experts_backend: Some("fast"),
    }
}

fn plan_qwen35_4b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-4B",
        base_weight_mb_q4: 3_600,
        kv_mb_per_1k_tokens_q4: 78,
        base_toks_per_sec_q4: 140.0,
        repeating_layers: 32,
        context_cap: 262_144,
        paged_attn: "auto",
        pa_cache_type: Some("f8e4m3"),
        pa_memory_fraction: Some("0.80"),
        force_no_mmap: true,
        force_language_model_only: false,
        disable_flash_attn: true,
        isq_singlethread: false,
        isq_cpu_threads: None,
        fixed_device_layers_3gpu: None,
        fixed_cuda_visible_devices_3gpu: None,
        topology_rel_path: Some("scripts/qwen35b_3gpu_striped.topology.yaml"),
        allow_device_layers_with_topology: true,
        nm_device_ordinal: Some(2),
        base_device_ordinal: Some(2),
        moe_experts_backend: Some("fast"),
    }
}

fn plan_qwen35_9b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-9B",
        base_weight_mb_q4: 6_700,
        kv_mb_per_1k_tokens_q4: 112,
        base_toks_per_sec_q4: 95.0,
        repeating_layers: 32,
        context_cap: 262_144,
        paged_attn: "auto",
        pa_cache_type: Some("f8e4m3"),
        pa_memory_fraction: Some("0.80"),
        force_no_mmap: true,
        force_language_model_only: false,
        disable_flash_attn: true,
        isq_singlethread: false,
        isq_cpu_threads: None,
        fixed_device_layers_3gpu: None,
        fixed_cuda_visible_devices_3gpu: None,
        topology_rel_path: Some("scripts/qwen35b_3gpu_striped.topology.yaml"),
        allow_device_layers_with_topology: true,
        nm_device_ordinal: Some(2),
        base_device_ordinal: Some(2),
        moe_experts_backend: Some("fast"),
    }
}

fn plan_qwen35_27b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-27B",
        base_weight_mb_q4: 18_300,
        kv_mb_per_1k_tokens_q4: 248,
        base_toks_per_sec_q4: 45.0,
        repeating_layers: 64,
        context_cap: 262_144,
        paged_attn: "auto",
        pa_cache_type: Some("f8e4m3"),
        pa_memory_fraction: Some("0.80"),
        force_no_mmap: false,
        force_language_model_only: false,
        disable_flash_attn: true,
        isq_singlethread: false,
        isq_cpu_threads: None,
        fixed_device_layers_3gpu: None,
        fixed_cuda_visible_devices_3gpu: None,
        topology_rel_path: None,
        allow_device_layers_with_topology: false,
        nm_device_ordinal: None,
        base_device_ordinal: None,
        moe_experts_backend: None,
    }
}

fn plan_qwen35_35b_a3b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-35B-A3B",
        base_weight_mb_q4: 21_500,
        kv_mb_per_1k_tokens_q4: 270,
        base_toks_per_sec_q4: 38.0,
        repeating_layers: 40,
        context_cap: 262_144,
        paged_attn: "off",
        pa_cache_type: None,
        pa_memory_fraction: None,
        force_no_mmap: true,
        force_language_model_only: true,
        disable_flash_attn: true,
        isq_singlethread: true,
        isq_cpu_threads: None,
        fixed_device_layers_3gpu: Some("0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:3;2:5"),
        fixed_cuda_visible_devices_3gpu: Some("0,1,2"),
        topology_rel_path: Some("scripts/qwen35b_3gpu_striped.topology.yaml"),
        allow_device_layers_with_topology: true,
        nm_device_ordinal: Some(2),
        base_device_ordinal: Some(2),
        moe_experts_backend: Some("fast"),
    }
}

fn plan_glm47_flash() -> ModelHarness {
    ModelHarness {
        model: "zai-org/GLM-4.7-Flash",
        base_weight_mb_q4: 22_000,
        kv_mb_per_1k_tokens_q4: 275,
        base_toks_per_sec_q4: 48.0,
        repeating_layers: 47,
        context_cap: 65_536,
        paged_attn: "off",
        pa_cache_type: None,
        pa_memory_fraction: None,
        force_no_mmap: true,
        force_language_model_only: false,
        disable_flash_attn: true,
        isq_singlethread: false,
        isq_cpu_threads: Some(4),
        fixed_device_layers_3gpu: None,
        fixed_cuda_visible_devices_3gpu: Some("0,2,1"),
        topology_rel_path: None,
        allow_device_layers_with_topology: false,
        nm_device_ordinal: None,
        base_device_ordinal: None,
        moe_experts_backend: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(count: usize, total_mb: u64) -> HardwareProfile {
        let gpus = (0..count)
            .map(|index| HardwareGpu {
                index,
                name: format!("GPU {index}"),
                total_mb,
            })
            .collect::<Vec<_>>();
        HardwareProfile {
            fingerprint: "test".to_string(),
            gpus,
            gpu0_desktop_reserve_mb: 1024,
        }
    }

    fn sum_device_layers(spec: &str) -> u32 {
        spec.split(';')
            .filter_map(|entry| entry.split_once(':'))
            .map(|(_, layers)| layers.parse::<u32>().unwrap())
            .sum()
    }

    #[test]
    fn performance_on_three_gpus_uses_power_of_two_subset() {
        let env_map = BTreeMap::new();
        let plan =
            build_gpt_oss_20b_bundle(ChatPreset::Performance, &hardware(3, 24_576), &env_map)
                .selected_plan;
        assert_eq!(plan.cuda_visible_devices, "1,2");
        assert_eq!(plan.tensor_parallel_backend.as_deref(), Some("nccl"));
        assert_eq!(plan.mn_local_world_size, Some(2));
    }

    #[test]
    fn max_context_keeps_policy_floor() {
        let env_map = BTreeMap::new();
        let plan = build_qwen35_4b_bundle(ChatPreset::MaxContext, &hardware(1, 24_576), &env_map)
            .selected_plan;
        assert!(plan.max_seq_len >= MIN_POLICY_CONTEXT);
        assert_eq!(plan.quantization, "Q4K");
    }

    #[test]
    fn gpt_oss_uses_native_quantization_without_isq() {
        let env_map = BTreeMap::new();
        let plan = build_gpt_oss_20b_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.quantization, "native_mxfp4");
        assert_eq!(plan.runtime_isq, None);
        assert_eq!(
            sum_device_layers(plan.device_layers.as_deref().unwrap()),
            24
        );
    }

    #[test]
    fn qwen35_moe_uses_aux_aware_runtime_constraints() {
        let env_map = BTreeMap::new();
        let plan = build_qwen35_35b_a3b_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.quantization, "Q4K");
        assert_eq!(plan.paged_attn, "off");
        let device_layers = plan.device_layers.as_deref().unwrap();
        assert_eq!(sum_device_layers(device_layers), 40);
        assert_eq!(
            device_layers,
            "0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:3;2:5"
        );
        assert!(plan.force_no_mmap);
        assert!(plan.force_language_model_only);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
    }

    #[test]
    fn qwen35_moe_drops_three_gpu_legacy_overrides_on_four_gpu_hosts() {
        let env_map = BTreeMap::new();
        let plan = build_qwen35_35b_a3b_bundle(ChatPreset::Quality, &hardware(4, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.quantization, "Q4K");
        assert_eq!(plan.paged_attn, "off");
        assert_eq!(plan.cuda_visible_devices, "0,1,2,3");
        assert_eq!(sum_device_layers(plan.device_layers.as_deref().unwrap()), 40);
        assert!(!plan.force_no_mmap);
        assert_eq!(plan.topology, None);
        assert!(!plan.allow_device_layers_with_topology);
        assert_eq!(plan.nm_device_ordinal, None);
        assert_eq!(plan.base_device_ordinal, None);
        assert_eq!(plan.moe_experts_backend, None);
        assert!(plan.force_language_model_only);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
    }

    #[test]
    fn glm_keeps_public_runtime_constraints() {
        let env_map = BTreeMap::new();
        let plan = build_glm47_flash_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.quantization, "Q4K");
        assert_eq!(plan.paged_attn, "off");
        assert_eq!(plan.cuda_visible_devices, "0,2,1");
        assert!(plan.force_no_mmap);
        assert!(plan.disable_flash_attn);
        assert_eq!(plan.isq_cpu_threads, Some(4));
    }

    #[test]
    fn qwen4b_presets_are_not_identical() {
        let env_map = BTreeMap::new();
        let bundle = build_qwen35_4b_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map);
        let quality = bundle
            .plans
            .iter()
            .find(|plan| plan.preset == ChatPreset::Quality)
            .unwrap();
        let max_context = bundle
            .plans
            .iter()
            .find(|plan| plan.preset == ChatPreset::MaxContext)
            .unwrap();
        let performance = bundle
            .plans
            .iter()
            .find(|plan| plan.preset == ChatPreset::Performance)
            .unwrap();
        assert!(max_context.max_seq_len >= quality.max_seq_len);
        assert!(performance.max_seq_len <= quality.max_seq_len);
        assert!(performance.max_seqs > quality.max_seqs);
        assert_eq!(
            sum_device_layers(quality.device_layers.as_deref().unwrap()),
            32
        );
    }
}
