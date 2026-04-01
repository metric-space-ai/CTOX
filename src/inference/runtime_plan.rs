use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::inference::engine;
use crate::inference::model_manifest;
use crate::inference::resource_state;
use crate::inference::runtime_env;

const MIN_POLICY_CONTEXT: u32 = 16_384;
const QUALITY_MIN_COMPACTION_TOKENS: u32 = 12_288;
const MAX_CONTEXT_MIN_COMPACTION_TOKENS: u32 = 16_384;
const PERFORMANCE_MIN_COMPACTION_TOKENS: u32 = 8_192;
const DEFAULT_GPU0_DESKTOP_RESERVE_MB: u64 = 1024;
const CHAT_PLAN_RELATIVE_PATH: &str = "runtime/chat_plan.json";

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
    pub repeating_weight_mb: u64,
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

impl ChatRuntimePlan {
    pub fn effective_cache_label(&self) -> &str {
        self.pa_cache_type.as_deref().unwrap_or_else(|| {
            if self.paged_attn.eq_ignore_ascii_case("off") {
                "off"
            } else {
                "auto"
            }
        })
    }
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
    match preset {
        ChatPreset::Quality => CompactionPolicy {
            threshold_percent: 75,
            min_tokens: QUALITY_MIN_COMPACTION_TOKENS,
        },
        ChatPreset::MaxContext => CompactionPolicy {
            threshold_percent: 85,
            min_tokens: MAX_CONTEXT_MIN_COMPACTION_TOKENS,
        },
        ChatPreset::Performance => CompactionPolicy {
            threshold_percent: 70,
            min_tokens: PERFORMANCE_MIN_COMPACTION_TOKENS,
        },
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
struct EmpiricalSizingProfile {
    // Tensors outside the repeating transformer stack: embeddings, lm_head, norms,
    // router/shared blocks and other per-model fixed tensors. This is measured per model.
    non_repeating_weight_mb_q4: u64,
    // Average footprint per repeating layer at Q4K after the model-specific load path settles.
    repeating_layer_weight_mb_q4: u64,
    // Extra headroom reserved for model-specific load/ISQ spikes observed empirically.
    load_peak_slack_mb_q4: u64,
    kv_mb_per_1k_tokens_q4: u64,
    base_toks_per_sec_q4: f64,
    repeating_layers: u32,
    context_cap: u32,
}

#[derive(Debug, Clone, Copy)]
struct ModelRuntimeHarness {
    paged_attn: &'static str,
    pa_cache_type: Option<&'static str>,
    pa_memory_fraction: Option<&'static str>,
    force_no_mmap: bool,
    force_language_model_only: bool,
    disable_flash_attn: bool,
    isq_singlethread: bool,
    isq_cpu_threads: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct ModelHarness {
    model: &'static str,
    sizing: EmpiricalSizingProfile,
    runtime: ModelRuntimeHarness,
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
    isq_singlethread: bool,
    isq_cpu_threads: Option<u32>,
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
const Q5K: QuantOption = QuantOption {
    label: "Q5K",
    runtime_isq: Some("Q5K"),
    weight_factor_milli: 1120,
    speed_factor_milli: 940,
};
const Q6K: QuantOption = QuantOption {
    label: "Q6K",
    runtime_isq: Some("Q6K"),
    weight_factor_milli: 1240,
    speed_factor_milli: 880,
};
#[derive(Debug, Clone, Copy)]
struct PlanSpec {
    quant: QuantOption,
    backend: BackendMode,
    context_target: Option<u32>,
    context_fraction_milli: u32,
    min_context_required: u32,
    per_gpu_headroom_mb: u64,
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
    root: &Path,
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
    let bundle = build_bundle_for_model(root, model, selected_preset, &hardware, env_map)?;
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
    env_map.insert("CTOX_ENGINE_MODEL".to_string(), plan.model.clone());
    env_map.insert(
        "CTOX_ENGINE_MAX_SEQ_LEN".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert(
        "CTOX_ENGINE_REALIZED_MAX_SEQ_LEN".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert(
        "CTOX_CHAT_MODEL_REALIZED_CONTEXT".to_string(),
        plan.max_seq_len.to_string(),
    );
    env_map.insert("CTOX_ENGINE_REALIZED_MODEL".to_string(), plan.model.clone());
    if let Some(runtime_isq) = &plan.runtime_isq {
        env_map.insert("CTOX_ENGINE_ISQ".to_string(), runtime_isq.clone());
    }
    env_map.insert(
        "CTOX_ENGINE_PAGED_ATTN".to_string(),
        plan.paged_attn.clone(),
    );
    if let Some(cache_type) = &plan.pa_cache_type {
        env_map.insert("CTOX_ENGINE_PA_CACHE_TYPE".to_string(), cache_type.clone());
    }
    if let Some(memory_fraction) = &plan.pa_memory_fraction {
        env_map.insert(
            "CTOX_ENGINE_PA_MEMORY_FRACTION".to_string(),
            memory_fraction.clone(),
        );
    }
    env_map.insert(
        "CTOX_ENGINE_DISABLE_NCCL".to_string(),
        if plan.disable_nccl { "1" } else { "0" }.to_string(),
    );
    if let Some(backend) = &plan.tensor_parallel_backend {
        env_map.insert(
            "CTOX_ENGINE_TENSOR_PARALLEL_BACKEND".to_string(),
            backend.clone(),
        );
    }
    if let Some(world_size) = plan.mn_local_world_size {
        env_map.insert(
            "CTOX_ENGINE_MN_LOCAL_WORLD_SIZE".to_string(),
            world_size.to_string(),
        );
    }
    env_map.insert(
        "CTOX_ENGINE_MAX_BATCH_SIZE".to_string(),
        plan.max_batch_size.to_string(),
    );
    env_map.insert(
        "CTOX_ENGINE_MAX_SEQS".to_string(),
        plan.max_seqs.to_string(),
    );
    env_map.insert(
        "CTOX_ENGINE_CUDA_VISIBLE_DEVICES".to_string(),
        plan.cuda_visible_devices.clone(),
    );
    if let Some(device_layers) = &plan.device_layers {
        env_map.insert(
            "CTOX_ENGINE_DEVICE_LAYERS".to_string(),
            device_layers.clone(),
        );
    }
    if let Some(topology) = &plan.topology {
        let topology_path = if Path::new(topology).is_absolute() {
            topology.clone()
        } else {
            root.join(topology).display().to_string()
        };
        env_map.insert("CTOX_ENGINE_TOPOLOGY".to_string(), topology_path);
    }
    if plan.allow_device_layers_with_topology {
        env_map.insert(
            "CTOX_ENGINE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY".to_string(),
            "1".to_string(),
        );
    }
    if let Some(ordinal) = plan.nm_device_ordinal {
        env_map.insert(
            "CTOX_ENGINE_NM_DEVICE_ORDINAL".to_string(),
            ordinal.to_string(),
        );
    }
    if let Some(ordinal) = plan.base_device_ordinal {
        env_map.insert(
            "CTOX_ENGINE_BASE_DEVICE_ORDINAL".to_string(),
            ordinal.to_string(),
        );
    }
    if let Some(backend) = &plan.moe_experts_backend {
        env_map.insert(
            "CTOX_ENGINE_MOE_EXPERTS_BACKEND".to_string(),
            backend.clone(),
        );
    }
    if plan.disable_flash_attn {
        env_map.insert(
            "CTOX_ENGINE_DISABLE_FLASH_ATTN".to_string(),
            "1".to_string(),
        );
    }
    if plan.force_no_mmap {
        env_map.insert("CTOX_ENGINE_NO_MMAP".to_string(), "1".to_string());
    }
    if plan.force_language_model_only {
        env_map.insert(
            "CTOX_ENGINE_LANGUAGE_MODEL_ONLY".to_string(),
            "1".to_string(),
        );
    }
    if plan.isq_singlethread {
        env_map.insert("CTOX_ENGINE_ISQ_SINGLETHREAD".to_string(), "1".to_string());
    }
    if let Some(cpu_threads) = plan.isq_cpu_threads {
        env_map.insert(
            "CTOX_ENGINE_ISQ_CPU_THREADS".to_string(),
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

pub fn reconcile_chat_runtime_plan(root: &Path) -> Result<Option<ChatRuntimePlan>> {
    let mut env_map = runtime_env::load_runtime_env_map(root).unwrap_or_default();
    let plan = apply_chat_runtime_plan(root, &mut env_map)?;
    runtime_env::save_runtime_env_map(root, &env_map)?;
    Ok(plan)
}

pub fn load_persisted_chat_runtime_plan(root: &Path) -> Result<Option<ChatRuntimePlan>> {
    let path = root.join(CHAT_PLAN_RELATIVE_PATH);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read chat runtime plan {}", path.display()))?;
    let plan = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse chat runtime plan {}", path.display()))?;
    Ok(Some(plan))
}

pub fn clear_chat_plan_env(env_map: &mut BTreeMap<String, String>) {
    for key in [
        "CTOX_CHAT_RUNTIME_PLAN_DIGEST",
        "CTOX_CHAT_RUNTIME_PLAN_ACTIVE",
        "CTOX_CHAT_RUNTIME_PLAN_PATH",
        "CTOX_ENGINE_FROM_UQFF",
        "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
        "CTOX_CHAT_COMPACTION_MIN_TOKENS",
        "CTOX_ENGINE_ISQ",
        "CTOX_ENGINE_PAGED_ATTN",
        "CTOX_ENGINE_PA_CACHE_TYPE",
        "CTOX_ENGINE_PA_MEMORY_FRACTION",
        "CTOX_ENGINE_PA_CONTEXT_LEN",
        "CTOX_ENGINE_DISABLE_NCCL",
        "CTOX_ENGINE_TENSOR_PARALLEL_BACKEND",
        "CTOX_ENGINE_MN_LOCAL_WORLD_SIZE",
        "CTOX_ENGINE_MAX_BATCH_SIZE",
        "CTOX_ENGINE_MAX_SEQS",
        "CTOX_ENGINE_DISABLE_FLASH_ATTN",
        "CTOX_ENGINE_NO_MMAP",
        "CTOX_ENGINE_LANGUAGE_MODEL_ONLY",
        "CTOX_ENGINE_ISQ_SINGLETHREAD",
        "CTOX_ENGINE_ISQ_CPU_THREADS",
        "CTOX_ENGINE_MAX_SEQ_LEN",
        "CTOX_ENGINE_NUM_DEVICE_LAYERS",
        "CTOX_ENGINE_DEVICE_LAYERS",
        "CTOX_ENGINE_TOPOLOGY",
        "CTOX_ENGINE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY",
        "CTOX_ENGINE_NM_DEVICE_ORDINAL",
        "CTOX_ENGINE_BASE_DEVICE_ORDINAL",
        "CTOX_ENGINE_MOE_EXPERTS_BACKEND",
        "CTOX_ENGINE_CUDA_VISIBLE_DEVICES",
        "CTOX_ENGINE_REALIZED_MAX_SEQ_LEN",
        "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
        "CTOX_ENGINE_REALIZED_MODEL",
    ] {
        env_map.remove(key);
    }
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
    root: &Path,
    model: &str,
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> Result<ChatPresetBundle> {
    if let Some(bundle) =
        build_manifest_bundle_with_root(Some(root), model, selected_preset, hardware, env_map)?
    {
        return Ok(bundle);
    }
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
        "nvidia/Nemotron-Cascade-2-30B-A3B" => Ok(build_nemotron_cascade_bundle_with_root(
            Some(root),
            selected_preset,
            hardware,
            env_map,
        )),
        "zai-org/GLM-4.7-Flash" => Ok(build_glm47_flash_bundle(selected_preset, hardware, env_map)),
        other => anyhow::bail!("unsupported runtime planner model: {other}"),
    }
}

fn build_manifest_bundle_with_root(
    root: Option<&Path>,
    model: &str,
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> Result<Option<ChatPresetBundle>> {
    let Some(manifest) = runtime_manifest(root, model) else {
        return Ok(None);
    };
    let harness = harness_from_manifest(&manifest);
    let quality = plan_from_specs(
        root,
        harness,
        ChatPreset::Quality,
        &plan_specs_from_manifest_profile(&manifest.quality),
        hardware,
        env_map,
    );
    let max_context = plan_from_specs(
        root,
        harness,
        ChatPreset::MaxContext,
        &plan_specs_from_manifest_profile(&manifest.max_context),
        hardware,
        env_map,
    );
    let performance = plan_from_specs(
        root,
        harness,
        ChatPreset::Performance,
        &plan_specs_from_manifest_profile(&manifest.performance),
        hardware,
        env_map,
    );
    Ok(Some(bundle_from_plans(
        harness.model,
        selected_preset,
        hardware,
        vec![quality, max_context, performance],
    )))
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
    root: Option<&Path>,
    harness: ModelHarness,
    preset: ChatPreset,
    specs: &[PlanSpec],
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatRuntimePlan {
    let snapshot = resource_state::inspect_resource_snapshot();
    let mut candidates = Vec::new();
    for spec in specs {
        if let Some(candidate) = build_candidate(
            root,
            harness,
            preset,
            *spec,
            hardware,
            env_map,
            snapshot.as_ref(),
        ) {
            candidates.push(candidate);
        }
    }
    if let Some(best) = choose_best_candidate(preset, &candidates) {
        return best;
    }
    let fallback_spec = select_floor_fallback_spec(specs).unwrap_or(PlanSpec {
        quant: Q4K,
        backend: BackendMode::DeviceLayers,
        context_target: Some(MIN_POLICY_CONTEXT),
        context_fraction_milli: 1000,
        min_context_required: MIN_POLICY_CONTEXT,
        per_gpu_headroom_mb: 0,
        max_batch_size: 1,
        max_seqs: 1,
    });
    build_floor_fallback_plan(
        root,
        harness,
        preset,
        fallback_spec,
        hardware,
        env_map,
        snapshot.as_ref(),
    )
}

fn select_floor_fallback_spec(specs: &[PlanSpec]) -> Option<PlanSpec> {
    specs.iter().copied().min_by(|left, right| {
        let backend_rank = |backend: BackendMode| match backend {
            BackendMode::DeviceLayers => 0u8,
            BackendMode::Nccl => 1u8,
        };
        (
            left.quant.weight_factor_milli,
            backend_rank(left.backend),
            left.per_gpu_headroom_mb,
            left.max_batch_size,
            left.max_seqs,
            left.context_target.unwrap_or(u32::MAX),
        )
            .cmp(&(
                right.quant.weight_factor_milli,
                backend_rank(right.backend),
                right.per_gpu_headroom_mb,
                right.max_batch_size,
                right.max_seqs,
                right.context_target.unwrap_or(u32::MAX),
            ))
    })
}

fn build_gpt_oss_20b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(
        None,
        "openai/gpt-oss-20b",
        selected_preset,
        hardware,
        env_map,
    )
    .expect("gpt-oss manifest bundle should resolve")
    .expect("gpt-oss manifest bundle should exist")
}

fn build_qwen35_4b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(None, "Qwen/Qwen3.5-4B", selected_preset, hardware, env_map)
        .expect("qwen3.5-4b manifest bundle should resolve")
        .expect("qwen3.5-4b manifest bundle should exist")
}

fn build_qwen35_9b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(None, "Qwen/Qwen3.5-9B", selected_preset, hardware, env_map)
        .expect("qwen3.5-9b manifest bundle should resolve")
        .expect("qwen3.5-9b manifest bundle should exist")
}

fn build_qwen35_27b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(None, "Qwen/Qwen3.5-27B", selected_preset, hardware, env_map)
        .expect("qwen3.5-27b manifest bundle should resolve")
        .expect("qwen3.5-27b manifest bundle should exist")
}

fn build_qwen35_35b_a3b_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(
        None,
        "Qwen/Qwen3.5-35B-A3B",
        selected_preset,
        hardware,
        env_map,
    )
    .expect("qwen3.5-35b-a3b manifest bundle should resolve")
    .expect("qwen3.5-35b-a3b manifest bundle should exist")
}

fn build_nemotron_cascade_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_nemotron_cascade_bundle_with_root(None, selected_preset, hardware, env_map)
}

fn build_nemotron_cascade_bundle_with_root(
    root: Option<&Path>,
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(
        root,
        "nvidia/Nemotron-Cascade-2-30B-A3B",
        selected_preset,
        hardware,
        env_map,
    )
    .expect("nemotron manifest bundle should resolve")
    .expect("nemotron manifest bundle should exist")
}

fn build_glm47_flash_bundle(
    selected_preset: ChatPreset,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
) -> ChatPresetBundle {
    build_manifest_bundle_with_root(
        None,
        "zai-org/GLM-4.7-Flash",
        selected_preset,
        hardware,
        env_map,
    )
    .expect("glm-4.7-flash manifest bundle should resolve")
    .expect("glm-4.7-flash manifest bundle should exist")
}

fn choose_best_candidate(
    preset: ChatPreset,
    candidates: &[ChatRuntimePlan],
) -> Option<ChatRuntimePlan> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|left, right| {
        let rank = |plan: &ChatRuntimePlan| -> (i64, i64, i64, i64) {
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
            let utilization_rank = -(plan
                .gpu_allocations
                .iter()
                .map(|gpu| gpu.free_headroom_mb as i64)
                .sum::<i64>());
            match preset {
                ChatPreset::Quality => (
                    quant_rank_high,
                    plan.max_seq_len as i64,
                    utilization_rank,
                    (plan.expected_tok_s * 100.0).round() as i64,
                ),
                ChatPreset::MaxContext => (
                    plan.max_seq_len as i64,
                    quant_rank_low,
                    utilization_rank,
                    (plan.expected_tok_s * 100.0).round() as i64,
                ),
                ChatPreset::Performance => (
                    (plan.expected_tok_s * 100.0).round() as i64,
                    plan.max_seqs as i64,
                    utilization_rank,
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
    root: Option<&Path>,
    harness: ModelHarness,
    preset: ChatPreset,
    spec: PlanSpec,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
    snapshot: Option<&resource_state::ResourceSnapshot>,
) -> Option<ChatRuntimePlan> {
    let placement = model_placement_profile(root, harness.model);
    let runtime = resolve_harness_runtime(harness, hardware);
    let reclaim_managed_chat_backend = env_map
        .get("CTOX_ACTIVE_MODEL")
        .map(String::as_str)
        .filter(|model| engine::uses_ctox_proxy_model(model))
        .is_some();
    let compaction_policy = compaction_policy_for_preset(preset);
    let quant = spec.quant;
    let backend = spec.backend;
    let mut gpu_indices =
        select_chat_gpu_indices(preset, backend, &hardware.gpus, placement.primary_gpu_index);
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

    let aux_reserves = compute_aux_reserves_mb(root, hardware, env_map, preset, &gpu_indices);
    let mut per_gpu_budgets = Vec::new();
    for gpu in &hardware.gpus {
        let desktop_reserve = desktop_reserve_mb_for_gpu(&placement, hardware, gpu.index);
        let aux_reserve = *aux_reserves.get(&gpu.index).unwrap_or(&0);
        let live_total_or_free = if reclaim_managed_chat_backend {
            gpu.total_mb
        } else {
            snapshot
                .and_then(|state| state.gpu(gpu.index))
                .map(|state| state.free_mb)
                .unwrap_or(gpu.total_mb)
        };
        let usable = live_total_or_free
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

    let non_repeating_weight_mb = empirical_non_repeating_weight_mb(harness, quant);
    let repeating_weight_mb = empirical_repeating_weight_mb(harness, quant);
    let weight_mb = non_repeating_weight_mb.saturating_add(repeating_weight_mb);
    let fixed_overhead_mb = fixed_overhead_mb(backend, gpu_indices.len())
        .saturating_add(empirical_load_peak_slack_mb(harness, quant, backend));
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
    let safety_headroom_mb = spec
        .per_gpu_headroom_mb
        .saturating_mul(gpu_indices.len() as u64);
    let kv_budget_cap_mb = kv_budget_cap_mb.saturating_sub(safety_headroom_mb);
    if kv_budget_cap_mb == 0 {
        return None;
    }

    let effective_concurrency = spec.max_seqs.max(spec.max_batch_size).max(1) as u64;
    let kv_mb_per_1k = scale_mb(
        harness.sizing.kv_mb_per_1k_tokens_q4,
        quant.weight_factor_milli,
    )
    .saturating_mul(effective_concurrency);
    let raw_context =
        (((kv_budget_cap_mb as f64) / (kv_mb_per_1k.max(1) as f64)) * 1024.0).floor() as u32;
    let mut plan_context = align_context(raw_context.min(harness.sizing.context_cap));
    plan_context =
        align_context(((plan_context as u64 * spec.context_fraction_milli as u64) / 1000) as u32);
    if let Some(target) = spec.context_target {
        plan_context = plan_context.min(align_context(target));
    }
    let policy_floor_ok = plan_context >= spec.min_context_required.max(MIN_POLICY_CONTEXT);
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
        harness,
        &placement,
        &runtime,
        fixed_device_layers_override,
        non_repeating_weight_mb,
        repeating_weight_mb,
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
        BackendMode::DeviceLayers => Some(device_layers_cli(
            &allocations,
            harness.sizing.repeating_layers,
        )),
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
    if let Some(snapshot) = snapshot {
        rationale.push(format!("resource snapshot {}", snapshot.source));
    }
    if reclaim_managed_chat_backend {
        rationale.push("planner treats managed chat backend VRAM as reclaimable".to_string());
    }
    if spec.per_gpu_headroom_mb > 0 {
        rationale.push(format!(
            "reserved {}MB per GPU for load/runtime headroom",
            spec.per_gpu_headroom_mb
        ));
    }
    if harness.runtime.disable_flash_attn {
        rationale.push("flash-attn disabled for this model/runtime path".to_string());
    }
    if runtime.force_no_mmap {
        rationale.push("mmap disabled for this model/runtime path".to_string());
    }
    if harness.runtime.force_language_model_only {
        rationale.push("vision tower disabled for text-only runtime path".to_string());
    }
    if runtime.isq_singlethread && quant.runtime_isq.is_some() {
        rationale.push("ISQ is serialized to avoid model-load VRAM spikes".to_string());
    }
    if let Some(cpu_threads) = runtime
        .isq_cpu_threads
        .filter(|_| quant.runtime_isq.is_some())
    {
        rationale.push(format!("ISQ cpu threads {cpu_threads}"));
    }
    if let Some(backend_override) = runtime.moe_experts_backend {
        rationale.push(format!("moe experts backend {}", backend_override));
    }
    let pa_cache_type =
        engine::resolve_model_pa_cache_type(harness.model, harness.runtime.pa_cache_type, env_map);
    let paged_attn = engine::resolve_model_paged_attn(
        harness.model,
        harness.runtime.paged_attn,
        pa_cache_type.as_deref(),
    );
    Some(ChatRuntimePlan {
        model: harness.model.to_string(),
        preset,
        quantization: quant.label.to_string(),
        runtime_isq: quant.runtime_isq.map(str::to_string),
        max_seq_len: plan_context,
        compaction_threshold_percent: compaction_policy.threshold_percent,
        compaction_min_tokens: compaction_policy.min_tokens,
        min_context_floor_applied: true,
        paged_attn,
        pa_cache_type,
        pa_memory_fraction: harness.runtime.pa_memory_fraction.map(str::to_string),
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
        disable_flash_attn: harness.runtime.disable_flash_attn,
        force_no_mmap: runtime.force_no_mmap,
        force_language_model_only: harness.runtime.force_language_model_only,
        isq_singlethread: runtime.isq_singlethread && quant.runtime_isq.is_some(),
        isq_cpu_threads: runtime
            .isq_cpu_threads
            .filter(|_| quant.runtime_isq.is_some()),
        expected_tok_s,
        hardware_fingerprint: hardware.fingerprint.clone(),
        rationale,
        gpu_allocations: allocations,
    })
}

fn build_floor_fallback_plan(
    root: Option<&Path>,
    harness: ModelHarness,
    preset: ChatPreset,
    fallback_spec: PlanSpec,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
    snapshot: Option<&resource_state::ResourceSnapshot>,
) -> ChatRuntimePlan {
    let runtime = resolve_harness_runtime(harness, hardware);
    let placement = model_placement_profile(root, harness.model);
    let plan = build_candidate(
        root,
        harness,
        preset,
        fallback_spec,
        hardware,
        env_map,
        snapshot,
    )
    .unwrap_or_else(|| {
        let compaction_policy = compaction_policy_for_preset(preset);
        let pa_cache_type = engine::resolve_model_pa_cache_type(
            harness.model,
            harness.runtime.pa_cache_type,
            env_map,
        );
        let paged_attn = engine::resolve_model_paged_attn(
            harness.model,
            harness.runtime.paged_attn,
            pa_cache_type.as_deref(),
        );
        ChatRuntimePlan {
            model: harness.model.to_string(),
            preset,
            quantization: fallback_spec.quant.label.to_string(),
            runtime_isq: fallback_spec.quant.runtime_isq.map(str::to_string),
            max_seq_len: MIN_POLICY_CONTEXT,
            compaction_threshold_percent: compaction_policy.threshold_percent,
            compaction_min_tokens: compaction_policy.min_tokens,
            min_context_floor_applied: false,
            paged_attn,
            pa_cache_type,
            pa_memory_fraction: harness.runtime.pa_memory_fraction.map(str::to_string),
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
            disable_flash_attn: harness.runtime.disable_flash_attn,
            force_no_mmap: runtime.force_no_mmap,
            force_language_model_only: harness.runtime.force_language_model_only,
            isq_singlethread: runtime.isq_singlethread && fallback_spec.quant.runtime_isq.is_some(),
            isq_cpu_threads: runtime
                .isq_cpu_threads
                .filter(|_| fallback_spec.quant.runtime_isq.is_some()),
            expected_tok_s: harness.sizing.base_toks_per_sec_q4 * 0.65,
            hardware_fingerprint: hardware.fingerprint.clone(),
            rationale: vec!["policy floor fallback".to_string()],
            gpu_allocations: hardware
                .gpus
                .iter()
                .map(|gpu| PlannedGpuAllocation {
                    gpu_index: gpu.index,
                    name: gpu.name.clone(),
                    total_mb: gpu.total_mb,
                    desktop_reserve_mb: desktop_reserve_mb_for_gpu(&placement, hardware, gpu.index),
                    aux_reserve_mb: 0,
                    chat_budget_mb: 0,
                    repeating_weight_mb: 0,
                    weight_mb: 0,
                    kv_cache_mb: 0,
                    free_headroom_mb: gpu.total_mb,
                    chat_enabled: true,
                })
                .collect(),
        }
    });
    let mut fallback = plan;
    fallback.rationale.push(format!(
        "hardware could not satisfy the full preset policy; kept {} and the 16k floor",
        fallback_spec.quant.label
    ));
    fallback
}

fn resolve_harness_runtime(
    harness: ModelHarness,
    _hardware: &HardwareProfile,
) -> ResolvedHarnessRuntime {
    ResolvedHarnessRuntime {
        fixed_device_layers: None,
        fixed_cuda_visible_devices: None,
        topology_rel_path: None,
        allow_device_layers_with_topology: false,
        // Keep non-repeating tensors and the base runtime device on the first
        // visible GPU. The planner already sorts visible devices so that the
        // manifest primary GPU comes first.
        nm_device_ordinal: Some(0),
        base_device_ordinal: Some(0),
        moe_experts_backend: None,
        force_no_mmap: harness.runtime.force_no_mmap,
        isq_singlethread: harness.runtime.isq_singlethread,
        isq_cpu_threads: harness.runtime.isq_cpu_threads,
    }
}

fn distribute_allocations(
    backend: BackendMode,
    hardware: &HardwareProfile,
    gpu_indices: &[usize],
    aux_reserves: &BTreeMap<usize, u64>,
    harness: ModelHarness,
    placement: &model_manifest::ManifestPlacementProfile,
    resolved_runtime: &ResolvedHarnessRuntime,
    fixed_device_layers: Option<&str>,
    non_repeating_weight_mb: u64,
    repeating_weight_mb: u64,
    kv_budget_mb: u64,
) -> Vec<PlannedGpuAllocation> {
    let selected = hardware
        .gpus
        .iter()
        .filter(|gpu| gpu_indices.contains(&gpu.index))
        .collect::<Vec<_>>();
    let fixed_layer_weights = fixed_device_layer_weights(
        fixed_device_layers,
        gpu_indices,
        harness.sizing.repeating_layers,
    );
    let base_weight_gpu = match backend {
        BackendMode::Nccl => None,
        BackendMode::DeviceLayers => resolved_runtime
            .base_device_ordinal
            .and_then(|gpu| {
                gpu_indices
                    .contains(&(gpu as usize))
                    .then_some(gpu as usize)
            })
            .or_else(|| {
                gpu_indices
                    .contains(&placement.primary_gpu_index)
                    .then_some(placement.primary_gpu_index)
            })
            .or_else(|| gpu_indices.first().copied()),
    };
    let device_layer_capacity_weights = selected
        .iter()
        .map(|gpu| {
            let capacity = gpu
                .total_mb
                .saturating_sub(desktop_reserve_mb_for_gpu(placement, hardware, gpu.index))
                .saturating_sub(*aux_reserves.get(&gpu.index).unwrap_or(&0));
            if matches!(backend, BackendMode::DeviceLayers) && base_weight_gpu == Some(gpu.index) {
                capacity.saturating_sub(non_repeating_weight_mb)
            } else {
                capacity
            }
        })
        .collect::<Vec<_>>();
    let weight_shares = match backend {
        BackendMode::Nccl => even_shares(
            non_repeating_weight_mb.saturating_add(repeating_weight_mb),
            selected.len(),
        ),
        BackendMode::DeviceLayers => match fixed_layer_weights.as_ref() {
            Some(weights) => proportional_shares(repeating_weight_mb, weights),
            None => proportional_shares(repeating_weight_mb, &device_layer_capacity_weights),
        },
    };
    let kv_shares = match backend {
        BackendMode::Nccl => even_shares(kv_budget_mb, selected.len()),
        BackendMode::DeviceLayers => match fixed_layer_weights.as_ref() {
            Some(weights) => proportional_shares(kv_budget_mb, weights),
            None => proportional_shares(kv_budget_mb, &device_layer_capacity_weights),
        },
    };

    let mut selected_index = 0usize;
    hardware
        .gpus
        .iter()
        .map(|gpu| {
            let desktop_reserve = desktop_reserve_mb_for_gpu(placement, hardware, gpu.index);
            let aux_reserve = *aux_reserves.get(&gpu.index).unwrap_or(&0);
            let chat_enabled = gpu_indices.contains(&gpu.index);
            let (repeating_weight_share, mut weight_share, kv_share) = if chat_enabled {
                let share = (
                    *weight_shares.get(selected_index).unwrap_or(&0),
                    *weight_shares.get(selected_index).unwrap_or(&0),
                    *kv_shares.get(selected_index).unwrap_or(&0),
                );
                selected_index += 1;
                share
            } else {
                (0, 0, 0)
            };
            if chat_enabled
                && matches!(backend, BackendMode::DeviceLayers)
                && base_weight_gpu == Some(gpu.index)
            {
                weight_share = weight_share.saturating_add(non_repeating_weight_mb);
            }
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
                repeating_weight_mb: repeating_weight_share,
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
        .filter(|allocation| allocation.chat_enabled && allocation.repeating_weight_mb > 0)
        .collect::<Vec<_>>();
    let total_budget = selected
        .iter()
        .map(|allocation| allocation.repeating_weight_mb)
        .sum::<u64>()
        .max(1);
    let total_layers = total_layers as u64;
    let mut raw = selected
        .iter()
        .map(|allocation| {
            (
                allocation.gpu_index,
                ((allocation.repeating_weight_mb as f64 / total_budget as f64)
                    * total_layers as f64)
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
    root: Option<&Path>,
    hardware: &HardwareProfile,
    env_map: &BTreeMap<String, String>,
    preset: ChatPreset,
    chat_gpu_indices: &[usize],
) -> BTreeMap<usize, u64> {
    let aux_models = [
        engine::auxiliary_model_selection(
            engine::AuxiliaryRole::Embedding,
            env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
        ),
        engine::auxiliary_model_selection(
            engine::AuxiliaryRole::Stt,
            env_map.get("CTOX_STT_MODEL").map(String::as_str),
        ),
        engine::auxiliary_model_selection(
            engine::AuxiliaryRole::Tts,
            env_map.get("CTOX_TTS_MODEL").map(String::as_str),
        ),
    ];

    let mut reserves: BTreeMap<usize, u64> = BTreeMap::new();
    for selection in aux_models {
        let reserve_mb = if selection.compute_target == engine::ComputeTarget::Cpu {
            0
        } else {
            auxiliary_manifest(root, selection.request_model)
                .map(|manifest| manifest.gpu_reserve_mb)
                .unwrap_or_else(|| selection.gpu_reserve_mb())
        };
        if reserve_mb == 0 {
            continue;
        }
        let explicit_devices =
            parse_csv_indices(env_map.get("CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES"));
        let default_distribution =
            default_aux_distribution(root, &selection, hardware, preset, chat_gpu_indices);
        let target_devices = if explicit_devices.is_empty() {
            default_distribution
        } else {
            explicit_devices
        };
        if target_devices.is_empty() {
            continue;
        }
        let shares = even_shares(reserve_mb, target_devices.len());
        for (idx, gpu_index) in target_devices.iter().enumerate() {
            let entry = reserves.entry(*gpu_index).or_insert(0);
            *entry = (*entry).saturating_add(*shares.get(idx).unwrap_or(&0u64));
        }
    }
    reserves
}

fn default_aux_distribution(
    root: Option<&Path>,
    selection: &engine::AuxiliaryModelSelection,
    hardware: &HardwareProfile,
    preset: ChatPreset,
    chat_gpu_indices: &[usize],
) -> Vec<usize> {
    let total_gpu_count = hardware.gpus.len();
    if total_gpu_count == 0 {
        return Vec::new();
    }
    if let Some(primary_gpu) = auxiliary_manifest(root, selection.request_model)
        .filter(|manifest| manifest.placement.use_primary_gpu_by_default)
        .map(|manifest| manifest.placement.primary_gpu_index)
        .filter(|index| hardware.gpus.iter().any(|gpu| gpu.index == *index))
    {
        return vec![primary_gpu];
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
    let mut tps = harness.sizing.base_toks_per_sec_q4 * (quant.speed_factor_milli as f64 / 1000.0);
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
    primary_gpu_index: usize,
) -> Vec<usize> {
    if gpus.is_empty() {
        return Vec::new();
    }
    match backend {
        BackendMode::Nccl | BackendMode::DeviceLayers => {
            let mut selected = gpus.iter().map(|gpu| gpu.index).collect::<Vec<_>>();
            selected.sort_by_key(|index| (*index != primary_gpu_index, *index));
            if matches!(backend, BackendMode::DeviceLayers)
                && matches!(preset, ChatPreset::Performance)
                && selected.len() > 1
            {
                return selected;
            }
            selected
        }
    }
}

fn scale_mb(base_mb: u64, factor_milli: u32) -> u64 {
    ((base_mb as u128 * factor_milli as u128) / 1000u128) as u64
}

fn empirical_repeating_weight_mb(harness: ModelHarness, quant: QuantOption) -> u64 {
    scale_mb(
        harness
            .sizing
            .repeating_layer_weight_mb_q4
            .saturating_mul(harness.sizing.repeating_layers as u64),
        quant.weight_factor_milli,
    )
}

fn empirical_non_repeating_weight_mb(harness: ModelHarness, quant: QuantOption) -> u64 {
    scale_mb(
        harness.sizing.non_repeating_weight_mb_q4,
        quant.weight_factor_milli,
    )
}

fn empirical_load_peak_slack_mb(
    harness: ModelHarness,
    quant: QuantOption,
    backend: BackendMode,
) -> u64 {
    let base = scale_mb(
        harness.sizing.load_peak_slack_mb_q4,
        quant.weight_factor_milli,
    );
    match backend {
        BackendMode::Nccl => base / 2,
        BackendMode::DeviceLayers => base,
    }
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
                .filter(|value| engine::is_openai_api_chat_model(value))
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
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 3_400,
            repeating_layer_weight_mb_q4: 516,
            load_peak_slack_mb_q4: 512,
            kv_mb_per_1k_tokens_q4: 220,
            base_toks_per_sec_q4: 90.0,
            repeating_layers: 24,
            context_cap: 131_072,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.80"),
            force_no_mmap: false,
            force_language_model_only: false,
            disable_flash_attn: false,
            isq_singlethread: false,
            isq_cpu_threads: None,
        },
    }
}

fn plan_qwen35_4b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-4B",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 720,
            repeating_layer_weight_mb_q4: 90,
            load_peak_slack_mb_q4: 384,
            kv_mb_per_1k_tokens_q4: 78,
            base_toks_per_sec_q4: 140.0,
            repeating_layers: 32,
            context_cap: 262_144,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.80"),
            force_no_mmap: false,
            force_language_model_only: false,
            disable_flash_attn: true,
            isq_singlethread: false,
            isq_cpu_threads: None,
        },
    }
}

fn plan_qwen35_9b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-9B",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 940,
            repeating_layer_weight_mb_q4: 180,
            load_peak_slack_mb_q4: 512,
            kv_mb_per_1k_tokens_q4: 112,
            base_toks_per_sec_q4: 95.0,
            repeating_layers: 32,
            context_cap: 262_144,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.80"),
            force_no_mmap: false,
            force_language_model_only: false,
            disable_flash_attn: true,
            isq_singlethread: false,
            isq_cpu_threads: None,
        },
    }
}

fn plan_qwen35_27b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-27B",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 1_660,
            repeating_layer_weight_mb_q4: 260,
            load_peak_slack_mb_q4: 768,
            kv_mb_per_1k_tokens_q4: 248,
            base_toks_per_sec_q4: 45.0,
            repeating_layers: 64,
            context_cap: 262_144,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.80"),
            force_no_mmap: false,
            force_language_model_only: false,
            disable_flash_attn: true,
            isq_singlethread: false,
            isq_cpu_threads: None,
        },
    }
}

fn plan_qwen35_35b_a3b() -> ModelHarness {
    ModelHarness {
        model: "Qwen/Qwen3.5-35B-A3B",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 3_500,
            repeating_layer_weight_mb_q4: 450,
            load_peak_slack_mb_q4: 2_200,
            kv_mb_per_1k_tokens_q4: 270,
            base_toks_per_sec_q4: 38.0,
            repeating_layers: 40,
            context_cap: 262_144,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.80"),
            force_no_mmap: true,
            force_language_model_only: true,
            disable_flash_attn: true,
            isq_singlethread: true,
            isq_cpu_threads: None,
        },
    }
}

fn manifest_candidate(
    quantization: model_manifest::ManifestQuantization,
    backend: model_manifest::ManifestBackendMode,
    max_batch_size: u32,
    max_seqs: u32,
    context_fraction_milli: u32,
    context_target_cap: Option<u32>,
    min_context_required: u32,
    per_gpu_headroom_mb: u64,
) -> model_manifest::PresetCandidateSpec {
    model_manifest::PresetCandidateSpec {
        quantization,
        backend,
        max_batch_size,
        max_seqs,
        context_fraction_milli,
        context_target_cap,
        min_context_required,
        per_gpu_headroom_mb,
    }
}

fn manifest_profile(
    objective: model_manifest::RuntimeObjectiveLabel,
    candidates: Vec<model_manifest::PresetCandidateSpec>,
) -> model_manifest::ManifestPresetProfile {
    model_manifest::ManifestPresetProfile {
        objective,
        candidates,
    }
}

fn manifest_from_harness(
    harness: ModelHarness,
    placement: model_manifest::ManifestPlacementProfile,
    quality: model_manifest::ManifestPresetProfile,
    max_context: model_manifest::ManifestPresetProfile,
    performance: model_manifest::ManifestPresetProfile,
) -> model_manifest::RuntimeModelManifest {
    model_manifest::RuntimeModelManifest {
        model: harness.model.to_string(),
        sizing: model_manifest::ManifestSizingProfile {
            non_repeating_weight_mb_q4: harness.sizing.non_repeating_weight_mb_q4,
            repeating_layer_weight_mb_q4: harness.sizing.repeating_layer_weight_mb_q4,
            load_peak_slack_mb_q4: harness.sizing.load_peak_slack_mb_q4,
            kv_mb_per_1k_tokens_q4: harness.sizing.kv_mb_per_1k_tokens_q4,
            base_toks_per_sec_q4: harness.sizing.base_toks_per_sec_q4,
            repeating_layers: harness.sizing.repeating_layers,
            context_cap: harness.sizing.context_cap,
        },
        runtime_defaults: model_manifest::ManifestRuntimeDefaults {
            paged_attn: harness.runtime.paged_attn.to_string(),
            pa_cache_type: harness.runtime.pa_cache_type.map(str::to_string),
            pa_memory_fraction: harness.runtime.pa_memory_fraction.map(str::to_string),
            force_no_mmap: harness.runtime.force_no_mmap,
            force_language_model_only: harness.runtime.force_language_model_only,
            disable_flash_attn: harness.runtime.disable_flash_attn,
            isq_singlethread: harness.runtime.isq_singlethread,
            isq_cpu_threads: harness.runtime.isq_cpu_threads,
        },
        placement,
        quality,
        max_context,
        performance,
    }
}

fn default_placement_profile() -> model_manifest::ManifestPlacementProfile {
    model_manifest::ManifestPlacementProfile {
        primary_gpu_index: 0,
        primary_gpu_holds_non_repeating: true,
        primary_gpu_desktop_reserve_mb: DEFAULT_GPU0_DESKTOP_RESERVE_MB,
    }
}

fn default_gpt_oss_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_gpt_oss_20b(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::NativeMxfp4,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(131_072),
                MIN_POLICY_CONTEXT,
                0,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::NativeMxfp4,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(131_072),
                MIN_POLICY_CONTEXT,
                0,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::NativeMxfp4,
                model_manifest::ManifestBackendMode::Nccl,
                2,
                2,
                1000,
                Some(65_536),
                MIN_POLICY_CONTEXT,
                0,
            )],
        ),
    )
}

fn default_qwen35_4b_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_qwen35_4b(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    512,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(262_144),
                MIN_POLICY_CONTEXT,
                0,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                4,
                4,
                1000,
                Some(24_576),
                MIN_POLICY_CONTEXT,
                512,
            )],
        ),
    )
}

fn default_qwen35_9b_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_qwen35_9b(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(262_144),
                MIN_POLICY_CONTEXT,
                0,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                2,
                2,
                1000,
                Some(24_576),
                MIN_POLICY_CONTEXT,
                768,
            )],
        ),
    )
}

fn default_qwen35_27b_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_qwen35_27b(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(262_144),
                MIN_POLICY_CONTEXT,
                512,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                2,
                2,
                1000,
                Some(20_480),
                MIN_POLICY_CONTEXT,
                768,
            )],
        ),
    )
}

fn default_qwen35_35b_a3b_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_qwen35_35b_a3b(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    0,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    MIN_POLICY_CONTEXT,
                    0,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(65_536),
                    MIN_POLICY_CONTEXT,
                    1_024,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(16_384),
                    MIN_POLICY_CONTEXT,
                    0,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    2,
                    2,
                    1000,
                    Some(20_480),
                    MIN_POLICY_CONTEXT,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(16_384),
                    MIN_POLICY_CONTEXT,
                    0,
                ),
            ],
        ),
    )
}

fn default_nemotron_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_nemotron_cascade_from_manifest_seed(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    24_576,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    16_384,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    16_384,
                    0,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(262_144),
                    32_768,
                    0,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    900,
                    Some(65_536),
                    16_384,
                    2_048,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    2,
                    2,
                    1000,
                    Some(32_768),
                    16_384,
                    512,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(65_536),
                    16_384,
                    768,
                ),
            ],
        ),
    )
}

fn plan_nemotron_cascade_from_manifest_seed() -> ModelHarness {
    ModelHarness {
        model: "nvidia/Nemotron-Cascade-2-30B-A3B",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 2_800,
            repeating_layer_weight_mb_q4: 390,
            load_peak_slack_mb_q4: 1_600,
            kv_mb_per_1k_tokens_q4: 140,
            base_toks_per_sec_q4: 42.0,
            repeating_layers: 52,
            context_cap: 262_144,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.45"),
            force_no_mmap: false,
            force_language_model_only: false,
            disable_flash_attn: true,
            isq_singlethread: true,
            isq_cpu_threads: None,
        },
    }
}

fn default_glm47_flash_manifest() -> model_manifest::RuntimeModelManifest {
    manifest_from_harness(
        plan_glm47_flash(),
        default_placement_profile(),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Quality,
            vec![
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q6k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(65_536),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q5k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(65_536),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
                manifest_candidate(
                    model_manifest::ManifestQuantization::Q4k,
                    model_manifest::ManifestBackendMode::DeviceLayers,
                    1,
                    1,
                    1000,
                    Some(65_536),
                    MIN_POLICY_CONTEXT,
                    768,
                ),
            ],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::MaxContext,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                1,
                1,
                1000,
                Some(65_536),
                MIN_POLICY_CONTEXT,
                512,
            )],
        ),
        manifest_profile(
            model_manifest::RuntimeObjectiveLabel::Performance,
            vec![manifest_candidate(
                model_manifest::ManifestQuantization::Q4k,
                model_manifest::ManifestBackendMode::DeviceLayers,
                2,
                2,
                1000,
                Some(16_384),
                MIN_POLICY_CONTEXT,
                768,
            )],
        ),
    )
}

fn default_runtime_manifest_for_model(model: &str) -> Option<model_manifest::RuntimeModelManifest> {
    Some(match model.trim() {
        "openai/gpt-oss-20b" => default_gpt_oss_manifest(),
        "Qwen/Qwen3.5-4B" => default_qwen35_4b_manifest(),
        "Qwen/Qwen3.5-9B" => default_qwen35_9b_manifest(),
        "Qwen/Qwen3.5-27B" => default_qwen35_27b_manifest(),
        "Qwen/Qwen3.5-35B-A3B" => default_qwen35_35b_a3b_manifest(),
        "nvidia/Nemotron-Cascade-2-30B-A3B" => default_nemotron_manifest(),
        "zai-org/GLM-4.7-Flash" => default_glm47_flash_manifest(),
        _ => return None,
    })
}

fn model_placement_profile(
    root: Option<&Path>,
    model: &str,
) -> model_manifest::ManifestPlacementProfile {
    runtime_manifest(root, model)
        .map(|manifest| manifest.placement)
        .unwrap_or_else(default_placement_profile)
}

fn desktop_reserve_mb_for_gpu(
    placement: &model_manifest::ManifestPlacementProfile,
    hardware: &HardwareProfile,
    gpu_index: usize,
) -> u64 {
    if gpu_index == placement.primary_gpu_index {
        placement
            .primary_gpu_desktop_reserve_mb
            .max(hardware.gpu0_desktop_reserve_mb)
    } else {
        0
    }
}

fn runtime_manifest(
    root: Option<&Path>,
    model: &str,
) -> Option<model_manifest::RuntimeModelManifest> {
    root.and_then(|root| {
        model_manifest::load_runtime_model_manifest(root, model)
            .ok()
            .flatten()
    })
    .or_else(|| default_runtime_manifest_for_model(model))
}

fn default_auxiliary_manifest_for_model(
    model: &str,
) -> Option<model_manifest::AuxiliaryModelManifest> {
    Some(match model.trim() {
        "Qwen/Qwen3-Embedding-0.6B" => model_manifest::AuxiliaryModelManifest {
            model: model.to_string(),
            role: "embedding".to_string(),
            gpu_reserve_mb: 1_100,
            placement: model_manifest::AuxiliaryPlacementProfile {
                primary_gpu_index: 0,
                use_primary_gpu_by_default: true,
                supports_multi_gpu_expansion: false,
            },
        },
        "mistralai/Voxtral-Mini-4B-Realtime-2602" => model_manifest::AuxiliaryModelManifest {
            model: model.to_string(),
            role: "stt".to_string(),
            gpu_reserve_mb: 4_200,
            placement: model_manifest::AuxiliaryPlacementProfile {
                primary_gpu_index: 0,
                use_primary_gpu_by_default: true,
                supports_multi_gpu_expansion: false,
            },
        },
        "mistralai/Voxtral-4B-TTS-2603" => model_manifest::AuxiliaryModelManifest {
            model: model.to_string(),
            role: "tts".to_string(),
            gpu_reserve_mb: 1_400,
            placement: model_manifest::AuxiliaryPlacementProfile {
                primary_gpu_index: 0,
                use_primary_gpu_by_default: true,
                supports_multi_gpu_expansion: false,
            },
        },
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" | "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => {
            model_manifest::AuxiliaryModelManifest {
                model: model.to_string(),
                role: "tts".to_string(),
                gpu_reserve_mb: 1_400,
                placement: model_manifest::AuxiliaryPlacementProfile {
                    primary_gpu_index: 0,
                    use_primary_gpu_by_default: true,
                    supports_multi_gpu_expansion: false,
                },
            }
        }
        _ => return None,
    })
}

fn auxiliary_manifest(
    root: Option<&Path>,
    model: &str,
) -> Option<model_manifest::AuxiliaryModelManifest> {
    root.and_then(|root| {
        model_manifest::load_auxiliary_model_manifest(root, model)
            .ok()
            .flatten()
    })
    .or_else(|| default_auxiliary_manifest_for_model(model))
}

fn plan_specs_from_manifest_profile(
    profile: &model_manifest::ManifestPresetProfile,
) -> Vec<PlanSpec> {
    profile
        .candidates
        .iter()
        .map(|candidate| PlanSpec {
            quant: match candidate.quantization {
                model_manifest::ManifestQuantization::Q4k => Q4K,
                model_manifest::ManifestQuantization::Q5k => Q5K,
                model_manifest::ManifestQuantization::Q6k => Q6K,
                model_manifest::ManifestQuantization::NativeMxfp4 => MXFP4_NATIVE,
            },
            backend: match candidate.backend {
                model_manifest::ManifestBackendMode::DeviceLayers => BackendMode::DeviceLayers,
                model_manifest::ManifestBackendMode::Nccl => BackendMode::Nccl,
            },
            context_target: candidate.context_target_cap,
            context_fraction_milli: candidate.context_fraction_milli,
            min_context_required: candidate.min_context_required,
            per_gpu_headroom_mb: candidate.per_gpu_headroom_mb,
            max_batch_size: candidate.max_batch_size,
            max_seqs: candidate.max_seqs,
        })
        .collect()
}

fn plan_nemotron_cascade() -> ModelHarness {
    plan_nemotron_cascade_from_manifest(&default_nemotron_manifest())
}

fn plan_nemotron_cascade_from_manifest(
    manifest: &model_manifest::RuntimeModelManifest,
) -> ModelHarness {
    harness_from_manifest(manifest)
}

fn harness_from_manifest(manifest: &model_manifest::RuntimeModelManifest) -> ModelHarness {
    ModelHarness {
        model: Box::leak(manifest.model.clone().into_boxed_str()),
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: manifest.sizing.non_repeating_weight_mb_q4,
            repeating_layer_weight_mb_q4: manifest.sizing.repeating_layer_weight_mb_q4,
            load_peak_slack_mb_q4: manifest.sizing.load_peak_slack_mb_q4,
            kv_mb_per_1k_tokens_q4: manifest.sizing.kv_mb_per_1k_tokens_q4,
            base_toks_per_sec_q4: manifest.sizing.base_toks_per_sec_q4,
            repeating_layers: manifest.sizing.repeating_layers,
            context_cap: manifest.sizing.context_cap,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: Box::leak(
                manifest
                    .runtime_defaults
                    .paged_attn
                    .clone()
                    .into_boxed_str(),
            ),
            pa_cache_type: manifest
                .runtime_defaults
                .pa_cache_type
                .clone()
                .map(|value| Box::leak(value.into_boxed_str()) as &'static str),
            pa_memory_fraction: manifest
                .runtime_defaults
                .pa_memory_fraction
                .clone()
                .map(|value| Box::leak(value.into_boxed_str()) as &'static str),
            force_no_mmap: manifest.runtime_defaults.force_no_mmap,
            force_language_model_only: manifest.runtime_defaults.force_language_model_only,
            disable_flash_attn: manifest.runtime_defaults.disable_flash_attn,
            isq_singlethread: manifest.runtime_defaults.isq_singlethread,
            isq_cpu_threads: manifest.runtime_defaults.isq_cpu_threads,
        },
    }
}

fn plan_glm47_flash() -> ModelHarness {
    ModelHarness {
        model: "zai-org/GLM-4.7-Flash",
        sizing: EmpiricalSizingProfile {
            non_repeating_weight_mb_q4: 3_190,
            repeating_layer_weight_mb_q4: 400,
            load_peak_slack_mb_q4: 1_800,
            kv_mb_per_1k_tokens_q4: 275,
            base_toks_per_sec_q4: 48.0,
            repeating_layers: 47,
            context_cap: 65_536,
        },
        runtime: ModelRuntimeHarness {
            paged_attn: "auto",
            pa_cache_type: Some("turboquant3"),
            pa_memory_fraction: Some("0.65"),
            force_no_mmap: true,
            force_language_model_only: false,
            disable_flash_attn: true,
            isq_singlethread: true,
            isq_cpu_threads: None,
        },
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
    fn aux_reserves_are_manifest_driven() {
        let env_map = BTreeMap::new();
        let reserves = compute_aux_reserves_mb(
            None,
            &hardware(3, 24_576),
            &env_map,
            ChatPreset::Quality,
            &[0, 1, 2],
        );
        assert_eq!(reserves.get(&0).copied(), Some(6_700));
        assert_eq!(reserves.get(&1).copied().unwrap_or(0), 0);
        assert_eq!(reserves.get(&2).copied().unwrap_or(0), 0);
    }

    #[test]
    fn aux_default_distribution_prefers_primary_gpu() {
        let env_map: BTreeMap<String, String> = BTreeMap::new();
        let selection = engine::auxiliary_model_selection(
            engine::AuxiliaryRole::Stt,
            env_map.get("CTOX_STT_MODEL").map(String::as_str),
        );
        let distribution = default_aux_distribution(
            None,
            &selection,
            &hardware(4, 24_576),
            ChatPreset::Performance,
            &[1, 2, 3],
        );
        assert_eq!(distribution, vec![0]);
    }

    #[test]
    fn performance_on_three_gpus_keeps_primary_gpu_in_the_nccl_set() {
        let env_map = BTreeMap::new();
        let plan =
            build_gpt_oss_20b_bundle(ChatPreset::Performance, &hardware(3, 24_576), &env_map)
                .selected_plan;
        assert_eq!(plan.cuda_visible_devices, "0,1,2");
        assert_eq!(plan.tensor_parallel_backend.as_deref(), Some("nccl"));
        assert_eq!(plan.mn_local_world_size, Some(3));
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
        assert!(matches!(plan.quantization.as_str(), "Q6K" | "Q5K" | "Q4K"));
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
        assert!(plan.device_layers.is_some(), "{plan:?}");
        let device_layers = plan.device_layers.as_deref().unwrap();
        assert_eq!(sum_device_layers(device_layers), 40);
        assert_eq!(plan.cuda_visible_devices, "0,1,2");
        assert!(device_layers.starts_with("0:"));
        assert!(plan.force_no_mmap);
        assert_eq!(plan.topology, None);
        assert!(!plan.allow_device_layers_with_topology);
        assert!(plan.force_language_model_only);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
    }

    #[test]
    fn qwen35_moe_keeps_dynamic_multi_gpu_planning_on_four_gpu_hosts() {
        let env_map = BTreeMap::new();
        let plan = build_qwen35_35b_a3b_bundle(ChatPreset::Quality, &hardware(4, 24_576), &env_map)
            .selected_plan;
        assert!(matches!(plan.quantization.as_str(), "Q6K" | "Q5K" | "Q4K"));
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
        assert_eq!(plan.cuda_visible_devices, "0,1,2,3");
        assert!(plan.device_layers.is_some(), "{plan:?}");
        assert_eq!(
            sum_device_layers(plan.device_layers.as_deref().unwrap()),
            40
        );
        assert!(plan.device_layers.as_deref().unwrap().starts_with("0:"));
        assert!(plan.force_no_mmap);
        assert_eq!(plan.topology, None);
        assert!(!plan.allow_device_layers_with_topology);
        assert_eq!(plan.nm_device_ordinal, Some(0));
        assert_eq!(plan.base_device_ordinal, Some(0));
        assert_eq!(plan.moe_experts_backend, None);
        assert!(plan.force_language_model_only);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
    }

    #[test]
    fn nemotron_uses_conservative_text_runtime_constraints() {
        let env_map = BTreeMap::new();
        let plan =
            build_nemotron_cascade_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
                .selected_plan;
        assert!(matches!(plan.quantization.as_str(), "Q6K" | "Q5K" | "Q4K"));
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
        assert!(!plan.force_no_mmap);
        assert!(!plan.force_language_model_only);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
        assert!(plan.device_layers.is_some(), "{plan:?}");
        assert_eq!(
            sum_device_layers(plan.device_layers.as_deref().unwrap()),
            52
        );
    }

    #[test]
    fn glm_keeps_public_runtime_constraints() {
        let env_map = BTreeMap::new();
        let plan = build_glm47_flash_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert!(matches!(plan.quantization.as_str(), "Q6K" | "Q5K" | "Q4K"));
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
        assert_eq!(plan.cuda_visible_devices, "0,1,2");
        assert_eq!(
            sum_device_layers(plan.device_layers.as_deref().unwrap()),
            47
        );
        assert!(plan.device_layers.as_deref().unwrap().starts_with("0:"));
        assert!(plan.force_no_mmap);
        assert!(plan.disable_flash_attn);
        assert!(plan.isq_singlethread);
        assert_eq!(plan.isq_cpu_threads, None);
    }

    #[test]
    fn checked_in_runtime_manifests_match_generated_runtime_defaults() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cases = [
            (
                "contracts/models/runtime_manifests/gpt_oss_20b.json",
                default_gpt_oss_manifest(),
            ),
            (
                "contracts/models/runtime_manifests/qwen3_5_4b.json",
                default_qwen35_4b_manifest(),
            ),
            (
                "contracts/models/runtime_manifests/qwen3_5_9b.json",
                default_qwen35_9b_manifest(),
            ),
            (
                "contracts/models/runtime_manifests/qwen3_5_27b.json",
                default_qwen35_27b_manifest(),
            ),
            (
                "contracts/models/runtime_manifests/qwen3_5_35b_a3b.json",
                default_qwen35_35b_a3b_manifest(),
            ),
            (
                "contracts/models/runtime_manifests/glm_4_7_flash.json",
                default_glm47_flash_manifest(),
            ),
        ];
        for (rel_path, generated) in cases {
            let raw = std::fs::read(root.join(rel_path)).unwrap_or_else(|err| {
                panic!("failed to read {rel_path}: {err}");
            });
            let checked_in: model_manifest::RuntimeModelManifest = serde_json::from_slice(&raw)
                .unwrap_or_else(|err| {
                    panic!("failed to parse {rel_path}: {err}");
                });
            assert_eq!(
                checked_in.runtime_defaults.paged_attn, generated.runtime_defaults.paged_attn,
                "paged_attn drifted in {rel_path}"
            );
            assert_eq!(
                checked_in.runtime_defaults.pa_cache_type, generated.runtime_defaults.pa_cache_type,
                "pa_cache_type drifted in {rel_path}"
            );
            assert_eq!(
                checked_in.runtime_defaults.pa_memory_fraction,
                generated.runtime_defaults.pa_memory_fraction,
                "pa_memory_fraction drifted in {rel_path}"
            );
        }
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
        let quant_rank = |value: &str| match value {
            "Q6K" => 3,
            "Q5K" => 2,
            "Q4K" => 1,
            _ => 0,
        };
        assert!(max_context.max_seq_len >= quality.max_seq_len);
        assert!(performance.max_seq_len <= quality.max_seq_len);
        assert!(performance.max_seqs > quality.max_seqs);
        assert!(quant_rank(&quality.quantization) >= quant_rank(&max_context.quantization));
        assert_eq!(performance.quantization, "Q4K");
        assert_eq!(
            sum_device_layers(quality.device_layers.as_deref().unwrap()),
            32
        );
    }

    #[test]
    fn nemotron_presets_prioritize_quantization_context_and_parallelism() {
        let env_map = BTreeMap::new();
        let bundle =
            build_nemotron_cascade_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map);
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
        let quant_rank = |value: &str| match value {
            "Q6K" => 3,
            "Q5K" => 2,
            "Q4K" => 1,
            _ => 0,
        };
        assert!(quant_rank(&quality.quantization) >= quant_rank(&max_context.quantization));
        assert!(max_context.max_seq_len >= quality.max_seq_len);
        assert!(performance.max_seqs >= 2);
        assert_eq!(performance.quantization, "Q4K");
    }

    #[test]
    fn compaction_policy_differs_by_preset() {
        let quality = compaction_policy_for_preset(ChatPreset::Quality);
        let max_context = compaction_policy_for_preset(ChatPreset::MaxContext);
        let performance = compaction_policy_for_preset(ChatPreset::Performance);

        assert_eq!(quality.threshold_percent, 75);
        assert_eq!(quality.min_tokens, 12_288);
        assert_eq!(max_context.threshold_percent, 85);
        assert_eq!(max_context.min_tokens, 16_384);
        assert_eq!(performance.threshold_percent, 70);
        assert_eq!(performance.min_tokens, 8_192);
    }

    #[test]
    fn pa_cache_type_override_flows_into_runtime_plan() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_ENGINE_PA_CACHE_TYPE_OVERRIDE".to_string(),
            "turboquant3".to_string(),
        );
        let plan = build_qwen35_27b_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
    }

    #[test]
    fn gpt_oss_turboquant_override_falls_back_to_fp8_runtime_plan() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_ENGINE_PA_CACHE_TYPE_OVERRIDE".to_string(),
            "turboquant3".to_string(),
        );
        let plan = build_gpt_oss_20b_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
    }

    #[test]
    fn glm_turboquant_override_flows_into_runtime_plan() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_ENGINE_PA_CACHE_TYPE_OVERRIDE".to_string(),
            "turboquant3".to_string(),
        );
        let plan = build_glm47_flash_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
            .selected_plan;
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
    }

    #[test]
    fn nemotron_turboquant_runtime_plan_is_enabled_by_default() {
        let env_map = BTreeMap::new();
        let plan =
            build_nemotron_cascade_bundle(ChatPreset::Quality, &hardware(3, 24_576), &env_map)
                .selected_plan;
        assert_eq!(plan.paged_attn, "auto");
        assert_eq!(plan.pa_cache_type.as_deref(), Some("turboquant3"));
    }

    #[test]
    fn nemotron_max_context_scales_beyond_old_static_cap() {
        let env_map = BTreeMap::new();
        let plan =
            build_nemotron_cascade_bundle(ChatPreset::MaxContext, &hardware(3, 24_576), &env_map)
                .selected_plan;
        assert!(plan.max_seq_len > 32_768);
    }

    #[test]
    fn nemotron_live_resource_snapshot_can_reduce_context() {
        let env_map = BTreeMap::new();
        let manifest = default_nemotron_manifest();
        let harness = plan_nemotron_cascade_from_manifest(&manifest);
        let spec = plan_specs_from_manifest_profile(&manifest.max_context)[0];
        let unconstrained_plan = build_candidate(
            None,
            harness,
            ChatPreset::MaxContext,
            spec,
            &hardware(3, 24_576),
            &env_map,
            None,
        )
        .unwrap();
        let snapshot = resource_state::ResourceSnapshot {
            source: "test".to_string(),
            gpus: vec![
                resource_state::GpuLiveState {
                    index: 0,
                    uuid: None,
                    name: "GPU 0".to_string(),
                    total_mb: 24_576,
                    used_mb: 10_576,
                    free_mb: 14_000,
                },
                resource_state::GpuLiveState {
                    index: 1,
                    uuid: None,
                    name: "GPU 1".to_string(),
                    total_mb: 24_576,
                    used_mb: 8_576,
                    free_mb: 16_000,
                },
                resource_state::GpuLiveState {
                    index: 2,
                    uuid: None,
                    name: "GPU 2".to_string(),
                    total_mb: 24_576,
                    used_mb: 8_576,
                    free_mb: 16_000,
                },
            ],
        };
        let plan = build_candidate(
            None,
            harness,
            ChatPreset::MaxContext,
            spec,
            &hardware(3, 24_576),
            &env_map,
            Some(&snapshot),
        )
        .unwrap();
        assert!(plan.max_seq_len < unconstrained_plan.max_seq_len);
    }
}
