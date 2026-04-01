use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObjectiveLabel {
    Quality,
    MaxContext,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestBackendMode {
    DeviceLayers,
    Nccl,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestQuantization {
    Q4k,
    Q5k,
    Q6k,
    NativeMxfp4,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestSizingProfile {
    pub non_repeating_weight_mb_q4: u64,
    pub repeating_layer_weight_mb_q4: u64,
    pub load_peak_slack_mb_q4: u64,
    pub kv_mb_per_1k_tokens_q4: u64,
    pub base_toks_per_sec_q4: f64,
    pub repeating_layers: u32,
    pub context_cap: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestRuntimeDefaults {
    pub paged_attn: String,
    pub pa_cache_type: Option<String>,
    pub pa_memory_fraction: Option<String>,
    pub force_no_mmap: bool,
    pub force_language_model_only: bool,
    pub disable_flash_attn: bool,
    pub isq_singlethread: bool,
    pub isq_cpu_threads: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestPlacementProfile {
    pub primary_gpu_index: usize,
    pub primary_gpu_holds_non_repeating: bool,
    pub primary_gpu_desktop_reserve_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PresetCandidateSpec {
    pub quantization: ManifestQuantization,
    pub backend: ManifestBackendMode,
    pub max_batch_size: u32,
    pub max_seqs: u32,
    pub context_fraction_milli: u32,
    pub context_target_cap: Option<u32>,
    pub min_context_required: u32,
    pub per_gpu_headroom_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestPresetProfile {
    pub objective: RuntimeObjectiveLabel,
    pub candidates: Vec<PresetCandidateSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeModelManifest {
    pub model: String,
    pub sizing: ManifestSizingProfile,
    pub runtime_defaults: ManifestRuntimeDefaults,
    pub placement: ManifestPlacementProfile,
    pub quality: ManifestPresetProfile,
    pub max_context: ManifestPresetProfile,
    pub performance: ManifestPresetProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuxiliaryPlacementProfile {
    pub primary_gpu_index: usize,
    pub use_primary_gpu_by_default: bool,
    pub supports_multi_gpu_expansion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuxiliaryModelManifest {
    pub model: String,
    pub role: String,
    pub gpu_reserve_mb: u64,
    pub placement: AuxiliaryPlacementProfile,
}

pub fn load_runtime_model_manifest(
    root: &Path,
    model: &str,
) -> anyhow::Result<Option<RuntimeModelManifest>> {
    let Some(path) = manifest_path_for_model(root, model) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read runtime model manifest {}", path.display()))?;
    let manifest: RuntimeModelManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse runtime model manifest {}", path.display()))?;
    Ok(Some(manifest))
}

pub fn load_auxiliary_model_manifest(
    root: &Path,
    model: &str,
) -> anyhow::Result<Option<AuxiliaryModelManifest>> {
    let Some(path) = auxiliary_manifest_path_for_model(root, model) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read auxiliary model manifest {}", path.display()))?;
    let manifest: AuxiliaryModelManifest = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse auxiliary model manifest {}",
            path.display()
        )
    })?;
    Ok(Some(manifest))
}

fn manifest_path_for_model(root: &Path, model: &str) -> Option<std::path::PathBuf> {
    let slug = match model.trim() {
        "openai/gpt-oss-20b" => "gpt_oss_20b",
        "Qwen/Qwen3.5-4B" => "qwen3_5_4b",
        "Qwen/Qwen3.5-9B" => "qwen3_5_9b",
        "Qwen/Qwen3.5-27B" => "qwen3_5_27b",
        "Qwen/Qwen3.5-35B-A3B" => "qwen3_5_35b_a3b",
        "nvidia/Nemotron-Cascade-2-30B-A3B" => "nemotron_cascade_2_30b_a3b",
        "zai-org/GLM-4.7-Flash" => "glm_4_7_flash",
        _ => return None,
    };
    Some(
        root.join("contracts")
            .join("models")
            .join("runtime_manifests")
            .join(format!("{slug}.json")),
    )
}

fn auxiliary_manifest_path_for_model(root: &Path, model: &str) -> Option<std::path::PathBuf> {
    let slug = match model.trim() {
        "Qwen/Qwen3-Embedding-0.6B" => "qwen3_embedding_0_6b",
        "mistralai/Voxtral-Mini-4B-Realtime-2602" => "voxtral_mini_4b_realtime_2602",
        "mistralai/Voxtral-4B-TTS-2603" => "voxtral_4b_tts_2603",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" => "qwen3_tts_12hz_0_6b_base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => "qwen3_tts_12hz_0_6b_customvoice",
        _ => return None,
    };
    Some(
        root.join("contracts")
            .join("models")
            .join("aux_runtime_manifests")
            .join(format!("{slug}.json")),
    )
}
