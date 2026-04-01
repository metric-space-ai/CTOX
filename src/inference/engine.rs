use anyhow::Context;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const DEFAULT_CODEX_REPO_REF: &str = "c6ab4ee537e5b118a20e9e0d3e0c0023cae2d982";
const DEFAULT_ENGINE_CANDLE_REF: &str = "master";
const EXPERIMENTAL_TURBOQUANT_ENV: &str = "CTOX_ENGINE_EXPERIMENTAL_TURBOQUANT";
const EXPERIMENTAL_TURBOQUANT_CHAT_ENV: &str = "CTOX_CHAT_EXPERIMENTAL_TURBOQUANT";
const PA_CACHE_TYPE_OVERRIDE_ENV: &str = "CTOX_ENGINE_PA_CACHE_TYPE_OVERRIDE";
const PA_CACHE_TYPE_CHAT_OVERRIDE_ENV: &str = "CTOX_CHAT_PA_CACHE_TYPE_OVERRIDE";
const TURBOQUANT2_CACHE_TYPE: &str = "turboquant2";
const TURBOQUANT3_CACHE_TYPE: &str = "turboquant3";
const TURBOQUANT4_CACHE_TYPE: &str = "turboquant4";

const DISALLOWED_ENGINE_FUNCTION_TOOLS: &[&str] = &["spawn_agent", "send_input"];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalModelFamily {
    GptOss,
    Qwen35Vision,
    NemotronCascade,
    Glm47Flash,
    Qwen3Embedding,
    VoxtralTranscription,
    Qwen3Speech,
    VoxtralSpeech,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryRole {
    Embedding,
    Stt,
    Tts,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComputeTarget {
    Gpu,
    Cpu,
}

impl ComputeTarget {
    pub fn as_env_value(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryBackendKind {
    MistralRs,
    Speaches,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct AuxiliaryModelSelection {
    pub role: AuxiliaryRole,
    pub choice: &'static str,
    pub request_model: &'static str,
    pub backend_kind: AuxiliaryBackendKind,
    pub compute_target: ComputeTarget,
    pub default_port: u16,
}

impl AuxiliaryModelSelection {
    pub fn gpu_reserve_mb(self) -> u64 {
        if self.compute_target == ComputeTarget::Cpu {
            return 0;
        }
        match self.role {
            AuxiliaryRole::Embedding => 1100,
            AuxiliaryRole::Stt => 4200,
            AuxiliaryRole::Tts => 1400,
        }
    }
}

impl FromStr for LocalModelFamily {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpt_oss" | "gpt-oss" | "gptoss" => Ok(Self::GptOss),
            "qwen3_5" | "qwen3.5" | "qwen3-5" | "qwen35" | "qwen3_5_vision" => {
                Ok(Self::Qwen35Vision)
            }
            "nemotron"
            | "nemotron_cascade"
            | "nemotron-cascade"
            | "nemotron_cascade_2"
            | "nemotron-cascade-2"
            | "nemotron-cascade-2-30b-a3b"
            | "nemotroncascade230ba3b" => Ok(Self::NemotronCascade),
            "glm4moelite" | "glm4_flash" | "glm4.7flash" | "glm-4.7-flash" | "gln-4.7-flash"
            | "gln4.7flash" => Ok(Self::Glm47Flash),
            "qwen3_embedding" | "qwen3-embedding" | "qwen3embedding" => Ok(Self::Qwen3Embedding),
            "voxtral" | "voxtral_realtime" | "voxtral-transcription" | "stt" => {
                Ok(Self::VoxtralTranscription)
            }
            "qwen3_tts" | "qwen3-tts" | "qwen3speech" | "tts" => Ok(Self::Qwen3Speech),
            "voxtral_tts" | "voxtral-tts" | "voxtral4btts" | "voxtral_speech" => {
                Ok(Self::VoxtralSpeech)
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
    pub ctox_engine_candle_root: PathBuf,
    pub ctox_engine_binary: PathBuf,
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EngineRuntimeConfig {
    pub family: LocalModelFamily,
    pub model: String,
    pub port: u16,
    pub proxy_port: Option<u16>,
    pub max_seq_len: Option<u32>,
    pub max_seqs: u32,
    pub max_batch_size: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EngineFamilyProfile {
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LocalModelProfile {
    pub runtime: EngineRuntimeConfig,
    pub family_profile: EngineFamilyProfile,
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
    pub runtime: EngineRuntimeConfig,
    pub family_profile: EngineFamilyProfile,
    pub engine_command: Vec<String>,
    pub codex_exec_command: Option<Vec<String>>,
    pub bridge_mode: String,
}

pub const SUPPORTED_CHAT_MODELS: &[&str] = &[
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "nvidia/Nemotron-Cascade-2-30B-A3B",
    "zai-org/GLM-4.7-Flash",
];

pub const SUPPORTED_OPENAI_API_CHAT_MODELS: &[&str] = &["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"];

pub const SUPPORTED_EMBEDDING_MODELS: &[&str] = &[
    "Qwen/Qwen3-Embedding-0.6B [GPU]",
    "Qwen/Qwen3-Embedding-0.6B [CPU]",
];

pub const SUPPORTED_STT_MODELS: &[&str] = &[
    "mistralai/Voxtral-Mini-4B-Realtime-2602 [GPU]",
    "Systran/faster-whisper-small [CPU]",
];

pub const SUPPORTED_TTS_MODELS: &[&str] = &[
    "mistralai/Voxtral-4B-TTS-2603 [GPU]",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base [GPU]",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice [GPU]",
    "speaches-ai/piper-de_DE-thorsten-high [CPU DE]",
    "speaches-ai/piper-fr_FR-siwis-medium [CPU FR]",
    "speaches-ai/piper-en_US-lessac-medium [CPU EN]",
];

pub fn auxiliary_model_selection(
    role: AuxiliaryRole,
    configured_model: Option<&str>,
) -> AuxiliaryModelSelection {
    let value = configured_model.map(str::trim).unwrap_or("");
    match role {
        AuxiliaryRole::Embedding => match value.to_ascii_lowercase().as_str() {
            "qwen/qwen3-embedding-0.6b [cpu]"
            | "qwen/qwen3-embedding-0.6b (cpu)"
            | "qwen3-embedding-0.6b [cpu]"
            | "qwen3-embedding-0.6b (cpu)" => AuxiliaryModelSelection {
                role,
                choice: "Qwen/Qwen3-Embedding-0.6B [CPU]",
                request_model: "Qwen/Qwen3-Embedding-0.6B",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Cpu,
                default_port: 1237,
            },
            _ => AuxiliaryModelSelection {
                role,
                choice: "Qwen/Qwen3-Embedding-0.6B [GPU]",
                request_model: "Qwen/Qwen3-Embedding-0.6B",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1237,
            },
        },
        AuxiliaryRole::Stt => match value.to_ascii_lowercase().as_str() {
            "systran/faster-whisper-small [cpu]"
            | "systran/faster-whisper-small (cpu)"
            | "systran/faster-whisper-small" => AuxiliaryModelSelection {
                role,
                choice: "Systran/faster-whisper-small [CPU]",
                request_model: "Systran/faster-whisper-small",
                backend_kind: AuxiliaryBackendKind::Speaches,
                compute_target: ComputeTarget::Cpu,
                default_port: 1238,
            },
            _ => AuxiliaryModelSelection {
                role,
                choice: "mistralai/Voxtral-Mini-4B-Realtime-2602 [GPU]",
                request_model: "mistralai/Voxtral-Mini-4B-Realtime-2602",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1238,
            },
        },
        AuxiliaryRole::Tts => match value.to_ascii_lowercase().as_str() {
            "mistralai/voxtral-4b-tts-2603 [gpu]"
            | "mistralai/voxtral-4b-tts-2603 (gpu)"
            | "mistralai/voxtral-4b-tts-2603"
            | "voxtral-4b-tts-2603"
            | "voxtral 4b tts 2603" => AuxiliaryModelSelection {
                role,
                choice: "mistralai/Voxtral-4B-TTS-2603 [GPU]",
                request_model: "mistralai/Voxtral-4B-TTS-2603",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1239,
            },
            "qwen/qwen3-tts-12hz-0.6b-base [gpu]"
            | "qwen/qwen3-tts-12hz-0.6b-base (gpu)"
            | "qwen/qwen3-tts-12hz-0.6b-base" => AuxiliaryModelSelection {
                role,
                choice: "Qwen/Qwen3-TTS-12Hz-0.6B-Base [GPU]",
                request_model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1239,
            },
            "qwen/qwen3-tts-12hz-0.6b-customvoice [gpu]"
            | "qwen/qwen3-tts-12hz-0.6b-customvoice (gpu)"
            | "qwen/qwen3-tts-12hz-0.6b-customvoice" => AuxiliaryModelSelection {
                role,
                choice: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice [GPU]",
                request_model: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1239,
            },
            "speaches-ai/piper-de_de-thorsten-high [cpu de]"
            | "speaches-ai/piper-de_de-thorsten-high (cpu de)"
            | "speaches-ai/piper-de_de-thorsten-high" => AuxiliaryModelSelection {
                role,
                choice: "speaches-ai/piper-de_DE-thorsten-high [CPU DE]",
                request_model: "speaches-ai/piper-de_DE-thorsten-high",
                backend_kind: AuxiliaryBackendKind::Speaches,
                compute_target: ComputeTarget::Cpu,
                default_port: 1239,
            },
            "speaches-ai/piper-fr_fr-siwis-medium [cpu fr]"
            | "speaches-ai/piper-fr_fr-siwis-medium (cpu fr)"
            | "speaches-ai/piper-fr_fr-siwis-medium" => AuxiliaryModelSelection {
                role,
                choice: "speaches-ai/piper-fr_FR-siwis-medium [CPU FR]",
                request_model: "speaches-ai/piper-fr_FR-siwis-medium",
                backend_kind: AuxiliaryBackendKind::Speaches,
                compute_target: ComputeTarget::Cpu,
                default_port: 1239,
            },
            "speaches-ai/piper-en_us-lessac-medium [cpu en]"
            | "speaches-ai/piper-en_us-lessac-medium (cpu en)"
            | "speaches-ai/piper-en_us-lessac-medium" => AuxiliaryModelSelection {
                role,
                choice: "speaches-ai/piper-en_US-lessac-medium [CPU EN]",
                request_model: "speaches-ai/piper-en_US-lessac-medium",
                backend_kind: AuxiliaryBackendKind::Speaches,
                compute_target: ComputeTarget::Cpu,
                default_port: 1239,
            },
            _ => AuxiliaryModelSelection {
                role,
                choice: "mistralai/Voxtral-4B-TTS-2603 [GPU]",
                request_model: "mistralai/Voxtral-4B-TTS-2603",
                backend_kind: AuxiliaryBackendKind::MistralRs,
                compute_target: ComputeTarget::Gpu,
                default_port: 1239,
            },
        },
    }
}

pub fn supported_local_model_profiles() -> Vec<LocalModelProfile> {
    vec![
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::GptOss,
                model: "openai/gpt-oss-20b".to_string(),
                port: 1234,
                proxy_port: Some(12434),
                max_seq_len: Some(131_072),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::GptOss,
                launcher_mode: "text".to_string(),
                arch: Some("gpt_oss".to_string()),
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.80".to_string()),
                pa_context_len: None,
                max_seq_len: 131_072,
                max_batch_size: 1,
                max_seqs: 1,
                isq: None,
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen35Vision,
                model: "Qwen/Qwen3.5-4B".to_string(),
                port: 1235,
                proxy_port: Some(12434),
                max_seq_len: Some(262_144),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen35Vision,
                launcher_mode: "vision".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.80".to_string()),
                pa_context_len: None,
                max_seq_len: 262_144,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen35Vision,
                model: "Qwen/Qwen3.5-9B".to_string(),
                port: 1235,
                proxy_port: Some(12434),
                max_seq_len: Some(262_144),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen35Vision,
                launcher_mode: "vision".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.80".to_string()),
                pa_context_len: None,
                max_seq_len: 262_144,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen35Vision,
                model: "Qwen/Qwen3.5-27B".to_string(),
                port: 1235,
                proxy_port: Some(12434),
                max_seq_len: Some(262_144),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen35Vision,
                launcher_mode: "vision".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.80".to_string()),
                pa_context_len: None,
                max_seq_len: 262_144,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(3),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen35Vision,
                model: "Qwen/Qwen3.5-35B-A3B".to_string(),
                port: 1235,
                proxy_port: Some(12434),
                max_seq_len: Some(262_144),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen35Vision,
                launcher_mode: "vision".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.80".to_string()),
                pa_context_len: None,
                max_seq_len: 262_144,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(3),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::NemotronCascade,
                model: "nvidia/Nemotron-Cascade-2-30B-A3B".to_string(),
                port: 1236,
                proxy_port: Some(12434),
                max_seq_len: Some(8_192),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::NemotronCascade,
                launcher_mode: "text".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.45".to_string()),
                pa_context_len: None,
                max_seq_len: 8_192,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(2),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Glm47Flash,
                model: "zai-org/GLM-4.7-Flash".to_string(),
                port: 1236,
                proxy_port: Some(12434),
                max_seq_len: Some(2_048),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Glm47Flash,
                launcher_mode: "text".to_string(),
                arch: Some("glm4moelite".to_string()),
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("turboquant3".to_string()),
                pa_memory_fraction: Some("0.45".to_string()),
                pa_context_len: None,
                max_seq_len: 2_048,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(3),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen3Embedding,
                model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                port: 1237,
                proxy_port: None,
                max_seq_len: Some(32_768),
                max_seqs: 8,
                max_batch_size: 8,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen3Embedding,
                launcher_mode: "embedding".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("f8e4m3".to_string()),
                pa_memory_fraction: Some("0.30".to_string()),
                pa_context_len: None,
                max_seq_len: 32_768,
                max_batch_size: 8,
                max_seqs: 8,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::VoxtralTranscription,
                model: "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string(),
                port: 1238,
                proxy_port: None,
                max_seq_len: Some(32_768),
                max_seqs: 2,
                max_batch_size: 2,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::VoxtralTranscription,
                launcher_mode: "vision".to_string(),
                arch: None,
                paged_attn: "auto".to_string(),
                pa_cache_type: Some("f8e4m3".to_string()),
                pa_memory_fraction: Some("0.55".to_string()),
                pa_context_len: None,
                max_seq_len: 32_768,
                max_batch_size: 2,
                max_seqs: 2,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::Qwen3Speech,
                model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
                port: 1239,
                proxy_port: None,
                max_seq_len: None,
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::Qwen3Speech,
                launcher_mode: "speech".to_string(),
                arch: None,
                paged_attn: "off".to_string(),
                pa_cache_type: None,
                pa_memory_fraction: None,
                pa_context_len: None,
                max_seq_len: 4_096,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
        LocalModelProfile {
            runtime: EngineRuntimeConfig {
                family: LocalModelFamily::VoxtralSpeech,
                model: "mistralai/Voxtral-4B-TTS-2603".to_string(),
                port: 1239,
                proxy_port: None,
                max_seq_len: Some(8_192),
                max_seqs: 1,
                max_batch_size: 1,
            },
            family_profile: EngineFamilyProfile {
                family: LocalModelFamily::VoxtralSpeech,
                launcher_mode: "speech".to_string(),
                arch: Some("voxtral_tts".to_string()),
                paged_attn: "off".to_string(),
                pa_cache_type: None,
                pa_memory_fraction: None,
                pa_context_len: None,
                max_seq_len: 8_192,
                max_batch_size: 1,
                max_seqs: 1,
                isq: Some("Q4K".to_string()),
                tensor_parallel_backend: None,
                disable_nccl: true,
                target_world_size: None,
                preferred_gpu_count: Some(1),
            },
        },
    ]
}

pub fn model_profile_for_model(model: &str) -> anyhow::Result<LocalModelProfile> {
    let normalized = normalize_supported_model(model);
    supported_local_model_profiles()
        .into_iter()
        .find(|profile| profile.runtime.model == normalized)
        .ok_or_else(|| anyhow::anyhow!("unsupported local model profile: {normalized}"))
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
    let ctox_engine_candle_root = root.join("engine/candle");
    VendoredDependencyPaths {
        references_root,
        codex_exec_binary: codex_rs_root.join("target/release/codex-exec"),
        codex_repo_root,
        codex_rs_root,
        ctox_engine_binary: ctox_engine_candle_root.join("target/release/ctox-engine"),
        ctox_engine_candle_root,
    }
}

pub fn default_runtime_config(family: LocalModelFamily) -> EngineRuntimeConfig {
    match family {
        LocalModelFamily::GptOss => EngineRuntimeConfig {
            family,
            model: "openai/gpt-oss-20b".to_string(),
            port: 1234,
            proxy_port: Some(12434),
            max_seq_len: Some(131_072),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::Qwen35Vision => EngineRuntimeConfig {
            family,
            model: "Qwen/Qwen3.5-27B".to_string(),
            port: 1235,
            proxy_port: None,
            max_seq_len: Some(262_144),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::NemotronCascade => EngineRuntimeConfig {
            family,
            model: "nvidia/Nemotron-Cascade-2-30B-A3B".to_string(),
            port: 1236,
            proxy_port: None,
            max_seq_len: Some(8_192),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::Glm47Flash => EngineRuntimeConfig {
            family,
            model: "zai-org/GLM-4.7-Flash".to_string(),
            port: 1236,
            proxy_port: None,
            max_seq_len: Some(65_536),
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::Qwen3Embedding => EngineRuntimeConfig {
            family,
            model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            port: 1237,
            proxy_port: None,
            max_seq_len: Some(32_768),
            max_seqs: 8,
            max_batch_size: 8,
        },
        LocalModelFamily::VoxtralTranscription => EngineRuntimeConfig {
            family,
            model: "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string(),
            port: 1238,
            proxy_port: None,
            max_seq_len: Some(32_768),
            max_seqs: 2,
            max_batch_size: 2,
        },
        LocalModelFamily::Qwen3Speech => EngineRuntimeConfig {
            family,
            model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            port: 1239,
            proxy_port: None,
            max_seq_len: None,
            max_seqs: 1,
            max_batch_size: 1,
        },
        LocalModelFamily::VoxtralSpeech => EngineRuntimeConfig {
            family,
            model: "mistralai/Voxtral-4B-TTS-2603".to_string(),
            port: 1239,
            proxy_port: None,
            max_seq_len: Some(8_192),
            max_seqs: 1,
            max_batch_size: 1,
        },
    }
}

pub fn runtime_config_for_model(model: &str) -> anyhow::Result<EngineRuntimeConfig> {
    Ok(model_profile_for_model(model)?.runtime)
}

pub fn is_openai_api_chat_model(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    SUPPORTED_OPENAI_API_CHAT_MODELS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&normalized))
}

pub fn uses_ctox_proxy_model(model: &str) -> bool {
    runtime_config_for_model(model).is_ok()
}

fn normalize_supported_model(model: &str) -> &str {
    let trimmed = model.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "glm-4.7-flash"
        | "glm 4.7 flash"
        | "gln-4.7-flash"
        | "gln 4.7 flash"
        | "zai/glm-4.7b-flash"
        | "zai-org/glm-4.7b-flash"
        | "zai/glm-4.7-flash"
        | "zai-org/glm-4.7-flash" => "zai-org/GLM-4.7-Flash",
        "nvidia/nemotron-cascade-2-30b-a3b"
        | "nemotron-cascade-2-30b-a3b"
        | "nemotron cascade 2 30b a3b"
        | "nemotron cascade 2" => "nvidia/Nemotron-Cascade-2-30B-A3B",
        "qwen/qwen3-embedding-0.6b" | "qwen3-embedding-0.6b" | "qwen3 embedding 0.6b" => {
            "Qwen/Qwen3-Embedding-0.6B"
        }
        "mistralai/voxtral-mini-4b-realtime-2602" | "voxtral-mini-4b-realtime-2602" => {
            "mistralai/Voxtral-Mini-4B-Realtime-2602"
        }
        "mistralai/voxtral-4b-tts-2603" | "voxtral-4b-tts-2603" | "voxtral 4b tts 2603" => {
            "mistralai/Voxtral-4B-TTS-2603"
        }
        "qwen/qwen3-tts-12hz-0.6b-base"
        | "qwen3-tts-12hz-0.6b-base"
        | "qwen3 tts 0.6b base"
        | "qwen3-tts 0.6b base" => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "qwen/qwen3-tts-12hz-0.6b-customvoice"
        | "qwen3-tts-12hz-0.6b-customvoice"
        | "qwen3 tts 0.6b customvoice"
        | "qwen3-tts 0.6b customvoice" => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        _ => trimmed,
    }
}

pub fn default_family_profile(family: LocalModelFamily) -> EngineFamilyProfile {
    match family {
        LocalModelFamily::GptOss => EngineFamilyProfile {
            family,
            launcher_mode: "text".to_string(),
            arch: Some("gpt_oss".to_string()),
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("turboquant3".to_string()),
            pa_memory_fraction: Some("0.80".to_string()),
            pa_context_len: None,
            max_seq_len: 131_072,
            max_batch_size: 1,
            max_seqs: 1,
            isq: None,
            tensor_parallel_backend: Some("nccl".to_string()),
            disable_nccl: false,
            target_world_size: Some(2),
            preferred_gpu_count: Some(2),
        },
        LocalModelFamily::Qwen35Vision => EngineFamilyProfile {
            family,
            launcher_mode: "vision".to_string(),
            arch: None,
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("turboquant3".to_string()),
            pa_memory_fraction: Some("0.80".to_string()),
            pa_context_len: None,
            max_seq_len: 32_768,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(3),
        },
        LocalModelFamily::NemotronCascade => EngineFamilyProfile {
            family,
            launcher_mode: "text".to_string(),
            arch: None,
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("turboquant3".to_string()),
            pa_memory_fraction: Some("0.45".to_string()),
            pa_context_len: None,
            max_seq_len: 8_192,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(2),
        },
        LocalModelFamily::Glm47Flash => EngineFamilyProfile {
            family,
            launcher_mode: "text".to_string(),
            arch: Some("glm4moelite".to_string()),
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("turboquant3".to_string()),
            pa_memory_fraction: Some("0.65".to_string()),
            pa_context_len: None,
            max_seq_len: 4_096,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(3),
        },
        LocalModelFamily::Qwen3Embedding => EngineFamilyProfile {
            family,
            launcher_mode: "embedding".to_string(),
            arch: None,
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("f8e4m3".to_string()),
            pa_memory_fraction: Some("0.30".to_string()),
            pa_context_len: None,
            max_seq_len: 32_768,
            max_batch_size: 8,
            max_seqs: 8,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(1),
        },
        LocalModelFamily::VoxtralTranscription => EngineFamilyProfile {
            family,
            launcher_mode: "vision".to_string(),
            arch: None,
            paged_attn: "auto".to_string(),
            pa_cache_type: Some("f8e4m3".to_string()),
            pa_memory_fraction: Some("0.55".to_string()),
            pa_context_len: None,
            max_seq_len: 32_768,
            max_batch_size: 2,
            max_seqs: 2,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(1),
        },
        LocalModelFamily::Qwen3Speech => EngineFamilyProfile {
            family,
            launcher_mode: "speech".to_string(),
            arch: None,
            paged_attn: "off".to_string(),
            pa_cache_type: None,
            pa_memory_fraction: None,
            pa_context_len: None,
            max_seq_len: 4_096,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(1),
        },
        LocalModelFamily::VoxtralSpeech => EngineFamilyProfile {
            family,
            launcher_mode: "speech".to_string(),
            arch: Some("voxtral_tts".to_string()),
            paged_attn: "off".to_string(),
            pa_cache_type: None,
            pa_memory_fraction: None,
            pa_context_len: None,
            max_seq_len: 8_192,
            max_batch_size: 1,
            max_seqs: 1,
            isq: Some("Q4K".to_string()),
            tensor_parallel_backend: None,
            disable_nccl: true,
            target_world_size: None,
            preferred_gpu_count: Some(1),
        },
    }
}

pub fn runtime_profile_for_model(model: &str) -> anyhow::Result<EngineFamilyProfile> {
    Ok(model_profile_for_model(model)?.family_profile)
}

pub fn resolve_pa_cache_type(
    default: Option<&str>,
    env_map: &BTreeMap<String, String>,
) -> Option<String> {
    let explicit_override = env_map
        .get(PA_CACHE_TYPE_OVERRIDE_ENV)
        .or_else(|| env_map.get(PA_CACHE_TYPE_CHAT_OVERRIDE_ENV))
        .and_then(|value| normalize_pa_cache_type_override(value));
    if let Some(override_value) = explicit_override {
        return override_value;
    }
    let turboquant_requested = env_map
        .get(EXPERIMENTAL_TURBOQUANT_ENV)
        .or_else(|| env_map.get(EXPERIMENTAL_TURBOQUANT_CHAT_ENV))
        .is_some_and(|value| env_truthy(value));
    if turboquant_requested && matches!(default, Some("f8e4m3")) {
        return Some(TURBOQUANT3_CACHE_TYPE.to_string());
    }
    default.map(str::to_string)
}

pub fn resolve_model_pa_cache_type(
    model: &str,
    default: Option<&str>,
    env_map: &BTreeMap<String, String>,
) -> Option<String> {
    let resolved = resolve_pa_cache_type(default, env_map);
    match resolved {
        Some(cache_type) if !model_supports_pa_cache_type(model, &cache_type) => default
            .filter(|cache_type| model_supports_pa_cache_type(model, cache_type))
            .map(str::to_string),
        other => other,
    }
}

pub fn resolve_model_paged_attn(model: &str, default: &str, cache_type: Option<&str>) -> String {
    if default != "off" {
        return default.to_string();
    }
    if cache_type.is_some() && model_supports_paged_attention_cache(model) {
        return "auto".to_string();
    }
    default.to_string()
}

pub fn model_supports_paged_attention_cache(model: &str) -> bool {
    model_profile_for_model(model)
        .map(|profile| {
            matches!(
                profile.runtime.family,
                LocalModelFamily::GptOss
                    | LocalModelFamily::Qwen35Vision
                    | LocalModelFamily::NemotronCascade
                    | LocalModelFamily::Glm47Flash
            )
        })
        .unwrap_or(false)
}

pub fn model_supports_pa_cache_type(model: &str, cache_type: &str) -> bool {
    let normalized = cache_type.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "auto" | "f8e4m3" => model_supports_paged_attention_cache(model),
        TURBOQUANT2_CACHE_TYPE | TURBOQUANT4_CACHE_TYPE => false,
        TURBOQUANT3_CACHE_TYPE => model_profile_for_model(model)
            .map(|profile| {
                matches!(
                    profile.runtime.family,
                    LocalModelFamily::GptOss
                        | LocalModelFamily::Qwen35Vision
                        | LocalModelFamily::NemotronCascade
                        | LocalModelFamily::Glm47Flash
                )
            })
            .unwrap_or(false),
        _ => false,
    }
}

fn normalize_pa_cache_type_override(value: &str) -> Option<Option<String>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "default" => None,
        "none" | "off" | "disable" | "disabled" => Some(None),
        "auto"
        | "f8e4m3"
        | TURBOQUANT2_CACHE_TYPE
        | TURBOQUANT3_CACHE_TYPE
        | TURBOQUANT4_CACHE_TYPE => Some(Some(value.trim().to_ascii_lowercase())),
        _ => None,
    }
}

fn env_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn build_engine_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &EngineRuntimeConfig,
) -> Vec<String> {
    let family_profile = runtime_profile_for_model(&runtime.model)
        .unwrap_or_else(|_| default_family_profile(runtime.family));
    let mut command = vec![dependencies.ctox_engine_binary.display().to_string()];
    match runtime.family {
        LocalModelFamily::GptOss
        | LocalModelFamily::NemotronCascade
        | LocalModelFamily::Glm47Flash => {
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
            ]);
            if let Some(arch) = family_profile.arch {
                command.extend(["-a".to_string(), arch]);
            }
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
        LocalModelFamily::Qwen3Embedding => {
            command.extend([
                "serve".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
                "embedding".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
        }
        LocalModelFamily::VoxtralTranscription => {
            command.extend([
                "serve".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
                "vision".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
        }
        LocalModelFamily::Qwen3Speech => {
            command.extend([
                "serve".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
            ]);
            if let Some(isq) = family_profile.isq.clone() {
                command.extend(["--isq".to_string(), isq]);
            }
            command.extend([
                "speech".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
        }
        LocalModelFamily::VoxtralSpeech => {
            command.extend([
                "serve".to_string(),
                "-p".to_string(),
                runtime.port.to_string(),
            ]);
            if let Some(isq) = family_profile.isq.clone() {
                command.extend(["--isq".to_string(), isq]);
            }
            command.extend([
                "speech".to_string(),
                "-m".to_string(),
                runtime.model.clone(),
            ]);
            if let Some(arch) = family_profile.arch.clone() {
                command.extend(["-a".to_string(), arch]);
            }
        }
    }
    command
}

pub fn build_codex_exec_command(
    dependencies: &VendoredDependencyPaths,
    runtime: &EngineRuntimeConfig,
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
        "web_search=\"live\"".to_string(),
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
    let engine_command = build_engine_command(&dependencies, &runtime);
    let codex_exec_command =
        build_codex_exec_command(&dependencies, &runtime, &CodexExecInvocation { prompt });
    let bridge_mode = match family {
        LocalModelFamily::GptOss => "codex_responses_proxy".to_string(),
        LocalModelFamily::Qwen35Vision => "qwen_custom_execution".to_string(),
        LocalModelFamily::NemotronCascade => "chatml_custom_execution".to_string(),
        LocalModelFamily::Glm47Flash => "codex_responses_proxy".to_string(),
        LocalModelFamily::Qwen3Embedding => "embedding_server".to_string(),
        LocalModelFamily::VoxtralTranscription => "transcription_server".to_string(),
        LocalModelFamily::Qwen3Speech => "speech_server".to_string(),
        LocalModelFamily::VoxtralSpeech => "speech_server".to_string(),
    };
    CleanRoomBaselinePlan {
        dependencies,
        runtime,
        family_profile,
        engine_command,
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

pub fn bootstrap_clean_room_dependencies(
    root: &Path,
) -> anyhow::Result<DependencyBootstrapOutcome> {
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
        "ctox-engine" => DEFAULT_ENGINE_CANDLE_REF,
        _ => DEFAULT_CODEX_REPO_REF,
    }
}

fn ensure_git_checkout(repo_url: &str, git_ref: &str, target_dir: &Path) -> anyhow::Result<String> {
    if target_dir.join(".git").exists() {
        ensure_git_worktree_clean(target_dir)?;
        run_git([
            "-C",
            path_arg(target_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            git_ref,
        ])?;
        run_git([
            "-C",
            path_arg(target_dir),
            "checkout",
            "--detach",
            "FETCH_HEAD",
        ])?;
        return Ok("updated".to_string());
    }

    if target_dir.exists() {
        anyhow::bail!(
            "refusing to overwrite non-git clean-room dependency path: {}",
            target_dir.display()
        );
    }

    if let Some(parent) = target_dir.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create dependency parent dir {}",
                parent.display()
            )
        })?;
    }

    run_git(["clone", "--no-checkout", repo_url, path_arg(target_dir)])?;
    run_git([
        "-C",
        path_arg(target_dir),
        "fetch",
        "--depth",
        "1",
        "origin",
        git_ref,
    ])?;
    run_git([
        "-C",
        path_arg(target_dir),
        "checkout",
        "--detach",
        "FETCH_HEAD",
    ])?;
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

pub fn rewrite_engine_responses_request(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;

    if let Some(tools) = payload.get_mut("tools").and_then(Value::as_array_mut) {
        let mut rewritten = Vec::new();
        for tool in tools.drain(..) {
            if let Some(tool) = rewrite_tool(tool) {
                rewritten.push(tool);
            }
        }
        *tools = rewritten;
    }

    if payload.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        payload["parallel_tool_calls"] = Value::Bool(true);
    }
    if let Some(object) = payload.as_object_mut() {
        object.remove("max_tool_calls");
    }

    serde_json::to_vec(&payload).context("failed to encode rewritten responses request")
}

pub fn rewrite_openai_responses_request(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;

    if let Some(tools) = payload.get_mut("tools").and_then(Value::as_array_mut) {
        let mut rewritten = Vec::new();
        for tool in tools.drain(..) {
            rewritten.extend(rewrite_openai_tool(tool));
        }
        *tools = rewritten;
    }

    serde_json::to_vec(&payload).context("failed to encode OpenAI responses request")
}

pub fn is_qwen_chat_model_id(model_id: &str) -> bool {
    let lowered = model_id.trim().to_ascii_lowercase();
    lowered.contains("qwen3.5") || lowered.contains("qwen/qwen3.5")
}

pub fn is_nemotron_chat_model_id(model_id: &str) -> bool {
    let lowered = model_id.trim().to_ascii_lowercase();
    lowered.contains("nemotron-cascade-2") || lowered.contains("nvidia/nemotron-cascade-2")
}

pub fn is_glm_chat_model_id(model_id: &str) -> bool {
    let lowered = model_id.trim().to_ascii_lowercase();
    lowered.contains("glm-4.7") || lowered.contains("glm 4.7") || lowered.contains("glm4.7")
}

pub fn rewrite_responses_to_qwen_chat_completions(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("Qwen/Qwen3.5-4B")
        .to_string();
    let instructions = payload
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let messages = build_qwen_chat_messages(
        &normalize_responses_input(payload.get("input")),
        instructions.as_deref(),
    );
    let mut merged_system_parts = Vec::new();
    let mut merged_messages = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if role == "system" {
            if !content.is_empty() {
                merged_system_parts.push(content);
            }
        } else {
            merged_messages.push(message);
        }
    }
    let mut messages = Vec::new();
    if !merged_system_parts.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": merged_system_parts.join("\n\n"),
        }));
    }
    messages.extend(merged_messages);

    let tools = payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .flat_map(|tool| rewrite_qwen_chat_tool(tool.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), Value::String(model));
    request.insert("messages".to_string(), Value::Array(messages));
    if !tools.is_empty() {
        request.insert("tools".to_string(), Value::Array(tools));
    }
    let enable_thinking = payload
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str)
        .is_some();
    request.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    for key in [
        "tool_choice",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_output_tokens",
    ] {
        if let Some(value) = payload.get(key) {
            let mapped_key = if key == "max_output_tokens" {
                "max_tokens"
            } else {
                key
            };
            request.insert(mapped_key.to_string(), value.clone());
        }
    }
    request.insert("stream".to_string(), Value::Bool(false));
    if payload.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        request.insert("parallel_tool_calls".to_string(), Value::Bool(true));
    } else if let Some(value) = payload.get("parallel_tool_calls") {
        request.insert("parallel_tool_calls".to_string(), value.clone());
    }

    serde_json::to_vec(&Value::Object(request))
        .context("failed to encode Qwen chat-completions payload")
}

pub fn rewrite_responses_to_nemotron_chat_completions(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("nvidia/Nemotron-Cascade-2-30B-A3B")
        .to_string();
    let instructions = payload
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let messages = build_qwen_chat_messages(
        &normalize_responses_input(payload.get("input")),
        instructions.as_deref(),
    );
    let mut merged_system_parts = Vec::new();
    let mut merged_messages = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if role == "system" {
            if !content.is_empty() {
                merged_system_parts.push(content);
            }
        } else {
            merged_messages.push(message);
        }
    }
    let mut messages = Vec::new();
    if !merged_system_parts.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": merged_system_parts.join("\n\n"),
        }));
    }
    messages.extend(merged_messages);

    let tools = payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .flat_map(|tool| rewrite_qwen_chat_tool(tool.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), Value::String(model));
    request.insert("messages".to_string(), Value::Array(messages));
    if !tools.is_empty() {
        request.insert("tools".to_string(), Value::Array(tools));
    }
    let enable_thinking = payload
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str)
        .is_some();
    request.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    for key in [
        "tool_choice",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_output_tokens",
    ] {
        if let Some(value) = payload.get(key) {
            let mapped_key = if key == "max_output_tokens" {
                "max_tokens"
            } else {
                key
            };
            request.insert(mapped_key.to_string(), value.clone());
        }
    }
    request.insert("stream".to_string(), Value::Bool(false));
    if payload.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        request.insert("parallel_tool_calls".to_string(), Value::Bool(true));
    } else if let Some(value) = payload.get("parallel_tool_calls") {
        request.insert("parallel_tool_calls".to_string(), value.clone());
    }

    serde_json::to_vec(&Value::Object(request))
        .context("failed to encode Nemotron chat-completions payload")
}

pub fn rewrite_responses_to_glm_chat_completions(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("zai-org/GLM-4.7-Flash")
        .to_string();
    let instructions = payload
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let messages = build_glm_chat_messages(
        &normalize_responses_input(payload.get("input")),
        instructions.as_deref(),
    );
    let tools = payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .flat_map(|tool| rewrite_glm_chat_tool(tool.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut request = serde_json::Map::new();
    request.insert("model".to_string(), Value::String(model));
    request.insert("messages".to_string(), Value::Array(messages));
    if !tools.is_empty() {
        request.insert("tools".to_string(), Value::Array(tools));
    }
    let (enable_thinking, reasoning_effort) = glm_chat_reasoning_config(&payload);
    request.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    if let Some(reasoning_effort) = reasoning_effort {
        request.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort),
        );
    }
    for key in [
        "tool_choice",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_output_tokens",
    ] {
        if let Some(value) = payload.get(key) {
            let mapped_key = if key == "max_output_tokens" {
                "max_tokens"
            } else {
                key
            };
            request.insert(mapped_key.to_string(), value.clone());
        }
    }
    request.insert("stream".to_string(), Value::Bool(false));
    if payload.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        request.insert("parallel_tool_calls".to_string(), Value::Bool(true));
    } else if let Some(value) = payload.get("parallel_tool_calls") {
        request.insert("parallel_tool_calls".to_string(), value.clone());
    }

    serde_json::to_vec(&Value::Object(request))
        .context("failed to encode GLM chat-completions payload")
}

fn glm_chat_reasoning_config(payload: &Value) -> (bool, Option<String>) {
    let effort = payload
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| matches!(value.as_str(), "low" | "medium" | "high"));
    (effort.is_some(), effort)
}

fn build_qwen_chat_messages(items: &[Value], instructions: Option<&str>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    let mut pending_assistant: Option<serde_json::Map<String, Value>> = None;
    let flush_pending_assistant =
        |pending_assistant: &mut Option<serde_json::Map<String, Value>>,
         messages: &mut Vec<Value>| {
            if let Some(assistant) = pending_assistant.take() {
                messages.push(Value::Object(assistant));
            }
        };

    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match item_type {
            "message" => {
                let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
                let mapped_role = match role {
                    "developer" => "system",
                    other => other,
                };
                let text = extract_message_content_text(object.get("content"));
                if mapped_role == "assistant" {
                    flush_pending_assistant(&mut pending_assistant, &mut messages);
                    let mut assistant = serde_json::Map::new();
                    assistant.insert("role".to_string(), Value::String("assistant".to_string()));
                    let (reasoning, content) = split_qwen_reasoning_and_content(&text);
                    assistant.insert("content".to_string(), Value::String(content));
                    if let Some(reasoning) = reasoning {
                        assistant.insert("reasoning_content".to_string(), Value::String(reasoning));
                    }
                    pending_assistant = Some(assistant);
                } else {
                    flush_pending_assistant(&mut pending_assistant, &mut messages);
                    messages.push(json!({
                        "role": mapped_role,
                        "content": text,
                    }));
                }
            }
            "function_call" => {
                let _call_id = object
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_ctox_proxy");
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let arguments = object
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let assistant = pending_assistant.get_or_insert_with(|| {
                    let mut assistant = serde_json::Map::new();
                    assistant.insert("role".to_string(), Value::String("assistant".to_string()));
                    assistant.insert("content".to_string(), Value::String(String::new()));
                    assistant
                });
                let existing = assistant
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let rendered = render_qwen_xml_tool_call(name, arguments);
                let combined = if existing.trim().is_empty() {
                    rendered
                } else {
                    format!("{existing}{rendered}")
                };
                assistant.insert("content".to_string(), Value::String(combined));
            }
            "function_call_output" => {
                flush_pending_assistant(&mut pending_assistant, &mut messages);
                let call_id = object
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_ctox_proxy");
                let output = extract_function_call_output_text(object.get("output"));
                messages.push(json!({
                    "role": "user",
                    "content": format!("<tool_response>\n{}\n</tool_response>", output.trim_end()),
                    "tool_call_id": call_id,
                }));
            }
            _ => {}
        }
    }

    flush_pending_assistant(&mut pending_assistant, &mut messages);
    messages
}

fn build_glm_chat_messages(items: &[Value], instructions: Option<&str>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    let mut pending_assistant: Option<serde_json::Map<String, Value>> = None;
    let flush_pending_assistant =
        |pending_assistant: &mut Option<serde_json::Map<String, Value>>,
         messages: &mut Vec<Value>| {
            if let Some(assistant) = pending_assistant.take() {
                messages.push(Value::Object(assistant));
            }
        };

    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match item_type {
            "message" => {
                let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
                let mapped_role = match role {
                    "developer" => "system",
                    other => other,
                };
                let text = extract_message_content_text(object.get("content"));
                if mapped_role == "assistant" {
                    flush_pending_assistant(&mut pending_assistant, &mut messages);
                    let mut assistant = serde_json::Map::new();
                    assistant.insert("role".to_string(), Value::String("assistant".to_string()));
                    let (reasoning, content) = split_qwen_reasoning_and_content(&text);
                    assistant.insert("content".to_string(), Value::String(content));
                    if let Some(reasoning) = reasoning {
                        assistant.insert("reasoning_content".to_string(), Value::String(reasoning));
                    }
                    pending_assistant = Some(assistant);
                } else {
                    flush_pending_assistant(&mut pending_assistant, &mut messages);
                    messages.push(json!({
                        "role": mapped_role,
                        "content": text,
                    }));
                }
            }
            "function_call" => {
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let arguments = object
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let assistant = pending_assistant.get_or_insert_with(|| {
                    let mut assistant = serde_json::Map::new();
                    assistant.insert("role".to_string(), Value::String("assistant".to_string()));
                    assistant.insert("content".to_string(), Value::String(String::new()));
                    assistant.insert("tool_calls".to_string(), Value::Array(Vec::new()));
                    assistant
                });
                let tool_calls = assistant
                    .entry("tool_calls".to_string())
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(tool_calls) = tool_calls.as_array_mut() {
                    let parsed_arguments = serde_json::from_str::<Value>(arguments)
                        .unwrap_or_else(|_| json!({ "input": arguments }));
                    tool_calls.push(json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": parsed_arguments
                        }
                    }));
                }
            }
            "function_call_output" => {
                flush_pending_assistant(&mut pending_assistant, &mut messages);
                let output = extract_function_call_output_text(object.get("output"));
                messages.push(json!({
                    "role": "tool",
                    "content": output.trim_end(),
                }));
            }
            _ => {}
        }
    }

    flush_pending_assistant(&mut pending_assistant, &mut messages);
    messages
}

fn render_qwen_xml_tool_call(name: &str, raw_arguments: &str) -> String {
    let argument_pairs = serde_json::from_str::<Value>(raw_arguments)
        .ok()
        .and_then(|parsed| parsed.as_object().cloned())
        .unwrap_or_else(|| {
            let mut fallback = serde_json::Map::new();
            fallback.insert(
                "input".to_string(),
                Value::String(raw_arguments.to_string()),
            );
            fallback
        });

    let mut rendered = format!("\n\n<tool_call>\n<function={name}>\n");
    for (argument_name, argument_value) in argument_pairs {
        rendered.push_str(&format!("<parameter={argument_name}>\n"));
        if argument_value.is_object() || argument_value.is_array() {
            rendered.push_str(&argument_value.to_string());
        } else if let Some(text) = argument_value.as_str() {
            rendered.push_str(text);
        } else {
            rendered.push_str(&argument_value.to_string());
        }
        rendered.push_str("\n</parameter>\n");
    }
    rendered.push_str("</function>\n</tool_call>");
    rendered
}

fn rewrite_qwen_chat_tool(tool: Value) -> Vec<Value> {
    let Some(object) = tool.as_object() else {
        return Vec::new();
    };
    let Some(tool_type) = object.get("type").and_then(Value::as_str) else {
        return Vec::new();
    };
    match tool_type {
        "function" => {
            let Some(name) = object.get("name").and_then(Value::as_str) else {
                return Vec::new();
            };
            if name == "apply_patch" {
                return Vec::new();
            }
            rewrite_tool(Value::Object(object.clone()))
                .into_iter()
                .collect()
        }
        "namespace" => object
            .get("tools")
            .and_then(Value::as_array)
            .map(|children| {
                children
                    .iter()
                    .flat_map(|child| rewrite_qwen_chat_tool(child.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn rewrite_glm_chat_tool(tool: Value) -> Vec<Value> {
    let Some(object) = tool.as_object() else {
        return Vec::new();
    };
    let Some(tool_type) = object.get("type").and_then(Value::as_str) else {
        return Vec::new();
    };
    match tool_type {
        "function" => {
            let Some(name) = object.get("name").and_then(Value::as_str) else {
                return Vec::new();
            };
            if name == "apply_patch" {
                return Vec::new();
            }
            rewrite_tool(Value::Object(object.clone()))
                .into_iter()
                .collect()
        }
        "namespace" => object
            .get("tools")
            .and_then(Value::as_array)
            .map(|children| {
                children
                    .iter()
                    .flat_map(|child| rewrite_glm_chat_tool(child.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

pub fn rewrite_qwen_chat_completions_to_responses(
    raw: &[u8],
    fallback_model: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse chat completion response")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(fallback_model)
        .unwrap_or("Qwen/Qwen3.5-4B")
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

    let mut output_items = Vec::new();
    let mut output_text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    if let Some(choices) = payload.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let message = choice.get("message").and_then(Value::as_object);
            if let Some(text) = message
                .and_then(|msg| msg.get("content"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|text| !text.is_empty())
            {
                let (plain_text, xml_tool_calls) = parse_qwen_xml_tool_calls(&text);
                if let Some(plain_text) = plain_text.filter(|text| !text.is_empty()) {
                    output_text_parts.push(plain_text.clone());
                    output_items.push(json!({
                        "type": "message",
                        "id": "msg_ctox_proxy",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": plain_text, "annotations": []}]
                    }));
                }
                for tool_call in xml_tool_calls {
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }));
                }
            }
            if let Some(reasoning) = message
                .and_then(|msg| msg.get("reasoning_content"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|text| !text.is_empty())
            {
                reasoning_parts.push(reasoning);
            }
            if let Some(tool_calls) = message
                .and_then(|msg| msg.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tool_call in tool_calls {
                    let name = tool_call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let arguments = tool_call
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let call_id = tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("call_ctox_proxy");
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments
                    }));
                }
            }
        }
    }
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
        "output": output_items,
        "output_text": if output_text_parts.is_empty() { Value::Null } else { Value::String(output_text_parts.join("")) },
        "reasoning": if reasoning_parts.is_empty() { Value::Null } else { Value::String(reasoning_parts.join("\n")) },
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0)
        }
    });
    serde_json::to_vec(&response_payload).context("failed to encode Qwen responses payload")
}

pub fn rewrite_nemotron_chat_completions_to_responses(
    raw: &[u8],
    fallback_model: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse chat completion response")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(fallback_model)
        .unwrap_or("nvidia/Nemotron-Cascade-2-30B-A3B")
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

    let mut output_items = Vec::new();
    let mut output_text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    if let Some(choices) = payload.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let message = choice.get("message").and_then(Value::as_object);
            let mut choice_reasoning = Vec::new();
            if let Some(reasoning) = message
                .and_then(|msg| msg.get("reasoning_content"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
            {
                choice_reasoning.push(reasoning.to_string());
            }
            if let Some(text) = message
                .and_then(|msg| msg.get("content"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|text| !text.is_empty())
            {
                let (inline_reasoning, visible_text) = split_qwen_reasoning_and_content(&text);
                if let Some(inline_reasoning) = inline_reasoning
                    .map(|text| text.trim().to_string())
                    .filter(|text| !text.is_empty())
                {
                    if !choice_reasoning
                        .iter()
                        .any(|existing| existing == &inline_reasoning)
                    {
                        choice_reasoning.push(inline_reasoning);
                    }
                }
                let (plain_text, xml_tool_calls) = parse_qwen_xml_tool_calls(&visible_text);
                if let Some(plain_text) = plain_text.filter(|text| !text.is_empty()) {
                    output_text_parts.push(plain_text.clone());
                    output_items.push(json!({
                        "type": "message",
                        "id": "msg_ctox_proxy",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": plain_text, "annotations": []}]
                    }));
                }
                for tool_call in xml_tool_calls {
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }));
                }
            }
            reasoning_parts.extend(choice_reasoning);
            if let Some(tool_calls) = message
                .and_then(|msg| msg.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tool_call in tool_calls {
                    let name = tool_call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let arguments = tool_call
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let call_id = tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("call_ctox_proxy");
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments
                    }));
                }
            }
        }
    }
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
        "output": output_items,
        "output_text": if output_text_parts.is_empty() { Value::Null } else { Value::String(output_text_parts.join("")) },
        "reasoning": if reasoning_parts.is_empty() { Value::Null } else { Value::String(reasoning_parts.join("\n")) },
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0)
        }
    });
    serde_json::to_vec(&response_payload).context("failed to encode Nemotron responses payload")
}

pub fn rewrite_glm_chat_completions_to_responses(
    raw: &[u8],
    fallback_model: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse chat completion response")?;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .or(fallback_model)
        .unwrap_or("zai-org/GLM-4.7-Flash")
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

    let mut output_items = Vec::new();
    let mut output_text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    if let Some(choices) = payload.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let message = choice.get("message").and_then(Value::as_object);
            if let Some(text) = message
                .and_then(|msg| msg.get("content"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|text| !text.is_empty())
            {
                let (plain_text, xml_tool_calls) = parse_glm_xml_tool_calls(&text);
                if let Some(plain_text) = plain_text.filter(|text| !text.is_empty()) {
                    output_text_parts.push(plain_text.clone());
                    output_items.push(json!({
                        "type": "message",
                        "id": "msg_ctox_proxy",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": plain_text, "annotations": []}]
                    }));
                }
                for tool_call in xml_tool_calls {
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }));
                }
            }
            if let Some(reasoning) = message
                .and_then(|msg| msg.get("reasoning_content"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|text| !text.is_empty())
            {
                reasoning_parts.push(reasoning);
            }
            if let Some(tool_calls) = message
                .and_then(|msg| msg.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tool_call in tool_calls {
                    let function = tool_call.get("function").unwrap_or(tool_call);
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let arguments = function
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    output_items.push(json!({
                        "type": "function_call",
                        "call_id": format!("call_ctox_proxy_{}", output_items.len()),
                        "name": name,
                        "arguments": if arguments.is_string() { arguments } else { Value::String(arguments.to_string()) }
                    }));
                }
            }
        }
    }
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
        "output": output_items,
        "output_text": if output_text_parts.is_empty() { Value::Null } else { Value::String(output_text_parts.join("")) },
        "reasoning": if reasoning_parts.is_empty() { Value::Null } else { Value::String(reasoning_parts.join("\n")) },
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or_else(|| usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0)),
            "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0)
        }
    });
    serde_json::to_vec(&response_payload).context("failed to encode GLM responses payload")
}

fn split_qwen_reasoning_and_content(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        if let Some((reasoning, content)) = rest.split_once("</think>") {
            let reasoning = reasoning.trim().to_string();
            let content = content.trim().to_string();
            return (
                if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
                content,
            );
        }
    }
    (None, trimmed.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QwenXmlToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

fn parse_qwen_xml_tool_calls(text: &str) -> (Option<String>, Vec<QwenXmlToolCall>) {
    let tool_call_re = Regex::new(
        r"(?s)<tool_call>\s*<function=([A-Za-z0-9_-]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    )
    .expect("valid Qwen tool call regex");
    let param_re = Regex::new(r"(?s)<parameter=([A-Za-z0-9_-]+)>\s*(.*?)\s*</parameter>")
        .expect("valid parameter regex");
    let mut tool_calls = Vec::new();
    let mut plain_text = String::new();
    let mut last_end = 0usize;
    for (index, captures) in tool_call_re.captures_iter(text).enumerate() {
        let Some(matched) = captures.get(0) else {
            continue;
        };
        plain_text.push_str(&text[last_end..matched.start()]);
        last_end = matched.end();
        let name = captures
            .get(1)
            .map(|capture| capture.as_str().trim().to_string())
            .unwrap_or_default();
        let body = captures
            .get(2)
            .map(|capture| capture.as_str())
            .unwrap_or_default();
        let mut arguments = serde_json::Map::new();
        for param in param_re.captures_iter(body) {
            let Some(param_name) = param.get(1).map(|capture| capture.as_str().trim()) else {
                continue;
            };
            let value = param
                .get(2)
                .map(|capture| capture.as_str().trim().to_string())
                .unwrap_or_default();
            arguments.insert(param_name.to_string(), Value::String(value));
        }
        tool_calls.push(QwenXmlToolCall {
            call_id: format!("call_ctox_proxy_{index}"),
            name,
            arguments: Value::Object(arguments).to_string(),
        });
    }
    plain_text.push_str(&text[last_end..]);
    let plain_text = plain_text.trim().to_string();
    (
        if plain_text.is_empty() {
            None
        } else {
            Some(plain_text)
        },
        tool_calls,
    )
}

fn parse_glm_xml_tool_calls(text: &str) -> (Option<String>, Vec<QwenXmlToolCall>) {
    let tool_call_re = Regex::new(r"(?s)<tool_call>\s*([A-Za-z0-9_.-]+)\s*(.*?)\s*</tool_call>")
        .expect("valid GLM tool call regex");
    let arg_re =
        Regex::new(r"(?s)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>")
            .expect("valid GLM arg regex");
    let mut tool_calls = Vec::new();
    let mut plain_text = String::new();
    let mut last_end = 0usize;
    for (index, captures) in tool_call_re.captures_iter(text).enumerate() {
        let Some(matched) = captures.get(0) else {
            continue;
        };
        plain_text.push_str(&text[last_end..matched.start()]);
        last_end = matched.end();
        let name = captures
            .get(1)
            .map(|capture| capture.as_str().trim().to_string())
            .unwrap_or_default();
        let body = captures
            .get(2)
            .map(|capture| capture.as_str())
            .unwrap_or_default();
        let mut arguments = serde_json::Map::new();
        for arg in arg_re.captures_iter(body) {
            let Some(key) = arg.get(1).map(|capture| capture.as_str().trim()) else {
                continue;
            };
            let value = arg
                .get(2)
                .map(|capture| capture.as_str().trim().to_string())
                .unwrap_or_default();
            arguments.insert(key.to_string(), Value::String(value));
        }
        tool_calls.push(QwenXmlToolCall {
            call_id: format!("call_ctox_proxy_{index}"),
            name,
            arguments: Value::Object(arguments).to_string(),
        });
    }
    plain_text.push_str(&text[last_end..]);
    let plain_text = plain_text.replace("<|observation|>", "").trim().to_string();
    (
        if plain_text.is_empty() {
            None
        } else {
            Some(plain_text)
        },
        tool_calls,
    )
}

pub fn rewrite_responses_payload_to_sse(raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses payload")?;
    let response_id = payload
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("resp_ctox_proxy");
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let usage = payload.get("usage").cloned().unwrap_or_else(|| {
        json!({
            "input_tokens": 0,
            "output_tokens": 0,
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
    if let Some(items) = payload.get("output").and_then(Value::as_array) {
        for item in items {
            push_response_output_item_frames(&mut frames, item);
        }
    }
    frames.push((
        "response.completed",
        json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "model": model,
                "usage": {
                    "input_tokens": usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
                    "output_tokens": usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
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

fn push_response_output_item_frames(frames: &mut Vec<(&'static str, String)>, item: &Value) {
    if let Some(partial_item) = partial_output_item_for_sse(item) {
        frames.push((
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "item": partial_item,
            })
            .to_string(),
        ));
    }

    frames.push((
        "response.output_item.done",
        json!({
            "type": "response.output_item.done",
            "item": item
        })
        .to_string(),
    ));
}

fn partial_output_item_for_sse(item: &Value) -> Option<Value> {
    let item_type = item.get("type").and_then(Value::as_str)?;
    if item_type != "web_search_call" {
        return None;
    }

    let id = item.get("id").and_then(Value::as_str)?;
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .filter(|status| !status.is_empty())
        .unwrap_or("in_progress");
    let partial_status = match status {
        "completed" | "failed" => "in_progress",
        other => other,
    };

    Some(json!({
        "type": item_type,
        "id": id,
        "status": partial_status,
    }))
}

fn responses_input_item_to_chat_message(item: &Value) -> Option<Value> {
    let object = item.as_object()?;
    let item_type = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    match item_type {
        "message" => {
            let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
            let mapped_role = match role {
                "developer" => "system",
                other => other,
            };
            let text = extract_message_content_text(object.get("content"));
            Some(json!({
                "role": mapped_role,
                "content": text,
            }))
        }
        "function_call" => {
            let call_id = object
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("call_ctox_proxy");
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = object
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            Some(json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }]
            }))
        }
        "function_call_output" => {
            let call_id = object
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("call_ctox_proxy");
            let output = extract_function_call_output_text(object.get("output"));
            Some(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            }))
        }
        _ => None,
    }
}

pub fn should_use_gpt_oss_harmony_proxy(raw: &[u8]) -> anyhow::Result<bool> {
    Ok(parse_harmony_proxy_request(raw)
        .map(|request| is_gpt_oss_model_id(&request.model))
        .unwrap_or(false))
}

pub fn responses_request_streams(raw: &[u8]) -> anyhow::Result<bool> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;
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
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse completion response")?;
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
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse completion response")?;
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
        let output_item = harmony_item_to_responses_output(item);
        push_response_output_item_frames(&mut frames, &output_item);
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
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse completion response")?;
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

    let mut request: Value = serde_json::from_slice(initial_request_raw)
        .context("failed to parse initial completion request")?;
    let first_payload: Value = serde_json::from_slice(first_completion_raw)
        .context("failed to parse first completion response")?;
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
                if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name) {
                    return None;
                }
                return Some(Value::Object(object.clone()));
            }

            let name = object.get("name").and_then(Value::as_str)?;
            if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name) {
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

fn rewrite_openai_tool(tool: Value) -> Vec<Value> {
    let Some(object) = tool.as_object() else {
        return Vec::new();
    };
    let Some(tool_type) = object.get("type").and_then(Value::as_str) else {
        return Vec::new();
    };
    match tool_type {
        "web_search" => vec![Value::Object(object.clone())],
        "function" => {
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .unwrap_or(object);
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                return Vec::new();
            };
            if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name) {
                return Vec::new();
            }
            let mut flattened = serde_json::Map::new();
            flattened.insert("type".to_string(), Value::String("function".to_string()));
            for key in ["name", "description", "parameters", "strict"] {
                if let Some(value) = function.get(key) {
                    flattened.insert(key.to_string(), value.clone());
                }
            }
            vec![Value::Object(flattened)]
        }
        "namespace" => object
            .get("tools")
            .and_then(Value::as_array)
            .map(|children| {
                children
                    .iter()
                    .flat_map(|child| rewrite_openai_tool(child.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn parse_harmony_proxy_request(raw: &[u8]) -> anyhow::Result<HarmonyProxyRequest> {
    let payload: Value =
        serde_json::from_slice(raw).context("failed to parse responses request")?;
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
            items
                .iter()
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
    if DISALLOWED_ENGINE_FUNCTION_TOOLS.contains(&name.as_str()) {
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
        "<|start|>assistant<|channel|>"
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
                let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
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
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
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
    let mut payload: Value = serde_json::from_slice(raw)
        .context("failed to parse responses request for materialization")?;
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
        block.push_str("After emitting a tool call, stop immediately.\n");
        block.push_str(
            "Use only the provided function tool definitions from the request metadata.\n",
        );
        block.push_str("Available tools: ");
        block.push_str(
            &tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        block.push('\n');
    }
    block.trim_end().to_string()
}

fn json_schema_to_typescript(schema: &Value) -> String {
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
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
                    line.push_str(&format!(
                        "{key}{optional}: {}",
                        json_schema_type_to_typescript(value)
                    ));
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
    sanitize_harmony_channel_leakage(&text)
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
        Value::String(text) => {
            serde_json::to_string(&json!({ "cmd": text })).unwrap_or_else(|_| arguments.to_string())
        }
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
    let key_re = Regex::new(&format!(r#""{}"\s*:"#, regex::escape(key))).ok()?;
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
    let key_re = Regex::new(&format!(r#""{}"\s*:"#, regex::escape(key))).ok()?;
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
    re.captures(source)?.get(1)?.as_str().parse::<i64>().ok()
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
    serde_json::from_str::<String>(&format!(
        "\"{}\"",
        text.replace('\\', "\\\\").replace('"', "\\\"")
    ))
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

fn sanitize_harmony_channel_leakage(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    let saw_plaintext_harmony = contains_plaintext_harmony_marker(&text);

    loop {
        let mut stripped_any = false;
        for prefix in [
            "assistantfinal",
            "final",
            "assistantcommentary",
            "commentary",
            "assistantanalysis",
            "analysis",
        ] {
            if let Some(stripped) = text.strip_prefix(prefix) {
                text = stripped.trim_start().to_string();
                stripped_any = true;
            }
        }
        if !stripped_any {
            break;
        }
    }

    if let Some(idx) = find_plaintext_harmony_marker(&text) {
        text.truncate(idx);
    }

    if saw_plaintext_harmony {
        text = trim_trailing_incomplete_harmony_token(&text).to_string();
    }

    text.trim().to_string()
}

fn find_plaintext_harmony_marker(text: &str) -> Option<usize> {
    let markers = [
        "assistantanalysis",
        "assistantfinal",
        "assistantcommentary",
        "analysis",
        "final",
        "commentary",
    ];
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|idx| (idx, *marker)))
        .filter(|(idx, marker)| {
            if *idx == 0 {
                return false;
            }
            let preceding = text[..*idx].chars().last();
            let following = text[idx + marker.len()..].chars().next();
            let preceding_is_payload = preceding
                .map(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ')' | ']' | '"' | '\''))
                .unwrap_or(false);
            let following_looks_like_channel = following
                .map(|ch| ch.is_ascii_uppercase() || ch == '<' || ch == '{' || ch == '[')
                .unwrap_or(false);
            preceding_is_payload && following_looks_like_channel
        })
        .map(|(idx, _)| idx)
        .min()
}

fn contains_plaintext_harmony_marker(text: &str) -> bool {
    [
        "assistantanalysis",
        "assistantfinal",
        "assistantcommentary",
        "analysis",
        "final",
        "commentary",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn trim_trailing_incomplete_harmony_token(text: &str) -> &str {
    text.strip_suffix("assistant")
        .or_else(|| text.strip_suffix("analysis"))
        .or_else(|| text.strip_suffix("commentary"))
        .or_else(|| text.strip_suffix("final"))
        .unwrap_or(text)
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
    fn gpt_oss_runtime_uses_engine_gpt_oss_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::GptOss);
        let command = build_engine_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert!(command.iter().any(|part| part == "gpt_oss"));
        assert!(command.iter().any(|part| part == "--max-seq-len"));
    }

    #[test]
    fn qwen_runtime_uses_engine_vision_startup() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));
        let runtime = default_runtime_config(LocalModelFamily::Qwen35Vision);
        let command = build_engine_command(&deps, &runtime);
        assert_eq!(command[1], "serve");
        assert_eq!(command[2], "-p");
        assert_eq!(command[4], "vision");
        assert!(command.iter().any(|part| part == "Qwen/Qwen3.5-27B"));
    }

    #[test]
    fn auxiliary_profiles_use_dedicated_engine_subcommands() {
        let deps = discover_vendored_dependency_paths(Path::new("/tmp/ctox"));

        let embedding = build_engine_command(
            &deps,
            &default_runtime_config(LocalModelFamily::Qwen3Embedding),
        );
        assert_eq!(embedding[4], "embedding");
        assert!(embedding
            .iter()
            .any(|part| part == "Qwen/Qwen3-Embedding-0.6B"));

        let stt = build_engine_command(
            &deps,
            &default_runtime_config(LocalModelFamily::VoxtralTranscription),
        );
        assert_eq!(stt[4], "vision");
        assert!(stt
            .iter()
            .any(|part| part == "mistralai/Voxtral-Mini-4B-Realtime-2602"));

        let tts = build_engine_command(
            &deps,
            &default_runtime_config(LocalModelFamily::VoxtralSpeech),
        );
        assert!(tts[0].ends_with("ctox-engine"));
        assert_eq!(tts[1], "serve");
        assert_eq!(tts[2], "-p");
        assert!(tts.iter().any(|part| part == "speech"));
        assert!(tts.windows(2).any(|pair| pair == ["--isq", "Q4K"]));
        assert!(tts
            .iter()
            .any(|part| part == "mistralai/Voxtral-4B-TTS-2603"));
    }

    #[test]
    fn auxiliary_model_choices_resolve_cpu_and_gpu_variants() {
        let embedding_cpu = auxiliary_model_selection(
            AuxiliaryRole::Embedding,
            Some("Qwen/Qwen3-Embedding-0.6B [CPU]"),
        );
        assert_eq!(embedding_cpu.request_model, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(embedding_cpu.compute_target, ComputeTarget::Cpu);
        assert_eq!(embedding_cpu.backend_kind, AuxiliaryBackendKind::MistralRs);
        assert_eq!(embedding_cpu.gpu_reserve_mb(), 0);

        let stt_gpu = auxiliary_model_selection(
            AuxiliaryRole::Stt,
            Some("mistralai/Voxtral-Mini-4B-Realtime-2602"),
        );
        assert_eq!(
            stt_gpu.choice,
            "mistralai/Voxtral-Mini-4B-Realtime-2602 [GPU]"
        );
        assert_eq!(stt_gpu.compute_target, ComputeTarget::Gpu);
        assert_eq!(stt_gpu.backend_kind, AuxiliaryBackendKind::MistralRs);
        assert_eq!(stt_gpu.gpu_reserve_mb(), 4200);

        let tts_cpu = auxiliary_model_selection(
            AuxiliaryRole::Tts,
            Some("speaches-ai/piper-fr_FR-siwis-medium [CPU FR]"),
        );
        assert_eq!(
            tts_cpu.request_model,
            "speaches-ai/piper-fr_FR-siwis-medium"
        );
        assert_eq!(tts_cpu.compute_target, ComputeTarget::Cpu);
        assert_eq!(tts_cpu.backend_kind, AuxiliaryBackendKind::Speaches);
        assert_eq!(tts_cpu.default_port, 1239);

        let tts_de = auxiliary_model_selection(
            AuxiliaryRole::Tts,
            Some("speaches-ai/piper-de_DE-thorsten-high [CPU DE]"),
        );
        assert_eq!(
            tts_de.request_model,
            "speaches-ai/piper-de_DE-thorsten-high"
        );
        assert_eq!(tts_de.compute_target, ComputeTarget::Cpu);
        assert_eq!(tts_de.backend_kind, AuxiliaryBackendKind::Speaches);

        let tts_qwen = auxiliary_model_selection(
            AuxiliaryRole::Tts,
            Some("Qwen/Qwen3-TTS-12Hz-0.6B-Base [GPU]"),
        );
        assert_eq!(tts_qwen.choice, "Qwen/Qwen3-TTS-12Hz-0.6B-Base [GPU]");
        assert_eq!(tts_qwen.request_model, "Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        assert_eq!(tts_qwen.compute_target, ComputeTarget::Gpu);
        assert_eq!(tts_qwen.backend_kind, AuxiliaryBackendKind::MistralRs);

        let tts_default = auxiliary_model_selection(AuxiliaryRole::Tts, None);
        assert_eq!(tts_default.choice, "mistralai/Voxtral-4B-TTS-2603 [GPU]");
        assert_eq!(tts_default.request_model, "mistralai/Voxtral-4B-TTS-2603");
        assert_eq!(tts_default.compute_target, ComputeTarget::Gpu);
    }

    #[test]
    fn family_profiles_drive_nccl_policy() {
        let gpt_oss = runtime_profile_for_model("openai/gpt-oss-20b").unwrap();
        let qwen = runtime_profile_for_model("Qwen/Qwen3.5-27B").unwrap();
        let glm = runtime_profile_for_model("zai-org/GLM-4.7-Flash").unwrap();
        let embedding = runtime_profile_for_model("Qwen/Qwen3-Embedding-0.6B").unwrap();
        let stt = runtime_profile_for_model("mistralai/Voxtral-Mini-4B-Realtime-2602").unwrap();
        let tts = runtime_profile_for_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base").unwrap();
        let voxtral_tts = runtime_profile_for_model("mistralai/Voxtral-4B-TTS-2603").unwrap();
        assert!(gpt_oss.disable_nccl);
        assert_eq!(gpt_oss.tensor_parallel_backend, None);
        assert_eq!(gpt_oss.target_world_size, None);
        assert_eq!(gpt_oss.preferred_gpu_count, Some(1));
        assert!(qwen.disable_nccl);
        assert_eq!(qwen.tensor_parallel_backend, None);
        assert_eq!(qwen.target_world_size, None);
        assert_eq!(qwen.preferred_gpu_count, Some(3));
        assert!(glm.disable_nccl);
        assert_eq!(glm.tensor_parallel_backend, None);
        assert_eq!(glm.preferred_gpu_count, Some(3));
        assert!(embedding.disable_nccl);
        assert_eq!(embedding.preferred_gpu_count, Some(1));
        assert!(stt.disable_nccl);
        assert_eq!(stt.preferred_gpu_count, Some(1));
        assert!(tts.disable_nccl);
        assert_eq!(tts.preferred_gpu_count, Some(1));
        assert_eq!(tts.isq.as_deref(), Some("Q4K"));
        assert!(voxtral_tts.disable_nccl);
        assert_eq!(voxtral_tts.preferred_gpu_count, Some(1));
        assert_eq!(voxtral_tts.isq.as_deref(), Some("Q4K"));
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
        assert_eq!(plan.family_profile.tensor_parallel_backend, None);
    }

    #[test]
    fn model_specific_runtime_profiles_match_size_class() {
        let qwen_small = runtime_profile_for_model("Qwen/Qwen3.5-4B").unwrap();
        assert!(qwen_small.disable_nccl);
        assert_eq!(qwen_small.preferred_gpu_count, Some(1));
        assert_eq!(qwen_small.max_seq_len, 262_144);

        let qwen_large = runtime_profile_for_model("Qwen/Qwen3.5-35B-A3B").unwrap();
        assert!(qwen_large.disable_nccl);
        assert_eq!(qwen_large.target_world_size, None);
        assert_eq!(qwen_large.preferred_gpu_count, Some(3));
        assert_eq!(qwen_large.paged_attn, "auto");
        assert_eq!(qwen_large.pa_cache_type.as_deref(), Some("turboquant3"));

        let nemotron = runtime_profile_for_model("nemotron-cascade-2-30b-a3b").unwrap();
        assert!(nemotron.disable_nccl);
        assert_eq!(nemotron.arch, None);
        assert_eq!(nemotron.max_seq_len, 8_192);

        let glm = runtime_profile_for_model("GLN 4.7 flash").unwrap();
        assert_eq!(glm.arch.as_deref(), Some("glm4moelite"));
        assert_eq!(glm.pa_cache_type.as_deref(), Some("turboquant3"));
    }

    #[test]
    fn pa_cache_type_override_can_enable_turboquant() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            PA_CACHE_TYPE_OVERRIDE_ENV.to_string(),
            TURBOQUANT3_CACHE_TYPE.to_string(),
        );
        assert_eq!(
            resolve_pa_cache_type(Some("f8e4m3"), &env_map),
            Some(TURBOQUANT3_CACHE_TYPE.to_string())
        );
    }

    #[test]
    fn experimental_turboquant_switches_fp8_defaults_only() {
        let mut env_map = BTreeMap::new();
        env_map.insert(EXPERIMENTAL_TURBOQUANT_ENV.to_string(), "1".to_string());
        assert_eq!(
            resolve_pa_cache_type(Some("f8e4m3"), &env_map),
            Some(TURBOQUANT3_CACHE_TYPE.to_string())
        );
        assert_eq!(resolve_pa_cache_type(None, &env_map), None);
    }

    #[test]
    fn turboquant_override_is_retained_for_gpt_oss() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            PA_CACHE_TYPE_OVERRIDE_ENV.to_string(),
            TURBOQUANT3_CACHE_TYPE.to_string(),
        );
        assert_eq!(
            resolve_model_pa_cache_type("openai/gpt-oss-20b", Some("f8e4m3"), &env_map),
            Some(TURBOQUANT3_CACHE_TYPE.to_string())
        );
    }

    #[test]
    fn turboquant_override_is_retained_for_supported_model_family() {
        let mut env_map = BTreeMap::new();
        env_map.insert(
            PA_CACHE_TYPE_OVERRIDE_ENV.to_string(),
            TURBOQUANT3_CACHE_TYPE.to_string(),
        );
        assert_eq!(
            resolve_model_pa_cache_type("Qwen/Qwen3.5-27B", Some("f8e4m3"), &env_map),
            Some(TURBOQUANT3_CACHE_TYPE.to_string())
        );
    }

    #[test]
    fn experimental_turboquant_is_retained_for_gpt_oss() {
        let mut env_map = BTreeMap::new();
        env_map.insert(EXPERIMENTAL_TURBOQUANT_ENV.to_string(), "1".to_string());
        assert_eq!(
            resolve_model_pa_cache_type("openai/gpt-oss-20b", Some("f8e4m3"), &env_map),
            Some(TURBOQUANT3_CACHE_TYPE.to_string())
        );
    }

    #[test]
    fn cache_override_is_ignored_for_non_paged_attention_family() {
        let mut env_map = BTreeMap::new();
        env_map.insert(PA_CACHE_TYPE_OVERRIDE_ENV.to_string(), "f8e4m3".to_string());
        assert_eq!(
            resolve_model_pa_cache_type("Qwen/Qwen3-TTS-12Hz-0.6B-Base", None, &env_map,),
            None
        );
    }

    #[test]
    fn supported_cache_override_promotes_paged_attention_from_off() {
        assert_eq!(
            resolve_model_paged_attn("Qwen/Qwen3.5-35B-A3B", "off", Some("turboquant3")),
            "auto"
        );
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
        assert!(command.iter().any(|part| part == "web_search=\"live\""));
    }

    #[test]
    fn harmony_prompt_does_not_inline_full_tool_schemas() {
        let prompt = build_gpt_oss_harmony_prompt(
            "Be precise.",
            &[json!({
                "type":"message",
                "role":"user",
                "content":[{"text":"Reply with OK"}]
            })],
            "medium",
            &[HarmonyToolSpec {
                name: "exec_command".to_string(),
                description: Some("Runs a shell command".to_string()),
                parameters: Some(json!({"type":"object"})),
            }],
        );
        assert!(prompt.contains("Available tools: exec_command"));
        assert!(prompt.contains(
            "Use only the provided function tool definitions from the request metadata."
        ));
        assert!(!prompt.contains("namespace functions"));
        assert!(!prompt.contains("type exec_command"));
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
    fn recognizes_openai_api_chat_models() {
        assert!(is_openai_api_chat_model("gpt-5.4"));
        assert!(is_openai_api_chat_model("GPT-5.4-MINI"));
        assert!(!is_openai_api_chat_model("openai/gpt-oss-20b"));
    }

    #[test]
    fn only_local_runtime_models_use_ctox_proxy_path() {
        assert!(!uses_ctox_proxy_model("gpt-5.4"));
        assert!(!uses_ctox_proxy_model("gpt-5.4-nano"));
        assert!(uses_ctox_proxy_model("Qwen/Qwen3.5-4B"));
        assert!(!uses_ctox_proxy_model("not-a-real-model"));
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
        let rewritten =
            rewrite_engine_responses_request(serde_json::to_vec(&payload).unwrap().as_slice())
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
    fn responses_rewrite_preserves_structured_input_and_instructions() {
        let payload = serde_json::json!({
            "instructions": "System rule",
            "input": [
                {"role": "developer", "content": [{"text": "Dev text"}]},
                {"role": "user", "content": [{"text": "User text"}]}
            ]
        });
        let rewritten =
            rewrite_engine_responses_request(serde_json::to_vec(&payload).unwrap().as_slice())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            value["instructions"],
            Value::String("System rule".to_string())
        );
        assert_eq!(value["input"], payload["input"]);
    }

    #[test]
    fn openai_responses_rewrite_flattens_namespace_tools() {
        let payload = serde_json::json!({
            "tools": [
                {"type":"web_search","search_context_size":"medium"},
                {
                    "type": "namespace",
                    "tools": [
                        {"type":"function","name":"exec_command","description":"run","parameters":{"type":"object"}},
                        {"type":"function","name":"spawn_agent","parameters":{"type":"object"}}
                    ]
                }
            ]
        });
        let rewritten =
            rewrite_openai_responses_request(serde_json::to_vec(&payload).unwrap().as_slice())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            value["tools"],
            serde_json::json!([
                {"type":"web_search","search_context_size":"medium"},
                {"type":"function","name":"exec_command","description":"run","parameters":{"type":"object"}}
            ])
        );
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
        let rewritten =
            rewrite_responses_to_gpt_oss_completion(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "openai/gpt-oss-20b");
        assert_eq!(value["max_tokens"], 333);
        assert!(value["prompt"].as_str().unwrap().contains("System rules"));
        assert!(value["prompt"].as_str().unwrap().contains("Do the thing"));
        assert!(value["prompt"]
            .as_str()
            .unwrap()
            .contains("<|start|>assistant"));
    }

    #[test]
    fn translates_responses_request_to_glm_chat_without_thinking_by_default() {
        let payload = serde_json::json!({
            "model": "zai-org/GLM-4.7-Flash",
            "instructions": "System rules",
            "input": [
                {"role":"user","content":[{"text":"Reply with CTOX_OK and nothing else."}]}
            ],
            "max_output_tokens": 64
        });
        let rewritten =
            rewrite_responses_to_glm_chat_completions(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "zai-org/GLM-4.7-Flash");
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["enable_thinking"], false);
        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn translates_responses_request_to_qwen_chat_without_thinking_by_default() {
        let payload = serde_json::json!({
            "model": "Qwen/Qwen3.5-4B",
            "instructions": "System rules",
            "input": [
                {"role":"user","content":[{"text":"Reply with CTOX_OK and nothing else."}]}
            ],
            "max_output_tokens": 64
        });
        let rewritten =
            rewrite_responses_to_qwen_chat_completions(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "Qwen/Qwen3.5-4B");
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["enable_thinking"], false);
    }

    #[test]
    fn translates_responses_request_to_nemotron_chat_without_thinking_by_default() {
        let payload = serde_json::json!({
            "model": "nvidia/Nemotron-Cascade-2-30B-A3B",
            "instructions": "System rules",
            "input": [
                {"role":"user","content":[{"text":"Reply with CTOX_OK and nothing else."}]}
            ],
            "max_output_tokens": 64
        });
        let rewritten =
            rewrite_responses_to_nemotron_chat_completions(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "nvidia/Nemotron-Cascade-2-30B-A3B");
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["enable_thinking"], false);
    }

    #[test]
    fn rewrites_nemotron_think_tags_back_to_responses_reasoning() {
        let payload = serde_json::json!({
            "id":"abc",
            "created":42,
            "model":"nvidia/Nemotron-Cascade-2-30B-A3B",
            "choices":[{"message":{"content":"<think>step by step</think>\nCTOX_OK"}}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten = rewrite_nemotron_chat_completions_to_responses(
            &serde_json::to_vec(&payload).unwrap(),
            None,
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "CTOX_OK");
        assert_eq!(value["reasoning"], "step by step");
        assert_eq!(value["output"][0]["content"][0]["text"], "CTOX_OK");
    }

    #[test]
    fn translates_responses_request_to_qwen_chat_with_explicit_reasoning() {
        let payload = serde_json::json!({
            "model": "Qwen/Qwen3.5-4B",
            "input": [
                {"role":"user","content":[{"text":"Solve this carefully."}]}
            ],
            "reasoning": {"effort":"high"}
        });
        let rewritten =
            rewrite_responses_to_qwen_chat_completions(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["enable_thinking"], true);
    }

    #[test]
    fn translates_responses_request_to_glm_chat_with_explicit_reasoning() {
        let payload = serde_json::json!({
            "model": "zai-org/GLM-4.7-Flash",
            "input": [
                {"role":"user","content":[{"text":"Solve this carefully."}]}
            ],
            "reasoning": {"effort":"high"}
        });
        let rewritten =
            rewrite_responses_to_glm_chat_completions(&serde_json::to_vec(&payload).unwrap())
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["enable_thinking"], true);
        assert_eq!(value["reasoning_effort"], "high");
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
        let rewritten =
            rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "CTOX_OK");
        assert_eq!(value["output"][0]["content"][0]["text"], "CTOX_OK");
        assert_eq!(value["usage"]["input_tokens"], 11);
        assert_eq!(value["usage"]["output_tokens"], 7);
    }

    #[test]
    fn strips_plaintext_harmony_channel_leakage_from_gpt_oss() {
        let payload = serde_json::json!({
            "id":"123",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"GPTOSS_OKanalysisThe user says assistantfinalGPTOSS_OK"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten =
            rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "GPTOSS_OK");
        assert_eq!(value["output"][0]["content"][0]["text"], "GPTOSS_OK");
    }

    #[test]
    fn strips_real_world_plaintext_harmony_channel_leakage_from_gpt_oss() {
        let payload = serde_json::json!({
            "id":"123",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"GPTOSS_OKassistantanalysisThe user says: \"Reply with GPTOSS_OK and nothing else.\"assistantfinalGPTOSS_OKassistantcommentaryto=functions.exec_command{\"cmd\":\"printf GPTOSS_OK\"}"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten =
            rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "GPTOSS_OK");
        assert_eq!(value["output"][0]["content"][0]["text"], "GPTOSS_OK");
    }

    #[test]
    fn strips_real_world_plaintext_harmony_channel_leakage_with_trailing_assistant() {
        let payload = serde_json::json!({
            "id":"123",
            "created":42,
            "model":"openai/gpt-oss-20b",
            "choices":[{"text":"GPTOSS_OKassistantanalysisThe user says: \"Reply with GPTOSS_OK and nothing else.\" So we should output exactly \"GPTOSS_OK\" with no other text.assistantfinalGPTOSS_OKassistantcommentaryWe have complied.assistantanalysisWe are done.assistantfinalGPTOSS_OKassistant"}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
        });
        let rewritten =
            rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "GPTOSS_OK");
        assert_eq!(value["output"][0]["content"][0]["text"], "GPTOSS_OK");
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
        let rewritten =
            rewrite_gpt_oss_completion_to_responses(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
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
        let rewritten =
            rewrite_gpt_oss_completion_to_sse(&serde_json::to_vec(&payload).unwrap(), None)
                .unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.contains("\"type\":\"response.created\""));
        assert!(text.contains("\"type\":\"response.output_item.done\""));
        assert!(text.contains("\"CTOX_STREAM_OK\""));
        assert!(text.contains("\"type\":\"response.completed\""));
    }

    #[test]
    fn emits_added_and_done_frames_for_web_search_calls() {
        let payload = serde_json::json!({
            "id": "resp_ws",
            "model": "openai/gpt-oss-20b",
            "output": [{
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "weather berlin"
                }
            }]
        });
        let rewritten =
            rewrite_responses_payload_to_sse(&serde_json::to_vec(&payload).unwrap()).unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.contains("\"type\":\"response.output_item.added\""));
        assert!(text.contains("\"type\":\"web_search_call\""));
        assert!(text.contains("\"id\":\"ws_1\""));
        assert!(text.contains("\"status\":\"in_progress\""));
        assert!(text.contains("\"type\":\"response.output_item.done\""));
        assert!(text.contains("\"query\":\"weather berlin\""));
    }
}
