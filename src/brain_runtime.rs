use crate::agentic::enforce_kleinhirn_ready;
use crate::agentic::wait_for_kleinhirn_startup_ready;
use crate::contracts::load_census;
use crate::contracts::find_local_kleinhirn_candidate;
use crate::contracts::load_model_policy;
use crate::contracts::recommended_browser_vision_kleinhirn;
use crate::contracts::recommended_kleinhirn;
use crate::contracts::BrainModel;
use crate::contracts::Paths;
use anyhow::Context;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;

#[derive(Debug, Clone, Default)]
pub struct KleinhirnRuntimeSnapshot {
    pub policy_model: Option<String>,
    pub runtime_model: Option<String>,
    pub official_label: Option<String>,
    pub adapter: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KleinhirnUpgradeOutcome {
    pub changed: bool,
    pub restarted: bool,
    pub summary: String,
    pub previous_runtime_model: Option<String>,
    pub current_runtime_model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GrosshirnActivationPrepOutcome {
    pub changed: bool,
    pub configured: bool,
    pub api_key_from_message: bool,
    pub target_model: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StartupCalibrationCandidate {
    max_batch_size: Option<u64>,
    max_seq_len: Option<u64>,
    pa_context_len: Option<u64>,
    pa_cache_type: Option<String>,
    paged_attn_mode: Option<String>,
    force_auto_device_mapping: bool,
}

pub fn load_kleinhirn_runtime_snapshot(paths: &Paths) -> Option<KleinhirnRuntimeSnapshot> {
    let env_map = load_kleinhirn_env_map(paths).ok()?;
    Some(KleinhirnRuntimeSnapshot {
        policy_model: env_map.get("CTO_AGENT_KLEINHIRN_MODEL").cloned(),
        runtime_model: env_map.get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL").cloned(),
        official_label: env_map.get("CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL").cloned(),
        adapter: env_map.get("CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER").cloned(),
    })
}

pub fn local_kleinhirn_upgrade_available(paths: &Paths) -> bool {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = recommended_kleinhirn(&policy, &census);
    let current = load_kleinhirn_runtime_snapshot(paths).unwrap_or_default();
    let selected_runtime = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    current.runtime_model.as_deref() != Some(selected_runtime.as_str())
        || current.policy_model.as_deref() != Some(selected.model_id.as_str())
}

pub fn local_browser_vision_kleinhirn_upgrade_available(paths: &Paths) -> bool {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let Some(selected) = recommended_browser_vision_kleinhirn(&policy, &census) else {
        return false;
    };
    let current = load_kleinhirn_runtime_snapshot(paths).unwrap_or_default();
    let selected_runtime = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    current.runtime_model.as_deref() != Some(selected_runtime.as_str())
        || current.policy_model.as_deref() != Some(selected.model_id.as_str())
}

pub fn grosshirn_runtime_configured(paths: &Paths) -> bool {
    let env_map = load_kleinhirn_env_map(paths).unwrap_or_default();
    runtime_env_value("CTO_AGENT_GROSSHIRN_API_KEY", &env_map)
        .or_else(|| runtime_env_value("OPENAI_API_KEY", &env_map))
        .is_some()
}

pub fn prepare_grosshirn_activation_from_message(
    paths: &Paths,
    message: &str,
) -> anyhow::Result<GrosshirnActivationPrepOutcome> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let original_text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    let mut env_map = parse_env_file(&original_text);
    let existing_api_key = runtime_env_value("CTO_AGENT_GROSSHIRN_API_KEY", &env_map)
        .or_else(|| runtime_env_value("OPENAI_API_KEY", &env_map));
    let extracted_api_key = extract_openai_api_key(message);
    let api_key_from_message = extracted_api_key.is_some();
    let chosen_api_key = extracted_api_key.or(existing_api_key.clone());
    let target_model = extract_requested_grosshirn_model(message)
        .or_else(|| runtime_env_value("CTO_AGENT_GROSSHIRN_MODEL", &env_map))
        .unwrap_or_else(|| "gpt-5.4".to_string());
    let owner_wants_openai_grosshirn = {
        let lowered = message.to_lowercase();
        lowered.contains("openai") || lowered.contains("gpt-5.4") || lowered.contains("gpt 5.4")
    };
    let base_url = if owner_wants_openai_grosshirn {
        runtime_env_value("CTO_AGENT_GROSSHIRN_BASE_URL", &env_map)
            .filter(|value| value.starts_with("https://"))
            .or_else(|| {
                std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| value.starts_with("https://"))
            })
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    } else {
        runtime_env_value("CTO_AGENT_GROSSHIRN_BASE_URL", &env_map)
            .or_else(|| {
                std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| value.starts_with("https://"))
            })
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    };
    let adapter = runtime_env_value("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER", &env_map)
        .unwrap_or_else(|| "openai_responses".to_string());
    let mut changed = false;

    if let Some(api_key) = chosen_api_key.as_ref() {
        if env_map
            .get("CTO_AGENT_GROSSHIRN_API_KEY")
            .map(|value| value != api_key)
            .unwrap_or(true)
        {
            env_map.insert(
                "CTO_AGENT_GROSSHIRN_API_KEY".to_string(),
                api_key.to_string(),
            );
            changed = true;
        }
    }

    if env_map
        .get("CTO_AGENT_GROSSHIRN_MODEL")
        .map(|value| value != &target_model)
        .unwrap_or(true)
    {
        env_map.insert(
            "CTO_AGENT_GROSSHIRN_MODEL".to_string(),
            target_model.clone(),
        );
        changed = true;
    }
    if env_map
        .get("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER")
        .map(|value| value != &adapter)
        .unwrap_or(true)
    {
        env_map.insert("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER".to_string(), adapter);
        changed = true;
    }
    if env_map
        .get("CTO_AGENT_GROSSHIRN_BASE_URL")
        .map(|value| value != &base_url)
        .unwrap_or(true)
    {
        env_map.insert("CTO_AGENT_GROSSHIRN_BASE_URL".to_string(), base_url);
        changed = true;
    }

    if changed {
        write_kleinhirn_env(&env_path, &env_map)?;
    }

    let configured = chosen_api_key
        .as_ref()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let summary = if configured {
        if api_key_from_message {
            format!(
                "Grosshirn-Aktivierung vorbereitet: API-Credential aus Owner-Signal uebernommen, Zielmodell {} gesetzt und Runtime-Konfiguration aktualisiert.",
                target_model
            )
        } else if changed {
            format!(
                "Grosshirn-Aktivierung vorbereitet: bestehende Runtime-Credentials und Zielmodell {} wurden fuer den naechsten bounded Schritt vereinheitlicht.",
                target_model
            )
        } else {
            format!(
                "Grosshirn-Aktivierung vorbereitet: Credentials und Zielmodell {} waren bereits in der Runtime vorhanden.",
                target_model
            )
        }
    } else {
        format!(
            "Grosshirn-Aktivierung erkannt, aber noch ohne API-Credential. Zielmodell {} wurde vorgemerkt; der naechste bounded Schritt muss fehlende Credentials explizit einfordern.",
            target_model
        )
    };

    Ok(GrosshirnActivationPrepOutcome {
        changed,
        configured,
        api_key_from_message,
        target_model,
        summary,
    })
}

pub fn apply_recommended_kleinhirn_upgrade(
    paths: &Paths,
) -> anyhow::Result<KleinhirnUpgradeOutcome> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = recommended_kleinhirn(&policy, &census);
    apply_selected_kleinhirn_upgrade(paths, selected, &census)
}

pub fn apply_recommended_browser_vision_kleinhirn_upgrade(
    paths: &Paths,
) -> anyhow::Result<KleinhirnUpgradeOutcome> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let Some(selected) = recommended_browser_vision_kleinhirn(&policy, &census) else {
        anyhow::bail!("no supported local vision-capable kleinhirn candidate available");
    };
    apply_selected_kleinhirn_upgrade(paths, selected, &census)
}

pub fn apply_targeted_kleinhirn_upgrade(
    paths: &Paths,
    target_model: Option<&str>,
) -> anyhow::Result<KleinhirnUpgradeOutcome> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = match target_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(needle) => find_local_kleinhirn_candidate(&policy, needle).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown local kleinhirn target '{}'; choose one of the curated local candidates",
                needle
            )
        })?,
        None => recommended_kleinhirn(&policy, &census),
    };
    apply_selected_kleinhirn_upgrade(paths, selected, &census)
}

fn apply_selected_kleinhirn_upgrade(
    paths: &Paths,
    selected: &BrainModel,
    census: &crate::contracts::SystemCensus,
) -> anyhow::Result<KleinhirnUpgradeOutcome> {
    let selected_runtime = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let original_text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    let env_map = parse_env_file(&original_text);
    let previous = KleinhirnRuntimeSnapshot {
        policy_model: env_map.get("CTO_AGENT_KLEINHIRN_MODEL").cloned(),
        runtime_model: env_map.get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL").cloned(),
        official_label: env_map.get("CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL").cloned(),
        adapter: env_map.get("CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER").cloned(),
    };

    if previous.runtime_model.as_deref() == Some(selected_runtime.as_str())
        && previous.policy_model.as_deref() == Some(selected.model_id.as_str())
    {
        return Ok(KleinhirnUpgradeOutcome {
            changed: false,
            restarted: false,
            summary: format!(
                "Lokales Kleinhirn laeuft bereits auf {} ({})",
                selected.official_label, selected_runtime
            ),
            previous_runtime_model: previous.runtime_model.clone(),
            current_runtime_model: previous.runtime_model,
        });
    }

    prefetch_selected_model_snapshot(selected)?;
    let startup_candidates = build_startup_calibration_candidates(selected, census);
    let mut failures = Vec::new();

    for candidate in &startup_candidates {
        let mut candidate_env_map = env_map.clone();
        apply_selected_model_to_env(&mut candidate_env_map, selected, census, candidate);
        write_kleinhirn_env(&env_path, &candidate_env_map)?;

        let upgrade_result = restart_kleinhirn_runtime(paths)
            .and_then(|_| wait_for_kleinhirn_startup_ready(paths))
            .and_then(|_| enforce_kleinhirn_ready(paths));

        match upgrade_result {
            Ok(()) => {
                let context_note = candidate
                    .max_seq_len
                    .map(|value| format!(" maxSeqLen={value}"))
                    .unwrap_or_default();
                return Ok(KleinhirnUpgradeOutcome {
                    changed: true,
                    restarted: true,
                    summary: format!(
                        "Lokales Kleinhirn auf {} ({}) umgestellt und erfolgreich neu gestartet.{}",
                        selected.official_label, selected_runtime, context_note
                    ),
                    previous_runtime_model: previous.runtime_model,
                    current_runtime_model: Some(selected_runtime),
                });
            }
            Err(err) => {
                failures.push(format!(
                    "maxBatchSize={:?}, maxSeqLen={:?}, paContext={:?}, paCacheType={:?}, pagedAttn={:?}, autoTp={} => {}",
                    candidate.max_batch_size,
                    candidate.max_seq_len,
                    candidate.pa_context_len,
                    candidate.pa_cache_type,
                    candidate.paged_attn_mode,
                    candidate.force_auto_device_mapping,
                    err
                ));
            }
        }
    }

    write_raw_env(&env_path, &original_text)?;
    let rollback_error = restart_kleinhirn_runtime(paths)
        .and_then(|_| wait_for_kleinhirn_startup_ready(paths))
        .and_then(|_| enforce_kleinhirn_ready(paths))
        .err()
        .map(|rollback| rollback.to_string());
    match rollback_error {
        Some(rollback) => anyhow::bail!(
            "local kleinhirn upgrade to {} failed across {} startup candidates: {}; rollback also failed: {}",
            selected_runtime,
            failures.len(),
            failures.join(" | "),
            rollback
        ),
        None => anyhow::bail!(
            "local kleinhirn upgrade to {} failed across {} startup candidates and was rolled back: {}",
            selected_runtime,
            failures.len(),
            failures.join(" | ")
        ),
    }
}

fn prefetch_selected_model_snapshot(selected: &BrainModel) -> anyhow::Result<()> {
    let runtime_model = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    if !runtime_model.contains('/') {
        return Ok(());
    }

    let python_code = r#"
from huggingface_hub import snapshot_download
import sys

repo_id = sys.argv[1]
snapshot_download(repo_id=repo_id, resume_download=True)
"#;

    let status = Command::new("python3")
        .arg("-c")
        .arg(python_code)
        .arg(&runtime_model)
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .status()
        .with_context(|| format!("failed to start Hugging Face prefetch for {}", runtime_model))?;

    if !status.success() {
        anyhow::bail!(
            "failed to prefetch model artifacts for {} before local kleinhirn switch",
            runtime_model
        );
    }

    Ok(())
}

fn apply_selected_model_to_env(
    env_map: &mut BTreeMap<String, String>,
    selected: &BrainModel,
    census: &crate::contracts::SystemCensus,
    calibration: &StartupCalibrationCandidate,
) {
    let runtime_model = selected
        .runtime_model_id
        .clone()
        .unwrap_or_else(|| selected.model_id.clone());
    env_map.insert(
        "CTO_AGENT_KLEINHIRN_PROFILE".to_string(),
        profile_name_for_model(&runtime_model),
    );
    env_map.insert(
        "CTO_AGENT_KLEINHIRN_MODEL".to_string(),
        selected.model_id.clone(),
    );
    env_map.insert(
        "CTO_AGENT_KLEINHIRN_RUNTIME_MODEL".to_string(),
        runtime_model,
    );
    env_map.insert(
        "CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL".to_string(),
        selected.official_label.clone(),
    );
    env_map.insert(
        "CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER".to_string(),
        selected
            .agentic_adapter
            .clone()
            .unwrap_or_else(default_adapter_for_model),
    );

    let tune = census
        .model_tune_candidates
        .as_ref()
        .and_then(|items| {
            items.iter().find(|candidate| {
                candidate.status.eq_ignore_ascii_case("supported")
                    && (candidate.model_id.eq_ignore_ascii_case(&selected.model_id)
                        || selected
                            .runtime_model_id
                            .as_deref()
                            .map(|value| candidate.model_id.eq_ignore_ascii_case(value))
                            .unwrap_or(false))
            })
        });
    let reliable_layout_tune =
        tune.filter(|candidate| candidate.max_context_tokens.unwrap_or(0) > 0);
    match tune.and_then(|candidate| candidate.recommended_isq.clone()) {
        Some(value) if !value.trim().is_empty() => {
            env_map.insert("CTO_AGENT_KLEINHIRN_ISQ".to_string(), value);
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_ISQ");
        }
    }

    match selected.startup_max_seqs {
        Some(value) if value > 0 => {
            env_map.insert("CTO_AGENT_KLEINHIRN_MAX_SEQS".to_string(), value.to_string());
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_MAX_SEQS");
        }
    }

    match calibration.max_batch_size {
        Some(value) if value > 0 => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE".to_string(),
                value.to_string(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE");
        }
    }

    match selected.startup_device_layers_cli.as_deref() {
        _ if selected
            .startup_topology_path
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false) =>
        {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_TOPOLOGY".to_string(),
                selected
                    .startup_topology_path
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
            );
            env_map.remove("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS");
            env_map.remove("CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS");
        }
        Some(value) if !value.trim().is_empty() => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_DEVICE_LAYERS".to_string(),
                value.trim().to_string(),
            );
            env_map.remove("CTO_AGENT_KLEINHIRN_TOPOLOGY");
            env_map.remove("CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS");
        }
        _ if calibration.force_auto_device_mapping || selected.prefer_auto_device_mapping => {
            env_map.remove("CTO_AGENT_KLEINHIRN_TOPOLOGY");
            env_map.remove("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS");
            env_map.remove("CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS");
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_TOPOLOGY");
            match reliable_layout_tune.and_then(|candidate| candidate.device_layers_cli.clone()) {
                Some(value) if !value.trim().is_empty() => {
                    env_map.insert("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS".to_string(), value);
                }
                _ => {
                    env_map.remove("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS");
                }
            }
        }
    }

    match calibration.max_seq_len {
        Some(value) if value > 0 => {
            env_map.insert("CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN".to_string(), value.to_string());
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN");
        }
    }

    match calibration.pa_context_len {
        Some(value) if value > 0 => {
            env_map.insert("CTO_AGENT_KLEINHIRN_PA_CTXT_LEN".to_string(), value.to_string());
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_PA_CTXT_LEN");
        }
    }

    match calibration.paged_attn_mode.as_deref() {
        Some("auto" | "on" | "off") => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE".to_string(),
                calibration.paged_attn_mode.clone().unwrap_or_default(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE");
        }
    }

    match calibration.pa_cache_type.as_deref() {
        Some("auto" | "f8e4m3") => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE".to_string(),
                calibration.pa_cache_type.clone().unwrap_or_default(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE");
        }
    }

    match selected.startup_chat_template_path.as_deref() {
        Some(value) if !value.trim().is_empty() => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE".to_string(),
                value.trim().to_string(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE");
        }
    }

    match selected.startup_jinja_explicit_path.as_deref() {
        Some(value) if !value.trim().is_empty() => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT".to_string(),
                value.trim().to_string(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT");
        }
    }

    match selected.startup_tokenizer_json_path.as_deref() {
        Some(value) if !value.trim().is_empty() => {
            env_map.insert(
                "CTO_AGENT_KLEINHIRN_TOKENIZER_JSON".to_string(),
                value.trim().to_string(),
            );
        }
        _ => {
            env_map.remove("CTO_AGENT_KLEINHIRN_TOKENIZER_JSON");
        }
    }

    env_map.remove("CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN");
    env_map.remove("CTO_AGENT_KLEINHIRN_PA_GPU_MEM");
    env_map.remove("CTO_AGENT_KLEINHIRN_PA_GPU_MEM_USAGE");
}

fn build_startup_calibration_candidates(
    selected: &BrainModel,
    census: &crate::contracts::SystemCensus,
) -> Vec<StartupCalibrationCandidate> {
    let tune = census
        .model_tune_candidates
        .as_ref()
        .and_then(|items| {
            items.iter().find(|candidate| {
                candidate.status.eq_ignore_ascii_case("supported")
                    && (candidate.model_id.eq_ignore_ascii_case(&selected.model_id)
                        || selected
                            .runtime_model_id
                            .as_deref()
                            .map(|value| candidate.model_id.eq_ignore_ascii_case(value))
                            .unwrap_or(false))
            })
        });
    let tuned_max_context = tune.and_then(|candidate| candidate.max_context_tokens);
    let multi_gpu_auto_tp =
        selected.prefer_auto_device_mapping && census.gpu_count.unwrap_or(0) > 1;

    if multi_gpu_auto_tp {
        let bootstrap_cap = multi_gpu_bootstrap_max_seq_len(selected);
        let mut lengths = Vec::new();
        if let Some(tuned) = tuned_max_context.filter(|value| *value > 0) {
            lengths.push(bootstrap_cap.map(|cap| tuned.min(cap)).unwrap_or(tuned));
        }
        if let Some(policy_cap) = selected.startup_max_seq_len.filter(|value| *value > 0) {
            lengths.push(bootstrap_cap.map(|cap| policy_cap.min(cap)).unwrap_or(policy_cap));
        }
        if let Some(policy_cap) = selected.startup_pa_context_len.filter(|value| *value > 0) {
            lengths.push(bootstrap_cap.map(|cap| policy_cap.min(cap)).unwrap_or(policy_cap));
        }
        for fallback in [131_072_u64, 65_536, 32_768, 16_384, 8_192, 4_096] {
            if tuned_max_context.map(|limit| fallback <= limit).unwrap_or(true)
                && bootstrap_cap.map(|cap| fallback <= cap).unwrap_or(true)
            {
                lengths.push(fallback);
            }
        }
        lengths.sort_unstable();
        lengths.dedup();
        lengths.reverse();
        if lengths.is_empty() {
            lengths.push(8_192);
        }
        return lengths
            .into_iter()
            .map(|length| StartupCalibrationCandidate {
                max_batch_size: selected.startup_max_batch_size.or(Some(1)),
                max_seq_len: Some(length),
                pa_context_len: Some(length),
                pa_cache_type: selected
                    .startup_pa_cache_type
                    .clone()
                    .or_else(|| Some("f8e4m3".to_string())),
                paged_attn_mode: Some("on".to_string()),
                force_auto_device_mapping: true,
            })
            .collect();
    }

    let effective_startup_max_seq_len = match (selected.startup_max_seq_len, tuned_max_context) {
        (_, Some(tuned)) if tuned > 0 => Some(tuned),
        (Some(policy_cap), _) => Some(policy_cap),
        (None, Some(tuned)) if tuned > 0 => Some(tuned),
        _ => None,
    };
    let effective_pa_context_len = match (selected.startup_pa_context_len, effective_startup_max_seq_len)
    {
        (Some(policy_cap), Some(seq_cap)) => Some(policy_cap.min(seq_cap)),
        (Some(policy_cap), None) => Some(policy_cap),
        (None, Some(seq_cap)) if selected.prefer_auto_device_mapping => Some(seq_cap),
        _ => None,
    };

    vec![StartupCalibrationCandidate {
        max_batch_size: selected.startup_max_batch_size.or(Some(1)),
        max_seq_len: effective_startup_max_seq_len,
        pa_context_len: effective_pa_context_len,
        pa_cache_type: selected.startup_pa_cache_type.clone(),
        paged_attn_mode: selected.startup_paged_attn_mode.clone(),
        force_auto_device_mapping: selected.prefer_auto_device_mapping,
    }]
}

fn multi_gpu_bootstrap_max_seq_len(selected: &BrainModel) -> Option<u64> {
    let env_cap = std::env::var("CTO_AGENT_MULTI_GPU_BOOTSTRAP_MAX_SEQ_LEN")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0);
    if env_cap.is_some() {
        return env_cap;
    }
    if selected.supports_vision {
        return Some(131_072);
    }
    None
}

fn restart_kleinhirn_runtime(paths: &Paths) -> anyhow::Result<()> {
    if try_restart_with_systemd() {
        return Ok(());
    }
    try_restart_with_local_script(paths)
}

fn try_restart_with_systemd() -> bool {
    Command::new("systemctl")
        .args(["--user", "restart", "cto-kleinhirn.service"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn try_restart_with_local_script(paths: &Paths) -> anyhow::Result<()> {
    let runtime_dir = paths.root.join("runtime");
    fs::create_dir_all(runtime_dir.join("logs"))
        .with_context(|| format!("failed to create {}", runtime_dir.display()))?;
    let pid_path = runtime_dir.join("kleinhirn.pid");
    if let Ok(existing) = fs::read_to_string(&pid_path)
        && let Ok(pid) = existing.trim().parse::<i32>()
    {
        let _ = Command::new("kill").arg("-TERM").arg(pid.to_string()).status();
    }

    let log_path = runtime_dir.join("logs/kleinhirn.log");
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("failed to clone {}", log_path.display()))?;
    let child = Command::new("sh")
        .arg(paths.root.join("scripts/run_kleinhirn.sh"))
        .current_dir(&paths.root)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .context("failed to spawn local kleinhirn runtime")?;
    fs::write(&pid_path, format!("{}\n", child.id()))
        .with_context(|| format!("failed to write {}", pid_path.display()))?;
    Ok(())
}

fn parse_env_file(text: &str) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        env_map.insert(key.trim().to_string(), unquote_env_value(raw_value.trim()));
    }
    env_map
}

fn runtime_env_value(key: &str, env_map: &BTreeMap<String, String>) -> Option<String> {
    std::env::var(key)
        .ok()
        .or_else(|| env_map.get(key).cloned())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_openai_api_key(message: &str) -> Option<String> {
    for raw in message.split_whitespace() {
        let candidate = raw.trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | '`' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
            )
        });
        if (candidate.starts_with("sk-proj-") || candidate.starts_with("sk-"))
            && candidate.len() > 20
        {
            return Some(candidate.to_string());
        }
    }
    None
}

fn extract_requested_grosshirn_model(message: &str) -> Option<String> {
    let lowered = message.to_lowercase();
    if lowered.contains("gpt-5.4-pro") || lowered.contains("gpt 5.4 pro") {
        return Some("gpt-5.4-pro".to_string());
    }
    if lowered.contains("gpt-5.4") || lowered.contains("gpt 5.4") {
        return Some("gpt-5.4".to_string());
    }
    None
}

fn unquote_env_value(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'))
    {
        value[1..value.len() - 1].replace("'\\''", "'")
    } else {
        value.to_string()
    }
}

fn write_kleinhirn_env(path: &std::path::Path, env_map: &BTreeMap<String, String>) -> anyhow::Result<()> {
    let mut rendered = String::new();
    for (key, value) in env_map {
        rendered.push_str(key);
        rendered.push('=');
        rendered.push_str(&shell_quote(value));
        rendered.push('\n');
    }
    write_raw_env(path, &rendered)
}

fn write_raw_env(path: &std::path::Path, text: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp_path = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = File::create(&tmp_path)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn profile_name_for_model(model_id: &str) -> String {
    let lowered = model_id.to_lowercase();
    if lowered.contains("qwen3.5") {
        "qwen35".to_string()
    } else {
        "gpt_oss".to_string()
    }
}

fn default_adapter_for_model() -> String {
    "openai_compatible_chat".to_string()
}

fn load_kleinhirn_env_map(paths: &Paths) -> anyhow::Result<BTreeMap<String, String>> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    Ok(parse_env_file(&text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{BrainModel, ModelTuneCandidate, Paths, SystemCensus};
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard(Option<OsString>);

    impl EnvGuard {
        fn set_cto_root(root: &std::path::Path) -> Self {
            let previous = std::env::var_os("CTO_AGENT_ROOT");
            unsafe {
                std::env::set_var("CTO_AGENT_ROOT", root);
            }
            Self(previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                unsafe {
                    std::env::set_var("CTO_AGENT_ROOT", previous);
                }
            } else {
                unsafe {
                    std::env::remove_var("CTO_AGENT_ROOT");
                }
            }
        }
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cto_agent_brain_runtime_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn qwen35_model() -> BrainModel {
        BrainModel {
            role: "kleinhirn_install_alternative".to_string(),
            provider: "qwen".to_string(),
            model_id: "Qwen3.5-35B-A3B".to_string(),
            runtime_model_id: Some("Qwen/Qwen3.5-35B-A3B".to_string()),
            official_label: "Qwen3.5 35B A3B".to_string(),
            agentic_adapter: Some("openai_compatible_chat".to_string()),
            reasoning_effort: "low".to_string(),
            deployment_mode: "local_or_self_hosted".to_string(),
            purpose: "vision".to_string(),
            supports_vision: true,
            min_cpu_threads: Some(16),
            min_memory_gb: Some(48),
            min_gpu_count: Some(3),
            min_total_gpu_memory_gb: Some(48),
            min_single_gpu_memory_gb: Some(12),
            startup_max_seqs: Some(1),
            startup_max_batch_size: Some(1),
            startup_max_seq_len: Some(8192),
            startup_pa_context_len: Some(8192),
            startup_pa_cache_type: Some("f8e4m3".to_string()),
            startup_paged_attn_mode: Some("auto".to_string()),
            startup_chat_template_path: None,
            startup_jinja_explicit_path: None,
            startup_tokenizer_json_path: None,
            startup_topology_path: None,
            startup_device_layers_cli: None,
            prefer_auto_device_mapping: true,
        }
    }

    #[test]
    fn multi_gpu_auto_tp_calibration_uses_descending_contexts() {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        unsafe {
            std::env::remove_var("CTO_AGENT_MULTI_GPU_BOOTSTRAP_MAX_SEQ_LEN");
        }
        let selected = qwen35_model();
        let census = SystemCensus {
            gpu_count: Some(5),
            model_tune_candidates: Some(vec![ModelTuneCandidate {
                model_id: "Qwen3.5-35B-A3B".to_string(),
                official_label: "Qwen3.5 35B A3B".to_string(),
                status: "supported".to_string(),
                recommended_isq: Some("Q6K".to_string()),
                device_layers_cli: Some("0:30;1:10".to_string()),
                max_context_tokens: Some(262_144),
                note: None,
            }]),
            ..Default::default()
        };

        let candidates = build_startup_calibration_candidates(&selected, &census);
        assert!(candidates.len() >= 4);
        assert_eq!(candidates[0].max_seq_len, Some(131_072));
        assert_eq!(candidates[0].max_batch_size, Some(1));
        assert_eq!(candidates[0].pa_cache_type.as_deref(), Some("f8e4m3"));
        assert_eq!(candidates[1].max_seq_len, Some(65_536));
        assert_eq!(candidates.last().and_then(|item| item.max_seq_len), Some(4_096));
        assert!(candidates.iter().all(|item| item.force_auto_device_mapping));
        assert!(candidates
            .iter()
            .all(|item| item.paged_attn_mode.as_deref() == Some("on")));
    }

    #[test]
    fn multi_gpu_auto_tp_does_not_persist_tune_device_layers() {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        unsafe {
            std::env::remove_var("CTO_AGENT_MULTI_GPU_BOOTSTRAP_MAX_SEQ_LEN");
        }
        let selected = qwen35_model();
        let census = SystemCensus {
            gpu_count: Some(5),
            model_tune_candidates: Some(vec![ModelTuneCandidate {
                model_id: "Qwen3.5-35B-A3B".to_string(),
                official_label: "Qwen3.5 35B A3B".to_string(),
                status: "supported".to_string(),
                recommended_isq: Some("Q6K".to_string()),
                device_layers_cli: Some("0:30;1:10".to_string()),
                max_context_tokens: Some(262_144),
                note: None,
            }]),
            ..Default::default()
        };
        let calibration = build_startup_calibration_candidates(&selected, &census)
            .into_iter()
            .next()
            .expect("calibration candidate");
        let mut env_map = BTreeMap::new();

        apply_selected_model_to_env(&mut env_map, &selected, &census, &calibration);

        assert_eq!(
            env_map.get("CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE").map(String::as_str),
            Some("on")
        );
        assert_eq!(
            env_map
                .get("CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE")
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(
            env_map
                .get("CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE")
                .map(String::as_str),
            Some("f8e4m3")
        );
        assert_eq!(
            env_map.get("CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN").map(String::as_str),
            Some("131072")
        );
        assert!(!env_map.contains_key("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS"));
        assert!(!env_map.contains_key("CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS"));
    }

    #[test]
    fn multi_gpu_bootstrap_cap_can_be_overridden_by_env() {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        unsafe {
            std::env::set_var("CTO_AGENT_MULTI_GPU_BOOTSTRAP_MAX_SEQ_LEN", "65536");
        }
        let selected = qwen35_model();
        let census = SystemCensus {
            gpu_count: Some(5),
            model_tune_candidates: Some(vec![ModelTuneCandidate {
                model_id: "Qwen3.5-35B-A3B".to_string(),
                official_label: "Qwen3.5 35B A3B".to_string(),
                status: "supported".to_string(),
                recommended_isq: Some("Q6K".to_string()),
                device_layers_cli: Some("0:30;1:10".to_string()),
                max_context_tokens: Some(262_144),
                note: None,
            }]),
            ..Default::default()
        };

        let candidates = build_startup_calibration_candidates(&selected, &census);
        assert_eq!(candidates[0].max_seq_len, Some(65_536));
        unsafe {
            std::env::remove_var("CTO_AGENT_MULTI_GPU_BOOTSTRAP_MAX_SEQ_LEN");
        }
    }

    #[test]
    fn explicit_topology_override_beats_auto_tp_and_device_layers() {
        let mut selected = qwen35_model();
        selected.startup_topology_path = Some("/tmp/qwen35-topology.json".to_string());
        let census = SystemCensus {
            gpu_count: Some(5),
            ..Default::default()
        };
        let calibration = StartupCalibrationCandidate {
            max_batch_size: Some(1),
            max_seq_len: Some(65_536),
            pa_context_len: Some(65_536),
            pa_cache_type: Some("f8e4m3".to_string()),
            paged_attn_mode: Some("on".to_string()),
            force_auto_device_mapping: true,
        };
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTO_AGENT_KLEINHIRN_DEVICE_LAYERS".to_string(),
            "0:30;1:10".to_string(),
        );

        apply_selected_model_to_env(&mut env_map, &selected, &census, &calibration);

        assert_eq!(
            env_map.get("CTO_AGENT_KLEINHIRN_TOPOLOGY").map(String::as_str),
            Some("/tmp/qwen35-topology.json")
        );
        assert!(!env_map.contains_key("CTO_AGENT_KLEINHIRN_DEVICE_LAYERS"));
        assert!(!env_map.contains_key("CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS"));
    }

    #[test]
    fn non_multi_gpu_prefers_tuned_max_context_for_gpt_oss() {
        let selected = BrainModel {
            role: "kleinhirn".to_string(),
            provider: "openai".to_string(),
            model_id: "gpt-oss-20b".to_string(),
            runtime_model_id: Some("openai/gpt-oss-20b".to_string()),
            official_label: "GPT-OSS 20B".to_string(),
            agentic_adapter: Some("openai_compatible_chat".to_string()),
            reasoning_effort: "low".to_string(),
            deployment_mode: "local_or_self_hosted".to_string(),
            purpose: "always-on".to_string(),
            supports_vision: false,
            min_cpu_threads: Some(8),
            min_memory_gb: Some(16),
            min_gpu_count: Some(1),
            min_total_gpu_memory_gb: Some(12),
            min_single_gpu_memory_gb: Some(12),
            startup_max_seqs: Some(1),
            startup_max_batch_size: Some(1),
            startup_max_seq_len: Some(8192),
            startup_pa_context_len: None,
            startup_pa_cache_type: None,
            startup_paged_attn_mode: Some("off".to_string()),
            startup_chat_template_path: None,
            startup_jinja_explicit_path: None,
            startup_tokenizer_json_path: None,
            startup_topology_path: None,
            startup_device_layers_cli: None,
            prefer_auto_device_mapping: false,
        };
        let census = SystemCensus {
            gpu_count: Some(1),
            model_tune_candidates: Some(vec![ModelTuneCandidate {
                model_id: "gpt-oss-20b".to_string(),
                official_label: "GPT-OSS 20B".to_string(),
                status: "supported".to_string(),
                recommended_isq: Some("Q6K".to_string()),
                device_layers_cli: Some("0:22;1:2".to_string()),
                max_context_tokens: Some(131_072),
                note: None,
            }]),
            ..Default::default()
        };

        let calibration = build_startup_calibration_candidates(&selected, &census);
        assert_eq!(calibration.len(), 1);
        assert_eq!(calibration[0].max_seq_len, Some(131_072));
        assert_eq!(calibration[0].paged_attn_mode.as_deref(), Some("off"));
    }

    #[test]
    fn grosshirn_activation_prepares_runtime_from_owner_message() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("grosshirn_prepare");
        std::fs::create_dir_all(root.join("runtime"))?;
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_KLEINHIRN_MODEL='gpt-oss-20b'\n",
        )?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        let outcome = prepare_grosshirn_activation_from_message(
            &paths,
            "Wechsle jetzt auf GPT-5.4 als Grosshirn. Nutze diesen API-Token: sk-proj-test-token-abcdefghijklmnopqrstuvwxyz123456",
        )?;
        let env_text = std::fs::read_to_string(root.join("runtime/kleinhirn.env"))?;

        assert!(outcome.configured);
        assert!(outcome.api_key_from_message);
        assert_eq!(outcome.target_model, "gpt-5.4");
        assert!(env_text.contains("CTO_AGENT_GROSSHIRN_MODEL='gpt-5.4'"));
        assert!(env_text.contains("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER='openai_responses'"));
        assert!(env_text.contains("CTO_AGENT_GROSSHIRN_BASE_URL='https://api.openai.com/v1'"));
        assert!(env_text.contains(
            "CTO_AGENT_GROSSHIRN_API_KEY='sk-proj-test-token-abcdefghijklmnopqrstuvwxyz123456'"
        ));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn grosshirn_runtime_configuration_is_visible_from_env_file() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("grosshirn_configured");
        std::fs::create_dir_all(root.join("runtime"))?;
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_KLEINHIRN_MODEL='gpt-oss-20b'\nCTO_AGENT_GROSSHIRN_API_KEY='sk-proj-test-token-abcdefghijklmnopqrstuvwxyz123456'\n",
        )?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;

        assert!(grosshirn_runtime_configured(&paths));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn openai_grosshirn_activation_resets_invalid_local_base_url() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("grosshirn_base_url_repair");
        std::fs::create_dir_all(root.join("runtime"))?;
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_GROSSHIRN_API_KEY='test'\nCTO_AGENT_GROSSHIRN_BASE_URL='http://127.0.0.1:9/v1'\n",
        )?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;

        let outcome = prepare_grosshirn_activation_from_message(
            &paths,
            "Wechsle jetzt auf GPT-5.4 als Grosshirn mit OpenAI.",
        )?;
        let env_text = std::fs::read_to_string(root.join("runtime/kleinhirn.env"))?;

        assert!(outcome.configured);
        assert!(env_text.contains("CTO_AGENT_GROSSHIRN_BASE_URL='https://api.openai.com/v1'"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
