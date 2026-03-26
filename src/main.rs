use anyhow::Context;
use std::path::PathBuf;

mod backend_manager;
mod channels;
mod chat_runtime;
mod execution_baseline;
mod follow_up;
mod lcm;
mod plan;
mod queue;
mod responses_proxy;
mod runtime_config;
mod runtime_planner;
mod schedule;
mod service;
mod tui;

fn apply_legacy_chat_runtime(root: &std::path::Path, model: &str) -> anyhow::Result<()> {
    let runtime = execution_baseline::runtime_config_for_model(model)?;
    let family_profile = execution_baseline::runtime_profile_for_model(model)?;
    let mut env_map = runtime_config::load_runtime_env_map(root).unwrap_or_default();
    runtime_planner::clear_persisted_chat_plan(root, &mut env_map)?;
    env_map.insert("CTOX_CHAT_SOURCE".to_string(), "local".to_string());
    env_map.insert("CTOX_CHAT_MODEL_BASE".to_string(), runtime.model.clone());
    env_map.insert("CTOX_ACTIVE_MODEL".to_string(), runtime.model.clone());
    env_map.insert("CTOX_CHAT_MODEL".to_string(), runtime.model.clone());
    env_map.remove("CTOX_BOOST_ACTIVE_UNTIL_EPOCH");
    env_map.remove("CTOX_BOOST_REASON");
    env_map.remove("CTOX_CHAT_LOCAL_PRESET");
    env_map.remove("CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN");
    env_map.remove("CTOX_CHAT_MODEL_REALIZED_CONTEXT");
    env_map.remove("CTOX_VLLM_SERVE_REALIZED_MODEL");
    for key in [
        "CTOX_VLLM_SERVE_NO_MMAP",
        "CTOX_VLLM_SERVE_LANGUAGE_MODEL_ONLY",
        "CTOX_VLLM_SERVE_ISQ_SINGLETHREAD",
        "CTOX_VLLM_SERVE_ISQ_CPU_THREADS",
        "CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN",
        "CTOX_VLLM_SERVE_TOPOLOGY",
        "CTOX_VLLM_SERVE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY",
        "CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL",
        "CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL",
        "CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND",
        "CTOX_VLLM_SERVE_DEVICE_LAYERS",
        "CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES",
    ] {
        env_map.remove(key);
    }
    env_map.insert("CTOX_VLLM_SERVE_ROLE".to_string(), "chat".to_string());
    env_map.insert("CTOX_VLLM_SERVE_MODEL".to_string(), runtime.model.clone());
    env_map.insert("CTOX_VLLM_SERVE_PORT".to_string(), runtime.port.to_string());
    env_map.insert(
        "CTOX_VLLM_SERVE_MAX_SEQS".to_string(),
        runtime.max_seqs.to_string(),
    );
    env_map.insert(
        "CTOX_VLLM_SERVE_MAX_BATCH_SIZE".to_string(),
        runtime.max_batch_size.to_string(),
    );
    if let Some(max_seq_len) = runtime.max_seq_len {
        env_map.insert(
            "CTOX_VLLM_SERVE_MAX_SEQ_LEN".to_string(),
            max_seq_len.to_string(),
        );
    } else {
        env_map.remove("CTOX_VLLM_SERVE_MAX_SEQ_LEN");
    }
    if let Some(arch) = family_profile.arch {
        env_map.insert("CTOX_VLLM_SERVE_ARCH".to_string(), arch);
    } else {
        env_map.remove("CTOX_VLLM_SERVE_ARCH");
    }
    env_map.insert(
        "CTOX_VLLM_SERVE_PAGED_ATTN".to_string(),
        family_profile.paged_attn,
    );
    if let Some(cache_type) = family_profile.pa_cache_type {
        env_map.insert("CTOX_VLLM_SERVE_PA_CACHE_TYPE".to_string(), cache_type);
    } else {
        env_map.remove("CTOX_VLLM_SERVE_PA_CACHE_TYPE");
    }
    if let Some(memory_fraction) = family_profile.pa_memory_fraction {
        env_map.insert(
            "CTOX_VLLM_SERVE_PA_MEMORY_FRACTION".to_string(),
            memory_fraction,
        );
    } else {
        env_map.remove("CTOX_VLLM_SERVE_PA_MEMORY_FRACTION");
    }
    if let Some(context_len) = family_profile.pa_context_len {
        env_map.insert(
            "CTOX_VLLM_SERVE_PA_CONTEXT_LEN".to_string(),
            context_len.to_string(),
        );
    } else {
        env_map.remove("CTOX_VLLM_SERVE_PA_CONTEXT_LEN");
    }
    if let Some(isq) = family_profile.isq {
        env_map.insert("CTOX_VLLM_SERVE_ISQ".to_string(), isq);
    } else {
        env_map.remove("CTOX_VLLM_SERVE_ISQ");
    }
    if let Some(backend) = family_profile.tensor_parallel_backend {
        env_map.insert(
            "CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND".to_string(),
            backend,
        );
    } else {
        env_map.remove("CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND");
    }
    env_map.insert(
        "CTOX_VLLM_SERVE_DISABLE_NCCL".to_string(),
        if family_profile.disable_nccl {
            "1"
        } else {
            "0"
        }
        .to_string(),
    );
    match model {
        "Qwen/Qwen3.5-35B-A3B" => {
            env_map.insert("CTOX_VLLM_SERVE_NO_MMAP".to_string(), "1".to_string());
            env_map.insert(
                "CTOX_VLLM_SERVE_LANGUAGE_MODEL_ONLY".to_string(),
                "1".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_ISQ_SINGLETHREAD".to_string(),
                "1".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN".to_string(),
                "1".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_TOPOLOGY".to_string(),
                root.join("scripts/qwen35b_3gpu_striped.topology.yaml")
                    .display()
                    .to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY".to_string(),
                "1".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL".to_string(),
                "2".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL".to_string(),
                "2".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND".to_string(),
                "fast".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_DEVICE_LAYERS".to_string(),
                "0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:4;2:4;0:2;1:3;2:5".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES".to_string(),
                "0,1,2".to_string(),
            );
        }
        "zai-org/GLM-4.7-Flash" => {
            env_map.insert("CTOX_VLLM_SERVE_NO_MMAP".to_string(), "1".to_string());
            env_map.insert(
                "CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN".to_string(),
                "1".to_string(),
            );
            env_map.insert(
                "CTOX_VLLM_SERVE_ISQ_CPU_THREADS".to_string(),
                "4".to_string(),
            );
        }
        _ => {}
    }
    if let Some(world_size) = family_profile.target_world_size {
        env_map.insert(
            "CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE".to_string(),
            world_size.to_string(),
        );
    } else {
        env_map.remove("CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE");
    }
    runtime_config::save_runtime_env_map(root, &env_map)?;
    println!("{}", serde_json::to_string_pretty(&env_map)?);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = resolve_workspace_root()?;

    match args.first().map(String::as_str) {
        None => tui::run_tui(&root),
        Some("clean-room-bootstrap-deps") => {
            let outcome = execution_baseline::bootstrap_clean_room_dependencies(&root)?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
            Ok(())
        }
        Some("clean-room-baseline-plan") => {
            let family = args
                .get(1)
                .map(String::as_str)
                .unwrap_or("gpt_oss")
                .parse()?;
            let prompt = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "Reply with CTOX_BASELINE_OK and nothing else.".to_string()
            };
            let plan = execution_baseline::build_clean_room_baseline_plan(&root, family, prompt);
            println!("{}", serde_json::to_string_pretty(&plan)?);
            Ok(())
        }
        Some("clean-room-rewrite-responses") => {
            let input_path = args
                .get(1)
                .context("usage: ctox clean-room-rewrite-responses <json-path>")?;
            let raw = std::fs::read(input_path)
                .with_context(|| format!("failed to read responses payload from {}", input_path))?;
            let rewritten = execution_baseline::rewrite_vllm_serve_responses_request(&raw)?;
            println!("{}", String::from_utf8_lossy(&rewritten));
            Ok(())
        }
        Some("chat-runtime-apply") => {
            let model = args.get(1).context(
                "usage: ctox chat-runtime-apply <model> <quality|max_context|performance>",
            )?;
            let preset = args.get(2).context(
                "usage: ctox chat-runtime-apply <model> <quality|max_context|performance>",
            )?;
            let mut env_map = runtime_config::load_runtime_env_map(&root).unwrap_or_default();
            env_map.insert("CTOX_CHAT_SOURCE".to_string(), "local".to_string());
            env_map.insert("CTOX_CHAT_MODEL_BASE".to_string(), model.clone());
            env_map.insert("CTOX_CHAT_MODEL".to_string(), model.clone());
            env_map.insert("CTOX_ACTIVE_MODEL".to_string(), model.clone());
            env_map.remove("CTOX_BOOST_ACTIVE_UNTIL_EPOCH");
            env_map.remove("CTOX_BOOST_REASON");
            env_map.insert(
                "CTOX_CHAT_LOCAL_PRESET".to_string(),
                runtime_planner::ChatPreset::from_label(preset)
                    .label()
                    .to_string(),
            );
            let plan = runtime_planner::apply_chat_runtime_plan(&root, &mut env_map)?
                .context("failed to resolve chat runtime plan")?;
            runtime_config::save_runtime_env_map(&root, &env_map)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
            Ok(())
        }
        Some("chat-runtime-apply-legacy") => {
            let model = args
                .get(1)
                .context("usage: ctox chat-runtime-apply-legacy <model>")?;
            apply_legacy_chat_runtime(&root, model)
        }
        Some("serve-responses-proxy") => {
            let config = responses_proxy::ProxyConfig::from_env_with_root(&root);
            eprintln!("{}", serde_json::to_string_pretty(&config)?);
            responses_proxy::serve_proxy(config)
        }
        Some("boost") => match args.get(1).map(String::as_str) {
            Some("status") => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&responses_proxy::boost_status(&root)?)?
                );
                Ok(())
            }
            Some("start") => {
                let minutes = find_flag_value(&args[2..], "--minutes")
                    .and_then(|value| value.parse::<u64>().ok());
                let model = find_flag_value(&args[2..], "--model");
                let reason = find_flag_value(&args[2..], "--reason");
                let result = responses_proxy::start_boost_lease(
                    &root,
                    model,
                    minutes,
                    reason,
                )?;
                println!("{}", serde_json::to_string_pretty(&result)?);
                Ok(())
            }
            Some("stop") => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&responses_proxy::stop_boost_lease(&root)?)?
                );
                Ok(())
            }
            _ => anyhow::bail!(
                "usage: ctox boost status | ctox boost start [--minutes <n>] [--model <id>] [--reason <text>] | ctox boost stop"
            ),
        },
        Some("service") => {
            if args.get(1).map(String::as_str) == Some("--foreground") {
                service::run_foreground(&root)
            } else {
                anyhow::bail!("usage: ctox service --foreground")
            }
        }
        Some("start") => {
            println!("{}", service::start_background(&root)?);
            Ok(())
        }
        Some("stop") => {
            println!("{}", service::stop_background(&root)?);
            Ok(())
        }
        Some("status") => {
            println!(
                "{}",
                serde_json::to_string_pretty(&service::service_status_snapshot(&root)?)?
            );
            Ok(())
        }
        Some("tui") => tui::run_tui(&root),
        Some("channel") => channels::handle_channel_command(&root, &args[1..]),
        Some("follow-up") => follow_up::handle_follow_up_command(&args[1..]),
        Some("plan") => plan::handle_plan_command(&root, &args[1..]),
        Some("queue") => queue::handle_queue_command(&root, &args[1..]),
        Some("schedule") => schedule::handle_schedule_command(&root, &args[1..]),
        Some("lcm-init") => {
            let db_path = args.get(1).context("usage: ctox lcm-init <db-path>")?;
            lcm::run_init(PathBuf::from(db_path).as_path())
        }
        Some("lcm-add-message") => {
            let db_path = args.get(1).context(
                "usage: ctox lcm-add-message <db-path> <conversation-id> <role> <content>",
            )?;
            let conversation_id: i64 = args
                .get(2)
                .context(
                    "usage: ctox lcm-add-message <db-path> <conversation-id> <role> <content>",
                )?
                .parse()
                .context("failed to parse conversation id")?;
            let role = args.get(3).context(
                "usage: ctox lcm-add-message <db-path> <conversation-id> <role> <content>",
            )?;
            let content = args
                .get(4..)
                .filter(|parts| !parts.is_empty())
                .map(|parts| parts.join(" "))
                .context(
                    "usage: ctox lcm-add-message <db-path> <conversation-id> <role> <content>",
                )?;
            let message = lcm::run_add_message(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                role,
                &content,
            )?;
            println!("{}", serde_json::to_string_pretty(&message)?);
            Ok(())
        }
        Some("lcm-compact") => {
            let db_path = args.get(1).context(
                "usage: ctox lcm-compact <db-path> <conversation-id> [token-budget] [--force]",
            )?;
            let conversation_id: i64 = args
                .get(2)
                .context(
                    "usage: ctox lcm-compact <db-path> <conversation-id> [token-budget] [--force]",
                )?
                .parse()
                .context("failed to parse conversation id")?;
            let token_budget = args
                .get(3)
                .filter(|value| !value.starts_with("--"))
                .map(|value| value.parse())
                .transpose()
                .context("failed to parse token budget")?
                .unwrap_or(24_000_i64);
            let force = args.iter().any(|arg| arg == "--force");
            let result = lcm::run_compact(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                token_budget,
                force,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-grep") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]")?;
            let conversation_arg = args
                .get(2)
                .context("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]")?;
            let conversation_id = if conversation_arg == "all" {
                None
            } else {
                Some(
                    conversation_arg
                        .parse()
                        .context("failed to parse conversation id")?,
                )
            };
            let scope = args
                .get(3)
                .context("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]")?;
            let mode = args
                .get(4)
                .context("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]")?;
            let tail = args
                .get(5..)
                .filter(|parts| !parts.is_empty())
                .context("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]")?;
            let (query, limit) = if let Some(last) = tail.last() {
                if let Ok(limit) = last.parse::<usize>() {
                    (tail[..tail.len().saturating_sub(1)].join(" "), limit)
                } else {
                    (tail.join(" "), 20_usize)
                }
            } else {
                anyhow::bail!("usage: ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]");
            };
            let result = lcm::run_grep(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                scope,
                mode,
                &query,
                limit,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-describe") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-describe <db-path> <summary-id>")?;
            let id = args
                .get(2)
                .context("usage: ctox lcm-describe <db-path> <summary-id>")?;
            let result = lcm::run_describe(PathBuf::from(db_path).as_path(), id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-expand") => {
            let db_path = args.get(1).context(
                "usage: ctox lcm-expand <db-path> <summary-id> [depth] [--messages] [token-cap]",
            )?;
            let summary_id = args.get(2).context(
                "usage: ctox lcm-expand <db-path> <summary-id> [depth] [--messages] [token-cap]",
            )?;
            let numeric_args = args
                .iter()
                .skip(3)
                .filter(|value| !value.starts_with("--"))
                .collect::<Vec<_>>();
            let depth = numeric_args
                .first()
                .map(|value| value.parse())
                .transpose()
                .context("failed to parse depth")?
                .unwrap_or(1_usize);
            let include_messages = args.iter().any(|arg| arg == "--messages");
            let token_cap = numeric_args
                .get(1)
                .map(|value| value.parse())
                .transpose()
                .context("failed to parse token cap")?
                .unwrap_or(8_000_i64);
            let result = lcm::run_expand(
                PathBuf::from(db_path).as_path(),
                summary_id,
                depth,
                include_messages,
                token_cap,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-dump") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-dump <db-path> <conversation-id>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox lcm-dump <db-path> <conversation-id>")?
                .parse()
                .context("failed to parse conversation id")?;
            let result = lcm::run_dump(PathBuf::from(db_path).as_path(), conversation_id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-refresh-continuity") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-refresh-continuity <db-path> <conversation-id>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox lcm-refresh-continuity <db-path> <conversation-id>")?
                .parse()
                .context("failed to parse conversation id")?;
            let result =
                lcm::run_refresh_continuity(PathBuf::from(db_path).as_path(), conversation_id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-show-continuity") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-show-continuity <db-path> <conversation-id>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox lcm-show-continuity <db-path> <conversation-id>")?
                .parse()
                .context("failed to parse conversation id")?;
            let result =
                lcm::run_show_continuity(PathBuf::from(db_path).as_path(), conversation_id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("lcm-run-fixture") => {
            let db_path = args
                .get(1)
                .context("usage: ctox lcm-run-fixture <db-path> <fixture-path>")?;
            let fixture_path = args
                .get(2)
                .context("usage: ctox lcm-run-fixture <db-path> <fixture-path>")?;
            let result = lcm::run_fixture(
                PathBuf::from(db_path).as_path(),
                PathBuf::from(fixture_path).as_path(),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-init") => {
            let db_path = args
                .get(1)
                .context("usage: ctox continuity-init <db-path> <conversation-id>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-init <db-path> <conversation-id>")?
                .parse()
                .context("failed to parse conversation id")?;
            let result =
                lcm::run_continuity_init(PathBuf::from(db_path).as_path(), conversation_id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-show") => {
            let db_path = args.get(1).context(
                "usage: ctox continuity-show <db-path> <conversation-id> [narrative|anchors|focus]",
            )?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-show <db-path> <conversation-id> [narrative|anchors|focus]")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args.get(3).map(String::as_str);
            let result =
                lcm::run_continuity_show(PathBuf::from(db_path).as_path(), conversation_id, kind)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-apply") => {
            let db_path = args
                .get(1)
                .context("usage: ctox continuity-apply <db-path> <conversation-id> <narrative|anchors|focus> <diff-path>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-apply <db-path> <conversation-id> <narrative|anchors|focus> <diff-path>")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args
                .get(3)
                .context("usage: ctox continuity-apply <db-path> <conversation-id> <narrative|anchors|focus> <diff-path>")?;
            let diff_path = args
                .get(4)
                .context("usage: ctox continuity-apply <db-path> <conversation-id> <narrative|anchors|focus> <diff-path>")?;
            let result = lcm::run_continuity_apply(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                kind,
                PathBuf::from(diff_path).as_path(),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-log") => {
            let db_path = args.get(1).context(
                "usage: ctox continuity-log <db-path> <conversation-id> [narrative|anchors|focus]",
            )?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-log <db-path> <conversation-id> [narrative|anchors|focus]")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args.get(3).map(String::as_str);
            let result =
                lcm::run_continuity_log(PathBuf::from(db_path).as_path(), conversation_id, kind)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-rebuild") => {
            let db_path = args
                .get(1)
                .context("usage: ctox continuity-rebuild <db-path> <conversation-id> <narrative|anchors|focus>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-rebuild <db-path> <conversation-id> <narrative|anchors|focus>")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args
                .get(3)
                .context("usage: ctox continuity-rebuild <db-path> <conversation-id> <narrative|anchors|focus>")?;
            let result = lcm::run_continuity_rebuild(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                kind,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-forgotten") => {
            let db_path = args
                .get(1)
                .context("usage: ctox continuity-forgotten <db-path> <conversation-id> [narrative|anchors|focus] [query]")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-forgotten <db-path> <conversation-id> [narrative|anchors|focus] [query]")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args.get(3).map(String::as_str);
            let query = args
                .get(4..)
                .filter(|parts| !parts.is_empty())
                .map(|parts| parts.join(" "));
            let result = lcm::run_continuity_forgotten(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                kind,
                query.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("continuity-build-prompt") => {
            let db_path = args
                .get(1)
                .context("usage: ctox continuity-build-prompt <db-path> <conversation-id> <narrative|anchors|focus>")?;
            let conversation_id: i64 = args
                .get(2)
                .context("usage: ctox continuity-build-prompt <db-path> <conversation-id> <narrative|anchors|focus>")?
                .parse()
                .context("failed to parse conversation id")?;
            let kind = args
                .get(3)
                .context("usage: ctox continuity-build-prompt <db-path> <conversation-id> <narrative|anchors|focus>")?;
            let result = lcm::run_continuity_build_prompt(
                PathBuf::from(db_path).as_path(),
                conversation_id,
                kind,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Some("context-retrieve") => {
            let conversation_id: i64 = find_flag_value(&args[1..], "--conversation-id")
                .unwrap_or("1")
                .parse()
                .context("failed to parse conversation id")?;
            let mode = find_flag_value(&args[1..], "--mode").unwrap_or("current");
            let db_path = find_flag_value(&args[1..], "--db")
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("runtime/ctox_lcm.db"));
            let query = find_flag_value(&args[1..], "--query").map(ToOwned::to_owned);
            let continuity_kind = find_flag_value(&args[1..], "--kind").map(ToOwned::to_owned);
            let summary_id = find_flag_value(&args[1..], "--summary-id").map(ToOwned::to_owned);
            let limit = find_flag_value(&args[1..], "--limit")
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("failed to parse limit")?
                .unwrap_or(10);
            let depth = find_flag_value(&args[1..], "--depth")
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("failed to parse depth")?
                .unwrap_or(1);
            let token_cap = find_flag_value(&args[1..], "--token-cap")
                .map(|value| value.parse::<i64>())
                .transpose()
                .context("failed to parse token cap")?
                .unwrap_or(8_000);
            let include_messages = args.iter().any(|arg| arg == "--messages");
            let result = lcm::run_context_retrieve(
                db_path.as_path(),
                conversation_id,
                mode,
                query.as_deref(),
                continuity_kind.as_deref(),
                summary_id.as_deref(),
                limit,
                depth,
                include_messages,
                token_cap,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        _ => {
            anyhow::bail!(
                "usage:\n  ctox\n  ctox start\n  ctox stop\n  ctox status\n  ctox service --foreground\n  ctox clean-room-bootstrap-deps\n  ctox clean-room-baseline-plan <gpt_oss|qwen3_5> [prompt]\n  ctox clean-room-rewrite-responses <json-path>\n  ctox chat-runtime-apply <model> <quality|max_context|performance>\n  ctox chat-runtime-apply-legacy <model>\n  ctox serve-responses-proxy\n  ctox boost status\n  ctox boost start [--minutes <n>] [--model <id>] [--reason <text>]\n  ctox boost stop\n  ctox tui\n  ctox channel <subcommand> ...\n  ctox follow-up <subcommand> ...\n  ctox plan <subcommand> ...\n  ctox schedule <subcommand> ...\n  ctox lcm-init <db-path>\n  ctox lcm-add-message <db-path> <conversation-id> <role> <content>\n  ctox lcm-compact <db-path> <conversation-id> [token-budget] [--force]\n  ctox lcm-grep <db-path> <conversation-id|all> <scope> <mode> <query> [limit]\n  ctox lcm-describe <db-path> <summary-id>\n  ctox lcm-expand <db-path> <summary-id> [depth] [--messages] [token-cap]\n  ctox lcm-dump <db-path> <conversation-id>\n  ctox lcm-refresh-continuity <db-path> <conversation-id>\n  ctox lcm-show-continuity <db-path> <conversation-id>\n  ctox lcm-run-fixture <db-path> <fixture-path>\n  ctox continuity-init <db-path> <conversation-id>\n  ctox continuity-show <db-path> <conversation-id> [narrative|anchors|focus]\n  ctox continuity-apply <db-path> <conversation-id> <narrative|anchors|focus> <diff-path>\n  ctox continuity-log <db-path> <conversation-id> [narrative|anchors|focus]\n  ctox continuity-rebuild <db-path> <conversation-id> <narrative|anchors|focus>\n  ctox continuity-forgotten <db-path> <conversation-id> [narrative|anchors|focus] [query]\n  ctox continuity-build-prompt <db-path> <conversation-id> <narrative|anchors|focus>\n  ctox context-retrieve [--db <path>] [--conversation-id <id>] --mode <current|continuity|forgotten|search|describe|expand> [--kind <narrative|anchors|focus>] [--query <text>] [--summary-id <id>] [--limit <n>] [--depth <n>] [--messages] [--token-cap <n>]"
            )
        }
    }
}

fn resolve_workspace_root() -> anyhow::Result<PathBuf> {
    if let Some(root) = std::env::var_os("CTOX_ROOT") {
        return Ok(PathBuf::from(root));
    }
    std::env::current_dir().context("failed to resolve CTOX workspace root")
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).map(String::as_str)
}
