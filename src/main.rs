mod attach;
mod agentic;
mod app;
mod brain_runtime;
mod browser_agent_bridge;
mod browser_engine;
mod browser_subworkers;
mod bootstrap;
mod codex_head_tail_buffer;
mod codex_text_encoding;
mod command_exec;
mod contracts;
mod context_controller;
mod desktop_session;
mod pages;
mod runtime_db;
mod storage;
mod supervisor;
mod tooling;

fn install_rustls_crypto_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_rustls_crypto_provider();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--init-only") => return app::init_only(),
        Some("install-bootstrap-tui") => {
            let paths = contracts::Paths::discover()?;
            bootstrap::run_install_bootstrap_tui(&paths)?;
            return Ok(());
        }
        Some("check-kleinhirn") => {
            let paths = contracts::Paths::discover()?;
            agentic::enforce_kleinhirn_ready(&paths)?;
            println!("READY");
            return Ok(());
        }
        Some("wait-kleinhirn-startup") => {
            let paths = contracts::Paths::discover()?;
            agentic::wait_for_kleinhirn_startup_ready(&paths)?;
            println!("READY");
            return Ok(());
        }
        Some("run-census") => {
            let paths = contracts::Paths::discover()?;
            let census = supervisor::run_system_census(&paths)?;
            println!("{}", serde_json::to_string_pretty(&census)?);
            return Ok(());
        }
        Some("recommend-kleinhirn") => {
            let paths = contracts::Paths::discover()?;
            let policy = contracts::load_model_policy(&paths);
            let census = contracts::load_census(&paths);
            let selected = contracts::recommended_kleinhirn(&policy, &census);
            println!("{}", serde_json::to_string_pretty(selected)?);
            return Ok(());
        }
        Some("recommend-browser-vision-kleinhirn") => {
            let paths = contracts::Paths::discover()?;
            let policy = contracts::load_model_policy(&paths);
            let census = contracts::load_census(&paths);
            let selected = contracts::recommended_browser_vision_kleinhirn(&policy, &census);
            println!("{}", serde_json::to_string_pretty(&selected)?);
            return Ok(());
        }
        Some("upgrade-kleinhirn") => {
            let paths = contracts::Paths::discover()?;
            let target = args.get(1).map(String::as_str);
            let outcome = brain_runtime::apply_targeted_kleinhirn_upgrade(&paths, target)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "changed": outcome.changed,
                "restarted": outcome.restarted,
                "summary": outcome.summary,
                "previousRuntimeModel": outcome.previous_runtime_model,
                "currentRuntimeModel": outcome.current_runtime_model,
            }))?);
            return Ok(());
        }
        Some("hard-reset-report") => {
            let paths = contracts::Paths::discover()?;
            let reason = if args.len() > 1 {
                args[1..].join(" ")
            } else {
                "automatic hard reset requested".to_string()
            };
            let report_path =
                runtime_db::write_pending_hard_reset_report(&paths, "automatic_restart", &reason)?;
            println!("{report_path}");
            return Ok(());
        }
        Some("attach") => {
            let paths = contracts::Paths::discover()?;
            return attach::run_attach_cli(&paths, &args[1..]);
        }
        Some("send") => {
            let paths = contracts::Paths::discover()?;
            let line = args[1..].join(" ");
            if line.trim().is_empty() {
                anyhow::bail!("usage: cto-agent send 'Michael Welsch: Nachricht'");
            }
            let output = attach::send_attach_line(&paths, &line)?;
            println!("{output}");
            return Ok(());
        }
        Some("channel-interrupt") => {
            let paths = contracts::Paths::discover()?;
            if args.len() < 4 {
                anyhow::bail!(
                    "usage: cto-agent channel-interrupt <source_channel> <speaker> <message>"
                );
            }
            let source_channel = args[1].trim();
            let speaker = args[2].trim();
            let message = args[3..].join(" ");
            if source_channel.is_empty() || speaker.is_empty() || message.trim().is_empty() {
                anyhow::bail!(
                    "usage: cto-agent channel-interrupt <source_channel> <speaker> <message>"
                );
            }
            let outcome =
                bootstrap::queue_channel_interrupt(&paths, source_channel, speaker, &message)?;
            println!("{}", outcome.output);
            return Ok(());
        }
        Some("status") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/status")?;
            println!("{output}");
            return Ok(());
        }
        Some("thread") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/thread")?;
            println!("{output}");
            return Ok(());
        }
        Some("signals") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/signals")?;
            println!("{output}");
            return Ok(());
        }
        Some("incidents") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/incidents")?;
            println!("{output}");
            return Ok(());
        }
        Some("events") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/events")?;
            println!("{output}");
            return Ok(());
        }
        Some("turns") => {
            let paths = contracts::Paths::discover()?;
            let output = attach::send_attach_line(&paths, "/turns")?;
            println!("{output}");
            return Ok(());
        }
        _ => {}
    }
    app::run().await
}
