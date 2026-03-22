use anyhow::Context;

mod execution_baseline;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = std::env::current_dir().context("failed to resolve CTOX workspace root")?;

    match args.first().map(String::as_str) {
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
            let plan = execution_baseline::build_clean_room_baseline_plan(
                &root,
                family,
                prompt,
            );
            println!("{}", serde_json::to_string_pretty(&plan)?);
            Ok(())
        }
        Some("clean-room-rewrite-responses") => {
            let input_path = args
                .get(1)
                .context("usage: ctox clean-room-rewrite-responses <json-path>")?;
            let raw = std::fs::read(input_path)
                .with_context(|| format!("failed to read responses payload from {}", input_path))?;
            let rewritten = execution_baseline::rewrite_mistralrs_responses_request(&raw)?;
            println!("{}", String::from_utf8_lossy(&rewritten));
            Ok(())
        }
        _ => {
            anyhow::bail!(
                "usage:\n  ctox clean-room-bootstrap-deps\n  ctox clean-room-baseline-plan <gpt_oss|qwen3_5> [prompt]\n  ctox clean-room-rewrite-responses <json-path>"
            )
        }
    }
}
