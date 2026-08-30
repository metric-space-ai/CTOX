// CLI dispatch for `ctox scrape ...`: flag parsing, usage texts, and the
// subcommand fan-out into the sibling modules.

use super::{
    DEFAULT_RUNTIME_ROOT, count_rows, execute_scrape, find_flag_value, list_targets,
    load_json_file, open_db, parse_where_filters, print_json, promote_template, query_records,
    rebuild_semantic_index, record_template_example, register_script, register_source_module,
    required_flag_value, resolve_db_path, semantic_search, show_api, show_latest, show_target,
    summary_payload, upsert_target,
};
use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;

pub fn handle_scrape_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "init" => {
            let conn = open_db(root)?;
            print_json(&json!({
                "ok": true,
                "db_path": resolve_db_path(root),
                "initialized": {
                    "targets_total": count_rows(&conn, "scrape_target")?,
                    "script_revisions_total": count_rows(&conn, "scrape_script_revision")?,
                    "runs_total": count_rows(&conn, "scrape_run")?,
                }
            }))
        }
        "summary" => print_json(&summary_payload(root)?),
        "list-targets" => print_json(&json!({ "ok": true, "targets": list_targets(root)? })),
        "show-target" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape show-target --target-key <key>")?;
            let target = show_target(root, target_key)?.context("target_key not found")?;
            print_json(&json!({ "ok": true, "target": target }))
        }
        "show-latest" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape show-latest --target-key <key> [--limit <n>]")?;
            let limit = find_flag_value(args, "--limit")
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("failed to parse --limit")?
                .unwrap_or(20);
            let latest = show_latest(root, target_key, limit)?.context("target_key not found")?;
            print_json(&json!({ "ok": true, "latest": latest }))
        }
        "show-api" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape show-api --target-key <key>")?;
            let api = show_api(root, target_key)?.context("target_key not found")?;
            print_json(&json!({ "ok": true, "api": api }))
        }
        "query-records" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape query-records --target-key <key> [--where field=value]... [--limit <n>]")?;
            let limit = find_flag_value(args, "--limit")
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("failed to parse --limit")?
                .unwrap_or(50);
            let filters = parse_where_filters(args)?;
            let response = query_records(root, target_key, &filters, limit)?
                .context("target_key not found")?;
            print_json(&json!({ "ok": true, "query": response }))
        }
        "semantic-search" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape semantic-search --target-key <key> --query <text> [--limit <n>]")?;
            let query = required_flag_value(args, "--query")
                .or_else(|| find_flag_value(args, "-q"))
                .context("usage: ctox scrape semantic-search --target-key <key> --query <text> [--limit <n>]")?;
            let limit = find_flag_value(args, "--limit")
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("failed to parse --limit")?
                .unwrap_or(12);
            let response =
                semantic_search(root, target_key, query, limit)?.context("target_key not found")?;
            print_json(&json!({ "ok": true, "semantic": response }))
        }
        "rebuild-semantic" => {
            let target_key = required_flag_value(args, "--target-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox scrape rebuild-semantic --target-key <key> [--limit <n>]")?;
            let response =
                rebuild_semantic_index(root, target_key)?.context("target_key not found")?;
            print_json(&json!({ "ok": true, "semantic_rebuild": response }))
        }
        "upsert-target" => {
            let input = required_flag_value(args, "--input").context(
                "usage: ctox scrape upsert-target --input <json-path> [--runtime-root <path>]",
            )?;
            let runtime_root =
                find_flag_value(args, "--runtime-root").unwrap_or(DEFAULT_RUNTIME_ROOT);
            let payload = load_json_file(root, input)?;
            let target = upsert_target(root, runtime_root, payload)?;
            print_json(&json!({ "ok": true, "target": target }))
        }
        "register-script" => {
            let target_key = required_flag_value(args, "--target-key")
                .context("usage: ctox scrape register-script --target-key <key> --script-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]")?;
            let script_file = required_flag_value(args, "--script-file")
                .context("usage: ctox scrape register-script --target-key <key> --script-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]")?;
            let runtime_root =
                find_flag_value(args, "--runtime-root").unwrap_or(DEFAULT_RUNTIME_ROOT);
            let language = find_flag_value(args, "--language").unwrap_or("javascript");
            let registered = register_script(
                root,
                runtime_root,
                target_key,
                script_file,
                language,
                find_flag_value(args, "--change-reason"),
                find_flag_value(args, "--notes"),
            )?;
            print_json(&json!({ "ok": true, "script": registered }))
        }
        "register-source-module" => {
            let target_key = required_flag_value(args, "--target-key")
                .context("usage: ctox scrape register-source-module --target-key <key> --source-key <key> --module-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]")?;
            let source_key = required_flag_value(args, "--source-key")
                .context("usage: ctox scrape register-source-module --target-key <key> --source-key <key> --module-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]")?;
            let module_file = required_flag_value(args, "--module-file")
                .context("usage: ctox scrape register-source-module --target-key <key> --source-key <key> --module-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]")?;
            let runtime_root =
                find_flag_value(args, "--runtime-root").unwrap_or(DEFAULT_RUNTIME_ROOT);
            let language = find_flag_value(args, "--language").unwrap_or("javascript");
            let registered = register_source_module(
                root,
                runtime_root,
                target_key,
                source_key,
                module_file,
                language,
                find_flag_value(args, "--change-reason"),
                find_flag_value(args, "--notes"),
            )?;
            print_json(&json!({ "ok": true, "source_module": registered }))
        }
        "record-template-example" => {
            let target_key = required_flag_value(args, "--target-key")
                .context("usage: ctox scrape record-template-example --target-key <key> --template-key <template> --script-file <path> [--language <lang>] [--result-count <n>] [--challenge-score <n>] [--reason <text>]")?;
            let template_key = required_flag_value(args, "--template-key")
                .context("usage: ctox scrape record-template-example --target-key <key> --template-key <template> --script-file <path> [--language <lang>] [--result-count <n>] [--challenge-score <n>] [--reason <text>]")?;
            let script_file = required_flag_value(args, "--script-file")
                .context("usage: ctox scrape record-template-example --target-key <key> --template-key <template> --script-file <path> [--language <lang>] [--result-count <n>] [--challenge-score <n>] [--reason <text>]")?;
            let language = find_flag_value(args, "--language").unwrap_or("javascript");
            let result_count = find_flag_value(args, "--result-count")
                .map(|value| value.parse::<i64>())
                .transpose()
                .context("failed to parse --result-count")?;
            let challenge_score = find_flag_value(args, "--challenge-score")
                .map(|value| value.parse::<i64>())
                .transpose()
                .context("failed to parse --challenge-score")?
                .unwrap_or(0);
            let result = record_template_example(
                root,
                target_key,
                template_key,
                script_file,
                language,
                result_count,
                challenge_score,
                find_flag_value(args, "--reason"),
            )?;
            print_json(&json!({ "ok": true, "template_event": result }))
        }
        "promote-template" => {
            let template_key = required_flag_value(args, "--template-key")
                .context("usage: ctox scrape promote-template --template-key <template> --script-file <path> [--language <lang>] --reason <text>")?;
            let script_file = required_flag_value(args, "--script-file")
                .context("usage: ctox scrape promote-template --template-key <template> --script-file <path> [--language <lang>] --reason <text>")?;
            let language = find_flag_value(args, "--language").unwrap_or("javascript");
            let reason = required_flag_value(args, "--reason")
                .context("usage: ctox scrape promote-template --template-key <template> --script-file <path> [--language <lang>] --reason <text>")?;
            let promoted = promote_template(root, template_key, script_file, language, reason)?;
            print_json(&json!({ "ok": true, "promoted_template": promoted }))
        }
        "execute" => execute_scrape(root, args),
        _ => anyhow::bail!(
            "usage:\n  ctox scrape init\n  ctox scrape summary\n  ctox scrape list-targets\n  ctox scrape show-target --target-key <key>\n  ctox scrape show-latest --target-key <key> [--limit <n>]\n  ctox scrape show-api --target-key <key>\n  ctox scrape query-records --target-key <key> [--where field=value]... [--limit <n>]\n  ctox scrape semantic-search --target-key <key> --query <text> [--limit <n>]\n  ctox scrape rebuild-semantic --target-key <key>\n  ctox scrape upsert-target --input <json-path> [--runtime-root <path>]\n  ctox scrape register-script --target-key <key> --script-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]\n  ctox scrape register-source-module --target-key <key> --source-key <key> --module-file <path> [--language <lang>] [--change-reason <text>] [--notes <text>] [--runtime-root <path>]\n  ctox scrape record-template-example --target-key <key> --template-key <template> --script-file <path> [--language <lang>] [--result-count <n>] [--challenge-score <n>] [--reason <text>]\n  ctox scrape promote-template --template-key <template> --script-file <path> [--language <lang>] --reason <text>\n  ctox scrape execute --target-key <key> [--trigger-kind <manual|scheduled|repair>] [--scheduled-for <iso>] [--timeout-seconds <n>] [--runtime-root <path>] [--allow-heal] [--thread-key <key>] [--queue-priority <urgent|high|normal|low>]"
        ),
    }
}
