// ref: cmd/validate_codex_models/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{env, fs, process::ExitCode};

use ctox_cliproxyapi::internal::registry::validate_codex_client_models_json;

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(flag) = args.next() else {
        eprintln!("error: --file is required");
        return ExitCode::from(2);
    };
    if flag != "--file" {
        eprintln!("error: --file is required");
        return ExitCode::from(2);
    }
    let Some(path) = args.next() else {
        eprintln!("error: --file is required");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("error: unexpected arguments");
        return ExitCode::from(2);
    }
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("error: read {}: {error}", path.to_string_lossy());
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = validate_codex_client_models_json(&data) {
        eprintln!(
            "error: invalid Codex client model catalog {}: {error}",
            path.to_string_lossy()
        );
        return ExitCode::FAILURE;
    }
    println!(
        "Validated Codex client model catalog: {}",
        path.to_string_lossy()
    );
    ExitCode::SUCCESS
}
