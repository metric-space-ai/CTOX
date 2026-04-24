// Origin: CTOX
// License: Apache-2.0

//! Per-model local backend registry — maps a request model ID onto
//! the matching server binary under `src/inference/models/<model>/`
//! and the CLI args needed to spawn it.
//!
//! Scope today: only the Qwen3.5-27B Q4_K_M + DFlash-draft pair.
//! Adding a second curated model means appending one entry to
//! [`resolve_local_model_backend`] plus any new env vars for
//! weight/tokenizer paths.
//!
//! The design is deliberately tiny — no config file, no dynamic
//! discovery. The supervisor is the only caller and needs one thing:
//! "for this request model, what binary + what args?".
//!
//! Weight paths come from env vars. Defaults match the
//! `dflash-ref` layout on a standard dev box, so the common case
//! works out of the box:
//!
//! | Env                              | Default                                                           |
//! |----------------------------------|-------------------------------------------------------------------|
//! | `CTOX_QWEN35_TARGET_GGUF`        | `$HOME/dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf`          |
//! | `CTOX_QWEN35_DRAFT_SAFETENSORS`  | `$HOME/dflash-ref/dflash/models/draft/model.safetensors`          |
//! | `CTOX_QWEN35_TOKENIZER`          | _discovered from the HF cache_                                    |

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// One local-model backend: binary to spawn + CLI args + env vars.
pub struct LocalModelBackend {
    pub binary: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(&'static str, OsString)>,
    /// Human-readable model id reported back in health/chat responses.
    pub model_id: &'static str,
}

/// Inputs the registry needs to build a launch spec.
pub struct LocalModelRequest<'a> {
    /// The request model (`launch_spec.request_model`).
    pub request_model: &'a str,
    /// The Unix-socket path the gateway expects.
    pub transport_endpoint: Option<&'a str>,
    /// CTOX install root — used to locate the built server binary.
    pub root: &'a Path,
}

/// Returns `Some(...)` if the request model has a local in-tree
/// server binary, else `None`. `None` means the caller must treat
/// this model as API-only.
pub fn resolve_local_model_backend(req: LocalModelRequest<'_>) -> Option<LocalModelBackend> {
    let model = req.request_model.trim();
    if is_qwen35_27b(model) {
        return Some(qwen35_27b_q4km_dflash_backend(req.root, req.transport_endpoint));
    }
    None
}

/// Is `model` handled by the Qwen3.5-27B Q4_K_M + DFlash server?
/// Matches the canonical request-model IDs we expect CTOX to
/// pipe in for local chat.
pub fn is_qwen35_27b(model: &str) -> bool {
    // Request model aliases — keep in sync with the model registry.
    model == "qwen35-27b-q4km-dflash"
        || model == "Qwen/Qwen3.5-27B"
        || model.starts_with("Qwen/Qwen3.5-27B-")
        || model.starts_with("unsloth/Qwen3.5-27B")
}

fn qwen35_27b_q4km_dflash_backend(
    root: &Path,
    transport_endpoint: Option<&str>,
) -> LocalModelBackend {
    let binary = root
        .join("src/inference/models/qwen35_27b_q4km_dflash/target/release")
        .join("qwen35-27b-q4km-dflash-server");

    let target = env_path_or(
        "CTOX_QWEN35_TARGET_GGUF",
        default_qwen35_target(),
    );
    let draft = env_path_or(
        "CTOX_QWEN35_DRAFT_SAFETENSORS",
        default_qwen35_draft(),
    );
    let tokenizer = std::env::var_os("CTOX_QWEN35_TOKENIZER");

    // Socket path comes from the gateway's runtime_state resolution.
    // Fallback to the canonical runtime/sockets/ path under `root`
    // if the caller didn't pass one — should only happen in tests.
    let socket = match transport_endpoint {
        Some(ep) => PathBuf::from(ep),
        None => root.join("runtime/sockets/primary_generation.sock"),
    };

    let mut args: Vec<OsString> = Vec::new();
    args.push("--target".into());
    args.push(target.into());
    args.push("--draft".into());
    args.push(draft.into());
    if let Some(tok) = tokenizer {
        args.push("--tokenizer".into());
        args.push(tok);
    }
    args.push("--socket".into());
    args.push(socket.into());
    args.push("--model-id".into());
    args.push("qwen35-27b-q4km-dflash".into());

    // No extra env currently. CUDA_VISIBLE_DEVICES / launch-spec
    // env gets layered by the supervisor's shared env helper.
    let env: Vec<(&'static str, OsString)> = Vec::new();

    LocalModelBackend {
        binary,
        args,
        env,
        model_id: "qwen35-27b-q4km-dflash",
    }
}

fn env_path_or(key: &str, fallback: PathBuf) -> PathBuf {
    match std::env::var_os(key) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => fallback,
    }
}

fn default_qwen35_target() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join("dflash-ref/dflash/models/Qwen3.5-27B-Q4_K_M.gguf")
}

fn default_qwen35_draft() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join("dflash-ref/dflash/models/draft/model.safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_aliases_resolve() {
        assert!(is_qwen35_27b("qwen35-27b-q4km-dflash"));
        assert!(is_qwen35_27b("Qwen/Qwen3.5-27B"));
        assert!(is_qwen35_27b("Qwen/Qwen3.5-27B-Instruct"));
        assert!(is_qwen35_27b("unsloth/Qwen3.5-27B-GGUF"));
        assert!(!is_qwen35_27b("Qwen/Qwen3-4B"));
        assert!(!is_qwen35_27b("anthropic/claude-sonnet-4.7"));
    }

    #[test]
    fn qwen35_backend_assembles_expected_cli() {
        let root = Path::new("/tmp/ctoxroot");
        let backend = resolve_local_model_backend(LocalModelRequest {
            request_model: "Qwen/Qwen3.5-27B-Instruct",
            transport_endpoint: Some("/tmp/ctoxroot/runtime/sockets/primary_generation.sock"),
            root,
        })
        .expect("qwen35 backend must resolve");
        let joined: String = backend
            .args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(joined.contains("--target"));
        assert!(joined.contains("--draft"));
        assert!(joined.contains("--socket"));
        assert!(joined.contains("primary_generation.sock"));
        assert!(joined.contains("--model-id qwen35-27b-q4km-dflash"));
        assert_eq!(backend.model_id, "qwen35-27b-q4km-dflash");
        assert!(backend
            .binary
            .ends_with("src/inference/models/qwen35_27b_q4km_dflash/target/release/qwen35-27b-q4km-dflash-server"));
    }

    #[test]
    fn unsupported_model_returns_none() {
        let root = Path::new("/tmp/ctoxroot");
        assert!(resolve_local_model_backend(LocalModelRequest {
            request_model: "anthropic/claude-sonnet-4.7",
            transport_endpoint: None,
            root,
        })
        .is_none());
    }
}
