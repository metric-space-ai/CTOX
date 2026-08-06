// ref: cmd/fetch_antigravity_models/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{collections::BTreeMap, fmt, path::Path, process::ExitCode, time::Duration};

use ctox_cliproxyapi::sdk::cliproxy::auth::{Auth, AuthStore};
use serde::Serialize;
use serde_json::Value;
use url::Url;

const BASE_URLS: [&str; 3] = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
];
const MODELS_PATH: &str = "/v1internal:fetchAvailableModels";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    pub output: String,
    pub pretty: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            output: "antigravity_models.json".to_owned(),
            pretty: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRequest {
    pub url: Url,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
    pub timeout: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

pub trait HttpTransport: Send + Sync {
    fn execute(&self, request: HttpRequest) -> Result<HttpResponse, CommandError>;
}

pub trait SecretResolver: Send + Sync {
    fn resolve(&self, auth: &Auth, name: &'static str) -> Result<Option<String>, CommandError>;
}

pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}
pub trait FileOutput: Send + Sync {
    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CommandError>;
}
pub trait CommandOutput: Send + Sync {
    fn info(&self, message: &str);
    fn warning(&self, message: &str);
}

pub struct Dependencies<'a> {
    pub auth_store: &'a dyn AuthStore,
    pub secrets: &'a dyn SecretResolver,
    pub http: &'a dyn HttpTransport,
    pub cancellation: &'a dyn Cancellation,
    pub files: &'a dyn FileOutput,
    pub output: &'a dyn CommandOutput,
    pub request_timeout: Duration,
    pub user_agent: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandError {
    Cancelled,
    Store,
    NoAuth,
    MissingAccessToken,
    InvalidUrl,
    Http,
    InvalidResponse,
    Write,
    InvalidArguments(String),
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("command cancelled"),
            Self::Store => formatter.write_str("auth store operation failed"),
            Self::NoAuth => formatter.write_str("no enabled antigravity auth found"),
            Self::MissingAccessToken => formatter.write_str("no access token found in auth"),
            Self::InvalidUrl => formatter.write_str("invalid Antigravity models URL"),
            Self::Http => formatter.write_str("all Antigravity model endpoints failed"),
            Self::InvalidResponse => formatter.write_str("invalid Antigravity models response"),
            Self::Write => formatter.write_str("failed to write output"),
            Self::InvalidArguments(value) => write!(formatter, "invalid arguments: {value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ModelOutput {
    models: Vec<ModelEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    display_name: String,
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
}

fn find_auth(auths: &[Auth]) -> Result<Auth, CommandError> {
    auths
        .iter()
        .find(|auth| !auth.disabled && auth.provider.trim().eq_ignore_ascii_case("antigravity"))
        .cloned()
        .ok_or(CommandError::NoAuth)
}

fn fetch_models(auth: &Auth, deps: &Dependencies<'_>) -> Result<Vec<ModelEntry>, CommandError> {
    let access_token = deps
        .secrets
        .resolve(auth, "access_token")?
        .filter(|value| !value.trim().is_empty())
        .ok_or(CommandError::MissingAccessToken)?;
    let project = auth
        .metadata
        .get("project_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty());
    let body = match project {
        Some(project) => serde_json::to_vec(&serde_json::json!({"project": project})),
        None => serde_json::to_vec(&serde_json::json!({})),
    }
    .map_err(|_| CommandError::InvalidResponse)?;
    for base in BASE_URLS {
        check_cancelled(deps.cancellation)?;
        let url =
            Url::parse(&format!("{base}{MODELS_PATH}")).map_err(|_| CommandError::InvalidUrl)?;
        let response = match deps.http.execute(HttpRequest {
            url,
            headers: BTreeMap::from([
                ("Content-Type".to_owned(), "application/json".to_owned()),
                ("Authorization".to_owned(), format!("Bearer {access_token}")),
                ("User-Agent".to_owned(), deps.user_agent.to_owned()),
            ]),
            body: body.clone(),
            timeout: deps.request_timeout,
        }) {
            Ok(response) if (200..300).contains(&response.status) => response,
            _ => continue,
        };
        if let Ok(models) = parse_models(&response.body) {
            return Ok(models);
        }
    }
    Err(CommandError::Http)
}

fn parse_models(raw: &[u8]) -> Result<Vec<ModelEntry>, CommandError> {
    const SKIP: [&str; 6] = [
        "chat_20706",
        "chat_23310",
        "tab_flash_lite_preview",
        "tab_jump_flash_lite_preview",
        "gemini-2.5-flash-thinking",
        "gemini-2.5-pro",
    ];
    let value: Value = serde_json::from_slice(raw).map_err(|_| CommandError::InvalidResponse)?;
    let models = value
        .get("models")
        .and_then(Value::as_object)
        .ok_or(CommandError::InvalidResponse)?;
    let mut output = Vec::new();
    for (original_name, model) in models {
        let id = original_name.trim();
        if id.is_empty() || SKIP.contains(&id) {
            continue;
        }
        let display_name = model
            .get("displayName")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .unwrap_or(id)
            .to_owned();
        output.push(ModelEntry {
            id: id.to_owned(),
            object: "model",
            owned_by: "antigravity",
            kind: "antigravity",
            display_name: display_name.clone(),
            name: id.to_owned(),
            description: display_name,
            context_length: model
                .get("maxTokens")
                .and_then(Value::as_u64)
                .filter(|v| *v > 0),
            max_completion_tokens: model
                .get("maxOutputTokens")
                .and_then(Value::as_u64)
                .filter(|v| *v > 0),
        });
    }
    Ok(output)
}

pub fn run(options: &Options, deps: &Dependencies<'_>) -> Result<usize, CommandError> {
    check_cancelled(deps.cancellation)?;
    let auth = find_auth(&deps.auth_store.list().map_err(|_| CommandError::Store)?)?;
    deps.output.info(&format!("Using auth: id={}", auth.id));
    let models = match fetch_models(&auth, deps) {
        Ok(models) => models,
        Err(CommandError::Http) => {
            deps.output
                .warning("no models returned (API may be unavailable or token expired)");
            Vec::new()
        }
        Err(error) => return Err(error),
    };
    let payload = ModelOutput { models };
    let mut raw = if options.pretty {
        serde_json::to_vec_pretty(&payload)
    } else {
        serde_json::to_vec(&payload)
    }
    .map_err(|_| CommandError::InvalidResponse)?;
    if options.pretty {
        raw.push(b'\n');
    }
    check_cancelled(deps.cancellation)?;
    deps.files
        .write(Path::new(&options.output), &raw)
        .map_err(|_| CommandError::Write)?;
    deps.output
        .info(&format!("Fetched {} models.", payload.models.len()));
    Ok(payload.models.len())
}

fn check_cancelled(cancellation: &dyn Cancellation) -> Result<(), CommandError> {
    if cancellation.is_cancelled() {
        Err(CommandError::Cancelled)
    } else {
        Ok(())
    }
}

fn parse_options(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, CommandError> {
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--output" => {
                options.output = args.next().ok_or_else(|| {
                    CommandError::InvalidArguments("--output requires a value".to_owned())
                })?
            }
            "--pretty" => options.pretty = true,
            "--pretty=false" => options.pretty = false,
            _ => return Err(CommandError::InvalidArguments(arg)),
        }
    }
    if options.output.trim().is_empty() {
        return Err(CommandError::InvalidArguments(
            "output path is empty".to_owned(),
        ));
    }
    Ok(Some(options))
}

fn usage() -> &'static str {
    "Usage: cliproxy-fetch-antigravity-models [--output PATH] [--pretty|--pretty=false]\nCredentials and transports are supplied by the CTOX host."
}

pub fn standalone_main(args: impl IntoIterator<Item = String>) -> ExitCode {
    match parse_options(args) {
        Ok(None) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Some(options)) => standalone_run(&options),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn standalone_run(_options: &Options) -> ExitCode {
    eprintln!("error: this adapter requires an in-process CTOX auth/secret/http/file authority");
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_filters_upstream_catalog() {
        let models = parse_models(br#"{"models":{"chat_20706":{"displayName":"hidden"},"gemini-3-pro":{"displayName":"Gemini 3","maxTokens":100,"maxOutputTokens":20}}}"#).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-3-pro");
        assert_eq!(models[0].context_length, Some(100));
    }

    #[test]
    fn options_do_not_accept_auth_directory_authority() {
        assert!(parse_options(["--auths-dir".to_owned(), "/tmp/auth".to_owned()]).is_err());
    }
}
