// ref: cmd/fetch_codex_models/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{collections::BTreeMap, fmt, path::Path, process::ExitCode, time::Duration};

use chrono::{DateTime, Utc};
use ctox_cliproxyapi::sdk::cliproxy::auth::{Auth, AuthStore};
use serde_json::Value;
use url::Url;

const CODEX_MODELS_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const CODEX_MODELS_PATH: &str = "/models";
const DEFAULT_CLIENT_VERSION: &str = "0.144.1";
const DEFAULT_CODEX_USER_AGENT: &str =
    "codex_cli_rs/0.144.1 (Mac OS 26.3.1; arm64) iTerm.app/3.6.9";
const DEFAULT_CODEX_ORIGINATOR: &str = "codex_cli_rs";
const ACCESS_TOKEN_REFRESH_LEEWAY: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    pub output: String,
    pub client_version: String,
    pub pretty: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            output: "codex_client_models.json".to_owned(),
            client_version: DEFAULT_CLIENT_VERSION.to_owned(),
            pretty: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRequest {
    pub url: Url,
    pub headers: BTreeMap<String, String>,
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
    fn store(&self, auth: &Auth, name: &'static str, value: &str) -> Result<(), CommandError>;
}

pub trait TokenRefresher: Send + Sync {
    fn refresh(
        &self,
        auth: &Auth,
        refresh_token: &str,
        cancelled: &dyn Cancellation,
    ) -> Result<RefreshedTokens, CommandError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshedTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

pub trait FileOutput: Send + Sync {
    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CommandError>;
}

pub trait CommandOutput: Send + Sync {
    fn info(&self, message: &str);
}

pub struct Dependencies<'a> {
    pub auth_store: &'a dyn AuthStore,
    pub secrets: &'a dyn SecretResolver,
    pub refresher: &'a dyn TokenRefresher,
    pub http: &'a dyn HttpTransport,
    pub clock: &'a dyn Clock,
    pub cancellation: &'a dyn Cancellation,
    pub files: &'a dyn FileOutput,
    pub output: &'a dyn CommandOutput,
    pub request_timeout: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandError {
    Cancelled,
    Store,
    NoCodexAuth,
    MissingTokens,
    InvalidRefresh,
    InvalidUrl,
    Http(String),
    InvalidResponse(String),
    Write,
    InvalidArguments(String),
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("command cancelled"),
            Self::Store => formatter.write_str("auth store operation failed"),
            Self::NoCodexAuth => formatter.write_str("no enabled codex auth found"),
            Self::MissingTokens => formatter.write_str("missing access_token and refresh_token"),
            Self::InvalidRefresh => {
                formatter.write_str("refresh response did not include access_token")
            }
            Self::InvalidUrl => formatter.write_str("invalid Codex models URL"),
            Self::Http(message) => write!(formatter, "models request failed: {message}"),
            Self::InvalidResponse(message) => {
                write!(formatter, "invalid models response: {message}")
            }
            Self::Write => formatter.write_str("failed to write output"),
            Self::InvalidArguments(message) => write!(formatter, "invalid arguments: {message}"),
        }
    }
}

fn find_codex_auth(auths: &[Auth], secrets: &dyn SecretResolver) -> Result<Auth, CommandError> {
    for auth in auths {
        if auth.disabled || !auth.provider.trim().eq_ignore_ascii_case("codex") {
            continue;
        }
        if secrets.resolve(auth, "access_token")?.is_some()
            || secrets.resolve(auth, "refresh_token")?.is_some()
        {
            return Ok(auth.clone());
        }
    }
    Err(CommandError::NoCodexAuth)
}

fn ensure_access_token(
    auth: &mut Auth,
    deps: &Dependencies<'_>,
) -> Result<(String, bool), CommandError> {
    check_cancelled(deps.cancellation)?;
    let access_token = deps.secrets.resolve(auth, "access_token")?;
    let refresh_due = auth
        .expiration_time()
        .is_some_and(|expires| deps.clock.now() + ACCESS_TOKEN_REFRESH_LEEWAY >= expires);
    if let Some(token) = access_token.as_ref().filter(|_| !refresh_due) {
        return Ok((token.clone(), false));
    }
    let Some(refresh_token) = deps.secrets.resolve(auth, "refresh_token")? else {
        return access_token
            .map(|token| (token, false))
            .ok_or(CommandError::MissingTokens);
    };
    let refreshed = deps
        .refresher
        .refresh(auth, &refresh_token, deps.cancellation)?;
    if refreshed.access_token.trim().is_empty() {
        return Err(CommandError::InvalidRefresh);
    }
    deps.secrets
        .store(auth, "access_token", refreshed.access_token.trim())?;
    if let Some(refresh_token) = refreshed
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        deps.secrets.store(auth, "refresh_token", refresh_token)?;
    }
    auth.last_refreshed_at = deps.clock.now();
    if let Some(expires_at) = refreshed.expires_at {
        auth.metadata
            .insert("expired".to_owned(), Value::String(expires_at.to_rfc3339()));
    }
    deps.auth_store
        .save(auth)
        .map_err(|_| CommandError::Store)?;
    Ok((refreshed.access_token, true))
}

fn codex_models_url(client_version: &str) -> Result<Url, CommandError> {
    let mut url = Url::parse(&format!("{CODEX_MODELS_BASE_URL}{CODEX_MODELS_PATH}"))
        .map_err(|_| CommandError::InvalidUrl)?;
    let version = client_version.trim();
    if !version.is_empty() {
        url.query_pairs_mut().append_pair("client_version", version);
    }
    Ok(url)
}

fn count_models(raw: &[u8]) -> Result<usize, CommandError> {
    let value: Value = serde_json::from_slice(raw)
        .map_err(|error| CommandError::InvalidResponse(error.to_string()))?;
    value
        .get("models")
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| {
            CommandError::InvalidResponse("response JSON does not contain models array".to_owned())
        })
}

fn pretty_json(raw: &[u8]) -> Result<Vec<u8>, CommandError> {
    let value: Value = serde_json::from_slice(raw)
        .map_err(|error| CommandError::InvalidResponse(error.to_string()))?;
    let mut output = serde_json::to_vec_pretty(&value)
        .map_err(|error| CommandError::InvalidResponse(error.to_string()))?;
    output.push(b'\n');
    Ok(output)
}

fn fetch_models(
    auth: &Auth,
    access_token: &str,
    client_version: &str,
    deps: &Dependencies<'_>,
) -> Result<(Vec<u8>, usize), CommandError> {
    check_cancelled(deps.cancellation)?;
    let mut headers = BTreeMap::from([
        ("Accept".to_owned(), "application/json".to_owned()),
        ("Authorization".to_owned(), format!("Bearer {access_token}")),
        ("Originator".to_owned(), DEFAULT_CODEX_ORIGINATOR.to_owned()),
        ("User-Agent".to_owned(), DEFAULT_CODEX_USER_AGENT.to_owned()),
    ]);
    if let Some(account_id) = auth.metadata.get("account_id").and_then(Value::as_str) {
        if !account_id.trim().is_empty() {
            headers.insert(
                "Chatgpt-Account-Id".to_owned(),
                account_id.trim().to_owned(),
            );
        }
    }
    for (name, value) in &auth.attributes {
        if let Some(header) = name.strip_prefix("header:") {
            if !header.trim().is_empty() && !value.contains(['\r', '\n']) {
                headers.insert(header.trim().to_owned(), value.clone());
            }
        }
    }
    let response = deps.http.execute(HttpRequest {
        url: codex_models_url(client_version)?,
        headers,
        timeout: deps.request_timeout,
    })?;
    if !(200..300).contains(&response.status) {
        return Err(CommandError::Http(format!("status {}", response.status)));
    }
    let count = count_models(&response.body)?;
    Ok((response.body, count))
}

pub fn run(options: &Options, deps: &Dependencies<'_>) -> Result<usize, CommandError> {
    check_cancelled(deps.cancellation)?;
    let auths = deps.auth_store.list().map_err(|_| CommandError::Store)?;
    let mut auth = find_codex_auth(&auths, deps.secrets)?;
    deps.output.info(&format!("Using auth: id={}", auth.id));
    let (access_token, refreshed) = ensure_access_token(&mut auth, deps)?;
    if refreshed {
        deps.output.info("Refreshed Codex access token.");
    }
    let (mut raw, count) = fetch_models(&auth, &access_token, &options.client_version, deps)?;
    if options.pretty {
        raw = pretty_json(&raw)?;
    }
    check_cancelled(deps.cancellation)?;
    deps.files
        .write(Path::new(&options.output), &raw)
        .map_err(|_| CommandError::Write)?;
    deps.output.info(&format!("Fetched {count} models."));
    Ok(count)
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
            "--client-version" => {
                options.client_version = args.next().ok_or_else(|| {
                    CommandError::InvalidArguments("--client-version requires a value".to_owned())
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
    "Usage: cliproxy-fetch-codex-models [--output PATH] [--client-version VERSION] [--pretty|--pretty=false]\nCredentials and transports are supplied by the CTOX host."
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
#[path = "main_test.rs"]
mod tests;
