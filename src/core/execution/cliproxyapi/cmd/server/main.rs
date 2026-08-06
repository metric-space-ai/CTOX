// ref: cmd/server/main.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{
    fmt,
    path::{Path, PathBuf},
    process::ExitCode,
    time::SystemTime,
};

use ctox_cliproxyapi::internal::{config::SdkConfig, safemode::has_example_api_keys};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServerConfig {
    pub sdk: SdkConfig,
    pub port: u16,
    pub home_enabled: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Options {
    pub config: Option<PathBuf>,
    pub tui: bool,
    pub standalone: bool,
    pub local_model: bool,
    pub cloud_config_missing: bool,
    pub mode: CommandMode,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum CommandMode {
    #[default]
    Server,
    CodexLogin,
    CodexDeviceLogin,
    ClaudeLogin,
    AntigravityLogin,
    KimiLogin,
    XaiLogin,
    VertexImport(PathBuf),
}

impl CommandMode {
    fn is_one_shot(&self) -> bool {
        !matches!(self, Self::Server)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogUpdaterPlan {
    pub start_models: bool,
    pub start_codex_client: bool,
}

pub trait ConfigSource: Send + Sync {
    fn load(&self, path: &Path) -> Result<Option<ServerConfig>, CommandError>;
}

pub trait FileSystem: Send + Sync {
    fn working_directory(&self) -> Result<PathBuf, CommandError>;
}

pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}
pub trait Clock: Send + Sync {
    fn now(&self) -> SystemTime;
}
pub trait CommandOutput: Send + Sync {
    fn info(&self, message: &str);
    fn warning(&self, message: &str);
}

pub trait ServiceHost: Send + Sync {
    fn run(&self, request: ServiceRequest<'_>) -> Result<(), CommandError>;
}

#[derive(Clone)]
pub struct ServiceRequest<'a> {
    pub config: &'a ServerConfig,
    pub config_path: &'a Path,
    pub mode: &'a CommandMode,
    pub tui: bool,
    pub standalone: bool,
    pub safe_mode: bool,
    pub catalog_plan: CatalogUpdaterPlan,
    pub cancellation: &'a dyn Cancellation,
}

pub struct Dependencies<'a> {
    pub config: &'a dyn ConfigSource,
    pub files: &'a dyn FileSystem,
    pub service: &'a dyn ServiceHost,
    pub clock: &'a dyn Clock,
    pub cancellation: &'a dyn Cancellation,
    pub output: &'a dyn CommandOutput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandError {
    InvalidArguments(String),
    WorkingDirectory,
    ConfigRead,
    ConfigMissing,
    Cancelled,
    ServiceUnavailable,
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArguments(value) => write!(formatter, "invalid arguments: {value}"),
            Self::WorkingDirectory => formatter.write_str("cannot resolve working directory"),
            Self::ConfigRead => formatter.write_str("failed to load typed server config"),
            Self::ConfigMissing => formatter.write_str("server config is missing"),
            Self::Cancelled => formatter.write_str("command cancelled"),
            Self::ServiceUnavailable => formatter.write_str("server host authority is unavailable"),
        }
    }
}

fn should_enable_example_api_key_safe_mode(
    config: Option<&ServerConfig>,
    command_mode: bool,
    tui_mode: bool,
    standalone: bool,
    cloud_config_missing: bool,
    home_mode: bool,
) -> bool {
    let Some(config) = config else {
        return false;
    };
    if command_mode || home_mode || cloud_config_missing || (tui_mode && !standalone) {
        return false;
    }
    has_example_api_keys(&config.sdk.api_keys)
}

fn model_catalog_updater_plan(local_model: bool, home_enabled: bool) -> CatalogUpdaterPlan {
    if local_model {
        return CatalogUpdaterPlan {
            start_models: false,
            start_codex_client: false,
        };
    }
    CatalogUpdaterPlan {
        start_models: !home_enabled,
        start_codex_client: true,
    }
}

fn plugin_bootstrap_config_path(
    args: &[String],
    default_path: Option<&Path>,
    working_directory: &Path,
) -> PathBuf {
    let default = || {
        default_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| working_directory.join("config.yaml"))
    };
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            return default();
        }
        if arg == "-config" || arg == "--config" {
            return args
                .get(index + 1)
                .map(PathBuf::from)
                .unwrap_or_else(default);
        }
        if let Some(value) = arg
            .strip_prefix("-config=")
            .or_else(|| arg.strip_prefix("--config="))
        {
            return PathBuf::from(value);
        }
        index += 1;
    }
    default()
}

pub fn run(
    options: &Options,
    raw_args: &[String],
    deps: &Dependencies<'_>,
) -> Result<(), CommandError> {
    if deps.cancellation.is_cancelled() {
        return Err(CommandError::Cancelled);
    }
    let working_directory = deps.files.working_directory()?;
    let config_path =
        plugin_bootstrap_config_path(raw_args, options.config.as_deref(), &working_directory);
    let config = deps
        .config
        .load(&config_path)?
        .ok_or(CommandError::ConfigMissing)?;
    let home_mode = config.home_enabled;
    let safe_mode = should_enable_example_api_key_safe_mode(
        Some(&config),
        options.mode.is_one_shot(),
        options.tui,
        options.standalone,
        options.cloud_config_missing,
        home_mode,
    );
    if safe_mode {
        deps.output
            .warning("unsafe example API key configured; proxy endpoints are disabled");
    }
    let catalog_plan = model_catalog_updater_plan(options.local_model, home_mode);
    deps.output.info(&format!(
        "CLIProxyAPI Rust startup at {:?} on port {}",
        deps.clock.now(),
        config.port
    ));
    if deps.cancellation.is_cancelled() {
        return Err(CommandError::Cancelled);
    }
    deps.service.run(ServiceRequest {
        config: &config,
        config_path: &config_path,
        mode: &options.mode,
        tui: options.tui,
        standalone: options.standalone,
        safe_mode,
        catalog_plan,
        cancellation: deps.cancellation,
    })
}

fn parse_options(args: &[String]) -> Result<Option<Options>, CommandError> {
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let mut value = || {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| CommandError::InvalidArguments(format!("{arg} requires a value")))
        };
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--config" | "-config" => options.config = Some(PathBuf::from(value()?)),
            "--tui" => options.tui = true,
            "--standalone" => options.standalone = true,
            "--local-model" => options.local_model = true,
            "--cloud-config-missing" => options.cloud_config_missing = true,
            "--codex-login" => options.mode = CommandMode::CodexLogin,
            "--codex-device-login" => options.mode = CommandMode::CodexDeviceLogin,
            "--claude-login" => options.mode = CommandMode::ClaudeLogin,
            "--antigravity-login" => options.mode = CommandMode::AntigravityLogin,
            "--kimi-login" => options.mode = CommandMode::KimiLogin,
            "--xai-login" => options.mode = CommandMode::XaiLogin,
            "--vertex-import" => options.mode = CommandMode::VertexImport(PathBuf::from(value()?)),
            _ if arg.starts_with("--config=") => {
                options.config = Some(PathBuf::from(arg.trim_start_matches("--config=")))
            }
            _ => return Err(CommandError::InvalidArguments(arg.clone())),
        }
        index += 1;
    }
    if options.standalone && !options.tui {
        return Err(CommandError::InvalidArguments(
            "--standalone requires --tui".to_owned(),
        ));
    }
    Ok(Some(options))
}

fn usage() -> &'static str {
    "Usage: cliproxy-server [--config PATH] [--tui [--standalone]] [--local-model] [provider login flag]\nRuntime config, secrets, stores, transports, and service lifecycle are supplied by CTOX."
}

pub fn standalone_main(args: Vec<String>) -> ExitCode {
    match parse_options(&args) {
        Ok(None) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Some(options)) => {
            let _ = options;
            eprintln!("error: this adapter requires an in-process CTOX config/service authority");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
#[path = "main_test.rs"]
mod tests;
