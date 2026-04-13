use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};

use crate::{
    command_catalog::CommandEntry,
    installations::{Installation, LaunchTarget, RemoteInstanceSource},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    Tui,
    Command,
}

#[derive(Debug, Clone)]
pub struct SessionLaunch {
    pub kind: SessionKind,
    pub title: String,
    pub spec: SessionSpec,
}

#[derive(Debug, Clone)]
pub enum SessionSpec {
    Local(LaunchTarget),
    Remote(RemoteSessionRequest),
}

#[derive(Debug, Clone)]
pub struct RemoteSessionRequest {
    pub kind: SessionKind,
    pub signaling_urls: Vec<String>,
    pub auth_token: String,
    pub password: String,
    pub room_id: String,
    pub client_name: String,
    pub command_args: Vec<String>,
    pub title: String,
}

pub trait InstanceConnector {
    fn launch_tui(&self, installation: &Installation) -> Result<SessionLaunch>;
    fn launch_command(&self, installation: &Installation, command: &CommandEntry) -> Result<SessionLaunch>;
    fn launch_command_with_extra_args(
        &self,
        installation: &Installation,
        command: &CommandEntry,
        extra_args: &[String],
    ) -> Result<SessionLaunch>;
    fn launch_custom_command(
        &self,
        installation: &Installation,
        args: &[String],
    ) -> Result<SessionLaunch>;
}

#[derive(Debug, Default)]
pub struct LocalProcessConnector;

impl InstanceConnector for LocalProcessConnector {
    fn launch_tui(&self, installation: &Installation) -> Result<SessionLaunch> {
        let title = format!("{} · TUI", installation.display_name());
        Ok(SessionLaunch {
            kind: SessionKind::Tui,
            title: title.clone(),
            spec: session_spec_for_installation(installation, &["tui"], title)?,
        })
    }

    fn launch_command(&self, installation: &Installation, command: &CommandEntry) -> Result<SessionLaunch> {
        self.launch_command_with_extra_args(installation, command, &[])
    }

    fn launch_command_with_extra_args(
        &self,
        installation: &Installation,
        command: &CommandEntry,
        extra_args: &[String],
    ) -> Result<SessionLaunch> {
        let title = format!("{} · {}", installation.display_name(), command.title);
        let mut merged_args: Vec<&str> = command.args.to_vec();
        for arg in extra_args {
            merged_args.push(arg.as_str());
        }
        Ok(SessionLaunch {
            kind: SessionKind::Command,
            title: title.clone(),
            spec: session_spec_for_installation(installation, &merged_args, title)?,
        })
    }

    fn launch_custom_command(
        &self,
        installation: &Installation,
        args: &[String],
    ) -> Result<SessionLaunch> {
        let title = if args.is_empty() {
            installation.display_name()
        } else {
            format!("{} · {}", installation.display_name(), args.join(" "))
        };
        let merged_args: Vec<&str> = args.iter().map(String::as_str).collect();
        Ok(SessionLaunch {
            kind: SessionKind::Command,
            title: title.clone(),
            spec: session_spec_for_installation(installation, &merged_args, title)?,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RemoteConnectorStub {
    pub signaling_endpoint: String,
}

#[allow(dead_code)]
impl RemoteConnectorStub {
    pub fn description(&self) -> String {
        format!(
            "reserved for future signaling/WebRTC transport via {}",
            self.signaling_endpoint
        )
    }
}

pub fn repo_root_from_manifest_dir(manifest_dir: &Path) -> Option<PathBuf> {
    manifest_dir.parent().map(Path::to_path_buf)
}

fn session_spec_for_installation(
    installation: &Installation,
    args: &[&str],
    title: String,
) -> Result<SessionSpec> {
    if installation.is_remote() {
        let remote = resolve_remote_session_request(installation, args, title)?;
        return Ok(SessionSpec::Remote(RemoteSessionRequest {
            kind: remote.kind,
            signaling_urls: remote.signaling_urls,
            auth_token: remote.auth_token,
            password: remote.password,
            room_id: remote.room_id,
            client_name: remote.client_name,
            command_args: remote.command_args,
            title: remote.title,
        }));
    }

    Ok(SessionSpec::Local(installation.command_launch_target(args)?))
}

fn resolve_remote_session_request(
    installation: &Installation,
    args: &[&str],
    title: String,
) -> Result<RemoteSessionRequest> {
    let config_map = load_ctox_remote_env_map(installation).unwrap_or_default();
    let allow_desktop_fallback =
        installation.remote.instance_source == RemoteInstanceSource::InstallNew;
    let signaling_urls = config_map
        .get("CTOX_WEBRTC_SIGNALING_URL")
        .map(|value| split_csv_values(value))
        .filter(|values| !values.is_empty())
        .or_else(|| allow_desktop_fallback.then(|| installation.remote.signaling_urls.clone()))
        .unwrap_or_default();
    let auth_token = first_non_empty(
        config_map.get("CTOX_WEBRTC_TOKEN").map(String::as_str),
        allow_desktop_fallback.then_some(installation.remote.auth_token.as_str()),
    )
    .unwrap_or_default()
    .to_owned();
    let password = first_non_empty(
        config_map
            .get("CTOX_WEBRTC_PASSWORD")
            .map(String::as_str),
        allow_desktop_fallback.then_some(installation.remote.password.as_str()),
    )
    .unwrap_or_default()
    .to_owned();
    let room_id = first_non_empty(
        config_map.get("CTOX_WEBRTC_ROOM").map(String::as_str),
        allow_desktop_fallback.then_some(installation.remote.room_id.as_str()),
    )
    .unwrap_or_default()
    .to_owned();
    let client_name = first_non_empty(
        Some(installation.remote.client_name.as_str()),
        Some(installation.display_name().as_str()),
    )
    .unwrap_or("desktop")
    .to_owned();

    if signaling_urls.is_empty() || room_id.trim().is_empty() || password.trim().is_empty() {
        if allow_desktop_fallback {
            bail!(
                "remote CTOX settings are missing. Finish host setup or configure Signaling Server, Remote Room, and Remote Password in CTOX Settings / Communication"
            );
        }
        bail!(
            "remote CTOX settings are missing in the target CTOX root. Configure Signaling Server, Remote Room, and Remote Password in CTOX Settings / Communication"
        );
    }

    Ok(RemoteSessionRequest {
        kind: if args == ["tui"] {
            SessionKind::Tui
        } else {
            SessionKind::Command
        },
        signaling_urls,
        auth_token,
        password,
        room_id,
        client_name,
        command_args: args.iter().map(|value| (*value).to_owned()).collect(),
        title,
    })
}

fn load_ctox_remote_env_map(installation: &Installation) -> Result<BTreeMap<String, String>> {
    if let Some(root) = installation.root_path.as_ref() {
        return load_env_map_from_local_root(root);
    }

    match installation.remote.host_target {
        crate::installations::RemoteHostTarget::Localhost => {
            let root = expand_local_root(&installation.remote.install_root)?;
            load_env_map_from_local_root(&root)
        }
        crate::installations::RemoteHostTarget::Ssh => {
            load_env_map_over_ssh(
                &installation.remote.ssh_user,
                &installation.remote.ssh_host,
                installation.remote.ssh_port,
                &installation.remote.ssh_password,
                &installation.remote.install_root,
            )
        }
        crate::installations::RemoteHostTarget::Unspecified => Ok(BTreeMap::new()),
    }
}

fn load_env_map_from_local_root(root: &Path) -> Result<BTreeMap<String, String>> {
    let path = root.join("runtime/engine.env");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(parse_env_map(&raw))
}

fn load_env_map_over_ssh(
    user: &str,
    host: &str,
    port: u16,
    password: &str,
    install_root: &str,
) -> Result<BTreeMap<String, String>> {
    if user.trim().is_empty()
        || host.trim().is_empty()
        || password.trim().is_empty()
        || install_root.trim().is_empty()
    {
        return Ok(BTreeMap::new());
    }

    let remote_root = remote_path_expr(install_root);
    let script = format!(
        r#"
set timeout 20
spawn ssh -p {port} {target} "cat {root}/runtime/engine.env 2>/dev/null || true"
expect {{
  "*password:*" {{ send "$env(CTOX_SSH_PASSWORD)\r"; exp_continue }}
  eof
}}
"#,
        port = port,
        target = format!("{}@{}", user.trim(), host.trim()),
        root = remote_root,
    );
    let output = Command::new("expect")
        .arg("-c")
        .arg(script)
        .env("CTOX_SSH_PASSWORD", password.trim())
        .output()
        .context("failed to load remote CTOX settings over SSH")?;
    if !output.status.success() {
        return Ok(BTreeMap::new());
    }
    Ok(parse_env_map(&String::from_utf8_lossy(&output.stdout)))
}

fn parse_env_map(raw: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        out.insert(key.to_owned(), value.trim().to_owned());
    }
    out
}

fn split_csv_values(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn first_non_empty<'a>(primary: Option<&'a str>, fallback: Option<&'a str>) -> Option<&'a str> {
    primary
        .filter(|value| !value.trim().is_empty())
        .or_else(|| fallback.filter(|value| !value.trim().is_empty()))
}

fn expand_local_root(raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("missing local CTOX root");
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = dirs::home_dir().context("home directory is not available")?;
        return Ok(home.join(rest));
    }
    Ok(PathBuf::from(trimmed))
}

fn remote_path_expr(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("~/") {
        format!("$HOME/{}", rest)
    } else {
        shell_quote(trimmed)
    }
}

fn shell_quote(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}
