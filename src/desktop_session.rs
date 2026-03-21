use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DesktopSessionInfo {
    pub display: Option<String>,
    pub wayland_display: Option<String>,
    pub xauthority: Option<String>,
    pub dbus_session_bus_address: Option<String>,
    pub xdg_runtime_dir: Option<String>,
    pub xdg_session_type: Option<String>,
    pub xdg_current_desktop: Option<String>,
}

impl DesktopSessionInfo {
    pub fn is_available(&self) -> bool {
        self.display.as_deref().is_some() || self.wayland_display.as_deref().is_some()
    }

    pub fn env_overrides(&self) -> Option<HashMap<String, Option<String>>> {
        let mut env = HashMap::new();
        push_env(&mut env, "DISPLAY", self.display.as_deref());
        push_env(&mut env, "WAYLAND_DISPLAY", self.wayland_display.as_deref());
        push_env(&mut env, "XAUTHORITY", self.xauthority.as_deref());
        push_env(
            &mut env,
            "DBUS_SESSION_BUS_ADDRESS",
            self.dbus_session_bus_address.as_deref(),
        );
        push_env(&mut env, "XDG_RUNTIME_DIR", self.xdg_runtime_dir.as_deref());
        push_env(
            &mut env,
            "XDG_SESSION_TYPE",
            self.xdg_session_type.as_deref(),
        );
        push_env(
            &mut env,
            "XDG_CURRENT_DESKTOP",
            self.xdg_current_desktop.as_deref(),
        );
        if env.is_empty() { None } else { Some(env) }
    }

    fn merge_missing(&mut self, other: DesktopSessionInfo) {
        if self.display.is_none() {
            self.display = other.display;
        }
        if self.wayland_display.is_none() {
            self.wayland_display = other.wayland_display;
        }
        if self.xauthority.is_none() {
            self.xauthority = other.xauthority;
        }
        if self.dbus_session_bus_address.is_none() {
            self.dbus_session_bus_address = other.dbus_session_bus_address;
        }
        if self.xdg_runtime_dir.is_none() {
            self.xdg_runtime_dir = other.xdg_runtime_dir;
        }
        if self.xdg_session_type.is_none() {
            self.xdg_session_type = other.xdg_session_type;
        }
        if self.xdg_current_desktop.is_none() {
            self.xdg_current_desktop = other.xdg_current_desktop;
        }
    }
}

pub fn detect_desktop_session() -> DesktopSessionInfo {
    #[cfg(target_os = "macos")]
    {
        return DesktopSessionInfo {
            xdg_session_type: Some("cocoa".to_string()),
            ..DesktopSessionInfo::default()
        };
    }

    #[cfg(target_os = "linux")]
    {
        return detect_linux_desktop_session();
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        DesktopSessionInfo::default()
    }
}

pub fn detect_desktop_session_env() -> Option<HashMap<String, Option<String>>> {
    detect_desktop_session().env_overrides()
}

#[cfg(target_os = "linux")]
fn detect_linux_desktop_session() -> DesktopSessionInfo {
    let mut info = current_process_session();
    if info.is_available() {
        return finalize_linux_session(info);
    }

    info.merge_missing(active_logind_session());
    info.merge_missing(process_scan_session());
    finalize_linux_session(info)
}

#[cfg(target_os = "linux")]
fn current_process_session() -> DesktopSessionInfo {
    DesktopSessionInfo {
        display: non_empty_env("DISPLAY"),
        wayland_display: non_empty_env("WAYLAND_DISPLAY"),
        xauthority: existing_file_env("XAUTHORITY"),
        dbus_session_bus_address: non_empty_env("DBUS_SESSION_BUS_ADDRESS"),
        xdg_runtime_dir: existing_dir_env("XDG_RUNTIME_DIR"),
        xdg_session_type: non_empty_env("XDG_SESSION_TYPE"),
        xdg_current_desktop: non_empty_env("XDG_CURRENT_DESKTOP"),
    }
}

#[cfg(target_os = "linux")]
fn active_logind_session() -> DesktopSessionInfo {
    let Some(uid) = current_uid_string() else {
        return DesktopSessionInfo::default();
    };
    let session_ids = list_logind_session_ids();
    let mut best: Option<(usize, DesktopSessionInfo)> = None;
    for session_id in session_ids {
        let Some(props) = load_logind_session(&session_id) else {
            continue;
        };
        if props.user.as_deref() != Some(uid.as_str()) {
            continue;
        }
        if props.remote.as_deref() == Some("yes") {
            continue;
        }
        if props.state.as_deref() != Some("active") {
            continue;
        }
        let mut info = DesktopSessionInfo {
            display: props.display.clone(),
            wayland_display: None,
            xauthority: None,
            dbus_session_bus_address: None,
            xdg_runtime_dir: None,
            xdg_session_type: props.session_type.clone(),
            xdg_current_desktop: props.desktop.clone(),
        };
        if let Some(pid) = props.leader_pid {
            info.merge_missing(load_proc_session_env(pid));
        }
        let score = logind_session_score(&info);
        match &best {
            Some((best_score, _)) if *best_score >= score => {}
            _ => best = Some((score, info)),
        }
    }
    best.map(|(_, info)| info).unwrap_or_default()
}

#[cfg(target_os = "linux")]
fn process_scan_session() -> DesktopSessionInfo {
    let Some(username) = current_username() else {
        return DesktopSessionInfo::default();
    };
    let Ok(output) = Command::new("ps").args(["eww", "-u", &username]).output() else {
        return DesktopSessionInfo::default();
    };
    if !output.status.success() {
        return DesktopSessionInfo::default();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    DesktopSessionInfo {
        display: env_value_from_ps_output(&text, "DISPLAY"),
        wayland_display: env_value_from_ps_output(&text, "WAYLAND_DISPLAY"),
        xauthority: env_value_from_ps_output(&text, "XAUTHORITY")
            .filter(|value| Path::new(value).is_file()),
        dbus_session_bus_address: env_value_from_ps_output(&text, "DBUS_SESSION_BUS_ADDRESS"),
        xdg_runtime_dir: env_value_from_ps_output(&text, "XDG_RUNTIME_DIR")
            .filter(|value| Path::new(value).is_dir()),
        xdg_session_type: env_value_from_ps_output(&text, "XDG_SESSION_TYPE"),
        xdg_current_desktop: env_value_from_ps_output(&text, "XDG_CURRENT_DESKTOP"),
    }
}

#[cfg(target_os = "linux")]
fn finalize_linux_session(mut info: DesktopSessionInfo) -> DesktopSessionInfo {
    if info.xdg_runtime_dir.is_none() {
        info.xdg_runtime_dir = runtime_dir_fallback();
    }
    if info.dbus_session_bus_address.is_none() {
        if let Some(runtime_dir) = info.xdg_runtime_dir.as_deref() {
            let bus_path = format!("{runtime_dir}/bus");
            if Path::new(&bus_path).exists() {
                info.dbus_session_bus_address = Some(format!("unix:path={bus_path}"));
            }
        }
    }
    if info.xauthority.is_none() {
        let mut candidates = Vec::new();
        if let Some(home) = std::env::var("HOME")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            candidates.push(Path::new(&home).join(".Xauthority"));
        }
        if let Some(runtime_dir) = info.xdg_runtime_dir.as_deref() {
            candidates.push(Path::new(runtime_dir).join("gdm/Xauthority"));
            candidates.push(Path::new(runtime_dir).join(".Xauthority"));
        }
        for candidate in candidates {
            if candidate.is_file() {
                info.xauthority = Some(candidate.display().to_string());
                break;
            }
        }
    }
    if info.display.is_none() {
        info.display = first_x11_display();
    }
    if info.wayland_display.is_none() {
        if let Some(runtime_dir) = info.xdg_runtime_dir.as_deref() {
            info.wayland_display = first_wayland_display(runtime_dir);
        }
    }
    info
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Default)]
struct LogindSessionProperties {
    user: Option<String>,
    state: Option<String>,
    remote: Option<String>,
    display: Option<String>,
    session_type: Option<String>,
    desktop: Option<String>,
    leader_pid: Option<u32>,
}

#[cfg(target_os = "linux")]
fn list_logind_session_ids() -> Vec<String> {
    let Ok(output) = Command::new("loginctl")
        .args(["list-sessions", "--no-legend"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

#[cfg(target_os = "linux")]
fn load_logind_session(session_id: &str) -> Option<LogindSessionProperties> {
    let output = Command::new("loginctl")
        .args([
            "show-session",
            session_id,
            "--property=User",
            "--property=State",
            "--property=Remote",
            "--property=Display",
            "--property=Type",
            "--property=Desktop",
            "--property=Leader",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_logind_session_properties(
        String::from_utf8_lossy(&output.stdout).as_ref(),
    ))
}

#[cfg(target_os = "linux")]
fn parse_logind_session_properties(text: &str) -> LogindSessionProperties {
    let mut props = LogindSessionProperties::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let trimmed = value.trim();
        match key.trim() {
            "User" if !trimmed.is_empty() => props.user = Some(trimmed.to_string()),
            "State" if !trimmed.is_empty() => props.state = Some(trimmed.to_string()),
            "Remote" if !trimmed.is_empty() => props.remote = Some(trimmed.to_string()),
            "Display" if !trimmed.is_empty() => props.display = Some(trimmed.to_string()),
            "Type" if !trimmed.is_empty() => props.session_type = Some(trimmed.to_string()),
            "Desktop" if !trimmed.is_empty() => props.desktop = Some(trimmed.to_string()),
            "Leader" => props.leader_pid = trimmed.parse::<u32>().ok(),
            _ => {}
        }
    }
    props
}

#[cfg(target_os = "linux")]
fn logind_session_score(info: &DesktopSessionInfo) -> usize {
    let mut score = 0;
    if info.display.is_some() || info.wayland_display.is_some() {
        score += 8;
    }
    if info.dbus_session_bus_address.is_some() {
        score += 4;
    }
    if info.xdg_runtime_dir.is_some() {
        score += 3;
    }
    if info.xauthority.is_some() {
        score += 2;
    }
    if info.xdg_session_type.is_some() {
        score += 2;
    }
    if info.xdg_current_desktop.is_some() {
        score += 1;
    }
    score
}

#[cfg(target_os = "linux")]
fn load_proc_session_env(pid: u32) -> DesktopSessionInfo {
    let path = format!("/proc/{pid}/environ");
    let Ok(bytes) = fs::read(path) else {
        return DesktopSessionInfo::default();
    };
    let env = parse_environ_bytes(&bytes);
    DesktopSessionInfo {
        display: env.get("DISPLAY").cloned(),
        wayland_display: env.get("WAYLAND_DISPLAY").cloned(),
        xauthority: env
            .get("XAUTHORITY")
            .cloned()
            .filter(|value| Path::new(value).is_file()),
        dbus_session_bus_address: env.get("DBUS_SESSION_BUS_ADDRESS").cloned(),
        xdg_runtime_dir: env
            .get("XDG_RUNTIME_DIR")
            .cloned()
            .filter(|value| Path::new(value).is_dir()),
        xdg_session_type: env.get("XDG_SESSION_TYPE").cloned(),
        xdg_current_desktop: env.get("XDG_CURRENT_DESKTOP").cloned(),
    }
}

#[cfg(target_os = "linux")]
fn parse_environ_bytes(bytes: &[u8]) -> HashMap<String, String> {
    bytes
        .split(|byte| *byte == 0)
        .filter_map(|entry| std::str::from_utf8(entry).ok())
        .filter_map(|entry| entry.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

#[cfg(target_os = "linux")]
fn env_value_from_ps_output(text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    for line in text.lines() {
        for token in line.split_whitespace() {
            if let Some(value) = token.strip_prefix(&prefix) {
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn runtime_dir_fallback() -> Option<String> {
    let uid = current_uid_string()?;
    let candidate = format!("/run/user/{uid}");
    Path::new(&candidate).is_dir().then_some(candidate)
}

#[cfg(target_os = "linux")]
fn first_x11_display() -> Option<String> {
    let dir = fs::read_dir("/tmp/.X11-unix").ok()?;
    let mut displays = dir
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| name.strip_prefix('X').map(str::to_string))
        .collect::<Vec<_>>();
    displays.sort();
    displays.first().map(|value| format!(":{value}"))
}

#[cfg(target_os = "linux")]
fn first_wayland_display(runtime_dir: &str) -> Option<String> {
    let dir = fs::read_dir(runtime_dir).ok()?;
    let mut displays = dir
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with("wayland-"))
        .collect::<Vec<_>>();
    displays.sort();
    displays.first().cloned()
}

#[cfg(target_os = "linux")]
fn current_uid_string() -> Option<String> {
    let output = Command::new("id").arg("-u").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let uid = String::from_utf8(output.stdout).ok()?;
    let uid = uid.trim();
    (!uid.is_empty()).then_some(uid.to_string())
}

#[cfg(target_os = "linux")]
fn current_username() -> Option<String> {
    if let Some(value) = std::env::var("USER")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return Some(value);
    }
    let output = Command::new("id").arg("-un").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let username = String::from_utf8(output.stdout).ok()?;
    let username = username.trim();
    (!username.is_empty()).then_some(username.to_string())
}

#[cfg(target_os = "linux")]
fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "linux")]
fn existing_file_env(key: &str) -> Option<String> {
    non_empty_env(key).filter(|value| Path::new(value).is_file())
}

#[cfg(target_os = "linux")]
fn existing_dir_env(key: &str) -> Option<String> {
    non_empty_env(key).filter(|value| Path::new(value).is_dir())
}

fn push_env(env: &mut HashMap<String, Option<String>>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        env.insert(key.to_string(), Some(value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_overrides_only_include_present_values() {
        let info = DesktopSessionInfo {
            display: Some(":0".to_string()),
            xauthority: Some("/home/test/.Xauthority".to_string()),
            xdg_session_type: Some("x11".to_string()),
            ..DesktopSessionInfo::default()
        };
        let env = info.env_overrides().expect("env overrides");
        assert_eq!(env.get("DISPLAY"), Some(&Some(":0".to_string())));
        assert_eq!(
            env.get("XAUTHORITY"),
            Some(&Some("/home/test/.Xauthority".to_string()))
        );
        assert_eq!(env.get("XDG_SESSION_TYPE"), Some(&Some("x11".to_string())));
        assert!(!env.contains_key("WAYLAND_DISPLAY"));
        assert!(!env.contains_key("DBUS_SESSION_BUS_ADDRESS"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_logind_session_properties_extracts_fields() {
        let props = parse_logind_session_properties(
            "User=1000\nState=active\nRemote=no\nDisplay=:0\nType=x11\nDesktop=KDE\nLeader=4262\n",
        );
        assert_eq!(props.user.as_deref(), Some("1000"));
        assert_eq!(props.state.as_deref(), Some("active"));
        assert_eq!(props.remote.as_deref(), Some("no"));
        assert_eq!(props.display.as_deref(), Some(":0"));
        assert_eq!(props.session_type.as_deref(), Some("x11"));
        assert_eq!(props.desktop.as_deref(), Some("KDE"));
        assert_eq!(props.leader_pid, Some(4262));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_environ_bytes_reads_null_separated_pairs() {
        let env = parse_environ_bytes(
            b"DISPLAY=:0\0DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus\0XDG_CURRENT_DESKTOP=KDE\0",
        );
        assert_eq!(env.get("DISPLAY").map(String::as_str), Some(":0"));
        assert_eq!(
            env.get("DBUS_SESSION_BUS_ADDRESS").map(String::as_str),
            Some("unix:path=/run/user/1000/bus")
        );
        assert_eq!(
            env.get("XDG_CURRENT_DESKTOP").map(String::as_str),
            Some("KDE")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn env_value_from_ps_output_finds_key_tokens() {
        let value = env_value_from_ps_output(
            "4262 /usr/bin/startplasma-x11 DISPLAY=:0 XAUTHORITY=/home/metricspace/.Xauthority XDG_SESSION_TYPE=x11",
            "XAUTHORITY",
        );
        assert_eq!(value.as_deref(), Some("/home/metricspace/.Xauthority"));
    }
}
