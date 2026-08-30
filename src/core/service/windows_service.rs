use anyhow::{Context, Result};
use serde::Serialize;
use std::ffi::{c_void, OsStr};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;
use std::thread;
use windows_sys::Win32::System::Services::{
    RegisterServiceCtrlHandlerExW, SetServiceStatus, StartServiceCtrlDispatcherW,
    SERVICE_ACCEPT_SHUTDOWN, SERVICE_ACCEPT_STOP, SERVICE_CONTROL_SHUTDOWN, SERVICE_CONTROL_STOP,
    SERVICE_RUNNING, SERVICE_START_PENDING, SERVICE_STATUS, SERVICE_STATUS_HANDLE, SERVICE_STOPPED,
    SERVICE_STOP_PENDING, SERVICE_TABLE_ENTRYW, SERVICE_WIN32_OWN_PROCESS,
};

const SERVICE_NAME: &str = "CTOX";
const DISPLAY_NAME: &str = "CTOX Backend";
static SERVICE_ROOT: OnceLock<PathBuf> = OnceLock::new();
static STATUS_HANDLE: AtomicPtr<c_void> = AtomicPtr::new(null_mut());

#[derive(Debug, Clone, Serialize)]
pub struct WindowsServiceStatus {
    pub installed: bool,
    pub running: bool,
    pub service_name: &'static str,
}

fn wide(value: impl AsRef<OsStr>) -> Vec<u16> {
    value.as_ref().encode_wide().chain(Some(0)).collect()
}

fn sc(args: &[&str]) -> Result<Output> {
    Command::new("sc.exe")
        .args(args)
        .output()
        .with_context(|| format!("failed to run sc.exe {}", args.join(" ")))
}

fn sc_success(args: &[&str]) -> Result<String> {
    let output = sc(args)?;
    if !output.status.success() {
        anyhow::bail!(
            "sc.exe {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn status() -> Result<WindowsServiceStatus> {
    let output = sc(&["query", SERVICE_NAME])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(WindowsServiceStatus {
        installed: output.status.success(),
        running: output.status.success() && stdout.contains("RUNNING"),
        service_name: SERVICE_NAME,
    })
}

pub fn install_current_executable(root: &Path) -> Result<WindowsServiceStatus> {
    let executable = std::env::current_exe().context("failed to resolve CTOX executable")?;
    let executable = executable
        .to_str()
        .context("CTOX executable path is not valid Unicode")?;
    let root = root
        .to_str()
        .context("CTOX root path is not valid Unicode")?;
    anyhow::ensure!(
        !executable.contains('"') && !root.contains('"'),
        "CTOX service paths must not contain a quote"
    );
    let bin_path = format!("\"{executable}\" service --windows-service --root \"{root}\"");
    if status()?.installed {
        sc_success(&[
            "config",
            SERVICE_NAME,
            "binPath=",
            &bin_path,
            "start=",
            "auto",
            "DisplayName=",
            DISPLAY_NAME,
        ])?;
    } else {
        sc_success(&[
            "create",
            SERVICE_NAME,
            "binPath=",
            &bin_path,
            "start=",
            "auto",
            "DisplayName=",
            DISPLAY_NAME,
        ])?;
    }
    sc_success(&[
        "failure",
        SERVICE_NAME,
        "reset=",
        "86400",
        "actions=",
        "restart/5000/restart/15000/restart/60000",
    ])?;
    status()
}

pub fn start() -> Result<()> {
    if status()?.running {
        return Ok(());
    }
    sc_success(&["start", SERVICE_NAME])?;
    Ok(())
}

pub fn stop() -> Result<()> {
    if !status()?.running {
        return Ok(());
    }
    sc_success(&["stop", SERVICE_NAME])?;
    Ok(())
}

fn report_status(state: u32, controls: u32, exit_code: u32, wait_hint: u32) {
    let handle = STATUS_HANDLE.load(Ordering::SeqCst) as SERVICE_STATUS_HANDLE;
    if handle.is_null() {
        return;
    }
    let status = SERVICE_STATUS {
        dwServiceType: SERVICE_WIN32_OWN_PROCESS,
        dwCurrentState: state,
        dwControlsAccepted: controls,
        dwWin32ExitCode: exit_code,
        dwServiceSpecificExitCode: 0,
        dwCheckPoint: 0,
        dwWaitHint: wait_hint,
    };
    unsafe {
        SetServiceStatus(handle, &status);
    }
}

unsafe extern "system" fn control_handler(
    control: u32,
    _event_type: u32,
    _event_data: *mut c_void,
    _context: *mut c_void,
) -> u32 {
    if matches!(control, SERVICE_CONTROL_STOP | SERVICE_CONTROL_SHUTDOWN) {
        report_status(SERVICE_STOP_PENDING, 0, 0, 15_000);
        if let Some(root) = SERVICE_ROOT.get().cloned() {
            thread::spawn(move || {
                let _ = crate::service::request_windows_service_shutdown(&root);
            });
        }
    }
    0
}

unsafe extern "system" fn service_main(_argc: u32, _argv: *mut *mut u16) {
    let name = wide(SERVICE_NAME);
    let handle =
        unsafe { RegisterServiceCtrlHandlerExW(name.as_ptr(), Some(control_handler), null()) };
    if handle.is_null() {
        return;
    }
    STATUS_HANDLE.store(handle, Ordering::SeqCst);
    report_status(SERVICE_START_PENDING, 0, 0, 30_000);
    let Some(root) = SERVICE_ROOT.get() else {
        report_status(SERVICE_STOPPED, 0, 2, 0);
        return;
    };
    report_status(
        SERVICE_RUNNING,
        SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN,
        0,
        0,
    );
    let exit_code = if crate::service::run_foreground(root).is_ok() {
        0
    } else {
        1
    };
    report_status(SERVICE_STOPPED, 0, exit_code, 0);
}

pub fn run_dispatcher(root: &Path) -> Result<()> {
    SERVICE_ROOT
        .set(root.to_path_buf())
        .map_err(|_| anyhow::anyhow!("CTOX Windows service root was already initialized"))?;
    let mut name = wide(SERVICE_NAME);
    let table = [
        SERVICE_TABLE_ENTRYW {
            lpServiceName: name.as_mut_ptr(),
            lpServiceProc: Some(service_main),
        },
        SERVICE_TABLE_ENTRYW {
            lpServiceName: null_mut(),
            lpServiceProc: None,
        },
    ];
    let started = unsafe { StartServiceCtrlDispatcherW(table.as_ptr()) };
    if started == 0 {
        return Err(std::io::Error::last_os_error())
            .context("failed to enter Windows SCM dispatcher");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::wide;

    #[test]
    fn wide_strings_are_nul_terminated() {
        let encoded = wide("CTOX");
        assert_eq!(encoded.last(), Some(&0));
        assert_eq!(encoded.iter().filter(|value| **value == 0).count(), 1);
    }
}
