use anyhow::Context;
use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const DEFAULT_REFERENCE_RELATIVE_DIR: &str = "runtime/browser/interactive-reference";

#[derive(Debug, Clone, Serialize)]
struct ToolStatus {
    available: bool,
    path: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BrowserDoctorReport {
    ok: bool,
    reference_dir: PathBuf,
    package_json_exists: bool,
    node_modules_exists: bool,
    playwright_dependency_declared: bool,
    chromium_fallback_executable: Option<String>,
    toolchain: serde_json::Value,
    codex_config_overrides: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BrowserInstallReport {
    ok: bool,
    reference_dir: PathBuf,
    package_json_created: bool,
    npm_install_ran: bool,
    browser_install_ran: bool,
    codex_config_overrides: Vec<String>,
}

pub fn handle_browser_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "doctor" => {
            let reference_dir = resolve_reference_dir(root, &args[1..]);
            let report = build_doctor_report(&reference_dir)?;
            print_json(&serde_json::to_value(report)?)
        }
        "install-reference" => {
            let reference_dir = resolve_reference_dir(root, &args[1..]);
            let install_browser = args.iter().any(|arg| arg == "--install-browser");
            let skip_npm_install = args.iter().any(|arg| arg == "--skip-npm-install");
            let report =
                install_reference(&reference_dir, !skip_npm_install, install_browser)?;
            print_json(&serde_json::to_value(report)?)
        }
        "bootstrap" => {
            let reference_dir = resolve_reference_dir(root, &args[1..]);
            let chromium_fallback_executable = find_playwright_cache_chromium_executable();
            let chromium_fallback_string = chromium_fallback_executable
                .as_ref()
                .map(|value| value.display().to_string());
            print_json(&json!({
                "ok": true,
                "reference_dir": reference_dir,
                "chromium_fallback_executable": chromium_fallback_string,
                "snippet": bootstrap_snippet(chromium_fallback_string.as_deref()),
            }))
        }
        _ => anyhow::bail!(
            "usage:\n  ctox browser doctor [--dir <path>]\n  ctox browser install-reference [--dir <path>] [--skip-npm-install] [--install-browser]\n  ctox browser bootstrap [--dir <path>]"
        ),
    }
}

pub fn codex_config_overrides(root: &Path) -> Vec<String> {
    let reference_dir = root.join(DEFAULT_REFERENCE_RELATIVE_DIR);
    codex_config_overrides_for_reference_dir(&reference_dir)
}

fn build_doctor_report(reference_dir: &Path) -> Result<BrowserDoctorReport> {
    let package_json_exists = reference_dir.join("package.json").exists();
    let node_modules_exists = reference_dir.join("node_modules").is_dir();
    let playwright_dependency_declared = read_playwright_dependency_declared(reference_dir)?;
    let chromium_fallback_executable =
        find_playwright_cache_chromium_executable().map(|value| value.display().to_string());
    let node = detect_tool("node", &["--version"]);
    let npm = detect_tool("npm", &["--version"]);
    let npx = detect_tool("npx", &["--version"]);
    let ok = node.available && npm.available && npx.available;
    Ok(BrowserDoctorReport {
        ok,
        reference_dir: reference_dir.to_path_buf(),
        package_json_exists,
        node_modules_exists,
        playwright_dependency_declared,
        chromium_fallback_executable,
        toolchain: json!({
            "node": node,
            "npm": npm,
            "npx": npx,
        }),
        codex_config_overrides: codex_config_overrides_for_reference_dir(reference_dir),
    })
}

fn install_reference(
    reference_dir: &Path,
    run_npm_install: bool,
    install_browser: bool,
) -> Result<BrowserInstallReport> {
    fs::create_dir_all(reference_dir).with_context(|| {
        format!(
            "failed to create interactive browser reference dir {}",
            reference_dir.display()
        )
    })?;
    let package_json_created = ensure_reference_package_json(reference_dir)?;
    if run_npm_install {
        run_command(
            reference_dir,
            "npm",
            &["install", "playwright"],
            "failed to install playwright reference",
        )?;
    }
    if install_browser {
        run_command(
            reference_dir,
            "npx",
            &["playwright", "install", "chromium"],
            "failed to install Playwright chromium browser",
        )?;
    }
    Ok(BrowserInstallReport {
        ok: true,
        reference_dir: reference_dir.to_path_buf(),
        package_json_created,
        npm_install_ran: run_npm_install,
        browser_install_ran: install_browser,
        codex_config_overrides: codex_config_overrides_for_reference_dir(reference_dir),
    })
}

fn ensure_reference_package_json(reference_dir: &Path) -> Result<bool> {
    let package_json_path = reference_dir.join("package.json");
    if package_json_path.exists() {
        return Ok(false);
    }
    let package_json = json!({
        "name": "ctox-interactive-browser-reference",
        "private": true,
        "type": "module",
        "description": "CTOX runtime reference workspace for js_repl-backed Playwright browser sessions.",
        "scripts": {
            "doctor": "node -e \"import('playwright').then(() => console.log('playwright import ok')).catch((error) => { console.error(error); process.exit(1); })\"",
            "install:chromium": "playwright install chromium"
        },
        "dependencies": {
            "playwright": "^1.53.0"
        }
    });
    fs::write(
        &package_json_path,
        serde_json::to_vec_pretty(&package_json)?,
    )
    .with_context(|| format!("failed to write {}", package_json_path.display()))?;
    Ok(true)
}

fn read_playwright_dependency_declared(reference_dir: &Path) -> Result<bool> {
    let package_json_path = reference_dir.join("package.json");
    if !package_json_path.exists() {
        return Ok(false);
    }
    let raw = fs::read(&package_json_path)
        .with_context(|| format!("failed to read {}", package_json_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&raw).context("failed to parse browser reference package.json")?;
    Ok(value
        .get("dependencies")
        .and_then(|value| value.get("playwright"))
        .and_then(serde_json::Value::as_str)
        .is_some())
}

fn run_command(cwd: &Path, program: &str, args: &[&str], error_message: &str) -> Result<()> {
    let output = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .output()
        .with_context(|| format!("{error_message}: failed to launch `{program}`"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit status {}", output.status)
    };
    anyhow::bail!("{error_message}: {detail}");
}

fn detect_tool(program: &str, version_args: &[&str]) -> ToolStatus {
    let path = find_command_on_path(program);
    let Some(path) = path else {
        return ToolStatus {
            available: false,
            path: None,
            version: None,
        };
    };
    let version = Command::new(&path)
        .args(version_args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if !stdout.is_empty() {
                stdout
            } else {
                stderr
            }
        })
        .filter(|value| !value.is_empty());
    ToolStatus {
        available: true,
        path: Some(path.display().to_string()),
        version,
    }
}

fn find_command_on_path(program: &str) -> Option<PathBuf> {
    if program.contains('/') {
        let path = PathBuf::from(program);
        return path.is_file().then_some(path);
    }
    let path_env = std::env::var_os("PATH")?;
    std::env::split_paths(&path_env)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

fn resolve_reference_dir(root: &Path, args: &[String]) -> PathBuf {
    find_flag_value(args, "--dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(DEFAULT_REFERENCE_RELATIVE_DIR))
}

fn codex_config_overrides_for_reference_dir(reference_dir: &Path) -> Vec<String> {
    let mut overrides = vec!["features.js_repl=true".to_string()];
    if let Some(node_path) = find_command_on_path("node") {
        overrides.push(format!(
            "js_repl_node_path=\"{}\"",
            escape_toml_string(&node_path.display().to_string())
        ));
    }
    let node_modules_dir = reference_dir.join("node_modules");
    if node_modules_dir.is_dir() {
        overrides.push(format!(
            "js_repl_node_module_dirs=[\"{}\"]",
            escape_toml_string(&node_modules_dir.display().to_string())
        ));
    }
    overrides
}

fn escape_toml_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).map(String::as_str)
}

fn bootstrap_snippet(chromium_fallback_executable: Option<&str>) -> String {
    let launch_options = if let Some(path) = chromium_fallback_executable {
        format!(
            "{{ headless: false, executablePath: \"{}\" }}",
            path.replace('\\', "\\\\").replace('"', "\\\"")
        )
    } else {
        "{ headless: false }".to_string()
    };
    format!(
        "var chromium;\nvar browser;\nvar context;\nvar page;\n({{ chromium }} = await import(\"playwright\"));\nbrowser ??= await chromium.launch({launch_options});\ncontext ??= await browser.newContext({{ viewport: {{ width: 1600, height: 900 }} }});\npage ??= await context.newPage();\nawait page.goto(\"http://127.0.0.1:3000\", {{ waitUntil: \"domcontentloaded\" }});\nconsole.log(\"Loaded:\", await page.title());"
    )
}

fn find_playwright_cache_chromium_executable() -> Option<PathBuf> {
    let cache_root = std::env::var_os("HOME")
        .map(PathBuf::from)?
        .join("Library/Caches/ms-playwright");
    let entries = fs::read_dir(&cache_root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("chromium-") || name.contains("headless") {
            continue;
        }
        for relative in [
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "chrome-mac/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "chrome-linux/chrome",
            "chrome-win/chrome.exe",
        ] {
            let candidate = path.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn print_json(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::codex_config_overrides_for_reference_dir;
    use super::ensure_reference_package_json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ctox-browser-{label}-{unique}"))
    }

    #[test]
    fn ensure_reference_package_json_writes_playwright_dependency() {
        let dir = temp_path("package-json");
        fs::create_dir_all(&dir).unwrap();
        let created = ensure_reference_package_json(&dir).unwrap();
        let raw = fs::read(dir.join("package.json")).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert!(created);
        assert_eq!(
            value["dependencies"]["playwright"].as_str(),
            Some("^1.53.0")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn codex_config_overrides_include_node_modules_when_reference_exists() {
        let dir = temp_path("overrides");
        fs::create_dir_all(dir.join("node_modules")).unwrap();
        let overrides = codex_config_overrides_for_reference_dir(&dir);
        assert!(overrides
            .iter()
            .any(|value| value == "features.js_repl=true"));
        assert!(overrides
            .iter()
            .any(|value| value.starts_with("js_repl_node_module_dirs=[")));
        let _ = fs::remove_dir_all(&dir);
    }
}
