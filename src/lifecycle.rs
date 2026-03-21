use crate::browser_agent_bridge::ensure_browser_agent_bridge;
use crate::contracts::Paths;
use crate::contracts::default_bios;
use crate::contracts::default_homepage_policy;
use crate::contracts::default_installation_bootstrap_state;
use crate::contracts::default_organigram;
use crate::contracts::default_root_auth;
use crate::contracts::default_self_preservation_state;
use crate::contracts::ensure_contract_files;
use crate::contracts::ensure_tls_files;
use crate::runtime_db::init_runtime_db;
use crate::runtime_db::seed_bootstrap_tasks;
use crate::storage::save_json;
use anyhow::Context;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub fn initialize_runtime(paths: &Paths) -> anyhow::Result<()> {
    ensure_contract_files(paths)?;
    ensure_tls_files(paths)?;
    init_runtime_db(paths)?;
    ensure_browser_agent_bridge(paths)?;
    seed_bootstrap_tasks(paths)?;
    Ok(())
}

pub fn factory_reset_installation(paths: &Paths) -> anyhow::Result<()> {
    reset_mutable_contract_state(paths)?;
    clear_runtime_artifacts(paths)?;
    initialize_runtime(paths)
}

fn reset_mutable_contract_state(paths: &Paths) -> anyhow::Result<()> {
    paths.ensure_dirs()?;
    save_json(&paths.bios_path, &default_bios())?;
    save_json(&paths.org_path, &default_organigram())?;
    save_json(&paths.root_auth_path, &default_root_auth())?;
    save_json(&paths.homepage_policy_path, &default_homepage_policy())?;
    save_json(
        &paths.installation_bootstrap_path,
        &default_installation_bootstrap_state(),
    )?;
    save_json(
        &paths.self_preservation_state_path,
        &default_self_preservation_state(),
    )?;
    Ok(())
}

fn clear_runtime_artifacts(paths: &Paths) -> anyhow::Result<()> {
    remove_dir_if_exists(&paths.runtime_dir.join("state"))?;
    remove_dir_if_exists(&paths.uploads_dir)?;
    remove_dir_if_exists(&paths.browser_artifacts_dir)?;
    remove_dir_if_exists(&paths.recovery_dir)?;
    remove_dir_if_exists(&paths.runtime_dir.join("browser-agent-bridge"))?;
    remove_file_if_exists(&paths.boot_log_path)?;
    remove_file_if_exists(&paths.runtime_db_path)?;
    remove_file_if_exists(&sqlite_sidecar_path(&paths.runtime_db_path, "-journal"))?;
    remove_file_if_exists(&sqlite_sidecar_path(&paths.runtime_db_path, "-shm"))?;
    remove_file_if_exists(&sqlite_sidecar_path(&paths.runtime_db_path, "-wal"))?;
    remove_file_if_exists(&paths.runtime_lock_path)?;
    remove_file_if_exists(&paths.attach_socket_path)?;
    Ok(())
}

fn remove_dir_if_exists(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove directory {}", path.display()))?;
    }
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("failed to remove file {}", path.display()))?;
    }
    Ok(())
}

fn sqlite_sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut raw = OsString::from(path.as_os_str());
    raw.push(suffix);
    PathBuf::from(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::Paths;
    use crate::contracts::load_bios;
    use crate::contracts::load_homepage_policy;
    use crate::contracts::load_installation_bootstrap_state;
    use crate::contracts::load_root_auth;
    use rusqlite::Connection;
    use rusqlite::params;
    use std::ffi::OsString;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_test_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cto_agent_lifecycle_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    struct EnvGuard(Option<OsString>);

    impl EnvGuard {
        fn set_cto_root(root: &Path) -> Self {
            let previous = std::env::var_os("CTO_AGENT_ROOT");
            unsafe {
                std::env::set_var("CTO_AGENT_ROOT", root);
            }
            Self(previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                unsafe {
                    std::env::set_var("CTO_AGENT_ROOT", previous);
                }
            } else {
                unsafe {
                    std::env::remove_var("CTO_AGENT_ROOT");
                }
            }
        }
    }

    #[test]
    fn factory_reset_restores_fresh_installation_state() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("factory_reset");
        fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;

        initialize_runtime(&paths)?;

        let mut bios = load_bios(&paths);
        bios.mission = "Mutated mission".to_string();
        bios.owner.name = "Michael Welsch".to_string();
        save_json(&paths.bios_path, &bios)?;

        let mut homepage = load_homepage_policy(&paths);
        homepage.homepage_ready = true;
        homepage.current_title = "Mutated homepage".to_string();
        save_json(&paths.homepage_policy_path, &homepage)?;

        let mut install = load_installation_bootstrap_state(&paths);
        install.status = "captured".to_string();
        install.owner_name = "Michael Welsch".to_string();
        save_json(&paths.installation_bootstrap_path, &install)?;

        crate::contracts::update_root_password(&paths, "factory-reset-test")?;
        let root_auth = crate::contracts::load_root_auth(&paths);
        assert!(root_auth.configured);

        fs::write(&paths.boot_log_path, "{\"boot\":true}\n")?;
        fs::write(paths.uploads_dir.join("upload.txt"), "upload")?;
        fs::write(paths.browser_artifacts_dir.join("artifact.txt"), "artifact")?;
        fs::write(
            paths
                .runtime_dir
                .join("browser-agent-bridge/jobs/job-1.json"),
            "{}",
        )?;

        let conn = Connection::open(&paths.runtime_db_path)?;
        let now = crate::contracts::now_iso();
        conn.execute(
            "INSERT INTO tasks(
                created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id,
                source_channel, speaker, task_kind, title, detail, trust_level, priority_score, status
             ) VALUES(?1, ?2, NULL, NULL, NULL, 'attach_terminal', 'Michael Welsch',
                'owner_interrupt', 'Temporary task', 'Reset me', 'owner_trust', 1000, 'queued')",
            params![now, now],
        )?;
        drop(conn);

        factory_reset_installation(&paths)?;

        let bios = load_bios(&paths);
        assert!(bios.mission.is_empty());
        assert!(bios.owner.name.is_empty());
        assert!(!bios.frozen);

        let homepage = load_homepage_policy(&paths);
        assert!(!homepage.homepage_ready);
        assert_eq!(homepage.current_title, "CTO-Agent Terminal Bridge");

        let install = load_installation_bootstrap_state(&paths);
        assert_eq!(install.status, "unconfigured");
        assert!(install.owner_name.is_empty());

        let root_auth = load_root_auth(&paths);
        assert!(!root_auth.configured);

        assert!(!paths.boot_log_path.exists());
        assert!(!paths.uploads_dir.join("upload.txt").exists());
        assert!(!paths.browser_artifacts_dir.join("artifact.txt").exists());
        assert!(
            !paths
                .runtime_dir
                .join("browser-agent-bridge/jobs/job-1.json")
                .exists()
        );
        assert!(paths.runtime_db_path.exists());
        assert!(paths.agent_state_path.exists());

        let conn = Connection::open(&paths.runtime_db_path)?;
        let leftover_tasks: i64 = conn.query_row(
            "SELECT COUNT(*) FROM tasks WHERE title = 'Temporary task'",
            [],
            |row| row.get(0),
        )?;
        let seeded: i64 =
            conn.query_row("SELECT COUNT(*) FROM bootstrap_task_seeds", [], |row| {
                row.get(0)
            })?;
        assert_eq!(leftover_tasks, 0);
        assert!(seeded > 0);

        fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
