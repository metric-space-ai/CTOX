use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

pub struct TestRoot {
    root: PathBuf,
}

#[allow(dead_code)]
impl TestRoot {
    pub fn new(label: &str) -> Self {
        let root = unique_test_root(label);
        fs::create_dir_all(root.join("runtime")).expect("failed to create runtime dir");
        Self { root }
    }

    pub fn run(&self, args: &[&str]) -> CmdOutput {
        let output = Command::new(env!("CARGO_BIN_EXE_ctox"))
            .args(args)
            .env("CTOX_ROOT", &self.root)
            .output()
            .expect("failed to execute ctox binary");
        CmdOutput { output }
    }

    pub fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[allow(dead_code)]
pub struct CmdOutput {
    output: std::process::Output,
}

#[allow(dead_code)]
impl CmdOutput {
    pub fn success(&self) -> &Self {
        assert!(
            self.output.status.success(),
            "command failed\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
            self.output.status.code(),
            self.stdout(),
            self.stderr()
        );
        self
    }

    pub fn stdout(&self) -> String {
        String::from_utf8_lossy(&self.output.stdout)
            .trim()
            .to_string()
    }

    pub fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr)
            .trim()
            .to_string()
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.output.stdout).expect("stdout was not valid json")
    }
}

#[allow(dead_code)]
impl TestRoot {
    pub fn db_path(&self) -> PathBuf {
        self.path("runtime/cto_agent.db")
    }
}

fn unique_test_root(label: &str) -> PathBuf {
    let mut slug = label
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .to_ascii_lowercase();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-');
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ctox-test-{}-{}-{}", slug, stamp, seq))
}
