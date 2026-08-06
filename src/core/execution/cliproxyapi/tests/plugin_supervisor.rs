// Origin: CTOX
// License: AGPL-3.0-only

#![cfg(any(unix, windows))]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use ctox_cliproxyapi::internal::pluginhost::process_transport::RequestMode;
use ctox_cliproxyapi::internal::pluginhost::supervisor::{
    PluginProcessConfig, PluginSupervisor, PluginSupervisorError, RestartPolicy,
};

fn runtime_root(label: &str) -> PathBuf {
    #[cfg(unix)]
    let base = PathBuf::from("/tmp");
    #[cfg(windows)]
    let base = std::env::temp_dir();
    let root = base.join(format!("cpa-supervisor-{label}-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    root
}

fn config(root: PathBuf, plugin_id: &str, max_restarts: u32) -> PluginProcessConfig {
    PluginProcessConfig {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_cliproxy-differential")),
        runtime_root: root,
        instance_id: "fixture".into(),
        plugin_id: plugin_id.into(),
        restart_policy: RestartPolicy {
            max_restarts,
            base_delay: Duration::from_millis(20),
            max_delay: Duration::from_millis(40),
        },
    }
}

#[tokio::test]
async fn spawned_child_has_empty_environment_and_shuts_down_gracefully() {
    let root = runtime_root("shutdown");
    let process_config = config(root.clone(), "fixture", 2);
    let debug = format!("{process_config:?}");
    assert!(!debug.contains("token"));

    let mut supervisor = PluginSupervisor::new(process_config).unwrap();
    supervisor.start().await.unwrap();
    assert!(supervisor.is_ready());
    supervisor.mark_stable();
    assert_eq!(supervisor.consecutive_failures(), 0);
    supervisor
        .begin_request(
            "request-before-shutdown".into(),
            RequestMode::Unary,
            None,
            0,
        )
        .unwrap();
    let report = supervisor.shutdown().await.unwrap();
    assert!(report.success);
    assert_eq!(report.code, Some(0));
    assert_eq!(report.aborted_request_ids, vec!["request-before-shutdown"]);
    assert!(!supervisor.is_ready());
    assert!(!root.join(".cpa/fixture/s").exists());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn registered_executor_capability_runs_a_real_bounded_unary_call() {
    let root = runtime_root("capability");
    let mut supervisor = PluginSupervisor::new(config(root.clone(), "fixture", 2)).unwrap();
    supervisor.start().await.unwrap();
    assert_eq!(
        supervisor.executor_identifier().await.unwrap_err(),
        PluginSupervisorError::NotRegistered
    );

    let registration = supervisor
        .register(b"mode: safe\ncredential: do-not-render".to_vec())
        .await
        .unwrap();
    assert!(supervisor.is_registered());
    assert_eq!(registration.metadata.name, "fixture-executor");
    assert!(registration.capabilities.executor);
    assert_eq!(
        supervisor.register(Vec::new()).await.unwrap_err(),
        PluginSupervisorError::AlreadyRegistered
    );
    assert_eq!(
        supervisor.executor_identifier().await.unwrap(),
        "fixture-executor"
    );

    let report = supervisor.shutdown().await.unwrap();
    assert!(report.success);
    assert!(report.aborted_request_ids.is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn unregistered_capability_is_rejected_before_child_dispatch() {
    let root = runtime_root("capability-gate");
    let mut supervisor =
        PluginSupervisor::new(config(root.clone(), "fixture-no-executor", 2)).unwrap();
    supervisor.start().await.unwrap();
    let registration = supervisor.register(Vec::new()).await.unwrap();
    assert!(!registration.capabilities.executor);
    assert_eq!(
        supervisor.executor_identifier().await.unwrap_err(),
        PluginSupervisorError::UnsupportedCapability
    );
    assert!(supervisor.shutdown().await.unwrap().success);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn crashes_abort_inflight_and_restart_with_capped_backoff() {
    let root = runtime_root("crash");
    let process_config = config(root.clone(), "fixture-crash", 2);
    let mut supervisor = PluginSupervisor::new(process_config).unwrap();
    supervisor.start().await.unwrap();
    supervisor
        .begin_request("request-before-crash".into(), RequestMode::Stream, None, 0)
        .unwrap();
    let first = supervisor.wait_for_exit().await.unwrap();
    assert!(!first.success);
    assert_eq!(first.code, Some(23));
    assert_eq!(first.aborted_request_ids, vec!["request-before-crash"]);
    assert!(!format!("{first:?}").contains("request-before-crash"));

    assert_eq!(
        supervisor.restart().await.unwrap(),
        Duration::from_millis(20)
    );
    let second = supervisor.wait_for_exit().await.unwrap();
    assert_eq!(second.code, Some(23));
    assert_eq!(
        supervisor.restart().await.unwrap(),
        Duration::from_millis(40)
    );
    let third = supervisor.wait_for_exit().await.unwrap();
    assert_eq!(third.code, Some(23));
    assert_eq!(
        supervisor.restart().await.unwrap_err(),
        PluginSupervisorError::RestartExhausted
    );
    assert!(!supervisor.is_ready());
    fs::remove_dir_all(root).unwrap();
}
