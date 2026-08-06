// ref: sdk/cliproxy/service_plugin_scheduler_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::providers::LoadContext;
use super::service_plugins::{
    run_model_registration_tasks, ModelRegistrationPhase, ModelRegistrationTask,
};
use super::service_test_support::{runtime_fixture, TestPluginRuntime};
use super::usage::{Plugin, Record, UsageContext};

#[test]
fn sync_plugin_runtime_binds_instance_owned_scheduler_and_ordered_hooks() {
    let fixture = runtime_fixture(None);
    let plugin = Arc::new(TestPluginRuntime::default());
    let erased: Arc<dyn super::service_plugins::ServicePluginRuntime> = plugin.clone();
    fixture.runtime.set_plugin_runtime(Some(erased.clone()));

    assert_eq!(fixture.runtime.sync_plugin_runtime_config(), Ok(true));
    let scheduler = fixture
        .runtime
        .plugin_scheduler()
        .expect("plugin scheduler");
    assert!(Arc::ptr_eq(&scheduler, &erased));
    assert_eq!(
        plugin.calls(),
        vec!["config", "frontend", "usage", "translator", "management"]
    );
}

#[test]
fn removing_plugin_runtime_clears_scheduler_binding() {
    let fixture = runtime_fixture(None);
    fixture
        .runtime
        .set_plugin_runtime(Some(Arc::new(TestPluginRuntime::default())));
    assert!(fixture.runtime.plugin_scheduler().is_some());
    fixture.runtime.set_plugin_runtime(None);
    assert_eq!(fixture.runtime.sync_plugin_runtime_config(), Ok(false));
    assert!(fixture.runtime.plugin_scheduler().is_none());
}

#[test]
fn model_registration_finishes_config_phase_before_other_phase() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let config_events = events.clone();
    let other_events = events.clone();
    run_model_registration_tasks(
        &LoadContext::default(),
        vec![
            ModelRegistrationTask::new(ModelRegistrationPhase::Other, "claude", move || {
                other_events.lock().expect("events").push("other");
            }),
            ModelRegistrationTask::new(
                ModelRegistrationPhase::ConfigApiKey,
                "openai-compatible-test",
                move || {
                    config_events.lock().expect("events").push("config");
                },
            ),
        ],
    );
    assert_eq!(*events.lock().expect("events"), vec!["config", "other"]);
}

struct UsageRecorder(Arc<Mutex<usize>>);

impl Plugin for UsageRecorder {
    fn handle_usage(&self, _context: &UsageContext, _record: &Record) {
        *self.0.lock().expect("usage count") += 1;
    }
}

#[test]
fn external_usage_plugin_is_registered_on_the_instance_owned_manager() {
    let fixture = runtime_fixture(None);
    let count = Arc::new(Mutex::new(0));
    fixture
        .runtime
        .register_usage_plugin(Arc::new(UsageRecorder(count.clone())));
    let usage = fixture.runtime.usage_manager();
    assert!(usage.publish(UsageContext::default(), Record::default()));
    usage.stop();
    assert_eq!(*count.lock().expect("usage count"), 1);
}
