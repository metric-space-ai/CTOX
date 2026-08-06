// ref: sdk/cliproxy/usage/manager_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::{generate_enabled, generate_flag, Manager, Plugin, Record, UsageContext};

#[test]
fn generate_defaults_and_explicit_values_match_upstream() {
    assert!(generate_enabled(None));
    assert!(!generate_enabled(generate_flag(false)));
    assert!(generate_enabled(generate_flag(true)));
    assert!(Record::default().generate.is_none());
    assert!(generate_enabled(Record::default().generate));
}

#[test]
fn typed_context_preserves_value_semantics() {
    let default = UsageContext::default();
    assert_eq!(default.requested_model_alias(), "");
    assert_eq!(default.reasoning_effort(), "");
    assert_eq!(default.service_tier(), "default");
    assert!(default.generate());

    let context = default
        .with_requested_model_alias("  client-model  ")
        .with_reasoning_effort(" high ")
        .with_service_tier("  ")
        .with_generate(false);
    assert_eq!(context.requested_model_alias(), "client-model");
    assert_eq!(context.reasoning_effort(), "high");
    assert_eq!(context.service_tier(), "default");
    assert!(!context.generate());
}

struct Collector {
    values: Arc<Mutex<Vec<String>>>,
    prefix: &'static str,
}

impl Plugin for Collector {
    fn handle_usage(&self, context: &UsageContext, record: &Record) {
        self.values.lock().unwrap().push(format!(
            "{}:{}:{}",
            self.prefix,
            context.requested_model_alias(),
            record.model
        ));
    }
}

#[test]
fn lazy_start_fifo_and_stop_drain() {
    let manager = Manager::new(2);
    let values = Arc::new(Mutex::new(Vec::new()));
    manager.register(Arc::new(Collector {
        values: Arc::clone(&values),
        prefix: "p",
    }));
    for model in ["one", "two", "three"] {
        assert!(manager.publish(
            UsageContext::default().with_requested_model_alias("alias"),
            Record {
                model: model.to_owned(),
                ..Record::default()
            },
        ));
    }
    manager.stop();
    assert_eq!(
        *values.lock().unwrap(),
        ["p:alias:one", "p:alias:two", "p:alias:three"]
    );
    assert!(!manager.publish(UsageContext::default(), Record::default()));
}

#[test]
fn named_registration_replaces_in_place() {
    let manager = Manager::new(1);
    let values = Arc::new(Mutex::new(Vec::new()));
    manager.register_named(
        "sink",
        Arc::new(Collector {
            values: Arc::clone(&values),
            prefix: "old",
        }),
    );
    manager.register_named(
        "sink",
        Arc::new(Collector {
            values: Arc::clone(&values),
            prefix: "new",
        }),
    );
    manager.publish(
        UsageContext::default(),
        Record {
            model: "m".to_owned(),
            ..Record::default()
        },
    );
    manager.stop();
    assert_eq!(*values.lock().unwrap(), ["new::m"]);
}

struct Panics;

impl Plugin for Panics {
    fn handle_usage(&self, _: &UsageContext, _: &Record) {
        panic!("fixture secret must remain isolated");
    }
}

#[test]
fn plugin_panic_isolated_from_later_plugins() {
    let manager = Manager::new(1);
    let values = Arc::new(Mutex::new(Vec::new()));
    manager.register(Arc::new(Panics));
    manager.register(Arc::new(Collector {
        values: Arc::clone(&values),
        prefix: "ok",
    }));
    manager.publish(
        UsageContext::default(),
        Record {
            model: "m".to_owned(),
            ..Record::default()
        },
    );
    manager.stop();
    assert_eq!(*values.lock().unwrap(), ["ok::m"]);
    assert_eq!(manager.plugin_panic_count(), 1);
}

#[test]
fn empty_named_registration_is_ignored() {
    let manager = Manager::new(0);
    let values = Arc::new(Mutex::new(Vec::new()));
    manager.register_named(
        "  ",
        Arc::new(Collector {
            values: Arc::clone(&values),
            prefix: "ignored",
        }),
    );
    manager.publish(UsageContext::default(), Record::default());
    manager.stop();
    assert!(values.lock().unwrap().is_empty());
}
