// ref: internal/registry/model_registry_hook_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{
    sync::{mpsc, Arc},
    time::Duration,
};

use super::*;

fn registry() -> ModelRegistry {
    ModelRegistry::new(Arc::new(embedded_models_catalog().unwrap()))
}

enum Call {
    Registered(String, String, Vec<RegistryModelInfo>),
    Unregistered(String, String),
}

struct CapturingHook(mpsc::Sender<Call>);

impl ModelRegistryHook for CapturingHook {
    fn on_models_registered(
        &self,
        context: HookContext,
        provider: &str,
        client_id: &str,
        models: Vec<RegistryModelInfo>,
    ) {
        assert!(!context.is_expired());
        self.0
            .send(Call::Registered(
                provider.to_owned(),
                client_id.to_owned(),
                models,
            ))
            .unwrap();
    }

    fn on_models_unregistered(&self, _: HookContext, provider: &str, client_id: &str) {
        self.0
            .send(Call::Unregistered(
                provider.to_owned(),
                client_id.to_owned(),
            ))
            .unwrap();
    }
}

fn model(id: &str) -> RegistryModelInfo {
    RegistryModelInfo {
        id: id.to_owned(),
        ..RegistryModelInfo::default()
    }
}

#[test]
fn registered_hook_is_async_normalized_and_deep_cloned() {
    let registry = registry();
    let (tx, rx) = mpsc::channel();
    registry.set_hook(Some(Arc::new(CapturingHook(tx))));
    let mut models = vec![model("m1"), model("m2"), model("m2")];
    registry.register_client("client-1", "OpenAI", &models);
    models[0].id = "mutated".to_owned();
    let Call::Registered(provider, client, captured) =
        rx.recv_timeout(Duration::from_secs(2)).unwrap()
    else {
        panic!("wrong hook call")
    };
    assert_eq!(provider, "openai");
    assert_eq!(client, "client-1");
    assert_eq!(
        captured.iter().map(|model| &model.id).collect::<Vec<_>>(),
        ["m1", "m2"]
    );
}

#[test]
fn unregistered_hook_receives_original_provider() {
    let registry = registry();
    let (tx, rx) = mpsc::channel();
    registry.set_hook(Some(Arc::new(CapturingHook(tx))));
    registry.register_client("client-1", "OpenAI", &[model("m1")]);
    let _ = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    registry.unregister_client("client-1");
    let Call::Unregistered(provider, client) = rx.recv_timeout(Duration::from_secs(2)).unwrap()
    else {
        panic!("wrong hook call")
    };
    assert_eq!((provider.as_str(), client.as_str()), ("openai", "client-1"));
}

struct BlockingHook {
    started: mpsc::Sender<()>,
    unblock: std::sync::Mutex<mpsc::Receiver<()>>,
}

impl ModelRegistryHook for BlockingHook {
    fn on_models_registered(&self, _: HookContext, _: &str, _: &str, _: Vec<RegistryModelInfo>) {
        self.started.send(()).unwrap();
        self.unblock.lock().unwrap().recv().unwrap();
    }
    fn on_models_unregistered(&self, _: HookContext, _: &str, _: &str) {}
}

#[test]
fn hook_does_not_block_registration() {
    let registry = Arc::new(registry());
    let (started_tx, started_rx) = mpsc::channel();
    let (unblock_tx, unblock_rx) = mpsc::channel();
    registry.set_hook(Some(Arc::new(BlockingHook {
        started: started_tx,
        unblock: std::sync::Mutex::new(unblock_rx),
    })));
    let target = Arc::clone(&registry);
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        target.register_client("client-1", "OpenAI", &[model("m1")]);
        done_tx.send(()).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    done_rx.recv_timeout(Duration::from_millis(200)).unwrap();
    assert!(registry.client_supports_model("client-1", "m1"));
    unblock_tx.send(()).unwrap();
}

struct PanicHook;

impl ModelRegistryHook for PanicHook {
    fn on_models_registered(&self, _: HookContext, _: &str, _: &str, _: Vec<RegistryModelInfo>) {
        panic!("boom")
    }
    fn on_models_unregistered(&self, _: HookContext, _: &str, _: &str) {
        panic!("boom")
    }
}

#[test]
fn hook_panic_does_not_affect_registry() {
    let registry = registry();
    registry.set_hook(Some(Arc::new(PanicHook)));
    registry.register_client("client-1", "OpenAI", &[model("m1")]);
    assert!(registry.client_supports_model("client-1", "m1"));
    registry.unregister_client("client-1");
    assert!(!registry.client_supports_model("client-1", "m1"));
}
