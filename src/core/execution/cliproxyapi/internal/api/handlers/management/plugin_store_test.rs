// ref: internal/api/handlers/management/plugin_store_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::test_store_test::{Runtime, Store};
use super::{
    ManagementPluginInstallRequest, ManagementPluginInstallResult, ManagementPluginService,
    ManagementPluginStagedInstall, ManagementPluginStoreAuthority,
    ManagementPluginStoreAuthorityError, ManagementPluginStoreCatalog, ManagementPluginStoreEntry,
    ManagementPluginStoreError, ManagementPluginStoreService, ManagementPluginStoreSource,
};

#[derive(Debug)]
struct Authority {
    catalog: ManagementPluginStoreCatalog,
    staged: Mutex<Option<ManagementPluginStagedInstall>>,
    rolled_back: Mutex<Vec<String>>,
    committed: Mutex<Vec<String>>,
}

impl ManagementPluginStoreAuthority for Authority {
    fn catalog(&self) -> Result<ManagementPluginStoreCatalog, ManagementPluginStoreAuthorityError> {
        Ok(self.catalog.clone())
    }

    fn stage_install(
        &self,
        _: &ManagementPluginInstallRequest,
    ) -> Result<ManagementPluginStagedInstall, ManagementPluginStoreAuthorityError> {
        self.staged
            .lock()
            .unwrap()
            .take()
            .ok_or(ManagementPluginStoreAuthorityError)
    }

    fn commit_install(
        &self,
        staged: ManagementPluginStagedInstall,
    ) -> ManagementPluginInstallResult {
        self.committed.lock().unwrap().push(staged.operation_id);
        ManagementPluginInstallResult {
            source_id: staged.source_id,
            id: staged.id,
            version: staged.version,
            status: "installed".to_owned(),
            restart_required: false,
        }
    }

    fn rollback_install(&self, staged: ManagementPluginStagedInstall) {
        self.rolled_back.lock().unwrap().push(staged.operation_id);
    }
}

fn setup(
    staged: ManagementPluginStagedInstall,
) -> (Arc<Store>, Arc<Authority>, ManagementPluginStoreService) {
    let store = Arc::new(Store::default());
    let plugins = Arc::new(ManagementPluginService::new(
        store.clone(),
        Arc::new(Runtime::default()),
    ));
    let authority = Arc::new(Authority {
        catalog: ManagementPluginStoreCatalog::default(),
        staged: Mutex::new(Some(staged)),
        rolled_back: Mutex::new(Vec::new()),
        committed: Mutex::new(Vec::new()),
    });
    let service = ManagementPluginStoreService::new(plugins, authority.clone());
    (store, authority, service)
}

fn staged() -> ManagementPluginStagedInstall {
    ManagementPluginStagedInstall {
        operation_id: "opaque-operation".to_owned(),
        source_id: "official".to_owned(),
        id: "sample".to_owned(),
        version: "1.2.3".to_owned(),
        install_ref: "opaque-install-ref".to_owned(),
    }
}

fn request() -> ManagementPluginInstallRequest {
    ManagementPluginInstallRequest {
        source_id: "official".to_owned(),
        id: "sample".to_owned(),
        version: Some("v1.2.3".to_owned()),
        platform: "linux/amd64".to_owned(),
    }
}

#[test]
fn install_commits_staged_artifact_only_after_durable_enable() {
    let (store, authority, service) = setup(staged());
    let result = service.install(request()).unwrap();
    assert_eq!(result.status, "installed");
    assert!(store.0.lock().unwrap().configs["sample"].enabled);
    assert_eq!(*authority.committed.lock().unwrap(), ["opaque-operation"]);
    assert!(authority.rolled_back.lock().unwrap().is_empty());
}

#[test]
fn mismatched_receipt_rolls_back_without_publishing_config() {
    let mut receipt = staged();
    receipt.id = "other".to_owned();
    let (store, authority, service) = setup(receipt);
    assert_eq!(
        service.install(request()),
        Err(ManagementPluginStoreError::InvalidReceipt)
    );
    assert!(store.0.lock().unwrap().configs.is_empty());
    assert_eq!(*authority.rolled_back.lock().unwrap(), ["opaque-operation"]);
}

#[test]
fn catalog_rejects_credential_bearing_source_urls() {
    let bad = Arc::new(Authority {
        catalog: ManagementPluginStoreCatalog {
            sources: vec![ManagementPluginStoreSource {
                id: "official".to_owned(),
                name: "Official".to_owned(),
                url: "https://token@example.test/registry.json".to_owned(),
            }],
            plugins: Vec::new(),
            source_errors: Vec::new(),
        },
        staged: Mutex::new(None),
        rolled_back: Mutex::new(Vec::new()),
        committed: Mutex::new(Vec::new()),
    });
    let plugins = Arc::new(ManagementPluginService::new(
        Arc::new(Store::default()),
        Arc::new(Runtime::default()),
    ));
    let bad_service = ManagementPluginStoreService::new(plugins, bad);
    assert_eq!(
        bad_service.catalog(),
        Err(ManagementPluginStoreError::InvalidReceipt)
    );
}

#[test]
fn catalog_sorts_duplicate_provider_entries_by_id_then_source() {
    let authority = Arc::new(Authority {
        catalog: ManagementPluginStoreCatalog {
            sources: vec![
                ManagementPluginStoreSource {
                    id: "z-source".to_owned(),
                    name: "Z".to_owned(),
                    url: "https://z.example.test/registry.json".to_owned(),
                },
                ManagementPluginStoreSource {
                    id: "a-source".to_owned(),
                    name: "A".to_owned(),
                    url: "https://a.example.test/registry.json".to_owned(),
                },
            ],
            plugins: vec![
                ManagementPluginStoreEntry {
                    source_id: "z-source".to_owned(),
                    id: "sample".to_owned(),
                    version: "1.0.0".to_owned(),
                    ..Default::default()
                },
                ManagementPluginStoreEntry {
                    source_id: "a-source".to_owned(),
                    id: "sample".to_owned(),
                    version: "1.0.0".to_owned(),
                    ..Default::default()
                },
            ],
            source_errors: Vec::new(),
        },
        staged: Mutex::new(None),
        rolled_back: Mutex::new(Vec::new()),
        committed: Mutex::new(Vec::new()),
    });
    let plugins = Arc::new(ManagementPluginService::new(
        Arc::new(Store::default()),
        Arc::new(Runtime::default()),
    ));
    let catalog = ManagementPluginStoreService::new(plugins, authority)
        .catalog()
        .unwrap();
    assert_eq!(
        catalog
            .plugins
            .iter()
            .map(|plugin| plugin.source_id.as_str())
            .collect::<Vec<_>>(),
        ["a-source", "z-source"]
    );
}

#[test]
fn staged_debug_redacts_operation_and_install_reference() {
    let debug = format!("{:?}", staged());
    assert!(debug.contains("[REDACTED]"));
    assert!(!debug.contains("opaque-operation"));
    assert!(!debug.contains("opaque-install-ref"));
}
