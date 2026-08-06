// ref: internal/api/handlers/management/auth_files_plugin_oauth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use super::auth_files_patch_fields_test::{record, Store};
use super::{
    ManagementCredentialService, ManagementOAuthClock, ManagementOAuthSessions,
    ManagementProviderOAuth, ManagementProviderOAuthAuthority,
    ManagementProviderOAuthAuthorityError, ManagementProviderOAuthError,
    ManagementProviderOAuthPoll,
};

#[derive(Debug)]
struct Clock;

impl ManagementOAuthClock for Clock {
    fn now_ms(&self) -> i64 {
        1_000
    }
}

#[derive(Debug, Default)]
struct Authority {
    polls: Mutex<VecDeque<ManagementProviderOAuthPoll>>,
}

impl ManagementProviderOAuthAuthority for Authority {
    fn begin(
        &self,
        provider: &str,
        state: &str,
        callback_path: &str,
    ) -> Result<String, ManagementProviderOAuthAuthorityError> {
        Ok(format!(
            "https://auth.invalid/{provider}?state={state}&callback={callback_path}"
        ))
    }

    fn poll(
        &self,
        _: &str,
        _: &str,
    ) -> Result<ManagementProviderOAuthPoll, ManagementProviderOAuthAuthorityError> {
        self.polls
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(ManagementProviderOAuthAuthorityError)
    }

    fn cancel(&self, _: &str, _: &str) -> Result<(), ManagementProviderOAuthAuthorityError> {
        Ok(())
    }
}

fn setup(polls: Vec<ManagementProviderOAuthPoll>) -> (Arc<Store>, ManagementProviderOAuth) {
    let store = Arc::new(Store::default());
    let credentials = Arc::new(ManagementCredentialService::new(store.clone()));
    let sessions = Arc::new(ManagementOAuthSessions::new(Arc::new(Clock)));
    let authority = Arc::new(Authority {
        polls: Mutex::new(polls.into()),
    });
    (
        store,
        ManagementProviderOAuth::new(sessions, credentials, authority),
    )
}

#[test]
fn plugin_poll_expands_multiple_credentials_atomically() {
    let mut one = record("one");
    one.provider = "plugin-x".to_owned();
    let mut two = record("two");
    two.provider = "plugin-x".to_owned();
    let (store, oauth) = setup(vec![ManagementProviderOAuthPoll {
        credentials: vec![one, two],
        ..Default::default()
    }]);
    oauth
        .begin_plugin("plugin-x", "state-1", BTreeMap::new())
        .unwrap();
    oauth.poll("plugin-x", "state-1").unwrap();
    let mut ids = store
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, ["one", "two"]);
}

#[test]
fn invalid_plugin_expansion_rolls_back_the_entire_poll() {
    let mut valid = record("valid");
    valid.provider = "plugin-x".to_owned();
    let mut invalid = record("invalid");
    invalid.id.clear();
    invalid.auth_index.clear();
    invalid.provider = "plugin-x".to_owned();
    let (store, oauth) = setup(vec![ManagementProviderOAuthPoll {
        credentials: vec![valid, invalid],
        ..Default::default()
    }]);
    store.0.lock().unwrap().push(record("existing"));
    let before = store.0.lock().unwrap().clone();
    oauth
        .begin_plugin("plugin-x", "state-2", BTreeMap::new())
        .unwrap();
    assert!(oauth.poll("plugin-x", "state-2").is_err());
    assert_eq!(*store.0.lock().unwrap(), before);
}

#[test]
fn plugin_source_controls_expanded_children_and_direct_mutation_conflicts() {
    let mut one = record("one");
    one.provider = "plugin-x".to_owned();
    let mut two = record("two");
    two.provider = "plugin-x".to_owned();
    let (store, oauth) = setup(vec![ManagementProviderOAuthPoll {
        credentials: vec![one, two],
        ..Default::default()
    }]);
    oauth
        .begin_plugin("plugin-x", "state-3", BTreeMap::new())
        .unwrap();
    oauth.poll("plugin-x", "state-3").unwrap();
    assert_eq!(
        oauth.guard_not_virtual_child("one"),
        Err(ManagementProviderOAuthError::VirtualChildConflict)
    );
    assert_eq!(
        oauth.set_plugin_source_disabled("plugin-x", true).unwrap(),
        2
    );
    assert!(store.0.lock().unwrap().iter().all(|record| record.disabled));
    assert_eq!(oauth.delete_plugin_source("plugin-x").unwrap(), 2);
    assert!(store.0.lock().unwrap().is_empty());
}

#[test]
fn provider_flow_uses_secret_free_store_projection() {
    let mut projected = record("one");
    projected.provider = "plugin-x".to_owned();
    let (_, oauth) = setup(vec![ManagementProviderOAuthPoll {
        credentials: vec![projected],
        ..Default::default()
    }]);
    let start = oauth
        .begin_plugin("plugin-x", "state-4", BTreeMap::new())
        .unwrap();
    assert!(format!("{start:?}").contains("[REDACTED]"));
    oauth.poll("plugin-x", "state-4").unwrap();
}
