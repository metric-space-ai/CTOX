// ref: internal/home/global.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! CTOX replacement for upstream's process-global `atomic.Pointer[Client]`.
//! Owners inject this handle explicitly, preventing unrelated harnesses from
//! sharing a mutable Home control-plane client.

use super::client::Client;
use std::sync::{Arc, RwLock};

#[derive(Default)]
pub struct HomeRuntime {
    current: RwLock<Option<Arc<Client>>>,
}
impl HomeRuntime {
    pub fn set_current(&self, client: Arc<Client>) {
        *self
            .current
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(client);
    }
    pub fn current(&self) -> Option<Arc<Client>> {
        self.current
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    pub fn clear_current(&self) {
        *self
            .current
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
    pub fn clear_current_if(&self, client: &Arc<Client>) -> bool {
        let mut current = self
            .current
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if current
            .as_ref()
            .is_some_and(|value| Arc::ptr_eq(value, client))
        {
            *current = None;
            true
        } else {
            false
        }
    }
}
