// ref: sdk/cliproxy/executionregistry/observation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::registry::{Registry, Scope};
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Immutable in-flight execution snapshot entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Observation {
    pub request_id: String,
    pub credential_id: String,
    pub model: String,
    pub request_kind: String,
    pub started_at: DateTime<Utc>,
    pub accounted: bool,
}

/// Immutable in-flight execution snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Freeze {
    pub revision: i64,
    pub barrier_revision: i64,
    pub executions: Vec<Observation>,
}

impl Registry {
    /// Records the latest Home observation barrier.
    pub fn observe_barrier(&self, revision: i64) {
        if revision <= 0 {
            return;
        }
        let mut data = super::registry::lock_unpoisoned(&self.shared().data);
        if revision > data.observed_barrier {
            data.observed_barrier = revision;
            data.pending_barrier_sequence = data.next;
        }
    }

    /// Copies all active executions into an immutable snapshot.
    pub fn freeze_in_flight(&self, _now: DateTime<Utc>) -> Freeze {
        let mut data = super::registry::lock_unpoisoned(&self.shared().data);
        if data.observed_barrier > data.published_barrier {
            let blocked = data
                .pending
                .iter()
                .any(|sequence| *sequence <= data.pending_barrier_sequence);
            if !blocked {
                data.published_barrier = data.observed_barrier;
            }
        }

        data.snapshot_revision += 1;
        Freeze {
            revision: data.snapshot_revision,
            barrier_revision: data.published_barrier,
            executions: data.scopes.values().map(scope_observation).collect(),
        }
    }
}

fn scope_observation(scope: &Arc<super::registry::ScopeInner>) -> Observation {
    let scope = Scope::from_inner(Arc::clone(scope));
    let spec = scope.spec();
    Observation {
        request_id: spec.request_id.clone(),
        credential_id: spec.credential_id.clone(),
        model: spec.model.clone(),
        request_kind: spec.kind.clone(),
        started_at: spec.started_at,
        accounted: spec.accounted,
    }
}
