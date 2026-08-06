// ref: sdk/cliproxy/executionregistry/registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Instance-owned lifecycle registry for Home-dispatched executions.
//!
//! The upstream implementation is already scoped to one Home subscriber. This
//! port preserves that ownership and deliberately exposes no process-global
//! default. Go channels and `context.Context` cancellation are represented by
//! condition variables and explicit [`WaitBudget`] values.

use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// Lifecycle failures returned by the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistryError {
    NotAccepting,
    Closed,
    InvalidPendingDispatch,
    InvalidExecutionResource,
    ExecutionResourceAlreadyBound,
    DeadlineExceeded,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotAccepting => "execution registry is not accepting dispatches",
            Self::Closed => "execution registry is closed",
            Self::InvalidPendingDispatch => "invalid pending dispatch",
            Self::InvalidExecutionResource => "invalid execution resource",
            Self::ExecutionResourceAlreadyBound => "execution resource is already bound",
            Self::DeadlineExceeded => "execution registry deadline exceeded",
        })
    }
}

impl Error for RegistryError {}

/// Lifecycle state of a [`Registry`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum State {
    Accepting = 0,
    Draining = 1,
    Closed = 2,
}

impl State {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Accepting,
            1 => Self::Draining,
            _ => Self::Closed,
        }
    }
}

/// A finite or unbounded wait, replacing Go's nullable `context.Context`.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaitBudget(Option<Instant>);

impl WaitBudget {
    pub fn unbounded() -> Self {
        Self(None)
    }

    pub fn for_duration(duration: Duration) -> Self {
        Self(Some(Instant::now() + duration))
    }

    fn remaining(self) -> Result<Option<Duration>, RegistryError> {
        match self.0 {
            None => Ok(None),
            Some(deadline) => deadline
                .checked_duration_since(Instant::now())
                .map(Some)
                .ok_or(RegistryError::DeadlineExceeded),
        }
    }
}

/// Describes one Home-dispatched execution.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScopeSpec {
    pub request_id: String,
    pub credential_id: String,
    pub model: String,
    pub kind: String,
    pub started_at: DateTime<Utc>,
    pub accounted: bool,
}

/// Identifies the cumulative release sequence for one credential and model.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReleaseGroup {
    pub credential_id: String,
    pub model: String,
}

/// Instance-owned acknowledgement primitive used by [`ReleaseTicket`].
#[derive(Clone, Debug, Default)]
pub struct ReleaseAcknowledgement {
    inner: Arc<(Mutex<bool>, Condvar)>,
}

impl ReleaseAcknowledgement {
    pub fn acknowledge(&self) {
        let (lock, changed) = &*self.inner;
        *lock_unpoisoned(lock) = true;
        changed.notify_all();
    }

    fn wait(&self, budget: WaitBudget) -> Result<(), RegistryError> {
        let (lock, changed) = &*self.inner;
        let mut acknowledged = lock_unpoisoned(lock);
        while !*acknowledged {
            acknowledged = wait_with_budget(changed, acknowledged, budget)?;
        }
        Ok(())
    }
}

/// Completes after Home acknowledges a cumulative release sequence.
#[derive(Clone, Debug)]
pub struct ReleaseTicket {
    pub group: ReleaseGroup,
    pub sequence: i64,
    acknowledgement: ReleaseAcknowledgement,
}

impl ReleaseTicket {
    /// Returns `None` for invalid or non-acknowledgeable releases, matching
    /// upstream `NewReleaseTicket` returning `nil`.
    pub fn new(
        group: ReleaseGroup,
        sequence: i64,
        acknowledgement: Option<ReleaseAcknowledgement>,
    ) -> Option<Self> {
        if sequence <= 0 {
            return None;
        }
        acknowledgement.map(|acknowledgement| Self {
            group,
            sequence,
            acknowledgement,
        })
    }

    pub fn wait(&self, budget: WaitBudget) -> Result<(), RegistryError> {
        self.acknowledgement.wait(budget)
    }
}

/// Receives the latest cumulative sequence for a release group.
pub type ReleaseSink =
    Arc<dyn Fn(ReleaseGroup, i64) -> Option<ReleaseTicket> + Send + Sync + 'static>;

type CloseResource = Box<dyn FnOnce() -> Result<(), String> + Send + 'static>;

/// Owns all dispatches accepted during one subscriber lifetime.
#[derive(Clone)]
pub struct Registry {
    shared: Arc<RegistryShared>,
}

pub(super) struct RegistryShared {
    state: AtomicU32,
    pub(super) data: Mutex<RegistryData>,
    changed: Condvar,
    close: Mutex<CloseState>,
    close_changed: Condvar,
}

pub(super) struct RegistryData {
    pub(super) next: u64,
    pub(super) snapshot_revision: i64,
    pub(super) observed_barrier: i64,
    pub(super) pending_barrier_sequence: u64,
    pub(super) published_barrier: i64,
    pub(super) pending: HashSet<u64>,
    pub(super) scopes: BTreeMap<u64, Arc<ScopeInner>>,
    release_sequences: HashMap<ReleaseGroup, i64>,
    release_sink: Option<ReleaseSink>,
}

#[derive(Default)]
struct CloseState {
    started: bool,
    done: bool,
}

/// Reserves an execution slot until it is installed or ended.
pub struct PendingDispatch {
    id: u64,
    registry: Weak<RegistryShared>,
    resolved: Mutex<bool>,
}

impl PendingDispatch {
    pub fn end(&self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut resolved = lock_unpoisoned(&self.resolved);
        if *resolved {
            return;
        }
        *resolved = true;
        let mut data = lock_unpoisoned(&registry.data);
        data.pending.remove(&self.id);
        registry.changed.notify_all();
    }
}

/// Owns the resource for one installed execution.
#[derive(Clone)]
pub struct Scope {
    inner: Arc<ScopeInner>,
}

pub(super) struct ScopeInner {
    id: u64,
    registry: Weak<RegistryShared>,
    pub(super) spec: ScopeSpec,
    state: Mutex<ScopeState>,
    changed: Condvar,
}

struct ScopeState {
    close_fn: Option<CloseResource>,
    close_started: bool,
    close_done: bool,
    release_ticket: Option<ReleaseTicket>,
    active: bool,
    end_started: bool,
    end_done: bool,
}

impl Registry {
    /// Creates an accepting, subscriber-owned registry.
    pub fn new() -> Self {
        Self {
            shared: Arc::new(RegistryShared {
                state: AtomicU32::new(State::Accepting as u32),
                data: Mutex::new(RegistryData {
                    next: 0,
                    snapshot_revision: 0,
                    observed_barrier: 0,
                    pending_barrier_sequence: 0,
                    published_barrier: 0,
                    pending: HashSet::new(),
                    scopes: BTreeMap::new(),
                    release_sequences: HashMap::new(),
                    release_sink: None,
                }),
                changed: Condvar::new(),
                close: Mutex::new(CloseState::default()),
                close_changed: Condvar::new(),
            }),
        }
    }

    pub fn state(&self) -> State {
        State::from_raw(self.shared.state.load(Ordering::Acquire))
    }

    /// Reserves a dispatch token while this registry accepts traffic.
    pub fn begin_dispatch(&self) -> Result<Arc<PendingDispatch>, RegistryError> {
        if self.state() != State::Accepting {
            return Err(RegistryError::NotAccepting);
        }
        let mut data = lock_unpoisoned(&self.shared.data);
        if self.state() != State::Accepting {
            return Err(RegistryError::NotAccepting);
        }
        data.next += 1;
        let id = data.next;
        data.pending.insert(id);
        Ok(Arc::new(PendingDispatch {
            id,
            registry: Arc::downgrade(&self.shared),
            resolved: Mutex::new(false),
        }))
    }

    /// Waits until every unresolved Home response was installed or ended.
    pub fn wait_pending(&self, budget: WaitBudget) -> Result<(), RegistryError> {
        let mut data = lock_unpoisoned(&self.shared.data);
        while !data.pending.is_empty() {
            data = wait_with_budget(&self.shared.changed, data, budget)?;
        }
        Ok(())
    }

    /// Atomically turns a pending dispatch into an active execution scope.
    pub fn install(
        &self,
        pending: &Arc<PendingDispatch>,
        spec: ScopeSpec,
    ) -> Result<Scope, RegistryError> {
        let Some(owner) = pending.registry.upgrade() else {
            return Err(RegistryError::InvalidPendingDispatch);
        };
        if !Arc::ptr_eq(&owner, &self.shared) {
            return Err(RegistryError::InvalidPendingDispatch);
        }

        let mut resolved = lock_unpoisoned(&pending.resolved);
        let mut data = lock_unpoisoned(&self.shared.data);
        if self.state() != State::Accepting {
            *resolved = true;
            data.pending.remove(&pending.id);
            self.shared.changed.notify_all();
            return Err(RegistryError::NotAccepting);
        }
        if *resolved || !data.pending.remove(&pending.id) {
            return Err(RegistryError::InvalidPendingDispatch);
        }
        *resolved = true;

        let inner = Arc::new(ScopeInner {
            id: pending.id,
            registry: Arc::downgrade(&self.shared),
            spec,
            state: Mutex::new(ScopeState {
                close_fn: None,
                close_started: false,
                close_done: false,
                release_ticket: None,
                active: true,
                end_started: false,
                end_done: false,
            }),
            changed: Condvar::new(),
        });
        data.scopes.insert(pending.id, Arc::clone(&inner));
        self.shared.changed.notify_all();
        Ok(Scope { inner })
    }

    /// Replaces the cumulative release sink and replays every known group.
    pub fn set_release_sink(&self, sink: Option<ReleaseSink>) {
        let sequences = {
            let mut data = lock_unpoisoned(&self.shared.data);
            data.release_sink = sink.clone();
            data.release_sequences.clone()
        };
        if let Some(sink) = sink {
            for (group, sequence) in sequences {
                if sequence > 0 {
                    sink(group, sequence);
                }
            }
        }
    }

    /// Convenience for legacy sinks that do not acknowledge releases.
    pub fn set_release_callback<F>(&self, sink: F)
    where
        F: Fn(ReleaseGroup, i64) + Send + Sync + 'static,
    {
        self.set_release_sink(Some(Arc::new(move |group, sequence| {
            sink(group, sequence);
            None
        })));
    }

    /// Rejects new work, starts resource cancellation, and waits for owners.
    pub fn drain(&self, budget: WaitBudget) -> Result<(), RegistryError> {
        match self.shared.state.compare_exchange(
            State::Accepting as u32,
            State::Draining as u32,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(current) if State::from_raw(current) == State::Draining => {}
            Err(_) => return Err(RegistryError::Closed),
        }

        let scopes = {
            let data = lock_unpoisoned(&self.shared.data);
            data.scopes.values().cloned().collect::<Vec<_>>()
        };
        for scope in scopes {
            Scope::start_bound_resource_close(&scope);
        }

        let mut data = lock_unpoisoned(&self.shared.data);
        while !data.pending.is_empty() || !data.scopes.is_empty() {
            data = wait_with_budget(&self.shared.changed, data, budget)?;
        }
        self.shared
            .state
            .store(State::Closed as u32, Ordering::Release);
        Ok(())
    }

    /// Permanently rejects work and waits for all currently bound resources.
    pub fn close(&self) -> Result<(), RegistryError> {
        let mut close = lock_unpoisoned(&self.shared.close);
        if close.started {
            while !close.done {
                close = wait_unpoisoned(&self.shared.close_changed, close);
            }
            return Ok(());
        }
        if self.state() == State::Closed {
            return Ok(());
        }
        close.started = true;
        drop(close);

        self.shared
            .state
            .store(State::Closed as u32, Ordering::Release);
        let scopes = {
            let data = lock_unpoisoned(&self.shared.data);
            data.scopes.values().cloned().collect::<Vec<_>>()
        };
        for scope in scopes {
            Scope::wait_for_bound_resource_close(&scope);
        }

        let mut close = lock_unpoisoned(&self.shared.close);
        close.done = true;
        self.shared.close_changed.notify_all();
        Ok(())
    }

    pub(super) fn shared(&self) -> &Arc<RegistryShared> {
        &self.shared
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Scope {
    pub fn spec(&self) -> &ScopeSpec {
        &self.inner.spec
    }

    pub(super) fn from_inner(inner: Arc<ScopeInner>) -> Self {
        Self { inner }
    }

    /// Attaches exactly one closeable execution resource.
    pub fn bind<F>(&self, close_fn: F) -> Result<(), RegistryError>
    where
        F: FnOnce() -> Result<(), String> + Send + 'static,
    {
        let Some(registry) = self.inner.registry.upgrade() else {
            return Err(RegistryError::InvalidExecutionResource);
        };
        let _data = lock_unpoisoned(&registry.data);
        if State::from_raw(registry.state.load(Ordering::Acquire)) != State::Accepting {
            return Err(RegistryError::NotAccepting);
        }
        let mut state = lock_unpoisoned(&self.inner.state);
        if !state.active {
            return Err(RegistryError::NotAccepting);
        }
        if state.close_fn.is_some() || state.close_started {
            return Err(RegistryError::ExecutionResourceAlreadyBound);
        }
        state.close_fn = Some(Box::new(close_fn));
        Ok(())
    }

    pub fn end(&self, reason: &str) {
        self.end_with_release(reason);
    }

    /// Ends exactly once and returns any acknowledgement ticket from the sink.
    pub fn end_with_release(&self, _reason: &str) -> Option<ReleaseTicket> {
        let registry = self.inner.registry.upgrade()?;

        {
            let _data = lock_unpoisoned(&registry.data);
            let mut state = lock_unpoisoned(&self.inner.state);
            if state.end_started {
                drop(state);
                drop(_data);
                return self.wait_for_end();
            }
            state.active = false;
            state.end_started = true;
        }

        Self::wait_for_bound_resource_close(&self.inner);

        let (sink, group, sequence) = {
            let mut data = lock_unpoisoned(&registry.data);
            if self.inner.spec.accounted {
                let group = ReleaseGroup {
                    credential_id: self.inner.spec.credential_id.clone(),
                    model: self.inner.spec.model.clone(),
                };
                let sequence = {
                    let sequence = data.release_sequences.entry(group.clone()).or_insert(0);
                    *sequence += 1;
                    *sequence
                };
                (data.release_sink.clone(), Some(group), sequence)
            } else {
                (None, None, 0)
            }
        };
        let ticket = match (sink, group) {
            (Some(sink), Some(group)) if sequence > 0 => sink(group, sequence),
            _ => None,
        };

        {
            let mut state = lock_unpoisoned(&self.inner.state);
            state.release_ticket = ticket.clone();
        }
        {
            let mut data = lock_unpoisoned(&registry.data);
            data.scopes.remove(&self.inner.id);
            registry.changed.notify_all();
        }
        {
            let mut state = lock_unpoisoned(&self.inner.state);
            state.end_done = true;
            self.inner.changed.notify_all();
        }
        ticket
    }

    fn wait_for_end(&self) -> Option<ReleaseTicket> {
        let mut state = lock_unpoisoned(&self.inner.state);
        while !state.end_done {
            state = wait_unpoisoned(&self.inner.changed, state);
        }
        state.release_ticket.clone()
    }

    fn start_bound_resource_close(inner: &Arc<ScopeInner>) {
        let close_fn = {
            let mut state = lock_unpoisoned(&inner.state);
            if state.close_started || state.close_done {
                return;
            }
            state.close_started = true;
            match state.close_fn.take() {
                Some(close_fn) => Some(close_fn),
                None => {
                    state.close_done = true;
                    inner.changed.notify_all();
                    None
                }
            }
        };
        if let Some(close_fn) = close_fn {
            let inner = Arc::clone(inner);
            std::thread::spawn(move || {
                // Upstream logs close failures but does not make them a drain
                // failure. The port retains that lifecycle contract.
                let _close_result = close_fn();
                let mut state = lock_unpoisoned(&inner.state);
                state.close_done = true;
                inner.changed.notify_all();
            });
        }
    }

    fn wait_for_bound_resource_close(inner: &Arc<ScopeInner>) {
        Self::start_bound_resource_close(inner);
        let mut state = lock_unpoisoned(&inner.state);
        while !state.close_done {
            state = wait_unpoisoned(&inner.changed, state);
        }
    }
}

pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait_unpoisoned<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar
        .wait(guard)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait_with_budget<'a, T>(
    condvar: &Condvar,
    guard: MutexGuard<'a, T>,
    budget: WaitBudget,
) -> Result<MutexGuard<'a, T>, RegistryError> {
    match budget.remaining()? {
        None => Ok(wait_unpoisoned(condvar, guard)),
        Some(remaining) => {
            let (guard, outcome) = condvar
                .wait_timeout(guard, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if outcome.timed_out() {
                Err(RegistryError::DeadlineExceeded)
            } else {
                Ok(guard)
            }
        }
    }
}
