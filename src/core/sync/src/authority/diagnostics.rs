//! Local, bounded counters: labels are internal operation names, never request data.
//! These observations do not establish authority and are not persisted or replicated.
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Queued,
    Waiting,
    Running,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PhaseTiming {
    pub active: u64,
    pub finished: u64,
    pub max_micros: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct OperationTiming {
    pub queued: PhaseTiming,
    pub waiting: PhaseTiming,
    pub running: PhaseTiming,
    pub succeeded: u64,
    pub failed: u64,
    /// Dropped or panicked observation; does not imply rollback of a Raft write.
    pub interrupted: u64,
}
impl OperationTiming {
    fn phase(&mut self, phase: Phase) -> &mut PhaseTiming {
        match phase {
            Phase::Queued => &mut self.queued,
            Phase::Waiting => &mut self.waiting,
            Phase::Running => &mut self.running,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct Timings(Arc<Mutex<BTreeMap<&'static str, OperationTiming>>>);
impl Timings {
    pub(crate) fn snapshot(&self) -> BTreeMap<&'static str, OperationTiming> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub(crate) fn start(&self, name: &'static str, phase: Phase) -> Observation {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(name)
            .or_default()
            .phase(phase)
            .active += 1;
        Observation {
            timings: self.clone(),
            name,
            phase,
            since: Instant::now(),
            outcome: None,
        }
    }
}
pub(crate) struct Observation {
    timings: Timings,
    name: &'static str,
    phase: Phase,
    since: Instant,
    outcome: Option<bool>,
}
impl Observation {
    pub(crate) fn enter(&mut self, phase: Phase) {
        self.finish_phase();
        self.phase = phase;
        self.since = Instant::now();
        self.timings
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(self.name)
            .or_default()
            .phase(phase)
            .active += 1;
    }
    pub(crate) fn finish(mut self, succeeded: bool) {
        self.outcome = Some(succeeded);
    }
    fn finish_phase(&self) {
        let mut timings = self.timings.0.lock().unwrap_or_else(|e| e.into_inner());
        let phase = timings.entry(self.name).or_default().phase(self.phase);
        phase.active -= 1;
        phase.finished += 1;
        phase.max_micros = phase
            .max_micros
            .max(self.since.elapsed().as_micros().min(u64::MAX as u128) as u64);
    }
}
impl Drop for Observation {
    fn drop(&mut self) {
        self.finish_phase();
        let mut timings = self.timings.0.lock().unwrap_or_else(|e| e.into_inner());
        let operation = timings.entry(self.name).or_default();
        match self.outcome {
            Some(true) => operation.succeeded += 1,
            Some(false) => operation.failed += 1,
            None => operation.interrupted += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phases_and_interruption_remain_distinct_under_overlap() {
        let timings = Timings::default();
        let mut first = timings.start("append", Phase::Queued);
        let second = timings.start("append", Phase::Queued);
        first.enter(Phase::Waiting);
        let snapshot = timings.snapshot();
        assert_eq!(snapshot["append"].queued.active, 1);
        assert_eq!(snapshot["append"].waiting.active, 1);
        assert_eq!(snapshot["append"].queued.finished, 1);
        drop(second);
        first.enter(Phase::Running);
        first.finish(true);
        timings.start("append", Phase::Running).finish(false);
        let snapshot = timings.snapshot();
        let op = &snapshot["append"];
        assert_eq!(
            (op.queued.active, op.waiting.active, op.running.active),
            (0, 0, 0)
        );
        assert_eq!((op.succeeded, op.failed, op.interrupted), (1, 1, 1));
        assert_eq!(
            (op.queued.finished, op.waiting.finished, op.running.finished),
            (2, 1, 2)
        );
    }
    #[test]
    fn history_is_aggregated_instead_of_retaining_each_request() {
        let timings = Timings::default();
        for _ in 0..10_000 {
            timings.start("client_write", Phase::Running).finish(true);
        }
        let snapshot = timings.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot["client_write"].succeeded, 10_000);
    }
}
