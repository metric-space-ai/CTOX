// ref: sdk/cliproxy/auth/home_in_flight_publisher.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: publisher pins one registry and injected Home transport lifetime
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::internal::home::{
    Client, HomeError, InFlightAccountedStatus, InFlightAggregate, InFlightFrameKind,
    InFlightRequestDetail, InFlightSnapshotFrame,
};
use crate::sdk::cliproxy::executionregistry::{Freeze, Observation, Registry};

use super::canonical_home_concurrency_model_key;
use super::conductor_home::{HomeClock, SystemHomeClock};

pub trait HomeInFlightTransport: Send + Sync {
    fn heartbeat_ok(&self) -> bool;
    fn push_in_flight_snapshot(&self, payload: &[u8]) -> Result<(), HomeError>;
}

impl HomeInFlightTransport for Client {
    fn heartbeat_ok(&self) -> bool {
        Client::heartbeat_ok(self)
    }

    fn push_in_flight_snapshot(&self, payload: &[u8]) -> Result<(), HomeError> {
        Client::push_in_flight_snapshot(self, payload)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HomeInFlightPublisherConfig {
    pub snapshot_interval: Duration,
    pub max_part_bytes: usize,
    pub max_part_count: usize,
    pub max_revision_bytes: usize,
    pub max_aggregate_groups: usize,
    pub max_details: usize,
    pub max_string_bytes: usize,
}

impl Default for HomeInFlightPublisherConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: Duration::from_secs(2),
            max_part_bytes: 64 * 1024,
            max_part_count: 16,
            max_revision_bytes: 512 * 1024,
            max_aggregate_groups: 4_096,
            max_details: 2_048,
            max_string_bytes: 256,
        }
    }
}

impl HomeInFlightPublisherConfig {
    pub fn validate(self) -> Result<Self, HomePublisherConfigError> {
        if self.snapshot_interval.is_zero()
            || self.max_part_bytes < 1_024
            || self.max_part_count == 0
            || self.max_part_count > 64
            || self.max_revision_bytes < self.max_part_bytes
            || self.max_revision_bytes > 4 * 1024 * 1024
            || self.max_aggregate_groups == 0
            || self.max_details > 16_384
            || self.max_string_bytes == 0
            || self.max_string_bytes > 4_096
            || self.max_revision_bytes.div_ceil(self.max_part_bytes) > self.max_part_count
        {
            return Err(HomePublisherConfigError);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HomePublisherConfigError;

impl fmt::Display for HomePublisherConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Home in-flight publisher configuration is invalid")
    }
}

impl std::error::Error for HomePublisherConfigError {}

pub struct HomeInFlightPublisher {
    transport: Arc<dyn HomeInFlightTransport>,
    registry: Arc<Registry>,
    config: RwLock<HomeInFlightPublisherConfig>,
    clock: Arc<dyn HomeClock>,
}

impl fmt::Debug for HomeInFlightPublisher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HomeInFlightPublisher")
            .field("heartbeat_ok", &self.transport.heartbeat_ok())
            .field("registry_state", &self.registry.state())
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

impl HomeInFlightPublisher {
    pub fn new(
        transport: Arc<dyn HomeInFlightTransport>,
        registry: Arc<Registry>,
        config: HomeInFlightPublisherConfig,
    ) -> Result<Self, HomePublisherConfigError> {
        Self::new_with_clock(transport, registry, config, Arc::new(SystemHomeClock))
    }

    pub fn new_with_clock(
        transport: Arc<dyn HomeInFlightTransport>,
        registry: Arc<Registry>,
        config: HomeInFlightPublisherConfig,
        clock: Arc<dyn HomeClock>,
    ) -> Result<Self, HomePublisherConfigError> {
        Ok(Self {
            transport,
            registry,
            config: RwLock::new(config.validate()?),
            clock,
        })
    }

    pub fn apply_config(
        &self,
        config: HomeInFlightPublisherConfig,
    ) -> Result<(), HomePublisherConfigError> {
        *self
            .config
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = config.validate()?;
        Ok(())
    }

    pub fn config(&self) -> HomeInFlightPublisherConfig {
        *self
            .config
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn publish_once(&self, observed_at: DateTime<Utc>) -> Result<usize, HomeError> {
        if !self.transport.heartbeat_ok() {
            return Ok(0);
        }
        let freeze = self.registry.freeze_in_flight(observed_at);
        let frames = encode_home_in_flight_freeze(&freeze, observed_at, self.config());
        for frame in &frames {
            let payload = serde_json::to_vec(frame).map_err(|_| {
                HomeError::InvalidRequest("in-flight frame encode failed".to_owned())
            })?;
            self.transport.push_in_flight_snapshot(&payload)?;
        }
        Ok(frames.len())
    }

    pub async fn run(&self, mut cancelled: tokio::sync::watch::Receiver<bool>) {
        loop {
            if *cancelled.borrow() {
                return;
            }
            let interval = self.config().snapshot_interval;
            tokio::select! {
                _ = tokio::time::sleep(interval) => { let _ = self.publish_once(self.clock.now()); }
                changed = cancelled.changed() => {
                    if changed.is_err() || *cancelled.borrow() { return; }
                }
            }
        }
    }
}

pub fn encode_home_in_flight_freeze(
    freeze: &Freeze,
    observed_at: DateTime<Utc>,
    config: HomeInFlightPublisherConfig,
) -> Vec<InFlightSnapshotFrame> {
    if config.validate().is_err() {
        return overflow(freeze, observed_at, 0);
    }
    let mut counts = BTreeMap::<(String, String, bool), i64>::new();
    let mut aggregate_valid = true;
    for observation in &freeze.executions {
        let model = observation_model(observation);
        aggregate_valid &= observation.credential_id.len() <= config.max_string_bytes
            && model.len() <= config.max_string_bytes;
        *counts
            .entry((
                observation.credential_id.clone(),
                model,
                observation.accounted,
            ))
            .or_default() += 1;
    }
    if !aggregate_valid || counts.len() > config.max_aggregate_groups {
        return overflow(freeze, observed_at, counts.len());
    }
    let aggregates = counts
        .into_iter()
        .map(
            |((credential_id, model, accounted), count)| InFlightAggregate {
                credential_id,
                model,
                status: if accounted {
                    InFlightAccountedStatus::Accounted
                } else {
                    InFlightAccountedStatus::Unaccounted
                },
                count,
            },
        )
        .collect::<Vec<_>>();
    let mut truncated = false;
    let mut details = freeze
        .executions
        .iter()
        .filter_map(|observation| {
            bounded_detail(observation, config.max_string_bytes, &mut truncated)
        })
        .collect::<Vec<_>>();
    details.sort_by(|left, right| {
        left.started_at
            .cmp(&right.started_at)
            .then_with(|| left.request_id.cmp(&right.request_id))
    });
    if details.len() > config.max_details {
        details.truncate(config.max_details);
        truncated = true;
    }
    pack_frames(freeze, observed_at, config, aggregates, details, truncated)
        .unwrap_or_else(|| overflow(freeze, observed_at, config.max_aggregate_groups + 1))
}

fn pack_frames(
    freeze: &Freeze,
    observed_at: DateTime<Utc>,
    config: HomeInFlightPublisherConfig,
    aggregates: Vec<InFlightAggregate>,
    details: Vec<InFlightRequestDetail>,
    mut truncated: bool,
) -> Option<Vec<InFlightSnapshotFrame>> {
    let mut frames = vec![part(freeze, observed_at, truncated)];
    for aggregate in aggregates {
        if !append_bounded_aggregate(
            &mut frames,
            aggregate,
            freeze,
            observed_at,
            config,
            truncated,
        ) {
            return None;
        }
    }
    for detail in details {
        if !append_bounded_detail(&mut frames, detail, freeze, observed_at, config, truncated) {
            truncated = true;
            break;
        }
    }
    let count = i32::try_from(frames.len()).ok()?;
    let mut total = 0;
    for (index, frame) in frames.iter_mut().enumerate() {
        frame.part_index = Some(i32::try_from(index).ok()?);
        frame.part_count = Some(count);
        frame.details_truncated |= truncated;
        let bytes = serde_json::to_vec(frame).ok()?.len();
        if bytes > config.max_part_bytes {
            return None;
        }
        total += bytes;
    }
    (total <= config.max_revision_bytes).then_some(frames)
}

fn append_bounded_aggregate(
    frames: &mut Vec<InFlightSnapshotFrame>,
    item: InFlightAggregate,
    freeze: &Freeze,
    observed_at: DateTime<Utc>,
    config: HomeInFlightPublisherConfig,
    truncated: bool,
) -> bool {
    frames.last_mut().unwrap().aggregates.push(item.clone());
    if frame_size(frames.last().unwrap()) <= config.max_part_bytes {
        return true;
    }
    frames.last_mut().unwrap().aggregates.pop();
    if frames.len() >= config.max_part_count {
        return false;
    }
    let mut next = part(freeze, observed_at, truncated);
    next.aggregates.push(item);
    if frame_size(&next) > config.max_part_bytes {
        return false;
    }
    frames.push(next);
    true
}

fn append_bounded_detail(
    frames: &mut Vec<InFlightSnapshotFrame>,
    item: InFlightRequestDetail,
    freeze: &Freeze,
    observed_at: DateTime<Utc>,
    config: HomeInFlightPublisherConfig,
    truncated: bool,
) -> bool {
    frames.last_mut().unwrap().details.push(item.clone());
    if frame_size(frames.last().unwrap()) <= config.max_part_bytes {
        return true;
    }
    frames.last_mut().unwrap().details.pop();
    if frames.len() >= config.max_part_count {
        return false;
    }
    let mut next = part(freeze, observed_at, truncated);
    next.details.push(item);
    if frame_size(&next) > config.max_part_bytes {
        return false;
    }
    frames.push(next);
    true
}

fn part(freeze: &Freeze, observed_at: DateTime<Utc>, truncated: bool) -> InFlightSnapshotFrame {
    InFlightSnapshotFrame {
        kind: InFlightFrameKind::Part,
        revision: freeze.revision,
        observed_at,
        barrier_revision: freeze.barrier_revision,
        part_index: Some(0),
        part_count: Some(1),
        details_truncated: truncated,
        aggregates: Vec::new(),
        details: Vec::new(),
        aggregate_group_count: 0,
    }
}

fn overflow(
    freeze: &Freeze,
    observed_at: DateTime<Utc>,
    groups: usize,
) -> Vec<InFlightSnapshotFrame> {
    vec![InFlightSnapshotFrame {
        kind: InFlightFrameKind::Overflow,
        revision: freeze.revision,
        observed_at,
        barrier_revision: freeze.barrier_revision,
        part_index: None,
        part_count: None,
        details_truncated: false,
        aggregates: Vec::new(),
        details: Vec::new(),
        aggregate_group_count: groups,
    }]
}

fn observation_model(observation: &Observation) -> String {
    if observation.accounted {
        observation.model.clone()
    } else {
        let model = canonical_home_concurrency_model_key(&observation.model);
        if model.is_empty() {
            "unknown".to_owned()
        } else {
            model
        }
    }
}

fn bounded_detail(
    observation: &Observation,
    max: usize,
    truncated: &mut bool,
) -> Option<InFlightRequestDetail> {
    let mut bound = |value: &str| {
        let bounded = truncate_utf8(value, max);
        *truncated |= bounded != value;
        bounded
    };
    let detail = InFlightRequestDetail {
        request_id: bound(&observation.request_id),
        credential_id: bound(&observation.credential_id),
        model: bound(&observation_model(observation)),
        request_kind: bound(&observation.request_kind),
        started_at: observation.started_at,
    };
    (!detail.request_id.trim().is_empty()
        && !detail.credential_id.trim().is_empty()
        && !detail.model.trim().is_empty()
        && !detail.request_kind.trim().is_empty())
    .then_some(detail)
}

fn truncate_utf8(value: &str, max: usize) -> String {
    if value.len() <= max {
        return value.to_owned();
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn frame_size(frame: &InFlightSnapshotFrame) -> usize {
    serde_json::to_vec(frame).map_or(usize::MAX, |raw| raw.len())
}
