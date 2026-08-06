// Origin: CTOX
// License: AGPL-3.0-only

pub mod certificate;
pub mod client;
pub mod concurrency_release;
pub mod global;
pub mod kv_helpers;
pub mod plugin_status;
pub mod requests;

pub use client::{
    Client, DispatchFailureStage, HomeConfig, HomeError, HomeTransport, KvSetOptions,
    TransportFailure,
};
pub use kv_helpers::hash_key_part;
pub use requests::{
    InFlightAccountedStatus, InFlightAggregate, InFlightFrameKind, InFlightRequestDetail,
    InFlightSnapshotFrame,
};

#[cfg(test)]
mod candidate_refresh_fingerprint_test;
#[cfg(test)]
mod client_test;
#[cfg(test)]
mod concurrency_release_test;
#[cfg(test)]
mod in_flight_contract_test;
#[cfg(test)]
mod kv_helpers_test;
#[cfg(test)]
mod plugin_status_test;
