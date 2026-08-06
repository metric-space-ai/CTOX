// ref: internal/logging/requestmeta.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, RwLock};

pub type ResponseHeaders = BTreeMap<String, Vec<String>>;

/// Immutable downstream request metadata. Cloning a [`RequestContext`] keeps a
/// value snapshot here while response holders remain request-local shared state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClientRequestMetadata {
    pub client_ip: String,
    pub x_forwarded_for: String,
    pub user_agent: String,
}

#[derive(Clone, Default)]
pub struct RequestContext {
    pub(super) request_id: Option<String>,
    pub(super) handler_request_id: Option<String>,
    endpoint: Option<String>,
    client_request_metadata: ClientRequestMetadata,
    response_status: Option<Arc<AtomicI32>>,
    response_headers: Option<Arc<RwLock<Option<ResponseHeaders>>>>,
    pub(super) cpa_trace_id: Option<Arc<RwLock<String>>>,
}

#[must_use]
pub fn with_endpoint(
    context: Option<&RequestContext>,
    endpoint: impl Into<String>,
) -> RequestContext {
    let mut derived = context.cloned().unwrap_or_default();
    derived.endpoint = Some(endpoint.into());
    derived
}

#[must_use]
pub fn get_endpoint(context: Option<&RequestContext>) -> &str {
    context
        .and_then(|context| context.endpoint.as_deref())
        .unwrap_or_default()
}

#[must_use]
pub fn with_client_request_metadata(
    context: Option<&RequestContext>,
    metadata: ClientRequestMetadata,
) -> RequestContext {
    let mut derived = context.cloned().unwrap_or_default();
    derived.client_request_metadata = metadata;
    derived
}

#[must_use]
pub fn get_client_request_metadata(context: Option<&RequestContext>) -> ClientRequestMetadata {
    context
        .map(|context| context.client_request_metadata.clone())
        .unwrap_or_default()
}

#[must_use]
pub fn with_response_status_holder(context: Option<&RequestContext>) -> RequestContext {
    let mut derived = context.cloned().unwrap_or_default();
    if derived.response_status.is_none() {
        derived.response_status = Some(Arc::new(AtomicI32::new(0)));
    }
    derived
}

#[must_use]
pub fn with_response_headers_holder(context: Option<&RequestContext>) -> RequestContext {
    let mut derived = context.cloned().unwrap_or_default();
    if derived.response_headers.is_none() {
        derived.response_headers = Some(Arc::new(RwLock::new(None)));
    }
    derived
}

pub fn set_response_status(context: Option<&RequestContext>, status: i32) {
    if status <= 0 {
        return;
    }
    if let Some(holder) = context.and_then(|context| context.response_status.as_ref()) {
        holder.store(status, Ordering::SeqCst);
    }
}

pub fn set_response_headers(context: Option<&RequestContext>, headers: &ResponseHeaders) {
    let Some(holder) = context.and_then(|context| context.response_headers.as_ref()) else {
        return;
    };
    *write_unpoisoned(holder) = clone_headers(headers);
}

#[must_use]
pub fn get_response_status(context: Option<&RequestContext>) -> i32 {
    context
        .and_then(|context| context.response_status.as_ref())
        .map(|holder| holder.load(Ordering::SeqCst))
        .unwrap_or_default()
}

#[must_use]
pub fn get_response_headers(context: Option<&RequestContext>) -> Option<ResponseHeaders> {
    let holder = context.and_then(|context| context.response_headers.as_ref())?;
    read_unpoisoned(holder).clone()
}

fn clone_headers(headers: &ResponseHeaders) -> Option<ResponseHeaders> {
    (!headers.is_empty()).then(|| headers.clone())
}

pub(super) fn read_unpoisoned<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(super) fn write_unpoisoned<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
