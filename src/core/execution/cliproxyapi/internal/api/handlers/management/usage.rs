// ref: internal/api/handlers/management/usage.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementUsageQueueError {
    StoreUnavailable,
}

impl fmt::Display for ManagementUsageQueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreUnavailable => formatter.write_str("usage queue unavailable"),
        }
    }
}

impl std::error::Error for ManagementUsageQueueError {}

/// Destructive, oldest-first access to a host-owned durable usage queue.
///
/// The upstream implementation reaches into a process-global Redis queue. The
/// portable boundary instead requires the host to provide an explicitly owned
/// adapter so queue authority and persistence remain visible to CTOX.
pub trait ManagementUsageQueue: Send + Sync {
    fn pop_oldest(&self, count: usize) -> Result<Vec<Vec<u8>>, ManagementUsageQueueError>;
}

pub fn parse_usage_queue_count(value: Option<&str>) -> Result<usize, &'static str> {
    let value = value.unwrap_or_default().trim();
    if value.is_empty() {
        return Ok(1);
    }
    value
        .parse::<usize>()
        .ok()
        .filter(|count| *count > 0)
        .ok_or("count must be a positive integer")
}

/// Encodes raw queue entries with the same semantics as Go's custom
/// `usageQueueRecord.MarshalJSON`: valid JSON is embedded as-is, while invalid
/// bytes are exposed as a JSON string.
pub fn usage_queue_payload(items: &[Vec<u8>]) -> Vec<u8> {
    let estimated_bytes = items
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .saturating_add(items.len())
        .saturating_add(2);
    let mut payload = Vec::with_capacity(estimated_bytes);
    payload.push(b'[');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            payload.push(b',');
        }
        if serde_json::from_slice::<serde_json::Value>(item).is_ok() {
            payload.extend_from_slice(item);
        } else {
            let string = String::from_utf8_lossy(item);
            payload.extend_from_slice(
                &serde_json::to_vec(string.as_ref()).unwrap_or_else(|_| b"\"\"".to_vec()),
            );
        }
    }
    payload.push(b']');
    payload
}
