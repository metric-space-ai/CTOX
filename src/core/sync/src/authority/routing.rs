//! The only execution-request retry boundary, shared by voters and workers.
//! Routing hints never authorize execution. Each attempt keeps the same request
//! and durable ID; a committed replay remains a replay, never a fresh effect grant.
use super::{
    network::{AuthorityFailure, Reply, Rpc},
    NodeId, Peer,
};
use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    io,
    time::Duration,
};
use tokio::time::{sleep_until, timeout_at, Instant};

impl std::fmt::Display for AuthorityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLeader { leader } => write!(f, "authority leader changed: {leader:?}"),
            Self::Unavailable { reason } | Self::Rejected { reason } => f.write_str(reason),
        }
    }
}
impl std::error::Error for AuthorityFailure {}

impl AuthorityFailure {
    pub(super) fn unavailable(reason: impl std::fmt::Display) -> Self {
        Self::Unavailable {
            reason: reason.to_string(),
        }
    }
    pub(super) fn rejected(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: reason.into(),
        }
    }
}

pub(super) async fn route<F, Fut, S>(
    rpc: Rpc,
    voters: &BTreeMap<NodeId, Peer>,
    mut preferred: Option<NodeId>,
    stopped: S,
    exchange: F,
) -> io::Result<(NodeId, Reply)>
where
    F: Fn(NodeId, Peer, Rpc) -> Fut,
    Fut: Future<Output = io::Result<Reply>>,
    S: Fn() -> bool,
{
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut backoff = Duration::from_millis(50);
    // Bounded to the three pinned voters; preserve every last route failure.
    let mut failures = BTreeMap::new();
    loop {
        let mut pending: VecDeque<_> = voters.keys().copied().collect();
        prioritize(&mut pending, preferred);
        while let Some(id) = pending.pop_front() {
            if stopped() {
                return Err(io::Error::other("execution authority is stopped"));
            }
            if Instant::now() >= deadline {
                break;
            }
            let attempt = (Instant::now() + Duration::from_secs(2)).min(deadline);
            let answer = timeout_at(attempt, exchange(id, voters[&id].clone(), rpc.clone())).await;
            if stopped() {
                return Err(io::Error::other(
                    "execution authority stopped before confirmation",
                ));
            }
            match answer {
                Ok(Ok(reply)) => {
                    let failure = match (&rpc, &reply) {
                        (Rpc::Propose(_), Reply::Propose(Ok(_)))
                        | (Rpc::WorkerMembership { .. }, Reply::WorkerMembership(Ok(_)))
                        | (Rpc::Validate { .. }, Reply::Validate(Ok(_))) => return Ok((id, reply)),
                        (Rpc::Propose(_), Reply::Propose(Err(failure)))
                        | (Rpc::WorkerMembership { .. }, Reply::WorkerMembership(Err(failure)))
                        | (Rpc::Validate { .. }, Reply::Validate(Err(failure))) => failure,
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "unexpected authority response",
                            ))
                        }
                    };
                    if let AuthorityFailure::Rejected { .. } = failure {
                        // Only an authenticated, typed rejection is terminal.
                        // Transport errors cannot speak for group membership.
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            failure.clone(),
                        ));
                    }
                    if let AuthorityFailure::NotLeader { leader } = failure {
                        if leader.is_some_and(|leader| voters.contains_key(&leader)) {
                            preferred = *leader;
                            // Never revisit a voter within a round, even for a cyclic hint.
                            prioritize(&mut pending, preferred);
                        }
                    }
                    failures.insert(id, failure.to_string());
                }
                Ok(Err(error)) => {
                    failures.insert(id, error.to_string());
                }
                Err(_) => {
                    failures.insert(
                        id,
                        "authority voter did not answer before its deadline".into(),
                    );
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::other(format!(
                "execution paused; authority outcome is unconfirmed (retain the request ID for reconciliation): {failures:?}"
            )));
        }
        // A single bounded recovery loop; success has no added delay. This
        // handles the interval where every voter still reports no leader.
        sleep_until((Instant::now() + backoff).min(deadline)).await;
        backoff = (backoff * 2).min(Duration::from_millis(250));
    }
}

fn prioritize(pending: &mut VecDeque<NodeId>, preferred: Option<NodeId>) {
    if let Some(index) = pending.iter().position(|id| Some(*id) == preferred) {
        let id = pending.remove(index).expect("position is in the queue");
        pending.push_front(id);
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
