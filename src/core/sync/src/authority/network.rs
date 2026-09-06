//! Raft RPCs use an authenticated CTOX Sync control channel, never an HTTP data bridge.
use super::{Job, NodeId, Ownership, Peer, Receipt, Request, TypeConfig, WorkerMembership};
pub use crate::contracts::AuthorityFailure;
use async_trait::async_trait;
use openraft::{
    error::{InstallSnapshotError, NetworkError, RPCError, RaftError, RemoteError},
    network::{RPCOption, RaftNetwork, RaftNetworkFactory},
    raft::{
        AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
        InstallSnapshotResponse, VoteRequest, VoteResponse,
    },
};
use serde::{Deserialize, Serialize};
use std::{io, sync::Arc};

pub const CONTROL_PROTOCOL: u32 = crate::contracts::CTOX_SYNC_AUTHORITY_PROTOCOL_VERSION;
pub const CONTROL_METHOD: &str = "ctox.sync.authority.v5";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Packet {
    pub version: u32,
    pub scope_id: String,
    pub from: NodeId,
    pub rpc: Rpc,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "camelCase")]
pub enum Rpc {
    Append(AppendEntriesRequest<TypeConfig>),
    Vote(VoteRequest<NodeId>),
    Snapshot(InstallSnapshotRequest<TypeConfig>),
    Propose(Request),
    WorkerMembership {
        node_id: NodeId,
    },
    Validate {
        job_id: String,
        ownership: Ownership,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", content = "result", rename_all = "camelCase")]
pub enum Reply {
    Append(Result<AppendEntriesResponse<NodeId>, RaftError<NodeId>>),
    Vote(Result<VoteResponse<NodeId>, RaftError<NodeId>>),
    Snapshot(Result<InstallSnapshotResponse<NodeId>, RaftError<NodeId, InstallSnapshotError>>),
    Propose(Result<Receipt, AuthorityFailure>),
    WorkerMembership(Result<Option<WorkerMembership>, AuthorityFailure>),
    Validate(Result<Job, AuthorityFailure>),
}

/// Implementations must bind the remote DataChannel identity to target.identity.
/// They must not route by an unverified peer-id field from a message body.
#[async_trait]
pub trait ControlTransport: Send + Sync + 'static {
    async fn exchange(&self, target: &Peer, packet: Packet) -> io::Result<Reply>;
}
#[derive(Clone)]
pub struct Factory {
    pub from: NodeId,
    pub scope_id: String,
    pub transport: Arc<dyn ControlTransport>,
}
pub struct Channel {
    factory: Factory,
    target: NodeId,
    peer: Peer,
}
impl RaftNetworkFactory<TypeConfig> for Factory {
    type Network = Channel;
    async fn new_client(&mut self, target: NodeId, peer: &Peer) -> Channel {
        Channel {
            factory: self.clone(),
            target,
            peer: peer.clone(),
        }
    }
}
impl Channel {
    async fn send(&self, rpc: Rpc, option: RPCOption) -> io::Result<Reply> {
        let packet = Packet {
            version: CONTROL_PROTOCOL,
            scope_id: self.factory.scope_id.clone(),
            from: self.factory.from,
            rpc,
        };
        tokio::time::timeout(
            option.hard_ttl(),
            self.factory.transport.exchange(&self.peer, packet),
        )
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "CTOX Sync authority RPC timed out"))?
    }
}
fn network_error<E: std::error::Error>(e: io::Error) -> RPCError<NodeId, Peer, E> {
    RPCError::Network(NetworkError::new(&e))
}
fn wrong_reply() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "CTOX Sync authority RPC reply type mismatch",
    )
}
impl RaftNetwork<TypeConfig> for Channel {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<AppendEntriesResponse<NodeId>, RPCError<NodeId, Peer, RaftError<NodeId>>> {
        match self
            .send(Rpc::Append(rpc), option)
            .await
            .map_err(network_error)?
        {
            Reply::Append(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(network_error(wrong_reply())),
        }
    }
    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        option: RPCOption,
    ) -> Result<VoteResponse<NodeId>, RPCError<NodeId, Peer, RaftError<NodeId>>> {
        match self
            .send(Rpc::Vote(rpc), option)
            .await
            .map_err(network_error)?
        {
            Reply::Vote(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(network_error(wrong_reply())),
        }
    }
    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<NodeId>,
        RPCError<NodeId, Peer, RaftError<NodeId, InstallSnapshotError>>,
    > {
        match self
            .send(Rpc::Snapshot(rpc), option)
            .await
            .map_err(network_error)?
        {
            Reply::Snapshot(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(network_error(wrong_reply())),
        }
    }
}
