//! Execution control over the existing multiplexed CTOX Sync DataChannel.
//! Routing hints may change on reconnect; configured signing keys remain authority.
use super::{
    auth::{self, ControlChannel, SigningIdentity},
    network::CONTROL_METHOD,
    node::AuthorityNode,
};
use async_trait::async_trait;
use rxdb::plugins::replication_webrtc::{
    send_message_and_await_answer, RxWebRTCReplicationPool, WebRTCConnectionHandler, WebRTCMessage,
    WebRTCRsConnectionHandler,
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    sync::{Arc, RwLock, Weak},
    time::Duration,
};

pub struct WebRtcControlChannel {
    pool: Weak<RxWebRTCReplicationPool<WebRTCRsConnectionHandler>>,
    allowed: BTreeSet<String>,
    routes: RwLock<BTreeMap<String, String>>,
    deadline: Duration,
}
impl WebRtcControlChannel {
    pub fn new(
        pool: &Arc<RxWebRTCReplicationPool<WebRTCRsConnectionHandler>>,
        allowed: BTreeSet<String>,
        deadline: Duration,
    ) -> io::Result<Self> {
        for identity in &allowed {
            auth::public_key(identity)?;
        }
        if allowed.len() != 3 || deadline.is_zero() || deadline > Duration::from_secs(30) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "control channel requires three configured keys and a bounded deadline",
            ));
        }
        Ok(Self {
            pool: Arc::downgrade(pool),
            allowed,
            routes: RwLock::new(BTreeMap::new()),
            deadline,
        })
    }
    /// Discovery supplies a route only. SignedTransport verifies the endpoint key.
    pub fn set_route(&self, identity: &str, peer: String) -> io::Result<()> {
        if !self.allowed.contains(identity) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "unconfigured control peer",
            ));
        }
        self.routes
            .write()
            .map_err(|_| io::Error::other("control route lock poisoned"))?
            .insert(identity.into(), peer);
        Ok(())
    }
}
#[async_trait]
impl ControlChannel for WebRtcControlChannel {
    async fn request(&self, target_identity: &str, envelope: Value) -> io::Result<Value> {
        let peer = self
            .routes
            .read()
            .map_err(|_| io::Error::other("control route lock poisoned"))?
            .get(target_identity)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotConnected,
                    "control peer has no live WebRTC route",
                )
            })?;
        let pool = self.pool.upgrade().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotConnected,
                "authority replication pool is stopped",
            )
        })?;
        // Configuration pins a routing hint, never a reusable connection handle.
        // Resolve it anew, then bind the entire request to that one lifetime.
        let peer = pool
            .connection_handler
            .connection_for_peer(&peer)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotConnected,
                    "control peer has no open WebRTC connection",
                )
            })?;
        if !pool.is_peer_ready_for_control(&peer) {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "authority peer has not completed room admission",
            ));
        }
        // A signed nonce also supplies the existing multiplexer's correlation ID.
        let nonce = envelope["body"]["nonce"]
            .as_str()
            .filter(|n| n.len() == 32 && n.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing control nonce"))?;
        let request = WebRTCMessage {
            id: format!("{CONTROL_METHOD}:{nonce}"),
            method: CONTROL_METHOD.into(),
            params: vec![envelope],
            collection: None,
        };
        let response = tokio::time::timeout(
            self.deadline,
            send_message_and_await_answer(pool.connection_handler.clone(), peer, request),
        )
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "authority WebRTC exchange timed out",
            )
        })?
        .map_err(|e| io::Error::other(e.to_string()))?;
        if let Some(error) = response.error {
            return Err(io::Error::other(error));
        }
        Ok(response.result)
    }
}
/// Install once during the host-owned pool lifecycle. The pool's room admission
/// still applies; the signed envelope then enforces the narrower voting group.
/// Weak node capture ensures a surviving pool cannot keep a stopped authority alive.
pub fn register_receiver<H: WebRTCConnectionHandler + 'static>(
    pool: &RxWebRTCReplicationPool<H>,
    identity: Arc<SigningIdentity>,
    scope: String,
    node: &Arc<AuthorityNode>,
) -> io::Result<()> {
    if !node.matches_local_identity(&scope, &identity.public_identity()) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "control receiver does not match its authority node",
        ));
    }
    let node = Arc::downgrade(node);
    pool.register_auxiliary_request_handler(
        CONTROL_METHOD,
        Arc::new(move |_route_identity, _capability, mut params| {
            let identity = identity.clone();
            let scope = scope.clone();
            let node = node.clone();
            Box::pin(async move {
                if params.len() != 1 {
                    return Err("authority expects one signed envelope".into());
                }
                let node = node
                    .upgrade()
                    .ok_or_else(|| "authority is stopped".to_owned())?;
                auth::receive(&identity, &scope, &node, params.remove(0))
                    .await
                    .map_err(|e| e.to_string())
            })
        }),
    )
    .map_err(|error| io::Error::other(error.to_string()))
}
