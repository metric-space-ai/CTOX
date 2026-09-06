//! Majority authority attached to an admitted native replication session.
#[cfg(unix)]
use crate::{
    authority::{
        auth::SignedTransport,
        webrtc::{register_receiver, WebRtcControlChannel},
    },
    local_host::LocalAuthorityHost,
};
use crate::{
    authority::{
        auth::SigningIdentity, client::ExecutionAuthority, node::AuthorityNode, NodeId, Peer,
    },
    native::NativePool,
};
#[cfg(unix)]
use futures_util::{FutureExt, StreamExt};
use rxdb::plugins::replication_webrtc::SignalingClient;
#[cfg(unix)]
use rxdb::plugins::replication_webrtc::WebRTCConnectionHandler;
#[cfg(unix)]
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    sync::{oneshot, Mutex},
    task::JoinHandle,
};

/// Supplied from the host's confirmed configuration, never from discovery.
/// Routing names are hints; SignedTransport independently verifies every key.
pub struct ExecutionGroupOptions {
    pub timing: crate::authority::timing::AuthorityTiming,
    pub node_id: NodeId,
    pub scope_id: String,
    pub room: String,
    pub peers: BTreeMap<NodeId, Peer>,
    pub routes: BTreeMap<NodeId, String>,
    pub store_path: PathBuf,
    /// Dedicated private local-runtime directory; never replicated to peers.
    pub ipc_directory: PathBuf,
}

/// Confirmed nonvoting membership and authority routes supplied by the host.
/// There is deliberately no Raft store path on a worker.
pub struct WorkerExecutionOptions {
    pub member: crate::authority::WorkerMembership,
    pub scope_id: String,
    pub room: String,
    pub voters: BTreeMap<NodeId, Peer>,
    pub routes: BTreeMap<NodeId, String>,
    pub ipc_directory: PathBuf,
}

pub type NativeExecutionGroup = NativeExecutionHost<AuthorityNode>;
pub type NativeExecutionWorker =
    NativeExecutionHost<crate::authority::client::WorkerAuthorityClient>;

#[cfg(unix)]
#[derive(Clone, Copy)]
enum PeerInitiation {
    VoterPair,
    WorkerToVoter,
}

pub struct NativeExecutionHost<A: ExecutionAuthority + 'static> {
    node: Arc<A>,
    endpoint: PathBuf,
    stop: Mutex<Option<oneshot::Sender<()>>>,
    task: Mutex<Option<JoinHandle<io::Result<()>>>>,
}
impl NativeExecutionHost<AuthorityNode> {
    pub(crate) async fn attach(
        pool: &NativePool,
        signaling: &Arc<SignalingClient>,
        room: &str,
        options: ExecutionGroupOptions,
        key: Arc<SigningIdentity>,
    ) -> io::Result<Self> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "execution group does not match its confirmed native session",
            )
        };
        if options.room != room
            || options.routes.keys().ne(options.peers.keys())
            || options.routes.values().collect::<BTreeSet<_>>().len() != 3
            || options
                .routes
                .values()
                .any(|route| route.is_empty() || route.trim() != route || route.len() > 128)
            || options.routes.get(&options.node_id) != signaling.own_peer_id().as_ref()
            || options
                .peers
                .get(&options.node_id)
                .is_none_or(|peer| peer.identity != key.public_identity())
            || !options.store_path.is_absolute()
            || !options.ipc_directory.is_absolute()
        {
            return Err(invalid());
        }
        #[cfg(not(unix))]
        {
            let _ = pool;
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native execution requires a certified local authority listener on this platform",
            ));
        }
        #[cfg(unix)]
        {
            let channel = Arc::new(WebRtcControlChannel::new(
                pool,
                options.peers.values().map(|p| p.identity.clone()).collect(),
                Duration::from_secs(3),
            )?);
            for (&id, peer) in &options.peers {
                if id != options.node_id {
                    channel.set_route(&peer.identity, options.routes[&id].clone())?;
                }
            }
            let transport = Arc::new(SignedTransport::new(
                key.clone(),
                options.scope_id.clone(),
                channel,
            ));
            let node = Arc::new(
                AuthorityNode::open(
                    options.node_id,
                    options.scope_id.clone(),
                    options.peers.clone(),
                    &options.store_path,
                    transport,
                    options.timing.raft_config()?,
                )
                .await?,
            );
            let mut group = Self {
                node,
                endpoint: PathBuf::new(),
                stop: Mutex::new(None),
                task: Mutex::new(None),
            };
            // On every subsequent error Drop stops the already opened Raft node.
            register_receiver(pool, key, options.scope_id, &group.node)?;
            if options.peers.keys().next() == Some(&options.node_id) {
                group.node.bootstrap().await?;
            }
            let routes: BTreeSet<_> = options
                .routes
                .into_iter()
                .filter_map(|(id, route)| (id != options.node_id).then_some(route))
                .collect();
            group
                .activate(
                    pool,
                    signaling,
                    options.ipc_directory,
                    routes,
                    PeerInitiation::VoterPair,
                )
                .await?;
            Ok(group)
        }
    }
}

impl<A: ExecutionAuthority + 'static> NativeExecutionHost<A> {
    #[cfg(unix)]
    async fn activate(
        &mut self,
        pool: &NativePool,
        signaling: &Arc<SignalingClient>,
        ipc_directory: PathBuf,
        routes: BTreeSet<String>,
        initiation: PeerInitiation,
    ) -> io::Result<()> {
        let group = self;
        let mut host = LocalAuthorityHost::start(ipc_directory, group.node.clone()).await?;
        group.endpoint = host.endpoint().to_path_buf();
        let (stop, mut stopped) = oneshot::channel();
        *group.stop.get_mut() = Some(stop);
        let handler = pool.connection_handler.clone();
        let signaling = signaling.clone();
        let node = group.node.clone();
        let mut peer_lists = signaling.peer_list_stream();
        let mut opened = handler.connect_stream();
        let mut disconnected = handler.disconnect_stream();
        let discover = async move {
            let mut advertised = BTreeSet::new();
            let mut connected = std::collections::BTreeMap::new();
            let mut retry = false;
            loop {
                tokio::select! {
                    list = peer_lists.next() => {
                        let Some(list) = list else { break; };
                        advertised = list.into_iter().collect();
                        connected.retain(|peer, _| advertised.contains(peer));
                    },
                    peer = opened.next() => {
                        let Some(peer) = peer else { break; };
                        if handler.connection_for_peer(peer.peer_id()).as_ref() == Some(&peer) {
                            connected.insert(peer.peer_id().to_owned(), peer);
                        }
                    },
                    peer = disconnected.next() => {
                        let Some(peer) = peer else { break; };
                        if connected.get(peer.peer_id()) == Some(&peer) {
                            connected.remove(peer.peer_id());
                        }
                    },
                    _ = tokio::time::sleep(Duration::from_secs(1)), if retry => {},
                }
                // Signaling addresses are ephemeral. The attached signing key,
                // member ID and scope remain pinned, and SignedTransport verifies
                // every exchange. Rejoining must not destroy that local authority.
                // During disconnect there is no address from which to negotiate.
                let Some(own_route) = signaling.own_peer_id() else {
                    continue;
                };
                retry = false;
                for route in &routes {
                    if (matches!(initiation, PeerInitiation::WorkerToVoter) || route > &own_route)
                        && advertised.contains(route)
                        && !connected.contains_key(route)
                        && signaling
                            .peer_role(route)
                            .as_deref()
                            .and_then(crate::native::NativePeerRole::from_wire)
                            .is_some()
                    {
                        retry = true;
                        let _ = tokio::time::timeout(Duration::from_secs(3), async {
                            match initiation {
                                PeerInitiation::VoterPair => {
                                    handler.connect_native_execution_peer(route.clone()).await
                                }
                                PeerInitiation::WorkerToVoter => {
                                    handler
                                        .connect_worker_to_authority_peer(route.clone())
                                        .await
                                }
                            }
                        })
                        .await;
                    }
                }
            }
            Err::<(), io::Error>(io::Error::new(
                io::ErrorKind::NotConnected,
                "native execution discovery ended",
            ))
        };
        *group.task.get_mut() = Some(tokio::spawn(async move {
            let outcome = std::panic::AssertUnwindSafe(async {
                tokio::select! {
                    biased;
                    _ = &mut stopped => Ok(()),
                    result = host.wait_stopped() => match result {
                        Ok(()) => Err(io::Error::new(io::ErrorKind::BrokenPipe,
                            "local execution authority listener stopped unexpectedly")),
                        Err(error) => Err(error),
                    },
                    result = discover => result,
                }
            })
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(io::Error::other("native execution supervisor panicked")));
            // A listener or discovery failure revokes authority before resources
            // are released. No surviving IPC/node handle can keep execution alive.
            let authority = node.shutdown().await;
            let listener = host.shutdown().await;
            outcome.and(authority).and(listener)
        }));
        Ok(())
    }

    pub fn node(&self) -> &Arc<A> {
        &self.node
    }

    pub fn ipc_endpoint(&self) -> &Path {
        &self.endpoint
    }

    /// Hosts await this to turn unexpected listener/discovery exit into a visible
    /// runtime failure. Cancelling the wait does not cancel the owned lifecycle.
    pub async fn wait_stopped(&self) -> io::Result<()> {
        let mut task = self.task.lock().await;
        let result = match task.as_mut() {
            Some(task) => task
                .await
                .map_err(io::Error::other)
                .and_then(|result| result),
            None => return Ok(()),
        };
        task.take();
        if result.is_err() {
            let _ = self.node.shutdown().await;
        }
        result
    }

    pub async fn shutdown(&self) -> io::Result<()> {
        if let Some(stop) = self.stop.lock().await.take() {
            let _ = stop.send(());
        }
        self.wait_stopped().await
    }
}
impl NativeExecutionHost<crate::authority::client::WorkerAuthorityClient> {
    pub(crate) async fn attach_worker(
        pool: &NativePool,
        signaling: &Arc<SignalingClient>,
        room: &str,
        options: WorkerExecutionOptions,
        key: Arc<SigningIdentity>,
    ) -> io::Result<Self> {
        #[cfg(not(unix))]
        {
            let _ = (pool, signaling, room, options, key);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native worker requires a certified local authority listener on this platform",
            ))
        }
        #[cfg(unix)]
        {
            let own_route = signaling
                .own_peer_id()
                .ok_or_else(|| io::Error::other("worker has no signaling identity"))?;
            if options.room != room
                || pool.connection_handler.local_peer_role()
                    != crate::native::NativePeerRole::WorkjetExecutor
                || options.routes.keys().ne(options.voters.keys())
                || options.routes.values().collect::<BTreeSet<_>>().len() != 3
                || options.routes.values().any(|route| {
                    route.is_empty()
                        || route.trim() != route
                        || route.len() > 128
                        || route == &own_route
                })
                || !options.ipc_directory.is_absolute()
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "worker does not match its confirmed native session and voter routes",
                ));
            }
            let channel = Arc::new(WebRtcControlChannel::new(
                pool,
                options
                    .voters
                    .values()
                    .map(|peer| peer.identity.clone())
                    .collect(),
                Duration::from_secs(3),
            )?);
            for (&id, peer) in &options.voters {
                channel.set_route(&peer.identity, options.routes[&id].clone())?;
            }
            let node = Arc::new(crate::authority::client::WorkerAuthorityClient::new(
                options.member,
                options.scope_id,
                options.voters,
                key,
                channel,
            )?);
            let mut worker = Self {
                node,
                endpoint: PathBuf::new(),
                stop: Mutex::new(None),
                task: Mutex::new(None),
            };
            worker
                .activate(
                    pool,
                    signaling,
                    options.ipc_directory,
                    options.routes.into_values().collect(),
                    PeerInitiation::WorkerToVoter,
                )
                .await?;
            Ok(worker)
        }
    }
}

impl<A: ExecutionAuthority + 'static> Drop for NativeExecutionHost<A> {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.get_mut().take() {
            let _ = stop.send(());
        }
        if let Some(task) = self.task.get_mut().take() {
            task.abort();
        }
        let node = self.node.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = node.shutdown().await;
            });
        }
    }
}
