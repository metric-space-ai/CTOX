use futures::{SinkExt, StreamExt};
#[cfg(test)]
use rxdb::plugins::replication_webrtc::WebRTCRsConnectionHandler;
use serde_json::{json, Value};
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle};
use tokio_tungstenite::{accept_async, tungstenite::Message};

type Members = Arc<Mutex<BTreeMap<String, mpsc::UnboundedSender<Message>>>>;

#[cfg(test)]
pub(crate) fn route_ready(
    pool: &rxdb::plugins::replication_webrtc::RxWebRTCReplicationPool<WebRTCRsConnectionHandler>,
    route: &str,
) -> bool {
    pool.connection_handler
        .connection_for_peer(route)
        .is_some_and(|connection| pool.is_peer_ready_for_control(&connection))
}

pub(crate) struct SignalingFixture {
    pub(crate) url: String,
    task: JoinHandle<()>,
    members: Members,
    joins: tokio::sync::broadcast::Sender<String>,
    pub(crate) offers: Arc<Mutex<BTreeSet<(String, String)>>>,
}
impl Drop for SignalingFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl SignalingFixture {
    #[cfg(test)]
    pub(crate) async fn start() -> Self {
        Self::with_roles(["ctox_instance"; 3]).await
    }

    pub(crate) async fn with_roles<const N: usize>(roles: [&'static str; N]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let members: Members = Arc::default();
        let retained_members = members.clone();
        let (joins, _) = tokio::sync::broadcast::channel(16);
        let retained_joins = joins.clone();
        let offers: Arc<Mutex<BTreeSet<(String, String)>>> = Arc::default();
        let retained_offers = offers.clone();
        let task = tokio::spawn(async move {
            let roles: Arc<BTreeMap<_, _>> = Arc::new(
                roles
                    .into_iter()
                    .enumerate()
                    .map(|(index, role)| (format!("native{:06}", index + 1), role))
                    .collect(),
            );
            let mut next_id = 0;
            let mut connections = tokio::task::JoinSet::new();
            loop {
                let (tcp, _) = listener.accept().await.unwrap();
                next_id += 1;
                let id = format!("native{next_id:06}");
                let members = members.clone();
                let roles = roles.clone();
                let joins = joins.clone();
                let offers = offers.clone();
                connections.spawn(async move {
                    let socket=accept_async(tcp).await.unwrap();let (mut writer,mut reader)=socket.split();
                    let (tx,mut rx)=mpsc::unbounded_channel();
                    tx.send(Message::text(json!({"type":"init","yourPeerId":id}).to_string())).unwrap();
                    loop {
                        tokio::select! {
                            output=rx.recv()=>{match output {Some(output)=>{let closing=matches!(output,Message::Close(_));if writer.send(output).await.is_err() || closing {break;}},None=>break}},
                            input=reader.next()=>{
                                let Some(Ok(Message::Text(input)))=input else {break;};
                                let value:Value=serde_json::from_str(&input).unwrap();
                                match value["type"].as_str() {
                                    Some("join")=>{
                                        let mut all=members.lock().unwrap();all.insert(id.clone(),tx.clone());
                                        let ids:Vec<_>=all.keys().cloned().collect();
                                        let peers:Vec<_>=ids.iter().map(|id|json!({"peerId":id,"role":roles.get(id).copied().unwrap_or("browser")})).collect();
                                        let joined=Message::text(json!({"type":"joined","otherPeerIds":ids,"peers":peers}).to_string());
                                        for target in all.values(){let _=target.send(joined.clone());}
                                        let _=joins.send(id.clone());
                                    },
                                    Some("signal")=>{
                                        assert_eq!(value["senderPeerId"],id);
                                        if value["data"]["type"] == "offer" {
                                            offers.lock().unwrap().insert((id.clone(), value["receiverPeerId"].as_str().unwrap().into()));
                                        }
                                        if let Some(target)=members.lock().unwrap().get(value["receiverPeerId"].as_str().unwrap()) {
                                            let _=target.send(Message::text(value.to_string()));
                                        }
                                    },
                                    Some("ping")=>{},
                                    other=>panic!("unexpected signaling request {other:?}"),
                                }
                            }
                        }
                    }
                    let mut all=members.lock().unwrap();
                    all.remove(&id);
                    let ids:Vec<_>=all.keys().cloned().collect();
                    let peers:Vec<_>=ids.iter().map(|id|json!({"peerId":id,"role":roles.get(id).copied().unwrap_or("browser")})).collect();
                    let joined=Message::text(json!({"type":"joined","otherPeerIds":ids,"peers":peers}).to_string());
                    for target in all.values(){let _=target.send(joined.clone());}
                });
            }
        });
        Self {
            url,
            task,
            members: retained_members,
            joins: retained_joins,
            offers: retained_offers,
        }
    }

    pub(crate) async fn disconnect_and_wait_for_rejoin(&self, peer: &str) -> String {
        let mut joins = self.joins.subscribe();
        let sender = self
            .members
            .lock()
            .unwrap()
            .get(peer)
            .cloned()
            .expect("connected fixture peer");
        sender.send(Message::Close(None)).unwrap();
        tokio::time::timeout(Duration::from_secs(15), joins.recv())
            .await
            .expect("peer did not reconnect to signaling")
            .unwrap()
    }
}
