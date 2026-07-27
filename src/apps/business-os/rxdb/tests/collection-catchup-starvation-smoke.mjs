import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'catchup-starvation-test',
  signalingUrl: 'wss://signaling.invalid/',
  room: 'catchup-starvation-room',
  iceServers: [],
  expectedNativePeerId: 'native-test',
});

shared.collectionCatchUpQueueSliceMs = 20;
const order = [];
let releaseFirst;
const firstBlocked = new Promise((resolve) => { releaseFirst = resolve; });
shared.catchUpRegisteredCollection = async (collection) => {
  order.push(`${collection}:start`);
  if (collection === 'first') await firstBlocked;
  order.push(`${collection}:done`);
};

const registration = { state: { emitError(error) { throw error; } } };
shared.scheduleCollectionCatchUp('first', registration);
shared.scheduleCollectionCatchUp('second', registration);

await new Promise((resolve) => setTimeout(resolve, 80));
if (!order.includes('second:start')) {
  throw new Error(`second collection was starved behind first: ${order.join(', ')}`);
}
if (order.includes('first:done')) {
  throw new Error('first collection unexpectedly completed before release');
}
if (!shared.collectionCatchUps.has('first')) {
  throw new Error('running first catch-up was removed from deduplication too early');
}

releaseFirst();
await new Promise((resolve) => setTimeout(resolve, 20));
if (!order.includes('first:done') || !order.includes('second:done')) {
  throw new Error(`catch-up completion missing: ${order.join(', ')}`);
}

console.log('collection catch-up starvation smoke passed');
