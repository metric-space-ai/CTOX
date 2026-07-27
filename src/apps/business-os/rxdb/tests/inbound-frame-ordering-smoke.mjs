import { CtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

function newPeer(room = 'ordering-room') {
  return new CtoxWebRtcNativePeer({
    signalingUrl: 'ws://localhost:0/ignored',
    room,
  });
}

function newConnection(peer, remotePeerId = 'ctox-core-ordering') {
  const connection = {
    remotePeerId,
    channel: null,
    inboundFrameChain: Promise.resolve(),
    inboundFrameGeneration: 0,
    lastError: null,
  };
  peer.connections.set(remotePeerId, connection);
  return connection;
}

function newChannel(label) {
  return {
    label,
    readyState: 'open',
    close() { this.readyState = 'closed'; },
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((promiseResolve) => { resolve = promiseResolve; });
  return { promise, resolve };
}

// A genuinely asynchronous first frame must hold every later frame on the same
// DataChannel behind it. Before the per-connection Promise chain, onmessage
// launched both async handlers independently and "second" completed first.
{
  const peer = newPeer('ordering-serial');
  const connection = newConnection(peer);
  const channel = newChannel('serial');
  const firstStarted = deferred();
  const releaseFirst = deferred();
  const bothCompleted = deferred();
  const completed = [];

  peer.handleDataChannelFrame = async (_peerId, payload) => {
    if (payload.id === 'first') {
      firstStarted.resolve();
      await releaseFirst.promise;
    }
    completed.push(payload.id);
    if (completed.length === 2) bothCompleted.resolve();
  };

  peer.attachChannel(connection, channel);
  channel.onmessage({ data: JSON.stringify({ id: 'first' }) });
  await firstStarted.promise;
  channel.onmessage({ data: JSON.stringify({ id: 'second' }) });

  await Promise.resolve();
  assert(completed.length === 0, 'the second frame cannot complete while the first frame is delayed');

  releaseFirst.resolve();
  await bothCompleted.promise;
  assert(completed.join(',') === 'first,second', `frames complete in arrival order (got ${completed.join(',')})`);
}

// Replacing a channel starts a new generation immediately. Work queued behind
// an in-flight old-channel frame must be invalidated and must not mutate the
// connection after the new channel has been attached.
{
  const peer = newPeer('ordering-generation');
  const connection = newConnection(peer);
  const oldChannel = newChannel('old');
  const replacementChannel = newChannel('new');
  const blockerStarted = deferred();
  const releaseBlocker = deferred();
  const mutations = [];

  peer.handleDataChannelFrame = async (_peerId, payload) => {
    if (payload.id === 'old-blocker') {
      blockerStarted.resolve();
      await releaseBlocker.promise;
      throw new Error('stale old-generation failure');
    }
    mutations.push(payload.id);
  };

  peer.attachChannel(connection, oldChannel);
  oldChannel.onmessage({ data: JSON.stringify({ id: 'old-blocker' }) });
  await blockerStarted.promise;
  oldChannel.onmessage({ data: JSON.stringify({ id: 'old-stale-mutation' }) });
  const oldGenerationChain = connection.inboundFrameChain;

  peer.attachChannel(connection, replacementChannel);
  replacementChannel.onmessage({ data: JSON.stringify({ id: 'new-generation-mutation' }) });
  const newGenerationChain = connection.inboundFrameChain;
  await newGenerationChain;
  assert(
    mutations.join(',') === 'new-generation-mutation',
    `the new generation runs without stale old-channel mutation (got ${mutations.join(',')})`,
  );

  releaseBlocker.resolve();
  await oldGenerationChain;
  assert(
    mutations.join(',') === 'new-generation-mutation',
    'queued work from the old channel generation remains invalid after its blocker resolves',
  );
  assert(
    connection.lastError === null,
    'an in-flight failure from the old generation cannot overwrite connection.lastError',
  );
}

// A rejected frame handler is observed on the connection, but the queue stores
// a fulfilled catch-chain so the next frame still executes.
{
  const peer = newPeer('ordering-error-recovery');
  const connection = newConnection(peer);
  const channel = newChannel('error-recovery');
  const processed = [];
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));

  peer.handleDataChannelFrame = async (_peerId, payload) => {
    if (payload.id === 'throws') throw new Error('synthetic delayed frame failure');
    processed.push(payload.id);
  };

  peer.attachChannel(connection, channel);
  channel.onmessage({ data: JSON.stringify({ id: 'throws' }) });
  channel.onmessage({ data: JSON.stringify({ id: 'after-error' }) });
  await connection.inboundFrameChain;

  assert(processed.join(',') === 'after-error', 'a failed frame does not poison the following frame');
  assert(
    connection.lastError?.code === 'ctox_webrtc_inbound_frame_failed',
    `connection records the inbound-frame failure code (got ${connection.lastError?.code || 'none'})`,
  );
  assert(
    errors.some((error) => error?.code === 'ctox_webrtc_inbound_frame_failed'),
    'the inbound-frame failure is emitted as an observable error',
  );
}

console.log('ctox-rxdb-js inbound frame ordering smoke OK');

function assert(c, m) { if (!c) throw new Error(m); }
