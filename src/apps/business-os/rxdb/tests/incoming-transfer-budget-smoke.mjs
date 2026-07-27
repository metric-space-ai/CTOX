import {
  CtoxWebRtcNativePeer,
  webrtcNativeTestInternals,
} from '../src/webrtc-native.mjs';

const {
  MAX_INCOMING_FRAME_TRANSFERS,
  MAX_INCOMING_FRAME_BUFFERED_BYTES,
} = webrtcNativeTestInternals;

function newPeer(room) {
  return new CtoxWebRtcNativePeer({
    signalingUrl: 'ws://localhost:0/ignored',
    room,
  });
}

function startFrame(transferId, totalBytes, attempt = 0) {
  return {
    kind: 'start',
    transferId,
    totalFrames: 2,
    totalBytes,
    attempt,
  };
}

// The transfer-count cap admits exactly eight independent reservations and
// rejects a ninth without disturbing any admitted transfer.
{
  const peer = newPeer('incoming-transfer-count');
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));

  for (let index = 0; index < MAX_INCOMING_FRAME_TRANSFERS; index += 1) {
    await peer.handleTransportFrame('native-count', startFrame(`count-${index}`, 1));
  }
  assert(
    peer.incomingFrames.size === MAX_INCOMING_FRAME_TRANSFERS,
    `${MAX_INCOMING_FRAME_TRANSFERS} concurrent transfers are admitted`,
  );

  await peer.handleTransportFrame('native-count', startFrame('count-overflow', 1));
  assert(
    peer.incomingFrames.size === MAX_INCOMING_FRAME_TRANSFERS,
    'the ninth transfer is rejected without evicting the first eight',
  );
  for (let index = 0; index < MAX_INCOMING_FRAME_TRANSFERS; index += 1) {
    assert(peer.incomingFrames.has(`count-${index}`), `admitted transfer count-${index} remains intact`);
  }
  assert(!peer.incomingFrames.has('count-overflow'), 'the ninth transfer receives no reservation');
  assert(
    errors.some((error) => error?.code === 'ctox_webrtc_incoming_transfer_budget_exceeded'),
    'transfer-count rejection emits ctox_webrtc_incoming_transfer_budget_exceeded',
  );
  peer.close();
}

// Four full-size declarations consume the aggregate byte reservation. A fifth
// one-byte declaration is rejected even though the transfer-count cap has room,
// and no chunk memory has been allocated.
{
  const peer = newPeer('incoming-byte-budget');
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));
  const perTransferBytes = MAX_INCOMING_FRAME_BUFFERED_BYTES / 4;

  for (let index = 0; index < 4; index += 1) {
    await peer.handleTransportFrame('native-bytes', startFrame(`bytes-${index}`, perTransferBytes));
  }
  assert(
    peer.incomingFrameReservedBytes() === MAX_INCOMING_FRAME_BUFFERED_BYTES,
    `four declarations reserve the full aggregate byte budget (${MAX_INCOMING_FRAME_BUFFERED_BYTES})`,
  );
  assert(peer.incomingFrameBufferedBytes() === 0, 'start-frame admission reserves bytes without buffering chunks');

  await peer.handleTransportFrame('native-bytes', startFrame('bytes-overflow', 1));
  assert(peer.incomingFrames.size === 4, 'byte overflow is rejected before a fifth transfer is stored');
  assert(!peer.incomingFrames.has('bytes-overflow'), 'byte-overflow transfer is not admitted');
  assert(peer.incomingFrameBufferedBytes() === 0, 'byte-budget rejection occurs before memory grows');
  assert(
    errors.some((error) => (
      error?.code === 'ctox_webrtc_incoming_transfer_budget_exceeded'
      && error.transferId === 'bytes-overflow'
      && error.reservedBytes === MAX_INCOMING_FRAME_BUFFERED_BYTES
    )),
    'aggregate-byte rejection reports the reserved budget',
  );
  peer.close();
}

// A retry start replaces its own reservation rather than counting the same
// transfer twice. The attempt metadata is refreshed with the replacement.
{
  const peer = newPeer('incoming-retry');
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));
  const retryBytes = Math.floor(MAX_INCOMING_FRAME_BUFFERED_BYTES / 8);

  await peer.handleTransportFrame('native-retry', startFrame('same-transfer', retryBytes, 0));
  const reservedBeforeRetry = peer.incomingFrameReservedBytes();
  await peer.handleTransportFrame('native-retry', startFrame('same-transfer', retryBytes, 1));

  assert(peer.incomingFrames.size === 1, 'a repeated start keeps one transfer entry');
  assert(
    peer.incomingFrameReservedBytes() === reservedBeforeRetry,
    'a repeated start with the same transferId does not double-count reserved bytes',
  );
  assert(peer.incomingFrames.get('same-transfer')?.attempt === 1, 'the retry replaces the prior attempt metadata');
  assert(
    !errors.some((error) => error?.code === 'ctox_webrtc_incoming_transfer_budget_exceeded'),
    'same-transfer retry is not rejected by the aggregate budget',
  );
  peer.close();
}

// A chunk that exceeds its transfer's declaration discards the transfer and
// releases the entire reservation immediately.
{
  const peer = newPeer('incoming-declared-size');
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));

  await peer.handleTransportFrame('native-size', startFrame('declared-three', 3));
  assert(peer.incomingFrameReservedBytes() === 3, 'the declared size is reserved on start');
  await peer.handleTransportFrame('native-size', {
    kind: 'chunk',
    transferId: 'declared-three',
    attempt: 0,
    seq: 0,
    data: 'four',
  });

  assert(!peer.incomingFrames.has('declared-three'), 'oversized buffered data discards the transfer');
  assert(peer.incomingFrameReservedBytes() === 0, 'discarding the transfer releases its reservation');
  assert(peer.incomingFrameBufferedBytes() === 0, 'discarding the transfer releases buffered-byte accounting');
  assert(
    errors.some((error) => (
      error?.code === 'ctox_webrtc_frame_buffer_exceeds_declared_size'
      && error.transferId === 'declared-three'
      && error.bufferedBytes === 4
      && error.declaredBytes === 3
    )),
    'declared-size overflow emits ctox_webrtc_frame_buffer_exceeds_declared_size',
  );
  peer.close();
}

// A successful joined broadcast proves an older socket's control-plane error is
// stale and clears it.
{
  const peer = newPeer('joined-clears-control-plane-error');
  peer.on('error', () => {});
  peer.handleSignalingMessage(JSON.stringify({
    type: 'ctoxError',
    scope: 'control-plane',
    code: 'control_plane_token_expired',
    reason: 'synthetic stale rejection',
  }));
  assert(
    peer.lastControlPlaneError?.code === 'control_plane_token_expired',
    'the retryable control-plane rejection is retained before rejoin',
  );

  peer.handleSignalingMessage(JSON.stringify({
    type: 'joined',
    yourPeerId: peer.options.clientId,
    peers: [],
  }));
  assert(peer.lastControlPlaneError === null, 'joined clears the stale lastControlPlaneError');
}

console.log('ctox-rxdb-js incoming transfer budget smoke OK');

function assert(c, m) { if (!c) throw new Error(m); }
