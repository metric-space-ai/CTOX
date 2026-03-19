import { rxdbWebRTC } from "./craft-sync-rxdb-shim.js";
import { REMOTE_COMPUTE_TOPIC } from "./remote_compute.mjs";

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function normalizeSignalingUrl(raw) {
  const value = asText(raw);
  if (!value) throw new Error("Signaling URL is empty.");
  const canonicalValue = /^ss:\/\//i.test(value) ? `wss://${value.slice(5)}` : value;
  if (!/^wss:\/\//i.test(canonicalValue)) {
    throw new Error("Signaling URL must start with ss:// or wss://");
  }
  const url = new URL(canonicalValue);
  url.search = "";
  url.hash = "";
  return url.toString();
}

function withToken(url, token) {
  const trimmedToken = asText(token);
  if (!trimmedToken) return url;
  const parsed = new URL(url);
  if (!parsed.searchParams.has("token")) {
    parsed.searchParams.set("token", trimmedToken);
  }
  return parsed.toString();
}

async function waitForDrain(channel, {
  maxBuffered = 2_000_000,
  lowThreshold = 512_000,
  timeoutMs = 15_000,
  isAbort = null,
} = {}) {
  if (!channel || typeof channel.bufferedAmount !== "number") return;
  if (channel.bufferedAmount <= maxBuffered) return;

  try {
    channel.bufferedAmountLowThreshold = Math.max(0, Math.min(lowThreshold, maxBuffered));
  } catch (_error) {}

  await new Promise((resolve, reject) => {
    let settled = false;
    let abortTimer = 0;
    let timeoutId = 0;

    const cleanup = () => {
      if (settled) return;
      settled = true;
      try {
        channel.removeEventListener("bufferedamountlow", onLow);
      } catch (_error) {}
      globalThis.clearInterval(abortTimer);
      globalThis.clearTimeout(timeoutId);
    };

    const onLow = () => {
      if (channel.bufferedAmount <= maxBuffered) {
        cleanup();
        resolve();
      }
    };

    if (typeof isAbort === "function") {
      abortTimer = globalThis.setInterval(() => {
        try {
          if (isAbort()) {
            cleanup();
            reject(new Error("datachannel-abort"));
          }
        } catch (_error) {}
      }, 100);
    }

    timeoutId = globalThis.setTimeout(() => {
      cleanup();
      reject(new Error("datachannel-backpressure-timeout"));
    }, timeoutMs);

    try {
      channel.addEventListener("bufferedamountlow", onLow);
    } catch (_error) {}
    onLow();
  });
}

function wrapSendGuard(handler) {
  if (!handler || typeof handler.send !== "function") return handler;
  const originalSend = handler.send.bind(handler);
  const encoder = new TextEncoder();

  handler.send = async (peer, message) => {
    if (!peer) throw new Error("peer-missing");
    if (peer.destroyed) throw new Error("peer-destroyed");

    const channel = peer._channel;
    const connection = peer._pc;
    if (channel?.readyState && channel.readyState !== "open") {
      throw new Error("datachannel-not-open");
    }

    const serializedMessage = typeof message === "string" ? JSON.stringify(message) : JSON.stringify(message);
    const bytes = encoder.encode(serializedMessage).byteLength;
    const maxMessageSize = connection?.sctp?.maxMessageSize;
    if (typeof maxMessageSize === "number" && maxMessageSize > 0 && bytes > maxMessageSize) {
      throw new Error(`msg-too-large bytes=${bytes} max=${maxMessageSize}`);
    }

    if (channel && typeof channel.bufferedAmount === "number") {
      await waitForDrain(channel, {
        isAbort: () => Boolean(peer.destroyed),
      });
    }
    return await originalSend(peer, message);
  };

  return handler;
}

function createSafeWebSocketClass() {
  const NativeWebSocket = globalThis.WebSocket;
  return class SafeWebSocket {
    constructor(url) {
      this._ws = new NativeWebSocket(url);
      this._ws.onopen = (event) => {
        try {
          this.onopen?.(event);
        } catch (_error) {}
      };
      this._ws.onclose = (event) => {
        try {
          this.onclose?.(event);
        } catch (_error) {}
      };
      this._ws.onerror = (event) => {
        try {
          this.onerror?.(event);
        } catch (_error) {}
      };
      this._ws.onmessage = (event) => {
        try {
          this.onmessage?.(event);
        } catch (_error) {
          try {
            this._ws.close();
          } catch (_closeError) {}
        }
      };
    }

    get readyState() {
      return this._ws.readyState;
    }

    send(data) {
      if (this._ws.readyState !== NativeWebSocket.OPEN) return;
      this._ws.send(data);
    }

    close() {
      try {
        this._ws.close();
      } catch (_error) {}
    }
  };
}

export async function createPeerComputeBus({
  settings = null,
  topic = REMOTE_COMPUTE_TOPIC,
  onConnect = null,
  onDisconnect = null,
  onMessage = null,
  onError = null,
} = {}) {
  const urls = asText(settings?.signalingUrls)
    .split(",")
    .map((entry) => asText(entry))
    .filter(Boolean)
    .map((entry) => normalizeSignalingUrl(entry));
  const signalingUrl = withToken(urls[0] || "", settings?.token);
  if (!signalingUrl) {
    throw new Error("Peer compute bus requires a signaling URL.");
  }

  const { getConnectionHandlerSimplePeer } = rxdbWebRTC();
  const handlerCreator = getConnectionHandlerSimplePeer({
    signalingServerUrl: signalingUrl,
    config: {
      iceServers: [
        {
          urls: ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"],
        },
      ],
    },
    webSocketConstructor: createSafeWebSocketClass(),
    debug: false,
  });
  const handler = wrapSendGuard(await handlerCreator({ topic }));
  const peers = new Set();
  const subscriptions = [];

  const subscribe = (stream, callback) => {
    const sub = stream?.subscribe?.(callback);
    if (sub) subscriptions.push(sub);
  };

  subscribe(handler.connect$, (peer) => {
    peers.add(peer);
    onConnect?.(peer);
  });
  subscribe(handler.disconnect$, (peer) => {
    peers.delete(peer);
    onDisconnect?.(peer);
  });
  subscribe(handler.message$, (event) => {
    onMessage?.(event);
  });
  subscribe(handler.response$, (event) => {
    onMessage?.(event);
  });
  subscribe(handler.error$, (error) => {
    onError?.(error);
  });

  return {
    topic,
    handler,
    listPeers() {
      return [...peers];
    },
    async send(peer, message) {
      await handler.send(peer, message);
    },
    async close() {
      while (subscriptions.length) {
        try {
          subscriptions.pop()?.unsubscribe?.();
        } catch (_error) {}
      }
      peers.clear();
      await handler.close();
    },
  };
}
