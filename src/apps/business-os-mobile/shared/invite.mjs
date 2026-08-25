import {
  DATA_PLANE,
  INVITE_TYPE,
  INVITE_VERSION,
  MOBILE_HOST,
  MOBILE_SCHEME,
  ROOM_PREFIX,
  TRANSPORT,
} from "./constants.mjs";

export class InviteValidationError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "InviteValidationError";
    this.code = code;
  }
}

function fail(code, message) {
  throw new InviteValidationError(code, message);
}

function requiredString(value, code, label) {
  if (typeof value !== "string" || !value.trim()) fail(code, `${label} is required`);
  return value.trim();
}

function parseTime(value, code, label) {
  const text = requiredString(value, code, label);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/.test(text)) {
    fail(code, `${label} must be RFC3339`);
  }
  const milliseconds = Date.parse(text);
  if (!Number.isFinite(milliseconds)) fail(code, `${label} must be RFC3339`);
  return milliseconds;
}

function validateSignalingUrl(raw) {
  const text = requiredString(raw, "signaling_url", "signaling URL");
  let parsed;
  try { parsed = new URL(text); } catch { fail("signaling_url", "signaling URL is invalid"); }
  if (parsed.username || parsed.password || parsed.hash) fail("signaling_url", "signaling URL contains unsupported credentials or fragment");
  if (parsed.protocol !== "wss:" || !parsed.hostname) fail("signaling_url", "signaling URLs must use wss");
  return parsed.toString();
}

function decodePayload(encoded) {
  if (!/^[A-Za-z0-9_-]+$/.test(encoded) || encoded.length > 262144) fail("payload", "pairing payload is invalid");
  let decoded;
  try { decoded = Buffer.from(encoded, "base64url").toString("utf8"); } catch { fail("payload", "pairing payload is invalid"); }
  try { return JSON.parse(decoded); } catch { fail("json", "pairing payload is not valid JSON"); }
}

export function parseMobilePairLink(raw, options = {}) {
  const input = String(raw || "").trim();
  if (!input) fail("empty", "pairing link is empty");
  let url;
  try { url = new URL(input); } catch { fail("url", "pairing link is invalid"); }
  if (url.protocol !== `${MOBILE_SCHEME}:`) fail("scheme", "unsupported pairing scheme");
  if (url.hostname !== MOBILE_HOST || (url.pathname && url.pathname !== "/")) fail("host", "unsupported pairing action");
  if (url.username || url.password || url.hash) fail("url", "pairing link contains unsupported components");
  const keys = [...url.searchParams.keys()];
  if (keys.length !== 1 || keys[0] !== "payload") fail("query", "pairing link must contain only payload");
  const encoded = url.searchParams.get("payload");
  if (!encoded) fail("payload", "pairing link is missing payload");
  return validateInviteV1(decodePayload(encoded), options);
}

export function validateInviteV1(input, { now = Date.now() } = {}) {
  if (!input || typeof input !== "object" || Array.isArray(input)) fail("object", "invite must be an object");
  if (input.type !== INVITE_TYPE) fail("type", "unsupported invite type");
  if (typeof input.version !== "number" || !Number.isInteger(input.version) || input.version !== INVITE_VERSION) fail("version", "unsupported invite version");
  const displayName = requiredString(input.display_name, "display_name", "display_name");
  const instanceId = requiredString(input.instance_id, "instance_id", "instance_id");
  const syncRoom = requiredString(input.sync_room, "sync_room", "sync_room");
  if (!syncRoom.startsWith(ROOM_PREFIX) || syncRoom.length <= ROOM_PREFIX.length) fail("sync_room", "sync_room must identify a CTOX Business OS room");
  const nativePeerId = requiredString(input.native_peer_id, "native_peer_id", "native_peer_id");
  if (!Array.isArray(input.signaling_urls) || input.signaling_urls.length === 0) fail("signaling_urls", "signaling_urls are required");
  const signalingUrls = input.signaling_urls.map(validateSignalingUrl);
  const password = requiredString(input.signaling_room_password, "password", "signaling_room_password");
  if (input.transport !== TRANSPORT) fail("transport", "invite transport must be webrtc");
  const expiresAtMs = parseTime(input.expires_at, "expires_at", "expires_at");
  if (expiresAtMs <= Number(now)) fail("expired", "invite is expired");
  if (input.data_plane !== DATA_PLANE) fail("data_plane", "invite data_plane must be rxdb-webrtc");
  if (input.http_bridge_available !== false) fail("http_bridge", "HTTP bridge must be disabled");
  const session = input.session;
  if (!session || typeof session !== "object" || Array.isArray(session) || session.authenticated !== true) fail("session", "authenticated session is required");
  const capabilityToken = requiredString(session.capability_token, "capability_token", "session capability_token");
  if (!Number.isSafeInteger(session.capability_expires_at_ms) || session.capability_expires_at_ms <= Number(now)) fail("capability_expired", "session capability is expired or invalid");
  if (session.capability_expires_at_ms > expiresAtMs) fail("capability_expiry", "session capability outlives the invite");
  const user = session.user;
  if (!user || typeof user !== "object" || Array.isArray(user)) fail("user", "session user is required");
  const userId = requiredString(user.id, "user_id", "session user id");
  const userDisplayName = requiredString(user.display_name, "user_display_name", "session user display_name");
  const role = requiredString(user.role, "user_role", "session user role");
  if (!new Set(["chef", "admin", "founder", "user"]).has(role)) fail("user_role", "session user role is unsupported");
  return Object.freeze({
    type: INVITE_TYPE,
    version: INVITE_VERSION,
    displayName,
    instanceId,
    syncRoom,
    nativePeerId,
    signalingUrls: Object.freeze(signalingUrls),
    password,
    expiresAt: new Date(expiresAtMs).toISOString(),
    expiresAtMs,
    session: Object.freeze({
      authenticated: true,
      source: typeof session.source === "string" ? session.source : "desktop_invite",
      capabilityToken,
      capabilityExpiresAtMs: session.capability_expires_at_ms,
      user: Object.freeze({ id: userId, displayName: userDisplayName, role, isAdmin: ["chef", "admin", "founder"].includes(role) }),
    }),
  });
}

export function encodeMobilePairLink(invite, options = {}) {
  const validated = validateInviteV1(invite, options);
  const copy = JSON.parse(JSON.stringify(invite));
  delete copy.desktop_link;
  const encoded = Buffer.from(JSON.stringify(copy), "utf8").toString("base64url");
  return `${MOBILE_SCHEME}://${MOBILE_HOST}?payload=${encoded}`;
}

export function confirmationMetadata(invite) {
  const normalized = invite.displayName ? invite : validateInviteV1(invite);
  return {
    displayName: normalized.displayName,
    expiresAt: normalized.expiresAt,
    signalingHosts: normalized.signalingUrls.map((value) => new URL(value).host),
  };
}
