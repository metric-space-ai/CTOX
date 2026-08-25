import { ANDROID_ENTRY, ANDROID_ORIGIN } from "./constants.mjs";

function scriptSafeJson(value) {
  const lineSeparator = new RegExp(String.fromCharCode(0x2028), "g");
  const paragraphSeparator = new RegExp(String.fromCharCode(0x2029), "g");
  return JSON.stringify(value)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026")
    .replace(lineSeparator, "\\u2028")
    .replace(paragraphSeparator, "\\u2029");
}

export function buildLaunchContext(metadata, secrets, platform = "mobile") {
  if (!metadata || !secrets?.password || !secrets?.capabilityToken) throw new Error("launch secrets are unavailable");
  if (metadata.capabilityExpiresAtMs <= Date.now()) throw new Error("pairing authorization is expired");
  const session = {
    authenticated: true,
    source: `${platform}_invite`,
    capability_token: secrets.capabilityToken,
    capability_expires_at_ms: metadata.capabilityExpiresAtMs,
    user: {
      id: metadata.sessionUser.id,
      display_name: metadata.sessionUser.displayName,
      role: metadata.sessionUser.role,
      is_admin: ["chef", "admin", "founder"].includes(metadata.sessionUser.role),
    },
  };
  const config = {
    instance_id: metadata.instanceId,
    peer_id: `${platform}:${metadata.id}`,
    peer_role: "business_os_client",
    native_peer_id: metadata.nativePeerId,
    sync_room: metadata.syncRoom,
    signaling_urls: metadata.signalingUrls,
    signaling_room_password: secrets.password,
    transport: "webrtc",
    data_plane: "rxdb-webrtc",
    http_bridge_available: false,
    app_hosting: `${platform}_bundled_shell`,
    ctox_instance_required: true,
    session,
  };
  return { session, config, designTemplates: [] };
}

export function injectLaunchContext(html, context) {
  const marker = /<head(?:\s[^>]*)?>/i;
  const match = marker.exec(String(html));
  if (!match) throw new Error("shell index is missing head");
  const script = `<script data-ctox-mobile-bootstrap>window.CTOX_BUSINESS_OS_SESSION=${scriptSafeJson(context.session)};window.CTOX_BUSINESS_OS_CONFIG=${scriptSafeJson(context.config)};window.CTOX_BUSINESS_OS_DESIGN_TEMPLATES=[];</script>`;
  return `${html.slice(0, match.index + match[0].length)}${script}${html.slice(match.index + match[0].length)}`;
}

export function iosEntry(instanceId) {
  return `ctox-business-os-mobile://${encodeURIComponent(instanceId)}/business-os/index.html`;
}

export function navigationDecision(raw, { platform, instanceId } = {}) {
  let url;
  try { url = new URL(raw); } catch { return "deny"; }
  if (platform === "ios" && url.protocol === "ctox-business-os-mobile:" && url.hostname === instanceId && url.pathname.startsWith("/business-os/")) return "allow";
  if (platform === "android" && url.origin === ANDROID_ORIGIN && url.pathname.startsWith("/business-os/")) return "allow";
  if (url.protocol === "https:") return "external";
  return "deny";
}

export { ANDROID_ENTRY };
