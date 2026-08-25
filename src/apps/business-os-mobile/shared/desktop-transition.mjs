import { validateInviteV1 } from "./invite.mjs";

export function parseDesktopInviteInput(raw, options = {}) {
  const input = String(raw || "").trim();
  if (!input) throw new Error("invite input is empty");
  let value;
  if (input.startsWith("ctox-business-os-desktop://")) {
    let url;
    try { url = new URL(input); } catch { throw new Error("desktop invite is invalid"); }
    if (url.protocol !== "ctox-business-os-desktop:" || url.hostname !== "pair" || (url.pathname && url.pathname !== "/")) {
      throw new Error("desktop invite action is unsupported");
    }
    const payload = url.searchParams.get("payload");
    if (!payload || [...url.searchParams.keys()].some((key) => key !== "payload")) throw new Error("desktop invite payload is invalid");
    try { value = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")); } catch { throw new Error("desktop invite payload is invalid"); }
  } else {
    try { value = JSON.parse(input); } catch { throw new Error("desktop invite JSON is invalid"); }
  }
  validateInviteV1(value, options);
  const copy = JSON.parse(JSON.stringify(value));
  delete copy.desktop_link;
  return copy;
}
