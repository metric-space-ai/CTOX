import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { confirmationMetadata, encodeMobilePairLink, InviteValidationError, parseMobilePairLink, validateInviteV1 } from "../shared/invite.mjs";
import { buildLaunchContext, injectLaunchContext, iosEntry, navigationDecision } from "../shared/launch.mjs";
import { importFromPaste } from "../shared/import-flow.mjs";
import { forgetInstance, MemoryRegistry, MemorySecretStore, pairAtomically } from "../shared/registry-model.mjs";
import { buildPackManifest, verifyPack } from "../shared/office-pack.mjs";
import { officeRequestState, requireAndroidMultiProfile } from "../shared/platform-policy.mjs";

const corpus = JSON.parse(await fs.readFile(new URL("../fixtures/invites.json", import.meta.url), "utf8"));
const now = Date.parse(corpus.now);
function clone(value) { return JSON.parse(JSON.stringify(value)); }
function setPath(object, dotted, value) {
  const parts = dotted.split(".");
  const last = parts.pop();
  let cursor = object;
  for (const part of parts) cursor = cursor[part];
  cursor[last] = value;
}

test("valid invite imports from the reserved mobile link", () => {
  const link = encodeMobilePairLink(corpus.valid, { now });
  const invite = parseMobilePairLink(link, { now });
  assert.equal(invite.instanceId, corpus.valid.instance_id);
  assert.deepEqual(confirmationMetadata(invite), {
    displayName: corpus.valid.display_name,
    expiresAt: corpus.valid.expires_at,
    signalingHosts: ["signal.example.test"],
  });
});

test("reserved mobile link rejects scheme, host and query deviations", () => {
  const link = encodeMobilePairLink(corpus.valid, { now });
  for (const candidate of [
    link.replace("ctox-business-os-mobile:", "https:"),
    link.replace("//pair", "//open"),
    `${link}&extra=1`,
    `${link}#fragment`,
  ]) assert.throws(() => parseMobilePairLink(candidate, { now }), InviteValidationError);
});

for (const rejection of corpus.rejections) {
  test(`invite rejects ${rejection.name}`, () => {
    const candidate = clone(corpus.valid);
    setPath(candidate, rejection.path, rejection.value);
    assert.throws(() => validateInviteV1(candidate, { now }), (error) => error instanceof InviteValidationError && error.code === rejection.code);
  });
}

test("safe registry stores opaque refs and metadata but no secrets", async () => {
  const invite = validateInviteV1(corpus.valid, { now });
  const registry = new MemoryRegistry();
  const secrets = new MemorySecretStore();
  const instance = await pairAtomically(invite, { registry, secrets });
  const serialized = JSON.stringify(await registry.load());
  assert.match(instance.passwordRef, /^device-secret:/);
  assert.equal(serialized.includes(invite.password), false);
  assert.equal(serialized.includes(invite.session.capabilityToken), false);
  assert.equal(await secrets.get(instance.passwordRef), invite.password);
});

test("re-pair commits new secrets before registry swap then removes old refs", async () => {
  const invite = validateInviteV1(corpus.valid, { now });
  const registry = new MemoryRegistry();
  const secrets = new MemorySecretStore();
  const first = await pairAtomically(invite, { registry, secrets });
  const changed = { ...invite, password: `${invite.password}-rotated`, session: { ...invite.session, capabilityToken: `${invite.session.capabilityToken}-rotated` } };
  const second = await pairAtomically(changed, { registry, secrets });
  assert.equal(first.storageIdentity, second.storageIdentity);
  assert.equal(await secrets.get(first.passwordRef), "");
  assert.equal(await secrets.get(second.passwordRef), changed.password);
});

test("failed re-pair leaves previous registry and secrets intact", async () => {
  const invite = validateInviteV1(corpus.valid, { now });
  const registry = new MemoryRegistry();
  const initialSecrets = new MemorySecretStore();
  const first = await pairAtomically(invite, { registry, secrets: initialSecrets });
  const failing = new MemorySecretStore({ failAt: 2 });
  failing.values = new Map(initialSecrets.values);
  await assert.rejects(pairAtomically({ ...invite, password: "rotated" }, { registry, secrets: failing }));
  assert.equal((await registry.load()).instances[0].passwordRef, first.passwordRef);
  assert.equal(await failing.get(first.passwordRef), invite.password);
});

test("forget deletes only selected secrets and persistent storage identity", async () => {
  const firstInvite = validateInviteV1(corpus.valid, { now });
  const secondInvite = { ...firstInvite, instanceId: "biz_second", syncRoom: "ctox-business-os:biz_second", displayName: "Second" };
  const registry = new MemoryRegistry();
  const secrets = new MemorySecretStore();
  const first = await pairAtomically(firstInvite, { registry, secrets });
  const second = await pairAtomically(secondInvite, { registry, secrets });
  const removed = [];
  assert.equal(await forgetInstance(first.id, { registry, secrets, removeStorage: async (id) => removed.push(id) }), true);
  assert.deepEqual(removed, [first.storageIdentity]);
  assert.equal(await secrets.get(first.passwordRef), "");
  assert.notEqual(await secrets.get(second.passwordRef), "");
  assert.deepEqual((await registry.load()).instances.map((item) => item.id), [second.id]);
});

test("paste clears only after confirmed secure commit", async () => {
  const link = encodeMobilePairLink(corpus.valid, { now });
  let clears = 0;
  const success = await importFromPaste({ readPaste: async () => link, clearPaste: async () => { clears += 1; }, parse: (raw) => parseMobilePairLink(raw, { now }), confirm: async () => true, commit: async () => ({ id: "safe" }) });
  assert.equal(success.cleared, true);
  assert.equal(clears, 1);
  const declined = await importFromPaste({ readPaste: async () => link, clearPaste: async () => { clears += 1; }, parse: (raw) => parseMobilePairLink(raw, { now }), confirm: async () => false, commit: async () => { throw new Error("must not run"); } });
  assert.equal(declined.cleared, false);
  assert.equal(clears, 1);
});

test("launch injection precedes every shell script and keeps secrets out of URL", () => {
  const invite = validateInviteV1(corpus.valid, { now });
  const metadata = {
    id: "paired-safe",
    instanceId: invite.instanceId,
    nativePeerId: invite.nativePeerId,
    syncRoom: invite.syncRoom,
    signalingUrls: invite.signalingUrls,
    capabilityExpiresAtMs: Date.now() + 60_000,
    sessionUser: invite.session.user,
  };
  const context = buildLaunchContext(metadata, { password: invite.password, capabilityToken: invite.session.capabilityToken }, "ios");
  const html = '<html><head><script src="first.js"></script></head></html>';
  const injected = injectLaunchContext(html, context);
  assert.ok(injected.indexOf("data-ctox-mobile-bootstrap") < injected.indexOf('src="first.js"'));
  assert.match(injected, /CTOX_BUSINESS_OS_DESIGN_TEMPLATES=\[\]/);
  assert.equal(iosEntry(invite.instanceId).includes(invite.password), false);
  assert.equal(iosEntry(invite.instanceId).includes("ctox_config"), false);
  const hostile = buildLaunchContext({ ...metadata, sessionUser: { ...metadata.sessionUser, displayName: "</script><script>throw 1</script>" } }, { password: invite.password, capabilityToken: invite.session.capabilityToken }, "ios");
  const hardened = injectLaunchContext(html, hostile);
  assert.equal(hardened.includes("</script><script>throw 1</script>"), false);
  assert.match(hardened, /\\u003c\/script\\u003e/);
});

test("each paired instance receives a distinct persistent storage identity", async () => {
  const firstInvite = validateInviteV1(corpus.valid, { now });
  const registry = new MemoryRegistry();
  const secrets = new MemorySecretStore();
  const first = await pairAtomically(firstInvite, { registry, secrets });
  const second = await pairAtomically({ ...firstInvite, instanceId: "biz_unique", syncRoom: "ctox-business-os:biz_unique" }, { registry, secrets });
  assert.notEqual(first.storageIdentity, second.storageIdentity);
});

test("navigation permits only bundled origin and externalizes safe HTTPS", () => {
  assert.equal(navigationDecision("ctox-business-os-mobile://biz/business-os/index.html", { platform: "ios", instanceId: "biz" }), "allow");
  assert.equal(navigationDecision("https://docs.example.test/help", { platform: "ios", instanceId: "biz" }), "external");
  assert.equal(navigationDecision("data:text/html,hello", { platform: "ios", instanceId: "biz" }), "deny");
  assert.equal(navigationDecision("https://appassets.androidplatform.net/business-os/index.html", { platform: "android" }), "allow");
  assert.equal(navigationDecision("https://appassets.androidplatform.net/other/index.html", { platform: "android" }), "external");
});

test("unsupported Android WebView profile fails closed", () => {
  assert.throws(() => requireAndroidMultiProfile(false), /MULTI_PROFILE/);
  assert.equal(requireAndroidMultiProfile(true), true);
});

test("office pack verifies success, resume state, cancellation, offline, revision and hash failures", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "ctox-mobile-pack-"));
  await fs.mkdir(path.join(root, "nested"));
  await fs.writeFile(path.join(root, "nested/file.bin"), "office-bytes");
  const manifest = await buildPackManifest(root, { sourceRevision: "rev-a", appVersion: "0.1.0" });
  const progress = [];
  await verifyPack(root, manifest, { sourceRevision: "rev-a", appVersion: "0.1.0", onProgress: (value) => progress.push(value) });
  assert.equal(progress.at(-1), 1);
  assert.equal(officeRequestState({ type: "cancel" }, officeRequestState({ type: "request", totalBytes: 12 })).retryable, true);
  assert.equal(officeRequestState({ type: "offline" }).retryable, true);
  assert.equal(officeRequestState({ type: "progress", value: 0.5 }, officeRequestState({ type: "download" })).progress, 0.5);
  await assert.rejects(verifyPack(root, manifest, { sourceRevision: "rev-b", appVersion: "0.1.0" }), /revision/);
  await fs.writeFile(path.join(root, "nested/file.bin"), "corrupt");
  await assert.rejects(verifyPack(root, manifest, { sourceRevision: "rev-a", appVersion: "0.1.0" }), /size|hash/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(verifyPack(root, manifest, { sourceRevision: "rev-a", appVersion: "0.1.0", signal: controller.signal }), /canceled/);
});
