import crypto from "node:crypto";

export class MemorySecretStore {
  constructor({ failAt = 0 } = {}) { this.values = new Map(); this.writes = 0; this.failAt = failAt; }
  async set(ref, value) { this.writes += 1; if (this.failAt === this.writes) throw new Error("synthetic secret write failure"); this.values.set(ref, value); }
  async get(ref) { return this.values.get(ref) || ""; }
  async delete(ref) { this.values.delete(ref); }
}

export class MemoryRegistry {
  constructor() { this.value = { version: 1, instances: [] }; }
  async load() { return structuredClone(this.value); }
  async save(value) { assertRegistrySafe(value); this.value = structuredClone(value); }
}

export function safeMetadata(invite, prior = null) {
  const id = prior?.id || `paired:${crypto.createHash("sha256").update(invite.instanceId).digest("hex").slice(0, 24)}`;
  const generation = crypto.randomUUID();
  return {
    id,
    displayName: invite.displayName,
    instanceId: invite.instanceId,
    syncRoom: invite.syncRoom,
    nativePeerId: invite.nativePeerId,
    signalingUrls: [...invite.signalingUrls],
    expiresAt: invite.expiresAt,
    capabilityExpiresAtMs: invite.session.capabilityExpiresAtMs,
    sessionUser: { ...invite.session.user },
    passwordRef: `device-secret://ctox-business-os-mobile/${id}/${generation}/room`,
    capabilityRef: `device-secret://ctox-business-os-mobile/${id}/${generation}/capability`,
    storageIdentity: prior?.storageIdentity || crypto.randomUUID(),
  };
}

export async function pairAtomically(invite, { registry, secrets }) {
  const state = await registry.load();
  const previous = state.instances.find((item) => item.instanceId === invite.instanceId) || null;
  const next = safeMetadata(invite, previous);
  const written = [];
  try {
    await secrets.set(next.passwordRef, invite.password); written.push(next.passwordRef);
    await secrets.set(next.capabilityRef, invite.session.capabilityToken); written.push(next.capabilityRef);
    const instances = state.instances.filter((item) => item.instanceId !== invite.instanceId);
    instances.push(next);
    await registry.save({ version: 1, instances });
  } catch (error) {
    await Promise.all(written.map((ref) => secrets.delete(ref)));
    throw error;
  }
  if (previous) {
    await secrets.delete(previous.passwordRef);
    await secrets.delete(previous.capabilityRef);
  }
  return next;
}

export async function forgetInstance(id, { registry, secrets, removeStorage }) {
  const state = await registry.load();
  const target = state.instances.find((item) => item.id === id);
  if (!target) return false;
  await registry.save({ version: 1, instances: state.instances.filter((item) => item.id !== id) });
  await secrets.delete(target.passwordRef);
  await secrets.delete(target.capabilityRef);
  await removeStorage(target.storageIdentity);
  return true;
}

export function assertRegistrySafe(value) {
  const serialized = JSON.stringify(value);
  for (const forbidden of ["signaling_room_password", "capability_token", "ctox_config", "desktop_link", "payload="]) {
    if (serialized.includes(forbidden)) throw new Error(`registry contains forbidden field: ${forbidden}`);
  }
}
