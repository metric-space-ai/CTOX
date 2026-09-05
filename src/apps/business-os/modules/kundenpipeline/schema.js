// Decision Hub / Kundenpipeline browser-side collection contract.
// The native owner is src/core/business_os/decision_hub.rs. Keep these
// declarations broad enough for forward-compatible projection fields while
// pinning the fields the UI actually reads.

import { collections as ctoxCoreCollections } from '../ctox/schema.js?v=20260816-browser-sync-guards-v141';

const jsonObject = { type: 'object', additionalProperties: true };

function recordSchema(properties, required = [], indexes = [], version = 0) {
  return {
    version,
    primaryKey: 'id',
    type: 'object',
    properties: {
      id: { type: 'string', maxLength: 256 },
      is_deleted: { type: 'boolean' },
      created_at_ms: { type: 'number' },
      updated_at_ms: { type: 'number' },
      deleted_at_ms: { type: 'number' },
      ...properties,
    },
    required: ['id', 'updated_at_ms', ...required],
    indexes: ['updated_at_ms', ['is_deleted', 'updated_at_ms'], ...indexes],
    additionalProperties: true,
  };
}

export const collections = {
  kundenpipeline_vorgaenge: recordSchema({
    title: { type: 'string' },
    status: { type: 'string' },
    kunde_id: { type: 'string' },
    kunde_name: { type: 'string' },
    quelle_json: jsonObject,
    triage_json: jsonObject,
    run_json: jsonObject,
    mails_json: { type: 'array', items: jsonObject },
    audit_json: { type: 'array', items: jsonObject },
    notes: { type: 'string' },
  }, ['title', 'status'], ['status', 'kunde_id']),

  kundenpipeline_entscheidungen: recordSchema({
    vorgang_id: { type: 'string' },
    typ: { type: 'string' },
    titel: { type: 'string' },
    title: { type: 'string' },
    frage_json: jsonObject,
    zeilen_json: { type: 'array', items: { type: 'string' } },
    detail_seiten_json: { type: 'array', items: jsonObject },
    aktionen_json: { type: 'array', items: jsonObject },
    backing_ref: { type: 'string' },
    source_json: jsonObject,
    correlation_json: jsonObject,
    request_fingerprint: { type: 'string' },
    status: { type: 'string' },
    antwort_json: jsonObject,
    expires_at_ms: { type: 'number' },
    requested_by: { type: 'string' },
    owner_user_id: { type: 'string' },
    assigned_user_id: { type: 'string' },
    participant_ids: { type: 'array', items: { type: 'string' } },
  }, ['status'], ['status', 'vorgang_id', 'typ']),

  kundenpipeline_projekte: recordSchema({
    name: { type: 'string' },
    code_projekt: { type: 'string' },
    adressen: { type: 'array', items: { type: 'string' } },
    domains: { type: 'array', items: { type: 'string' } },
    active: { type: 'boolean' },
  }, ['name'], ['name', 'active']),

  // Keep the command receipt schema byte-compatible with the shell's
  // canonical declaration; command dispatch itself remains ctx.commandBus.
  business_commands: ctoxCoreCollections.business_commands,
};

export const migrationStrategies = {
  business_commands: {
    1: (oldDoc) => ({
      ...oldDoc,
      inbound_channel: oldDoc.inbound_channel || oldDoc.module || '',
    }),
    2: (oldDoc) => oldDoc,
  },
  kundenpipeline_vorgaenge: { 1: (oldDoc) => oldDoc },
  kundenpipeline_entscheidungen: { 1: (oldDoc) => oldDoc },
  kundenpipeline_projekte: { 1: (oldDoc) => oldDoc },
};
