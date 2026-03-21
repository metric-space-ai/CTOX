#!/usr/bin/env node

import * as childProcess from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as url from "url";

const execFileSync = childProcess.execFileSync;
const mkdir = fs.promises.mkdir;
const readFile = fs.promises.readFile;
const readdir = fs.promises.readdir;
const rename = fs.promises.rename;
const unlink = fs.promises.unlink;
const writeFile = fs.promises.writeFile;
const copyFile = fs.promises.copyFile;
const dirname = path.dirname;
const extname = path.extname;
const join = path.join;
const resolve = path.resolve;
const basename = path.basename;
const fileURLToPath = url.fileURLToPath;
const randomUUID = crypto.randomUUID
  ? function randomUuidCompat() {
      return crypto.randomUUID();
    }
  : function randomUuidCompat() {
      return crypto.randomBytes(16).toString("hex");
    };

const __dirname = dirname(fileURLToPath(import.meta.url));

function defaultAgentBinary() {
  if (process.env.CTO_AGENT_BINARY) return process.env.CTO_AGENT_BINARY;
  const releasePath = resolve(process.cwd(), "target/release/cto-agent");
  if (fs.existsSync(releasePath)) return releasePath;
  return resolve(process.cwd(), "target/debug/cto-agent");
}

const DEFAULTS = {
  channel: "jami",
  provider: "jami",
  db: resolve(process.cwd(), "runtime/cto_agent.db"),
  rawDir: resolve(process.cwd(), "runtime/communication/jami/raw"),
  inboxDir: resolve(process.cwd(), "runtime/communication/jami/inbox"),
  outboxDir: resolve(process.cwd(), "runtime/communication/jami/outbox"),
  archiveDir: resolve(process.cwd(), "runtime/communication/jami/archive"),
  schema: resolve(process.cwd(), "scripts/communication_schema.sql"),
  agentBinary: defaultAgentBinary(),
  accountId: process.env.CTO_JAMI_ACCOUNT_ID || "",
  profileName: process.env.CTO_JAMI_PROFILE_NAME || "",
  limit: 50,
  trustLevel: "low",
  emitInterrupts: "false",
  interruptChannel: "jami",
};

function nowIso() {
  return new Date().toISOString();
}

function fail(message) {
  throw new Error(message);
}

function sqlValue(value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "NULL";
  if (typeof value === "boolean") return value ? "1" : "0";
  const text = String(value).replace(/\u0000/g, "").replace(/'/g, "''");
  return `'${text}'`;
}

function parseJsonOutput(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return [];
  return JSON.parse(trimmed);
}

function toBool(value) {
  const normalized = String(value == null ? "" : value)
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function runSql(dbPath, sql, { json = false } = {}) {
  const input = `.timeout 5000\n${json ? ".mode json\n" : ""}${sql.trim().endsWith(";") ? sql.trim() : `${sql.trim()};`}\n`;
  return execFileSync("sqlite3", [dbPath], {
    input,
    encoding: "utf8",
    stdio: ["pipe", "pipe", "pipe"],
  });
}

function firstRow(dbPath, sql) {
  return parseJsonOutput(runSql(dbPath, sql, { json: true }))[0] || null;
}

async function ensureSchema(dbPath, schemaPath) {
  await mkdir(dirname(dbPath), { recursive: true });
  const sql = await readFile(schemaPath, "utf8");
  runSql(dbPath, sql);
}

function previewText(input = "") {
  return String(input || "").replace(/\s+/g, " ").trim().slice(0, 280);
}

function sanitizeFileComponent(value = "") {
  return String(value || "")
    .trim()
    .replace(/[^A-Za-z0-9_.-]/g, "_")
    .slice(0, 120);
}

function normalizeText(value) {
  return String(value == null ? "" : value).trim();
}

function normalizeAddress(value) {
  return normalizeText(value);
}

function normalizeList(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeAddress(entry)).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/[,\n]/)
      .map((entry) => normalizeAddress(entry))
      .filter(Boolean);
  }
  return [];
}

function accountKeyFromJami(accountId) {
  const normalized = normalizeText(accountId).toLowerCase();
  if (normalized.startsWith("jami:")) return normalized;
  return `jami:${normalized}`;
}

function messageKeyFromRemote(accountKey, folder, remoteId) {
  return `${accountKey}::${folder}::${remoteId}`;
}

function buildProfileJson(options) {
  return JSON.stringify({
    inboxDir: options.inboxDir,
    outboxDir: options.outboxDir,
    archiveDir: options.archiveDir,
    profileName: options.profileName,
  });
}

function upsertAccount(dbPath, account) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_accounts (
      account_key, channel, address, provider, profile_json, created_at, updated_at, last_inbound_ok_at, last_outbound_ok_at
    ) VALUES (
      ${sqlValue(account.accountKey)},
      ${sqlValue(account.channel)},
      ${sqlValue(account.address)},
      ${sqlValue(account.provider)},
      ${sqlValue(account.profileJson)},
      ${sqlValue(account.createdAt)},
      ${sqlValue(account.updatedAt)},
      ${sqlValue(account.lastInboundOkAt)},
      ${sqlValue(account.lastOutboundOkAt)}
    )
    ON CONFLICT(account_key) DO UPDATE SET
      channel=excluded.channel,
      address=excluded.address,
      provider=excluded.provider,
      profile_json=excluded.profile_json,
      updated_at=excluded.updated_at,
      last_inbound_ok_at=COALESCE(excluded.last_inbound_ok_at, communication_accounts.last_inbound_ok_at),
      last_outbound_ok_at=COALESCE(excluded.last_outbound_ok_at, communication_accounts.last_outbound_ok_at)
    `
  );
}

function upsertMessage(dbPath, message) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_messages (
      message_key, channel, account_key, thread_key, remote_id, direction, folder_hint, sender_display, sender_address,
      recipient_addresses_json, cc_addresses_json, bcc_addresses_json, subject, preview, body_text, body_html,
      raw_payload_ref, trust_level, status, seen, has_attachments, external_created_at, observed_at, metadata_json
    ) VALUES (
      ${sqlValue(message.messageKey)},
      ${sqlValue(message.channel)},
      ${sqlValue(message.accountKey)},
      ${sqlValue(message.threadKey)},
      ${sqlValue(message.remoteId)},
      ${sqlValue(message.direction)},
      ${sqlValue(message.folderHint)},
      ${sqlValue(message.senderDisplay)},
      ${sqlValue(message.senderAddress)},
      ${sqlValue(message.recipientAddressesJson)},
      ${sqlValue(message.ccAddressesJson)},
      ${sqlValue(message.bccAddressesJson)},
      ${sqlValue(message.subject)},
      ${sqlValue(message.preview)},
      ${sqlValue(message.bodyText)},
      ${sqlValue(message.bodyHtml)},
      ${sqlValue(message.rawPayloadRef)},
      ${sqlValue(message.trustLevel)},
      ${sqlValue(message.status)},
      ${sqlValue(message.seen)},
      ${sqlValue(message.hasAttachments)},
      ${sqlValue(message.externalCreatedAt)},
      ${sqlValue(message.observedAt)},
      ${sqlValue(message.metadataJson)}
    )
    ON CONFLICT(message_key) DO UPDATE SET
      thread_key=excluded.thread_key,
      sender_display=excluded.sender_display,
      sender_address=excluded.sender_address,
      recipient_addresses_json=excluded.recipient_addresses_json,
      cc_addresses_json=excluded.cc_addresses_json,
      bcc_addresses_json=excluded.bcc_addresses_json,
      subject=excluded.subject,
      preview=excluded.preview,
      body_text=excluded.body_text,
      body_html=excluded.body_html,
      raw_payload_ref=excluded.raw_payload_ref,
      trust_level=excluded.trust_level,
      status=excluded.status,
      seen=excluded.seen,
      has_attachments=excluded.has_attachments,
      external_created_at=excluded.external_created_at,
      observed_at=excluded.observed_at,
      metadata_json=excluded.metadata_json
    `
  );
}

function messageExists(dbPath, messageKey) {
  const row = firstRow(
    dbPath,
    `
    SELECT message_key
    FROM communication_messages
    WHERE message_key = ${sqlValue(messageKey)}
    LIMIT 1
    `
  );
  return !!(row && row.message_key);
}

function refreshThread(dbPath, threadKey) {
  const latestRows = parseJsonOutput(
    runSql(
      dbPath,
      `
      SELECT subject, message_key, external_created_at, sender_address, recipient_addresses_json
      FROM communication_messages
      WHERE thread_key = ${sqlValue(threadKey)}
      ORDER BY external_created_at DESC, observed_at DESC
      LIMIT 1
      `,
      { json: true }
    )
  );
  if (!latestRows.length) return;

  const latest = latestRows[0];
  const counts = parseJsonOutput(
    runSql(
      dbPath,
      `
      SELECT
        COUNT(*) AS message_count,
        SUM(CASE WHEN seen = 0 THEN 1 ELSE 0 END) AS unread_count
      FROM communication_messages
      WHERE thread_key = ${sqlValue(threadKey)}
      `,
      { json: true }
    )
  )[0];

  const participants = Array.from(
    new Set([
      String(latest.sender_address || "").trim(),
      ...JSON.parse(latest.recipient_addresses_json || "[]"),
    ].filter(Boolean))
  );

  runSql(
    dbPath,
    `
    INSERT INTO communication_threads (
      thread_key, channel, account_key, subject, participant_keys_json, last_message_key,
      last_message_at, message_count, unread_count, metadata_json, updated_at
    )
    SELECT
      ${sqlValue(threadKey)},
      channel,
      account_key,
      ${sqlValue(latest.subject || "(Jami)")},
      ${sqlValue(JSON.stringify(participants))},
      ${sqlValue(latest.message_key || "")},
      ${sqlValue(latest.external_created_at || nowIso())},
      ${sqlValue(Number((counts && counts.message_count) || 0))},
      ${sqlValue(Number((counts && counts.unread_count) || 0))},
      ${sqlValue("{}")},
      ${sqlValue(nowIso())}
    FROM communication_messages
    WHERE thread_key = ${sqlValue(threadKey)}
    LIMIT 1
    ON CONFLICT(thread_key) DO UPDATE SET
      subject=excluded.subject,
      participant_keys_json=excluded.participant_keys_json,
      last_message_key=excluded.last_message_key,
      last_message_at=excluded.last_message_at,
      message_count=excluded.message_count,
      unread_count=excluded.unread_count,
      metadata_json=excluded.metadata_json,
      updated_at=excluded.updated_at
    `
  );
}

function recordSyncRun(dbPath, run) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_sync_runs (
      run_key, channel, account_key, folder_hint, started_at, finished_at,
      ok, fetched_count, stored_count, error_text, metadata_json
    ) VALUES (
      ${sqlValue(run.runKey)},
      ${sqlValue(run.channel)},
      ${sqlValue(run.accountKey)},
      ${sqlValue(run.folderHint)},
      ${sqlValue(run.startedAt)},
      ${sqlValue(run.finishedAt)},
      ${sqlValue(run.ok)},
      ${sqlValue(run.fetchedCount)},
      ${sqlValue(run.storedCount)},
      ${sqlValue(run.errorText)},
      ${sqlValue(run.metadataJson)}
    )
    `
  );
}

function emitAgentInterrupt(options, speaker, summary) {
  const output = execFileSync(
    options.agentBinary,
    ["channel-interrupt", options.interruptChannel, speaker, summary],
    {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  return String(output || "").trim();
}

async function ensureDir(dirPath) {
  await mkdir(dirPath, { recursive: true });
}

async function writeRawPayload(rawDir, remoteId, payload) {
  await ensureDir(rawDir);
  const safeId = sanitizeFileComponent(remoteId || randomUUID()) || randomUUID();
  const fullPath = join(rawDir, `${safeId}.json`);
  await writeFile(fullPath, `${JSON.stringify(payload, null, 2)}\n`);
  return fullPath;
}

function normalizeInboundEntry(options, entry, sourceName, index) {
  const accountKey = accountKeyFromJami(options.accountId);
  const senderAddress = normalizeAddress(
    entry.senderAddress || entry.senderUri || entry.fromAddress || entry.fromUri || entry.author || ""
  );
  const senderDisplay = normalizeText(
    entry.senderDisplay || entry.senderName || entry.fromName || senderAddress || "unknown sender"
  );
  const recipients = normalizeList(
    entry.recipientAddresses || entry.to || entry.participants || entry.recipientUris || []
  );
  const remoteId = normalizeText(
    entry.remoteId || entry.id || entry.messageId || entry.uri || `${basename(sourceName)}:${index}`
  );
  const subject = normalizeText(entry.subject || entry.conversationLabel || entry.threadLabel || "(Jami)");
  const bodyText = normalizeText(entry.bodyText || entry.body || entry.text || entry.message || "");
  const externalCreatedAt = normalizeText(
    entry.externalCreatedAt || entry.createdAt || entry.timestamp || entry.sentAt || nowIso()
  );
  const threadKey = normalizeText(
    entry.threadKey || entry.conversationId || entry.conversationUri || entry.threadId || ""
  ) || `${accountKey}::${sanitizeFileComponent(senderAddress || senderDisplay || subject || remoteId)}`;
  const messageKey = messageKeyFromRemote(accountKey, "INBOX", remoteId);
  return {
    accountKey,
    threadKey,
    messageKey,
    remoteId,
    senderDisplay,
    senderAddress,
    recipients,
    subject,
    bodyText,
    preview: previewText(entry.preview || bodyText || subject),
    seen: toBool(entry.seen) ? 1 : 0,
    hasAttachments:
      toBool(entry.hasAttachments) || (Array.isArray(entry.attachments) && entry.attachments.length > 0) ? 1 : 0,
    externalCreatedAt,
    metadataJson: JSON.stringify({
      sourceFile: sourceName,
      conversationId: entry.conversationId || entry.conversationUri || "",
      threadLabel: entry.threadLabel || entry.conversationLabel || "",
      rawEntry: entry,
    }),
  };
}

async function archiveSourceFile(sourcePath, archiveDir) {
  await ensureDir(archiveDir);
  const destinationBase = join(archiveDir, basename(sourcePath));
  let destination = destinationBase;
  if (fs.existsSync(destination)) {
    destination = join(
      archiveDir,
      `${path.parse(basename(sourcePath)).name}-${Date.now()}${extname(sourcePath)}`
    );
  }
  try {
    await rename(sourcePath, destination);
  } catch {
    await copyFile(sourcePath, destination);
    await unlink(sourcePath);
  }
  return destination;
}

async function loadSourceEntries(filePath) {
  const text = await readFile(filePath, "utf8");
  if (extname(filePath).toLowerCase() === ".jsonl") {
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }
  const parsed = JSON.parse(text);
  if (Array.isArray(parsed)) return parsed;
  return [parsed];
}

function normalizeOptions(rawArgv) {
  const argv = [...rawArgv];
  const command = argv.shift();
  if (!command) fail("Usage: communication_jami_cli.mjs <sync|send|list> [options]");

  const options = {
    command,
    to: [],
    db: DEFAULTS.db,
    rawDir: DEFAULTS.rawDir,
    inboxDir: process.env.CTO_JAMI_INBOX_DIR || DEFAULTS.inboxDir,
    outboxDir: process.env.CTO_JAMI_OUTBOX_DIR || DEFAULTS.outboxDir,
    archiveDir: process.env.CTO_JAMI_ARCHIVE_DIR || DEFAULTS.archiveDir,
    schema: DEFAULTS.schema,
    agentBinary: DEFAULTS.agentBinary,
    provider: DEFAULTS.provider,
    limit: DEFAULTS.limit,
    trustLevel: DEFAULTS.trustLevel,
    emitInterrupts: DEFAULTS.emitInterrupts,
    interruptChannel: DEFAULTS.interruptChannel,
    accountId: DEFAULTS.accountId,
    profileName: DEFAULTS.profileName,
  };

  while (argv.length) {
    const token = argv.shift();
    if (!token.startsWith("--")) fail(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[0] && !argv[0].startsWith("--") ? argv.shift() : "true";
    if (key === "to") {
      options.to.push(value);
      continue;
    }
    options[key.replace(/-([a-z])/g, (_m, char) => char.toUpperCase())] = value;
  }

  options.limit = Number.parseInt(options.limit, 10) || DEFAULTS.limit;
  if (!options.profileName) options.profileName = options.accountId;
  return options;
}

function requireAccount(options) {
  if (!normalizeText(options.accountId)) {
    fail("Missing --account-id or CTO_JAMI_ACCOUNT_ID.");
  }
}

async function sendJami(options) {
  requireAccount(options);
  if (!options.to.length) fail("Need at least one --to recipient.");
  if (!options.body) fail("Missing --body for send.");
  await ensureSchema(options.db, options.schema);
  await ensureDir(options.outboxDir);

  const accountKey = accountKeyFromJami(options.accountId);
  const timestamp = nowIso();
  const remoteId = `queued-${randomUUID()}`;
  const threadKey =
    normalizeText(options.threadKey || options.to[0]) || `${accountKey}::outbox::${sanitizeFileComponent(remoteId)}`;
  const subject = normalizeText(options.subject || options.threadLabel || "(Jami)");
  const payload = {
    accountId: options.accountId,
    profileName: options.profileName,
    threadKey,
    remoteId,
    to: options.to,
    subject,
    bodyText: options.body,
    createdAt: timestamp,
    metadata: {
      queuedBy: "communication_jami_cli",
    },
  };
  const outboxPath = join(
    options.outboxDir,
    `${new Date().toISOString().replace(/[:.]/g, "-")}-${sanitizeFileComponent(remoteId)}.json`
  );
  await writeFile(outboxPath, `${JSON.stringify(payload, null, 2)}\n`);

  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.accountId,
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: timestamp,
    updatedAt: timestamp,
    lastInboundOkAt: null,
    lastOutboundOkAt: timestamp,
  });

  upsertMessage(options.db, {
    messageKey: messageKeyFromRemote(accountKey, "outbox", remoteId),
    channel: DEFAULTS.channel,
    accountKey,
    threadKey,
    remoteId,
    direction: "outbound",
    folderHint: "outbox",
    senderDisplay: options.profileName || options.accountId,
    senderAddress: options.accountId,
    recipientAddressesJson: JSON.stringify(options.to.map((value) => normalizeAddress(value))),
    ccAddressesJson: JSON.stringify([]),
    bccAddressesJson: JSON.stringify([]),
    subject,
    preview: previewText(options.body),
    bodyText: options.body,
    bodyHtml: "",
    rawPayloadRef: outboxPath,
    trustLevel: options.trustLevel,
    status: "queued",
    seen: 1,
    hasAttachments: 0,
    externalCreatedAt: timestamp,
    observedAt: timestamp,
    metadataJson: JSON.stringify({
      bridge: "jami-file-bridge",
      delivery: "queued_for_bridge",
      outboxPath,
    }),
  });
  refreshThread(options.db, threadKey);

  return {
    ok: true,
    queued: true,
    accountKey,
    to: options.to,
    subject,
    outboxPath,
    dbPath: options.db,
  };
}

async function syncJami(options) {
  requireAccount(options);
  await ensureSchema(options.db, options.schema);
  await ensureDir(options.inboxDir);
  await ensureDir(options.archiveDir);

  const accountKey = accountKeyFromJami(options.accountId);
  const startedAt = nowIso();
  let fetchedCount = 0;
  let storedCount = 0;

  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.accountId,
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: startedAt,
    updatedAt: startedAt,
    lastInboundOkAt: null,
    lastOutboundOkAt: null,
  });

  try {
    const sourceFiles = (await readdir(options.inboxDir))
      .filter((name) => [".json", ".jsonl"].includes(extname(name).toLowerCase()))
      .sort()
      .slice(0, options.limit);

    for (const name of sourceFiles) {
      const sourcePath = join(options.inboxDir, name);
      const entries = await loadSourceEntries(sourcePath);
      fetchedCount += entries.length;
      for (let index = 0; index < entries.length; index += 1) {
        const inbound = normalizeInboundEntry(options, entries[index], name, index);
        const alreadyKnown = messageExists(options.db, inbound.messageKey);
        const observedAt = nowIso();
        const rawPayloadRef = await writeRawPayload(options.rawDir, inbound.remoteId, entries[index]);
        upsertMessage(options.db, {
          messageKey: inbound.messageKey,
          channel: DEFAULTS.channel,
          accountKey: inbound.accountKey,
          threadKey: inbound.threadKey,
          remoteId: inbound.remoteId,
          direction: "inbound",
          folderHint: "INBOX",
          senderDisplay: inbound.senderDisplay,
          senderAddress: inbound.senderAddress,
          recipientAddressesJson: JSON.stringify(inbound.recipients),
          ccAddressesJson: JSON.stringify([]),
          bccAddressesJson: JSON.stringify([]),
          subject: inbound.subject,
          preview: inbound.preview,
          bodyText: inbound.bodyText,
          bodyHtml: "",
          rawPayloadRef,
          trustLevel: options.trustLevel,
          status: "received",
          seen: inbound.seen,
          hasAttachments: inbound.hasAttachments,
          externalCreatedAt: inbound.externalCreatedAt,
          observedAt,
          metadataJson: inbound.metadataJson,
        });
        refreshThread(options.db, inbound.threadKey);
        if (!alreadyKnown && toBool(options.emitInterrupts)) {
          const speaker = inbound.senderAddress
            ? `${inbound.senderDisplay} <${inbound.senderAddress}>`
            : inbound.senderDisplay;
          const summary = [
            `Jami-Nachricht eingegangen von ${speaker || "unknown sender"}.`,
            inbound.subject ? `Konversation: ${inbound.subject}` : "",
            inbound.preview ? `Vorschau: ${inbound.preview}` : "",
          ]
            .filter(Boolean)
            .join("\n");
          emitAgentInterrupt(options, speaker || "unknown sender", summary);
        }
        storedCount += 1;
      }
      await archiveSourceFile(sourcePath, options.archiveDir);
    }

    const finishedAt = nowIso();
    upsertAccount(options.db, {
      accountKey,
      channel: DEFAULTS.channel,
      address: options.accountId,
      provider: options.provider,
      profileJson: buildProfileJson(options),
      createdAt: startedAt,
      updatedAt: finishedAt,
      lastInboundOkAt: finishedAt,
      lastOutboundOkAt: null,
    });
    recordSyncRun(options.db, {
      runKey: randomUUID(),
      channel: DEFAULTS.channel,
      accountKey,
      folderHint: "INBOX",
      startedAt,
      finishedAt,
      ok: true,
      fetchedCount,
      storedCount,
      errorText: "",
      metadataJson: JSON.stringify({ adapter: "js-jami-file-bridge" }),
    });
    return {
      ok: true,
      accountKey,
      fetchedCount,
      storedCount,
      inboxDir: options.inboxDir,
      archiveDir: options.archiveDir,
      dbPath: options.db,
    };
  } catch (error) {
    const finishedAt = nowIso();
    recordSyncRun(options.db, {
      runKey: randomUUID(),
      channel: DEFAULTS.channel,
      accountKey,
      folderHint: "INBOX",
      startedAt,
      finishedAt,
      ok: false,
      fetchedCount,
      storedCount,
      errorText: String((error && error.message) || error),
      metadataJson: JSON.stringify({ adapter: "js-jami-file-bridge" }),
    });
    throw error;
  }
}

async function listMessages(options) {
  await ensureSchema(options.db, options.schema);
  const rows = parseJsonOutput(
    runSql(
      options.db,
      `
      SELECT channel, account_key, folder_hint, direction, subject, sender_address, external_created_at, preview
      FROM communication_messages
      WHERE channel = 'jami'
      ORDER BY external_created_at DESC, observed_at DESC
      LIMIT ${sqlValue(options.limit)}
      `,
      { json: true }
    )
  );
  return {
    ok: true,
    count: rows.length,
    dbPath: options.db,
    messages: rows,
  };
}

async function main() {
  const options = normalizeOptions(process.argv.slice(2));
  let result;
  if (options.command === "send") {
    result = await sendJami(options);
  } else if (options.command === "sync") {
    result = await syncJami(options);
  } else if (options.command === "list") {
    result = await listMessages(options);
  } else {
    fail(`Unsupported command: ${options.command}`);
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

export { main, normalizeInboundEntry, previewText };

const executedPath = process.argv[1] ? resolve(process.argv[1]) : "";
const currentModulePath = fileURLToPath(import.meta.url);

if (executedPath === currentModulePath) {
  main().catch((error) => {
    process.stdout.write(
      `${JSON.stringify({ ok: false, error: String((error && error.message) || error) }, null, 2)}\n`
    );
    process.exitCode = 1;
  });
}
