#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { connect as tlsConnect } from "node:tls";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULTS = {
  channel: "email",
  provider: "one.com",
  db: resolve(process.cwd(), "runtime/cto_agent.db"),
  rawDir: resolve(process.cwd(), "runtime/communication/raw"),
  schema: join(__dirname, "communication_schema.sql"),
  agentBinary: process.env.CTO_AGENT_BINARY || resolve(process.cwd(), "target/debug/cto-agent"),
  imapHost: "imap.one.com",
  imapPort: 993,
  smtpHost: "send.one.com",
  smtpPort: 465,
  folder: "INBOX",
  limit: 20,
  trustLevel: "low",
  emitInterrupts: "false",
  interruptChannel: "email",
};

const DEBUG = process.env.COMM_DEBUG === "1";

function nowIso() {
  return new Date().toISOString();
}

function fail(message) {
  throw new Error(message);
}

function debugLog(...args) {
  if (!DEBUG) return;
  process.stderr.write(`[comm-debug] ${args.join(" ")}\n`);
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
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function runSql(dbPath, sql, { json = false } = {}) {
  const input = `${json ? ".mode json\n" : ""}${sql.trim().endsWith(";") ? sql.trim() : `${sql.trim()};`}\n`;
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

function decodeMimeHeader(value = "") {
  return String(value).replace(/=\?([^?]+)\?([bqBQ])\?([^?]*)\?=/g, (_full, _charset, encoding, payload) => {
    try {
      if (String(encoding).toUpperCase() === "B") {
        return Buffer.from(payload, "base64").toString("utf8");
      }
      const qp = payload
        .replace(/_/g, " ")
        .replace(/=([0-9A-Fa-f]{2})/g, (_match, hex) => String.fromCharCode(Number.parseInt(hex, 16)));
      return Buffer.from(qp, "binary").toString("utf8");
    } catch {
      return payload;
    }
  });
}

function unfoldHeaders(headerText) {
  return headerText.replace(/\r?\n[ \t]+/g, " ");
}

function parseHeaders(headerText) {
  const unfolded = unfoldHeaders(headerText);
  const headers = {};
  for (const line of unfolded.split(/\r?\n/)) {
    const index = line.indexOf(":");
    if (index <= 0) continue;
    const name = line.slice(0, index).trim().toLowerCase();
    const value = decodeMimeHeader(line.slice(index + 1).trim());
    headers[name] = value;
  }
  return headers;
}

function extractAddress(token = "") {
  const bracket = token.match(/<([^>]+)>/);
  if (bracket?.[1]) return bracket[1].trim().toLowerCase();
  const naked = token.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i);
  return naked?.[0]?.trim().toLowerCase() || "";
}

function extractDisplayName(token = "") {
  const bracket = token.match(/^(.*)<[^>]+>/);
  if (bracket?.[1]) return decodeMimeHeader(bracket[1].replace(/"/g, "").trim());
  return "";
}

function extractAddresses(raw = "") {
  const seen = new Set();
  const out = [];
  for (const token of String(raw || "").split(",")) {
    const address = extractAddress(token);
    if (!address || seen.has(address)) continue;
    seen.add(address);
    out.push(address);
  }
  return out;
}

function previewText(input = "") {
  return String(input || "").replace(/\s+/g, " ").trim().slice(0, 280);
}

function parseRfc822(rawBuffer) {
  const rawText = rawBuffer.toString("utf8");
  const separator = rawText.search(/\r?\n\r?\n/);
  const headerText = separator === -1 ? rawText : rawText.slice(0, separator);
  const bodyText = separator === -1 ? "" : rawText.slice(separator).replace(/^\r?\n\r?\n/, "");
  const headers = parseHeaders(headerText);
  return {
    headers,
    bodyText,
    subject: headers.subject || "(ohne Betreff)",
    fromHeader: headers.from || "",
    toHeader: headers.to || "",
    ccHeader: headers.cc || "",
    messageId: headers["message-id"] || "",
    references: headers.references || "",
    inReplyTo: headers["in-reply-to"] || "",
    sentAt: headers.date || "",
    hasAttachments: /content-disposition:\s*attachment/i.test(rawText) || /multipart\/mixed/i.test(rawText),
  };
}

function threadKeyFromEmail(parsed, fallback) {
  const references = String(parsed.references || "")
    .split(/\s+/)
    .map((value) => value.trim())
    .filter(Boolean);
  return references[0] || parsed.inReplyTo || parsed.messageId || fallback;
}

function accountKeyFromEmail(address) {
  return `email:${String(address || "").trim().toLowerCase()}`;
}

function messageKeyFromRemote(accountKey, folder, remoteId) {
  return `${accountKey}::${folder}::${remoteId}`;
}

async function writeRawPayload(rawDir, remoteId, rawBuffer) {
  await mkdir(rawDir, { recursive: true });
  const safeId = String(remoteId || randomUUID()).replace(/[^A-Za-z0-9_.-]/g, "_");
  const fullPath = join(rawDir, `${safeId}.eml`);
  await writeFile(fullPath, rawBuffer);
  return fullPath;
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
  return !!row?.message_key;
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
      String(latest.sender_address || "").trim().toLowerCase(),
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
      ${sqlValue(latest.subject || "(ohne Betreff)")},
      ${sqlValue(JSON.stringify(participants))},
      ${sqlValue(latest.message_key || "")},
      ${sqlValue(latest.external_created_at || nowIso())},
      ${sqlValue(Number(counts?.message_count || 0))},
      ${sqlValue(Number(counts?.unread_count || 0))},
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

class BufferSocket {
  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.closed = false;
    this.waiters = [];
    socket.on("data", (chunk) => {
      this.buffer = Buffer.concat([this.buffer, chunk]);
      this.flush();
    });
    socket.on("close", () => {
      this.closed = true;
      this.flush();
    });
  }

  flush() {
    const waiters = [...this.waiters];
    this.waiters = [];
    for (const waiter of waiters) {
      if (!waiter()) this.waiters.push(waiter);
    }
  }

  waitFor(test, timeoutMs = 20000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((entry) => entry !== runner);
        reject(new Error("socket timeout"));
      }, timeoutMs);

      const runner = () => {
        try {
          const result = test(this.buffer);
          if (result === false || result === null || result === undefined) {
            if (this.closed) {
              clearTimeout(timer);
              reject(new Error("socket closed before response"));
              return true;
            }
            return false;
          }
          clearTimeout(timer);
          resolve(result);
          return true;
        } catch (error) {
          clearTimeout(timer);
          reject(error);
          return true;
        }
      };

      this.waiters.push(runner);
      runner();
    });
  }

  async readLine(timeoutMs = 20000) {
    return this.waitFor((buffer) => {
      const index = buffer.indexOf("\r\n");
      if (index === -1) return false;
      const line = buffer.subarray(0, index).toString("utf8");
      this.buffer = buffer.subarray(index + 2);
      return line;
    }, timeoutMs);
  }

  async readUntilTagged(tag, timeoutMs = 20000) {
    return this.waitFor((buffer) => {
      const startMarker = Buffer.from(`${tag} `);
      const inline = buffer.indexOf(startMarker) === 0 ? 0 : -1;
      const tagged = inline === 0 ? 0 : buffer.indexOf(Buffer.from(`\r\n${tag} `));
      const lineStart = inline === 0 ? 0 : tagged === -1 ? -1 : tagged + 2;
      if (lineStart === -1) return false;
      const lineEnd = buffer.indexOf(Buffer.from("\r\n"), lineStart);
      if (lineEnd === -1) return false;
      const out = buffer.subarray(0, lineEnd + 2);
      this.buffer = buffer.subarray(lineEnd + 2);
      return out;
    }, timeoutMs);
  }
}

async function writeSocket(socket, data) {
  debugLog("write", JSON.stringify(String(data).slice(0, 400)));
  if (socket.write(data)) return;
  await once(socket, "drain");
}

function imapQuote(value) {
  return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function extractFetchLiteral(buffer) {
  const latin1 = buffer.toString("latin1");
  const match = /\{(\d+)\}\r\n/.exec(latin1);
  if (!match) {
    return { prefix: latin1, literal: Buffer.alloc(0) };
  }
  const literalLength = Number.parseInt(match[1], 10);
  const literalStart = match.index + match[0].length;
  return {
    prefix: latin1.slice(0, literalStart),
    literal: buffer.subarray(literalStart, literalStart + literalLength),
  };
}

class ImapClient {
  constructor(config) {
    this.config = config;
    this.tagCounter = 0;
    this.socket = null;
    this.reader = null;
  }

  nextTag() {
    this.tagCounter += 1;
    return `A${String(this.tagCounter).padStart(4, "0")}`;
  }

  async connect() {
    this.socket = await new Promise((resolveSocket, rejectSocket) => {
      const socket = tlsConnect(
        {
          host: this.config.imapHost,
          port: this.config.imapPort,
          servername: this.config.imapHost,
        },
        () => resolveSocket(socket)
      );
      socket.setTimeout(20000, () => socket.destroy(new Error("IMAP socket timeout")));
      socket.once("error", rejectSocket);
    });
    this.reader = new BufferSocket(this.socket);
    const greeting = await this.reader.readLine();
    debugLog("imap-greeting", greeting);
    if (!/\bOK\b/i.test(greeting)) fail(`IMAP greeting failed: ${greeting}`);
  }

  async command(commandText) {
    const tag = this.nextTag();
    await writeSocket(this.socket, `${tag} ${commandText}\r\n`);
    const response = await this.reader.readUntilTagged(tag);
    const text = response.toString("utf8");
    debugLog("imap-response", commandText, JSON.stringify(text.slice(0, 400)));
    const statusLine = text.match(new RegExp(`(?:^|\\r\\n)${tag} (OK|NO|BAD)`, "i"));
    if (!statusLine || statusLine[1].toUpperCase() !== "OK") {
      fail(`IMAP command failed: ${commandText}`);
    }
    return { tag, buffer: response, text };
  }

  async login(emailAddress, password) {
    await this.command(`LOGIN ${imapQuote(emailAddress)} ${imapQuote(password)}`);
  }

  async select(folder) {
    await this.command(`SELECT ${imapQuote(folder)}`);
  }

  async searchAllUids() {
    const response = await this.command("UID SEARCH ALL");
    const match = response.text.match(/\* SEARCH ?([0-9 ]*)/);
    return String(match?.[1] || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
  }

  async fetchRaw(uid) {
    const response = await this.command(`UID FETCH ${uid} (UID FLAGS RFC822)`);
    const { prefix, literal } = extractFetchLiteral(response.buffer);
    const flagsMatch = prefix.match(/FLAGS \(([^)]*)\)/i);
    const flags = String(flagsMatch?.[1] || "")
      .split(/\s+/)
      .map((value) => value.trim())
      .filter(Boolean);
    return { uid, flags, raw: literal };
  }

  async logout() {
    if (!this.socket) return;
    try {
      await this.command("LOGOUT");
    } catch {}
    this.socket.end();
  }
}

class SmtpClient {
  constructor(config) {
    this.config = config;
    this.socket = null;
    this.reader = null;
  }

  async connect() {
    this.socket = await new Promise((resolveSocket, rejectSocket) => {
      const socket = tlsConnect(
        {
          host: this.config.smtpHost,
          port: this.config.smtpPort,
          servername: this.config.smtpHost,
        },
        () => resolveSocket(socket)
      );
      socket.setTimeout(20000, () => socket.destroy(new Error("SMTP socket timeout")));
      socket.once("error", rejectSocket);
    });
    this.reader = new BufferSocket(this.socket);
    await this.expect([220]);
  }

  async expect(allowedCodes) {
    const first = await this.reader.readLine();
    const code = Number.parseInt(first.slice(0, 3), 10);
    const lines = [first];
    debugLog("smtp-response", JSON.stringify(first));
    let current = first;
    while (current[3] === "-") {
      current = await this.reader.readLine();
      debugLog("smtp-response", JSON.stringify(current));
      lines.push(current);
    }
    if (!allowedCodes.includes(code)) {
      fail(`SMTP failed: ${lines.join(" | ")}`);
    }
    return { code, lines };
  }

  async sendCommand(commandText, allowedCodes) {
    debugLog("smtp-command", JSON.stringify(commandText));
    await writeSocket(this.socket, `${commandText}\r\n`);
    return this.expect(allowedCodes);
  }

  async login(emailAddress, password) {
    await this.sendCommand(`EHLO localhost`, [250]);
    const payload = Buffer.from(`\u0000${emailAddress}\u0000${password}`, "utf8").toString("base64");
    await this.sendCommand(`AUTH PLAIN ${payload}`, [235]);
  }

  async sendMail(message) {
    await this.sendCommand(`MAIL FROM:<${message.from}>`, [250]);
    for (const recipient of [...message.to, ...message.cc, ...message.bcc]) {
      await this.sendCommand(`RCPT TO:<${recipient}>`, [250, 251]);
    }
    await this.sendCommand("DATA", [354]);

    const lines = [
      `From: ${message.from}`,
      `To: ${message.to.join(", ")}`,
      ...(message.cc.length ? [`Cc: ${message.cc.join(", ")}`] : []),
      `Subject: ${message.subject}`,
      `Message-ID: ${message.messageId}`,
      `Date: ${new Date().toUTCString()}`,
      "Content-Type: text/plain; charset=utf-8",
      "",
      ...String(message.body || "").split(/\r?\n/).map((line) => (line.startsWith(".") ? `.${line}` : line)),
      ".",
    ];
    await writeSocket(this.socket, `${lines.join("\r\n")}\r\n`);
    await this.expect([250]);
  }

  async close() {
    if (!this.socket) return;
    try {
      this.socket.end("QUIT\r\n");
    } catch {}
  }
}

function normalizeOptions(rawArgv) {
  const argv = [...rawArgv];
  const command = argv.shift();
  if (!command) fail("Usage: communication_mail_cli.mjs <sync|send|list> [options]");

  const options = {
    command,
    to: [],
    cc: [],
    bcc: [],
    db: DEFAULTS.db,
    rawDir: DEFAULTS.rawDir,
    schema: DEFAULTS.schema,
    agentBinary: DEFAULTS.agentBinary,
    provider: DEFAULTS.provider,
    folder: DEFAULTS.folder,
    limit: DEFAULTS.limit,
    trustLevel: DEFAULTS.trustLevel,
    emitInterrupts: DEFAULTS.emitInterrupts,
    interruptChannel: DEFAULTS.interruptChannel,
    imapHost: DEFAULTS.imapHost,
    imapPort: DEFAULTS.imapPort,
    smtpHost: DEFAULTS.smtpHost,
    smtpPort: DEFAULTS.smtpPort,
    email: process.env.CTO_EMAIL_ADDRESS || "",
    passwordEnv: "CTO_EMAIL_PASSWORD",
  };

  while (argv.length) {
    const token = argv.shift();
    if (!token.startsWith("--")) fail(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[0] && !argv[0].startsWith("--") ? argv.shift() : "true";
    if (["to", "cc", "bcc"].includes(key)) {
      options[key].push(value);
      continue;
    }
    options[key.replace(/-([a-z])/g, (_m, char) => char.toUpperCase())] = value;
  }

  options.limit = Number.parseInt(options.limit, 10) || DEFAULTS.limit;
  options.imapPort = Number.parseInt(options.imapPort, 10) || DEFAULTS.imapPort;
  options.smtpPort = Number.parseInt(options.smtpPort, 10) || DEFAULTS.smtpPort;
  options.password = process.env[options.passwordEnv] || "";
  return options;
}

function requireCredentials(options) {
  if (!options.email) fail("Missing --email or CTO_EMAIL_ADDRESS.");
  if (!options.password) fail(`Missing password in env ${options.passwordEnv}.`);
}

function buildProfileJson(options) {
  return JSON.stringify({
    imapHost: options.imapHost,
    imapPort: options.imapPort,
    smtpHost: options.smtpHost,
    smtpPort: options.smtpPort,
    folder: options.folder,
  });
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

function outboundMessageRecord(options, body, messageId) {
  const accountKey = accountKeyFromEmail(options.email);
  const observedAt = nowIso();
  const remoteId = messageId;
  const threadKey = remoteId;
  return {
    accountKey,
    message: {
      messageKey: messageKeyFromRemote(accountKey, "sent", remoteId),
      channel: DEFAULTS.channel,
      accountKey,
      threadKey,
      remoteId,
      direction: "outbound",
      folderHint: "sent",
      senderDisplay: "",
      senderAddress: options.email.toLowerCase(),
      recipientAddressesJson: JSON.stringify(options.to.map((value) => value.toLowerCase())),
      ccAddressesJson: JSON.stringify(options.cc.map((value) => value.toLowerCase())),
      bccAddressesJson: JSON.stringify(options.bcc.map((value) => value.toLowerCase())),
      subject: options.subject || "(ohne Betreff)",
      preview: previewText(body),
      bodyText: body,
      bodyHtml: "",
      rawPayloadRef: "",
      trustLevel: options.trustLevel,
      status: "sent",
      seen: 1,
      hasAttachments: 0,
      externalCreatedAt: observedAt,
      observedAt,
      metadataJson: JSON.stringify({ messageId }),
    },
  };
}

async function sendMail(options) {
  requireCredentials(options);
  await ensureSchema(options.db, options.schema);
  const accountKey = accountKeyFromEmail(options.email);
  const timestamp = nowIso();
  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.email.toLowerCase(),
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: timestamp,
    updatedAt: timestamp,
    lastInboundOkAt: null,
    lastOutboundOkAt: timestamp,
  });

  const smtp = new SmtpClient(options);
  const messageId = `<${randomUUID()}@${options.email.split("@").at(-1)}>`;
  try {
    await smtp.connect();
    await smtp.login(options.email, options.password);
    await smtp.sendMail({
      from: options.email,
      to: options.to.map((value) => value.toLowerCase()),
      cc: options.cc.map((value) => value.toLowerCase()),
      bcc: options.bcc.map((value) => value.toLowerCase()),
      subject: options.subject || "(ohne Betreff)",
      body: options.body || "",
      messageId,
    });
  } finally {
    await smtp.close();
  }

  const record = outboundMessageRecord(options, options.body || "", messageId);
  upsertMessage(options.db, record.message);
  refreshThread(options.db, record.message.threadKey);
  return {
    ok: true,
    accountKey,
    to: options.to,
    subject: options.subject || "(ohne Betreff)",
    messageId,
    dbPath: options.db,
  };
}

async function syncMail(options) {
  requireCredentials(options);
  await ensureSchema(options.db, options.schema);
  const accountKey = accountKeyFromEmail(options.email);
  const startedAt = nowIso();
  let fetchedCount = 0;
  let storedCount = 0;

  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.email.toLowerCase(),
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: startedAt,
    updatedAt: startedAt,
    lastInboundOkAt: null,
    lastOutboundOkAt: null,
  });

  const imap = new ImapClient(options);
  try {
    await imap.connect();
    await imap.login(options.email, options.password);
    await imap.select(options.folder);
    const allUids = await imap.searchAllUids();
    const selected = allUids.slice(-options.limit).reverse();
    fetchedCount = selected.length;

    for (const uid of selected) {
      const fetched = await imap.fetchRaw(uid);
      const parsed = parseRfc822(fetched.raw);
      const senderAddress = extractAddress(parsed.fromHeader);
      const senderDisplay = extractDisplayName(parsed.fromHeader) || senderAddress || "unknown";
      const remoteId = uid;
      const threadKey = threadKeyFromEmail(parsed, `${accountKey}::${remoteId}`);
      const messageKey = messageKeyFromRemote(accountKey, options.folder, remoteId);
      const alreadyKnown = messageExists(options.db, messageKey);
      const rawPayloadRef = await writeRawPayload(options.rawDir, remoteId, fetched.raw);
      const observedAt = nowIso();
      const preview = previewText(parsed.bodyText || parsed.subject);

      upsertMessage(options.db, {
        messageKey,
        channel: DEFAULTS.channel,
        accountKey,
        threadKey,
        remoteId,
        direction: "inbound",
        folderHint: options.folder,
        senderDisplay,
        senderAddress,
        recipientAddressesJson: JSON.stringify(extractAddresses(parsed.toHeader)),
        ccAddressesJson: JSON.stringify(extractAddresses(parsed.ccHeader)),
        bccAddressesJson: JSON.stringify([]),
        subject: parsed.subject,
        preview,
        bodyText: parsed.bodyText,
        bodyHtml: "",
        rawPayloadRef,
        trustLevel: options.trustLevel,
        status: "received",
        seen: fetched.flags.includes("\\Seen") ? 1 : 0,
        hasAttachments: parsed.hasAttachments ? 1 : 0,
        externalCreatedAt: parsed.sentAt || observedAt,
        observedAt,
        metadataJson: JSON.stringify({
          messageId: parsed.messageId,
          references: parsed.references,
          inReplyTo: parsed.inReplyTo,
          imapFlags: fetched.flags,
        }),
      });
      refreshThread(options.db, threadKey);
      if (!alreadyKnown && toBool(options.emitInterrupts)) {
        const speaker = senderAddress ? `${senderDisplay} <${senderAddress}>` : senderDisplay;
        const summary = [
          `E-Mail eingegangen von ${speaker || "unknown sender"}.`,
          `Betreff: ${parsed.subject || "(ohne Betreff)"}`,
          preview ? `Vorschau: ${preview}` : "",
        ]
          .filter(Boolean)
          .join("\n");
        emitAgentInterrupt(options, speaker || "unknown sender", summary);
      }
      storedCount += 1;
    }

    const finishedAt = nowIso();
    upsertAccount(options.db, {
      accountKey,
      channel: DEFAULTS.channel,
      address: options.email.toLowerCase(),
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
      folderHint: options.folder,
      startedAt,
      finishedAt,
      ok: true,
      fetchedCount,
      storedCount,
      errorText: "",
      metadataJson: JSON.stringify({ adapter: "js-mail-template" }),
    });
    return {
      ok: true,
      accountKey,
      folder: options.folder,
      fetchedCount,
      storedCount,
      dbPath: options.db,
    };
  } catch (error) {
    const finishedAt = nowIso();
    recordSyncRun(options.db, {
      runKey: randomUUID(),
      channel: DEFAULTS.channel,
      accountKey,
      folderHint: options.folder,
      startedAt,
      finishedAt,
      ok: false,
      fetchedCount,
      storedCount,
      errorText: String(error?.message || error),
      metadataJson: JSON.stringify({ adapter: "js-mail-template" }),
    });
    throw error;
  } finally {
    await imap.logout();
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
    if (!options.subject) fail("Missing --subject for send.");
    if (!options.body) fail("Missing --body for send.");
    if (!options.to.length) fail("Need at least one --to recipient.");
    result = await sendMail(options);
  } else if (options.command === "sync") {
    result = await syncMail(options);
  } else if (options.command === "list") {
    result = await listMessages(options);
  } else {
    fail(`Unsupported command: ${options.command}`);
  }

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

main().catch((error) => {
  process.stdout.write(`${JSON.stringify({ ok: false, error: String(error?.message || error) }, null, 2)}\n`);
  process.exitCode = 1;
});
