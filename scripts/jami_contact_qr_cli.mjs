#!/usr/bin/env node

import * as childProcess from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

const SERVICE = "cx.ring.Ring";
const OBJECT_PATH = "/cx/ring/Ring/ConfigurationManager";
const INTERFACE = "cx.ring.Ring.ConfigurationManager";
const DEFAULT_TIMEOUT_SECS = 45;

function fail(message, code = 1) {
  fs.writeSync(process.stderr.fd, `${String(message || "").trim()}\n`);
  process.exitCode = code;
  process.exit();
}

function parseArgs(argv) {
  const args = {
    command: "ensure-contact",
    timeoutSecs: DEFAULT_TIMEOUT_SECS,
    displayName: "",
    accountId: process.env.CTO_JAMI_ACCOUNT_ID || "",
  };
  const input = [...argv];
  if (input[0] && !input[0].startsWith("--")) {
    args.command = String(input.shift() || "");
  }
  for (let index = 0; index < input.length; index += 1) {
    const token = input[index];
    switch (token) {
      case "--timeout-secs":
        args.timeoutSecs = Number.parseInt(input[index + 1] || "", 10) || DEFAULT_TIMEOUT_SECS;
        index += 1;
        break;
      case "--display-name":
        args.displayName = String(input[index + 1] || "");
        index += 1;
        break;
      case "--account-id":
        args.accountId = String(input[index + 1] || "");
        index += 1;
        break;
      case "--help":
      case "-h":
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${token}`);
    }
  }
  return args;
}

function dbusEnvFileCandidates() {
  const runtimeDir =
    process.env.XDG_RUNTIME_DIR ||
    (typeof process.getuid === "function" ? path.join("/run/user", String(process.getuid())) : "");
  const candidates = [];
  if (process.env.CTO_JAMI_DBUS_ENV_FILE) {
    candidates.push(process.env.CTO_JAMI_DBUS_ENV_FILE);
  }
  if (runtimeDir) {
    candidates.push(path.join(runtimeDir, "cto-jami-dbus.env"));
  }
  candidates.push("/tmp/cto-jami-dbus.env");
  return [...new Set(candidates.filter(Boolean))];
}

function loadJamiDbusEnvironment() {
  for (const candidate of dbusEnvFileCandidates()) {
    if (!fs.existsSync(candidate)) continue;
    const raw = fs.readFileSync(candidate, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("export ")) continue;
      const match = trimmed.match(/^([A-Z0-9_]+)=(.*?);?$/);
      if (!match) continue;
      const key = match[1];
      let value = match[2].trim();
      if (
        (value.startsWith("'") && value.endsWith("'")) ||
        (value.startsWith('"') && value.endsWith('"'))
      ) {
        value = value.slice(1, -1);
      }
      process.env[key] = value;
    }
    process.env.CTO_JAMI_DBUS_ENV_FILE = candidate;
    return candidate;
  }
  return null;
}

function decodeDbusString(value) {
  return String(value)
    .replace(/\\'/g, "'")
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\");
}

function parseCompactDetails(rawDetails) {
  const trimmed = String(rawDetails || "").trim();
  if (trimmed === "{}") {
    return {};
  }
  const body = trimmed.replace(/^\{\s*/, "").replace(/\s*\}$/, "");
  const details = {};
  let index = 0;

  while (index < body.length) {
    while (index < body.length && /[\s,]/.test(body[index])) index += 1;
    if (index >= body.length || body[index] !== "'") break;
    index += 1;

    let key = "";
    while (index < body.length) {
      const char = body[index];
      if (char === "\\") {
        key += body.slice(index, index + 2);
        index += 2;
        continue;
      }
      if (char === "'") {
        index += 1;
        break;
      }
      key += char;
      index += 1;
    }

    while (index < body.length && /\s/.test(body[index])) index += 1;
    if (body.slice(index, index + 1) !== ":") break;
    index += 1;
    while (index < body.length && /\s/.test(body[index])) index += 1;

    let value = "";
    if (body[index] === "'") {
      index += 1;
      while (index < body.length) {
        const char = body[index];
        if (char === "\\") {
          value += body.slice(index, index + 2);
          index += 2;
          continue;
        }
        if (char === "'") {
          index += 1;
          break;
        }
        value += char;
        index += 1;
      }
      details[decodeDbusString(key)] = decodeDbusString(value);
    } else {
      while (index < body.length && body[index] !== ",") {
        value += body[index];
        index += 1;
      }
      details[decodeDbusString(key)] = value.trim();
    }
  }

  return details;
}

function parseStringArrayTuple(output) {
  const values = [];
  const regex = /'((?:[^'\\]|\\.)*)'/g;
  for (const match of String(output || "").matchAll(regex)) {
    values.push(decodeDbusString(match[1]));
  }
  return values;
}

function parseSingleStringTuple(output) {
  const match = String(output || "").match(/\(\s*'((?:[^'\\]|\\.)*)'\s*,?\s*\)/);
  if (!match) {
    throw new Error(`Could not parse DBus string result: ${String(output || "").trim()}`);
  }
  return decodeDbusString(match[1]);
}

function parseStringDictTuple(output) {
  const text = String(output || "").trim();
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end < start) {
    return {};
  }
  return parseCompactDetails(text.slice(start, end + 1));
}

function parseBoolean(value) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function buildStringDict(details) {
  const entries = Object.entries(details).map(([key, value]) => {
    const safeKey = String(key).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    const safeValue = String(value).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    return `'${safeKey}': '${safeValue}'`;
  });
  return `{${entries.join(", ")}}`;
}

function runGdbusCall(method, args = []) {
  const result = childProcess.spawnSync(
    "gdbus",
    [
      "call",
      "--session",
      "--dest",
      SERVICE,
      "--object-path",
      OBJECT_PATH,
      "--method",
      `${INTERFACE}.${method}`,
      ...args,
    ],
    {
      encoding: "utf8",
    }
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error((result.stderr || result.stdout || `gdbus ${method} failed`).trim());
  }
  return String(result.stdout || "").trim();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hostnameLabel() {
  const host = String(os.hostname() || "").trim();
  return host || "cto-agent";
}

function preferredDisplayName(rawDisplayName) {
  const cleaned = String(rawDisplayName || "").trim();
  return cleaned || "CTO-Agent";
}

function looksLikeJamiId(value) {
  return /^[a-f0-9]{40}$/i.test(String(value || "").trim());
}

function summarizeAccount(accountId, details, volatileDetails) {
  const jamiId = String(details["Account.username"] || "").trim();
  const registrationStatus = String(volatileDetails["Account.registrationStatus"] || "").trim();
  const displayName = String(details["Account.displayName"] || details["Account.alias"] || "").trim();
  const deviceName = String(details["Account.deviceName"] || "").trim();
  return {
    accountId,
    jamiId,
    displayName,
    deviceName,
    registrationStatus,
    active: parseBoolean(volatileDetails["Account.active"]),
    deviceAnnounced: parseBoolean(volatileDetails["Account.deviceAnnounced"]),
    accountType: String(details["Account.type"] || "").trim(),
    details,
    volatileDetails,
  };
}

function accountScore(account, requestedAccountId, requestedDisplayName) {
  let score = 0;
  if (requestedAccountId && account.accountId === requestedAccountId) score += 1000;
  if (account.accountType === "RING") score += 100;
  if (account.registrationStatus === "REGISTERED") score += 300;
  if (account.active) score += 80;
  if (account.deviceAnnounced) score += 60;
  if (account.jamiId) score += 50;
  const preferredName = String(requestedDisplayName || "").trim().toLowerCase();
  if (
    preferredName &&
    [account.displayName, account.deviceName]
      .map((value) => String(value || "").trim().toLowerCase())
      .includes(preferredName)
  ) {
    score += 40;
  }
  return score;
}

function chooseBestExistingAccount(accounts, requestedAccountId, requestedDisplayName) {
  const candidates = accounts.filter((account) => account.accountType === "RING");
  if (!candidates.length) return null;
  return [...candidates].sort((left, right) => {
    const scoreDelta =
      accountScore(right, requestedAccountId, requestedDisplayName) -
      accountScore(left, requestedAccountId, requestedDisplayName);
    if (scoreDelta !== 0) return scoreDelta;
    return left.accountId.localeCompare(right.accountId);
  })[0];
}

function listAccounts() {
  return parseStringArrayTuple(runGdbusCall("getAccountList"));
}

function getAccountDetails(accountId) {
  return parseStringDictTuple(runGdbusCall("getAccountDetails", [accountId]));
}

function getVolatileAccountDetails(accountId) {
  return parseStringDictTuple(runGdbusCall("getVolatileAccountDetails", [accountId]));
}

async function loadExistingAccounts() {
  const accounts = [];
  for (const accountId of listAccounts()) {
    try {
      const details = getAccountDetails(accountId);
      const volatileDetails = getVolatileAccountDetails(accountId);
      accounts.push(summarizeAccount(accountId, details, volatileDetails));
    } catch {
      // Skip stale or partially initialized accounts.
    }
  }
  return accounts;
}

async function waitForUsableAccount(accountId, timeoutSecs) {
  const deadline = Date.now() + Math.max(10, timeoutSecs) * 1000;
  let latest = summarizeAccount(accountId, getAccountDetails(accountId), getVolatileAccountDetails(accountId));
  while (Date.now() < deadline) {
    latest = summarizeAccount(accountId, getAccountDetails(accountId), getVolatileAccountDetails(accountId));
    if (latest.registrationStatus === "REGISTERED" && latest.jamiId) {
      return latest;
    }
    await sleep(1200);
  }
  return latest;
}

async function ensureAgentAccount(options) {
  const accounts = await loadExistingAccounts();
  let account = chooseBestExistingAccount(accounts, options.accountId, options.displayName);
  let created = false;

  if (!account) {
    const displayName = preferredDisplayName(options.displayName);
    const details = {
      "Account.type": "RING",
      "Account.displayName": displayName,
      "Account.alias": displayName,
    };
    const createdAccountId = parseSingleStringTuple(runGdbusCall("addAccount", [buildStringDict(details)]));
    if (!createdAccountId) {
      throw new Error("Jami did not return a new account id.");
    }
    created = true;
    try {
      runGdbusCall("sendRegister", [createdAccountId, "true"]);
    } catch {
      // The daemon often registers automatically; this is only a best-effort nudge.
    }
    account = await waitForUsableAccount(createdAccountId, options.timeoutSecs);
  } else {
    account = await waitForUsableAccount(account.accountId, options.timeoutSecs);
  }

  if (!account.jamiId) {
    throw new Error(`Jami account ${account.accountId} is missing a shareable Jami ID.`);
  }

  const shareValue = account.jamiId;
  const shareLabel = looksLikeJamiId(shareValue) ? "Jami ID" : "Jami Username";

  return {
    ok: true,
    created,
    account_id: account.accountId,
    share_value: shareValue,
    share_label: shareLabel,
    registration_status: account.registrationStatus || "UNKNOWN",
    display_name: account.displayName || preferredDisplayName(options.displayName),
    device_name: account.deviceName || hostnameLabel(),
    account_active: account.active,
    dbus_env_file: process.env.CTO_JAMI_DBUS_ENV_FILE || "",
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(
      "Usage: jami_contact_qr_cli.mjs [ensure-contact] [--display-name <name>] [--account-id <id>] [--timeout-secs <secs>]\n"
    );
    return;
  }
  if (args.command !== "ensure-contact") {
    fail(`Unsupported command: ${args.command}`);
    return;
  }

  const dbusEnvFile = loadJamiDbusEnvironment();
  if (!dbusEnvFile && !process.env.DBUS_SESSION_BUS_ADDRESS) {
    fail("No Jami DBus session environment is available.");
    return;
  }

  const result = await ensureAgentAccount(args);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((error) => {
  fail(error instanceof Error ? error.message : String(error));
});
