#!/usr/bin/env node

import * as childProcess from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

const SERVICE = "cx.ring.Ring";
const OBJECT_PATH = "/cx/ring/Ring/ConfigurationManager";
const INTERFACE = "cx.ring.Ring.ConfigurationManager";
const DEFAULT_TIMEOUT_SECS = 900;

function nowIso() {
  return new Date().toISOString();
}

function emit(event) {
  fs.writeSync(process.stdout.fd, `${JSON.stringify({ updated_at: nowIso(), ...event })}\n`);
}

function fail(message, code = 1) {
  emit({ kind: "error", message });
  process.exitCode = code;
  process.exit();
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

function parseArgs(argv) {
  const args = { timeoutSecs: DEFAULT_TIMEOUT_SECS, displayName: "" };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--timeout-secs":
        args.timeoutSecs = Number.parseInt(argv[i + 1] || "", 10) || DEFAULT_TIMEOUT_SECS;
        i += 1;
        break;
      case "--display-name":
        args.displayName = String(argv[i + 1] || "");
        i += 1;
        break;
      case "--help":
      case "-h":
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
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

function parseSingleStringTuple(output) {
  const match = output.match(/\(\s*'([^']*)'\s*,?\s*\)/);
  if (!match) {
    throw new Error(`Could not parse DBus string result: ${output}`);
  }
  return match[1];
}

function decodeDbusString(value) {
  return String(value)
    .replace(/\\'/g, "'")
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\");
}

function parseCompactSignalLine(line) {
  const match = line.match(
    /deviceAuthStateChanged\s+\('((?:[^'\\]|\\.)*)',\s*(-?\d+),\s*(\{.*\})\)\s*$/
  );
  if (!match) {
    return null;
  }
  return {
    accountId: decodeDbusString(match[1]),
    stateCode: Number.parseInt(match[2], 10),
    details: parseCompactDetails(match[3]),
  };
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

function parseSignalBlock(lines) {
  let accountId = null;
  let stateCode = null;
  const details = {};
  let inDict = false;
  let pendingKey = null;

  for (const rawLine of lines.slice(1)) {
    const line = rawLine.trim();
    const intMatch = line.match(/^(?:int32|uint32)\s+(-?\d+)/);
    if (intMatch && stateCode == null) {
      stateCode = Number.parseInt(intMatch[1], 10);
      continue;
    }
    if (line.startsWith("dict entry(")) {
      inDict = true;
      pendingKey = null;
      continue;
    }
    if (line === ")") {
      inDict = false;
      pendingKey = null;
      continue;
    }
    const stringMatch = line.match(/^string\s+"((?:[^"\\]|\\.)*)"$/);
    if (!stringMatch) continue;
    const value = decodeDbusString(stringMatch[1]);
    if (inDict) {
      if (pendingKey == null) {
        pendingKey = value;
      } else {
        details[pendingKey] = value;
        pendingKey = null;
      }
    } else if (accountId == null) {
      accountId = value;
    }
  }

  return { accountId, stateCode, details };
}

function stateLabel(stateCode) {
  switch (stateCode) {
    case 0:
      return "init";
    case 1:
      return "token_available";
    case 2:
      return "connecting";
    case 3:
      return "authenticating";
    case 4:
      return "in_progress";
    case 5:
      return "done";
    default:
      return "unknown";
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(
      "Usage: jami_device_link_cli.mjs [--display-name <name>] [--timeout-secs <secs>]\n"
    );
    return;
  }

  const jamiDbusEnvFile = loadJamiDbusEnvironment();
  emit({
    kind: "log",
    level: "info",
    message: jamiDbusEnvFile
      ? `Loaded Jami DBus environment from ${jamiDbusEnvFile}.`
      : "No dedicated Jami DBus environment file found. Using current session bus.",
  });

  const details = {
    "Account.type": "RING",
    "Account.archiveURL": "jami-auth",
  };
  if (args.displayName.trim()) {
    details["Account.displayName"] = args.displayName.trim();
    details["Account.alias"] = args.displayName.trim();
  }

  const monitor = childProcess.spawn(
    "gdbus",
    ["monitor", "--session", "--dest", SERVICE, "--object-path", OBJECT_PATH],
    {
      stdio: ["ignore", "pipe", "pipe"],
    }
  );

  let activeBlock = [];
  const pendingSignals = [];
  let monitorReady = false;
  let settled = false;
  let timeoutHandle = null;
  let accountId = "";

  let resolveMonitorReady;
  const monitorReadyPromise = new Promise((resolve) => {
    resolveMonitorReady = resolve;
  });

  function markMonitorReady() {
    if (monitorReady) return;
    monitorReady = true;
    resolveMonitorReady();
  }

  function shutdown(code = 0) {
    if (settled) return;
    settled = true;
    if (timeoutHandle) clearTimeout(timeoutHandle);
    monitor.kill("SIGTERM");
    setTimeout(() => {
      process.exit(code);
    }, 10).unref();
  }

  function handleParsedSignal(parsed) {
    if (!parsed || !parsed.accountId) return;
    if (!accountId) {
      pendingSignals.push(parsed);
      return;
    }
    if (parsed.accountId !== accountId) return;
    if (parsed.stateCode == null) {
      emit({ kind: "log", level: "warn", message: "Ignored Jami deviceAuthStateChanged signal without a state code." });
      return;
    }
    const label = stateLabel(parsed.stateCode);
    emit({
      kind: "state_changed",
      account_id: accountId,
      state_code: parsed.stateCode,
      state_label: label,
      details: parsed.details,
    });
    if (parsed.stateCode === 5) {
      const error = parsed.details.error || "";
      emit({
        kind: "completed",
        account_id: accountId,
        success: error === "" || error === "none",
        error,
      });
      shutdown(error === "" || error === "none" ? 0 : 1);
    }
  }

  function handleSignal(lines) {
    if (!lines.length) return;
    if (!lines[0].includes("member deviceAuthStateChanged")) return;
    handleParsedSignal(parseSignalBlock(lines));
  }

  const rl = readline.createInterface({ input: monitor.stdout });
  rl.on("line", (line) => {
    const trimmed = String(line || "").trim();
    if (trimmed.startsWith("Monitoring signals on object") || trimmed.startsWith("The name ")) {
      markMonitorReady();
      return;
    }
    if (trimmed.includes(".deviceAuthStateChanged")) {
      handleParsedSignal(parseCompactSignalLine(trimmed));
      return;
    }
    if (/^\/.+: interface /.test(line)) {
      if (activeBlock.length) handleSignal(activeBlock);
      activeBlock = [line];
      return;
    }
    if (!activeBlock.length) return;
    activeBlock.push(line);
  });
  rl.on("close", () => {
    if (activeBlock.length) handleSignal(activeBlock);
    if (!settled) {
      emit({ kind: "error", message: "Jami DBus monitor closed unexpectedly." });
      shutdown(1);
    }
  });

  const stderrRl = readline.createInterface({ input: monitor.stderr });
  stderrRl.on("line", (line) => {
    const trimmed = String(line || "").trim();
    if (trimmed) {
      emit({ kind: "log", level: "stderr", message: trimmed });
    }
  });

  try {
    await Promise.race([
      monitorReadyPromise,
      new Promise((resolve) => setTimeout(resolve, 800)),
    ]);
    accountId = parseSingleStringTuple(runGdbusCall("addAccount", [buildStringDict(details)]));
  } catch (error) {
    fail(`Failed to start Jami device link session: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }

  emit({ kind: "session_started", account_id: accountId });
  while (pendingSignals.length) {
    handleParsedSignal(pendingSignals.shift());
  }

  const stdinRl = readline.createInterface({ input: process.stdin });
  stdinRl.on("line", (line) => {
    if (settled) return;
    let command = null;
    try {
      command = JSON.parse(line);
    } catch {
      emit({ kind: "log", level: "warn", message: "Ignored non-JSON command on stdin." });
      return;
    }
    if (!command || typeof command !== "object") return;

    if (command.cmd === "cancel") {
      try {
        runGdbusCall("removeAccount", [accountId]);
      } catch (error) {
        emit({
          kind: "log",
          level: "warn",
          message: `Failed to cancel pending Jami link session: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
      emit({ kind: "cancelled", account_id: accountId });
      shutdown(0);
      return;
    }

    if (command.cmd === "provide_auth") {
      const password = String(command.password || "");
      if (!password) {
        emit({ kind: "log", level: "warn", message: "Jami link password command was empty." });
        return;
      }
      const authScheme = String(command.scheme || "password");
      try {
        runGdbusCall("provideAccountAuthentication", [accountId, password, authScheme]);
      } catch (firstError) {
        emit({
          kind: "error",
          message: `Failed to submit Jami link password: ${
            firstError instanceof Error ? firstError.message : String(firstError)
          }`,
        });
      }
    }
  });

  timeoutHandle = setTimeout(() => {
    emit({ kind: "error", message: "Timed out while waiting for Jami device linking to complete." });
    shutdown(1);
  }, Math.max(30, args.timeoutSecs) * 1000);
  timeoutHandle.unref();
}

main().catch((error) => {
  fail(error instanceof Error ? error.message : String(error));
});
