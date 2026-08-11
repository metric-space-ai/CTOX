// Real-turn smoke: drives the built sidecar through one NON-faux turn against a
// REAL model, verifying the whole pipeline end-to-end — sidecar -> pi-ai provider
// -> real LLM -> real tool call -> applied to the projected source. This is the
// same path a live CTOX-gateway turn takes. The model must point at an
// owner-controlled loopback gateway; direct provider endpoints and provider
// keys in the sidecar environment are intentionally unsupported.
//
// SKIPS by default. Run it only with the public model descriptor issued by the
// Rust owner for a live, capability-protected turn. For the main CTOX gateway,
// use api "openai-responses", provider "ctox-gateway", loopback baseUrl and the
// active gateway model. Coding-plan live receipts must use the native owner so
// it can issue the per-turn capability; this script cannot manufacture one.
import assert from "node:assert/strict";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const modelEnv = process.env.CTOX_PI_REAL_MODEL;
if (!modelEnv) {
  console.log(
    "SKIP: set CTOX_PI_REAL_MODEL to an owner-issued CTOX loopback model descriptor",
  );
  process.exit(0);
}
const model = JSON.parse(modelEnv);

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, "..", "dist", "ctox-pi-sidecar.mjs");
const dir = mkdtempSync(path.join(os.tmpdir(), "ctox-pi-real-"));
const socketPath = path.join(dir, "sidecar.sock");

// NON-faux daemon; inherit env so the provider picks up its API key.
const child = spawn(process.execPath, [dist, socketPath], {
  env: process.env,
  stdio: ["ignore", "ignore", "inherit"],
});
const cleanup = () => {
  try {
    child.kill("SIGKILL");
  } catch {
    /* ignore */
  }
  rmSync(dir, { recursive: true, force: true });
};

const waitForSocket = async (timeoutMs = 15000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(socketPath)) return;
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error("daemon socket did not appear");
};

try {
  await waitForSocket();
  const response = await new Promise((resolve, reject) => {
    const client = net.createConnection(socketPath, () => {
      client.write(
        `${JSON.stringify({
          id: "real-1",
          prompt:
            "In index.js, change the exported value of v from 1 to 2. Use the edit tool to make the change, then you are done.",
          files: { "index.js": "export const v = 1;\n" },
          maxAssistantTurns: 6,
          model,
        })}\n`,
      );
    });
    let buffer = "";
    client.on("data", (chunk) => {
      buffer += chunk.toString("utf8");
      const nl = buffer.indexOf("\n");
      if (nl >= 0) {
        client.end();
        try {
          resolve(JSON.parse(buffer.slice(0, nl)));
        } catch (error) {
          reject(error);
        }
      }
    });
    client.on("error", reject);
    setTimeout(() => reject(new Error("real turn timed out")), 120000);
  });

  assert.equal(response.ok, true, `real turn ok (error: ${response.error})`);
  const index = (response.snapshot || []).find((e) => String(e.path).endsWith("index.js"));
  assert.ok(
    index && String(index.content).includes("v = 2"),
    "the real LLM edited the projected source (v=1 -> v=2)",
  );
  console.log("REAL TURN OK — real LLM drove the full pipeline end-to-end");
} finally {
  cleanup();
}
