#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const roots = [path.join(root, "shared"), path.join(root, "ios"), path.join(root, "android")];
const sourceExtensions = new Set([".js", ".mjs", ".swift", ".kt", ".kts", ".xml", ".plist"]);
const findings = [];
const ignoredDirectories = new Set([
  ".build",
  ".gradle",
  "build",
  "DerivedData",
  "dist",
  "node_modules",
]);
const rules = [
  ["native business-data endpoint", /\/api\/business-os\/(?!sync\/config|status)[A-Za-z0-9_./-]*/i],
  ["native RxDB HTTP endpoint", /\/rxdb\/(?:pull|push)|\/commands(?:\/|["'])/i],
  ["native business WebSocket bridge", /(?:OkHttpClient|URLSession|HttpURLConnection).{0,200}(?:business[_-]?commands|business[_-]?records|rxdb)/is],
  ["secret in URL", /(?:URL|Uri|URLComponents|loadUrl)[^\n]{0,200}(?:ctox_config|room_password|capability_token|signaling_room_password)/i],
  ["secret logging", /(?:print|logger|Log\.[diewv]|NSLog|console\.)[^\n]{0,160}(?:payload|password|capability|token|packed config|raw link)/i],
  ["secret registry field", /(?:registry|instances?)[^\n]{0,160}(?:signaling_room_password|capability_token|ctox_config)/i],
];

async function walk(directory) {
  for (const entry of await fs.readdir(directory, { withFileTypes: true }).catch(() => [])) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory() && !ignoredDirectories.has(entry.name)) await walk(file);
    else if (sourceExtensions.has(path.extname(file))) {
      const text = await fs.readFile(file, "utf8");
      for (const [label, rule] of rules) if (rule.test(text)) findings.push(`${path.relative(root, file)}: ${label}`);
    }
  }
}
for (const directory of roots) await walk(directory);
if (findings.length) {
  process.stderr.write(`${findings.join("\n")}\n`);
  process.exitCode = 1;
} else process.stdout.write("Mobile native/shared static guard passed.\n");
