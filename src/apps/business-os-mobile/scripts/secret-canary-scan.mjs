#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fixtures = JSON.parse(await fs.readFile(path.join(root, "fixtures/invites.json"), "utf8"));
const canaries = [
  fixtures.valid.signaling_room_password,
  fixtures.valid.session.capability_token,
].map((value) => Buffer.from(value));
const outputRoots = [
  path.join(root, "dist"),
  path.join(root, "ios/DerivedData"),
  path.join(root, "android/app/build"),
];
const findings = [];

async function scan(target) {
  const stat = await fs.stat(target).catch(() => null);
  if (!stat) return;
  if (stat.isDirectory()) {
    for (const entry of await fs.readdir(target)) await scan(path.join(target, entry));
    return;
  }
  if (!stat.isFile()) return;
  const bytes = await fs.readFile(target);
  if (canaries.some((canary) => bytes.includes(canary))) findings.push(path.relative(root, target));
}

for (const target of outputRoots) await scan(target);
if (findings.length) {
  process.stderr.write(`Synthetic credential canary reached generated output:\n${findings.join("\n")}\n`);
  process.exitCode = 1;
} else {
  process.stdout.write("Generated outputs contain no synthetic credential canaries.\n");
}
