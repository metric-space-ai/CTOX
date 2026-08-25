#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

export const INSTALL_MANIFEST_SCHEMA = "ctox.install-manifest.v1";

const TARGETS = [
  ["macos", "arm64", "ctox-macos-arm64.tar.gz"],
  ["macos", "x64", "ctox-macos-x64.tar.gz"],
  ["linux", "arm64", "ctox-linux-arm64.tar.gz"],
  ["linux", "x64", "ctox-linux-x64.tar.gz"],
  ["windows", "x64", "ctox-windows-x64.zip"],
];

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const [key, inline] = token.split("=", 2);
    const value = inline ?? argv[index + 1];
    if (inline === undefined) index += 1;
    values.set(key, value);
  }
  return values;
}

async function sha256(filePath) {
  return createHash("sha256").update(await readFile(filePath)).digest("hex");
}

export async function generateInstallManifest({ artifactsDir, tag, repository, output }) {
  if (!artifactsDir || !tag || !repository || !output) {
    throw new Error("artifactsDir, tag, repository and output are required");
  }
  const releaseBase = `https://github.com/${repository}/releases/download/${encodeURIComponent(tag)}`;
  const artifacts = [];
  for (const [platform, arch, filename] of TARGETS) {
    const filePath = path.join(artifactsDir, filename);
    artifacts.push({
      platform,
      arch,
      filename,
      url: `${releaseBase}/${filename}`,
      sha256: await sha256(filePath),
    });
  }
  const manifest = {
    schema: INSTALL_MANIFEST_SCHEMA,
    version: 1,
    release: tag,
    channel: "stable",
    repository,
    compatibility: {
      workjetHostProvisioning: 1,
      businessOsInvite: 1,
      businessOsDataPlane: "rxdb-webrtc",
      httpDataBridge: false,
    },
    artifacts,
  };
  await writeFile(output, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return manifest;
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const args = parseArgs(process.argv.slice(2));
  await generateInstallManifest({
    artifactsDir: args.get("--artifacts-dir"),
    tag: args.get("--tag"),
    repository: args.get("--repository"),
    output: args.get("--output"),
  });
}
