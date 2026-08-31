#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { verifyPack } from "../shared/office-pack.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const base = JSON.parse(await fs.readFile(path.join(dist, "base-manifest.json"), "utf8"));
const pack = JSON.parse(await fs.readFile(path.join(dist, "office-pack-manifest.json"), "utf8"));
if (base.source_revision !== pack.source_revision || base.app_version !== pack.app_version) throw new Error("base and office pack versions differ");
await fs.access(path.join(dist, "base/business-os/index.html"));
try {
  await fs.access(path.join(dist, "base/business-os/vendor/ctox-office"));
  throw new Error("base shell contains ctox-office");
} catch (error) {
  if (error.message === "base shell contains ctox-office") throw error;
}
await verifyPack(path.join(dist, "office-pack/vendor/ctox-office"), pack, { sourceRevision: base.source_revision, appVersion: base.app_version });
process.stdout.write(`Verified ${pack.files.length} office pack files and base exclusion.\n`);
