#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { buildPackManifest } from "../shared/office-pack.mjs";

const mobileRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(mobileRoot, "../../..");
const sourceRoot = path.join(repoRoot, "src/apps/business-os");
const outputRoot = path.join(mobileRoot, "dist");
const baseRoot = path.join(outputRoot, "base/business-os");
const packRoot = path.join(outputRoot, "office-pack/vendor/ctox-office");
const packageJson = JSON.parse(await fs.readFile(path.join(mobileRoot, "package.json"), "utf8"));
const gitRevision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" }).trim();

async function sourceTreeDigest(directory, relative = "") {
  const hash = createHash("sha256");
  for (const entry of await fs.readdir(directory, { withFileTypes: true }).then((items) => items.sort((a, b) => a.name.localeCompare(b.name)))) {
    const childRelative = path.posix.join(relative, entry.name);
    const child = path.join(directory, entry.name);
    if (entry.isDirectory()) hash.update(await sourceTreeDigest(child, childRelative));
    else if (entry.isFile()) {
      const name = Buffer.from(childRelative);
      const data = await fs.readFile(child);
      hash.update(Buffer.from(String(name.length))); hash.update("\\0"); hash.update(name);
      hash.update(Buffer.from(String(data.length))); hash.update("\\0"); hash.update(data);
    }
  }
  return hash.digest();
}

const sourceRevision = `git:${gitRevision}:sha256:${(await sourceTreeDigest(sourceRoot)).toString("hex")}`;

async function copyTree(source, destination, shouldExclude = () => false) {
  await fs.mkdir(destination, { recursive: true });
  for (const entry of await fs.readdir(source, { withFileTypes: true })) {
    const relative = path.relative(sourceRoot, path.join(source, entry.name)).split(path.sep).join("/");
    if (shouldExclude(relative)) continue;
    const from = path.join(source, entry.name);
    const to = path.join(destination, entry.name);
    if (entry.isDirectory()) await copyTree(from, to, shouldExclude);
    else if (entry.isFile()) await fs.copyFile(from, to);
  }
}

await fs.chmod(baseRoot, 0o755).catch(() => {});
await fs.chmod(packRoot, 0o755).catch(() => {});
await fs.rm(outputRoot, { recursive: true, force: true });
await copyTree(sourceRoot, baseRoot, (relative) => relative === "vendor/ctox-office" || relative.startsWith("vendor/ctox-office/"));
await copyTree(path.join(sourceRoot, "vendor/ctox-office"), packRoot);
const officeManifest = await buildPackManifest(packRoot, { sourceRevision, appVersion: packageJson.version });
const baseMetadata = {
  format: "ctox.mobile.shell-base.v1",
  source_revision: sourceRevision,
  app_version: packageJson.version,
  entry: "business-os/index.html",
  excluded_pack: "ctox-office",
};
await fs.writeFile(path.join(outputRoot, "base-manifest.json"), `${JSON.stringify(baseMetadata, null, 2)}\n`);
await fs.writeFile(path.join(outputRoot, "office-pack-manifest.json"), `${JSON.stringify(officeManifest, null, 2)}\n`);
async function markReadOnly(directory) {
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) await markReadOnly(target);
    else if (entry.isFile()) await fs.chmod(target, 0o444);
  }
}
await markReadOnly(baseRoot);
await markReadOnly(packRoot);
process.stdout.write(`Staged Business OS mobile shell revision ${sourceRevision.slice(0, 12)}.\n`);
