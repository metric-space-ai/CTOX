import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { PACK_FORMAT, PACK_ID } from "./constants.mjs";

export async function sha256File(file) {
  const bytes = await fs.readFile(file);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

export async function buildPackManifest(root, { sourceRevision, appVersion }) {
  const files = [];
  async function walk(relative = "") {
    const dir = path.join(root, relative);
    for (const entry of (await fs.readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const child = path.posix.join(relative.split(path.sep).join(path.posix.sep), entry.name);
      const absolute = path.join(root, child);
      if (entry.isDirectory()) await walk(child);
      else if (entry.isFile()) {
        const stat = await fs.stat(absolute);
        files.push({ path: child, size: stat.size, sha256: await sha256File(absolute) });
      }
    }
  }
  await walk();
  return {
    format: PACK_FORMAT,
    pack_id: PACK_ID,
    source_revision: sourceRevision,
    app_version: appVersion,
    total_bytes: files.reduce((sum, file) => sum + file.size, 0),
    files,
  };
}

export async function verifyPack(root, manifest, { sourceRevision, appVersion, signal, onProgress = () => {} } = {}) {
  if (manifest?.format !== PACK_FORMAT || manifest?.pack_id !== PACK_ID) throw new Error("unsupported office pack manifest");
  if (manifest.source_revision !== sourceRevision) throw new Error("office pack revision mismatch");
  if (manifest.app_version !== appVersion) throw new Error("office pack app version mismatch");
  let verified = 0;
  for (const file of manifest.files || []) {
    if (signal?.aborted) throw new Error("office pack activation canceled");
    if (!file.path || path.isAbsolute(file.path) || file.path.includes("..")) throw new Error("office pack path is invalid");
    const absolute = path.resolve(root, file.path);
    const canonicalRoot = path.resolve(root);
    if (!absolute.startsWith(`${canonicalRoot}${path.sep}`)) throw new Error("office pack path escapes root");
    const stat = await fs.stat(absolute).catch(() => null);
    if (!stat?.isFile() || stat.size !== file.size) throw new Error(`office pack file size mismatch: ${file.path}`);
    if (await sha256File(absolute) !== file.sha256) throw new Error(`office pack hash mismatch: ${file.path}`);
    verified += file.size;
    onProgress(manifest.total_bytes ? verified / manifest.total_bytes : 1);
  }
  if (verified !== manifest.total_bytes) throw new Error("office pack total byte mismatch");
  return true;
}
