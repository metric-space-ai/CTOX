import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { generateInstallManifest, INSTALL_MANIFEST_SCHEMA } from "./generate-install-manifest.mjs";

const filenames = [
  "ctox-macos-arm64.tar.gz",
  "ctox-macos-x64.tar.gz",
  "ctox-linux-arm64.tar.gz",
  "ctox-linux-x64.tar.gz",
  "ctox-windows-x64.zip",
];

test("emits a deterministic five-platform CTOX install manifest", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "ctox-install-manifest-"));
  try {
    for (const filename of filenames) await writeFile(path.join(root, filename), filename);
    const output = path.join(root, "ctox-install-manifest-v1.json");
    const manifest = await generateInstallManifest({
      artifactsDir: root,
      tag: "v1.2.3",
      repository: "metric-space-ai/ctox",
      output,
    });

    assert.equal(manifest.schema, INSTALL_MANIFEST_SCHEMA);
    assert.equal(manifest.release, "v1.2.3");
    assert.equal(manifest.artifacts.length, 5);
    assert.deepEqual(
      manifest.artifacts.map(({ platform, arch }) => `${platform}/${arch}`),
      ["macos/arm64", "macos/x64", "linux/arm64", "linux/x64", "windows/x64"],
    );
    assert.ok(manifest.artifacts.every(({ sha256 }) => /^[a-f0-9]{64}$/.test(sha256)));
    assert.equal(manifest.compatibility.businessOsDataPlane, "rxdb-webrtc");
    assert.equal(manifest.compatibility.httpDataBridge, false);
    assert.deepEqual(JSON.parse(await readFile(output, "utf8")), manifest);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("fails closed when a required target artifact is absent", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "ctox-install-manifest-missing-"));
  try {
    await assert.rejects(
      generateInstallManifest({
        artifactsDir: root,
        tag: "v1.2.3",
        repository: "metric-space-ai/ctox",
        output: path.join(root, "manifest.json"),
      }),
      /ENOENT/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
