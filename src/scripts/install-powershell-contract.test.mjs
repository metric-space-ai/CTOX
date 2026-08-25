import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const installer = await readFile(new URL("../../install.ps1", import.meta.url), "utf8");
const releaseWorkflow = await readFile(
  new URL("../../.github/workflows/release.yml", import.meta.url),
  "utf8",
);

test("Windows installer is admin-gated and validates the official release", () => {
  assert.match(installer, /Assert-Administrator/);
  assert.match(installer, /ctox\.install-manifest\.v1/);
  assert.match(installer, /metric-space-ai\/ctox/);
  assert.match(installer, /Scheme -ne "https"/);
  assert.match(installer, /Host -ne "github\.com"/);
  assert.match(installer, /Get-FileHash.+SHA256/);
  assert.match(installer, /archive SHA-256 mismatch/);
});

test("Windows installer activates an atomic release and owns the service lifecycle", () => {
  assert.match(installer, /current\.next/);
  assert.match(installer, /ItemType Junction/);
  assert.match(installer, /service install --root/);
  assert.match(installer, /New-ItemProperty.+Services\\CTOX/s);
  assert.match(installer, /& \$ctox start/);
  assert.match(installer, /& \$ctox status \| ConvertFrom-Json/);
  assert.doesNotMatch(installer, /\buninstall\b/i);
});

test("Windows installer emits sanitized machine-readable lifecycle events", () => {
  assert.match(installer, /ctox\.install-event\.v1/);
  for (const phase of ["preflight", "download", "verify", "install", "service", "complete"]) {
    assert.match(installer, new RegExp(`-Phase "${phase}"`));
  }
  assert.doesNotMatch(installer, /Read-Host|Get-Credential|ConvertTo-SecureString/);
});

test("Windows release bundle is checked against the binary bundle contract", () => {
  assert.match(releaseWorkflow, /Package runtime bundle \(Windows\)/);
  assert.match(releaseWorkflow, /contracts\\binary_bundle_manifest\.txt/);
  assert.match(releaseWorkflow, /bundle missing required path/);
  assert.match(releaseWorkflow, /src\\apps\\business-os/);
  assert.match(releaseWorkflow, /install\.ps1/);
});
