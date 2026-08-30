// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import test from 'node:test';

import { createShellSbom } from './build-shell-sbom.mjs';

test('SPDX inventory mirrors the immutable shell manifest', () => {
  const sbom = createShellSbom({
    schema: 'ctox.business-os-shell.v1',
    version: '1.2.3',
    sourceCommit: 'a'.repeat(40),
    embeddedManifestSha256: 'b'.repeat(64),
    files: [{ path: 'index.html', byteSize: 12, sha256: 'c'.repeat(64) }],
  }, 'https://github.com/metric-space-ai/ctox/releases/download/business-os-shell-v1.2.3/sbom');
  assert.equal(sbom.spdxVersion, 'SPDX-2.3');
  assert.equal(sbom.files[0].fileName, './index.html');
  assert.equal(sbom.files[0].checksums[0].checksumValue, 'c'.repeat(64));
  assert.equal(sbom.relationships[0].relatedSpdxElement, 'SPDXRef-File-1');
});

test('empty inventories and invalid namespaces fail closed', () => {
  assert.throws(() => createShellSbom({ schema: 'ctox.business-os-shell.v1', files: [] }, 'https://example.test/sbom'), /empty/);
  assert.throws(() => createShellSbom({ schema: 'ctox.business-os-shell.v1', files: [{}] }, 'not a url'), /Invalid URL/);
});
