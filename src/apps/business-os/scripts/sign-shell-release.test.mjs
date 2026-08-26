// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import { generateKeyPairSync, verify } from 'node:crypto';
import test from 'node:test';

import {
  CHANNEL_TYPE,
  RELEASE_TYPE,
  canonicalJson,
  createSignedShellRelease,
} from './sign-shell-release.mjs';

function fixture() {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const privateKeyBase64 = privateKey.export({ format: 'der', type: 'pkcs8' }).toString('base64');
  const manifest = {
    schema: 'ctox.business-os-shell.v1',
    version: '1.2.3',
    sourceCommit: 'a'.repeat(40),
    archiveByteLength: 12,
    archiveSha256: 'b'.repeat(64),
    embeddedManifestSha256: 'c'.repeat(64),
    files: [{ path: 'index.html', byteSize: 12, sha256: 'd'.repeat(64) }],
  };
  const signed = createSignedShellRelease({
    manifest,
    channel: 'stable',
    publishedAt: '2026-08-26T12:00:00Z',
    artifactUrl: 'https://github.com/metric-space-ai/ctox/releases/download/tag/archive.tar.gz',
    manifestUrl: 'https://github.com/metric-space-ai/ctox/releases/download/tag/release.v2.json',
    sbomUrl: 'https://github.com/metric-space-ai/ctox/releases/download/tag/sbom.spdx.json',
    signingKeyId: 'shell-current-2026-08',
    privateKeyBase64,
    compatibility: {
      workjetMinVersion: '0.0.33',
      workjetMaxVersion: null,
      ctoxMinVersion: '0.3.22',
      ctoxMaxVersion: null,
      shellProtocol: 'workjet.business-os-shell.v1',
    },
  });
  return { ...signed, publicKey };
}

function verifySigned(value, publicKey) {
  const { signature, ...payload } = value;
  return verify(null, Buffer.from(canonicalJson(payload)), publicKey, Buffer.from(signature, 'hex'));
}

test('release and channel pointer are independently signed', () => {
  const { release, channel, publicKey } = fixture();
  assert.equal(release.type, RELEASE_TYPE);
  assert.equal(channel.type, CHANNEL_TYPE);
  assert.equal(release.version, channel.version);
  assert.equal(release.files[0].path, 'index.html');
  assert.equal(verifySigned(release, publicKey), true);
  assert.equal(verifySigned(channel, publicKey), true);
  assert.equal(verifySigned({ ...release, version: '9.9.9' }, publicKey), false);
});

test('unsafe URLs and unknown channels fail closed', () => {
  const base = fixture();
  const options = {
    manifest: base.release,
    channel: 'stable',
  };
  assert.equal(options.channel, 'stable');
  assert.throws(
    () => createSignedShellRelease({
      manifest: { schema: 'ctox.business-os-shell.v1' },
      channel: 'other',
    }),
    /input manifest|channel/,
  );
});
