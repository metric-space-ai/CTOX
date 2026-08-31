// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import { generateKeyPairSync, verify } from 'node:crypto';
import test from 'node:test';

import {
  canonicalJson,
  CHANNEL_TYPE,
  createSignedShellRelease,
  RELEASE_TYPE,
} from './sign-shell-release.mjs';

function fixture() {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const privateKeyBase64 = privateKey.export({ format: 'der', type: 'pkcs8' }).toString('base64');
  const manifest = {
    schema: 'ctox.business-os-shell.v1',
    version: '1.2.3',
    sourceCommit: '0123456789abcdef0123456789abcdef01234567',
    archiveByteLength: 123,
    archiveSha256: 'ab'.repeat(32),
    embeddedManifestSha256: 'cd'.repeat(32),
    files: [{ path: 'index.html', byteSize: 7, sha256: 'ef'.repeat(32) }],
  };
  const signed = createSignedShellRelease({
    manifest,
    channel: 'stable',
    publishedAt: '2026-08-26T00:00:00Z',
    artifactUrl: 'https://example.test/shell.tar.gz',
    manifestUrl: 'https://example.test/shell.release.v2.json',
    sbomUrl: 'https://example.test/shell.spdx.json',
    signingKeyId: 'shell-test-current',
    privateKeyBase64,
    compatibility: {
      workjetMinVersion: '0.1.0',
      workjetMaxVersion: null,
      ctoxMinVersion: '0.3.0',
      ctoxMaxVersion: null,
      shellProtocol: 'workjet.business-os-shell.v1',
    },
  });
  return { publicKey, signed };
}

function verifies(publicKey, payload, signature) {
  return verify(
    null,
    Buffer.from(canonicalJson(payload), 'utf8'),
    publicKey,
    Buffer.from(signature, 'hex'),
  );
}

test('migrates the readable v1 producer manifest into a signed immutable v2 release', () => {
  const { publicKey, signed } = fixture();
  assert.equal(signed.release.type, RELEASE_TYPE);
  assert.equal(signed.release.version, '1.2.3');
  assert.equal(signed.release.files[0].size, 7);
  assert.equal(signed.release.artifact.size, 123);
  const { signature: releaseSignature, ...releasePayload } = signed.release;
  assert.equal(verifies(publicKey, releasePayload, releaseSignature), true);

  assert.equal(signed.channel.type, CHANNEL_TYPE);
  const { signature: channelSignature, ...channelPayload } = signed.channel;
  assert.equal(verifies(publicKey, channelPayload, channelSignature), true);
});

test('signatures fail after release or channel metadata is changed', () => {
  const { publicKey, signed } = fixture();
  const { signature: releaseSignature, ...releasePayload } = signed.release;
  const { signature: channelSignature, ...channelPayload } = signed.channel;
  assert.equal(
    verifies(publicKey, { ...releasePayload, version: '1.2.4' }, releaseSignature),
    false,
  );
  assert.equal(
    verifies(publicKey, { ...channelPayload, manifestSha256: '00'.repeat(32) }, channelSignature),
    false,
  );
});

test('rejects non-v1 producers, insecure URLs, empty inventories and unknown channels', () => {
  const { privateKey } = generateKeyPairSync('ed25519');
  const key = privateKey.export({ format: 'der', type: 'pkcs8' }).toString('base64');
  const base = {
    manifest: {
      schema: 'ctox.business-os-shell.v1',
      version: '1.2.3',
      sourceCommit: '0123456789abcdef0123456789abcdef01234567',
      archiveByteLength: 1,
      archiveSha256: 'ab'.repeat(32),
      embeddedManifestSha256: 'cd'.repeat(32),
      files: [{ path: 'index.html', byteSize: 1, sha256: 'ef'.repeat(32) }],
    },
    channel: 'stable',
    publishedAt: '2026-08-26T00:00:00Z',
    artifactUrl: 'https://example.test/shell.tar.gz',
    manifestUrl: 'https://example.test/shell.json',
    sbomUrl: 'https://example.test/sbom.json',
    signingKeyId: 'shell-test',
    privateKeyBase64: key,
    compatibility: {},
  };
  assert.throws(() => createSignedShellRelease({ ...base, channel: 'dev' }), /Unsupported/);
  assert.throws(
    () => createSignedShellRelease({ ...base, artifactUrl: 'http://example.test/shell' }),
    /HTTPS/,
  );
  assert.throws(
    () => createSignedShellRelease({ ...base, manifest: { ...base.manifest, files: [] } }),
    /empty/,
  );
  assert.throws(
    () =>
      createSignedShellRelease({
        ...base,
        manifest: { ...base.manifest, schema: 'ctox.business-os-shell.release.v2' },
      }),
    /v1/,
  );
});
