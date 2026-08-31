// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import { createHash, createPrivateKey, sign } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HEX_64 = /^[0-9a-f]{64}$/;
const KEY_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const CHANNELS = new Set(['stable', 'beta', 'nightly']);
const HTTPS_URL = /^https:\/\//;

export const RELEASE_TYPE = 'ctox.business-os-shell.release.v2';
export const CHANNEL_TYPE = 'ctox.business-os-shell.channel.v1';

export function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

export function canonicalJson(value) {
  return JSON.stringify(value);
}

function safeHttpsUrl(value, label) {
  const raw = String(value || '');
  if (!HTTPS_URL.test(raw)) throw new Error(`${label} must be HTTPS`);
  const parsed = new URL(raw);
  if (parsed.username || parsed.password || parsed.hash) {
    throw new Error(`${label} must not contain credentials or a fragment`);
  }
  return parsed.toString();
}

function signingKey(privateKeyBase64) {
  const bytes = Buffer.from(String(privateKeyBase64 || ''), 'base64');
  if (bytes.length < 32 || bytes.length > 4096) throw new Error('Signing key is missing or invalid');
  return createPrivateKey({ key: bytes, format: 'der', type: 'pkcs8' });
}

function signatureFor(payload, key) {
  return sign(null, Buffer.from(canonicalJson(payload), 'utf8'), key).toString('hex');
}

export function createSignedShellRelease({
  manifest,
  channel,
  publishedAt,
  artifactUrl,
  manifestUrl,
  sbomUrl,
  signingKeyId,
  privateKeyBase64,
  compatibility,
}) {
  if (!manifest || manifest.schema !== 'ctox.business-os-shell.v1') {
    throw new Error('Expected ctox.business-os-shell.v1 input manifest');
  }
  if (!CHANNELS.has(channel)) throw new Error('Unsupported shell release channel');
  if (!KEY_ID.test(signingKeyId)) throw new Error('Invalid signing key id');
  if (!Number.isSafeInteger(manifest.archiveByteLength) || manifest.archiveByteLength < 1) {
    throw new Error('Invalid archive byte length');
  }
  if (!HEX_64.test(String(manifest.archiveSha256 || ''))) throw new Error('Invalid archive SHA-256');
  if (!Array.isArray(manifest.files) || manifest.files.length < 1) throw new Error('Shell file inventory is empty');
  const key = signingKey(privateKeyBase64);
  const artifact = {
    url: safeHttpsUrl(artifactUrl, 'Artifact URL'),
    size: manifest.archiveByteLength,
    sha256: manifest.archiveSha256,
    contentType: 'application/gzip',
  };
  const releasePayload = {
    type: RELEASE_TYPE,
    version: manifest.version,
    channel,
    sourceCommit: manifest.sourceCommit,
    publishedAt,
    artifact,
    compatibility,
    files: manifest.files.map((file) => ({
      path: file.path,
      size: file.byteSize,
      sha256: file.sha256,
    })),
    provenance: {
      embeddedManifestSha256: manifest.embeddedManifestSha256,
      sbomUrl: safeHttpsUrl(sbomUrl, 'SBOM URL'),
    },
    signingKeyId,
  };
  const release = { ...releasePayload, signature: signatureFor(releasePayload, key) };
  const releaseBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
  const channelPayload = {
    type: CHANNEL_TYPE,
    channel,
    version: manifest.version,
    manifestUrl: safeHttpsUrl(manifestUrl, 'Manifest URL'),
    manifestSha256: sha256(releaseBytes),
    publishedAt,
    signingKeyId,
  };
  return {
    release,
    releaseBytes,
    channel: { ...channelPayload, signature: signatureFor(channelPayload, key) },
  };
}

function parseArgs(argv) {
  const allowed = new Set([
    '--manifest', '--output-dir', '--channel', '--published-at', '--artifact-url',
    '--manifest-url', '--sbom-url', '--signing-key-id', '--private-key-base64',
    '--workjet-min', '--ctox-min',
  ]);
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!allowed.has(flag) || value === undefined || allowed.has(value)) {
      throw new Error(`Invalid argument: ${flag || '<missing>'}`);
    }
    if (values.has(flag)) throw new Error(`Duplicate argument: ${flag}`);
    values.set(flag, value);
  }
  for (const flag of allowed) if (!values.has(flag)) throw new Error(`Missing argument: ${flag}`);
  return Object.fromEntries([...values].map(([key, value]) => [key.slice(2), value]));
}

async function main(argv) {
  const args = parseArgs(argv);
  const manifest = JSON.parse(await readFile(resolve(args.manifest), 'utf8'));
  const signed = createSignedShellRelease({
    manifest,
    channel: args.channel,
    publishedAt: args['published-at'],
    artifactUrl: args['artifact-url'],
    manifestUrl: args['manifest-url'],
    sbomUrl: args['sbom-url'],
    signingKeyId: args['signing-key-id'],
    privateKeyBase64: args['private-key-base64'],
    compatibility: {
      workjetMinVersion: args['workjet-min'],
      workjetMaxVersion: null,
      ctoxMinVersion: args['ctox-min'],
      ctoxMaxVersion: null,
      shellProtocol: 'workjet.business-os-shell.v1',
    },
  });
  const outputDir = resolve(args['output-dir']);
  await mkdir(outputDir, { recursive: true });
  const releaseName = `ctox-business-os-shell-${manifest.version}.release.v2.json`;
  const channelName = `business-os-shell-${args.channel}.json`;
  await writeFile(resolve(outputDir, releaseName), signed.releaseBytes, { flag: 'wx', mode: 0o600 });
  await writeFile(resolve(outputDir, channelName), `${JSON.stringify(signed.channel, null, 2)}\n`, {
    flag: 'wx',
    mode: 0o600,
  });
  process.stdout.write(`${JSON.stringify({ releaseName, channelName, source: basename(args.manifest) })}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`sign-shell-release: ${error?.message || error}\n`);
    process.exitCode = 1;
  });
}
