import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash, generateKeyPairSync, verify } from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const SCRIPT_PATH = fileURLToPath(new URL('./build-appstore-index.mjs', import.meta.url));
const MODULES_DIR = fileURLToPath(new URL('../modules/', import.meta.url));
const TIMESTAMP = '2026-08-31T00:00:00Z';

function fixture(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ctox-appstore-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const keyPath = path.join(root, 'private-key.pem');
  writeFileSync(keyPath, privateKey.export({ format: 'pem', type: 'pkcs8' }));
  return { root, keyPath, publicKey };
}

function runBuild(out, { keyPath, unsigned = false, timestamp = TIMESTAMP } = {}) {
  const args = [SCRIPT_PATH, '--out', out, '--timestamp', timestamp];
  if (unsigned) args.push('--unsigned');
  else args.push('--key', keyPath);
  const result = spawnSync(process.execPath, args, {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  assert.equal(
    result.status,
    0,
    `publisher failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
  );
  return result;
}

function snapshotTree(directory, relative = '') {
  const snapshot = new Map();
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
    const absolutePath = path.join(directory, entry.name);
    const relativePath = relative ? `${relative}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      for (const [name, bytes] of snapshotTree(absolutePath, relativePath)) {
        snapshot.set(name, bytes);
      }
    } else if (entry.isFile()) {
      snapshot.set(relativePath, readFileSync(absolutePath));
    } else {
      assert.fail(`unexpected output entry type: ${absolutePath}`);
    }
  }
  return snapshot;
}

function assertTreesEqual(leftDirectory, rightDirectory) {
  const left = snapshotTree(leftDirectory);
  const right = snapshotTree(rightDirectory);
  assert.deepEqual([...left.keys()], [...right.keys()]);
  for (const [relativePath, leftBytes] of left) {
    assert.deepEqual(leftBytes, right.get(relativePath), `bytes differ: ${relativePath}`);
  }
}

function readBase64Signature(signaturePath) {
  return Buffer.from(readFileSync(signaturePath, 'utf8').trim(), 'base64');
}

function findEndOfCentralDirectory(zip) {
  const minimumOffset = Math.max(0, zip.length - 22 - 0xffff);
  for (let offset = zip.length - 22; offset >= minimumOffset; offset -= 1) {
    if (zip.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  throw new Error('ZIP end of central directory not found');
}

function readStoredZip(zipPath) {
  const zip = readFileSync(zipPath);
  const eocdOffset = findEndOfCentralDirectory(zip);
  const entryCount = zip.readUInt16LE(eocdOffset + 10);
  let centralOffset = zip.readUInt32LE(eocdOffset + 16);
  const entries = new Map();

  for (let index = 0; index < entryCount; index += 1) {
    assert.equal(zip.readUInt32LE(centralOffset), 0x02014b50, 'invalid central directory header');
    const method = zip.readUInt16LE(centralOffset + 10);
    const compressedSize = zip.readUInt32LE(centralOffset + 20);
    const uncompressedSize = zip.readUInt32LE(centralOffset + 24);
    const nameLength = zip.readUInt16LE(centralOffset + 28);
    const extraLength = zip.readUInt16LE(centralOffset + 30);
    const commentLength = zip.readUInt16LE(centralOffset + 32);
    const localOffset = zip.readUInt32LE(centralOffset + 42);
    const name = zip.subarray(centralOffset + 46, centralOffset + 46 + nameLength).toString('utf8');

    assert.equal(method, 0, `${name} is not stored`);
    assert.equal(compressedSize, uncompressedSize, `${name} has different stored sizes`);
    assert.equal(extraLength, 0, `${name} has a central extra field`);
    assert.equal(zip.readUInt32LE(localOffset), 0x04034b50, `invalid local header for ${name}`);
    const localNameLength = zip.readUInt16LE(localOffset + 26);
    const localExtraLength = zip.readUInt16LE(localOffset + 28);
    assert.equal(localExtraLength, 0, `${name} has a local extra field`);
    const dataOffset = localOffset + 30 + localNameLength + localExtraLength;
    entries.set(name, zip.subarray(dataOffset, dataOffset + compressedSize));

    centralOffset += 46 + nameLength + extraLength + commentLength;
  }

  return entries;
}

test('full run publishes the 18 real store apps with matching hashes', (t) => {
  const { root, keyPath } = fixture(t);
  const out = path.join(root, 'dist');
  runBuild(out, { keyPath });

  const indexPath = path.join(out, 'index.json');
  const index = JSON.parse(readFileSync(indexPath, 'utf8'));
  assert.equal(index.schema, 'ctox.appstore.index.v1');
  assert.equal(index.generated_at, TIMESTAMP);
  assert.equal(index.apps.length, 18);
  assert.deepEqual(index.apps.map((app) => app.id), [...index.apps.map((app) => app.id)].sort());
  assert.ok(existsSync(`${indexPath}.sig`));

  for (const app of index.apps) {
    const iconPath = path.join(out, app.icon);
    const bundlePath = path.join(out, app.bundle);
    assert.ok(existsSync(iconPath), `missing icon: ${app.icon}`);
    assert.ok(existsSync(bundlePath), `missing bundle: ${app.bundle}`);
    assert.ok(existsSync(`${bundlePath}.sig`), `missing bundle signature: ${app.bundle}.sig`);
    const bundleBytes = readFileSync(bundlePath);
    assert.equal(bundleBytes.length, app.bundle_size, `bundle size mismatch: ${app.id}`);
    assert.equal(
      createHash('sha256').update(bundleBytes).digest('hex'),
      app.bundle_sha256,
      `bundle hash mismatch: ${app.id}`,
    );
  }
});

test('two runs with the same key and timestamp are byte-identical', (t) => {
  const { root, keyPath } = fixture(t);
  const first = path.join(root, 'first');
  const second = path.join(root, 'second');
  runBuild(first, { keyPath });
  runBuild(second, { keyPath });
  assertTreesEqual(first, second);
});

test('index and bundle signatures verify with the generated Ed25519 key', (t) => {
  const { root, keyPath, publicKey } = fixture(t);
  const out = path.join(root, 'dist');
  runBuild(out, { keyPath });

  const indexPath = path.join(out, 'index.json');
  const indexBytes = readFileSync(indexPath);
  const index = JSON.parse(indexBytes);
  const publicDer = publicKey.export({ format: 'der', type: 'spki' });
  assert.equal(
    index.signing_key_id,
    createHash('sha256').update(publicDer).digest('hex').slice(0, 16),
  );
  assert.equal(
    verify(null, indexBytes, publicKey, readBase64Signature(`${indexPath}.sig`)),
    true,
  );

  const app = index.apps[0];
  const bundlePath = path.join(out, app.bundle);
  const bundleBytes = readFileSync(bundlePath);
  assert.equal(
    verify(null, bundleBytes, publicKey, readBase64Signature(`${bundlePath}.sig`)),
    true,
  );
});

test('bundle central directory contains the exact on-disk module manifest', (t) => {
  const { root, keyPath } = fixture(t);
  const out = path.join(root, 'dist');
  runBuild(out, { keyPath });

  const index = JSON.parse(readFileSync(path.join(out, 'index.json'), 'utf8'));
  const app = index.apps[0];
  const zipEntries = readStoredZip(path.join(out, app.bundle));
  const manifestEntry = `${app.id}/module.json`;
  assert.ok(zipEntries.has(manifestEntry), `missing ZIP entry: ${manifestEntry}`);
  assert.deepEqual(
    zipEntries.get(manifestEntry),
    readFileSync(path.join(MODULES_DIR, app.id, 'module.json')),
  );
});

test('--unsigned emits no signatures and uses a null signing key id', (t) => {
  const { root } = fixture(t);
  const out = path.join(root, 'dist');
  runBuild(out, { unsigned: true });

  const index = JSON.parse(readFileSync(path.join(out, 'index.json'), 'utf8'));
  assert.equal(index.signing_key_id, null);
  for (const relativePath of snapshotTree(out).keys()) {
    assert.equal(relativePath.endsWith('.sig'), false, `unexpected signature: ${relativePath}`);
  }
});
