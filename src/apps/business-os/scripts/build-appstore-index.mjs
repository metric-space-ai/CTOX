import {
  copyFileSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import {
  createHash,
  createPrivateKey,
  createPublicKey,
  sign,
} from 'node:crypto';
import { fileURLToPath } from 'node:url';

const DEFAULT_TIMESTAMP = '2026-01-01T00:00:00Z';
const INDEX_SCHEMA = 'ctox.appstore.index.v1';
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const BUSINESS_OS_DIR = path.dirname(path.dirname(SCRIPT_PATH));
const MODULES_DIR = path.join(BUSINESS_OS_DIR, 'modules');

const CRC32_TABLE = new Uint32Array(256);
for (let index = 0; index < CRC32_TABLE.length; index += 1) {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = (value & 1) === 1
      ? (value >>> 1) ^ 0xedb88320
      : value >>> 1;
  }
  CRC32_TABLE[index] = value >>> 0;
}

function usageError(message) {
  throw new Error(`${message}\nUsage: node build-appstore-index.mjs --out <dir> [--key <ed25519-private-pem>] [--unsigned] [--timestamp <ISO8601>]`);
}

function parseArguments(argv) {
  const options = {
    out: null,
    key: null,
    unsigned: false,
    timestamp: DEFAULT_TIMESTAMP,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--unsigned') {
      options.unsigned = true;
    } else if (argument === '--out' || argument === '--key' || argument === '--timestamp') {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith('--')) {
        usageError(`Missing value for ${argument}`);
      }
      index += 1;
      if (argument === '--out') options.out = value;
      if (argument === '--key') options.key = value;
      if (argument === '--timestamp') options.timestamp = value;
    } else {
      usageError(`Unknown argument: ${argument}`);
    }
  }

  if (!options.out) usageError('Missing required --out argument');
  if (!options.unsigned && !options.key) {
    usageError('Signed builds require --key');
  }
  if (options.unsigned && options.key) {
    usageError('--key cannot be used with --unsigned');
  }

  return options;
}

function dosTimestamp(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    throw new Error(`Invalid ISO8601 timestamp: ${timestamp}`);
  }
  const year = date.getUTCFullYear();
  if (year < 1980 || year > 2107) {
    throw new Error(`ZIP timestamp year must be between 1980 and 2107: ${timestamp}`);
  }

  return {
    time: (date.getUTCHours() << 11)
      | (date.getUTCMinutes() << 5)
      | Math.floor(date.getUTCSeconds() / 2),
    date: ((year - 1980) << 9)
      | ((date.getUTCMonth() + 1) << 5)
      | date.getUTCDate(),
  };
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = CRC32_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function comparePaths(left, right) {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function collectFiles(directory, relativeDirectory = '') {
  const files = [];
  const names = readdirSync(directory).sort(comparePaths);

  for (const name of names) {
    const absolutePath = path.join(directory, name);
    const relativePath = relativeDirectory
      ? `${relativeDirectory}/${name}`
      : name;
    const stats = statSync(absolutePath, { throwIfNoEntry: true });
    if (stats.isDirectory()) {
      files.push(...collectFiles(absolutePath, relativePath));
    } else if (stats.isFile()) {
      files.push({ absolutePath, relativePath });
    } else {
      throw new Error(`Unsupported module entry: ${absolutePath}`);
    }
  }

  return files;
}

function createStoredZip(moduleDirectory, moduleId, timestamp) {
  const fixedTime = dosTimestamp(timestamp);
  const entries = collectFiles(moduleDirectory)
    .map(({ absolutePath, relativePath }) => {
      const data = readFileSync(absolutePath);
      const name = Buffer.from(`${moduleId}/${relativePath}`, 'utf8');
      if (name.length > 0xffff) {
        throw new Error(`ZIP entry path is too long: ${absolutePath}`);
      }
      if (data.length > 0xffffffff) {
        throw new Error(`ZIP entry is too large without zip64: ${absolutePath}`);
      }
      return { data, name, crc: crc32(data), absolutePath };
    })
    .sort((left, right) => Buffer.compare(left.name, right.name));

  if (entries.length > 0xffff) {
    throw new Error(`Too many ZIP entries without zip64: ${moduleDirectory}`);
  }

  const localParts = [];
  const centralParts = [];
  let localOffset = 0;

  for (const entry of entries) {
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt16LE(0x0800, 6);
    localHeader.writeUInt16LE(0, 8);
    localHeader.writeUInt16LE(fixedTime.time, 10);
    localHeader.writeUInt16LE(fixedTime.date, 12);
    localHeader.writeUInt32LE(entry.crc, 14);
    localHeader.writeUInt32LE(entry.data.length, 18);
    localHeader.writeUInt32LE(entry.data.length, 22);
    localHeader.writeUInt16LE(entry.name.length, 26);
    localHeader.writeUInt16LE(0, 28);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt16LE(0x0800, 8);
    centralHeader.writeUInt16LE(0, 10);
    centralHeader.writeUInt16LE(fixedTime.time, 12);
    centralHeader.writeUInt16LE(fixedTime.date, 14);
    centralHeader.writeUInt32LE(entry.crc, 16);
    centralHeader.writeUInt32LE(entry.data.length, 20);
    centralHeader.writeUInt32LE(entry.data.length, 24);
    centralHeader.writeUInt16LE(entry.name.length, 28);
    centralHeader.writeUInt16LE(0, 30);
    centralHeader.writeUInt16LE(0, 32);
    centralHeader.writeUInt16LE(0, 34);
    centralHeader.writeUInt16LE(0, 36);
    centralHeader.writeUInt32LE(0, 38);
    centralHeader.writeUInt32LE(localOffset, 42);

    localParts.push(localHeader, entry.name, entry.data);
    centralParts.push(centralHeader, entry.name);
    localOffset += localHeader.length + entry.name.length + entry.data.length;
    if (localOffset > 0xffffffff) {
      throw new Error(`ZIP local data is too large without zip64: ${moduleDirectory}`);
    }
  }

  const centralDirectory = Buffer.concat(centralParts);
  if (centralDirectory.length > 0xffffffff) {
    throw new Error(`ZIP central directory is too large without zip64: ${moduleDirectory}`);
  }

  const endOfCentralDirectory = Buffer.alloc(22);
  endOfCentralDirectory.writeUInt32LE(0x06054b50, 0);
  endOfCentralDirectory.writeUInt16LE(0, 4);
  endOfCentralDirectory.writeUInt16LE(0, 6);
  endOfCentralDirectory.writeUInt16LE(entries.length, 8);
  endOfCentralDirectory.writeUInt16LE(entries.length, 10);
  endOfCentralDirectory.writeUInt32LE(centralDirectory.length, 12);
  endOfCentralDirectory.writeUInt32LE(localOffset, 16);
  endOfCentralDirectory.writeUInt16LE(0, 20);

  return Buffer.concat([...localParts, centralDirectory, endOfCentralDirectory]);
}

function readCatalog() {
  const modules = [];
  const seenIds = new Map();
  const directoryNames = readdirSync(MODULES_DIR).sort(comparePaths);

  for (const directoryName of directoryNames) {
    const moduleDirectory = path.join(MODULES_DIR, directoryName);
    if (!statSync(moduleDirectory).isDirectory()) continue;
    const manifestPath = path.join(moduleDirectory, 'module.json');
    let manifestBytes;
    try {
      manifestBytes = readFileSync(manifestPath);
    } catch (error) {
      if (error.code === 'ENOENT') continue;
      throw new Error(`Cannot read manifest ${manifestPath}: ${error.message}`);
    }

    let manifest;
    try {
      manifest = JSON.parse(manifestBytes);
    } catch (error) {
      throw new Error(`Invalid JSON in manifest ${manifestPath}: ${error.message}`);
    }
    if (manifest.install_scope !== 'store') continue;

    for (const field of ['id', 'version', 'entry']) {
      if (typeof manifest[field] !== 'string' || manifest[field].length === 0) {
        throw new Error(`Manifest missing ${field}: ${manifestPath}`);
      }
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(manifest.id)) {
      throw new Error(`Manifest has unsafe id: ${manifestPath}`);
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(manifest.version)) {
      throw new Error(`Manifest has unsafe version: ${manifestPath}`);
    }
    const previousManifest = seenIds.get(manifest.id);
    if (previousManifest) {
      throw new Error(`Duplicate app id ${manifest.id}: ${manifestPath} (already in ${previousManifest})`);
    }
    seenIds.set(manifest.id, manifestPath);
    modules.push({ manifest, manifestPath, moduleDirectory });
  }

  return modules.sort((left, right) => comparePaths(left.manifest.id, right.manifest.id));
}

function loadSigningKey(keyPath) {
  let privateKey;
  try {
    privateKey = createPrivateKey(readFileSync(keyPath));
  } catch (error) {
    throw new Error(`Cannot load Ed25519 private key ${keyPath}: ${error.message}`);
  }
  if (privateKey.asymmetricKeyType !== 'ed25519') {
    throw new Error(`Signing key is not Ed25519: ${keyPath}`);
  }
  const publicDer = createPublicKey(privateKey).export({ format: 'der', type: 'spki' });
  return {
    privateKey,
    keyId: createHash('sha256').update(publicDer).digest('hex').slice(0, 16),
  };
}

function writeSignature(filePath, bytes, privateKey) {
  const signature = sign(null, bytes, privateKey);
  writeFileSync(`${filePath}.sig`, `${signature.toString('base64')}\n`);
}

export function buildAppstore(options) {
  const outputDirectory = path.resolve(options.out);
  if (outputDirectory === path.parse(outputDirectory).root) {
    throw new Error(`Refusing to use filesystem root as --out: ${outputDirectory}`);
  }

  const timestamp = options.timestamp ?? DEFAULT_TIMESTAMP;
  dosTimestamp(timestamp);
  const signing = options.unsigned ? null : loadSigningKey(path.resolve(options.key));
  const catalog = readCatalog();

  rmSync(outputDirectory, { recursive: true, force: true });
  mkdirSync(outputDirectory, { recursive: true });

  const apps = [];
  for (const { manifest, moduleDirectory } of catalog) {
    const appDirectory = path.join(outputDirectory, 'apps', manifest.id);
    mkdirSync(appDirectory, { recursive: true });

    const bundleName = `${manifest.id}-${manifest.version}.zip`;
    const bundlePath = path.join(appDirectory, bundleName);
    const bundleBytes = createStoredZip(moduleDirectory, manifest.id, timestamp);
    writeFileSync(bundlePath, bundleBytes);
    if (signing) writeSignature(bundlePath, bundleBytes, signing.privateKey);

    const iconSource = path.join(moduleDirectory, 'icon.svg');
    const iconDestination = path.join(appDirectory, 'icon.svg');
    try {
      copyFileSync(iconSource, iconDestination);
    } catch (error) {
      if (error.code === 'ENOENT') {
        console.log(`Warning: no icon.svg for ${manifest.id}; skipping icon copy`);
      } else {
        throw new Error(`Cannot copy icon ${iconSource}: ${error.message}`);
      }
    }

    apps.push({
      id: manifest.id,
      title: typeof manifest.title === 'string' ? manifest.title : manifest.id,
      description: typeof manifest.description === 'string' ? manifest.description : '',
      category: typeof manifest.category === 'string' ? manifest.category : '',
      version: manifest.version,
      tags: Array.isArray(manifest.tags) ? manifest.tags : [],
      entry: manifest.entry,
      icon: `apps/${manifest.id}/icon.svg`,
      bundle: `apps/${manifest.id}/${bundleName}`,
      bundle_sha256: createHash('sha256').update(bundleBytes).digest('hex'),
      bundle_size: bundleBytes.length,
    });
  }

  const index = {
    schema: INDEX_SCHEMA,
    generated_at: timestamp,
    signing_key_id: signing?.keyId ?? null,
    apps,
  };
  const indexBytes = Buffer.from(`${JSON.stringify(index, null, 2)}\n`);
  const indexPath = path.join(outputDirectory, 'index.json');
  writeFileSync(indexPath, indexBytes);
  if (signing) writeSignature(indexPath, indexBytes, signing.privateKey);

  console.log(`Built ${apps.length} app bundles in ${outputDirectory}`);
  return index;
}

function main() {
  try {
    buildAppstore(parseArguments(process.argv.slice(2)));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT_PATH) {
  main();
}
