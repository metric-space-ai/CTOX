// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function requiredString(value, label) {
  const normalized = String(value || '').trim();
  if (!normalized) throw new Error(`${label} is required`);
  return normalized;
}

export function createShellSbom(manifest, documentNamespace) {
  if (!manifest || manifest.schema !== 'ctox.business-os-shell.v1') {
    throw new Error('Expected ctox.business-os-shell.v1 input manifest');
  }
  const namespace = new URL(requiredString(documentNamespace, 'Document namespace')).toString();
  const files = Array.isArray(manifest.files) ? manifest.files : [];
  if (files.length === 0) throw new Error('Shell file inventory is empty');
  return {
    spdxVersion: 'SPDX-2.3',
    dataLicense: 'CC0-1.0',
    SPDXID: 'SPDXRef-DOCUMENT',
    name: `ctox-business-os-shell-${manifest.version}`,
    documentNamespace: namespace,
    creationInfo: { created: '1970-01-01T00:00:00Z', creators: ['Tool: ctox-business-os-shell-release'] },
    packages: [{
      name: 'ctox-business-os-shell',
      SPDXID: 'SPDXRef-Package-Shell',
      versionInfo: manifest.version,
      downloadLocation: 'NOASSERTION',
      filesAnalyzed: true,
      packageVerificationCode: { packageVerificationCodeValue: manifest.embeddedManifestSha256 },
      externalRefs: [{
        referenceCategory: 'OTHER',
        referenceType: 'ctox-source-commit',
        referenceLocator: manifest.sourceCommit,
      }],
    }],
    files: files.map((file, index) => ({
      fileName: `./${file.path}`,
      SPDXID: `SPDXRef-File-${index + 1}`,
      checksums: [{ algorithm: 'SHA256', checksumValue: file.sha256 }],
    })),
    relationships: files.map((_, index) => ({
      spdxElementId: 'SPDXRef-Package-Shell',
      relationshipType: 'CONTAINS',
      relatedSpdxElement: `SPDXRef-File-${index + 1}`,
    })),
  };
}

async function main(argv) {
  if (argv.length !== 6 || argv[0] !== '--manifest' || argv[2] !== '--output' || argv[4] !== '--namespace') {
    throw new Error('Usage: build-shell-sbom --manifest <path> --output <path> --namespace <https-url>');
  }
  const manifest = JSON.parse(await readFile(resolve(argv[1]), 'utf8'));
  const sbom = createShellSbom(manifest, argv[5]);
  await writeFile(resolve(argv[3]), `${JSON.stringify(sbom, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`build-shell-sbom: ${error?.message || error}\n`);
    process.exitCode = 1;
  });
}
