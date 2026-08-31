import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { describe, it } from 'node:test';

const explorerUrl = new URL('./index.js', import.meta.url);
const explorerSource = await readFile(explorerUrl, 'utf8');
const explorerCss = await readFile(new URL('./index.css', import.meta.url), 'utf8');
const explorerHtml = await readFile(new URL('./index.html', import.meta.url), 'utf8');
const manifest = JSON.parse(await readFile(new URL('./module.json', import.meta.url), 'utf8'));
const collectionSchemas = JSON.parse(await readFile(new URL('./collections.schema.json', import.meta.url), 'utf8'));
const bundledSource = explorerSource;
const { __explorerTestHooks: explorer } = await import(explorerUrl);

describe('Explorer module contract', () => {
  it('uses the Business OS module entry contract and a fragment-only view', () => {
    assert.match(explorerSource, /export async function mount\(ctx\)/);
    assert.match(explorerSource, /const container = ctx\.host/);
    assert.doesNotMatch(explorerSource, /export const manifest/);
    assert.doesNotMatch(explorerSource, /desktop-apps\//);
    assert.match(explorerHtml, /^<main class="ctox-workspace app-explorer"/);
    assert.doesNotMatch(explorerHtml, /<!doctype|<(?:html|head|script|style)\b/i);
  });

  it('declares the full core system-module manifest and source collections', () => {
    assert.equal(manifest.id, 'explorer');
    assert.equal(manifest.title, 'Explorer');
    assert.equal(manifest.install_scope, 'core');
    assert.equal(manifest.default_installed, true);
    assert.equal(manifest.launch_kind, 'desktop-app');
    assert.equal(manifest.layout.shell_contract, 'v2');
    assert.equal(manifest.layout.shell_geometry_contract, 'business-os-v2-global-1');
    assert.equal(manifest.store.source_path, 'modules/explorer');
    assert.deepEqual(manifest.collections, [
      'desktop_file_chunks', 'desktop_files', 'documents', 'knowledge_items',
      'matching_objects', 'outbound_companies', 'spreadsheets',
    ]);
    assert.deepEqual(Object.keys(collectionSchemas.collections), manifest.collections);
  });

  it('uses only shell-sanctioned container breakpoints', () => {
    assert.match(explorerCss, /@container business-app-window \(max-width: 1024px\)/);
    assert.match(explorerCss, /@container business-app-window \(max-width: 768px\)/);
    assert.doesNotMatch(explorerCss, /@media\s*\([^)]*(?:max|min)-width/);
  });
});

describe('Explorer helpers', () => {
  it('keeps existing records visible unless they are explicitly deleted', () => {
    const rows = explorer.normalizeRowsForSource([
      { id: 'doc-1', title: 'Proposal', updated_at_ms: 1000 },
      { id: 'doc-2', title: 'Deleted', is_deleted: true, updated_at_ms: 2000 },
      { id: 'doc-3', title: 'Legacy', is_deleted: undefined, updated_at_ms: 3000 },
    ], explorer.SOURCES.find((source) => source.id === 'documents'));

    assert.deepEqual(rows.map((row) => row.label), ['Proposal', 'Legacy']);
  });

  it('filters filesystem rows by the current folder and preserves folders', () => {
    const rows = explorer.normalizeRowsForSource([
      { id: 'fs_root', parent_id: '', path: '/', name: 'Files', kind: 'folder' },
      { id: 'folder-a', parent_id: 'fs_root', path: '/A', name: 'A', kind: 'folder' },
      { id: 'file-a', parent_id: 'folder-a', path: '/A/a.txt', name: 'a.txt', kind: 'file', size_bytes: 12 },
      { id: 'file-root', parent_id: 'fs_root', path: '/root.txt', name: 'root.txt', kind: 'file', size_bytes: 20 },
    ], explorer.FILE_SOURCE, 'fs_root');

    assert.deepEqual(rows.map((row) => [row.label, row.isFolder]), [
      ['A', true],
      ['root.txt', false],
    ]);
  });

  it('keeps the CTOX-published file area visible at the Files root', async () => {
    const upserts = [];
    const db = {
      collection(name) {
        if (name !== 'desktop_files') return null;
        return {
          findOne: () => ({ exec: async () => null }),
          upsert: async (doc) => { upserts.push(doc); },
        };
      },
    };

    await explorer.ensureFileSystem(db);

    assert.ok(upserts.some((doc) => (
      doc.id === 'fs_ctox'
      && doc.parent_id === 'fs_root'
      && doc.path === '/CTOX'
      && doc.name === 'CTOX'
    )));
  });

  it('waits for desktop file replication before seeding and rendering Files', () => {
    const replicationStart = explorerSource.indexOf("startCollection?.('desktop_files')");
    const seedStart = explorerSource.indexOf('await ensureFileSystem(ctx.db)');
    const renderStart = explorerSource.indexOf('await selectSource(FILE_SOURCE)');

    assert.ok(replicationStart >= 0, 'Files must explicitly start desktop_files replication');
    assert.ok(replicationStart < seedStart, 'native file metadata must arrive before browser seeds');
    assert.ok(seedStart < renderStart, 'the populated collection must be rendered last');
  });

  it('replicates every selected source through its declared collection', () => {
    assert.match(explorerSource, /const collectionId = sourceCollectionId\(state\.activeSource\)[\s\S]*?startCollection\?\.\(collectionId\)/);
    for (const collection of ['documents', 'spreadsheets', 'knowledge_items', 'matching_objects', 'outbound_companies']) {
      assert.ok(manifest.collections.includes(collection));
      assert.ok(collectionSchemas.collections[collection]);
    }
  });

  it('offers discoverable recent-file views and explicit sort choices', () => {
    assert.match(explorerSource, /get label\(\) \{ return T\.recentCreated; \}[\s\S]*?recentSort: 'created'/);
    assert.match(explorerSource, /get label\(\) \{ return T\.recentModified; \}[\s\S]*?recentSort: 'modified'/);
    assert.match(explorerHtml, /data-explorer-sort/);
    assert.match(explorerHtml, /<option value="created"/);
    assert.match(explorerSource, /activeData\.filter\(\(item\) => item\.kind !== 'folder'\)\.map\(normalizeFileRow\)/);
    assert.match(explorerSource, /requireRevision: `files-search:/);
    assert.match(explorerSource, /\{ name: \{ \$regex: pattern \} \}/);
  });

  it('keeps the legacy schema-registration capability guarded', () => {
    assert.match(explorerSource, /ctx\.ensureModuleData\?\.\(source\.moduleId\)/);
    assert.ok(manifest.collections.includes('documents'));
  });

  it('validates new folder and rename inputs before persistence', () => {
    const existing = new Set(['reports']);

    assert.equal(explorer.validateEntryName('', existing), 'Name ist erforderlich.');
    assert.equal(explorer.validateEntryName('../x', existing), 'Name darf keine Schrägstriche enthalten.');
    assert.equal(explorer.validateEntryName('reports', existing), 'Name existiert bereits in diesem Ordner.');
    assert.equal(explorer.validateEntryName('Quarterly Reports', existing), '');
  });

  it('creates deterministic unique names for uploads', () => {
    assert.equal(explorer.uniqueName('Report.pdf', ['Report.pdf']), 'Report 2.pdf');
    assert.equal(explorer.uniqueName('Report.pdf', ['Report.pdf', 'Report 2.pdf']), 'Report 3.pdf');
  });

  it('keeps grid rows inside the visible explorer main column', () => {
    assert.match(explorerCss, /\.app-explorer-grid \{[\s\S]*?min-width: 0;/);
    assert.match(explorerCss, /\.app-explorer-row \{[\s\S]*?min-width: 0;/);
  });

  it('offers a direct Files download action independent of the file viewer', () => {
    assert.match(bundledSource, /data-preview-download/);
    assert.match(bundledSource, /anchor\.download\s*=\s*row\.label/);
    assert.match(bundledSource, /Herunterladen/);
    assert.match(bundledSource, /Download fehlgeschlagen/);
    assert.match(bundledSource, /reportFileIntegrityError/);
  });

  it('supports bidirectional desktop file drag without an HTTP data path', () => {
    assert.equal(explorer.dataTransferContainsFiles({ files: [{ name: 'input.csv' }], types: [] }), true);
    assert.equal(explorer.dataTransferContainsFiles({ files: [], types: ['Files'] }), true);
    assert.equal(explorer.dataTransferContainsFiles({ files: [], types: ['text/plain'] }), false);
    assert.equal(explorer.safeDownloadName('../unsafe:report?.csv'), '_unsafe_report_.csv');
    assert.match(explorerSource, /state\.activeSource\.recentSort/);
    assert.match(explorerSource, /setData\('DownloadURL'/);
    assert.match(explorerSource, /ctoxBusinessOsDesktop[\s\S]*?startFileDrag/);
    assert.doesNotMatch(bundledSource, /fetch\([^)]*desktop_files/);
  });

  it('routes office files to their editing apps and keeps media in the viewer', () => {
    assert.equal(explorer.associatedAppFor({ label: 'loads.csv', mimeType: 'text/plain' }), 'spreadsheets');
    assert.equal(explorer.associatedAppFor({ label: 'loads.xlsx', mimeType: 'application/octet-stream' }), 'spreadsheets');
    assert.equal(explorer.associatedAppFor({ label: 'report.docx', mimeType: 'application/octet-stream' }), 'documents');
    assert.equal(explorer.associatedAppFor({ label: 'notes.txt', mimeType: 'text/plain' }), 'documents');
    assert.equal(explorer.associatedAppFor({ label: 'manual.pdf', mimeType: 'application/pdf' }), '');
    assert.equal(explorer.normalizedMimeType({ label: 'loads.csv', mimeType: 'text/plain' }), 'text/csv');
    assert.equal(explorer.mimeFromName('report.docx'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    assert.match(explorerSource, /associatedAppFor\(row\)[\s\S]*?openDesktopApp\(associatedApp/);
  });

  it('stores uploaded file chunks in one bulk write without DataURL materialization', async () => {
    assert.doesNotMatch(bundledSource, /readAsDataURL/);
    const chunkWrites = [];
    const fileWrites = [];
    const db = {
      collection(name) {
        if (name === 'desktop_file_chunks') {
          return {
            bulkUpsert: async (docs) => { chunkWrites.push(docs); },
            upsert: async () => { throw new Error('desktop_file_chunks upsert must not run per chunk'); },
          };
        }
        if (name === 'desktop_files') {
          return { upsert: async (doc) => { fileWrites.push(doc); } };
        }
        return null;
      },
    };
    const bytes = new Uint8Array(24 * 1024);
    bytes.fill(65);
    const file = typeof File === 'function'
      ? new File([bytes], 'bulk.txt', { type: 'text/plain' })
      : { type: 'text/plain', size: bytes.length, arrayBuffer: async () => bytes.buffer };
    await explorer.storeFile(db, 'fs_root', '/', 'bulk.txt', file);

    assert.equal(chunkWrites.length, 1, 'chunks are written through one bulkUpsert call');
    assert.ok(chunkWrites[0].length > 1, 'test payload spans multiple chunk documents');
    assert.equal(fileWrites.length, 1, 'file metadata is written once');
  });
});
