import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  buildDocumentsDeepLink,
  normalizeDocumentArtifact,
  normalizeEditorMode,
  openDocumentsArtifact,
  validateEasyEmailHandle,
  validateEasyEmailRuntime,
} from '../mail-content-editor.mjs';

test('normalizes the two supported editor modes', () => {
  assert.equal(normalizeEditorMode(), 'rich-text');
  assert.equal(normalizeEditorMode('word'), 'rich-text');
  assert.equal(normalizeEditorMode('easy-email'), 'html');
  assert.throws(() => normalizeEditorMode('markdown'), /Unsupported mail editor mode/);
});

test('normalizes Documents references without taking ownership of document bytes', () => {
  const artifact = normalizeDocumentArtifact({
    document_id: 'doc_sales_1',
    version_id: 'doc_sales_1_v3',
    label: 'Sales follow-up',
  });
  assert.deepEqual(artifact, {
    documentId: 'doc_sales_1',
    versionId: 'doc_sales_1_v3',
    title: 'Sales follow-up',
    deepLink: '#documents?record=doc_sales_1&record_id=doc_sales_1&version=doc_sales_1_v3&version_id=doc_sales_1_v3',
  });
  assert.throws(() => normalizeDocumentArtifact({ title: 'Missing id' }), /requires documentId/);
});

test('Documents deep links carry current and canonical record aliases', () => {
  assert.equal(buildDocumentsDeepLink({}), '#documents');
  assert.equal(
    buildDocumentsDeepLink({ documentId: 'doc / 4', versionId: 'v#1' }),
    '#documents?record=doc+%2F+4&record_id=doc+%2F+4&version=v%231&version_id=v%231',
  );
});

test('opens a Word artifact through the shell desktop launcher', async () => {
  const calls = [];
  const result = await openDocumentsArtifact({
    openDesktopApp: async (...args) => {
      calls.push(args);
      return 'window-documents-1';
    },
  }, { documentId: 'doc_9', versionId: 'doc_9_v2' });

  assert.equal(result, 'window-documents-1');
  assert.deepEqual(calls, [[
    'documents',
    {
      args: {
        record: 'doc_9',
        record_id: 'doc_9',
        documentId: 'doc_9',
        version: 'doc_9_v2',
        version_id: 'doc_9_v2',
        versionId: 'doc_9_v2',
      },
    },
  ]]);
});

test('fails closed when the Business OS launcher is unavailable', async () => {
  await assert.rejects(
    openDocumentsArtifact({}, { documentId: 'doc_9' }),
    (error) => error.code === 'documents_launcher_unavailable',
  );
});

test('validates the expected local Easy Email API', () => {
  const factory = () => null;
  assert.equal(validateEasyEmailRuntime({ createEasyEmailEditor: factory }), factory);
  assert.throws(() => validateEasyEmailRuntime({}), /must export createEasyEmailEditor/);

  const handle = {
    getDocument() {},
    getHtml() {},
    setDocument() {},
    getSelectedBlockId() {},
    onSelectionChange() {},
    setMergeTags() {},
    focus() {},
    destroy() {},
  };
  assert.equal(validateEasyEmailHandle(handle), handle);
  assert.throws(() => validateEasyEmailHandle({}), /missing getDocument/);
});

test('the vendored Easy Email port exposes the adapter factory', async () => {
  const runtime = await import('../../../../vendor/easy-email-editor/index.mjs');
  assert.equal(validateEasyEmailRuntime(runtime), runtime.createEasyEmailEditor);
});

test('keeps the HTML canvas central and optional panels in one right drawer', async () => {
  const source = await readFile(new URL('../mail-content-editor.mjs', import.meta.url), 'utf8');
  const css = await readFile(new URL('../editor.css', import.meta.url), 'utf8');

  assert.match(source, /panelHosts.*blocks.*design.*logic.*source/s);
  assert.match(source, /commandHost\.replaceChildren\(commandbar\)/);
  assert.match(source, /requestPanel:\s*\(name\)\s*=>\s*openHtmlPanel\(name,\s*\{\s*fromRuntime:\s*true\s*\}\)/);
  assert.match(source, /panelOptions\.fromRuntime\s*!==\s*true/);
  assert.match(source, /drawer\.hidden = true/);
  assert.match(source, /htmlPanel\.append\(htmlHost, drawerBackdrop, drawer\)/);
  assert.match(css, /\.mail-content-editor-html-host[\s\S]*position:\s*absolute;[\s\S]*inset:\s*0;/);
  assert.match(css, /\.mail-content-editor-drawer[\s\S]*position:\s*absolute;[\s\S]*inset-inline-end:\s*0;/);
  assert.doesNotMatch(css, /grid-template-columns:\s*[^;]*240px[^;]*1fr[^;]*280px/);
});
