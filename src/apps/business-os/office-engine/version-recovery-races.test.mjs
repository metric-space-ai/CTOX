import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { build } from '../node_modules/esbuild/lib/main.js';

const base = fileURLToPath(new URL('../modules', import.meta.url));
for (const [module, kind, records] of [
  ['documents', 'ctox-documents', 'documents'],
  ['spreadsheets', 'ctox-spreadsheets', 'spreadsheets'],
]) {
  // Test-only export: production source and its public API remain untouched.
  const source = await readFile(`${base}/${module}/index.js`, 'utf8');
  const bundle = await build({
    stdin: { contents: `${source}\nexport { loadSelectedVersion as testLoadSelectedVersion };`,
      resolveDir: `${base}/${module}`, sourcefile: 'index.js', loader: 'js' },
    bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent',
  });
  const { testLoadSelectedVersion: load } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
  for (const change of ['edit', 'saving', 'selection', 'new-version']) {
    test(`${module}: reconnect completion preserves concurrent ${change}`, async () => {
      let ready;
      let started;
      const recovering = new Promise(resolve => { started = resolve; });
      const peer = new Promise(resolve => { ready = resolve; });
      let reads = 0;
      let directStarts = 0;
      const original = { id: 'old' };
      const handle = { kind, recordId: 'file', activity: 0, saving: false };
      const state = {
        [records]: [{ id: 'file', current_version_id: 'v1' }],
        selectedId: 'file', selectedVersion: original, editorHandle: handle,
        dirty: false, saving: false,
        ctx: {
          db: { collection() { return {
            findOne() { return { async exec() {
              reads += 1;
              if (reads === 1) throw new Error('query_cancelled');
              return { toJSON: () => ({ id: 'v1' }) };
            } }; },
            find() { assert.fail('exact metadata record exists'); },
          }; } },
          sync: { async startCollection(name, options) {
            if (!options?.forceDirect) return { mode: 'follower' };
            directStarts += 1;
            assert.equal(name, module === 'documents' ? 'document_versions' : 'spreadsheet_versions');
            started();
            return { ready: peer };
          } },
        },
      };
      const pending = load(state);
      await recovering;
      if (change === 'edit') { state.dirty = true; handle.activity += 1; }
      if (change === 'saving') { handle.saving = true; state.saving = true; }
      if (change === 'selection') { state.selectedId = 'other'; state.selectedVersion = { id: 'other-v1' }; }
      if (change === 'new-version') state.selectedVersion = { id: 'v2' };
      const expected = { selectedId: state.selectedId, selectedVersion: state.selectedVersion,
        dirty: state.dirty, saving: state.saving, handleSaving: handle.saving };
      ready({ state: {
        async waitForOpenPeerId() { return 'native'; },
        async awaitInSync() { assert.fail('no full collection replication'); },
      } });
      assert.equal(await pending, null);
      assert.deepEqual({ selectedId: state.selectedId, selectedVersion: state.selectedVersion,
        dirty: state.dirty, saving: state.saving, handleSaving: handle.saving }, expected);
      assert.equal(state.selectedVersion, expected.selectedVersion);
      assert.equal(directStarts, 1);
    });
  }
}
