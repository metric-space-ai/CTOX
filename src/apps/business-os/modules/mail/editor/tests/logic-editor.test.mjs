import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  createMailLogicDefinition,
  createMailLogicGroup,
  createMailLogicRule,
  blockAtEasyEmailPath,
  evaluateMailLogic,
  evaluateRule,
  insertMailLogicNode,
  moveMailLogicNode,
  normalizeMailLogicDefinition,
  readBlockMailLogic,
  removeMailLogicNode,
  setMailLogicTestData,
  updateMailLogicNode,
  writeBlockMailLogic,
} from '../logic-editor-v1.mjs';

test('evaluates every supported operator with typed values', () => {
  const data = {
    contact: { name: 'Ada Lovelace', age: 36, active: true, note: '', tags: ['vip', 'sales'] },
  };
  const cases = [
    [{ field: 'contact.name', operator: 'equals', valueType: 'string', value: 'Ada Lovelace' }, true],
    [{ field: 'contact.name', operator: 'not-equals', valueType: 'string', value: 'Grace' }, true],
    [{ field: 'contact.name', operator: 'contains', valueType: 'string', value: 'Love' }, true],
    [{ field: 'contact.tags', operator: 'not-contains', valueType: 'string', value: 'blocked' }, true],
    [{ field: 'contact.active', operator: 'exists' }, true],
    [{ field: 'contact.note', operator: 'empty' }, true],
    [{ field: 'contact.age', operator: 'greater', valueType: 'number', value: 35 }, true],
    [{ field: 'contact.age', operator: 'less', valueType: 'number', value: 40 }, true],
    [{ field: 'contact.active', operator: 'equals', valueType: 'boolean', value: true }, true],
    [{ field: 'contact.missing', operator: 'equals', valueType: 'null', value: null }, true],
  ];
  for (const [rule, expected] of cases) assert.equal(evaluateRule(rule, data), expected, rule.operator);
});

test('evaluates nested AND/OR groups and returns per-node results', () => {
  const definition = createMailLogicDefinition({
    root: createMailLogicGroup({
      id: 'root',
      combinator: 'and',
      children: [
        createMailLogicRule({ id: 'country', field: 'contact.country', operator: 'equals', value: 'DE' }),
        createMailLogicGroup({
          id: 'intent',
          combinator: 'or',
          children: [
            createMailLogicRule({ id: 'hot', field: 'contact.score', operator: 'greater', valueType: 'number', value: 80 }),
            createMailLogicRule({ id: 'vip', field: 'contact.tags', operator: 'contains', value: 'vip' }),
          ],
        }),
      ],
    }),
  });
  const result = evaluateMailLogic(definition, { contact: { country: 'DE', score: 42, tags: ['vip'] } });
  assert.equal(result.matched, true);
  assert.equal(result.results.hot, false);
  assert.equal(result.results.vip, true);
  assert.equal(result.results.intent, true);
  assert.equal(result.results.root, true);
});

test('creates, edits, reorders, nests, and deletes rules immutably', () => {
  const initial = createMailLogicDefinition({ root: createMailLogicGroup({ id: 'root' }) });
  const withFirst = insertMailLogicNode(initial, 'root', createMailLogicRule({ id: 'first', field: 'a' }));
  const withSecond = insertMailLogicNode(withFirst, 'root', createMailLogicRule({ id: 'second', field: 'b' }));
  const withGroup = insertMailLogicNode(withSecond, 'root', createMailLogicGroup({ id: 'nested', combinator: 'or' }), 1);
  const nested = insertMailLogicNode(withGroup, 'nested', createMailLogicRule({ id: 'inside', field: 'c' }));
  const edited = updateMailLogicNode(nested, 'inside', { operator: 'exists' });
  const moved = moveMailLogicNode(edited, 'second', -1);
  const removed = removeMailLogicNode(moved, 'first');

  assert.deepEqual(initial.root.children, []);
  assert.equal(edited.root.children[1].children[0].operator, 'exists');
  assert.equal(moved.root.children[1].id, 'second');
  assert.deepEqual(removed.root.children.map((node) => node.id), ['second', 'nested']);
});

test('normalization repairs duplicate ids and invalid enum values', () => {
  const normalized = normalizeMailLogicDefinition({
    root: {
      id: 'same', kind: 'group', combinator: 'xor', children: [
        { id: 'same', kind: 'rule', field: 'x', operator: 'wat', valueType: 'date', value: 2 },
      ],
    },
  });
  assert.equal(normalized.root.combinator, 'and');
  assert.notEqual(normalized.root.id, normalized.root.children[0].id);
  assert.equal(normalized.root.children[0].operator, 'equals');
  assert.equal(normalized.root.children[0].valueType, 'number');
});

test('persists logic and live test data inside the selected versioned source block', () => {
  const document = {
    version: 1,
    content: {
      id: 'page', type: 'page', data: { value: {} }, children: [{
        id: 'text-1', type: 'text', data: { value: { content: 'Hallo' } }, children: [],
      }],
    },
  };
  let definition = createMailLogicDefinition({ root: createMailLogicGroup({ id: 'root' }) });
  definition = insertMailLogicNode(definition, 'root', createMailLogicRule({
    field: 'contact.segment', operator: 'equals', value: 'kunde',
  }));
  definition = setMailLogicTestData(definition, { contact: { segment: 'kunde' } });
  const written = writeBlockMailLogic(document, 'text-1', definition);

  assert.equal(document.content.children[0].data.value.logic, undefined);
  assert.equal(readBlockMailLogic(written, 'text-1').testData.contact.segment, 'kunde');
  assert.equal(evaluateMailLogic(readBlockMailLogic(written, 'text-1')).matched, true);
});

test('reads and writes canonical Easy Email idx paths when upstream blocks have no id', async () => {
  // Shape follows upstream Page/Section/Column/Text create() output and the
  // `content.children.N` path emitted by useFocusIdx().
  const upstreamTemplate = JSON.parse(await readFile(
    new URL('./fixtures/easy-email-upstream-template.json', import.meta.url),
    'utf8',
  ));
  const idx = 'content.children.0.children[0].children.0';
  assert.equal(blockAtEasyEmailPath(upstreamTemplate, idx).type, 'text');
  let logic = createMailLogicDefinition({ root: createMailLogicGroup({ id: 'root' }) });
  logic = insertMailLogicNode(logic, 'root', createMailLogicRule({
    field: 'contact.locale', operator: 'equals', value: 'de',
  }));
  const written = writeBlockMailLogic(upstreamTemplate, idx, logic);
  assert.equal(readBlockMailLogic(written, idx).root.children[0].field, 'contact.locale');
  assert.equal(upstreamTemplate.content.children[0].children[0].children[0].data.value.logic, undefined);
  assert.equal(blockAtEasyEmailPath(upstreamTemplate, 'content.__proto__.x'), null);
});
