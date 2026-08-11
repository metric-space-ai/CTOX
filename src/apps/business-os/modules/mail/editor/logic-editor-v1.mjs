export const MAIL_LOGIC_VERSION = 1;

export const MAIL_LOGIC_OPERATORS = Object.freeze([
  'equals',
  'not-equals',
  'contains',
  'not-contains',
  'exists',
  'empty',
  'greater',
  'less',
]);

export const MAIL_LOGIC_VALUE_TYPES = Object.freeze(['string', 'number', 'boolean', 'null']);

const OPERATOR_SET = new Set(MAIL_LOGIC_OPERATORS);
const VALUE_TYPE_SET = new Set(MAIL_LOGIC_VALUE_TYPES);
const COMBINATOR_SET = new Set(['and', 'or']);
const VALUELESS_OPERATORS = new Set(['exists', 'empty']);

const LABELS = Object.freeze({
  de: Object.freeze({
    title: 'Logik',
    noBlock: 'Wähle im Layout einen Inhaltsblock aus.',
    applies: 'Block wird angezeigt',
    hidden: 'Block wird für diese Testdaten ausgeblendet',
    addRule: 'Regel hinzufügen',
    addGroup: 'Gruppe hinzufügen',
    delete: 'Löschen',
    moveUp: 'Nach oben',
    moveDown: 'Nach unten',
    all: 'Alle Regeln (UND)',
    any: 'Mindestens eine Regel (ODER)',
    field: 'Feld / Merge-Tag',
    operator: 'Operator',
    type: 'Werttyp',
    value: 'Wert',
    testData: 'Live-Testdaten (JSON)',
    invalidJson: 'Testdaten müssen gültiges JSON-Objekt sein.',
    equals: 'ist gleich',
    'not-equals': 'ist nicht gleich',
    contains: 'enthält',
    'not-contains': 'enthält nicht',
    exists: 'ist vorhanden',
    empty: 'ist leer',
    greater: 'ist größer als',
    less: 'ist kleiner als',
    string: 'Text',
    number: 'Zahl',
    boolean: 'Ja/Nein',
    null: 'Null',
    true: 'Ja',
    false: 'Nein',
    matches: 'trifft zu',
    misses: 'trifft nicht zu',
  }),
  en: Object.freeze({
    title: 'Logic',
    noBlock: 'Select a content block in Layout.',
    applies: 'Block is visible',
    hidden: 'Block is hidden for this test data',
    addRule: 'Add rule',
    addGroup: 'Add group',
    delete: 'Delete',
    moveUp: 'Move up',
    moveDown: 'Move down',
    all: 'All rules (AND)',
    any: 'At least one rule (OR)',
    field: 'Field / merge tag',
    operator: 'Operator',
    type: 'Value type',
    value: 'Value',
    testData: 'Live test data (JSON)',
    invalidJson: 'Test data must be a valid JSON object.',
    equals: 'equals',
    'not-equals': 'does not equal',
    contains: 'contains',
    'not-contains': 'does not contain',
    exists: 'exists',
    empty: 'is empty',
    greater: 'is greater than',
    less: 'is less than',
    string: 'Text',
    number: 'Number',
    boolean: 'Yes/No',
    null: 'Null',
    true: 'Yes',
    false: 'No',
    matches: 'matches',
    misses: 'does not match',
  }),
});

export function createMailLogicDefinition(input = {}) {
  return normalizeMailLogicDefinition({
    version: MAIL_LOGIC_VERSION,
    root: input.root || createMailLogicGroup({ combinator: 'and' }),
    testData: isPlainObject(input.testData) ? input.testData : {},
  });
}

export function createMailLogicGroup(input = {}) {
  return {
    id: normalizedId(input.id, 'logic-group'),
    kind: 'group',
    combinator: COMBINATOR_SET.has(input.combinator) ? input.combinator : 'and',
    children: Array.isArray(input.children) ? input.children : [],
  };
}

export function createMailLogicRule(input = {}) {
  const operator = OPERATOR_SET.has(input.operator) ? input.operator : 'equals';
  const valueType = VALUE_TYPE_SET.has(input.valueType) ? input.valueType : inferValueType(input.value);
  return {
    id: normalizedId(input.id, 'logic-rule'),
    kind: 'rule',
    field: String(input.field || '').trim(),
    operator,
    valueType,
    value: normalizeTypedValue(input.value, valueType, operator),
  };
}

export function normalizeMailLogicDefinition(input = {}) {
  const usedIds = new Set();
  const normalizeNode = (node, fallbackKind = 'rule') => {
    if (node?.kind === 'group' || fallbackKind === 'group') {
      const group = createMailLogicGroup(node || {});
      group.id = uniqueNodeId(group.id, 'logic-group', usedIds);
      group.children = (Array.isArray(node?.children) ? node.children : [])
        .map((child) => normalizeNode(child, child?.kind === 'group' ? 'group' : 'rule'));
      return group;
    }
    const rule = createMailLogicRule(node || {});
    rule.id = uniqueNodeId(rule.id, 'logic-rule', usedIds);
    return rule;
  };
  const root = normalizeNode(input?.root?.kind === 'group' ? input.root : createMailLogicGroup(), 'group');
  return {
    version: MAIL_LOGIC_VERSION,
    root,
    testData: isPlainObject(input?.testData) ? cloneJson(input.testData) : {},
  };
}

export function evaluateMailLogic(input, data = undefined) {
  const definition = normalizeMailLogicDefinition(input || {});
  const source = data === undefined ? definition.testData : data;
  const results = {};
  const evaluateNode = (node) => {
    let matched;
    if (node.kind === 'group') {
      const childResults = node.children.map(evaluateNode);
      matched = childResults.length === 0
        ? true
        : node.combinator === 'or'
          ? childResults.some(Boolean)
          : childResults.every(Boolean);
    } else {
      matched = evaluateRule(node, source);
    }
    results[node.id] = matched;
    return matched;
  };
  return { matched: evaluateNode(definition.root), results, definition };
}

export function evaluateRule(input, data) {
  const rule = createMailLogicRule(input);
  const actual = getPath(data, rule.field);
  if (rule.operator === 'exists') return actual !== undefined && actual !== null;
  if (rule.operator === 'empty') return isEmptyValue(actual);
  const expected = normalizeTypedValue(rule.value, rule.valueType, rule.operator);
  const actualTyped = coerceActualValue(actual, rule.valueType);
  if (rule.operator === 'equals') return equalTyped(actualTyped, expected);
  if (rule.operator === 'not-equals') return !equalTyped(actualTyped, expected);
  if (rule.operator === 'contains' || rule.operator === 'not-contains') {
    const contains = Array.isArray(actual)
      ? actual.some((entry) => equalTyped(coerceActualValue(entry, rule.valueType), expected))
      : String(actual ?? '').includes(String(expected ?? ''));
    return rule.operator === 'contains' ? contains : !contains;
  }
  if (rule.operator === 'greater') return comparable(actualTyped, expected, (left, right) => left > right);
  if (rule.operator === 'less') return comparable(actualTyped, expected, (left, right) => left < right);
  return false;
}

export function insertMailLogicNode(input, parentId, node, index = Number.POSITIVE_INFINITY) {
  const definition = normalizeMailLogicDefinition(input);
  const parent = findLogicNode(definition.root, parentId)?.node;
  if (!parent || parent.kind !== 'group') throw new RangeError(`Unknown logic group: ${parentId}`);
  const normalized = node?.kind === 'group' ? createMailLogicGroup(node) : createMailLogicRule(node);
  const existingIds = new Set(flattenLogicNodes(definition.root).map((entry) => entry.node.id));
  normalized.id = uniqueNodeId(normalized.id, normalized.kind === 'group' ? 'logic-group' : 'logic-rule', existingIds);
  if (normalized.kind === 'group') normalized.children = normalizeMailLogicDefinition({ root: normalized }).root.children;
  const target = Math.max(0, Math.min(parent.children.length, Number(index)));
  parent.children.splice(Number.isFinite(target) ? target : parent.children.length, 0, normalized);
  return definition;
}

export function updateMailLogicNode(input, id, patch) {
  const definition = normalizeMailLogicDefinition(input);
  const match = findLogicNode(definition.root, id);
  if (!match) throw new RangeError(`Unknown logic node: ${id}`);
  if (match.node.kind === 'group') {
    const next = typeof patch === 'function' ? patch(cloneJson(match.node)) : { ...match.node, ...patch };
    match.node.combinator = COMBINATOR_SET.has(next.combinator) ? next.combinator : match.node.combinator;
  } else {
    const next = typeof patch === 'function' ? patch(cloneJson(match.node)) : { ...match.node, ...patch };
    Object.assign(match.node, createMailLogicRule({ ...match.node, ...next, id: match.node.id }));
  }
  return definition;
}

export function removeMailLogicNode(input, id) {
  const definition = normalizeMailLogicDefinition(input);
  const match = findLogicNode(definition.root, id);
  if (!match) return definition;
  if (!match.parent) throw new TypeError('The root logic group cannot be removed');
  match.parent.children.splice(match.index, 1);
  return definition;
}

export function moveMailLogicNode(input, id, delta) {
  const definition = normalizeMailLogicDefinition(input);
  const match = findLogicNode(definition.root, id);
  if (!match?.parent) return definition;
  const nextIndex = Math.max(0, Math.min(match.parent.children.length - 1, match.index + Number(delta)));
  if (nextIndex === match.index) return definition;
  const [node] = match.parent.children.splice(match.index, 1);
  match.parent.children.splice(nextIndex, 0, node);
  return definition;
}

export function setMailLogicTestData(input, testData) {
  const definition = normalizeMailLogicDefinition(input);
  if (!isPlainObject(testData)) throw new TypeError('Mail logic test data must be an object');
  definition.testData = cloneJson(testData);
  return definition;
}

export function readBlockMailLogic(document, blockId) {
  const block = findBlockByReference(document, blockId);
  return block ? normalizeMailLogicDefinition(block.data?.value?.logic || {}) : null;
}

export function writeBlockMailLogic(document, blockId, logic) {
  const next = cloneJson(document);
  const block = findBlockByReference(next, blockId);
  if (!block) throw new RangeError(`Unknown email block: ${blockId}`);
  block.data = isPlainObject(block.data) ? block.data : {};
  block.data.value = isPlainObject(block.data.value) ? block.data.value : {};
  block.data.value.logic = normalizeMailLogicDefinition(logic);
  // The rule tree replaces the historical single visibility predicate.
  delete block.data.value.visibility;
  return next;
}

export function flattenMergeTagPaths(value, prefix = '') {
  if (!isPlainObject(value)) return prefix ? [prefix] : [];
  const paths = [];
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (isPlainObject(child)) paths.push(...flattenMergeTagPaths(child, path));
    else paths.push(path);
  }
  return paths;
}

/** Mount a DOM editor for one selected Easy Email block. */
export function mountMailLogicEditor(options = {}) {
  const host = options.host;
  if (!host?.ownerDocument?.createElement) throw new TypeError('Mail logic editor requires a DOM host');
  if (typeof options.getDocument !== 'function' || typeof options.setDocument !== 'function') {
    throw new TypeError('Mail logic editor requires getDocument() and setDocument()');
  }
  const locale = String(options.locale || 'de').toLowerCase().startsWith('en') ? 'en' : 'de';
  const t = { ...LABELS[locale], ...(options.labels || {}) };
  const doc = host.ownerDocument;
  const root = doc.createElement('section');
  root.className = 'mail-logic-editor';
  root.dataset.mailLogicEditor = 'true';
  host.replaceChildren(root);
  let selectedBlockId = selectionReference(options.getSelectedBlockId?.());
  let definition = null;
  let testDataDraft = '';
  let jsonError = '';
  let disposed = false;
  let writeChain = Promise.resolve();
  let readOnly = options.readOnly === true;
  const fieldListId = normalizedId('', 'mail-logic-fields');
  const actionIcon = (name, fallback) => options.getActionIcon?.(name) || escapeHtml(fallback);

  const selectionCleanup = options.onSelectionChange?.((selection) => {
    selectedBlockId = selectionReference(selection);
    reload();
  });

  async function reload() {
    if (disposed) return;
    const blockId = selectedBlockId;
    const documentValue = await options.getDocument();
    if (disposed || blockId !== selectedBlockId) return;
    definition = blockId ? readBlockMailLogic(documentValue, blockId) : null;
    testDataDraft = definition ? JSON.stringify(definition.testData, null, 2) : '';
    jsonError = '';
    render();
    if (definition) await emitPreview(blockId, definition);
  }

  function persist(nextDefinition, reason) {
    definition = normalizeMailLogicDefinition(nextDefinition);
    testDataDraft = JSON.stringify(definition.testData, null, 2);
    const snapshot = definition;
    const blockId = selectedBlockId;
    writeChain = writeChain.then(async () => {
      if (disposed || !blockId) return;
      const currentDocument = await options.getDocument();
      const nextDocument = writeBlockMailLogic(currentDocument, blockId, snapshot);
      await options.setDocument(nextDocument);
      await emitPreview(blockId, snapshot);
      options.onChange?.({ blockId, logic: cloneJson(snapshot), reason });
    });
    render();
    return writeChain;
  }

  function render() {
    if (disposed) return;
    if (!selectedBlockId || !definition) {
      root.innerHTML = `<div class="ctox-empty"><strong>${escapeHtml(t.title)}</strong><span>${escapeHtml(t.noBlock)}</span></div>`;
      return;
    }
    const evaluation = evaluateMailLogic(definition, definition.testData);
    const paths = [...new Set([
      ...flattenMergeTagPaths(options.mergeTags || {}),
      ...flattenMergeTagPaths(definition.testData),
      ...flattenLogicNodes(definition.root).filter(({ node }) => node.kind === 'rule').map(({ node }) => node.field).filter(Boolean),
    ])].sort();
    root.innerHTML = `
      <div class="mail-logic-editor-result ${evaluation.matched ? 'is-match' : 'is-miss'}" role="status" data-logic-preview-result>
        <strong>${escapeHtml(evaluation.matched ? t.applies : t.hidden)}</strong>
      </div>
      <div class="mail-logic-editor-tree" data-logic-tree>
        ${renderGroup(definition.root, evaluation.results, t, fieldListId, readOnly, actionIcon, true, 0, 1)}
      </div>
      <label class="mail-logic-editor-testdata">
        <span class="ctox-field-label">${escapeHtml(t.testData)}</span>
        <textarea class="ctox-textarea" rows="8" spellcheck="false" data-logic-test-data aria-invalid="${jsonError ? 'true' : 'false'}" ${readOnly ? 'disabled' : ''}>${escapeHtml(testDataDraft)}</textarea>
        <small class="mail-logic-editor-error" data-logic-json-error>${escapeHtml(jsonError)}</small>
      </label>
      <datalist id="${escapeAttr(fieldListId)}">${paths.map((path) => `<option value="${escapeAttr(path)}"></option>`).join('')}</datalist>
    `;
  }

  root.addEventListener('click', (event) => {
    const action = event.target.closest?.('[data-logic-action]');
    if (!action || !definition || readOnly) return;
    const id = String(action.dataset.logicNodeId || '');
    const kind = action.dataset.logicAction;
    if (kind === 'add-rule') persist(insertMailLogicNode(definition, id, createMailLogicRule()), 'rule.create');
    else if (kind === 'add-group') persist(insertMailLogicNode(definition, id, createMailLogicGroup()), 'group.create');
    else if (kind === 'delete') persist(removeMailLogicNode(definition, id), 'node.delete');
    else if (kind === 'up') persist(moveMailLogicNode(definition, id, -1), 'node.reorder');
    else if (kind === 'down') persist(moveMailLogicNode(definition, id, 1), 'node.reorder');
  });

  root.addEventListener('change', (event) => {
    const control = event.target.closest?.('[data-logic-field]');
    if (!control || !definition || readOnly) return;
    const id = String(control.dataset.logicNodeId || '');
    const field = control.dataset.logicField;
    if (!id || !field) return;
    const patch = { [field]: valueFromControl(control, field) };
    persist(updateMailLogicNode(definition, id, patch), 'node.edit');
  });

  root.addEventListener('input', (event) => {
    if (!event.target.matches?.('[data-logic-test-data]') || !definition || readOnly) return;
    testDataDraft = event.target.value;
    try {
      const parsed = JSON.parse(testDataDraft || '{}');
      if (!isPlainObject(parsed)) throw new TypeError();
      jsonError = '';
      persist(setMailLogicTestData(definition, parsed), 'test-data.edit');
    } catch {
      jsonError = t.invalidJson;
      event.target.setAttribute('aria-invalid', 'true');
      root.querySelector('[data-logic-json-error]').textContent = jsonError;
    }
  });

  reload();

  return Object.freeze({
    reload,
    getDefinition: () => definition ? cloneJson(definition) : null,
    async setSelectedBlock(blockId) {
      selectedBlockId = selectionReference(blockId);
      await reload();
    },
    async flush() { await writeChain; },
    setReadOnly(nextReadOnly) {
      readOnly = nextReadOnly === true;
      render();
    },
    destroy() {
      disposed = true;
      selectionCleanup?.();
      root.remove();
    },
  });

  async function emitPreview(blockId, value) {
    const evaluation = evaluateMailLogic(value, value.testData);
    const detail = {
      blockId,
      matched: evaluation.matched,
      results: cloneJson(evaluation.results),
      testData: cloneJson(value.testData),
      logic: cloneJson(value),
    };
    await options.onTestDataChange?.(detail.testData);
    await options.onPreviewChange?.(detail);
  }
}

function renderGroup(group, results, t, fieldListId, readOnly, actionIcon, isRoot = false, index = 0, count = 1) {
  const children = group.children.map((node, index) => node.kind === 'group'
    ? renderGroup(node, results, t, fieldListId, readOnly, actionIcon, false, index, group.children.length)
    : renderRule(node, results, t, fieldListId, readOnly, actionIcon, index, group.children.length)).join('');
  return `
    <section class="mail-logic-group" data-logic-node-id="${escapeAttr(group.id)}" data-context-record-id="${escapeAttr(group.id)}" data-context-record-type="mail-logic-group" data-context-label="${escapeAttr(group.combinator.toUpperCase())}">
      <header class="mail-logic-group-header">
        <select class="ctox-select" data-logic-field="combinator" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.operator)}" ${readOnly ? 'disabled' : ''}>
          <option value="and" ${group.combinator === 'and' ? 'selected' : ''}>${escapeHtml(t.all)}</option>
          <option value="or" ${group.combinator === 'or' ? 'selected' : ''}>${escapeHtml(t.any)}</option>
        </select>
        <span class="mail-logic-match ${results[group.id] ? 'is-match' : 'is-miss'}">${escapeHtml(results[group.id] ? t.matches : t.misses)}</span>
        <span class="mail-logic-node-actions">
          <button class="ctox-pane-icon" type="button" data-logic-action="add-rule" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.addRule)}" title="${escapeAttr(t.addRule)}" ${readOnly ? 'disabled' : ''}>${actionIcon('add', '+')}</button>
          <button class="ctox-pane-icon" type="button" data-logic-action="add-group" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.addGroup)}" title="${escapeAttr(t.addGroup)}" ${readOnly ? 'disabled' : ''}>${actionIcon('folder', '+')}</button>
          ${isRoot ? '' : `<button class="ctox-pane-icon" type="button" data-logic-action="up" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.moveUp)}" title="${escapeAttr(t.moveUp)}" ${readOnly || index === 0 ? 'disabled' : ''}>${actionIcon('chevronUp', '↑')}</button><button class="ctox-pane-icon" type="button" data-logic-action="down" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.moveDown)}" title="${escapeAttr(t.moveDown)}" ${readOnly || index === count - 1 ? 'disabled' : ''}>${actionIcon('chevronDown', '↓')}</button><button class="ctox-pane-icon" type="button" data-logic-action="delete" data-logic-node-id="${escapeAttr(group.id)}" aria-label="${escapeAttr(t.delete)}" title="${escapeAttr(t.delete)}" ${readOnly ? 'disabled' : ''}>${actionIcon('trash', '×')}</button>`}
        </span>
      </header>
      <div class="mail-logic-group-children">${children}</div>
    </section>`;
}

function renderRule(rule, results, t, fieldListId, readOnly, actionIcon, index, count) {
  const noValue = VALUELESS_OPERATORS.has(rule.operator);
  return `
    <div class="mail-logic-rule" data-logic-node-id="${escapeAttr(rule.id)}" data-context-record-id="${escapeAttr(rule.id)}" data-context-record-type="mail-logic-rule" data-context-label="${escapeAttr(rule.field || t.field)}">
      <input class="ctox-input" list="${escapeAttr(fieldListId)}" value="${escapeAttr(rule.field)}" placeholder="${escapeAttr(t.field)}" data-logic-field="field" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.field)}" ${readOnly ? 'disabled' : ''}>
      <select class="ctox-select" data-logic-field="operator" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.operator)}" ${readOnly ? 'disabled' : ''}>
        ${MAIL_LOGIC_OPERATORS.map((operator) => `<option value="${operator}" ${rule.operator === operator ? 'selected' : ''}>${escapeHtml(t[operator])}</option>`).join('')}
      </select>
      <select class="ctox-select" data-logic-field="valueType" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.type)}" ${readOnly || noValue ? 'disabled' : ''}>
        ${MAIL_LOGIC_VALUE_TYPES.map((type) => `<option value="${type}" ${rule.valueType === type ? 'selected' : ''}>${escapeHtml(t[type])}</option>`).join('')}
      </select>
      ${renderValueControl(rule, t, noValue, readOnly)}
      <span class="mail-logic-match ${results[rule.id] ? 'is-match' : 'is-miss'}">${escapeHtml(results[rule.id] ? t.matches : t.misses)}</span>
      <span class="mail-logic-node-actions">
        <button class="ctox-pane-icon" type="button" data-logic-action="up" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.moveUp)}" title="${escapeAttr(t.moveUp)}" ${readOnly || index === 0 ? 'disabled' : ''}>${actionIcon('chevronUp', '↑')}</button>
        <button class="ctox-pane-icon" type="button" data-logic-action="down" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.moveDown)}" title="${escapeAttr(t.moveDown)}" ${readOnly || index === count - 1 ? 'disabled' : ''}>${actionIcon('chevronDown', '↓')}</button>
        <button class="ctox-pane-icon" type="button" data-logic-action="delete" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.delete)}" title="${escapeAttr(t.delete)}" ${readOnly ? 'disabled' : ''}>${actionIcon('trash', '×')}</button>
      </span>
    </div>`;
}

function renderValueControl(rule, t, noValue, readOnly) {
  if (noValue || rule.valueType === 'null') {
    return `<input class="ctox-input" value="" disabled aria-label="${escapeAttr(t.value)}">`;
  }
  if (rule.valueType === 'boolean') {
    return `<select class="ctox-select" data-logic-field="value" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.value)}" ${readOnly ? 'disabled' : ''}><option value="true" ${rule.value === true ? 'selected' : ''}>${escapeHtml(t.true)}</option><option value="false" ${rule.value === false ? 'selected' : ''}>${escapeHtml(t.false)}</option></select>`;
  }
  return `<input class="ctox-input" type="${rule.valueType === 'number' ? 'number' : 'text'}" value="${escapeAttr(rule.value ?? '')}" placeholder="${escapeAttr(t.value)}" data-logic-field="value" data-logic-node-id="${escapeAttr(rule.id)}" aria-label="${escapeAttr(t.value)}" ${readOnly ? 'disabled' : ''}>`;
}

function valueFromControl(control, field) {
  if (field === 'value') {
    const row = control.closest('.mail-logic-rule');
    const type = row?.querySelector('[data-logic-field="valueType"]')?.value || 'string';
    return normalizeTypedValue(control.value, type, row?.querySelector('[data-logic-field="operator"]')?.value || 'equals');
  }
  return control.value;
}

function flattenLogicNodes(root) {
  const entries = [];
  const visit = (node, parent = null, index = 0) => {
    entries.push({ node, parent, index });
    if (node.kind === 'group') node.children.forEach((child, childIndex) => visit(child, node, childIndex));
  };
  visit(root);
  return entries;
}

function findLogicNode(root, id) {
  return flattenLogicNodes(root).find((entry) => entry.node.id === id) || null;
}

export function findBlockByReference(document, reference) {
  const direct = blockAtEasyEmailPath(document, reference);
  if (direct) return direct;
  let result = null;
  const visit = (block) => {
    if (!block || result) return;
    if (block.id === reference) {
      result = block;
      return;
    }
    for (const child of block.children || []) visit(child);
  };
  visit(document?.content);
  return result;
}

export function blockAtEasyEmailPath(document, reference) {
  const raw = String(reference || '').trim();
  if (!raw) return null;
  const tokens = raw
    .replace(/\[(\d+)\]/g, '.$1')
    .replace(/^\./, '')
    .split('.')
    .filter(Boolean);
  if (!tokens.length || tokens[0] !== 'content') return null;
  let current = document;
  for (const token of tokens) {
    if (token === 'content' || token === 'children') {
      current = current?.[token];
      continue;
    }
    if (!/^\d+$/.test(token) || !Array.isArray(current)) return null;
    current = current[Number(token)];
  }
  return current && typeof current === 'object' && typeof current.type === 'string' ? current : null;
}

function getPath(source, path) {
  return String(path || '').split('.').filter(Boolean).reduce((current, key) => current?.[key], source);
}

function isEmptyValue(value) {
  return value === undefined || value === null || value === '' || (Array.isArray(value) && value.length === 0);
}

function coerceActualValue(value, type) {
  if (type === 'null') return value == null ? null : value;
  if (type === 'number') {
    const number = Number(value);
    return Number.isFinite(number) ? number : Number.NaN;
  }
  if (type === 'boolean') {
    if (value === true || value === 'true' || value === 1 || value === '1') return true;
    if (value === false || value === 'false' || value === 0 || value === '0') return false;
    return Boolean(value);
  }
  return String(value ?? '');
}

function normalizeTypedValue(value, type, operator = 'equals') {
  if (VALUELESS_OPERATORS.has(operator)) return '';
  if (type === 'null') return null;
  if (type === 'number') {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }
  if (type === 'boolean') return value === true || value === 'true' || value === 1 || value === '1';
  return String(value ?? '');
}

function inferValueType(value) {
  if (value === null) return 'null';
  if (typeof value === 'number') return 'number';
  if (typeof value === 'boolean') return 'boolean';
  return 'string';
}

function equalTyped(left, right) {
  return Object.is(left, right) || (Number.isNaN(left) && Number.isNaN(right));
}

function comparable(left, right, comparator) {
  if (typeof left === 'number' && (Number.isNaN(left) || Number.isNaN(right))) return false;
  return comparator(left, right);
}

function normalizedId(value, prefix) {
  const id = String(value || '').trim();
  if (id) return id;
  if (globalThis.crypto?.randomUUID) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function selectionReference(value) {
  if (value && typeof value === 'object') {
    return String(value.blockId || value.idx || value.path || value.reference || '');
  }
  return String(value || '');
}

function uniqueNodeId(candidate, prefix, used) {
  let id = candidate;
  while (used.has(id)) id = normalizedId('', prefix);
  used.add(id);
  return id;
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function cloneJson(value) {
  return typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

function escapeHtml(value) {
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
