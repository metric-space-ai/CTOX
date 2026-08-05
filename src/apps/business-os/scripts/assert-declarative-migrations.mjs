// Guard the JSON-only module migration DSL used by app.js and module
// standalone data sources. JSON is canonical; schema.js must expose the same
// collections, target versions, and ordered operations for browser consumers.

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { executableDeclarativeMigrationStrategies } from '../shared/declarative-migrations.js';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(scriptDir, '..');
const repoRoot = resolve(appRoot, '../../..');
const modulesRoot = join(appRoot, 'modules');
const failures = [];

const documentsByModule = new Map();

for (const id of readdirSync(modulesRoot).sort()) {
  const dir = join(modulesRoot, id);
  const schemaPath = join(dir, 'collections.schema.json');
  if (!statSync(dir).isDirectory() || !existsSync(schemaPath)) continue;

  const document = JSON.parse(readFileSync(schemaPath, 'utf8'));
  documentsByModule.set(id, document);

  const moduleSchemaPath = join(dir, 'schema.js');
  let moduleStrategies = {};
  if (existsSync(moduleSchemaPath)) {
    try {
      const schemaModule = await import(pathToFileURL(moduleSchemaPath).href);
      moduleStrategies = schemaModule.migrationStrategies || {};
    } catch (error) {
      failures.push(`${relative(repoRoot, moduleSchemaPath)}: failed to load schema.js: ${error.message}`);
    }
  }

  compareMigrationMirrors({
    id,
    jsonPath: schemaPath,
    jsonStrategies: document.migration_strategies || {},
    schemaPath: moduleSchemaPath,
    schemaStrategies: moduleStrategies,
  });
}

await compareStarterTemplates();

expectMigration('ctox', 'business_commands', '1', {
  id: 'cmd_1',
  module: 'tickets',
  status: 'pending',
}, (doc) => doc.inbound_channel === 'tickets');

expectMigration('ctox', 'business_commands', '1', {
  id: 'cmd_2',
  module: 'tickets',
  inbound_channel: 'email',
  status: 'pending',
}, (doc) => doc.inbound_channel === 'email');

expectMigration('notes', 'notes', '1', {
  id: 'note_1',
  title: 'Legacy',
  is_favorite: 1,
}, (doc) => (
  doc.notebook === ''
  && doc.tags === ''
  && doc.is_favorite === true
  && doc.is_trashed === false
  && doc.is_locked === false
  && doc.lock_passcode === ''
));

expectMigration('outbound', 'outbound_messages', '1', {
  id: 'msg_1',
  payload: {
    channel: 'letter',
    recipient_address_text: 'Ada Lovelace',
  },
}, (doc) => (
  doc.channel === 'letter'
  && doc.recipient_address_text === 'Ada Lovelace'
  && doc.document_id === ''
  && doc.document_version_id === ''
  && doc.document_pdf_url === ''
  && doc.physical_sent_at_ms === 0
));

expectMigration('matching', 'matching_results', '1', {
  id: 'match_1',
  score: 0.9,
}, (doc) => doc.id === 'match_1' && doc.score === 0.9);

if (failures.length) {
  console.error(`Business OS declarative migrations failed:\n${failures.map((line) => `- ${line}`).join('\n')}`);
  process.exit(1);
}

console.log(`Business OS declarative migrations OK (${documentsByModule.size} modules checked)`);

function compareMigrationMirrors({ id, jsonPath, jsonStrategies, schemaPath, schemaStrategies }) {
  compileStrategies(jsonPath, jsonStrategies);
  compileStrategies(schemaPath, schemaStrategies);

  const jsonCollections = nonEmptyStrategyCollections(jsonStrategies);
  const schemaCollections = nonEmptyStrategyCollections(schemaStrategies);
  const collections = new Set([...jsonCollections.keys(), ...schemaCollections.keys()]);

  for (const collection of [...collections].sort()) {
    const jsonVersions = jsonCollections.get(collection);
    const schemaVersions = schemaCollections.get(collection);

    if (!jsonVersions) {
      failures.push(`${id}/${collection}: collections.schema.json missing migration strategies mirrored by schema.js`);
      continue;
    }
    if (!schemaVersions) {
      failures.push(`${id}/${collection}: schema.js missing migration strategies declared by collections.schema.json`);
      continue;
    }

    const versions = new Set([...Object.keys(jsonVersions), ...Object.keys(schemaVersions)]);
    for (const version of [...versions].sort(numericVersionCompare)) {
      if (!Object.hasOwn(jsonVersions, version)) {
        failures.push(`${id}/${collection}: collections.schema.json missing migration ${version} mirrored by schema.js`);
        continue;
      }
      if (!Object.hasOwn(schemaVersions, version)) {
        failures.push(`${id}/${collection}: schema.js missing migration ${version} declared by collections.schema.json`);
        continue;
      }

      try {
        const jsonOperations = normalizedOperations(jsonVersions[version]);
        const schemaOperations = normalizedOperations(schemaVersions[version]);
        if (JSON.stringify(jsonOperations) !== JSON.stringify(schemaOperations)) {
          failures.push(
            `${id}/${collection}: migration ${version} operations differ; `
            + `collections.schema.json=${JSON.stringify(jsonOperations)} `
            + `schema.js=${JSON.stringify(schemaOperations)}`,
          );
        }
      } catch (error) {
        failures.push(`${id}/${collection}: migration ${version} mirror cannot be compared: ${error.message}`);
      }
    }
  }
}

function compileStrategies(path, strategies) {
  for (const [collection, versions] of Object.entries(strategies || {})) {
    try {
      executableDeclarativeMigrationStrategies(versions);
    } catch (error) {
      failures.push(`${relative(repoRoot, path)}: ${collection} migrations failed to compile: ${error.message}`);
    }
  }
}

function nonEmptyStrategyCollections(strategies) {
  return new Map(
    Object.entries(strategies || {}).filter(([, versions]) => (
      versions && typeof versions === 'object' && Object.keys(versions).length > 0
    )),
  );
}

function normalizedOperations(spec) {
  if (typeof spec === 'function') return operationsFromFunction(spec);
  const operations = Array.isArray(spec) ? spec : spec?.operations;
  if (!Array.isArray(operations)) {
    throw new Error('declarative migration spec must contain an operations array');
  }
  return operations.map(normalizeOperation);
}

function normalizeOperation(operation) {
  if (operation?.op === 'set_from_first_truthy') {
    const normalized = {
      op: operation.op,
      field: operation.field,
      paths: [...operation.paths],
    };
    if (Object.hasOwn(operation, 'default')) normalized.default = operation.default;
    return normalized;
  }
  if (operation?.op === 'set_boolean') {
    return {
      op: operation.op,
      field: operation.field,
      path: operation.path || operation.field,
    };
  }
  throw new Error(`unsupported declarative migration operation ${operation?.op}`);
}

function operationsFromFunction(migrate) {
  const source = migrate.toString().trim();
  const arrow = source.match(/^\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*=>\s*([\s\S]+)$/);
  if (!arrow) throw new Error('schema.js migration must be an arrow function');

  const [, parameter] = arrow;
  const body = stripWrappingParentheses(arrow[2].trim());
  if (body === parameter) return [];
  if (!body.startsWith('{') || !body.endsWith('}')) {
    throw new Error('schema.js migration must return the old document or an object literal');
  }

  const properties = splitTopLevel(body.slice(1, -1), ',').map((part) => part.trim()).filter(Boolean);
  let preservesOldDocument = false;
  const operations = [];
  for (const property of properties) {
    if (property === `...${parameter}`) {
      preservesOldDocument = true;
      continue;
    }

    const colon = topLevelSeparatorIndex(property, ':');
    if (colon < 0) throw new Error(`unsupported schema.js migration property ${property}`);
    const field = propertyKey(property.slice(0, colon).trim());
    const expression = stripWrappingParentheses(property.slice(colon + 1).trim());
    operations.push(operationFromExpression(parameter, field, expression));
  }

  if (!preservesOldDocument) {
    throw new Error('schema.js migration object must spread the old document');
  }
  return operations;
}

function operationFromExpression(parameter, field, expression) {
  const boolean = expression.match(new RegExp(`^!!\\s*${escapeRegExp(parameter)}((?:\\?\\.[A-Za-z_$][\\w$]*|\\.[A-Za-z_$][\\w$]*)+)$`));
  if (boolean) {
    return {
      op: 'set_boolean',
      field,
      path: memberPath(boolean[1]),
    };
  }

  const operands = splitTopLevel(expression, '||').map((part) => stripWrappingParentheses(part.trim()));
  const paths = [];
  let fallback;
  let hasFallback = false;
  for (const operand of operands) {
    const path = operand.match(new RegExp(`^${escapeRegExp(parameter)}((?:\\?\\.[A-Za-z_$][\\w$]*|\\.[A-Za-z_$][\\w$]*)+)$`));
    if (path && !hasFallback) {
      paths.push(memberPath(path[1]));
      continue;
    }
    if (hasFallback || operand !== operands.at(-1)) {
      throw new Error(`unsupported truthy-chain operand ${operand}`);
    }
    fallback = literalValue(operand);
    hasFallback = true;
  }
  if (!paths.length) throw new Error(`unsupported schema.js migration expression ${expression}`);

  const operation = { op: 'set_from_first_truthy', field, paths };
  if (hasFallback) operation.default = fallback;
  return operation;
}

function memberPath(suffix) {
  return suffix.replaceAll('?.', '.').replace(/^\./, '');
}

function literalValue(source) {
  if (/^'(?:[^'\\]|\\.)*'$/.test(source)) {
    return source.slice(1, -1).replace(/\\'/g, "'").replace(/\\\\/g, '\\');
  }
  if (/^"(?:[^"\\]|\\.)*"$/.test(source)) return JSON.parse(source);
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(source)) return Number(source);
  if (source === 'true') return true;
  if (source === 'false') return false;
  if (source === 'null') return null;
  throw new Error(`unsupported schema.js migration literal ${source}`);
}

function propertyKey(source) {
  if (/^[A-Za-z_$][\w$]*$/.test(source)) return source;
  const value = literalValue(source);
  if (typeof value === 'string') return value;
  throw new Error(`unsupported schema.js migration field ${source}`);
}

function splitTopLevel(source, separator) {
  const parts = [];
  let start = 0;
  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = null;
      continue;
    }
    if (char === "'" || char === '"' || char === '`') {
      quote = char;
      continue;
    }
    if (char === '(' || char === '[' || char === '{') depth += 1;
    else if (char === ')' || char === ']' || char === '}') depth -= 1;
    else if (depth === 0 && source.startsWith(separator, index)) {
      parts.push(source.slice(start, index));
      index += separator.length - 1;
      start = index + 1;
    }
  }
  parts.push(source.slice(start));
  return parts;
}

function topLevelSeparatorIndex(source, separator) {
  const first = splitTopLevel(source, separator)[0];
  return first.length === source.length ? -1 : first.length;
}

function stripWrappingParentheses(source) {
  let stripped = source;
  while (stripped.startsWith('(') && stripped.endsWith(')') && wrappingPairCoversSource(stripped)) {
    stripped = stripped.slice(1, -1).trim();
  }
  return stripped;
}

function wrappingPairCoversSource(source) {
  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = null;
      continue;
    }
    if (char === "'" || char === '"' || char === '`') quote = char;
    else if (char === '(') depth += 1;
    else if (char === ')') {
      depth -= 1;
      if (depth === 0 && index !== source.length - 1) return false;
    }
  }
  return depth === 0;
}

function escapeRegExp(source) {
  return source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function numericVersionCompare(left, right) {
  return Number(left) - Number(right) || left.localeCompare(right);
}

async function compareStarterTemplates() {
  const starterRoot = join(appRoot, 'app-starter', 'v2');
  const jsonPath = join(starterRoot, 'collections.schema.json.tpl');
  const schemaPath = join(starterRoot, 'schema.js.tpl');
  const collection = 'starter_records';
  try {
    const document = JSON.parse(readFileSync(jsonPath, 'utf8').replaceAll('__COLLECTION__', collection));
    const source = readFileSync(schemaPath, 'utf8').replaceAll('__COLLECTION__', collection);
    const schemaModule = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
    compareMigrationMirrors({
      id: 'app-starter/v2',
      jsonPath,
      jsonStrategies: document.migration_strategies || {},
      schemaPath,
      schemaStrategies: schemaModule.migrationStrategies || {},
    });
  } catch (error) {
    failures.push(`app-starter/v2: failed to compare migration templates: ${error.message}`);
  }
}

function expectMigration(moduleId, collection, version, oldDoc, predicate) {
  const document = documentsByModule.get(moduleId);
  const strategy = document?.migration_strategies?.[collection];
  const executable = executableDeclarativeMigrationStrategies(strategy);
  const migrate = executable?.[version];
  if (typeof migrate !== 'function') {
    failures.push(`${moduleId}/${collection}: missing migration ${version}`);
    return;
  }
  const migrated = migrate(oldDoc);
  if (!predicate(migrated)) {
    failures.push(`${moduleId}/${collection}: migration ${version} produced ${JSON.stringify(migrated)}`);
  }
}
