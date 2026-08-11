import test from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';

const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8');
const schemaSource = readFileSync(new URL('../schema.js', import.meta.url), 'utf8');
const collectionsManifest = JSON.parse(readFileSync(new URL('../collections.schema.json', import.meta.url), 'utf8'));
const moduleManifest = JSON.parse(readFileSync(new URL('../module.json', import.meta.url), 'utf8'));
const bundled = await build({
  entryPoints: [new URL('../index.js', import.meta.url).pathname],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  write: false,
  plugins: [{
    name: 'isolated-runtime-shared',
    setup(buildApi) {
      buildApi.onResolve({ filter: /^\.\.\/\.\.\/shared\// }, (args) => ({
        path: new URL(args.path.slice('../../shared/'.length), new URL('../../../../src/apps/business-os/shared/', import.meta.url)).pathname,
      }));
    },
  }],
});
const module = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
const hooks = module.__thesenOutboundTestHooks;

test('THESEN source catalog includes protected and public research providers', () => {
  const ids = hooks.SOURCE_DEFS.map((item) => item.id);
  assert.deepEqual(ids.sort(), [
    'bundesanzeiger.de', 'companyhouse.de', 'dnbhoovers.com', 'experte.de',
    'firmenabc.at', 'google.de', 'handelsregister.de', 'leadfeeder.com',
    'maps.google.com', 'moneyhouse.ch', 'northdata.de', 'rocketreach.com',
    'xing.com', 'zefix.ch',
  ].sort());
  assert.ok(hooks.SOURCE_DEFS.every((item) => item.targetKey));
  assert.equal(hooks.SOURCE_DEFS.find((item) => item.id === 'dnbhoovers.com').credentialSecretName, 'DNB_HOOVERS_BROWSER_LOGIN');
  assert.equal(hooks.SOURCE_DEFS.find((item) => item.id === 'leadfeeder.com').credentialSecretName, 'LEADFEEDER_BROWSER_LOGIN');
  assert.equal(hooks.SOURCE_DEFS.find((item) => item.id === 'xing.com').credentialSecretName, 'XING_BROWSER_LOGIN');
  assert.equal(hooks.SOURCE_DEFS.find((item) => item.id === 'rocketreach.com').url, 'https://rocketreach.co/');
});

test('access state is derived from persisted records, never from a hardcoded account list', () => {
  // The old ACCOUNT_DEFS literal and its renderer are deleted; one truth per source.
  assert.doesNotMatch(source, /ACCOUNT_DEFS/);
  assert.doesNotMatch(source, /renderAccountInventory/);
  assert.doesNotMatch(source, /accountStatusLabel|adapterStatusLabel/);
  assert.match(source, /function sourceStatus\(item, adapter/);
  // LinkedIn existed only inside the deleted array and is not a registered source.
  assert.ok(!hooks.SOURCE_DEFS.some((item) => item.id === 'linkedin.com'));
  assert.doesNotMatch(source, /'linkedin\.com'/);
  // The old hardcoded login hints are gone; no secrets were ever embedded.
  assert.doesNotMatch(source, /phillip\.thesen@etis-gmbh\.de/);
  assert.doesNotMatch(source, /stefanie\.pechacek@thesen-ag\.com/);
  assert.doesNotMatch(source, /thesen@thesen-ag\.com/);
  assert.doesNotMatch(source, /Arbeit2025|thesenag#2020|ThesenBurg1#/);
});

test('source panel fuses sources and access into one tab with one derived status per source', () => {
  assert.match(source, /data-action="source-view" data-view="sources"/);
  assert.match(source, /data-action="source-view" data-view="policy"/);
  assert.doesNotMatch(source, /data-view="accounts"/);
  assert.match(source, /Quellen & Zugänge/);
  assert.match(source, /Status unbekannt — diese Quelle wurde noch nie geprüft\./);
  assert.match(source, /Zugang abgelehnt/);
  assert.match(source, /Zugang fehlt/);
  assert.doesNotMatch(source, /Adapter geprüft|Adapter offen/);
});

test('source status line and credential chip are both derived from the same adapter and source record', () => {
  const item = { id: 'xing.com', label: 'XING', enabled: true, requires_credential: true, countries: ['DE'], field_keys: [] };
  const adapter = {
    source_id: 'xing.com',
    status: 'test_ok',
    scrape_status: 'test_executed',
    auth_status: 'credential_available',
    updated_at_ms: Date.now(),
  };
  const ready = hooks.sourceStatus(item, adapter);
  assert.equal(ready.code, 'ready');
  assert.match(ready.label, /^Bereit · zuletzt geprüft am /);
  assert.deepEqual(ready.chip, { code: 'available', label: 'Zugang hinterlegt' });

  const missing = hooks.sourceStatus(item, { ...adapter, auth_status: 'required' });
  assert.equal(missing.code, 'credential_missing');
  assert.deepEqual(missing.chip, { code: 'missing', label: 'Zugang fehlt' });

  const rejected = hooks.sourceStatus(item, { ...adapter, auth_status: 'invalid_credentials' });
  assert.equal(rejected.code, 'credential_rejected');
  assert.equal(rejected.chip.code, 'invalid');
  assert.match(rejected.label, /^Zugang abgelehnt/);

  const openSource = hooks.sourceStatus({ ...item, requires_credential: false }, adapter);
  assert.equal(openSource.code, 'ready');
  assert.equal(openSource.chip, null);

  const unknown = hooks.sourceStatus({ id: 'neu.example', enabled: true, requires_credential: false }, null);
  assert.equal(unknown.code, 'unknown');
  assert.match(unknown.label, /noch nie geprüft/);
});

test('workspace exposes campaign creation, research settings, and table or shard views', () => {
  assert.match(source, /data-action="new-campaign"/);
  assert.match(source, /data-action="open-policy"/);
  assert.match(source, /data-action="view-mode" data-view="shards"/);
  assert.match(source, /data-action="view-mode" data-view="table"/);
  assert.match(source, /function createCampaign/);
  assert.match(source, /function saveResearchPolicy/);
  assert.match(source, /RESEARCH_POLICY_ID/);
  assert.match(source, /Verbindlicher Rechercheablauf/);
});

test('research policy uses its authoritative collection and preserves the backend contract', () => {
  const policySchema = collectionsManifest.collections.thesen_outbound_research_policies;
  assert.equal(policySchema.version, 0);
  assert.equal(policySchema.primaryKey, 'id');
  assert.equal(policySchema.additionalProperties, true);
  for (const field of [
    'id', 'title', 'version_number', 'status', 'skill_name', 'skill_version',
    'min_independent_sources', 'rules', 'created_at_ms', 'updated_at_ms',
  ]) {
    assert.ok(policySchema.properties[field], `missing research policy field ${field}`);
  }
  assert.ok(moduleManifest.collections.includes('thesen_outbound_research_policies'));
  assert.match(schemaSource, /thesen_outbound_research_policies:\s*researchPolicySchema/);
  assert.match(source, /researchPolicies:\s*ctx\.db\.collection\(['"]thesen_outbound_research_policies['"]\)/);
  assert.match(source, /state\.collections\.researchPolicies\.findOne\(RESEARCH_POLICY_ID\)/);
  assert.doesNotMatch(source, /state\.collections\.imports\.findOne\(RESEARCH_POLICY_ID\)/);

  const rules = [{
    id: 'new_record_DE_firma_name',
    mode: 'new_record',
    country: 'DE',
    field_key: 'firma_name',
    source_ids: ['impressum', 'northdata.de', 'handelsregister.de'],
    enabled: true,
    min_sources: 2,
  }];
  const existing = {
    id: 'thesen_research_policy_v1',
    title: 'THESEN Quellenstandard',
    version_number: 7,
    status: 'active',
    skill_name: 'thesen-outbound-research',
    skill_version: '1.0.0',
    min_independent_sources: 3,
    rules,
    instructions: 'Bisheriger Ablauf',
    created_at_ms: 100,
    updated_at_ms: 200,
  };
  const saved = hooks.researchPolicyRecord(existing, '  Neuer verbindlicher Ablauf  ', 300);
  assert.equal(saved.id, 'thesen_research_policy_v1');
  assert.equal(saved.instructions, 'Neuer verbindlicher Ablauf');
  assert.equal(saved.version_number, 8);
  assert.equal(saved.min_independent_sources, 3);
  assert.equal(saved.skill_name, existing.skill_name);
  assert.equal(saved.skill_version, existing.skill_version);
  assert.deepEqual(saved.rules, rules);
  assert.equal(saved.created_at_ms, 100);
  assert.equal(saved.updated_at_ms, 300);
  assert.equal(hooks.researchPolicyInstructions(saved), 'Neuer verbindlicher Ablauf');
});

test('workspace exposes campaign maintenance, lead editing, and scoped multi-selection', () => {
  assert.match(source, /data-action="rename-campaign"/);
  assert.match(source, /data-action="delete-campaign"/);
  assert.match(source, /showBusinessConfirm/);
  assert.match(source, /data-action="edit-lead"/);
  assert.match(source, /data-lead-edit-field/);
  assert.match(source, /data-action="toggle-visible-leads"/);
  assert.match(source, /data-action="toggle-lead"/);
  assert.match(source, /data-action="research-selection"/);
  assert.match(source, /selectionAnchorId/);
  assert.match(source, /event\.shiftKey/);
  assert.match(source, /function startScopedResearch/);
});

test('source and lead dialogs close only from their backdrop or explicit close buttons', () => {
  assert.match(
    source,
    /action === 'close-sources' && \(event\.target === trigger \|\| trigger\.matches\('button\[data-action="close-sources"\]'\)\)/,
  );
  assert.match(
    source,
    /action === 'close-lead-editor' && \(event\.target === trigger \|\| trigger\.matches\('button\[data-action="close-lead-editor"\]'\)\)/,
  );
  assert.doesNotMatch(source, /trigger === event\.target\.closest\('\[data-action\]'\)/);
});

test('fresh browsers synchronize every private outbound collection before the first data render', () => {
  for (const collection of [
    'thesen_outbound_sources',
    'thesen_outbound_adapters',
    'thesen_outbound_imports',
    'thesen_outbound_research_policies',
    'thesen_outbound_leads',
    'business_commands',
  ]) {
    assert.match(source, new RegExp(`['"]${collection}['"]`));
  }
  assert.match(source, /REPLICATED_COLLECTIONS\.map/);
  assert.match(source, /state\.ctx\.sync\?\.startCollection\?\.\(collection\)/);
  assert.match(source, /Promise\.allSettled\(replicationBridges\.map/);
  assert.match(source, /waitForReplicationBridge\(bridge, collection\)/);
  assert.match(source, /synchronizeInitialData\(\)\.catch/);
  assert.match(source, /Daten werden synchronisiert\./);
});

test('green adapter state requires a registered or tested scraper', () => {
  assert.equal(hooks.adapterReady({ status: 'draft', scrape_status: 'target_available' }), false);
  assert.equal(hooks.adapterReady({ status: 'adapter_ready', scrape_status: 'script_required' }), false);
  assert.equal(hooks.adapterReady({ status: 'adapter_ready', scrape_status: 'registered' }), true);
  assert.equal(hooks.adapterReady({ status: 'test_ok', scrape_status: 'test_executed' }), true);
});

test('browser authorization is offered for protected and challenge-blocked sources', () => {
  assert.equal(hooks.sourceNeedsBrowserAuthorization(
    { requires_credential: true },
    { status: 'test_auth_required', scrape_status: 'auth_required', auth_status: 'required' },
  ), true);
  assert.equal(hooks.sourceNeedsBrowserAuthorization(
    { requires_credential: false },
    { status: 'test_auth_required', scrape_status: 'blocked', auth_status: 'required' },
  ), true);
  assert.equal(hooks.sourceNeedsBrowserAuthorization(
    { requires_credential: false },
    { status: 'test_ok', scrape_status: 'test_executed', auth_status: 'not_required' },
  ), false);
  assert.equal(
    hooks.rxdbIdSlug('cmd_thesen_source_2814ef73-c432-4b16-89c2-2649d953fed2'),
    'cmd_thesen_source_2814ef73_c432_4b16_89c2_2649d953fed2',
  );
});

test('lead validation opens once a company name is known and research is idle', () => {
  const oneSource = [{ field_key: 'firma_name', source_id: 'northdata.de', source_url: 'https://www.northdata.de/example' }];
  const duplicateDomain = [...oneSource, { field_key: 'firma_name', source_url: 'https://northdata.de/another' }];
  const twoSources = [...oneSource, { field_key: 'firma_name', source_id: 'handelsregister.de', source_url: 'https://www.handelsregister.de/example' }];
  assert.equal(hooks.independentEvidenceCount({ evidence: duplicateDomain }), 1);
  assert.equal(hooks.independentFieldEvidenceCount({ evidence: twoSources }, 'firma_name'), 2);
  const named = { research_status: 'needs_review', name: 'Beispiel GmbH', payload: { researched_field_keys: ['firma_name'] } };
  assert.equal(hooks.leadReadyForValidation({ ...named, research_status: 'queued' }), false);
  assert.equal(hooks.leadReadyForValidation({ ...named, research_status: 'running' }), false);
  assert.equal(hooks.leadReadyForValidation({ ...named, evidence: oneSource }), true);
  assert.equal(hooks.leadReadyForValidation({ research_status: 'needs_review', name: '', data: {}, payload: {} }), false);
});

test('field evidence is grouped into compact, independent source rows', () => {
  const groups = hooks.sourceEvidenceGroups({
    evidence: [
      { field_key: 'firma_name', source_id: 'northdata.de', source_url: 'https://www.northdata.de/example' },
      { field_key: 'firma_ort', source_id: 'northdata.de', source_url: 'https://www.northdata.de/example' },
      { field_key: 'firma_name', source_id: 'handelsregister.de', source_url: 'https://www.handelsregister.de/example' },
    ],
  });
  assert.equal(groups.length, 2);
  assert.equal(groups.find((group) => group.key === 'northdata.de').fieldCount, 2);
  assert.equal(groups.find((group) => group.key === 'handelsregister.de').fieldCount, 1);
});

test('review field with two distinct source ids counts as sufficiently evidenced', () => {
  const lead = {
    data: { firma_anschrift: 'PC-Str. 1' },
    contacts: [],
    evidence: [
      { field: 'firma_anschrift', value: 'PC-Str. 1', confidence: 'high', source_id: 'northdata.de', source_url: 'https://www.northdata.de/example' },
      { field: 'firma_anschrift', value: 'PC-Str. 1', confidence: 'medium', source_id: 'handelsregister.de', source_url: 'https://www.handelsregister.de/example' },
    ],
  };
  const review = hooks.researchFieldReview(lead);
  const field = review.fields.find((item) => item.key === 'firma_anschrift');
  assert.equal(field.value, 'PC-Str. 1');
  assert.equal(field.filled, true);
  assert.equal(field.independentCount, 2);
  assert.equal(field.sufficient, true);
  assert.ok(!review.missingEvidenceKeys.includes('firma_anschrift'));
});

test('review field with two entries from the same source id is not sufficient', () => {
  const lead = {
    research_status: 'completed',
    data: { firma_anschrift: 'PC-Str. 1' },
    contacts: [],
    evidence: [
      { field_key: 'firma_anschrift', value: 'PC-Str. 1', source_id: 'northdata.de', source_url: 'https://www.northdata.de/first' },
      { field_key: 'firma_anschrift', value: 'PC-Str. 1', source_id: 'northdata.de', source_url: 'https://www.northdata.de/second' },
    ],
  };
  const review = hooks.researchFieldReview(lead);
  const field = review.fields.find((item) => item.key === 'firma_anschrift');
  assert.equal(field.filled, true);
  assert.equal(field.independentCount, 1);
  assert.equal(field.sufficient, false);
  assert.ok(review.missingEvidenceKeys.includes('firma_anschrift'));
  assert.ok(hooks.validationBlockers(lead).some((line) => line.includes('Firmenname fehlt')));
  assert.equal(hooks.validationBlockers({ ...lead, name: 'Beispiel GmbH' }).length, 0);
});

test('unfilled review field renders compact with fehlt badge', () => {
  const html = hooks.renderResearchReview({ data: {}, contacts: [], evidence: [], research_status: 'needs_review' });
  assert.match(html, /fehlt/);
  assert.ok(!/Vertrauen:/.test(html), 'keine Vertrauens-Prosa in der Feldkarte');
});


test('detail pane reviews all 21 research fields in three groups and keeps scroll position', () => {
  const keys = hooks.RESEARCH_FIELD_GROUPS.flatMap((group) => group.fields.map(([key]) => key));
  assert.deepEqual(keys, [
    'firma_name', 'firma_anschrift', 'firma_plz', 'firma_ort', 'firma_email',
    'firma_domain', 'firma_telefon', 'wz_code', 'umsatz', 'mitarbeiter',
    'person_geschlecht', 'person_titel', 'person_vorname', 'person_nachname',
    'person_funktion', 'person_position', 'person_email', 'person_email_validation',
    'person_telefon', 'person_linkedin', 'person_xing',
  ]);
  assert.deepEqual(hooks.RESEARCH_FIELD_GROUPS.map((group) => group.label), ['Unternehmen', 'Klassifikation', 'Ansprechpartner']);
  const html = hooks.renderResearchReview({ data: {}, contacts: [], evidence: [] });
  for (const key of keys) assert.ok(html.includes(`data-review-field="${key}"`), `missing review row for ${key}`);
  assert.match(html, /0 von 21 Feldern ermittelt/);
  assert.match(source, /const scroller = pane\.querySelector\('\.thesen-detail-body'\)/);
  assert.match(source, /nextScroller\.scrollTop = scrollTop/);
  assert.match(source, /Noch nicht freigebbar/);
  assert.match(source, /data-action="research-lead"/);
  assert.match(source, /data-action="validate-lead"/);
  // The single "Zu Sellify" button was split: the operator now chooses whether
  // the handover only updates the record or also adds it to the campaign.
  assert.match(source, /data-action="sellify-update-only"/);
  assert.match(source, /data-action="sellify-update-campaign"/);
  assert.match(source, /data-action="mail-series-email"/);
  assert.match(source, /openSeriesEmailFromLead/);
  assert.match(source, /data-action="edit-lead"/);
});

test('Sellify recipient selection hands eligible addresses to Mail series email', () => {
  const lead = {
    id: 'lead-mail', campaign: 'August-Welle',
    contacts: [
      { id: 'a', email: 'A@Example.test' },
      { id: 'b', person_email: 'b@example.test' },
      { id: 'blocked', email: 'blocked@example.test' },
    ],
    selected_contact_ids: ['a', 'b', 'blocked'],
  };
  const decisions = new Map([
    ['a', { status: 'free' }], ['b', { status: 'free' }], ['blocked', { status: 'blocked' }],
  ]);
  const handoff = hooks.seriesEmailHandoffForLead(lead, decisions);
  assert.deepEqual(handoff.recipients, ['a@example.test', 'b@example.test']);
  assert.match(handoff.hash, /^#mail\?action=series-email&source_module=sellify/);
  assert.match(handoff.hash, /recipients=a%40example\.test%2Cb%40example\.test/);
  assert.equal(handoff.excluded.length, 1);
});

test('native person research result writes typed values, contacts, and field evidence', () => {
  const candidate = (sourceId, value) => ({
    value,
    confidence: 'high',
    source_id: sourceId,
    source_url: `https://${sourceId}/record`,
    tier: 'P',
    via: 'scrape_target',
  });
  const result = hooks.researchOutcomeWriteback({
    data: {},
    contacts: [],
    evidence: [],
    payload: {},
  }, {
    status: 'completed',
    result: {
      tool: 'ctox_person_research',
      fields: {
        firma_name: {
          value: 'Example GmbH',
          candidates: [
            candidate('handelsregister.de', 'Example GmbH'),
            candidate('northdata.de', 'Example GmbH'),
          ],
        },
        person_vorname: {
          value: 'Ada',
          candidates: [
            candidate('xing.com', 'Ada'),
            candidate('rocketreach.com', 'Ada'),
          ],
        },
        person_nachname: {
          value: 'Lovelace',
          candidates: [
            candidate('xing.com', 'Lovelace'),
            candidate('rocketreach.com', 'Lovelace'),
          ],
        },
      },
    },
  });
  assert.equal(result.research_status, 'completed');
  assert.equal(result.data.firma_name, 'Example GmbH');
  assert.equal(result.contacts[0].name, 'Ada Lovelace');
  assert.equal(result.evidence.length, 6);
  assert.deepEqual(result.payload.unverified_field_keys, []);
  assert.deepEqual(result.payload.verified_field_keys.sort(), [
    'firma_name',
    'person_nachname',
    'person_vorname',
  ]);
});

test('terminal research control command is deterministically reconciled without free-text placeholders', () => {
  const candidate = (sourceId, value) => ({
    value,
    confidence: 'high',
    source_id: sourceId,
    source_url: `https://${sourceId}/record`,
    tier: 'P',
    via: 'scrape_target',
  });
  const patch = hooks.researchCommandLeadPatch({
    command_id: 'cmd_research_1',
    data: {},
    contacts: [],
    evidence: [],
    payload: {},
  }, {
    id: 'cmd_research_1',
    command_id: 'cmd_research_1',
    command_type: 'web_stack.person_research',
    terminal_status: 'completed',
    result: {
      ok: true,
      tool: 'ctox_person_research',
      mode: 'new_record',
      fields: {
        firma_name: {
          value: 'Example GmbH',
          candidates: [
            candidate('handelsregister.de', 'Example GmbH'),
            candidate('northdata.de', 'Example GmbH'),
          ],
        },
        firma_email: { value: null, candidates: [] },
      },
    },
  });

  assert.equal(patch.research_status, 'completed');
  assert.equal(patch.data.firma_name, 'Example GmbH');
  assert.equal(patch.data.firma_email, undefined);
  assert.deepEqual(patch.contacts, []);
  assert.equal(patch.payload.reconciled_research_command_id, 'cmd_research_1');
  assert.equal(JSON.stringify(patch).includes('</email>'), false);
});

test('Sellify handoff preserves SQL source of truth and allows typed commands only', () => {
  const contract = hooks.sellifyWritebackContract();
  assert.equal(contract.source_of_truth, 'sellify-original-sql');
  assert.equal(contract.direct_sql_write_allowed, false);
  assert.deepEqual(contract.allowed_commands, ['external_sql.write']);
  assert.ok(contract.allowed_operations.includes('company_create'));
  assert.ok(contract.allowed_operations.includes('company_update'));
  assert.ok(contract.allowed_operations.includes('person_create'));
  assert.ok(contract.allowed_operations.includes('person_update'));
  assert.ok(contract.required_result.includes('deduplication_decision'));
});

test('Sellify handoff maps researched company and person fields to typed SQL operations', () => {
  assert.deepEqual(hooks.sellifyCompanyValues({
    name: 'Acme GmbH',
    website: 'https://acme.example',
    city: 'Köln',
    country: 'DE',
    data: {
      email: 'info@acme.example',
      telefon: '+49 221 123',
      strasse: 'Domstraße 1',
      plz: '50667',
    },
  }), {
    name: 'Acme GmbH',
    short_name: '',
    number1: '',
    number2: '',
    email: 'info@acme.example',
    phone: '+49 221 123',
    fax: '',
    website_url: 'https://acme.example',
    address_line: 'Domstraße 1',
    postal_code: '50667',
    city: 'Köln',
    country_code: 'DE',
  });
  assert.equal(hooks.sellifyPersonValues({
    name: 'Ada Lovelace',
    role: 'Geschäftsführung',
    email: 'ada@acme.example',
  }).last_name, 'Lovelace');
});

test('runtime uses scoped collections and tracked business chat tasks without HTTP data paths', () => {
  assert.match(source, /thesen_outbound_sources/);
  assert.match(source, /thesen_outbound_leads/);
  assert.match(source, /commandBus\.dispatch/);
  assert.match(source, /businessChat\.submitTask/);
  assert.match(source, /requireTrackedSubmission/);
  assert.match(source, /web_stack\.person_research/);
  assert.match(source, /fields:\s*\[\.\.\.RESEARCH_FIELDS\]/);
  assert.match(source, /person_email_validation/);
  assert.match(source, /person_linkedin/);
  assert.match(source, /external_sql\.write/);
  assert.match(source, /findSellifyCompanyDuplicate/);
  assert.match(source, /waitForProjectedRecord/);
  assert.match(source, /until:\s*["']terminal["']/);
  assert.match(source, /response_channel:\s*["']business_os_chat["']/);
  assert.match(source, /thesen-outbound-research/);
  assert.match(source, /web-unlock/);
  assert.doesNotMatch(source, /fetch\s*\(\s*["']\/(?:api|rxdb|business-os)/);
  assert.match(source, /ctx\.db\.collection\(['"]business_commands['"]\)/);
  assert.match(source, /data-context-record-id/);
  assert.match(source, /data-context-record-type/);
  assert.match(source, /data-context-label/);
  assert.match(source, /Browser-Anmeldung angefordert/);
  assert.match(source, /data-thesen-outbound-styles/);
  assert.match(source, /classList\.add\(["']thesen-outbound["']\)/);
});

test('adapter source actions submit typed operations instead of orchestration prompts', () => {
  for (const commandType of [
    'outbound.research_source.generate_adapter',
    'outbound.research_source.test',
  ]) {
    assert.deepEqual(
      hooks.adapterCommandOperation(commandType, 'handelsregister-de'),
      { operation: commandType, target_key: 'handelsregister-de' },
    );
  }
  assert.throws(
    () => hooks.adapterCommandOperation('outbound.research_source.test', ''),
    /Operation und Ziel/,
  );

  const start = source.indexOf('async function runAdapterCommand');
  const end = source.indexOf('async function openSourceAuthorization', start);
  const adapterCommandSource = source.slice(start, end);
  assert.match(adapterCommandSource, /const operation = adapterCommandOperation\(commandType, item\.target_key\)/);
  assert.match(adapterCommandSource, /payload:\s*\{\s*\.\.\.operation,/);
  assert.match(adapterCommandSource, /text:\s*operationText/);
  assert.match(adapterCommandSource, /instruction:\s*operationText/);
  assert.match(adapterCommandSource, /user_message:\s*operationText/);
  assert.doesNotMatch(adapterCommandSource, /Verwende den THESEN-Recherche-Standard/);
  assert.doesNotMatch(adapterCommandSource, /\bprompt\b/);
});

test('generic adapter control commands do not request a private cross-module writeback', () => {
  assert.match(source, /module:\s*['"]outbound['"]/);
  assert.doesNotMatch(
    source,
    /writeback:\s*\{\s*collection:\s*['"]thesen_outbound_adapters['"]/,
  );
});

test('terminal adapter controls are tracked by command id without inventing a queue task', () => {
  assert.deepEqual(
    hooks.requireTrackedSubmission(
      { status: 'completed', command_id: 'cmd_adapter', task_id: '' },
      { allowTerminalCommand: true },
    ),
    { status: 'completed', command_id: 'cmd_adapter', task_id: '' },
  );
  assert.throws(
    () => hooks.requireTrackedSubmission({ status: 'queued', command_id: 'cmd_adapter', task_id: '' }),
    /keinen verfolgbaren Task/,
  );
  assert.throws(
    () => hooks.requireTrackedSubmission(
      { status: 'queued', command_id: 'cmd_adapter', task_id: '' },
      { allowTerminalCommand: true },
    ),
    /keinen verfolgbaren Task/,
  );
  assert.deepEqual(
    hooks.requireTrackedSubmission(
      { status: 'accepted', command_id: 'cmd_adapter', task_id: '' },
      { allowControlCommand: true },
    ),
    { status: 'accepted', command_id: 'cmd_adapter', task_id: '' },
  );
});

test('terminal adapter command projection replaces optimistic adapter state', () => {
  const patch = hooks.adapterCommandRecordPatch({
    id: 'adapter_thesen_bundesanzeiger-de',
    source_id: 'bundesanzeiger.de',
    status: 'test_requested',
    scrape_status: 'test_requested',
    auth_status: 'not_required',
    payload: {},
  }, {
    id: 'cmd_adapter',
    status: 'completed',
    result: {
      status: 'completed',
      adapter: {
        status: 'test_zero_records',
        scrape_status: 'test_zero_records',
        auth_status: 'not_required',
      },
    },
  });
  assert.equal(patch.status, 'test_zero_records');
  assert.equal(patch.scrape_status, 'test_zero_records');
  assert.equal(patch.payload.reconciled_command_id, 'cmd_adapter');
  assert.equal(hooks.adapterReady(patch), false);
  assert.equal(hooks.adapterCommandRecordPatch(
    { payload: { reconciled_command_id: 'cmd_adapter' } },
    { id: 'cmd_adapter', status: 'completed' },
  ), null);
});

test('campaign progress exposes all five user-visible states without double counting', () => {
  const leads = [
    { campaign: 'Chemie 2026', research_status: 'new', validation_status: 'pending' },
    { campaign: 'Chemie 2026', research_status: 'running', validation_status: 'pending' },
    { campaign: 'Chemie 2026', research_status: 'completed', validation_status: 'pending' },
    { campaign: 'Chemie 2026', research_status: 'failed', validation_status: 'pending' },
    { campaign: 'Chemie 2026', research_status: 'completed', validation_status: 'validated' },
    { campaign: 'Andere', research_status: 'completed', validation_status: 'validated' },
  ];
  const progress = hooks.campaignResearchProgress('Chemie 2026', leads);
  assert.deepEqual(progress.counts, {
    new: 1,
    queued: 0,
    running: 1,
    completed: 1,
    failed: 1,
    validated: 1,
  });
  assert.equal(progress.total, 5);
  assert.equal(progress.processed, 3);
  assert.equal(progress.percent, 60);
  assert.equal(progress.status, 'running');
});

test('completed, failed, and validated campaign summaries are derived from lead truth', () => {
  const lead = (research_status, validation_status = 'pending') => ({
    campaign: 'C',
    research_status,
    validation_status,
  });
  assert.equal(hooks.campaignResearchProgress('C', [lead('completed')]).status, 'completed');
  assert.equal(hooks.campaignResearchProgress('C', [lead('failed')]).status, 'failed');
  assert.equal(hooks.campaignResearchProgress('C', [lead('completed', 'validated')]).status, 'validated');
});

test('campaign queue preserves order and retries only leads that still need research', () => {
  const queue = hooks.campaignResearchQueue([
    { id: 'new', research_status: 'new', validation_status: 'pending' },
    { id: 'done', research_status: 'completed', validation_status: 'pending' },
    { id: 'review', research_status: 'needs_review', validation_status: 'pending' },
    { id: 'failed', research_status: 'failed', validation_status: 'pending' },
    { id: 'validated', research_status: 'completed', validation_status: 'validated' },
    { id: 'stale-running', research_status: 'running', validation_status: 'pending' },
  ]);
  assert.deepEqual(queue, ['new', 'review', 'failed', 'stale-running']);
  const stale = hooks.campaignResearchProgress('C', [{
    campaign: 'C',
    research_status: 'running',
    validation_status: 'pending',
  }]);
  assert.equal(stale.status, 'running');
  assert.equal(stale.active, false);
});

test('a new import starts a fresh research period for an existing lead', () => {
  const lifecycle = hooks.resetLeadForImport({
    research_status: 'completed',
    validation_status: 'validated',
    sellify_status: 'completed',
    task_id: 'task_old',
    command_id: 'command_old',
    evidence: [{ field: 'firma_name', source_id: 'old-source' }],
  });
  assert.deepEqual(lifecycle, {
    research_status: 'new',
    validation_status: 'pending',
    sellify_status: 'not_started',
    task_id: '',
    command_id: '',
    // A re-import starts a fresh research period, so the operator's previous
    // recipient choice must not silently carry over into the new one.
    selected_contact_ids: [],
    evidence: [],
  });
});

test('campaign task prompt states scope, validation, unlocking, and controlled serial processing', () => {
  const prompt = hooks.campaignResearchPrompt('Chemie Testimport 2026', [
    { id: 'lead_1', name: 'Additiv-Chemie Luers GmbH & Co. KG' },
    { id: 'lead_2', name: 'Aeroxon Insect Control GmbH' },
  ], 'run_1');
  assert.match(prompt, /thesen-outbound-research/);
  assert.match(prompt, /Kampagne: Chemie Testimport 2026/);
  assert.match(prompt, /web_stack\.person_research/);
  assert.match(prompt, /"company":"<Lead-Name>"/);
  assert.match(prompt, /Verwende company, niemals company_name/);
  assert.match(prompt, /running-Status ist ausdrücklich kein Erfolg/);
  assert.match(prompt, /seriell/);
  assert.match(prompt, /mindestens zwei unabhängige Quellen/);
  assert.match(prompt, /Web-Stack-Unlocking/);
  assert.match(prompt, /keine SQL-Schreiboperation/i);
  assert.match(prompt, /lead_1/);
  assert.match(prompt, /lead_2/);
});

test('campaign reconciliation matches the latest child research command by lead id', () => {
  const lead = {
    id: 'lead_1',
    command_id: 'parent_campaign_command',
    payload: { campaign_run_id: 'run_1' },
  };
  const command = hooks.researchCommandForLead(lead, [
    {
      id: 'wrong_type',
      command_type: 'business_os.chat.task',
      record_id: 'lead_1',
      updated_at_ms: 30,
    },
    {
      id: 'old_child',
      command_type: 'web_stack.person_research',
      record_id: 'lead_1',
      payload: { campaign_run_id: 'run_1' },
      updated_at_ms: 10,
    },
    {
      id: 'latest_child',
      command_type: 'web_stack.person_research',
      record_id: 'lead_1',
      payload: { campaign_run_id: 'run_1' },
      updated_at_ms: 20,
    },
    {
      id: 'other_run',
      command_type: 'web_stack.person_research',
      record_id: 'lead_1',
      payload: { campaign_run_id: 'run_2' },
      updated_at_ms: 40,
    },
  ]);
  assert.equal(command.id, 'latest_child');
});

test('campaign reconciliation accepts workflow_id and rejects another campaign run', () => {
  const lead = { id: 'lead_1', payload: { campaign_run_id: 'run_1' } };
  const command = hooks.researchCommandForLead(lead, [
    {
      id: 'current',
      command_type: 'web_stack.person_research',
      record_id: 'lead_1',
      payload: { workflow_id: 'run_1' },
      updated_at_ms: 10,
    },
    {
      id: 'newer_other_run',
      command_type: 'web_stack.person_research',
      record_id: 'lead_1',
      payload: { workflow_id: 'run_2' },
      updated_at_ms: 20,
    },
  ]);
  assert.equal(command.id, 'current');
});

test('accepted child command is the only state that moves a queued lead to running', () => {
  const queued = { command_id: 'parent', task_id: 'campaign-task', payload: {} };
  const patch = hooks.researchCommandLeadPatch(queued, {
    id: 'child',
    command_type: 'web_stack.person_research',
    record_id: 'lead_1',
    status: 'accepted',
    task_status: 'queued',
    terminal_status: 'none',
  });
  assert.equal(patch.research_status, 'running');
  assert.equal(patch.command_id, 'child');
  assert.equal(patch.task_id, 'campaign-task');
});

test('non-terminal sentinel never masks the accepted command status', () => {
  assert.equal(hooks.normalizedResearchCommandStatus({
    terminal_status: 'none',
    task_status: 'queued',
    status: 'accepted',
    result: { status: 'running' },
  }), 'accepted');
  assert.equal(hooks.normalizedResearchCommandStatus({
    terminal_status: 'completed',
    status: 'accepted',
  }), 'completed');
});

test('lead research maps app state to a typed web-stack research mode', () => {
  assert.equal(hooks.leadResearchMode({ sellify_status: 'not_started' }), 'new_record');
  assert.equal(hooks.leadResearchMode({ sellify_status: 'completed' }), 'update_firm');
  assert.doesNotMatch(source, /command_type:\s*['"]web_stack\.person_research['"][\s\S]{0,500}mode:\s*['"]data['"]/);
  assert.match(source, /command_type:\s*['"]web_stack\.person_research['"]/);
  assert.match(source, /command_type:\s*['"]web_stack\.person_research['"][\s\S]{0,300}control_command:\s*true/);
  assert.match(source, /allowControlCommand:\s*true/);
  assert.match(source, /campaign_run_id:\s*options\.campaignRunId/);
  assert.match(source, /requireTrackedSubmission\(await state\.ctx\.businessChat\.submitTask/);
  assert.match(source, /reconcileResearchCommands/);
});

test('terminal research reconciliation notifies the global business chat', () => {
  assert.match(source, /ctox-business-os-command-observed/);
  assert.match(source, /window\.top/);
  assert.match(source, /postMessage/);
  assert.ok(source.indexOf('chatEventTarget.postMessage') < source.indexOf('chatEventTarget.dispatchEvent'));
  assert.match(source, /if \(chatEventTarget === window\)/);
  assert.match(source, /detail:\s*\{\s*command\s*\}/);
});

test('campaign start submits a new validated CTOX task immediately', async () => {
  const submitted = [];
  const detail = await hooks.openCampaignResearchChat({
    campaign: 'Chemie Testimport 2026',
    leads: [
      { id: 'lead_1', name: 'Additiv-Chemie Luers GmbH & Co. KG' },
      { id: 'lead_2', name: 'Aeroxon Insect Control GmbH' },
    ],
    runId: 'run_1',
    prompt: 'Präziser Kampagnenauftrag mit validiertem Scope.',
    async submitTask(value) {
      submitted.push(value);
      return { status: 'queued', task_id: 'task_1', command_id: 'run_1' };
    },
  });
  assert.equal(submitted.length, 1);
  assert.equal(detail.task_id, 'task_1');
  assert.equal(detail.command_id, 'run_1');
  assert.equal(submitted[0].action, 'context-chat');
  assert.equal(submitted[0].reuseActive, false);
  assert.equal(submitted[0].open, true);
  assert.equal(submitted[0].command_type, 'business_os.chat.task');
  assert.equal(submitted[0].record_id, 'campaign:Chemie Testimport 2026');
  assert.deepEqual(submitted[0].required_skills, [
    'thesen-outbound-research',
    'universal-scraping',
    'web-unlock',
  ]);
  assert.deepEqual(submitted[0].writeback_contract.record_ids, ['lead_1', 'lead_2']);
  assert.equal(submitted[0].writeback_contract.min_independent_sources, 2);
  assert.equal(submitted[0].payload.response_channel, 'business_os_chat');
  assert.match(submitted[0].payload.thread_key, /run_1/);
});

test('campaign task refuses invalid scope or a missing chat API', async () => {
  await assert.rejects(() => hooks.openCampaignResearchChat({
    campaign: '',
    leads: [],
    runId: 'run_1',
    prompt: 'Prompt',
    submitTask() {},
  }), /Kampagne|campaign/i);
  await assert.rejects(() => hooks.openCampaignResearchChat({
    campaign: 'Chemie 2026',
    leads: [{ id: 'lead_1', name: 'Acme GmbH' }],
    runId: 'run_1',
    prompt: 'Prompt',
    submitTask: null,
  }), /Chat/i);
  await assert.rejects(() => hooks.openCampaignResearchChat({
    campaign: 'Chemie 2026',
    leads: [{ id: 'lead_1', name: 'Acme GmbH' }],
    runId: 'run_1',
    prompt: 'Prompt',
    async submitTask() { return { status: 'queued', command_id: 'run_1', task_id: '' }; },
  }), /verfolgbaren Task/i);
});

test('campaign controller submits recoverable native research commands with one tracked chat per lead', () => {
  assert.match(source, /class="ctox-button thesen-campaign-research"/);
  assert.match(source, /const campaignAction = campaignProgress\.trackingTaskId \? 'track-task' : 'research-campaign'/);
  assert.match(source, /data-action="\$\{campaignAction\}"/);
  assert.match(source, /businessChat\?\.open\?\.\(\{/);
  assert.match(source, /focus:\s*\{\s*task_id:\s*normalizedTaskId,\s*command_id:\s*normalizedCommandId/);
  assert.match(source, /aria-label=/);
  assert.match(source, /async function startCampaignResearch/);
  assert.match(source, /for \(const lead of eligibleLeads\)/);
  assert.match(source, /await researchLead\(lead\.id,\s*\{/);
  assert.match(source, /campaignRunId:\s*runId/);
  assert.match(source, /threadKey:\s*`business-os\/thesen-outbound\/campaign\/\$\{runId\}\/lead\/\$\{lead\.id\}`/);
  assert.match(source, /businessChat\?\.submitTask/);
  assert.match(source, /open:\s*true/);
  assert.doesNotMatch(source, /processCampaignResearch/);
  assert.doesNotMatch(source, /fetch\s*\(\s*["']\/(?:api|rxdb|business-os)/);
});

test('untracked queued or running records are repaired instead of displayed as active', () => {
  assert.match(source, /async function repairUntrackedResearchStatuses/);
  assert.match(source, /research_recovery_reason:\s*['"]untracked_automation_reset['"]/);
  assert.match(source, /await repairUntrackedResearchStatuses\(\)/);
  assert.match(source, /function requireTrackedSubmission/);
});

test('reload keeps the selected lead inside the selected campaign', () => {
  assert.match(
    source,
    /const selectedCampaignLeads = campaignLeads\(state\.selectedCampaign\);[\s\S]*selectedCampaignLeads\.some\(\(lead\) => lead\.id === state\.selectedLeadId\)/,
  );
  assert.doesNotMatch(
    source,
    /state\.leads\.some\(\(lead\) => lead\.id === state\.selectedLeadId\)/,
  );
});

test('the Sellify handoff declares the campaign write and never invents SQL', () => {
  const contract = hooks.sellifyWritebackContract();
  // The source itself owns campaign_create; the app may only name it.
  assert.ok(contract.allowed_operations.includes('campaign_create'));
  assert.equal(contract.direct_sql_write_allowed, false);
  assert.deepEqual(contract.allowed_commands, ['external_sql.write']);
  assert.ok(contract.required_result.includes('campaign_ids'));
});

test('campaign membership is written per selected recipient, never per lead', () => {
  const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8');
  // One campaign write per person id, and only after the people exist.
  assert.match(source, /for \(const personId of personIds\)\s*\{\s*const campaignId = await addSellifyCampaignMember/);
  assert.match(source, /dispatchExternalSqlWrite\('campaign_create'/);
  // A handover that claims a campaign but confirms none must fail loudly.
  assert.match(source, /if \(!campaignIds\.length\) throw new Error/);
});

// --- Regression tests for the 2026-07-31 review findings -------------------
// renderDetail() referenced sellifyHandoffPrecondition and the click handler
// referenced setContactRecipientSelection, but neither function existed:
// selecting a lead crashed the detail pane and the recipient checkboxes could
// never persist a selection, so the entire Sellify flow was unreachable while
// every source-level test stayed green. These tests EXECUTE the flow.

// The real shared/dialogs.js reaches for `document` when it shows an alert, and
// the gate under test deliberately shows one. Supply the minimum DOM it touches
// so the assertion below is about the gate, not about the environment.
if (typeof globalThis.window === 'undefined') {
  globalThis.window = globalThis;
}
if (typeof globalThis.requestAnimationFrame !== 'function') {
  globalThis.requestAnimationFrame = (callback) => setTimeout(() => callback(0), 0);
  globalThis.cancelAnimationFrame = (handle) => clearTimeout(handle);
}
if (typeof globalThis.document === 'undefined') {
  const stubNode = () => ({
    style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    appendChild() {}, removeChild() {}, remove() {}, setAttribute() {},
    append() {}, prepend() {}, insertBefore() {}, getAttribute: () => null,
    addEventListener() {}, focus() {}, querySelector: () => null,
    querySelectorAll: () => [], innerHTML: '', textContent: '',
  });
  globalThis.document = {
    createElement: stubNode,
    createTextNode: () => stubNode(),
    getElementById: () => null,
    body: stubNode(),
    head: stubNode(),
    documentElement: stubNode(),
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener() {},
    removeEventListener() {},
  };
}

function flowElement() {
  return {
    innerHTML: '', scrollTop: 0, dataset: {}, children: new Map(),
    querySelector(selector) { return this.children.get(selector) || null; },
    querySelectorAll() { return []; },
    addEventListener() {},
  };
}

function flowHost(extraPanes = {}) {
  const host = flowElement();
  const panes = ['[data-campaigns-pane]', '[data-leads-pane]', '[data-detail-pane]', '[data-source-panel]', '[data-app-dialog]'];
  for (const selector of panes) host.children.set(selector, extraPanes[selector] || flowElement());
  return host;
}

function flowWorld({ campaignCreateResult } = {}) {
  const dispatches = [];
  const lead = {
    id: 'lead_flow', name: 'Destilla GmbH', campaign: 'Chemie Kampagne',
    city: 'Nördlingen', country: 'DE', domain: 'destilla.example', website: '',
    research_status: 'completed', validation_status: 'validated', sellify_status: 'not_started',
    data: {}, evidence: [], payload: {},
    contacts: [
      { id: 'contact_a', first_name: 'Ada', last_name: 'Lovelace', function: 'GF' },
      { id: 'contact_b', first_name: 'Bob', last_name: 'Bauer', function: 'Vertrieb' },
    ],
    selected_contact_ids: [],
    updated_at_ms: 0,
  };
  const collections = {
    leads: {
      find: () => ({
        exec: async () => [{
          toJSON: () => lead,
          incrementalPatch: async (patch) => { Object.assign(lead, patch); },
        }],
      }),
      findOne: () => ({
        exec: async () => ({
          toJSON: () => lead,
          incrementalPatch: async (patch) => { Object.assign(lead, patch); },
        }),
      }),
    },
  };
  const sellifyCompany = { id: 'sellify-company-11', contact_id: 11, name: 'Destilla GmbH', is_deleted: false, updated_at_ms: 1, payload: {} };
  const sellifyPerson = { id: 'sellify-person-22', person_id: 22, contact_id: 11, email: '', is_deleted: false, updated_at_ms: 1, payload: {} };
  const ctx = {
    host: flowHost(), session: {},
    db: { collection: () => ({ find: () => ({ exec: async () => [] }) }) },
    commandBus: {
      dispatch: async (command) => {
        dispatches.push(command);
        if (command.payload.operation_id === 'campaign_create' && campaignCreateResult) return campaignCreateResult;
        return { status: 'completed', result: { returned_rows: [{ contact_id: 11, person_id: 22, selection_id: 33, selectionmember_id: 44 }] } };
      },
    },
  };
  hooks.__render.setState({
    ctx, messages: {}, leads: [lead], collections,
    sellifyCompanies: { find: () => ({ exec: async () => [] }), findOne: () => ({ exec: async () => sellifyCompany }) },
    sellifyPeople: { find: () => ({ exec: async () => [] }), findOne: () => ({ exec: async () => sellifyPerson }) },
    selectedLeadId: lead.id, selectedLeadIds: new Set(), selectedCampaign: lead.campaign,
    adapters: [], sources: [], commands: [], imports: [], researchPolicies: [],
    campaignRuns: new Map(), sourcePanelOpen: false, leadEditorOpen: false,
    recipientEligibility: new Map(lead.contacts.map((contact) => [`${lead.id}|${contact.id}`, hooks.classifySellifyPerson(contact)])),
    recipientEligibilityReady: new Set([lead.id]),
    recipientEligibilitySignatures: new Map(), recipientRemovalNotices: new Map(),
    reconcilingRecipientEligibility: false,
    syncPending: false, syncError: '', search: '', sourceSearch: '',
  });
  return { dispatches, lead };
}

const flowOps = (dispatches) => dispatches.map((command) => command.payload.operation_id);

test('renderDetail with a selected lead renders the recipient checkboxes and both Sellify actions', () => {
  const world = flowWorld();
  world.lead.selected_contact_ids = ['contact_a'];
  hooks.__render.renderDetail();
  const html = hooks.__render.getState().ctx.host.querySelector('[data-detail-pane]').innerHTML;
  assert.ok(html.includes('data-action="toggle-contact-recipient"'), 'recipient checkboxes never reached the markup');
  assert.ok(html.includes('data-action="sellify-update-only"'), 'update-only button missing');
  assert.ok(html.includes('data-action="sellify-update-campaign"'), 'campaign button missing');
});

test('the recipient toggle is the manual gate: no selection, no enabled handoff', async () => {
  const world = flowWorld();
  assert.match(hooks.sellifyHandoffPrecondition(world.lead), /mindestens eine Person/);
  await hooks.setContactRecipientSelection(world.lead.id, 'contact_a', true);
  assert.deepEqual(world.lead.selected_contact_ids, ['contact_a']);
  assert.equal(hooks.sellifyHandoffPrecondition(world.lead), '');
  await hooks.setContactRecipientSelection(world.lead.id, 'contact_a', false);
  assert.deepEqual(world.lead.selected_contact_ids, []);
  assert.match(hooks.sellifyHandoffPrecondition(world.lead), /mindestens eine Person/);
});

test('a handoff with nobody selected dispatches no SQL write and leaves the lead untouched', async () => {
  const world = flowWorld();
  await hooks.sendLeadToSellify(world.lead.id, { includeCampaign: true });
  assert.equal(world.dispatches.length, 0, 'an empty recipient selection still reached Sellify');
  assert.equal(world.lead.sellify_status, 'not_started');
  await hooks.sendLeadToSellify(world.lead.id, { includeCampaign: false });
  assert.equal(world.dispatches.length, 0, 'update-only bypassed the recipient gate');
});

test('update-only and update+campaign genuinely diverge: only the latter writes campaign members', async () => {
  const only = flowWorld();
  only.lead.selected_contact_ids = ['contact_a', 'contact_b'];
  await hooks.sendLeadToSellify(only.lead.id, { includeCampaign: false });
  assert.ok(!flowOps(only.dispatches).includes('campaign_create'), 'update-only wrote a campaign');
  assert.ok(flowOps(only.dispatches).includes('company_create'), 'update-only skipped the company write');
  assert.deepEqual(only.lead.payload.sellify_campaign_ids, []);
  assert.equal(only.lead.payload.sellify_campaign_name, '');
  assert.equal(only.lead.sellify_status, 'completed');

  const withCampaign = flowWorld();
  withCampaign.lead.selected_contact_ids = ['contact_a', 'contact_b'];
  await hooks.sendLeadToSellify(withCampaign.lead.id, { includeCampaign: true });
  const memberWrites = withCampaign.dispatches.filter((command) => command.payload.operation_id === 'campaign_create');
  assert.equal(memberWrites.length, 2, 'expected one campaign write per selected recipient');
  assert.ok(memberWrites.every((command) => command.payload.name === 'Chemie Kampagne'));
  assert.ok(memberWrites.every((command) => command.payload.person_id === 22));
  assert.deepEqual(withCampaign.lead.payload.sellify_campaign_ids, [33, 33]);
  assert.equal(withCampaign.lead.payload.sellify_campaign_name, 'Chemie Kampagne');
  assert.equal(withCampaign.lead.sellify_status, 'completed');
});

test('a failed campaign assignment surfaces as a failed handoff, never as success', async () => {
  const world = flowWorld({ campaignCreateResult: { status: 'completed', result: { returned_rows: [{}] } } });
  world.lead.selected_contact_ids = ['contact_a'];
  await hooks.sendLeadToSellify(world.lead.id, { includeCampaign: true });
  assert.equal(world.lead.sellify_status, 'failed');
  assert.match(String(world.lead.payload.sellify_error), /keine Kampagnen-Zuordnung/);
});

test('a pane re-render preserves the scroll offset of its .thesen-scroll region', () => {
  const region = { current: [{ scrollTop: 142 }] };
  const pane = {
    stored: '',
    set innerHTML(value) { this.stored = value; region.current = [{ scrollTop: 0 }]; },
    get innerHTML() { return this.stored; },
    querySelectorAll(selector) { return selector === '.thesen-scroll' ? region.current : []; },
  };
  const world = flowWorld();
  const host = flowHost({ '[data-campaigns-pane]': pane });
  hooks.__render.setState({ ctx: { ...hooks.__render.getState().ctx, host } });
  hooks.__render.renderCampaigns();
  assert.ok(pane.innerHTML.includes('Chemie Kampagne'), 'campaign pane did not render');
  assert.equal(region.current[0].scrollTop, 142, 'the re-render dropped the scroll position');
});

// --- Contact protection: Sellify remarks are a hard campaign boundary -------

test('" Kontaktsperre !!!" is normalized and blocks the person', () => {
  const decision = hooks.classifySellifyPerson({ note_text: ' Kontaktsperre !!!' });
  assert.equal(decision.status, 'blocked');
  assert.equal(decision.label, 'Kontaktsperre');
  assert.equal(decision.originalRemark, ' Kontaktsperre !!!');
});

test('"Im Ruhestand" blocks the person', () => {
  const decision = hooks.classifySellifyPerson({ title: 'Im Ruhestand' });
  assert.equal(decision.status, 'blocked');
  assert.equal(decision.label, 'Ruhestand');
  assert.equal(decision.originalRemark, 'Im Ruhestand');
});

test('the Babis successor remark blocks Britta Babis and puts Angela Grün into review', () => {
  const originalRemark = '06.10.2025: Anruf Frau Lindner, Frau Britta Babis ist verstorben, die neue Leiterin für den Einkauf ist Frau Angela Grün';
  const lead = {
    id: 'lead_successor',
    contacts: [
      { id: 'britta', name: 'Britta Babis' },
      { id: 'angela', name: 'Angela Grün' },
    ],
    selected_contact_ids: [],
  };
  const decisions = hooks.deriveLeadRecipientEligibility(lead, {
    people: [
      { id: 'sellify-person-1', person_id: 1, display_name: 'Britta Babis', note_text: originalRemark, title: '' },
      { id: 'sellify-person-2', person_id: 2, display_name: 'Angela Grün', note_text: '', title: '' },
    ],
    companies: [],
    contextAvailable: true,
  });
  assert.equal(decisions.get('britta').status, 'blocked');
  assert.equal(decisions.get('britta').label, 'verstorben');
  assert.equal(decisions.get('angela').status, 'review');
  assert.equal(decisions.get('angela').label, 'zu prüfen');
  assert.equal(decisions.get('angela').originalRemark, originalRemark);
});

test('a previously selected person that later becomes blocked is deselected and reported', async () => {
  const world = flowWorld();
  world.lead.selected_contact_ids = ['contact_a'];
  world.lead.contacts[0].note_text = ' Kontaktsperre !!!';
  hooks.__render.setState({
    recipientEligibilityReady: new Set([world.lead.id]),
    recipientEligibility: new Map([
      [`${world.lead.id}|contact_a`, hooks.classifySellifyPerson(world.lead.contacts[0])],
      [`${world.lead.id}|contact_b`, hooks.classifySellifyPerson(world.lead.contacts[1])],
    ]),
    recipientRemovalNotices: new Map(),
  });
  const repaired = await hooks.repairLeadRecipientSelections();
  assert.equal(repaired, 1);
  assert.deepEqual(world.lead.selected_contact_ids, []);
  const notice = hooks.__render.getState().recipientRemovalNotices.get(world.lead.id);
  assert.equal(notice.length, 1);
  assert.equal(notice[0].decision.label, 'Kontaktsperre');
  assert.equal(await hooks.repairLeadRecipientSelections(), 0);
  assert.equal(hooks.__render.getState().recipientRemovalNotices.get(world.lead.id).length, 1);
  const summary = hooks.renderCampaignRecipientExclusions(world.lead.campaign);
  assert.match(summary, /1 Kontakte ausgeschlossen · 0 zu prüfen/);
  assert.match(summary, /bereits ausgewählte Kontakte wurden abgewählt/);
  assert.match(summary, /Kontaktsperre !!!/);
});

test('the canonical Sellify and mail-merge recipient list rejects blocked and review contacts even after selection manipulation', async () => {
  const world = flowWorld();
  world.lead.contacts.push({
    id: 'contact_c', first_name: 'Clara', last_name: 'Castor', note_text: 'Nachfolger noch unklar',
  });
  world.lead.contacts[1].note_text = 'keine Werbung';
  // Simulates a manipulated/stale persisted selection containing all contacts.
  world.lead.selected_contact_ids = ['contact_a', 'contact_b', 'contact_c'];
  await hooks.sendLeadToSellify(world.lead.id, { includeCampaign: true });
  assert.deepEqual(world.lead.selected_contact_ids, ['contact_a']);
  const personWrites = world.dispatches.filter((command) => command.payload.operation_id === 'person_create');
  const campaignWrites = world.dispatches.filter((command) => command.payload.operation_id === 'campaign_create');
  assert.equal(personWrites.length, 1, 'a blocked or review contact reached the Sellify person list');
  assert.equal(campaignWrites.length, 1, 'a blocked or review contact reached the campaign/serial-letter list');
  assert.equal(personWrites[0].payload.first_name, 'Ada');
});

test('a person without a remark remains free', () => {
  assert.deepEqual(hooks.classifySellifyPerson({ note_text: '', title: 'Dr.' }), {
    status: 'free',
    label: 'frei',
    reason: '',
    originalRemark: '',
    sourceField: '',
    sourceRecordId: '',
  });
});

test('a company-level remark reviews only the named person instead of blocking the company', () => {
  const lead = {
    id: 'lead_company_note',
    contacts: [
      { id: 'named', name: 'Nora Beispiel' },
      { id: 'other', name: 'Otto Beispiel' },
    ],
    selected_contact_ids: [],
  };
  const decisions = hooks.deriveLeadRecipientEligibility(lead, {
    people: [],
    companies: [{ id: 'company-1', note_text: 'Nora Beispiel hat die Firma verlassen; Nachfolger noch offen.' }],
    contextAvailable: true,
  });
  assert.equal(decisions.get('named').status, 'review');
  assert.equal(decisions.get('other').status, 'free');
});

test('die Beobachtungskennung ignoriert reines Neuschreiben ohne Inhaltsaenderung', () => {
  // Auf der Kundeninstanz schreibt eine Schleife im nativen Peer dieselben sechs
  // Vorgangsdokumente unveraendert neu — am 11.08.2026 gemessen mit zeitweise
  // ueber 100 Revisionen je Minute, unabhaengig nachgewiesen bei
  // replicationUp=false, also ganz ohne Browser. Solange updated_at_ms im
  // Schluessel stand, galt jedes dieser Neuschreiben als neue Beobachtung und der
  // Browser schrieb den Lead erneut: die Anzeige haette die Schleife angeheizt.
  const vorgang = { command_id: 'cmd_research_1', terminal_status: 'completed', updated_at_ms: 1_000 };
  const dasselbeNurNeuGeschrieben = { ...vorgang, updated_at_ms: 9_999_999 };
  assert.equal(
    hooks.researchCommandObservationKey(dasselbeNurNeuGeschrieben),
    hooks.researchCommandObservationKey(vorgang),
  );

  // Der Statuswechsel muss weiterhin durchkommen — darauf beruht das Nachholen
  // verspaeteter Ergebnisse, das heute vier Firmen ihre Belege zurueckgab.
  const vorherGescheitert = { command_id: 'cmd_research_1', terminal_status: 'failed', updated_at_ms: 1_000 };
  assert.notEqual(
    hooks.researchCommandObservationKey(vorherGescheitert),
    hooks.researchCommandObservationKey(vorgang),
  );
});

test('die Recherche bekommt die vorhandenen CRM-Kontakte als Vorwissen mit', () => {
  // Bis zum 11.08.2026 fragte der Auftrag Sellify nur, OB die Firma existiert.
  // Die dort gefuehrten Ansprechpartner — in diesem Mandanten 60.639 Personen zu
  // 17.516 Firmen — blieben unbeachtet, und die Recherche suchte dieselben Namen
  // im offenen Netz neu zusammen.
  const text = hooks.sellifyVorwissenAlsText({
    contact_id: 4711,
    name: 'Destilla GmbH',
    anschrift: 'Industriestr. 1', plz: '88326', ort: 'Aulendorf',
    domain: 'destilla.de', telefon: '+4975251234',
    personen: [
      { vorname: 'Anna', nachname: 'Berg', funktion: 'Einkauf', email: 'a.berg@destilla.de', telefon: '' },
      { vorname: 'Jo', nachname: 'Klein', funktion: '', email: '', telefon: '+49752512399' },
    ],
  });
  assert.match(text, /Destilla GmbH/);
  assert.match(text, /contact_id 4711/);
  assert.match(text, /Anna Berg \| Einkauf \| a\.berg@destilla\.de/);
  assert.match(text, /Jo Klein/);
  assert.match(text, /NICHT erneut erraten/);

  // Ohne CRM-Treffer darf kein leerer Block in den Auftrag geraten.
  assert.equal(hooks.sellifyVorwissenAlsText(null), '');

  // Eine bekannte Firma ohne Kontakte ist trotzdem Vorwissen.
  const ohne = hooks.sellifyVorwissenAlsText({ contact_id: 1, name: 'Leer GmbH', personen: [] });
  assert.match(ohne, /noch keine Ansprechpartner gefuehrt/);
});

test('ein unvollstaendig geladener Lead meldet nicht "0 Quellen"', () => {
  // Am 11.08.2026 hing der Dienst 33 Stunden mit 4,2 GB fest, die Replikation
  // lieferte nichts mehr, und die Feldkarten meldeten fuer ANGUS Chemie
  // durchgaengig "0 Quellen" — waehrend serverseitig ZWOELF Belege lagen. Der
  // Nutzer haette die Firma auf dieser Anzeige als unbelegt verworfen. Eine
  // Aussage ueber die Welt darf nur fallen, wenn wir sie treffen koennen.
  const lead = {
    name: 'ANGUS Chemie GmbH',
    data: { firma_name: 'Angus Chemie GmbH', firma_plz: '49479', firma_ort: 'Ibbenbueren' },
    contacts: [],
    evidence: [{ field_key: 'firma_name', source_id: 'northdata.de', source_url: 'https://northdata.de/x' }],
  };
  // Belege da, aber die Auswertung findet keine Quelle -> unvollstaendig geladen.
  const kaputt = hooks.researchFieldReview({ ...lead, evidence: [{ nutzlos: true }] });
  assert.equal(kaputt.evidenceUnloaded, true);

  // Mit auswertbaren Belegen ist es eine echte Messung.
  const echt = hooks.researchFieldReview(lead);
  assert.equal(echt.evidenceUnloaded, false);

  // Ein Lead ganz ohne Belege ist ebenfalls eine echte Aussage: nichts gefunden.
  const leer = hooks.researchFieldReview({ ...lead, evidence: [] });
  assert.equal(leer.evidenceUnloaded, false);
});

test('gepflegte CRM-Kontaktdaten werden uebernommen, nicht neu erraten', async () => {
  // Am 11.08.2026 stand im gesamten Lead-Bestand KEINE einzige E-Mail-Adresse an
  // einem Ansprechpartner — die Serien-E-Mail war damit unmoeglich. Im CRM lagen
  // gleichzeitig 60.021 von 60.640 Personen MIT Adresse, fuer denselben Roger
  // Wintzen woertlich roger.wintzen@chemofast.com.
  const lead = {
    id: 'lead_test',
    contacts: [
      { id: 'c1', name: 'Roger Wintzen', email: '', phone: '', position: '' },
      { id: 'c2', name: 'Eigene Recherche', email: 'schon@da.example', phone: '' },
      { id: 'c3', name: 'Nicht im CRM', email: '', phone: '' },
    ],
  };
  const context = {
    contextAvailable: true,
    people: [
      { display_name: 'Roger Wintzen', email: 'roger.wintzen@chemofast.com', phone: '+4921548123', position: 'Geschäftsführer', person_id: 4711 },
      { display_name: 'Eigene Recherche', email: 'crm@example.com' },
    ],
  };
  const geaendert = await hooks.uebernehmeCrmKontaktdaten(lead, context);
  assert.equal(geaendert, true);
  assert.equal(lead.contacts[0].email, 'roger.wintzen@chemofast.com', 'fehlende Adresse wird uebernommen');
  assert.equal(lead.contacts[0].sellify_person_id, 4711);
  assert.equal(lead.contacts[1].email, 'schon@da.example', 'vorhandener Wert bleibt stehen — er kann aktueller sein');
  assert.equal(lead.contacts[2].email, '', 'wer nicht im CRM steht, bekommt nichts erfunden');

  // Ohne CRM-Kontext bleibt alles unveraendert.
  const unberuehrt = { id: 'x', contacts: [{ id: 'c', name: 'A', email: '' }] };
  assert.equal(await hooks.uebernehmeCrmKontaktdaten(unberuehrt, { contextAvailable: false, people: [] }), false);
  assert.equal(unberuehrt.contacts[0].email, '');
});

test('ein Registerzeichen darf den CRM-Abgleich nicht abschneiden', () => {
  // Der Lead hiess "CHEMOFAST Anchoring GmbH", die CRM-Organisation
  // "CHEMOFAST® Anchoring GmbH". Der exakte Vergleich fand das nicht — und damit
  // blieb am 11.08.2026 der GESAMTE CRM-Pfad wirkungslos: keine Dublette, kein
  // Vorwissen im Auftrag, keine uebernommenen Kontaktdaten, keine Serien-E-Mail.
  // Wegen eines Registerzeichens.
  const k = hooks.firmenSchluessel;
  assert.equal(k('CHEMOFAST® Anchoring GmbH'), k('CHEMOFAST Anchoring GmbH'));
  assert.equal(k('Destilla GmbH & Co. KG'), k('Destilla GmbH & Co KG'));
  assert.equal(k('  Mueller-Meier  GmbH '), k('Mueller Meier GmbH'));

  // Die Rechtsform bleibt unterscheidend: das sind verschiedene Firmen.
  assert.notEqual(k('Muster GmbH'), k('Muster AG'));
  assert.notEqual(k('Muster Chemie GmbH'), k('Muster Technik GmbH'));
  assert.equal(k(''), '');
});

test('Namensvarianten werden gezielt abgefragt, nicht die ganze Collection geladen', () => {
  // Der Lead heisst "CHEMOFAST Anchoring GmbH", die CRM-Organisation
  // "CHEMOFAST® Anchoring GmbH". Ein normalisierter Vergleich braeuchte alle
  // 17.520 Organisationen — am 12.08.2026 fror genau das die Seite ein
  // (CDP-Timeout 45 s, renderer unresponsive) und musste zurueckgenommen werden.
  // Wenige gezielte Punktabfragen kosten dagegen nichts, egal wie gross das CRM ist.
  const v = hooks.firmenNamensvarianten('CHEMOFAST Anchoring GmbH');
  assert.ok(v.includes('CHEMOFAST® Anchoring GmbH'), 'die CRM-Schreibweise muss dabei sein');
  assert.ok(!v.includes('CHEMOFAST Anchoring GmbH'), 'der schon gepruefte Originalname faellt raus');
  assert.ok(v.length <= 16, `Kandidatenliste bleibt klein, war ${v.length}`);

  // Kein Name, keine Abfragen.
  assert.deepEqual(hooks.firmenNamensvarianten(''), []);
  assert.deepEqual(hooks.firmenNamensvarianten(null), []);
});
