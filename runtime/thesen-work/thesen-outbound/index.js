import {
  extractCompanyRowsFromWorkbookFile,
  extractCompanyRowsFromText,
  normalizeCompanyRow,
  openUniversalImporter,
  parseDelimitedText,
} from '../../shared/universal-importer.js';
import { showBusinessAlert, showBusinessConfirm, showBusinessPrompt } from '../../shared/dialogs.js';
import { loadModuleMessages } from '../../shared/i18n.js';

const SOURCE_DEFS = Object.freeze([
  source('handelsregister.de', 'Handelsregister', 'https://www.handelsregister.de/', ['DE'], ['Firma', 'Sitz', 'Status', 'Geschäftsführung']),
  source('bundesanzeiger.de', 'Bundesanzeiger', 'https://www.bundesanzeiger.de/', ['DE'], ['Abschlüsse', 'Umsatz', 'Beschäftigte']),
  source('northdata.de', 'Northdata', 'https://www.northdata.de/', ['DE', 'AT', 'CH'], ['Firma', 'Sitz', 'Verflechtungen']),
  source('companyhouse.de', 'CompanyHouse', 'https://www.companyhouse.de/', ['DE'], ['Firma', 'Sitz', 'Geschäftsführung']),
  source('dnbhoovers.com', 'D&B Hoovers', 'https://app.dnbhoovers.com/', ['DE', 'AT', 'CH'], ['Branche', 'Umsatz', 'Beschäftigte'], 'DNB_HOOVERS_BROWSER_LOGIN'),
  source('leadfeeder.com', 'Leadfeeder', 'https://app.leadfeeder.com/f/200538/dashboard?organization_id=wmNE2MdyDk&full_view=false&tab=overview', ['DE', 'AT', 'CH'], ['Domain', 'Besuche', 'Interesse'], 'LEADFEEDER_BROWSER_LOGIN'),
  source('xing.com', 'XING', 'https://www.xing.com/', ['DE', 'AT', 'CH'], ['Person', 'Funktion', 'Position'], 'XING_BROWSER_LOGIN'),
  source('google.de', 'Google', 'https://www.google.de/', ['DE', 'AT', 'CH'], ['Website', 'Aktuelles', 'Kontakte']),
  source('maps.google.com', 'Google Maps', 'https://www.google.de/maps', ['DE', 'AT', 'CH'], ['Adresse', 'Telefon', 'Website']),
  source('rocketreach.com', 'RocketReach', 'https://rocketreach.co/', ['DE', 'AT', 'CH'], ['Person', 'E-Mail', 'Telefon']),
  source('firmenabc.at', 'FirmenABC', 'https://www.firmenabc.at/', ['AT'], ['Firma', 'Adresse', 'Telefon']),
  source('moneyhouse.ch', 'Moneyhouse', 'https://www.moneyhouse.ch/', ['CH'], ['Firma', 'Sitz', 'Management']),
  source('zefix.ch', 'Zefix', 'https://www.zefix.ch/', ['CH'], ['Register', 'Sitz', 'Status']),
  source('experte.de', 'E-Mail-Prüfung', 'https://www.experte.de/email-pruefen', ['DE', 'AT', 'CH'], ['E-Mail-Prüfung']),
]);

const RESEARCH_FIELDS = Object.freeze([
  'firma_name',
  'firma_anschrift',
  'firma_plz',
  'firma_ort',
  'firma_email',
  'firma_domain',
  'firma_telefon',
  'wz_code',
  'umsatz',
  'mitarbeiter',
  'person_geschlecht',
  'person_titel',
  'person_vorname',
  'person_nachname',
  'person_funktion',
  'person_position',
  'person_email',
  'person_email_validation',
  'person_telefon',
  'person_linkedin',
  'person_xing',
]);

const RESEARCH_FIELD_GROUPS = Object.freeze([
  {
    id: 'company',
    label: 'Unternehmen',
    fields: Object.freeze([
      ['firma_name', 'Firmenname'],
      ['firma_anschrift', 'Anschrift'],
      ['firma_plz', 'PLZ'],
      ['firma_ort', 'Ort'],
      ['firma_email', 'E-Mail'],
      ['firma_domain', 'Domain'],
      ['firma_telefon', 'Telefon'],
    ]),
  },
  {
    id: 'classification',
    label: 'Klassifikation',
    fields: Object.freeze([
      ['wz_code', 'WZ-Code'],
      ['umsatz', 'Umsatz'],
      ['mitarbeiter', 'Mitarbeiter'],
    ]),
  },
  {
    id: 'contact',
    label: 'Ansprechpartner',
    fields: Object.freeze([
      ['person_geschlecht', 'Geschlecht'],
      ['person_titel', 'Titel'],
      ['person_vorname', 'Vorname'],
      ['person_nachname', 'Nachname'],
      ['person_funktion', 'Funktion'],
      ['person_position', 'Position'],
      ['person_email', 'E-Mail'],
      ['person_email_validation', 'E-Mail-Prüfung'],
      ['person_telefon', 'Telefon'],
      ['person_linkedin', 'LinkedIn'],
      ['person_xing', 'XING'],
    ]),
  },
]);

const RESEARCH_FIELD_VALUE_KEYS = Object.freeze({
  firma_name: ['firma_name', 'company_name', 'firma'],
  firma_anschrift: ['firma_anschrift', 'address_line', 'address', 'street', 'strasse'],
  firma_plz: ['firma_plz', 'postal_code', 'postcode', 'plz'],
  firma_ort: ['firma_ort', 'city', 'ort'],
  firma_email: ['firma_email', 'email', 'company_email', 'e_mail'],
  firma_domain: ['firma_domain', 'domain', 'website', 'website_url', 'internet'],
  firma_telefon: ['firma_telefon', 'phone', 'company_phone', 'telefon'],
  wz_code: ['wz_code', 'wzcode'],
  umsatz: ['umsatz', 'revenue_mio', 'umsatz_mio'],
  mitarbeiter: ['mitarbeiter', 'employees'],
  person_geschlecht: ['person_geschlecht', 'geschlecht', 'gender'],
  person_titel: ['person_titel', 'titel'],
  person_vorname: ['person_vorname', 'vorname', 'first_name'],
  person_nachname: ['person_nachname', 'nachname', 'last_name'],
  person_funktion: ['person_funktion', 'funktion', 'role'],
  person_position: ['person_position', 'position'],
  person_email: ['person_email', 'email'],
  person_email_validation: ['person_email_validation', 'email_validation'],
  person_telefon: ['person_telefon', 'telefon', 'phone'],
  person_linkedin: ['person_linkedin', 'linkedin'],
  person_xing: ['person_xing', 'xing'],
});

// ACHTUNG: jede Aenderung an index.css braucht hier eine NEUE Nummer. Das
// Stylesheet wird unter genau dieser URL geholt, und Cloudflare speichert
// jede URL ein Jahr lang unveraenderlich. Blieb die Nummer stehen, kam keine
// einzige CSS-Aenderung beim Nutzer an — neun Tage lang unbemerkt.
const STYLE_BUILD = '20260806-thesen-outbound-contact-protection-2';
const COMMAND_REFRESH_MS = 5000;
const PERSON_BLOCK_PATTERNS = Object.freeze([
  ['Kontaktsperre', 'kontaktsperre'],
  ['keine Werbung', 'keine werbung'],
  ['nicht kontaktieren', 'nicht kontaktieren'],
  ['kein Kontakt', 'kein kontakt'],
  ['Werbesperre', 'werbesperre'],
  ['verstorben', 'verstorben'],
  ['ausgeschieden', 'ausgeschieden'],
  ['Ruhestand', 'im ruhestand'],
  ['Ruhestand', 'ruhestand'],
  ['Rente', 'in rente'],
  ['Rente', 'rente'],
  ['pensioniert', 'pensioniert'],
  ['Firma verlassen', 'firma verlassen'],
  ['nicht mehr im Haus', 'nicht mehr im haus'],
  ['nicht mehr tätig', 'nicht mehr tätig'],
]);
const PERSON_REVIEW_PATTERNS = Object.freeze([
  ['Nachfolgehinweis', 'nachfolger'],
  ['Nachfolgehinweis', 'nachfolgerin'],
  ['Nachfolgehinweis', 'übernimmt'],
  ['Wechselhinweis', 'wechsel'],
  ['Wechselhinweis', 'gewechselt'],
  ['Zuständigkeit unklar', 'nicht mehr zuständig'],
  ['Zuständigkeit unklar', 'unklar'],
]);
const SOURCE_REQUEST_TIMEOUT_MS = 15 * 60 * 1000;
const RESEARCH_POLICY_ID = 'thesen_research_policy_v1';
// A research run reports progress continuously. Silence past this window means
// the task is gone even though the durable status still reads `running`.
const RESEARCH_HEARTBEAT_STALE_MS = 10 * 60 * 1000;
// Obergrenze fuer einen Lead im Zustand "laeuft", wenn zu ihm ueberhaupt kein
// Vorgang mehr auffindbar ist. Ein einzelner Lauf dauert gemessen rund 90
// Sekunden; 30 Minuten lassen Warteschlange und Wiederholung reichlich Luft und
// beenden trotzdem den Zustand, in dem ANGUS Chemie ueber fuenf Stunden hing.
const RESEARCH_RUNNING_MAX_MS = 30 * 60 * 1000;
const LEGACY_RESEARCH_POLICY_IMPORT_ID = 'settings_research_policy';
const REPLICATED_COLLECTIONS = Object.freeze([
  'thesen_outbound_sources',
  'thesen_outbound_adapters',
  'thesen_outbound_imports',
  'thesen_outbound_research_policies',
  'thesen_outbound_leads',
  'business_commands',
]);
const DEFAULT_RESEARCH_POLICY = [
  '1. Identität und Registerdaten zuerst über Handelsregister, Northdata oder das zuständige Landesregister klären.',
  '2. Website, Anschrift und Kommunikation über Unternehmenswebsite, Google und Google Maps ergänzen.',
  '3. Kennzahlen über Bundesanzeiger, D&B Hoovers oder eine zweite belastbare Unternehmensquelle prüfen.',
  '4. Ansprechpartner über Unternehmenswebsite, XING und RocketReach recherchieren.',
  '5. Jedes übernommene Feld mit mindestens zwei unabhängigen Quellen belegen.',
  '6. Bei Zugriffshürden Web-Stack-Unlocking verwenden; bei Anmeldung den CTOX-Browser öffnen.',
  '7. Unklare oder widersprüchliche Daten als prüfbedürftig markieren und nicht automatisch an Sellify übergeben.',
].join('\n');
const CURRENT_USER_COPY = Object.freeze({
  de: {
    adapterPending: 'Noch nicht eingerichtet',
    adapterBuilding: 'Datenzugriff wird eingerichtet',
    buildAdapter: 'Datenzugriff einrichten',
    authRequested: 'Browser-Anmeldung angefordert',
    credentialReady: 'Zugang hinterlegt',
  },
  en: {
    adapterPending: 'Not configured',
    adapterBuilding: 'Data access is being configured',
    buildAdapter: 'Configure data access',
    authRequested: 'Browser sign-in requested',
    credentialReady: 'Credentials available',
  },
});

const state = {
  ctx: null,
  collections: {},
  sources: [],
  adapters: [],
  imports: [],
  leads: [],
  selectedLeadId: '',
  selectedLeadIds: new Set(),
  selectionAnchorId: '',
  selectedCampaign: '',
  campaignViewMode: 'table',
  leadEditorOpen: false,
  leadEditorId: '',
  leadDraft: null,
  sourcePanelOpen: false,
  sourcePanelView: 'sources',
  commandCollection: null,
  reconcilingCommands: false,
  reconcilingAdapterCommands: false,
  commandRefreshTimer: null,
  sellifyCompanies: null,
  sellifyPeople: null,
  recipientEligibility: new Map(),
  recipientEligibilityReady: new Set(),
  recipientEligibilitySignatures: new Map(),
  recipientRemovalNotices: new Map(),
  reconcilingRecipientEligibility: false,
  search: '',
  sourceSearch: '',
  researchPolicy: DEFAULT_RESEARCH_POLICY,
  researchPolicyDraft: DEFAULT_RESEARCH_POLICY,
  subscriptions: [],
  messages: {},
  campaignRuns: new Map(),
  syncPending: true,
  syncError: '',
};

export async function mount(ctx) {
  await ensureStyles();
  state.ctx = ctx;
  ctx.host.classList.add('thesen-outbound');
  const locale = ctx.locale === 'en' ? 'en' : 'de';
  const loadedMessages = await loadModuleMessages(import.meta.url, locale).catch(() => ({}));
  state.messages = { ...loadedMessages, ...CURRENT_USER_COPY[locale] };
  state.collections = {
    sources: ctx.db.collection('thesen_outbound_sources'),
    adapters: ctx.db.collection('thesen_outbound_adapters'),
    imports: ctx.db.collection('thesen_outbound_imports'),
    researchPolicies: ctx.db.collection('thesen_outbound_research_policies'),
    leads: ctx.db.collection('thesen_outbound_leads'),
  };
  state.commandCollection = ctx.db.collection('business_commands') || null;
  state.sellifyCompanies = ctx.db.collection('sellify_companies') || null;
  state.sellifyPeople = ctx.db.collection('sellify_people') || null;
  state.syncPending = true;
  state.syncError = '';
  bindCollections();
  bindUi();
  await reload();
  render();
  synchronizeInitialData().catch((error) => {
    state.syncPending = false;
    state.syncError = error?.message || String(error);
    render();
  });
  state.commandRefreshTimer = globalThis.setInterval(() => {
    Promise.all([
      reconcileResearchCommands({ authoritative: true }),
      reconcileAdapterCommands({ authoritative: true }),
    ]).then(async (changes) => {
      if (!changes.some(Boolean)) return;
      await reload();
      render();
    }).catch(() => {});
  }, COMMAND_REFRESH_MS);
  return () => {
    if (state.commandRefreshTimer) globalThis.clearInterval(state.commandRefreshTimer);
    state.commandRefreshTimer = null;
    state.subscriptions.forEach((subscription) => subscription?.unsubscribe?.());
    state.subscriptions = [];
    ctx.host.classList.remove('thesen-outbound');
    ctx.host.replaceChildren();
  };
}

async function synchronizeInitialData() {
  const replicationBridges = await Promise.all(REPLICATED_COLLECTIONS.map(async (collection) => {
    try {
      const bridge = await withTimeout(
        Promise.resolve(state.ctx.sync?.startCollection?.(collection)),
        `${collection} konnte nicht gestartet werden.`,
      );
      return [collection, bridge];
    } catch (error) {
      console.warn(`[thesen-outbound] ${collection} sync start delayed`, error);
      return [collection, null];
    }
  }));
  await Promise.allSettled(replicationBridges.map(([collection, bridge]) => (
    waitForReplicationBridge(bridge, collection)
  )));
  await seedSources();
  await reload();
  await repairUntrackedResearchStatuses();
  await reload();
  await reconcileResearchCommands({ authoritative: true });
  await reconcileAdapterCommands({ authoritative: true });
  await reload();
  await enforceRecipientEligibility();
  state.syncPending = false;
  state.syncError = '';
  render();
}

async function withTimeout(promise, message, timeoutMs = 20000) {
  let timer = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function waitForReplicationBridge(bridge, collection, timeoutMs = 20000) {
  const replicationState = bridge?.state;
  const wait = typeof replicationState?.awaitInSync === 'function'
    ? replicationState.awaitInSync.bind(replicationState)
    : typeof replicationState?.awaitInitialReplication === 'function'
      ? replicationState.awaitInitialReplication.bind(replicationState)
      : null;
  if (!wait) return;
  await Promise.race([
    wait(),
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`${collection} konnte nicht synchronisiert werden.`)), timeoutMs);
    }),
  ]);
}

async function ensureStyles() {
  const href = new URL(`./index.css?v=${STYLE_BUILD}`, import.meta.url).href;
  if (document.querySelector(`link[data-thesen-outbound-styles="${STYLE_BUILD}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  link.dataset.thesenOutboundStyles = STYLE_BUILD;
  document.head.append(link);
  await new Promise((resolve, reject) => {
    link.addEventListener('load', resolve, { once: true });
    link.addEventListener('error', () => reject(new Error('THESEN Outbound styles could not be loaded.')), { once: true });
  });
}

function source(id, label, url, countries, fieldKeys, credentialSecretName = '') {
  return {
    id, label, url, countries, fieldKeys,
    targetKey: id.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, ''),
    credentialSecretName,
  };
}

async function seedSources() {
  const now = Date.now();
  for (const definition of SOURCE_DEFS) {
    const existing = await state.collections.sources.findOne(definition.id).exec();
    if (existing) continue;
    await state.collections.sources.insert({
      id: definition.id,
      label: definition.label,
      url: definition.url,
      countries: definition.countries,
      field_keys: definition.fieldKeys,
      enabled: true,
      requires_credential: Boolean(definition.credentialSecretName),
      credential_secret_name: definition.credentialSecretName,
      target_key: definition.targetKey,
      adapter_status: 'draft',
      scrape_status: 'target_available',
      auth_status: definition.credentialSecretName ? 'required' : 'not_required',
      payload: { builtin: true, secret_value_in_payload: false },
      created_at_ms: now,
      updated_at_ms: now,
    });
  }
}

function bindCollections() {
  for (const collection of Object.values(state.collections)) {
    const subscription = collection.find().$.subscribe(() => {
      reload().then(() => enforceRecipientEligibility()).catch(() => render());
    });
    state.subscriptions.push(subscription);
  }
  if (state.commandCollection) {
    const subscription = state.commandCollection.find().$.subscribe(() => {
      Promise.all([reconcileResearchCommands(), reconcileAdapterCommands()]).then(async (changes) => {
        if (!changes.some(Boolean)) return;
        await reload();
        render();
      });
    });
    state.subscriptions.push(subscription);
  }
}

function bindUi() {
  const host = state.ctx.host;
  host.addEventListener('click', handleClick);
  host.addEventListener('keydown', handleKeydown);
  host.addEventListener('input', (event) => {
    if (event.target.matches('[data-lead-search]')) {
      state.search = event.target.value;
      renderCenter();
    }
    if (event.target.matches('[data-source-search]')) {
      state.sourceSearch = event.target.value;
      renderSourcePanel();
    }
    if (event.target.matches('[data-research-policy]')) {
      state.researchPolicyDraft = event.target.value;
    }
    if (event.target.matches('[data-lead-edit-field]') && state.leadDraft) {
      state.leadDraft[event.target.dataset.leadEditField] = event.target.value;
    }
  });
  host.addEventListener('pointerdown', startResize);
}

function researchPolicyInstructions(policy) {
  return String(policy?.instructions || DEFAULT_RESEARCH_POLICY).trim() || DEFAULT_RESEARCH_POLICY;
}

function researchPolicyRecord(existing, instructions, now = Date.now()) {
  const currentVersion = Number(existing?.version_number);
  const minIndependentSources = Number(existing?.min_independent_sources);
  const createdAt = Number(existing?.created_at_ms);
  return {
    id: RESEARCH_POLICY_ID,
    title: String(existing?.title ?? 'THESEN Quellenstandard'),
    version_number: (Number.isFinite(currentVersion) ? currentVersion : 0) + 1,
    status: String(existing?.status ?? 'active'),
    skill_name: String(existing?.skill_name ?? 'thesen-outbound-research'),
    skill_version: String(existing?.skill_version ?? '1.0.0'),
    min_independent_sources: Number.isFinite(minIndependentSources) ? minIndependentSources : 2,
    rules: Array.isArray(existing?.rules) ? existing.rules : [],
    instructions: String(instructions || '').trim(),
    created_at_ms: Number.isFinite(createdAt) ? createdAt : now,
    updated_at_ms: now,
  };
}

async function reload() {
  const [sources, adapters, imports, researchPolicies, leads] = await Promise.all(
    Object.values(state.collections).map(async (collection) => {
      const docs = await collection.find().exec();
      return docs.map((doc) => doc.toJSON());
    }),
  );
  state.sources = sources.sort((a, b) => a.label.localeCompare(b.label, 'de'));
  state.adapters = adapters;
  const policy = researchPolicies.find((item) => item.id === RESEARCH_POLICY_ID);
  state.researchPolicy = researchPolicyInstructions(policy);
  state.researchPolicyDraft = state.researchPolicy;
  state.imports = imports
    .filter((item) => item.id !== LEGACY_RESEARCH_POLICY_IMPORT_ID)
    .sort((a, b) => b.updated_at_ms - a.updated_at_ms);
  state.leads = leads
    .map(normalizeLeadRecipientShape)
    .sort((a, b) => b.updated_at_ms - a.updated_at_ms);
  invalidateChangedRecipientEligibility(state.leads);
  if (state.leads.length || state.imports.length) {
    state.syncPending = false;
    state.syncError = '';
  }
  const campaigns = campaignRows();
  if (!state.selectedCampaign || !campaigns.some((campaign) => campaign.name === state.selectedCampaign)) {
    state.selectedCampaign = campaigns[0]?.name || '';
  }
  const selectedCampaignLeads = campaignLeads(state.selectedCampaign);
  const campaignLeadIds = new Set(selectedCampaignLeads.map((lead) => lead.id));
  state.selectedLeadIds = new Set(
    [...state.selectedLeadIds].filter((id) => campaignLeadIds.has(id)),
  );
  if (!selectedCampaignLeads.some((lead) => lead.id === state.selectedLeadId)) {
    state.selectedLeadId = selectedCampaignLeads[0]?.id || '';
  }
}

function render() {
  const syncStatus = state.syncPending
    ? '<div class="ctox-syncing thesen-sync-status" role="status" aria-live="polite">Daten werden synchronisiert.</div>'
    : state.syncError
      ? '<div class="thesen-sync-status thesen-sync-delayed" role="status">Datenverbindung verzögert. Lokale Daten werden angezeigt.</div>'
      : '';
  state.ctx.host.innerHTML = `
    ${syncStatus}
    <div class="thesen-outbound-layout">
      <aside class="thesen-pane thesen-campaigns" data-campaigns-pane></aside>
      <div class="ctox-column-resizer thesen-resizer" data-resizer="left" role="separator" aria-label="Kampagnenbreite ändern"></div>
      <main class="thesen-pane thesen-leads" data-leads-pane></main>
      <div class="ctox-column-resizer thesen-resizer" data-resizer="right" role="separator" aria-label="Detailbreite ändern"></div>
      <aside class="thesen-pane thesen-detail" data-detail-pane></aside>
    </div>
    <div data-source-panel></div>
    <div data-app-dialog></div>`;
  renderCampaigns();
  renderCenter();
  renderDetail();
  renderSourcePanel();
  renderLeadEditor();
}

function campaignRows() {
  const counts = new Map();
  for (const lead of state.leads) {
    const name = String(lead.campaign || tr('annualResearch', 'Jahresrecherche')).trim();
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => a.name.localeCompare(b.name, 'de'));
}

// Every pane re-renders by rewriting innerHTML, which destroys the scroll
// containers and drops the reader back to the top on ANY click. Capture the
// offset of each `.thesen-scroll` region before the rewrite and restore it
// afterwards, matched by position so a list keeps its place while its rows
// update around it.
function setPaneHtml(pane, html) {
  if (!pane) return;
  const previous = Array.from(pane.querySelectorAll('.thesen-scroll')).map((node) => node.scrollTop);
  pane.innerHTML = html;
  if (!previous.length) return;
  pane.querySelectorAll('.thesen-scroll').forEach((node, index) => {
    const offset = previous[index];
    if (typeof offset === 'number' && offset > 0) node.scrollTop = offset;
  });
}

function renderCampaigns() {
  const pane = state.ctx.host.querySelector('[data-campaigns-pane]');
  if (!pane) return;
  const campaigns = campaignRows();
  setPaneHtml(pane, `
    <header class="ctox-pane-header ctox-pane-band thesen-header">
      <div class="thesen-pane-heading"><h2 class="ctox-pane-title">${tr('campaigns', 'Kampagnen')}</h2><span>${campaigns.length}</span></div>
      <div class="thesen-header-actions">
        <button class="ctox-pane-icon" data-action="new-campaign" title="${tr('newCampaign', 'Neue Kampagne')}" aria-label="${tr('newCampaign', 'Neue Kampagne')}">${icon('plus')}</button>
      </div>
    </header>
    <div class="thesen-scroll thesen-campaign-list">
      ${campaigns.map((campaign) => `
        <div class="thesen-campaign-row" data-selected="${campaign.name === state.selectedCampaign}">
          <button class="thesen-campaign-select" data-action="select-campaign" data-campaign="${escapeHtml(campaign.name)}" aria-pressed="${campaign.name === state.selectedCampaign}">
            <span>${escapeHtml(campaign.name)}</span><strong>${campaign.count}</strong>
          </button>
          <div class="thesen-campaign-actions">
            <button class="ctox-pane-icon" data-action="rename-campaign" data-campaign="${escapeHtml(campaign.name)}" title="Kampagne umbenennen" aria-label="${escapeHtml(campaign.name)} umbenennen">${icon('edit')}</button>
            <button class="ctox-pane-icon is-danger" data-action="delete-campaign" data-campaign="${escapeHtml(campaign.name)}" title="Kampagne löschen" aria-label="${escapeHtml(campaign.name)} löschen">${icon('trash')}</button>
          </div>
        </div>`).join('') || `<div class="thesen-empty">${tr('noCampaigns', 'Noch keine Kampagne.')}</div>`}
    </div>`);
}

function renderSourcePanel() {
  const mount = state.ctx.host.querySelector('[data-source-panel]');
  if (!mount) return;
  if (!state.sourcePanelOpen) {
    mount.replaceChildren();
    return;
  }
  const needle = state.sourceSearch.trim().toLowerCase();
  const sources = state.sources.filter((item) => !needle || `${item.label} ${item.url}`.toLowerCase().includes(needle));
  const showingPolicy = state.sourcePanelView === 'policy';
  const panelCount = showingPolicy ? 7 : sources.length;
  const panelTitle = showingPolicy
    ? tr('researchPolicy', 'Rechercheablauf')
    : tr('sourcesAndAccounts', 'Quellen & Zugänge');
  setPaneHtml(mount, `
    <div class="thesen-source-backdrop" data-action="close-sources">
      <section class="thesen-source-panel" role="dialog" aria-modal="true" aria-label="${panelTitle}" data-source-dialog>
        <header class="ctox-pane-header ctox-pane-band thesen-header">
          <div><span class="ctox-pane-kicker">${panelCount}</span><h2 class="ctox-pane-title">${panelTitle}</h2></div>
          <div class="thesen-header-actions">
            ${!showingPolicy ? `<button class="ctox-pane-icon" data-action="add-source" title="${tr('addSource', 'Quelle hinzufügen')}" aria-label="${tr('addSource', 'Quelle hinzufügen')}">${icon('plus')}</button>` : ''}
            <button class="ctox-pane-icon" data-action="close-sources" title="${tr('close', 'Schließen')}" aria-label="${tr('close', 'Schließen')}">${icon('close')}</button>
          </div>
        </header>
        <div class="thesen-source-tabs" role="tablist" aria-label="${tr('sourceSettings', 'Recherche-Einstellungen')}">
          <button role="tab" aria-selected="${!showingPolicy}" data-action="source-view" data-view="sources">${tr('sourcesAndAccounts', 'Quellen & Zugänge')}</button>
          <button role="tab" aria-selected="${showingPolicy}" data-action="source-view" data-view="policy">${tr('researchPolicy', 'Rechercheablauf')}</button>
        </div>
        ${showingPolicy ? '' : `<div class="thesen-toolbar"><input class="ctox-pane-search" data-source-search value="${escapeHtml(state.sourceSearch)}" placeholder="${tr('searchSource', 'Quelle suchen')}" /></div>`}
        ${showingPolicy
          ? renderResearchPolicy()
          : `<div class="thesen-scroll thesen-source-list">${sources.map(renderSourceRow).join('')}</div>`}
      </section>
    </div>`);
}

function renderResearchPolicy() {
  return `<section class="thesen-policy-editor">
    <textarea data-research-policy aria-label="${tr('researchPolicy', 'Rechercheablauf')}">${escapeHtml(state.researchPolicyDraft)}</textarea>
    <footer>
      <button class="ctox-button" data-action="reset-policy">${tr('reset', 'Zurücksetzen')}</button>
      <button class="ctox-button is-primary" data-action="save-policy">${icon('check')}<span>${tr('save', 'Speichern')}</span></button>
    </footer>
  </section>`;
}

function sourceStatus(item, adapter, now = Date.now()) {
  const status = String(adapter?.status ?? item?.adapter_status ?? '').toLowerCase();
  const scrapeStatus = String(adapter?.scrape_status ?? item?.scrape_status ?? '').toLowerCase();
  const authStatus = String(adapter?.auth_status ?? item?.auth_status ?? '').toLowerCase();
  const lastError = String(adapter?.last_error ?? '').trim();
  const updatedAt = Number(adapter?.updated_at_ms ?? item?.updated_at_ms) || 0;
  const requiresCredential = Boolean(item?.requires_credential);
  const checkedAt = updatedAt ? new Date(updatedAt).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric' }) : '';

  // Der Zugangs-Chip speist sich aus denselben persistierten Feldern wie die Statuszeile.
  // credential_present und auth_verified_at_ms existieren serverseitig noch nicht (Spec §1.2)
  // und werden daher nie als bewiesen dargestellt.
  const credentialStored = ['credential_available', 'authenticated'].includes(authStatus);
  let chip = null;
  if (requiresCredential) {
    chip = authStatus === 'invalid_credentials'
      ? { code: 'invalid', label: tr('accountInvalid', 'Zugang abgelehnt') }
      : credentialStored
        ? { code: 'available', label: tr('accountCredentialsAvailable', 'Zugang hinterlegt') }
        : { code: 'missing', label: tr('accountMissingShort', 'Zugang fehlt') };
  }

  // §1.4: *_requested gilt nur mit verfolgtem Kommando und nur 15 Minuten lang.
  const requestPending = Boolean(adapter?.last_command_id) && updatedAt > 0 && now - updatedAt <= SOURCE_REQUEST_TIMEOUT_MS;
  const requestExpired = 'Die letzte Anfrage ist ohne Ergebnis abgelaufen. Status unbekannt.';

  // §1.3: strikte Priorität — Deaktiviert → Prüfung läuft → Anmeldung läuft →
  // Zugang abgelehnt → Zugang fehlt → Blockiert → Fehlgeschlagen → Bereit →
  // Registriert-ungeprüft → Zugang-hinterlegt-ungeprüft → Unbekannt.
  if (item?.enabled === false) {
    return { code: 'disabled', label: tr('sourceDisabled', 'Deaktiviert — wird bei der Recherche übersprungen.'), chip };
  }
  if (status === 'generation_queued' || status.endsWith('_requested') || scrapeStatus.endsWith('_requested')) {
    if (requestPending) {
      return {
        code: 'check_running',
        label: status === 'generation_queued'
          ? tr('adapterBuilding', 'Datenzugriff wird eingerichtet')
          : tr('checkRunning', 'Prüfung läuft …'),
        chip,
      };
    }
    return { code: 'request_expired', label: tr('requestExpired', requestExpired), chip };
  }
  if (['auth_requested', 'browser_session_requested'].includes(authStatus)) {
    return requestPending
      ? { code: 'auth_running', label: tr('authRequested', 'Browser-Anmeldung angefordert') + ' — bitte im Browser-Fenster abschließen.', chip }
      : { code: 'request_expired', label: tr('requestExpired', requestExpired), chip };
  }
  if (authStatus === 'invalid_credentials') {
    return {
      code: 'credential_rejected',
      label: checkedAt
        ? `Zugang abgelehnt: Die Quelle hat die Zugangsdaten zuletzt am ${checkedAt} zurückgewiesen.`
        : tr('credentialRejected', 'Zugang abgelehnt: Die Quelle hat die Zugangsdaten zurückgewiesen.'),
      chip,
    };
  }
  if (requiresCredential && !credentialStored) {
    return { code: 'credential_missing', label: tr('credentialMissing', 'Zugang erforderlich — keine Zugangsdaten hinterlegt.'), chip };
  }
  if (status.includes('blocked') || scrapeStatus === 'blocked') {
    // blocked_reason/failure_mode fehlen serverseitig (Spec §1.2) — Grund bleibt ehrlich unbekannt.
    return { code: 'blocked', label: tr('sourceBlocked', 'Zugriff blockiert — der Grund ist unbekannt. Automatischer Zugriff ist derzeit nicht möglich.'), chip };
  }
  if (status.includes('failed') || scrapeStatus.includes('failed') || lastError) {
    const detail = lastError ? `: ${lastError}` : '';
    return {
      code: 'failed',
      label: checkedAt
        ? `Letzte Prüfung fehlgeschlagen (${checkedAt})${detail}`
        : `Letzte Prüfung fehlgeschlagen${detail}`,
      chip,
    };
  }
  if (adapterReady({ status, scrape_status: scrapeStatus })) {
    return {
      code: 'ready',
      label: checkedAt ? `Bereit · zuletzt geprüft am ${checkedAt}` : tr('adapterReady', 'Datenzugriff bereit'),
      chip,
    };
  }
  if (status.includes('zero_records') || scrapeStatus.includes('zero_records')) {
    return { code: 'empty_result', label: tr('checkEmpty', 'Letzte Prüfung ohne Treffer — die Quelle lieferte keine Einträge.'), chip };
  }
  if (scrapeStatus === 'registered') {
    // adapter_revision fehlt serverseitig (Spec §1.2) — keine Revisionsnummer behaupten.
    return { code: 'registered', label: tr('adapterRegistered', 'Adapter registriert · noch nicht geprüft'), chip };
  }
  if (requiresCredential && credentialStored) {
    return { code: 'credential_unverified', label: tr('credentialStored', 'Zugang hinterlegt · Anmeldung noch nicht geprüft'), chip };
  }
  return { code: 'unknown', label: tr('sourceUnknown', 'Status unbekannt — diese Quelle wurde noch nie geprüft.'), chip };
}

function renderSourceRow(item) {
  const adapter = state.adapters.find((entry) => entry.source_id === item.id);
  const status = sourceStatus(item, adapter);
  const ready = status.code === 'ready';
  const needsBrowserAuthorization = sourceNeedsBrowserAuthorization(item, adapter);
  return `<div class="thesen-source-row" data-source-id="${escapeHtml(item.id)}" data-context-record-id="${escapeHtml(item.id)}" data-context-record-type="research-source" data-context-label="${escapeHtml(item.label)}">
    <button class="thesen-source-toggle" data-action="toggle-source" aria-pressed="${item.enabled}" title="Quelle ${item.enabled ? 'deaktivieren' : 'aktivieren'}"><span class="thesen-toggle-dot"></span></button>
    <div class="thesen-source-copy">
      <strong>${escapeHtml(item.label)}</strong>
      <span>${escapeHtml(item.countries.join('/'))} · ${escapeHtml(item.field_keys.join(', '))}</span>
      <span class="thesen-adapter-state ${ready ? 'is-ready' : ''}">${ready ? icon('check') : '<i></i>'}${escapeHtml(status.label)}</span>
      ${status.chip ? `<span class="thesen-source-credential is-${escapeHtml(status.chip.code)}">${escapeHtml(status.chip.label)}</span>` : ''}
    </div>
    <div class="thesen-source-actions">
      <button class="ctox-pane-icon" data-action="build-adapter" title="${tr('buildAdapter', 'Datenzugriff einrichten')}" aria-label="${tr('buildAdapter', 'Datenzugriff einrichten')}">${icon('wrench')}</button>
      <button class="ctox-pane-icon" data-action="test-adapter" title="${tr('testAdapter', 'Datenzugriff prüfen')}" aria-label="${tr('testAdapter', 'Datenzugriff prüfen')}">${icon('test')}</button>
      ${needsBrowserAuthorization ? `<button class="ctox-pane-icon" data-action="auth-source" title="${tr('browserSignIn', 'Im CTOX-Browser anmelden')}" aria-label="${tr('signIn', 'Anmelden')}">${icon('login')}</button>` : ''}
    </div>
  </div>`;
}

function renderCenter() {
  const pane = state.ctx.host.querySelector('[data-leads-pane]');
  if (!pane) return;
  const needle = state.search.trim().toLowerCase();
  const leads = state.leads.filter((lead) => {
    const campaign = String(lead.campaign || tr('annualResearch', 'Jahresrecherche')).trim();
    return (!state.selectedCampaign || campaign === state.selectedCampaign)
      && (!needle || `${lead.name} ${lead.domain} ${lead.city}`.toLowerCase().includes(needle));
  });
  const campaignProgress = campaignResearchProgress(
    state.selectedCampaign,
    state.leads,
    state.campaignRuns.get(state.selectedCampaign),
  );
  const campaignAction = campaignProgress.trackingTaskId ? 'track-task' : 'research-campaign';
  const campaignActionLabel = campaignProgress.trackingTaskId
    ? tr('trackResearch', 'Recherche in CTOX öffnen')
    : tr('researchCampaign', 'Kampagne recherchieren');
  const selectedVisibleCount = selectedVisibleLeadCount(leads);
  const selectedCount = state.selectedLeadIds.size;
  const selectedActionable = campaignLeads(state.selectedCampaign)
    .filter((lead) => state.selectedLeadIds.has(lead.id) && !['queued', 'running'].includes(lead.research_status))
    .length;
  const allVisibleSelected = leads.length > 0 && selectedVisibleCount === leads.length;
  const someVisibleSelected = selectedVisibleCount > 0 && !allVisibleSelected;
  setPaneHtml(pane, `
    <header class="ctox-pane-header ctox-pane-band thesen-header">
      <div class="thesen-pane-heading"><h2 class="ctox-pane-title">${escapeHtml(state.selectedCampaign || tr('newResearch', 'Neu- und Nachrecherche'))}</h2><span>${leads.length} Leads</span></div>
      <div class="thesen-header-actions">
        <button class="ctox-button thesen-campaign-research" data-action="${campaignAction}" data-campaign="${escapeHtml(state.selectedCampaign)}"
          data-task-id="${escapeHtml(campaignProgress.trackingTaskId)}" data-command-id="${escapeHtml(campaignProgress.trackingCommandId)}"
          title="${escapeHtml(campaignActionLabel)}" aria-label="${escapeHtml(campaignActionLabel)}"
          ${!state.selectedCampaign || (!campaignProgress.trackingTaskId && (campaignProgress.active || campaignProgress.actionable === 0)) ? 'disabled' : ''}>
          ${icon(campaignProgress.trackingTaskId ? 'external' : 'search')}<span>${escapeHtml(campaignProgress.trackingTaskId ? 'Task öffnen' : `Alle recherchieren (${campaignProgress.actionable})`)}</span>
        </button>
        <button class="ctox-pane-icon" data-action="open-policy" title="${tr('researchPolicy', 'Rechercheablauf')}" aria-label="${tr('researchPolicy', 'Rechercheablauf')}">${icon('book')}</button>
        <button class="ctox-pane-icon" data-action="open-sources" title="${tr('sourceSettings', 'Recherche-Einstellungen')}" aria-label="${tr('sourceSettings', 'Recherche-Einstellungen')}">${icon('settings')}</button>
        <div class="ctox-view-toggle" role="group" aria-label="${tr('view', 'Darstellung')}">
          <button class="ctox-pane-icon" data-action="view-mode" data-view="shards" aria-pressed="${state.campaignViewMode === 'shards'}" title="${tr('shardView', 'Shard-Ansicht')}" aria-label="${tr('shardView', 'Shard-Ansicht')}">${icon('shards')}</button>
          <button class="ctox-pane-icon" data-action="view-mode" data-view="table" aria-pressed="${state.campaignViewMode === 'table'}" title="${tr('tableView', 'Tabellenansicht')}" aria-label="${tr('tableView', 'Tabellenansicht')}">${icon('table')}</button>
        </div>
        <button class="ctox-pane-icon is-primary" data-action="import-leads" title="${tr('import', 'Importieren')}" aria-label="${tr('import', 'Importieren')}">${icon('import')}</button>
      </div>
    </header>
    ${selectedCount ? `
      <div class="thesen-toolbar thesen-selection-bar" role="status" aria-live="polite">
        <strong>${selectedCount} ausgewählt</strong>
        ${selectedVisibleCount !== selectedCount ? `<span>${selectedVisibleCount} sichtbar</span>` : ''}
        <button class="ctox-button is-primary" data-action="research-selection" ${selectedActionable ? '' : 'disabled'}>${icon('search')}<span>Auswahl recherchieren (${selectedActionable})</span></button>
        <button class="ctox-button" data-action="clear-selection">Auswahl aufheben</button>
      </div>` : `
      <div class="thesen-toolbar">
        <input class="ctox-pane-search" data-lead-search value="${escapeHtml(state.search)}" placeholder="${tr('searchLead', 'Firma, Domain oder Ort')}" />
        <span>${state.imports[0] ? `${tr('lastImport', 'Letzter Import')}: ${escapeHtml(state.imports[0].title)}` : tr('noImport', 'Noch kein Import')}</span>
      </div>`}
    ${renderCampaignProgress(campaignProgress)}
    ${renderCampaignRecipientExclusions(state.selectedCampaign)}
    <div class="thesen-scroll">
      ${state.campaignViewMode === 'shards'
        ? `<div class="thesen-lead-shards">${leads.map(renderLeadShard).join('') || `<div class="thesen-empty">${tr('noLeads', 'Noch keine Leads importiert.')}</div>`}</div>`
        : `<table class="thesen-table"><thead><tr><th class="thesen-select-column"><input type="checkbox" data-action="toggle-visible-leads" aria-label="Alle sichtbaren Leads auswählen" ${allVisibleSelected ? 'checked' : ''} ${someVisibleSelected ? 'data-indeterminate="true"' : ''}></th><th>${tr('organization', 'Organisation')}</th><th>${tr('location', 'Ort')}</th><th>${tr('research', 'Recherche')}</th><th>Sellify</th></tr></thead>
          <tbody>${leads.map(renderLeadRow).join('') || `<tr><td colspan="5" class="thesen-empty">${tr('noLeads', 'Noch keine Leads importiert.')}</td></tr>`}</tbody></table>`}
    </div>`);
  const headerCheckbox = pane.querySelector('[data-action="toggle-visible-leads"]');
  if (headerCheckbox) headerCheckbox.indeterminate = someVisibleSelected;
}

function campaignRecipientExclusions(campaign) {
  const rows = [];
  const removedKeys = new Set();
  for (const [leadId, notices] of state.recipientRemovalNotices) {
    for (const notice of notices || []) removedKeys.add(recipientEligibilityKey(leadId, notice.contact?.id));
  }
  for (const lead of campaignLeads(campaign)) {
    for (const contact of lead.contacts || []) {
      const decision = currentContactEligibility(lead, contact);
      if (decision.status === 'free') continue;
      rows.push({
        lead,
        contact,
        decision,
        removed: removedKeys.has(recipientEligibilityKey(lead.id, contact.id)),
      });
    }
  }
  return rows;
}

function renderCampaignRecipientExclusions(campaign) {
  const rows = campaignRecipientExclusions(campaign);
  if (!rows.length) return '';
  const reviewCount = rows.filter((row) => row.decision.status === 'review').length;
  const removedCount = rows.filter((row) => row.removed).length;
  return `<details class="thesen-recipient-exclusions" ${removedCount ? 'open' : ''}>
    <summary>${rows.length} Kontakte ausgeschlossen · ${reviewCount} zu prüfen</summary>
    ${removedCount ? `<p role="status">${removedCount} bereits ausgewählte Kontakte wurden abgewählt.</p>` : ''}
    <ul>${rows.map(({ lead, contact, decision, removed }) => `<li>
      <strong>${escapeHtml(personDisplayName(contact) || tr('contact', 'Kontakt'))}</strong>
      <span>${escapeHtml(lead.name)} · ${escapeHtml(decision.label)}${removed ? ' · abgewählt' : ''}</span>
      ${decision.originalRemark ? `<q>${escapeHtml(decision.originalRemark)}</q>` : `<span>${escapeHtml(decision.reason)}</span>`}
    </li>`).join('')}</ul>
  </details>`;
}

function renderCampaignProgress(progress) {
  const current = progress.currentLeadName
    ? `${tr('currentLead', 'Aktuell')}: ${progress.currentLeadName}`
    : campaignResearchStatusLabel(progress.status);
  return `
    <div class="thesen-campaign-progress" role="status" aria-live="polite" data-campaign-status="${escapeHtml(progress.status)}">
      <div class="thesen-progress-copy">
        <strong>${escapeHtml(current)}</strong>
        <span>${progress.processed} / ${progress.total}</span>
      </div>
      <div class="thesen-progress-track" aria-hidden="true"><i style="width:${progress.percent}%"></i></div>
      <div class="thesen-progress-states">
        ${campaignProgressState('new', tr('statusNew', 'Neu'), progress.counts.new)}
        ${campaignProgressState('queued', tr('statusQueued', 'Wartet'), progress.counts.queued)}
        ${campaignProgressState('running', tr('statusRunning', 'Laufend'), progress.counts.running)}
        ${campaignProgressState('completed', tr('statusCompleted', 'Abgeschlossen'), progress.counts.completed)}
        ${campaignProgressState('failed', tr('statusFailed', 'Unvollständig'), progress.counts.failed)}
        ${campaignProgressState('validated', tr('statusValidated', 'Validiert'), progress.counts.validated)}
      </div>
    </div>`;
}

function campaignProgressState(status, label, count) {
  return `<span class="thesen-progress-state is-${status}"><i></i>${escapeHtml(label)} <strong>${count}</strong></span>`;
}

function renderLeadRow(lead) {
  return `<tr data-action="select-lead" data-id="${escapeHtml(lead.id)}" tabindex="0" data-context-record-id="${escapeHtml(lead.id)}" data-context-record-type="lead" data-context-label="${escapeHtml(lead.name)}" aria-selected="${lead.id === state.selectedLeadId}" data-checked="${state.selectedLeadIds.has(lead.id)}">
    <td class="thesen-select-column"><input type="checkbox" data-action="toggle-lead" data-id="${escapeHtml(lead.id)}" aria-label="${escapeHtml(lead.name)} auswählen" ${state.selectedLeadIds.has(lead.id) ? 'checked' : ''}></td>
    <td><strong>${escapeHtml(lead.name)}</strong>${lead.domain || lead.website ? `<span>${escapeHtml(lead.domain || lead.website)}</span>` : ''}</td>
    <td>${escapeHtml([lead.city, lead.country].filter(Boolean).join(', ') || '—')}</td>
    <td><span class="ctox-badge ${lead.validation_status === 'validated' ? 'is-success' : ''}">${escapeHtml(researchLabel(lead))}</span></td>
    <td>${escapeHtml(sellifyLabel(lead))}</td>
  </tr>`;
}

function renderLeadShard(lead) {
  return `<article class="thesen-lead-shard" data-checked="${state.selectedLeadIds.has(lead.id)}"
    data-context-record-id="${escapeHtml(lead.id)}" data-context-record-type="lead" data-context-label="${escapeHtml(lead.name)}">
    <input type="checkbox" data-action="toggle-lead" data-id="${escapeHtml(lead.id)}" aria-label="${escapeHtml(lead.name)} auswählen" ${state.selectedLeadIds.has(lead.id) ? 'checked' : ''}>
    <button class="thesen-shard-main" data-action="select-lead" data-id="${escapeHtml(lead.id)}" aria-pressed="${lead.id === state.selectedLeadId}">
      <strong>${escapeHtml(lead.name)}</strong>
      <span>${escapeHtml([lead.city, lead.country].filter(Boolean).join(', ') || lead.domain || '—')}</span>
      <footer><span class="ctox-badge ${lead.validation_status === 'validated' ? 'is-success' : ''}">${escapeHtml(researchLabel(lead))}</span><span>${escapeHtml(lead.sellify_status === 'completed' ? 'Sellify' : '')}</span></footer>
    </button>
    <button class="ctox-pane-icon thesen-shard-research" data-action="research-lead" data-id="${escapeHtml(lead.id)}" title="Lead nachrecherchieren" aria-label="${escapeHtml(lead.name)} nachrecherchieren">${icon('search')}</button>
  </article>`;
}

// Jede Collection-Änderung ruft render() und damit renderDetail(). Ein
// innerHTML-Neuaufbau reisst dabei den Fokus aus einem Eingabefeld und setzt
// die Scrollposition zurueck — Tippen war praktisch unmoeglich. Solange der
// Nutzer in diesem Bereich schreibt, wird der Neuaufbau aufgeschoben und nach
// dem Verlassen des Feldes einmal nachgeholt.
function detailPaneHasActiveInput(pane) {
  const active = state.ctx.host.ownerDocument?.activeElement || globalThis.document?.activeElement;
  if (!active || !pane.contains(active)) return false;
  const tag = String(active.tagName || '').toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || active.isContentEditable;
}

function renderDetail() {
  const pane = state.ctx.host.querySelector('[data-detail-pane]');
  if (!pane) return;
  if (detailPaneHasActiveInput(pane)) {
    if (!state.detailRenderDeferred) {
      state.detailRenderDeferred = true;
      const active = state.ctx.host.ownerDocument?.activeElement || globalThis.document?.activeElement;
      active.addEventListener('blur', () => {
        state.detailRenderDeferred = false;
        renderDetail();
      }, { once: true });
    }
    return;
  }
  state.detailRenderDeferred = false;
  const lead = selectedLead();
  if (!lead) {
    pane.innerHTML = `<div class="thesen-empty">${tr('selectLead', 'Lead auswählen.')}</div>`;
    return;
  }
  const scroller = pane.querySelector('.thesen-detail-body');
  const scrollTop = scroller ? scroller.scrollTop : 0;
  const review = researchFieldReview(lead);
  const readyForValidation = leadReadyForValidation(lead);
  const blockers = validationBlockers(lead);
  const validateLabel = readyForValidation
    ? tr('validate', 'Validieren')
    : `${tr('notReadyForValidation', 'Noch nicht freigebbar')}: ${blockers.join(' · ') || tr('researchIncomplete', 'Recherche noch nicht abgeschlossen.')}`;
  const handoffPrecondition = sellifyHandoffPrecondition(lead);
  const handoffRunning = lead.sellify_status === 'queued';
  const handoffDisabled = Boolean(handoffPrecondition) || handoffRunning;
  const handoffTitle = handoffRunning ? tr('handoffRunning', 'Übergabe läuft') : handoffPrecondition;
  pane.innerHTML = `
    <header class="ctox-pane-header ctox-pane-band thesen-header">
      <div><span class="ctox-pane-kicker">Lead</span><h2 class="ctox-pane-title">${escapeHtml(lead.name)}</h2></div>
      <div class="thesen-header-actions">
        <button class="ctox-pane-icon" data-action="edit-lead" data-id="${escapeHtml(lead.id)}" title="Lead bearbeiten" aria-label="${escapeHtml(lead.name)} bearbeiten">${icon('edit')}</button>
      </div>
    </header>
    <div class="thesen-scroll thesen-detail-body">
      ${renderResearchReviewSummary(review, lead)}
      <div class="thesen-detail-actions">
        <button class="ctox-button is-primary" data-action="research-lead" data-id="${escapeHtml(lead.id)}">${icon('search')}<span>${tr('researchLead', 'Nachrecherche')}</span></button>
        <button class="ctox-button" data-action="validate-lead" data-id="${escapeHtml(lead.id)}" ${readyForValidation ? '' : 'disabled'} title="${escapeHtml(validateLabel)}">${icon('check')}<span>${escapeHtml(validateLabel)}</span></button>
        <button class="ctox-button" data-action="sellify-update-only" data-id="${escapeHtml(lead.id)}" ${handoffDisabled ? 'disabled' : ''} title="${escapeHtml(handoffTitle || tr('sellifyUpdateOnlyTitle', 'Organisation und ausgewählte Personen in Sellify aktualisieren.'))}">${icon('send')}<span>${tr('sellifyUpdateOnly', 'Zu Sellify (nur aktualisieren)')}</span></button>
        <button class="ctox-button" data-action="sellify-update-campaign" data-id="${escapeHtml(lead.id)}" ${handoffDisabled ? 'disabled' : ''} title="${escapeHtml(handoffTitle || tr('sellifyUpdateCampaignTitle', 'Organisation und ausgewählte Personen aktualisieren und der Kampagne hinzufügen.'))}">${icon('send')}<span>${tr('sellifyUpdateCampaign', 'Zu Sellify (aktualisieren & Kampagne)')}</span></button>
        <button class="ctox-button" data-action="mail-series-email" data-id="${escapeHtml(lead.id)}" ${handoffDisabled ? 'disabled' : ''} title="${escapeHtml(handoffTitle || tr('mailSeriesEmailTitle', 'Ausgewählte, freigegebene Personen in eine Serien-E-Mail übernehmen.'))}">${icon('send')}<span>${tr('mailSeriesEmail', 'Als Serien-E-Mail')}</span></button>
      </div>
      ${renderResearchReviewGroups(review, lead)}
    </div>`;
  const nextScroller = pane.querySelector('.thesen-detail-body');
  if (nextScroller) nextScroller.scrollTop = scrollTop;
}

function renderLeadEditor() {
  const mount = state.ctx.host.querySelector('[data-app-dialog]');
  if (!mount) return;
  if (!state.leadEditorOpen || !state.leadDraft) {
    mount.replaceChildren();
    return;
  }
  const draft = state.leadDraft;
  const fieldInput = (key, label, value, options = {}) => `
    <label class="thesen-form-field ${options.wide ? 'is-wide' : ''}">
      <span>${escapeHtml(label)}</span>
      <input data-lead-edit-field="${escapeHtml(key)}" value="${escapeHtml(value || '')}" ${options.required ? 'required' : ''}>
    </label>`;
  mount.innerHTML = `
    <div class="thesen-source-backdrop" data-action="close-lead-editor">
      <section class="thesen-lead-editor" role="dialog" aria-modal="true" aria-label="Lead bearbeiten" data-lead-editor>
        <header class="ctox-pane-header ctox-pane-band thesen-header">
          <div><span class="ctox-pane-kicker">Lead</span><h2 class="ctox-pane-title">Bearbeiten</h2></div>
          <button class="ctox-pane-icon" data-action="close-lead-editor" title="Schließen" aria-label="Schließen">${icon('close')}</button>
        </header>
        <div class="thesen-lead-form">
          ${fieldInput('name', 'Organisation', draft.name, { required: true, wide: true })}
          ${fieldInput('website', 'Website', draft.website, { wide: true })}
          ${fieldInput('address_line', 'Straße', draft.address_line, { wide: true })}
          ${fieldInput('postal_code', 'PLZ', draft.postal_code)}
          ${fieldInput('city', 'Ort', draft.city)}
          ${fieldInput('country', 'Land', draft.country)}
          ${fieldInput('email', 'E-Mail', draft.email)}
          ${fieldInput('phone', 'Telefon', draft.phone)}
          ${fieldInput('campaign', 'Kampagne', draft.campaign, { required: true, wide: true })}
        </div>
        <footer class="thesen-dialog-actions">
          <button class="ctox-button" data-action="close-lead-editor">Abbrechen</button>
          <button class="ctox-button is-primary" data-action="save-lead-editor">Speichern</button>
        </footer>
      </section>
    </div>`;
}

async function handleClick(event) {
  const trigger = event.target.closest('[data-action]');
  if (!trigger) return;
  const action = trigger.dataset.action;
  const id = trigger.dataset.id || trigger.closest('[data-source-id]')?.dataset.sourceId || '';
  if (action === 'select-lead') {
    if (event.metaKey || event.ctrlKey) {
      toggleLeadSelection(id, !state.selectedLeadIds.has(id), event.shiftKey);
      renderCenter();
    } else {
      state.selectedLeadId = id;
      renderCenter();
      renderDetail();
    }
  }
  if (action === 'select-campaign') {
    state.selectedCampaign = trigger.dataset.campaign || '';
    state.selectedLeadIds.clear();
    state.selectionAnchorId = '';
    const first = state.leads.find((lead) => String(lead.campaign || tr('annualResearch', 'Jahresrecherche')).trim() === state.selectedCampaign);
    state.selectedLeadId = first?.id || '';
    renderCampaigns(); renderCenter(); renderDetail();
  }
  if (action === 'open-sources') {
    state.sourcePanelView = 'sources';
    state.sourcePanelOpen = true;
    renderSourcePanel();
  }
  if (action === 'open-policy') {
    state.sourcePanelView = 'policy';
    state.sourcePanelOpen = true;
    state.researchPolicyDraft = state.researchPolicy;
    renderSourcePanel();
  }
  if (action === 'source-view') {
    state.sourcePanelView = trigger.dataset.view === 'policy' ? 'policy' : 'sources';
    state.sourceSearch = '';
    state.researchPolicyDraft = state.researchPolicy;
    renderSourcePanel();
  }
  if (action === 'close-sources' && (event.target === trigger || trigger.matches('button[data-action="close-sources"]'))) {
    state.sourcePanelOpen = false; renderSourcePanel();
  }
  // Der Import-Knopf im mittleren Fenster fuegt Unternehmen zur AUSGEWAEHLTEN
  // Kampagne hinzu. Vorher rief er den Importer ohne Kampagnenname auf; der
  // Importer setzte seinen Standardtitel, und weil die Kampagne aus dem Titel
  // abgeleitet wird, entstand jedes Mal eine neue. Neue Kampagnen legt man
  // ausdruecklich ueber "Neue Kampagne" an.
  if (action === 'import-leads') await openImporter(state.selectedCampaign || '');
  if (action === 'new-campaign') await createCampaign();
  if (action === 'rename-campaign') await renameCampaign(trigger.dataset.campaign || state.selectedCampaign);
  if (action === 'delete-campaign') await deleteCampaign(trigger.dataset.campaign || state.selectedCampaign);
  if (action === 'toggle-lead') {
    toggleLeadSelection(id, Boolean(trigger.checked), event.shiftKey);
    renderCenter();
  }
  if (action === 'toggle-visible-leads') {
    setVisibleLeadSelection(Boolean(trigger.checked));
    renderCenter();
  }
  if (action === 'clear-selection') {
    state.selectedLeadIds.clear();
    state.selectionAnchorId = '';
    renderCenter();
  }
  if (action === 'research-selection') await startSelectionResearch();
  if (action === 'edit-lead') openLeadEditor(id);
  if (action === 'close-lead-editor' && (event.target === trigger || trigger.matches('button[data-action="close-lead-editor"]'))) {
    closeLeadEditor();
  }
  if (action === 'save-lead-editor') await saveLeadEditor();
  if (action === 'view-mode') {
    state.campaignViewMode = trigger.dataset.view === 'shards' ? 'shards' : 'table';
    renderCenter();
  }
  if (action === 'save-policy') await saveResearchPolicy();
  if (action === 'reset-policy') {
    state.researchPolicyDraft = DEFAULT_RESEARCH_POLICY;
    renderSourcePanel();
  }
  if (action === 'toggle-source') await toggleSource(id);
  if (action === 'build-adapter') await runAdapterCommand(id, 'outbound.research_source.generate_adapter');
  if (action === 'test-adapter') await runAdapterCommand(id, 'outbound.research_source.test');
  if (action === 'auth-source') await runAdapterCommand(id, 'outbound.research_source.auth_assist');
  if (action === 'research-campaign') await startCampaignResearch(trigger.dataset.campaign || state.selectedCampaign);
  if (action === 'track-task') await openCtoxTask(trigger.dataset.taskId || '', trigger.dataset.commandId || '');
  if (action === 'research-lead') await researchLead(id);
  if (action === 'validate-lead') await validateLead(id);
  if (action === 'approve-field') await approveResearchField(trigger.dataset.field || '');
  if (action === 'edit-lead' && !id && selectedLead()) {
    state.leadEditorFocusField = trigger.dataset.field || '';
    openLeadEditor(selectedLead().id);
    return;
  }
  // Der Zustand wird aus der gespeicherten Auswahl abgeleitet, nicht aus
  // `trigger.checked`: bei einem Klick auf das umgebende Label meldet die
  // Checkbox je nach Ereignisreihenfolge noch den alten Wert, und die Auswahl
  // fiel danach still auf 0 zurueck — der Grund, warum die Sellify-Uebergabe
  // nie ansprang.
  // Ein Klick auf die Empfaengerzeile erzeugt ZWEI Klickereignisse: eines vom
  // umgebenden Label, eines von der Checkbox, die der Browser daraufhin selbst
  // ausloest. Gemessen. Beide erreichten den Handler, der die Auswahl damit
  // ab- und sofort wieder anschaltete — die Auswahl bewegte sich nie. Das
  // zweite Ereignis desselben Kontakts wird deshalb verworfen; der Zielzustand
  // kommt aus der gespeicherten Auswahl, nicht aus `trigger.checked` (das bei
  // Label-Klicks je nach Reihenfolge noch den alten Wert meldet).
  if (action === 'toggle-contact-recipient') {
    const contactId = trigger.dataset.contactId || '';
    const stempel = `${id}|${contactId}`;
    const jetzt = Date.now();
    if (state.lastRecipientToggle?.key === stempel && jetzt - state.lastRecipientToggle.at < 400) return;
    state.lastRecipientToggle = { key: stempel, at: jetzt };
    const current = new Set((state.leads.find((entry) => entry.id === id)?.selected_contact_ids) || []);
    await setContactRecipientSelection(id, contactId, !current.has(contactId));
  }
  if (action === 'sellify-update-only') await sendLeadToSellify(id, { includeCampaign: false });
  if (action === 'sellify-update-campaign') await sendLeadToSellify(id, { includeCampaign: true });
  if (action === 'mail-series-email') await openSeriesEmailFromLead(id);
  if (action === 'add-source') await addSource();
}

function handleKeydown(event) {
  if (event.key === 'Escape') {
    if (state.leadEditorOpen) closeLeadEditor();
    else if (state.sourcePanelOpen) {
      state.sourcePanelOpen = false;
      renderSourcePanel();
    }
    return;
  }
  const row = event.target.closest('[data-action="select-lead"]');
  if (!row || !['Enter', ' '].includes(event.key)) return;
  event.preventDefault();
  const id = row.dataset.id || '';
  if (event.key === ' ' || event.metaKey || event.ctrlKey) {
    toggleLeadSelection(id, !state.selectedLeadIds.has(id), event.shiftKey);
    renderCenter();
    return;
  }
  state.selectedLeadId = id;
  renderCenter();
  renderDetail();
}

function visibleCampaignLeads() {
  const needle = state.search.trim().toLowerCase();
  return campaignLeads(state.selectedCampaign).filter((lead) => (
    !needle || `${lead.name} ${lead.domain} ${lead.website} ${lead.city}`.toLowerCase().includes(needle)
  ));
}

function selectedVisibleLeadCount(leads = visibleCampaignLeads()) {
  return leads.filter((lead) => state.selectedLeadIds.has(lead.id)).length;
}

function toggleLeadSelection(id, selected, range = false) {
  if (!id) return;
  const visible = visibleCampaignLeads();
  if (range && state.selectionAnchorId) {
    const start = visible.findIndex((lead) => lead.id === state.selectionAnchorId);
    const end = visible.findIndex((lead) => lead.id === id);
    if (start >= 0 && end >= 0) {
      for (const lead of visible.slice(Math.min(start, end), Math.max(start, end) + 1)) {
        if (selected) state.selectedLeadIds.add(lead.id);
        else state.selectedLeadIds.delete(lead.id);
      }
      return;
    }
  }
  if (selected) state.selectedLeadIds.add(id);
  else state.selectedLeadIds.delete(id);
  state.selectionAnchorId = id;
}

function setVisibleLeadSelection(selected) {
  for (const lead of visibleCampaignLeads()) {
    toggleLeadSelection(lead.id, selected);
  }
  state.selectionAnchorId = '';
}

async function startSelectionResearch() {
  const leads = campaignLeads(state.selectedCampaign)
    .filter((lead) => state.selectedLeadIds.has(lead.id));
  if (!leads.length) {
    await showBusinessAlert('Bitte mindestens einen Lead auswählen.');
    return null;
  }
  return startScopedResearch(state.selectedCampaign, leads, {
    scope: 'selection',
    title: `Nachrecherche: ${leads.length} ausgewählte Leads`,
  });
}

async function startCampaignResearch(campaignName) {
  const campaign = String(campaignName || '').trim();
  const leads = campaignLeads(campaign);
  return startScopedResearch(campaign, leads, {
    scope: 'campaign',
    title: `Kampagnenrecherche: ${campaign}`,
  });
}

async function startScopedResearch(campaign, leads, options = {}) {
  const eligibleIds = options.scope === 'selection'
    ? new Set((leads || [])
      .filter((lead) => !['queued', 'running'].includes(lead.research_status))
      .map((lead) => lead.id))
    : new Set(campaignResearchQueue(leads));
  const eligibleLeads = (leads || []).filter((lead) => eligibleIds.has(lead.id));
  const validation = validateCampaignResearchRequest({ campaign, leads: eligibleLeads });
  if (!validation.valid) {
    showBusinessAlert(validation.error);
    return null;
  }
  if (options.scope === 'campaign' && state.campaignRuns.get(campaign)?.status === 'running') {
    showBusinessAlert(tr('campaignAlreadyRunning', 'Für diese Kampagne läuft bereits eine Recherche.'));
    return null;
  }

  const runId = `thesen_${options.scope || 'campaign'}_research_${crypto.randomUUID()}`;
  const runKey = options.scope === 'campaign' ? campaign : runId;
  state.campaignRuns.set(runKey, {
    id: runId,
    taskId: '',
    commandId: '',
    status: 'running',
    currentLeadId: '',
    currentLeadName: '',
    startedAtMs: Date.now(),
    finishedAtMs: 0,
    error: '',
  });
  renderCenter();

  const submissions = [];
  for (const lead of eligibleLeads) {
    const result = await researchLead(lead.id, {
      campaignRunId: runId,
      threadKey: `business-os/thesen-outbound/campaign/${runId}/lead/${lead.id}`,
      suppressAlerts: true,
    });
    submissions.push({ lead, result });
  }
  const accepted = submissions.filter(({ result }) => result?.status !== 'failed');
  const failed = submissions.filter(({ result }) => result?.status === 'failed');
  const run = state.campaignRuns.get(runKey);
  if (run) Object.assign(run, {
    commandId: accepted[0]?.result?.commandId || '',
    status: accepted.length ? 'running' : 'failed',
    finishedAtMs: accepted.length ? 0 : Date.now(),
    error: failed.map(({ lead, result }) => `${lead.name}: ${result.error}`).join('\n'),
  });
  await reload();
  if (failed.length) {
    showBusinessAlert(
      accepted.length
        ? `${accepted.length} Recherchen wurden gestartet. ${failed.length} konnten nicht gestartet werden.`
        : failed[0].result.error,
    );
  }
  if (options.scope === 'selection') {
    state.selectedLeadIds.clear();
    state.selectionAnchorId = '';
    renderCenter();
  }
  return runId;
}

function campaignLeads(campaign) {
  return state.leads.filter((lead) => String(lead.campaign || tr('annualResearch', 'Jahresrecherche')).trim() === campaign);
}

function campaignResearchQueue(leads) {
  return (leads || [])
    .filter((lead) => lead.validation_status !== 'validated' && lead.research_status !== 'completed')
    .map((lead) => lead.id);
}

function validateCampaignResearchRequest({ campaign, leads }) {
  if (!String(campaign || '').trim()) {
    return { valid: false, error: tr('selectCampaignFirst', 'Bitte zuerst eine Kampagne auswählen.') };
  }
  if (!Array.isArray(leads) || leads.length === 0) {
    return { valid: false, error: tr('campaignHasNoLeads', 'Diese Kampagne enthält keine Leads.') };
  }
  if (leads.some((lead) => !String(lead?.id || '').trim() || !String(lead?.name || '').trim())) {
    return { valid: false, error: tr('campaignHasInvalidLeads', 'Die Kampagne enthält unvollständige Lead-Datensätze.') };
  }
  return { valid: true, error: '' };
}

function campaignResearchPrompt(campaign, leads, runId, options = {}) {
  const leadList = leads.map((lead, index) => `${index + 1}. ${lead.name} [${lead.id}]`).join('\n');
  return [
    'Nutze den CTOX Skill thesen-outbound-research und den CTOX Web Stack.',
    '',
    `Kampagne: ${campaign}`,
    `Umfang: ${options.scope === 'selection' ? `Auswahl mit ${leads.length} Leads` : `gesamte Kampagne mit ${leads.length} Leads`}`,
    `Workflow-ID: ${runId}`,
    `Leads: ${leads.length}`,
    '',
    'Aufgabe:',
    'Du bist der alleinige dauerhafte Orchestrator dieser Kampagnenrecherche. Die Ausführung darf nicht von einem geöffneten Browser-Tab abhängen.',
    'Verbindlicher Rechercheablauf:',
    state.researchPolicy,
    '',
    'Arbeite die Leads seriell ab. Lies die Datensätze über business_os.query_records aus thesen_outbound_leads. Starte für jeden Lead exakt einen business_os.dispatch_command mit module_id thesen-outbound, command_type web_stack.person_research und der jeweiligen Lead-ID als record_id.',
    `Der Payload ist strikt: {"company":"<Lead-Name>","country":"DE|AT|CH","mode":"new_record|update_firm","fields":${JSON.stringify(RESEARCH_FIELDS)},"include_private":[],"auto_browser_capture":true}. Verwende company, niemals company_name. Sende keine Felder research_scope, min_independent_sources oder campaign im Recherche-Payload.`,
    'Warte nach jedem Dispatch über business_os.get_command_status bis completed, failed, blocked oder cancelled. Ein anfänglicher running-Status ist ausdrücklich kein Erfolg. Schreibe erst nach dem terminalen Readback den belegten Zustand über business_os.upsert_record zurück.',
    '',
    'Validierungsregeln:',
    '- Jedes übernommene Feld benötigt mindestens zwei unabhängige Quellen.',
    '- Nutze konfigurierte Playwright-Adapter; bei Zugriffshürden den Web-Stack-Unlocking- und Browser-Anmeldeprozess.',
    '- Terminal, Shell, curl, direkte HTTP-Aufrufe oder eigene Browserautomation sind kein Ersatz für business_os.dispatch_command.',
    '- Keine stillen Fehler: Fehler je Lead protokollieren und mit dem nächsten Lead fortfahren.',
    '- Setze je Lead queued -> running -> completed, needs_review oder failed und erhalte campaign_run_id sowie alle bestehenden Felder.',
    '- Keine direkte Übergabe an Sellify und keine SQL-Schreiboperation in diesem Lauf.',
    '- Ergebnisse ausschließlich in den zugehörigen thesen_outbound_leads zurückschreiben.',
    '',
    'Lead-Liste:',
    leadList,
  ].join('\n');
}

async function openCampaignResearchChat({
  campaign,
  leads,
  runId,
  prompt,
  title: requestedTitle = '',
  scope = 'campaign',
  submitTask = state.ctx?.businessChat?.submitTask,
}) {
  const validation = validateCampaignResearchRequest({ campaign, leads });
  if (!validation.valid) throw new Error(validation.error);
  if (!String(prompt || '').trim() || !String(runId || '').trim()) {
    throw new Error(tr('campaignTaskInvalid', 'Der Recherche-Task ist unvollständig und wurde nicht gestartet.'));
  }
  if (typeof submitTask !== 'function') {
    throw new Error(tr('chatUnavailable', 'CTOX Chat ist nicht verfügbar. Die Recherche wurde nicht gestartet.'));
  }
  const title = String(requestedTitle || `${tr('campaignResearchTitle', 'Kampagnenrecherche')}: ${campaign}`).trim();
  const detail = {
    text: prompt,
    module: 'thesen-outbound',
    source_module: 'thesen-outbound',
    source_title: tr('title', 'THESEN Outbound'),
    action: 'context-chat',
    reuseActive: false,
    open: true,
    command_id: runId,
    command_type: 'business_os.chat.task',
    record_id: `campaign:${campaign}`,
    title,
    command_title: title,
    instruction: prompt,
    mode: 'data',
    target: 'data',
    required_skills: ['thesen-outbound-research', 'universal-scraping', 'web-unlock'],
    writeback_contract: {
      collection: 'thesen_outbound_leads',
      allowed_collections: ['thesen_outbound_leads'],
      command_type: 'web_stack.person_research',
      record_ids: leads.map((lead) => lead.id),
      min_independent_sources: 2,
    },
    payload: {
      campaign,
      scope,
      lead_ids: leads.map((lead) => lead.id),
      lead_count: leads.length,
      prompt,
      response_channel: 'business_os_chat',
      thread_key: `business-os/thesen-outbound/campaign/${runId}`,
    },
    client_context: {
      action: 'context-chat',
      source: 'thesen-outbound-campaign-research',
      module: 'thesen-outbound',
      campaign,
      scope,
      workflow_id: runId,
      response_channel: 'business_os_chat',
    },
  };
  const submission = requireTrackedSubmission(await submitTask(detail));
  return { ...submission, detail };
}

function requireTrackedSubmission(submission, { allowTerminalCommand = false, allowControlCommand = false } = {}) {
  const taskId = String(submission?.task_id || submission?.taskId || '').trim();
  const commandId = String(submission?.command_id || submission?.commandId || '').trim();
  const status = String(submission?.terminal_status || submission?.status || '').trim().toLowerCase();
  const terminalCommand = allowTerminalCommand
    && ['completed', 'failed', 'blocked', 'cancelled', 'canceled'].includes(status);
  const trackedControlCommand = allowControlCommand && Boolean(commandId) && Boolean(status);
  if (!commandId || (!taskId && !terminalCommand && !trackedControlCommand)) {
    throw new Error(tr('taskNotConfirmed', 'CTOX hat keinen verfolgbaren Task bestätigt. Die Automatisierung wurde nicht gestartet.'));
  }
  return { ...submission, task_id: taskId, command_id: commandId };
}

function normalizeProtectionText(value) {
  return String(value || '')
    .normalize('NFKC')
    .toLocaleLowerCase('de-DE')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

function containsProtectionPhrase(text, phrase) {
  const normalizedText = normalizeProtectionText(text);
  const normalizedPhrase = normalizeProtectionText(phrase);
  if (` ${normalizedText} `.includes(` ${normalizedPhrase} `)) return true;
  const punctuationRemoved = String(text || '')
    .normalize('NFKC')
    .toLocaleLowerCase('de-DE')
    .replace(/[^\p{L}\p{N}\s]+/gu, '')
    .trim()
    .replace(/\s+/g, ' ');
  return ` ${punctuationRemoved} `.includes(` ${normalizedPhrase} `);
}

function personDisplayName(person) {
  return String(person?.display_name || person?.name || [
    person?.first_name || person?.person_vorname,
    person?.last_name || person?.person_nachname,
  ].filter(Boolean).join(' ')).trim();
}

function personProtectionFields(person) {
  return [
    ['note_text', String(person?.note_text || '')],
    ['title', String(person?.title || '')],
  ].filter(([, value]) => value.trim());
}

function findPersonBlockMatch(value) {
  for (const [label, phrase] of PERSON_BLOCK_PATTERNS) {
    if (containsProtectionPhrase(value, phrase)) return { label, phrase };
  }
  return null;
}

function findPersonReviewMatch(value) {
  const normalized = normalizeProtectionText(value);
  if (/\b(?:die|der) neue\b.{0,120}\bist\b/u.test(normalized)
    || normalized.includes('neue leitung')
    || normalized.includes('neuer ansprechpartner')
    || normalized.includes('neue ansprechpartnerin')) {
    return { label: 'Nachfolgehinweis', phrase: 'neue Person' };
  }
  for (const [label, phrase] of PERSON_REVIEW_PATTERNS) {
    if (containsProtectionPhrase(normalized, phrase)) return { label, phrase };
  }
  return null;
}

function freePersonEligibility() {
  return {
    status: 'free',
    label: 'frei',
    reason: '',
    originalRemark: '',
    sourceField: '',
    sourceRecordId: '',
  };
}

function personEligibilityDecision(status, match, originalRemark, sourceField, sourceRecordId = '') {
  return {
    status,
    label: status === 'blocked' ? match.label : 'zu prüfen',
    reason: match.label,
    originalRemark: String(originalRemark || ''),
    sourceField,
    sourceRecordId: String(sourceRecordId || ''),
  };
}

function classifySellifyPerson(person) {
  for (const [sourceField, originalRemark] of personProtectionFields(person)) {
    const blocked = findPersonBlockMatch(originalRemark);
    if (blocked) {
      return personEligibilityDecision('blocked', blocked, originalRemark, sourceField, person?.id || person?.person_id);
    }
  }
  for (const [sourceField, originalRemark] of personProtectionFields(person)) {
    const review = findPersonReviewMatch(originalRemark);
    if (review) {
      return personEligibilityDecision('review', review, originalRemark, sourceField, person?.id || person?.person_id);
    }
  }
  return freePersonEligibility();
}

function strongerPersonEligibility(left, right) {
  const rank = { free: 0, review: 1, blocked: 2 };
  if (!left) return right || freePersonEligibility();
  if (!right) return left;
  return rank[right.status] > rank[left.status] ? right : left;
}

function normalizedPersonId(value) {
  const raw = String(value || '').trim();
  const match = raw.match(/(?:sellify-person-)?(\d+)$/);
  return match ? Number(match[1]) : 0;
}

function personMatchesContact(person, contact) {
  const explicitIds = [
    contact?.sellify_person_id,
    contact?.person_id,
    contact?.id,
  ].map(normalizedPersonId).filter(Boolean);
  if (explicitIds.length && explicitIds.includes(Number(person?.person_id))) return true;
  const contactEmail = String(contact?.email || contact?.person_email || '').trim().toLowerCase();
  const personEmail = String(person?.email || '').trim().toLowerCase();
  if (contactEmail && personEmail && contactEmail === personEmail) return true;
  const contactName = normalizeProtectionText(personDisplayName(contact));
  const sellifyName = normalizeProtectionText(personDisplayName(person));
  return contactName.split(' ').length >= 2 && contactName === sellifyName;
}

function successorReferenceMentions(originalRemark, personName) {
  const text = normalizeProtectionText(originalRemark);
  const name = normalizeProtectionText(personName);
  if (!text || !name || name.split(' ').length < 2) return false;
  const nameIndex = text.indexOf(name);
  if (nameIndex < 0) return false;
  for (const marker of ['die neue', 'der neue', 'nachfolger', 'nachfolgerin']) {
    const markerIndex = text.indexOf(marker);
    if (markerIndex >= 0 && nameIndex > markerIndex && nameIndex - markerIndex < 180) return true;
  }
  const takeoverIndex = text.indexOf('übernimmt');
  return takeoverIndex >= 0 && Math.abs(nameIndex - takeoverIndex) < 120;
}

function companyRemarkMentionsPerson(company, personName) {
  const name = normalizeProtectionText(personName);
  if (!name || name.split(' ').length < 2) return null;
  for (const [sourceField, originalRemark] of personProtectionFields(company)) {
    if (!normalizeProtectionText(originalRemark).includes(name)) continue;
    const signal = findPersonBlockMatch(originalRemark) || findPersonReviewMatch(originalRemark);
    if (signal) return personEligibilityDecision('review', { label: 'Firmenvermerk prüfen' }, originalRemark, sourceField, company?.id || company?.contact_id);
  }
  return null;
}

function deriveLeadRecipientEligibility(lead, { people = [], companies = [], contextAvailable = true } = {}) {
  const normalizedLead = normalizeLeadRecipientShape(lead || {});
  const decisions = new Map();
  for (const contact of normalizedLead.contacts) {
    let decision = classifySellifyPerson(contact);
    const matchedPeople = people.filter((person) => personMatchesContact(person, contact));
    for (const person of matchedPeople) {
      decision = strongerPersonEligibility(decision, classifySellifyPerson(person));
    }
    for (const person of people) {
      if (matchedPeople.includes(person)) continue;
      for (const [, originalRemark] of personProtectionFields(person)) {
        if (!successorReferenceMentions(originalRemark, personDisplayName(contact))) continue;
        decision = strongerPersonEligibility(decision, personEligibilityDecision(
          'review',
          { label: 'Nachfolgehinweis' },
          originalRemark,
          'note_text',
          person?.id || person?.person_id,
        ));
      }
    }
    for (const company of companies) {
      decision = strongerPersonEligibility(decision, companyRemarkMentionsPerson(company, personDisplayName(contact)));
    }
    if (!contextAvailable && decision.status === 'free') {
      decision = {
        ...personEligibilityDecision('review', { label: 'Sellify-Sperrvermerk nicht prüfbar' }, '', '', ''),
        pending: true,
      };
    }
    decisions.set(contact.id, decision);
  }
  return decisions;
}

function recipientEligibilityKey(leadId, contactId) {
  return `${String(leadId || '')}|${String(contactId || '')}`;
}

function currentContactEligibility(lead, contact) {
  const local = classifySellifyPerson(contact);
  if (local.status !== 'free') return local;
  const cached = state.recipientEligibility.get(recipientEligibilityKey(lead?.id, contact?.id));
  if (cached) return cached;
  return {
    ...personEligibilityDecision('review', { label: 'Sellify-Sperrvermerk wird geprüft' }, '', '', ''),
    pending: true,
  };
}

function buildCampaignRecipientList(lead, decisions = null) {
  const normalized = normalizeLeadRecipientShape(lead || {});
  const selectedIds = new Set(normalized.selected_contact_ids);
  const recipients = [];
  const excluded = [];
  for (const contact of normalized.contacts) {
    if (!selectedIds.has(contact.id)) continue;
    const decision = decisions?.get?.(contact.id)
      || decisions?.[contact.id]
      || currentContactEligibility(normalized, contact);
    if (decision?.status === 'free') recipients.push(contact);
    else excluded.push({ contact, decision: decision || personEligibilityDecision('review', { label: 'Sperrstatus unbekannt' }, '', '', '') });
  }
  return { recipients, excluded, selectedCount: selectedIds.size };
}

function recipientEligibilitySignature(lead) {
  return JSON.stringify({
    name: lead?.name || '',
    sellify_contact_id: lead?.payload?.sellify_contact_id || '',
    contacts: (lead?.contacts || []).map((contact) => ({
      id: contact?.id || '',
      person_id: contact?.person_id || contact?.sellify_person_id || '',
      name: personDisplayName(contact),
      email: contact?.email || contact?.person_email || '',
      note_text: contact?.note_text || '',
      title: contact?.title || '',
    })),
  });
}

function invalidateChangedRecipientEligibility(leads) {
  const liveLeadIds = new Set((leads || []).map((lead) => lead.id));
  for (const lead of leads || []) {
    const signature = recipientEligibilitySignature(lead);
    if (state.recipientEligibilitySignatures.get(lead.id) === signature) continue;
    state.recipientEligibilitySignatures.set(lead.id, signature);
    state.recipientEligibilityReady.delete(lead.id);
    for (const key of state.recipientEligibility.keys()) {
      if (key.startsWith(`${lead.id}|`)) state.recipientEligibility.delete(key);
    }
  }
  for (const leadId of [...state.recipientEligibilitySignatures.keys()]) {
    if (liveLeadIds.has(leadId)) continue;
    state.recipientEligibilitySignatures.delete(leadId);
    state.recipientEligibilityReady.delete(leadId);
    state.recipientRemovalNotices.delete(leadId);
  }
}

function docJson(doc) {
  return doc?.toJSON?.() || doc;
}

function uniqueSellifyRecords(records) {
  const unique = new Map();
  for (const record of records || []) {
    const value = docJson(record);
    const key = String(value?.id || value?.person_id || value?.contact_id || JSON.stringify(value));
    if (!unique.has(key)) unique.set(key, value);
  }
  return [...unique.values()];
}

async function findSellifyRecords(collection, selector) {
  if (!collection?.find) return [];
  const docs = await collection.find({ selector }).exec();
  return (docs || []).map(docJson).filter((record) => !record?.is_deleted);
}

async function loadSellifyRecipientContext(lead) {
  // Die Sperrvermerkspruefung darf nicht daran scheitern, dass zufaellig noch
  // niemand sellifyReadCollection() gerufen hat. state.sellifyPeople ist ein
  // Zwischenspeicher, der erst beim ersten Zugriff gefuellt wird — beim Rendern
  // der Empfaengerliste passierte das nie. Ergebnis am 11.08.2026: jeder Kontakt
  // stand auf "Sellify-Sperrvermerk nicht pruefbar", das Haekchen war
  // deaktiviert, und damit endete die Kette vor der Uebergabe. Das System hat
  // dabei richtig gehandelt — ohne Pruefung der Kontaktsperre darf niemand
  // angeschrieben werden; es konnte nur nicht pruefen.
  if (!state.sellifyPeople?.find || !state.sellifyCompanies?.find) {
    try {
      sellifyReadCollection('sellify_companies');
      sellifyReadCollection('sellify_people');
    } catch (error) {
      console.warn('[thesen-outbound] Sellify-Projektion fuer die Sperrvermerkspruefung nicht erreichbar', error);
    }
  }
  if (!state.sellifyPeople?.find || !state.sellifyCompanies?.find) {
    return { people: [], companies: [], contextAvailable: false };
  }
  try {
    const companies = [];
    const linkedContactId = Number(lead?.payload?.sellify_contact_id) || 0;
    if (linkedContactId && state.sellifyCompanies.findOne) {
      const linked = await state.sellifyCompanies.findOne(`sellify-company-${linkedContactId}`).exec();
      if (linked) companies.push(docJson(linked));
    }
    if (String(lead?.name || '').trim()) {
      companies.push(...await findSellifyRecords(state.sellifyCompanies, {
        name: { $eq: String(lead.name).trim() },
      }));
    }
    const activeCompanies = uniqueSellifyRecords(companies);
    const people = [];
    for (const company of activeCompanies) {
      const contactId = Number(company?.contact_id) || 0;
      if (contactId) people.push(...await findSellifyRecords(state.sellifyPeople, { contact_id: { $eq: contactId } }));
    }
    for (const contact of lead?.contacts || []) {
      const personId = normalizedPersonId(contact?.sellify_person_id || contact?.person_id || contact?.id);
      if (personId && state.sellifyPeople.findOne) {
        const person = await state.sellifyPeople.findOne(`sellify-person-${personId}`).exec();
        if (person) people.push(docJson(person));
      }
      const email = String(contact?.email || contact?.person_email || '').trim();
      if (email) people.push(...await findSellifyRecords(state.sellifyPeople, { email: { $eq: email } }));
      const displayName = personDisplayName(contact);
      if (normalizeProtectionText(displayName).split(' ').length >= 2) {
        people.push(...await findSellifyRecords(state.sellifyPeople, { display_name: { $eq: displayName } }));
      }
    }
    return { people: uniqueSellifyRecords(people), companies: activeCompanies, contextAvailable: true };
  } catch {
    return { people: [], companies: [], contextAvailable: false };
  }
}

async function refreshLeadRecipientEligibility(lead, { force = false } = {}) {
  if (!lead?.id) return new Map();
  if (!force && state.recipientEligibilityReady.has(lead.id)) {
    return new Map((lead.contacts || []).map((contact) => [
      contact.id,
      currentContactEligibility(lead, contact),
    ]));
  }
  const context = await loadSellifyRecipientContext(lead);
  const decisions = deriveLeadRecipientEligibility(lead, context);
  for (const [contactId, decision] of decisions) {
    state.recipientEligibility.set(recipientEligibilityKey(lead.id, contactId), decision);
  }
  state.recipientEligibilityReady.add(lead.id);
  return decisions;
}

async function refreshAllRecipientEligibility() {
  for (const lead of state.leads) await refreshLeadRecipientEligibility(lead);
}

async function enforceRecipientEligibility() {
  if (state.reconcilingRecipientEligibility) return 0;
  state.reconcilingRecipientEligibility = true;
  try {
    await refreshAllRecipientEligibility();
    const repaired = await repairLeadRecipientSelections();
    if (repaired) await reload();
    render();
    return repaired;
  } finally {
    state.reconcilingRecipientEligibility = false;
  }
}

function stableContactIdentity(contact, index = 0) {
  const parts = [
    contact?.name,
    contact?.first_name,
    contact?.last_name,
    contact?.person_vorname,
    contact?.person_nachname,
    contact?.email,
    contact?.person_email,
    contact?.phone,
    contact?.person_telefon,
    contact?.linkedin,
    contact?.xing,
  ].map((value) => String(value || '').trim().toLowerCase());
  return parts.some(Boolean) ? parts.join('|') : `position-${index}`;
}

function withStableContactIds(leadId, contacts = []) {
  const used = new Set();
  return (Array.isArray(contacts) ? contacts : []).map((value, index) => {
    const contact = value && typeof value === 'object' ? value : {};
    const explicitId = String(contact.id || '').trim();
    const baseId = explicitId || `contact_${fingerprint(`${leadId}|${stableContactIdentity(contact, index)}`)}`;
    let id = baseId;
    let suffix = 2;
    while (used.has(id)) id = `${baseId}_${suffix++}`;
    used.add(id);
    return contact.id === id ? contact : { ...contact, id };
  });
}

function normalizeLeadRecipientShape(lead) {
  const contacts = withStableContactIds(lead?.id || 'lead', lead?.contacts || []);
  const contactIds = new Set(contacts.map((contact) => contact.id));
  const selected = Array.isArray(lead?.selected_contact_ids) ? lead.selected_contact_ids : [];
  const selected_contact_ids = [...new Set(selected
    .map((id) => String(id || '').trim())
    .filter((id) => contactIds.has(id)))];
  return { ...lead, contacts, selected_contact_ids };
}

async function repairLeadRecipientSelections() {
  const docs = await state.collections.leads.find().exec();
  let repaired = 0;
  for (const doc of docs) {
    const current = doc.toJSON?.() || doc;
    const normalized = normalizeLeadRecipientShape(current);
    const decisions = new Map(normalized.contacts.map((contact) => [
      contact.id,
      currentContactEligibility(normalized, contact),
    ]));
    const plan = buildCampaignRecipientList(normalized, decisions);
    const selected_contact_ids = plan.recipients.map((contact) => contact.id);
    const contactsChanged = JSON.stringify(current.contacts || []) !== JSON.stringify(normalized.contacts);
    const selectionChanged = JSON.stringify(current.selected_contact_ids || []) !== JSON.stringify(selected_contact_ids);
    if (plan.excluded.length) {
      state.recipientRemovalNotices.set(normalized.id, plan.excluded);
    } else {
      const stillRestricted = (state.recipientRemovalNotices.get(normalized.id) || []).filter(({ contact }) => (
        currentContactEligibility(normalized, contact).status !== 'free'
      ));
      if (stillRestricted.length) state.recipientRemovalNotices.set(normalized.id, stillRestricted);
      else state.recipientRemovalNotices.delete(normalized.id);
    }
    if (!contactsChanged && !selectionChanged && Array.isArray(current.selected_contact_ids)) continue;
    await doc.incrementalPatch({
      contacts: normalized.contacts,
      selected_contact_ids,
      updated_at_ms: Date.now(),
    });
    repaired += 1;
  }
  return repaired;
}

async function repairUntrackedResearchStatuses() {
  const invalid = state.leads.filter((lead) => (
    ['queued', 'running'].includes(lead.research_status)
    && !String(lead.command_id || '').trim()
  ));
  if (!invalid.length) return 0;
  await Promise.all(invalid.map((lead) => patchLead(lead.id, {
    research_status: 'new',
    task_id: '',
    command_id: '',
    payload: {
      ...(lead.payload || {}),
      research_recovery_reason: 'untracked_automation_reset',
      research_recovered_at_ms: Date.now(),
    },
  })));
  return invalid.length;
}

async function toggleSource(id) {
  const doc = await state.collections.sources.findOne(id).exec();
  if (doc) await doc.incrementalPatch({ enabled: !doc.enabled, updated_at_ms: Date.now() });
}

async function addSource() {
  const url = String(await showBusinessPrompt('Vollständige Adresse der Recherchequelle', {
    title: 'Quelle hinzufügen',
    defaultValue: 'https://',
    confirmLabel: 'Weiter',
    cancelLabel: 'Abbrechen',
  }) || '').trim();
  if (!url) return;
  let parsed;
  try { parsed = new URL(url); } catch { showBusinessAlert('Bitte eine gültige URL eingeben.'); return; }
  if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname) {
    showBusinessAlert('Recherchequellen müssen eine vollständige HTTP- oder HTTPS-Adresse verwenden.');
    return;
  }
  const id = parsed.hostname.replace(/^www\./, '').toLowerCase();
  const credentialSecretName = String(await showBusinessPrompt('Name der CTOX-Zugangsreferenz. Leer lassen, wenn keine Anmeldung nötig ist.', {
    title: 'Zugang',
    defaultValue: '',
    confirmLabel: 'Quelle anlegen',
    cancelLabel: 'Abbrechen',
  }) || '').trim();
  const now = Date.now();
  const existing = await state.collections.sources.findOne(id).exec();
  if (existing) { showBusinessAlert('Diese Quelle ist bereits vorhanden.'); return; }
  await state.collections.sources.insert({
    id, label: id, url, countries: ['DE', 'AT', 'CH'], field_keys: [], enabled: true,
    requires_credential: Boolean(credentialSecretName), credential_secret_name: credentialSecretName, target_key: id.replace(/[^a-z0-9]+/g, '-'),
    adapter_status: 'draft', scrape_status: 'target_available', auth_status: credentialSecretName ? 'required' : 'not_required',
    payload: { builtin: false, secret_value_in_payload: false }, created_at_ms: now, updated_at_ms: now,
  });
}

function adapterCommandOperation(commandType, targetKey) {
  const operation = String(commandType || '').trim();
  const target_key = String(targetKey || '').trim();
  if (!operation || !target_key) throw new Error('Adapter-Operation und Ziel sind erforderlich.');
  return { operation, target_key };
}

async function runAdapterCommand(sourceId, commandType) {
  const item = state.sources.find((entry) => entry.id === sourceId);
  if (!item) return;
  const commandId = `cmd_thesen_source_${crypto.randomUUID()}`;
  const adapter = {
    id: `adapter_thesen_${item.target_key}`,
    source_id: item.id,
    label: item.label,
    url: item.url,
    adapter_kind: 'scrape_target',
    target_key: item.target_key,
    countries: item.countries,
    field_keys: item.field_keys,
    enabled: item.enabled,
    requires_credential: item.requires_credential,
    credential_secret_name: item.credential_secret_name,
    auth_mode: item.requires_credential ? 'browser_session' : 'none',
    secret_value_in_payload: false,
  };
  const title = commandType.endsWith('.generate_adapter')
    ? `Datenzugriff einrichten: ${item.label}`
    : commandType.endsWith('.test')
      ? `Datenzugriff prüfen: ${item.label}`
      : `Anmeldung vorbereiten: ${item.label}`;
  const operation = adapterCommandOperation(commandType, item.target_key);
  // The payload stays typed (operation + target_key) so the server never has to
  // interpret prose. What the operator reads is a plain sentence naming exactly
  // that operation and target — deterministic underneath, legible on screen.
  const operationText = `${title} — ${operation.operation} · Ziel ${operation.target_key}`;
  const command = {
    id: commandId, command_id: commandId, module: 'outbound', type: commandType, command_type: commandType,
    operation: operation.operation, target_key: operation.target_key,
    sync_queue_tasks: false,
    record_id: adapter.id, inbound_channel: 'business_os.thesen_outbound',
    payload: {
      ...operation,
      adapter_id: adapter.id, source_id: item.id, adapter,
      required_skills: ['thesen-outbound-research', 'universal-scraping', 'web-unlock'],
      scrape_contract: scrapeContract(item), secret_value_in_payload: false,
    },
    client_context: {
      source_module: 'thesen-outbound',
      source_id: item.id,
      actor: {
        id: state.ctx?.session?.user?.id || state.ctx?.session?.userId || '',
      },
    },
  };
  let result;
  try {
    result = requireTrackedSubmission(await state.ctx.businessChat.submitTask({
      ...command,
      title,
      text: operationText,
      instruction: operationText,
      user_message: operationText,
      // Ein Rechercheauftrag darf das Chatfenster nicht aufreissen. Bei einem
      // Kampagnenlauf sprang es sonst je Lead erneut auf.
      open: false,
      control_command: true,
      client_context: {
        ...command.client_context,
        action: 'context-chat',
        response_channel: 'business_os_chat',
      },
    }), { allowTerminalCommand: true, allowControlCommand: true });
  } catch (error) {
    result = { status: 'failed', error: String(error?.message || error), command_id: commandId, task_id: '' };
    showBusinessAlert(result.error);
  }
  if (commandType.endsWith('.auth_assist') && result.status !== 'failed') {
    await openSourceAuthorization(item, commandId);
  }
  const serverAdapter = result?.result?.adapter || result?.result?.outcome?.adapter || result?.adapter || {};
  const now = Date.now();
  const requestedStatus = commandType.endsWith('.generate_adapter')
    ? 'generation_queued'
    : commandType.endsWith('.test') ? 'test_requested' : 'auth_requested';
  const status = serverAdapter.status || (result.task_id ? requestedStatus : result.status || 'failed');
  const doc = await state.collections.adapters.findOne(adapter.id).exec();
  const patch = {
    id: adapter.id, source_id: item.id, status,
    scrape_status: serverAdapter.scrape_status || (commandType.endsWith('.test') ? 'test_requested' : 'registration_requested'),
    auth_status: serverAdapter.auth_status || (commandType.endsWith('.auth_assist') ? 'browser_session_requested' : item.auth_status),
    last_command_id: result.command_id || commandId,
    last_task_id: result.task_id || '',
    last_error: result.error || serverAdapter.last_error || '',
    payload: { result: sanitizeCommandResult(result), secret_value_in_payload: false },
    created_at_ms: doc?.created_at_ms || now, updated_at_ms: now,
  };
  if (doc) await doc.incrementalPatch(patch); else await state.collections.adapters.insert(patch);
  const sourceDoc = await state.collections.sources.findOne(item.id).exec();
  await sourceDoc?.incrementalPatch({ adapter_status: patch.status, scrape_status: patch.scrape_status, auth_status: patch.auth_status, updated_at_ms: now });
}

async function openSourceAuthorization(item, commandId) {
  const openApp = state.ctx?.openApp || state.ctx?.openDesktopApp;
  if (typeof openApp !== 'function') {
    showBusinessAlert('Die Browser-App konnte nicht geöffnet werden. Öffnen Sie „Browser“ und wählen Sie die angeforderte Anmeldung.');
    return;
  }
  const sourceSlug = rxdbIdSlug(item.id);
  const commandSlug = rxdbIdSlug(commandId);
  const sessionId = `browser_session_web_stack_auth_${sourceSlug}_${commandSlug}`;
  const tabId = `browser_tab_web_stack_auth_${sourceSlug}_${commandSlug}`;
  const hostname = new URL(item.url).hostname;
  await openApp('browser', {
    args: {
      session_id: sessionId,
      tab_id: tabId,
      source_id: item.id,
      purpose: 'web_stack_auth',
      target_url: item.url,
      allowed_domains: [hostname],
      secret_name: item.credential_secret_name || '',
    },
  });
}

async function reconcileAdapterCommands({ authoritative = false } = {}) {
  if (!state.commandCollection || state.reconcilingAdapterCommands) return false;
  const trackedAdapters = state.adapters.filter((adapter) => String(adapter.last_command_id || '').trim());
  if (!trackedAdapters.length) return false;
  state.reconcilingAdapterCommands = true;
  let changed = false;
  try {
    const commandIds = [...new Set(trackedAdapters.map((adapter) => String(adapter.last_command_id).trim()))];
    const query = authoritative
      ? {
          selector: { id: { $in: commandIds } },
          limit: Math.max(50, commandIds.length * 2),
          requireRevision: `thesen-outbound-adapters:${Math.floor(Date.now() / COMMAND_REFRESH_MS)}`,
        }
      : {};
    const docs = await state.commandCollection.find(query).exec();
    const commands = new Map(docs.map((doc) => {
      const command = doc.toJSON?.() || doc;
      return [String(command.command_id || command.id || '').trim(), command];
    }));
    for (const adapter of trackedAdapters) {
      const command = commands.get(String(adapter.last_command_id || '').trim());
      const patch = adapterCommandRecordPatch(adapter, command);
      if (!patch) continue;
      const adapterDoc = await state.collections.adapters.findOne(adapter.id).exec();
      if (!adapterDoc) continue;
      await adapterDoc.incrementalPatch(patch);
      const sourceDoc = await state.collections.sources.findOne(adapter.source_id).exec();
      await sourceDoc?.incrementalPatch({
        adapter_status: patch.status,
        scrape_status: patch.scrape_status,
        auth_status: patch.auth_status,
        updated_at_ms: patch.updated_at_ms,
      });
      changed = true;
    }
  } finally {
    state.reconcilingAdapterCommands = false;
  }
  return changed;
}

function adapterCommandRecordPatch(adapter, command) {
  if (!command) return null;
  const commandId = String(command.command_id || command.id || '').trim();
  const status = String(
    command.terminal_status || command.status || command.result?.status || '',
  ).trim().toLowerCase();
  if (!['completed', 'failed', 'blocked', 'cancelled', 'canceled'].includes(status)) return null;
  if (adapter.payload?.reconciled_command_id === commandId) return null;
  const serverAdapter = command.result?.adapter
    || command.result?.outcome?.adapter
    || command.payload?.outcome?.adapter
    || {};
  const failed = ['failed', 'blocked', 'cancelled', 'canceled'].includes(status);
  const nextStatus = String(serverAdapter.status || (failed ? 'failed' : status));
  const nextScrapeStatus = String(
    serverAdapter.scrape_status || (failed ? 'failed' : adapter.scrape_status || 'test_requested'),
  );
  const nextAuthStatus = String(serverAdapter.auth_status || adapter.auth_status || 'not_required');
  const error = String(
    command.error_message
      || command.error
      || command.result?.error
      || serverAdapter.last_error
      || '',
  );
  return {
    status: nextStatus,
    scrape_status: nextScrapeStatus,
    auth_status: nextAuthStatus,
    last_error: error,
    payload: {
      ...(adapter.payload || {}),
      reconciled_command_id: commandId,
      reconciled_command_status: status,
      result: sanitizeCommandResult(command),
      secret_value_in_payload: false,
    },
    updated_at_ms: Date.now(),
  };
}

function sourceNeedsBrowserAuthorization(item, adapter) {
  if (item?.requires_credential) return true;
  const status = String(adapter?.status || item?.adapter_status || '').toLowerCase();
  const scrapeStatus = String(adapter?.scrape_status || item?.scrape_status || '').toLowerCase();
  const authStatus = String(adapter?.auth_status || item?.auth_status || '').toLowerCase();
  return status.includes('auth_required')
    || ['blocked', 'auth_required', 'browser_required'].includes(scrapeStatus)
    || ['required', 'auth_required', 'browser_session_requested'].includes(authStatus);
}

function rxdbIdSlug(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function scrapeContract(item) {
  return {
    skill: 'thesen-outbound-research',
    skills: ['thesen-outbound-research', 'universal-scraping', 'web-unlock'],
    target_key: item.target_key,
    source_id: item.id,
    output_schema: 'prospect.v1',
    min_independent_sources: 2,
    fallback: {
      allow_browser_assist: true,
      credential_ref: item.credential_secret_name ? `ctox-secret://credentials/${item.credential_secret_name}` : '',
    },
    unlock: { detect_access_challenge: true, record_signal: true, allow_access_control_bypass: false },
  };
}

async function createCampaign() {
  const name = String(await showBusinessPrompt(tr('campaignNamePrompt', 'Name der neuen Kampagne'), {
    title: tr('newCampaign', 'Neue Kampagne'),
    placeholder: tr('campaignName', 'Kampagnenname'),
    confirmLabel: tr('continue', 'Weiter'),
  }) || '').trim();
  if (!name) return;
  await openImporter(name);
}

async function renameCampaign(currentName) {
  const current = String(currentName || '').trim();
  if (!current) return;
  const next = String(await showBusinessPrompt('Neuer Name der Kampagne', {
    title: 'Kampagne umbenennen',
    defaultValue: current,
    confirmLabel: 'Umbenennen',
  }) || '').trim();
  if (!next || next === current) return;
  if (campaignRows().some((campaign) => campaign.name === next)) {
    await showBusinessAlert('Eine Kampagne mit diesem Namen existiert bereits.');
    return;
  }
  const leads = campaignLeads(current);
  await Promise.all(leads.map((lead) => patchLead(lead.id, {
    campaign: next,
    payload: {
      ...(lead.payload || {}),
      previous_campaign: current,
      campaign_renamed_at_ms: Date.now(),
    },
  })));
  const matchingImports = state.imports.filter((item) => String(item.title || '').trim() === current);
  await Promise.all(matchingImports.map(async (item) => {
    const doc = await state.collections.imports.findOne(item.id).exec();
    if (doc) await doc.incrementalPatch({ title: next, updated_at_ms: Date.now() });
  }));
  const run = state.campaignRuns.get(current);
  if (run) {
    state.campaignRuns.delete(current);
    state.campaignRuns.set(next, run);
  }
  state.selectedCampaign = next;
  await reload();
  render();
}

async function deleteCampaign(campaignName) {
  const campaign = String(campaignName || '').trim();
  if (!campaign) return;
  const leads = campaignLeads(campaign);
  const running = leads.filter((lead) => ['queued', 'running'].includes(lead.research_status));
  // A lead can keep `running` long after its task is gone — the status is a
  // durable field, not a heartbeat. Refusing on it alone made campaigns
  // permanently undeletable. Only a run that is still reporting blocks; a
  // stale one is named and left to the operator to decide.
  const live = running.filter((lead) => Date.now() - Number(
    lead.research_updated_at_ms || lead.research_started_at_ms || lead.updated_at_ms || 0,
  ) < RESEARCH_HEARTBEAT_STALE_MS);
  if (live.length > 0) {
    await showBusinessAlert(`Die Kampagne kann nicht gelöscht werden, solange ${live.length} Recherche-Task${live.length === 1 ? '' : 's'} aktiv sind.`);
    return;
  }
  const staleNote = running.length
    ? ` ${running.length} Lauf${running.length === 1 ? '' : 'e'} steht noch auf „läuft“, meldet sich aber nicht mehr — der wird mitgelöscht.`
    : '';
  const confirmed = await showBusinessConfirm(
    `Die Kampagne „${campaign}“ und ${leads.length} Lead${leads.length === 1 ? '' : 's'} werden dauerhaft gelöscht.${staleNote}`,
    {
      title: 'Kampagne löschen',
      confirmLabel: 'Endgültig löschen',
      requireText: campaign,
      kind: 'danger',
    },
  );
  if (!confirmed) return;
  await Promise.all(leads.map(async (lead) => {
    const doc = await state.collections.leads.findOne(lead.id).exec();
    if (doc) await doc.remove();
  }));
  const matchingImports = state.imports.filter((item) => String(item.title || '').trim() === campaign);
  await Promise.all(matchingImports.map(async (item) => {
    const doc = await state.collections.imports.findOne(item.id).exec();
    if (doc) await doc.remove();
  }));
  state.campaignRuns.delete(campaign);
  state.selectedLeadIds.clear();
  state.selectedLeadId = '';
  state.selectedCampaign = '';
  await reload();
  render();
}

function openLeadEditor(id) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead) return;
  state.leadEditorId = lead.id;
  state.leadDraft = {
    name: lead.name || '',
    website: lead.website || '',
    address_line: firstValue(lead.data, ['address_line', 'address', 'street', 'strasse']) || '',
    postal_code: firstValue(lead.data, ['postal_code', 'postcode', 'plz']) || '',
    city: lead.city || firstValue(lead.data, ['city', 'ort']) || '',
    country: lead.country || 'DE',
    email: firstValue(lead.data, ['email', 'company_email', 'e_mail']) || '',
    phone: firstValue(lead.data, ['phone', 'company_phone', 'telefon']) || '',
    campaign: lead.campaign || '',
  };
  state.leadEditorOpen = true;
  renderLeadEditor();
}

function closeLeadEditor() {
  state.leadEditorOpen = false;
  state.leadEditorId = '';
  state.leadDraft = null;
  renderLeadEditor();
}

async function saveLeadEditor() {
  const lead = state.leads.find((entry) => entry.id === state.leadEditorId);
  const draft = state.leadDraft;
  if (!lead || !draft) return;
  const name = String(draft.name || '').trim();
  const campaign = String(draft.campaign || '').trim();
  if (!name || !campaign) {
    await showBusinessAlert('Organisation und Kampagne sind Pflichtfelder.');
    return;
  }
  const website = String(draft.website || '').trim();
  const nextData = {
    ...(lead.data || {}),
    address_line: String(draft.address_line || '').trim(),
    postal_code: String(draft.postal_code || '').trim(),
    city: String(draft.city || '').trim(),
    email: String(draft.email || '').trim(),
    phone: String(draft.phone || '').trim(),
  };
  await patchLead(lead.id, {
    name,
    campaign,
    website,
    domain: website ? domainFromUrl(website) : '',
    city: nextData.city,
    country: normalizedResearchCountry(draft.country),
    data: nextData,
    payload: {
      ...(lead.payload || {}),
      manually_edited_at_ms: Date.now(),
    },
  });
  state.selectedCampaign = campaign;
  closeLeadEditor();
  await reload();
  render();
}

async function saveResearchPolicy() {
  const instructions = String(state.researchPolicyDraft || '').trim();
  if (!instructions) {
    showBusinessAlert(tr('policyRequired', 'Der Rechercheablauf darf nicht leer sein.'));
    return;
  }
  const existingDoc = await state.collections.researchPolicies.findOne(RESEARCH_POLICY_ID).exec();
  const existing = existingDoc?.toJSON?.() || existingDoc;
  const record = researchPolicyRecord(existing, instructions);
  if (existingDoc) await existingDoc.incrementalPatch(record);
  else await state.collections.researchPolicies.insert(record);
  state.researchPolicy = instructions;
  state.researchPolicyDraft = instructions;
  showBusinessAlert(tr('policySaved', 'Rechercheablauf gespeichert.'));
}

async function openImporter(defaultTitle = '') {
  await openUniversalImporter(state.ctx, {
    side: 'right', moduleId: 'thesen-outbound', entityType: 'lead', commandType: 'thesen-outbound.import',
    title: 'Leads importieren', kicker: 'Neu- und Nachrecherche', defaultSource: 'excel', showFileExplorer: false,
    defaultTitle: defaultTitle || `Recherche ${new Date().getFullYear()}`, helperText: 'Excel, Text oder URL importieren. Die Recherche beginnt erst nach der Sichtprüfung.',
    // Das Importfenster ist ein Vollbild-Overlay ueber der GESAMTEN Shell, nicht
    // nur ueber dieser App. Es offen zu halten blockiert das ganze System,
    // solange der Import laeuft. Es schliesst deshalb sofort; die Rueckmeldung
    // ist die Kampagne, die mit ihrer Lead-Zahl in der Liste erscheint.
    submitLabel: 'Importieren', submittingLabel: 'Import wird verarbeitet...', doneLabel: 'Leads importiert.', closeOnSubmit: true, dispatch: false,
    onImport: async ({ payload }) => importPayload(payload),
  });
}

async function importPayload(payload) {
  const now = Date.now();
  const importId = `import_${crypto.randomUUID()}`;
  const rows = await extractRows(payload);
  const normalizedRows = rows.length ? rows : payload.source_type === 'url' ? [{ name: new URL(payload.source.url).hostname, website: payload.source.url, raw: {} }] : [];
  await state.collections.imports.insert({
    id: importId, title: payload.title || 'Lead-Import', source_type: payload.source_type || 'text',
    status: normalizedRows.length ? 'imported' : 'empty', lead_count: normalizedRows.length,
    payload: { source_url: payload.source?.url || '', file_names: (payload.source?.files || []).map((file) => file.name), secret_value_in_payload: false },
    created_at_ms: now, updated_at_ms: now,
  });
  for (const [index, raw] of normalizedRows.entries()) {
    const row = normalizeCompanyRow(raw, index);
    if (!row.name) continue;
    const id = `lead_${fingerprint(`${row.name}|${row.domain || row.website}|${row.country}`)}`;
    const existing = await state.collections.leads.findOne(id).exec();
    const lifecycle = resetLeadForImport(existing);
    const lead = {
      id, import_id: importId, campaign: payload.title || `Recherche ${new Date().getFullYear()}`,
      name: row.name, domain: row.domain || domainFromUrl(row.website), website: row.website || '', city: row.city || '', country: row.country || 'DE',
      ...lifecycle, data: existing?.data || {}, contacts: withStableContactIds(id, existing?.contacts || []),
      payload: {
        ...(existing?.payload || {}),
        imported_row: row.raw || raw,
        min_independent_sources: 2,
        previous_import_id: existing?.import_id || '',
        previous_campaign: existing?.campaign || '',
      },
      created_at_ms: existing?.created_at_ms || now, updated_at_ms: now,
    };
    if (existing) await existing.incrementalPatch(lead); else await state.collections.leads.insert(lead);
  }
  return { status: 'completed', message: `${normalizedRows.length} Leads importiert.` };
}

function resetLeadForImport(existing) {
  return {
    research_status: 'new',
    validation_status: 'pending',
    sellify_status: 'not_started',
    task_id: '',
    command_id: '',
    selected_contact_ids: [],
    evidence: [],
  };
}

async function extractRows(payload) {
  if (payload.source_type === 'text') return extractCompanyRowsFromText(payload.source?.text || '');
  const rows = [];
  for (const file of payload.source?.files || []) {
    if (/\.xlsx$/i.test(file.name || '')) rows.push(...await extractCompanyRowsFromWorkbookFile(file));
    else if (/\.(csv|tsv|txt)$/i.test(file.name || '')) rows.push(...parseDelimitedText(file.text || '').map((row, index) => normalizeCompanyRow(row, index)));
  }
  return rows;
}

async function researchLead(id, options = {}) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead) return { status: 'missing', error: 'Lead nicht gefunden.' };
  const commandId = `thesen-lead-research-${crypto.randomUUID()}`;
  // Die Dublettenpruefung entscheidet ueber Neu- oder Nachrecherche. Faellt sie
  // aus (Projektion noch nicht da), bleibt es bei der bisherigen Herleitung.
  let bekannteFirma = null;
  let vorwissen = null;
  try {
    vorwissen = await sellifyVorwissen(lead);
    bekannteFirma = vorwissen;
  } catch (error) {
    console.warn('[thesen-outbound] Sellify-Dublettenpruefung vor der Recherche fehlgeschlagen', error);
  }
  const researchMode = leadResearchMode(lead, bekannteFirma);
  const crmWissen = sellifyVorwissenAlsText(vorwissen);
  const prompt = [
    `Recherchiere den Lead ${lead.name} [${lead.id}] nach dem THESEN-Quellenstandard.`,
    // Das CRM steht bewusst GANZ OBEN im Auftrag: was das Haus schon weiss, soll
    // die Recherche leiten, nicht am Ende mit ihr abgeglichen werden.
    ...(crmWissen ? [crmWissen] : []),
    'Nutze die konfigurierten Quellen und Playwright-Adapter; bei Zugriffshürden den Web-Stack-Unlocking- und Browser-Anmeldeprozess.',
    'Sichere jedes übernommene Feld mit mindestens zwei unabhängigen Quellen ab.',
    'Erfasse ALLE auffindbaren Ansprechpartner des Unternehmens, nicht nur einen: '
    + 'Geschäftsführung, Vertretungsberechtigte, Vertriebsleitung, Einkauf. '
    + 'Gib sie als Liste zurück, jeden mit Anrede, Titel, Vorname, Nachname, Funktion, '
    + 'und — falls auffindbar — Telefon und E-Mail, jeweils mit Quelle.',
    'Der strukturierte Recherche-Command läuft in diesem Chat. Das Ergebnis wird danach ohne freie Texttransformation in den Lead übernommen.',
  ].join('\n');
  try {
    if (typeof state.ctx?.businessChat?.submitTask !== 'function') {
      throw new Error('CTOX Chat ist nicht verfügbar. Die Recherche wurde nicht gestartet.');
    }
    const result = requireTrackedSubmission(await state.ctx.businessChat.submitTask({
      instruction: prompt,
      prompt,
      user_message: prompt,
      title: `Nachrecherche: ${lead.name}`,
      // Ein Rechercheauftrag darf das Chatfenster nicht aufreissen. Bei einem
      // Kampagnenlauf sprang es sonst je Lead erneut auf.
      open: false,
      module: 'thesen-outbound',
      source_module: 'thesen-outbound',
      command_id: commandId,
      command_type: 'web_stack.person_research',
      control_command: true,
      record_id: lead.id,
      thread_key: options.threadKey || `business-os/thesen-outbound/lead/${lead.id}`,
      mode: researchMode,
      target: 'data',
      required_skills: ['thesen-outbound-research', 'universal-scraping', 'web-unlock'],
      writeback_contract: {
        collection: 'thesen_outbound_leads',
        allowed_collections: ['thesen_outbound_leads'],
        command_type: 'web_stack.person_research',
        record_ids: [lead.id],
        min_independent_sources: 2,
      },
      payload: {
        lead_id: lead.id,
        company: lead.name,
        country: normalizedResearchCountry(lead.country),
        mode: researchMode,
        fields: [...RESEARCH_FIELDS],
        include_private: [],
        auto_browser_capture: true,
        campaign_run_id: options.campaignRunId || '',
        workflow_id: options.campaignRunId || '',
        response_channel: 'business_os_chat',
        title: `Nachrecherche: ${lead.name}`,
        source_policy: enabledSourcePolicy(),
      },
      client_context: {
        action: 'context-chat',
        source: 'thesen-outbound-lead-research',
        source_module: 'thesen-outbound',
        record_id: lead.id,
        response_channel: 'business_os_chat',
        writeback_required: true,
        campaign_run_id: options.campaignRunId || '',
      },
    }), { allowControlCommand: true });
    await patchLead(id, {
      research_status: 'running',
      command_id: result.command_id,
      task_id: result.task_id,
      payload: {
        ...lead.payload,
        campaign_run_id: options.campaignRunId || '',
        research_started_at_ms: Date.now(),
      },
    });
    return { status: 'queued', commandId: result.command_id, taskId: result.task_id || '' };
  } catch (error) {
    const message = String(error?.message || error);
    // Ein abgelaufener Wartender ist KEIN gescheiterter Auftrag. Der Command-Bus
    // meldet den Zeitablauf ausdruecklich als transient/projection_delayed: der
    // Vorgang laeuft weiter, nur die Projektion kam nicht rechtzeitig hinterher.
    // Vorher setzte jeder Fehler den Lead auf failed — am 11.08.2026 lagen
    // dadurch bei vier Firmen (ANGUS, Berg, Dr. Kurt Richter, Aeroxon) je 21
    // eingebettete Felder samt Quellen fertig im Ergebnis, waehrend der Lead
    // "fehlgeschlagen, 0 belegt" zeigte. Der Lead bleibt jetzt verfolgbar; der
    // Abgleich in reconcileResearchCommands wendet das Ergebnis an, sobald die
    // Projektion es sichtbar macht.
    if (isTransientResearchWaitError(error)) {
      await patchLead(id, {
        research_status: 'running',
        command_id: commandId,
        payload: {
          ...lead.payload,
          campaign_run_id: options.campaignRunId || '',
          research_wait_note: message,
          research_wait_since_ms: Number(lead.payload?.research_wait_since_ms || Date.now()),
        },
      });
      return { status: 'running', commandId, note: message };
    }
    await patchLead(id, {
      research_status: 'failed',
      command_id: commandId,
      payload: {
        ...lead.payload,
        campaign_run_id: options.campaignRunId || '',
        research_error: message,
        research_finished_at_ms: Date.now(),
      },
    });
    if (!options.suppressAlerts) showBusinessAlert(message);
    return { status: 'failed', commandId, error: message };
  }
}

// Eine Firma, die im CRM bereits existiert, ist eine NACHrecherche — auch wenn
// wir sie selbst noch nie uebergeben haben. Vorher entschied allein
// `sellify_status`, also ob WIR schon einmal geschrieben hatten; eine seit
// Jahren in Sellify gefuehrte Firma lief damit als Neuanlage, und die im CRM
// vorhandenen Ansprechpartner (im Schnitt 3,5 je Firma) blieben unbeachtet.
function leadResearchMode(lead, existingSellifyCompany = null) {
  if (lead?.sellify_status === 'completed') return 'update_firm';
  return existingSellifyCompany ? 'update_firm' : 'new_record';
}

async function reconcileResearchCommands({ authoritative = false } = {}) {
  if (!state.commandCollection || state.reconcilingCommands) return false;
  state.reconcilingCommands = true;
  let changed = false;
  try {
    // Auch failed/needs_review abgleichen: ein Lead, dessen früher Lauf
    // scheiterte, verließ sonst die Menge für immer — ein später doch noch
    // eingetroffenes completed-Ergebnis (samt gefundener Kontakte) wurde nie
    // mehr angewendet. Der Beobachtungsschlüssel verhindert Doppelarbeit.
    const pendingLeads = state.leads.filter((lead) => (
      ['queued', 'running', 'failed', 'needs_review'].includes(lead.research_status)
    ));
    const demandedCommands = authoritative
      ? await demandResearchCommands(pendingLeads)
      : [];
    const commandDocs = await state.commandCollection.find().exec();
    const commands = uniqueCommands([
      ...demandedCommands,
      ...commandDocs.map((doc) => doc.toJSON?.() || doc),
    ]);
    for (const lead of pendingLeads) {
      const command = researchCommandForLead(lead, commands);
      // Ein Lead ohne auffindbaren Vorgang darf nicht ewig "laeuft" anzeigen.
      // ANGUS Chemie stand am 11.08.2026 ueber fuenf Stunden auf running,
      // obwohl sein Vorgang um 12:04 fertig war — fuer den Nutzer nicht von
      // einem haengenden System zu unterscheiden. Nach der Obergrenze sagt der
      // Lead ehrlich, dass die Rueckmeldung ausblieb; ein spaeter doch noch
      // gefundener Vorgang wird oben trotzdem weiter angewendet, weil failed
      // Teil der abgeglichenen Menge bleibt.
      if (!command) {
        if (lead.research_status !== 'running' && lead.research_status !== 'queued') continue;
        const startedAt = Number(
          lead.payload?.research_wait_since_ms || lead.payload?.research_started_at_ms || 0,
        );
        if (!startedAt || Date.now() - startedAt < RESEARCH_RUNNING_MAX_MS) continue;
        await patchLead(lead.id, {
          research_status: 'failed',
          payload: {
            ...lead.payload,
            research_finished_at_ms: Date.now(),
            research_error: 'Der Vorgang hat sich nicht zurueckgemeldet. Recherche bitte erneut starten.',
          },
        });
        changed = true;
        continue;
      }
      const observedCommandId = String(command.command_id || command.id || '').trim();
      const observationKey = researchCommandObservationKey(command);
      if (lead.payload?.observed_research_command_key === observationKey) continue;
      const patch = researchCommandLeadPatch(lead, command);
      if (!patch) continue;
      patch.payload = {
        ...(patch.payload || lead.payload || {}),
        observed_research_command_key: observationKey,
      };
      const chatEventTarget = window.top && window.top !== window ? window.top : window;
      chatEventTarget.postMessage({
        type: 'ctox-business-os-command-observed',
        command,
      }, window.location.origin);
      if (chatEventTarget === window) {
        chatEventTarget.dispatchEvent(new CustomEvent('ctox-business-os-command-observed', {
          detail: { command },
        }));
      }
      await patchLead(lead.id, patch);
      changed = true;
    }
  } finally {
    state.reconcilingCommands = false;
  }
  return changed;
}

async function demandResearchCommands(leads = []) {
  const leadIds = [...new Set(leads.map((lead) => String(lead?.id || '').trim()).filter(Boolean))];
  if (!leadIds.length) return [];
  const workflowIds = [...new Set(leads
    .map((lead) => String(lead?.payload?.campaign_run_id || '').trim())
    .filter(Boolean))]
    .sort();
  const revisionWindow = Math.floor(Date.now() / COMMAND_REFRESH_MS);
  const docs = await state.commandCollection.find({
    selector: {
      command_type: 'web_stack.person_research',
      record_id: { $in: leadIds },
    },
    limit: Math.max(200, leadIds.length * 8),
    requireRevision: `thesen-outbound:${workflowIds.join(',')}:${revisionWindow}`,
  }).exec();
  return docs.map((doc) => doc.toJSON?.() || doc);
}

function uniqueCommands(commands = []) {
  const byId = new Map();
  for (const command of commands) {
    const id = String(command?.command_id || command?.id || '').trim();
    if (id) byId.set(id, command);
  }
  return [...byId.values()];
}

// Der Command-Bus kennzeichnet einen Zeitablauf beim Warten selbst als transient
// und wiederholbar (shared/command-bus.js, code 'projection_delayed'). Wir lesen
// diese Kennzeichnung, statt den Text zu vergleichen; der Text bleibt nur als
// letzter Rueckfall, falls ein aelterer Bus die Felder noch nicht mitschickt.
function isTransientResearchWaitError(error) {
  if (!error) return false;
  const code = String(error.code || error.details?.code || '').trim();
  if (code === 'projection_delayed' || code === 'projection_pending') return true;
  const status = String(error.status || error.details?.status || '').trim();
  if (status === 'projection_pending') return true;
  if (error.transient === true || error.details?.transient === true) return true;
  return /wartet noch auf die R/i.test(String(error.message || ''));
}

// Kennung, MIT DER SICH INHALTLICH ETWAS GEAENDERT HAT — bewusst ohne Zeitstempel.
//
// Vorher stand updated_at_ms mit im Schluessel. Auf der Kundeninstanz laeuft eine
// Schreibschleife im nativen Peer, die dieselben sechs Vorgangsdokumente
// unveraendert immer wieder neu schreibt (am 11.08.2026 gemessen: zeitweise ueber
// 100 Revisionen pro Minute, unabhaengig nachgewiesen bei replicationUp=false,
// also voellig ohne Browser). Jedes dieser Neuschreiben hob updated_at_ms an, damit
// aenderte sich der Schluessel, damit galt der Vorgang als neu beobachtet — und der
// Browser schrieb den Lead erneut. Die Anzeige haette die Schleife also zusaetzlich
// angeheizt, sobald jemand das Modul offen laesst.
//
// Der Status bleibt im Schluessel: der Uebergang failed -> completed muss weiterhin
// durchkommen, denn genau darauf beruht das Nachholen verspaeteter Ergebnisse.
// Die Ursache der Schleife selbst liegt im Peer und gehoert nicht hierher; dies ist
// die Bremse auf unserer Seite, keine Behebung.
function researchCommandObservationKey(command) {
  return [
    String(command?.command_id || command?.id || '').trim(),
    String(command?.terminal_status || command?.task_status || command?.status || command?.execution_phase || '').trim(),
  ].join(':');
}

function researchCommandForLead(lead, commands = []) {
  const campaignRunId = String(lead?.payload?.campaign_run_id || '').trim();
  return commands
    .filter((command) => command?.command_type === 'web_stack.person_research')
    .filter((command) => String(command.record_id || '').trim() === String(lead?.id || '').trim())
    .filter((command) => {
      const commandRunId = String(
        command?.payload?.campaign_run_id
        || command?.payload?.workflow_id
        || command?.workflow_id
        || '',
      ).trim();
      return !campaignRunId || !commandRunId || commandRunId === campaignRunId;
    })
    .sort((left, right) => {
      // Ein später fehlgeschlagener oder abgebrochener Lauf darf ein früheres
      // completed-Ergebnis nicht dauerhaft verdecken: sonst gehen die dort
      // gefundenen Kontakte verloren, obwohl die Recherche sie geliefert hat.
      // Rangfolge: aktive Läufe (Live-Status) > completed > failed/cancelled,
      // innerhalb der Stufe entscheidet die Aktualität.
      const tier = (command) => {
        const status = normalizedResearchCommandStatus(command);
        if (['accepted', 'queued', 'running', 'leased', 'retry_wait', 'working'].includes(status)) return 0;
        if (status === 'completed') return 1;
        return 2;
      };
      if (tier(left) !== tier(right)) return tier(left) - tier(right);
      return (
        Number(right.updated_at_ms || right.created_at_ms || 0)
        - Number(left.updated_at_ms || left.created_at_ms || 0)
      );
    })[0] || null;
}

// Ein Feld mit genau einer fundierten Quelle darf der Nutzer bewusst
// freigeben. Die Entscheidung wird als eigener, unabhängiger Beleg mit
// source_id "operator" protokolliert — sichtbar in der Quellenliste, zählbar
// für die Zwei-Quellen-Regel, und nachvollziehbar statt versteckt.
async function approveResearchField(fieldKey) {
  const lead = selectedLead();
  if (!lead || !fieldKey) return;
  const value = researchFieldValue(lead, fieldKey);
  if (!value) return;
  const evidence = deduplicateEvidence([...(lead.evidence || []), {
    field_key: fieldKey,
    value,
    confidence: 'operator',
    source_id: 'operator',
    source_url: '',
    tier: 'O',
    via: 'manual-approval',
    label: tr('operatorApproved', 'Vom Nutzer freigegeben'),
  }]);
  const draft = { ...lead, evidence };
  const researched = lead.payload?.researched_field_keys || [];
  const allProven = researched.length > 0
    && researched.every((key) => independentFieldEvidenceCount(draft, key) >= 2);
  const research_status = allProven && lead.research_status === 'needs_review'
    ? 'completed'
    : lead.research_status;
  const payload = {
    ...lead.payload,
    operator_approved_field_keys: [...new Set([
      ...(lead.payload?.operator_approved_field_keys || []),
      fieldKey,
    ])],
  };
  // Die Freigabe wartete bisher auf den vollen Replikationsumlauf, bevor sich in
  // der Ansicht etwas bewegte — bis zu einer Minute Stille nach dem Klick. Der
  // lokale Stand wird deshalb sofort gesetzt und gezeichnet; der Schreibvorgang
  // laeuft dahinter und korrigiert bei Bedarf.
  const eintrag = state.leads.find((entry) => entry.id === lead.id);
  if (eintrag) {
    eintrag.evidence = evidence;
    eintrag.research_status = research_status;
    eintrag.payload = payload;
  }
  renderDetail();
  try {
    await patchLead(lead.id, { evidence, research_status, payload });
  } catch (error) {
    console.warn('[thesen-outbound] Freigabe konnte nicht gespeichert werden', error);
    await reload();
    renderDetail();
  }
}

async function validateLead(id) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead) return;
  if (!leadReadyForValidation(lead)) {
    showBusinessAlert('Für die Freigabe sind eine abgeschlossene Recherche und mindestens zwei unabhängige Quellen erforderlich.');
    return;
  }
  await patchLead(id, { validation_status: 'validated', payload: { ...lead.payload, validated_at_ms: Date.now() } });
}

// The detail pane asks this before it enables either Sellify action, so the
// operator sees WHY a handoff is unavailable instead of a dead button. It was
// called at index.js:813 but never defined — every lead selection therefore
// threw a ReferenceError and the pane never rendered.
function sellifyHandoffPrecondition(lead) {
  if (!lead || lead.validation_status !== 'validated') {
    return 'Bitte den Lead vor der Übergabe validieren.';
  }
  if (!state.recipientEligibilityReady.has(lead.id)) {
    return 'Die Sellify-Sperrvermerke werden noch geprüft.';
  }
  const plan = buildCampaignRecipientList(lead);
  if (!plan.recipients.length) {
    return plan.excluded.length
      ? 'Alle ausgewählten Personen sind gesperrt oder müssen zuerst geprüft werden.'
      : 'Bitte mindestens eine Person auswählen, bevor der Lead an Sellify übergeben wird.';
  }
  return '';
}

// The recipient checkbox writes through here. It was wired at index.js:960 and
// never defined, so no selection could ever be persisted — which is why picking
// three people still showed zero selected.
async function setContactRecipientSelection(id, contactId, selected) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead || !contactId) return;
  if (!state.recipientEligibilityReady.has(id)) await refreshLeadRecipientEligibility(lead);
  const normalized = normalizeLeadRecipientShape(lead);
  const contact = normalized.contacts.find((entry) => entry.id === contactId);
  if (!contact) return;
  const decision = currentContactEligibility(normalized, contact);
  if (selected && decision.status !== 'free') {
    renderDetail();
    return;
  }
  const selectedIds = new Set(normalized.selected_contact_ids);
  if (selected) selectedIds.add(contactId);
  else selectedIds.delete(contactId);
  const selected_contact_ids = [...selectedIds].filter((entry) => (
    normalized.contacts.some((candidate) => candidate.id === entry
      && currentContactEligibility(normalized, candidate).status === 'free')
  ));
  lead.selected_contact_ids = selected_contact_ids;
  await patchLead(id, { selected_contact_ids });
  renderDetail();
}

async function sendLeadToSellify(id, { includeCampaign = false } = {}) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead || lead.validation_status !== 'validated') { showBusinessAlert('Bitte den Lead vor der Übergabe validieren.'); return; }
  // This is the canonical recipient-list boundary. Even a manipulated checkbox
  // state is rechecked against current Sellify person/company remarks before any
  // person or campaign command can be created.
  const decisions = await refreshLeadRecipientEligibility(lead, { force: true });
  const recipientPlan = buildCampaignRecipientList(lead, decisions);
  const selectedContacts = recipientPlan.recipients;
  if (recipientPlan.excluded.length) {
    const selected_contact_ids = selectedContacts.map((contact) => contact.id);
    lead.selected_contact_ids = selected_contact_ids;
    state.recipientRemovalNotices.set(id, recipientPlan.excluded);
    await patchLead(id, { selected_contact_ids });
    renderCenter();
    renderDetail();
  }
  if (!selectedContacts.length) {
    showBusinessAlert(recipientPlan.excluded.length
      ? 'Die ausgewählten Personen sind gesperrt oder müssen zuerst geprüft werden und wurden abgewählt.'
      : 'Bitte mindestens eine Person auswählen, bevor der Lead an Sellify übergeben wird.');
    return;
  }
  const campaignName = String(lead.campaign || '').trim();
  if (includeCampaign && !campaignName) {
    showBusinessAlert('Der Lead hat keine Kampagne. Ohne Kampagnenname kann er nicht als Kampagne übertragen werden.');
    return;
  }
  const workflowId = `thesen-sellify-${crypto.randomUUID()}`;
  const prompt = `Übergebe den validierten Lead ${lead.name} kontrolliert an Sellify. Prüfe Dubletten, schreibe ausschließlich über typisierte SQL-Operationen und bestätige jeden Datensatz durch den synchronisierten Readback.`;
  await patchLead(id, {
    sellify_status: 'queued',
    command_id: workflowId,
    payload: { ...lead.payload, sellify_started_at_ms: Date.now() },
  });
  try {
    const duplicate = await findSellifyCompanyDuplicate(lead);
    let company = duplicate
      ? await updateSellifyCompany(duplicate, lead, workflowId, prompt)
      : await createSellifyCompany(lead, workflowId, prompt);
    // create+update im selben Lauf: die Version nach create kommt aus returned_rows
    // und wird hier an den zweiten Schreibvorgang weitergereicht. Ohne das
    // kollidiert die Uebergabe mit sich selbst.
    if (!duplicate) {
      company = await updateSellifyCompany(company, lead, workflowId, prompt, {
        expectedSourceVersion: Number(company.authoritative_source_version) || 0,
      });
    }
    const personIds = [];
    for (const contact of selectedContacts) {
      const person = await upsertSellifyPerson(company, contact, workflowId, prompt);
      if (person?.person_id) personIds.push(person.person_id);
    }
    // Sellify models a campaign per contact and person, so each selected
    // recipient joins the campaign in its own typed write. Only after the
    // organisation and its people exist — a campaign row pointing at a person
    // Sellify does not know is rejected by the source itself.
    const campaignIds = [];
    if (includeCampaign) {
      for (const personId of personIds) {
        const campaignId = await addSellifyCampaignMember(
          company, personId, campaignName, workflowId, prompt,
        );
        if (campaignId) campaignIds.push(campaignId);
      }
      if (!campaignIds.length) throw new Error('Sellify hat keine Kampagnen-Zuordnung bestätigt.');
    }
    await patchLead(id, {
      sellify_status: 'completed',
      command_id: workflowId,
      payload: {
        ...lead.payload,
        sellify_finished_at_ms: Date.now(),
        sellify_company_id: company.id,
        sellify_contact_id: company.contact_id,
        sellify_person_ids: personIds,
        sellify_campaign_ids: campaignIds,
        sellify_campaign_name: includeCampaign ? campaignName : '',
        sellify_deduplication: duplicate ? 'updated_existing' : 'created_new',
      },
    });
  } catch (error) {
    // Eine ausstehende Ruecksynchronisierung ist kein Fehlschlag: Sellify hat
    // geschrieben, nur die lokale Kopie hinkt hinterher. Diesen Zustand als
    // "gescheitert" zu melden hat den Nutzer wiederholt in die Irre gefuehrt.
    const pending = error?.readbackPending === true;
    await patchLead(id, {
      sellify_status: pending ? 'pending_readback' : 'failed',
      command_id: workflowId,
      payload: {
        ...lead.payload,
        [pending ? 'sellify_pending_reason' : 'sellify_error']: String(error?.message || error),
        sellify_finished_at_ms: Date.now(),
      },
    });
    showBusinessAlert(pending
      ? `${String(error?.message || error)} Der Lead wird als „Übergeben, Bestätigung ausstehend“ geführt.`
      : String(error?.message || error));
  }
}

function seriesEmailHandoffForLead(lead, decisions = null) {
  const plan = buildCampaignRecipientList(lead, decisions);
  const recipients = [...new Set(plan.recipients
    .map((contact) => String(contact.email || contact.person_email || '').trim().toLowerCase())
    .filter((address) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(address)))];
  const params = new URLSearchParams({
    action: 'series-email',
    source_module: 'sellify',
    recipients: recipients.join(','),
    subject: String(lead?.campaign || state.selectedCampaign || '').trim(),
  });
  return { recipients, hash: `#mail?${params.toString()}`, excluded: plan.excluded };
}

async function openSeriesEmailFromLead(id) {
  const lead = state.leads.find((entry) => entry.id === id);
  if (!lead || lead.validation_status !== 'validated') {
    showBusinessAlert('Bitte den Lead vor der Serien-E-Mail validieren.');
    return;
  }
  const decisions = await refreshLeadRecipientEligibility(lead, { force: true });
  const handoff = seriesEmailHandoffForLead(lead, decisions);
  if (!handoff.recipients.length) {
    showBusinessAlert(handoff.excluded.length
      ? 'Die ausgewählten Personen sind gesperrt oder müssen zuerst geprüft werden.'
      : 'Bitte mindestens eine Person mit gültiger E-Mail-Adresse auswählen.');
    return;
  }
  location.hash = handoff.hash;
  await state.ctx?.openApp?.('mail');
}

async function findSellifyCompanyDuplicate(lead) {
  const collection = sellifyReadCollection('sellify_companies');
  const matches = await collection.find({
    selector: { name: { $eq: String(lead.name || '').trim() } },
  }).exec();
  const active = matches.filter((entry) => !entry.is_deleted);
  if (active.length <= 1) return active[0] || null;
  const domain = normalizedDomain(lead.website || lead.domain);
  const postalCode = String(lead.data?.postal_code || lead.data?.plz || '').trim();
  const city = String(lead.city || lead.data?.city || lead.data?.ort || '').trim().toLowerCase();
  const strong = active.filter((entry) => {
    const sameDomain = domain && normalizedDomain(entry.website_url) === domain;
    const sameAddress = postalCode && city
      && String(entry.postal_code || '').trim() === postalCode
      && String(entry.city || '').trim().toLowerCase() === city;
    return sameDomain || sameAddress;
  });
  if (strong.length === 1) return strong[0];
  throw new Error('Die Sellify-Dublettenprüfung ist nicht eindeutig. Bitte den Lead und die vorhandenen Organisationen prüfen.');
}

// Das CRM ist die erste Quelle, nicht die letzte Ablage.
//
// Bis zum 11.08.2026 wurde Sellify vor einer Recherche nur gefragt, OB die Firma
// existiert — die dort gefuehrten Ansprechpartner (im Schnitt 3,5 je Firma, in
// diesem Mandanten 60.639 Personen zu 17.516 Firmen) blieben unbeachtet. Der
// Auftrag ging dann los und liess dieselben Namen, Anschriften und Telefonnummern
// im offenen Netz neu zusammensuchen, die zwei Handgriffe entfernt schon
// vorlagen. Das ist nicht nur Verschwendung: extern Erratenes ist schlechter
// belegt als ein gepflegter CRM-Eintrag.
//
// Was hier eingesammelt wird, geht als Vorwissen in den Auftrag. Es ersetzt die
// externe Pruefung nicht — der Zwei-Quellen-Nachweis bleibt unangetastet —, aber
// die Recherche weiss ab jetzt, was das Haus bereits kennt.
async function sellifyVorwissen(lead) {
  const firma = await findSellifyCompanyDuplicate(lead);
  if (!firma) return null;
  let personen = [];
  try {
    const gefunden = await sellifyReadCollection('sellify_people')
      .find({ selector: { contact_id: { $eq: firma.contact_id } } })
      .exec();
    personen = gefunden
      .filter((entry) => !entry.is_deleted)
      .map((entry) => ({
        vorname: String(entry.first_name || '').trim(),
        nachname: String(entry.last_name || '').trim(),
        funktion: String(entry.position || entry.function || '').trim(),
        email: String(entry.email || '').trim(),
        telefon: String(entry.phone || entry.telephone || '').trim(),
      }))
      .filter((p) => p.vorname || p.nachname);
  } catch (error) {
    // Eine Firma ohne lesbare Kontakte ist immer noch Vorwissen.
    console.warn('[thesen-outbound] Sellify-Kontakte nicht lesbar', error);
  }
  return {
    contact_id: firma.contact_id,
    name: String(firma.name || '').trim(),
    anschrift: String(firma.street || firma.address || '').trim(),
    plz: String(firma.postal_code || '').trim(),
    ort: String(firma.city || '').trim(),
    domain: String(firma.website_url || '').trim(),
    telefon: String(firma.phone || '').trim(),
    personen,
  };
}

function sellifyVorwissenAlsText(vorwissen) {
  if (!vorwissen) return '';
  const zeilen = [
    'BEKANNT AUS DEM EIGENEN CRM (Sellify) — diese Angaben sind gepflegt und haben Vorrang',
    'vor extern Gefundenem. Pruefe sie, widerlege sie wenn noetig, aber erfinde sie nicht neu:',
    `- Organisation: ${vorwissen.name}${vorwissen.contact_id ? ` [contact_id ${vorwissen.contact_id}]` : ''}`,
  ];
  if (vorwissen.anschrift || vorwissen.plz || vorwissen.ort) {
    zeilen.push(`- Anschrift: ${[vorwissen.anschrift, [vorwissen.plz, vorwissen.ort].filter(Boolean).join(' ')].filter(Boolean).join(', ')}`);
  }
  if (vorwissen.domain) zeilen.push(`- Domain: ${vorwissen.domain}`);
  if (vorwissen.telefon) zeilen.push(`- Telefon: ${vorwissen.telefon}`);
  if (vorwissen.personen.length) {
    zeilen.push(`- Bereits gefuehrte Ansprechpartner (${vorwissen.personen.length}):`);
    for (const p of vorwissen.personen.slice(0, 25)) {
      const teile = [[p.vorname, p.nachname].filter(Boolean).join(' '), p.funktion, p.email, p.telefon].filter(Boolean);
      zeilen.push(`  · ${teile.join(' | ')}`);
    }
    zeilen.push('  Diese Personen NICHT erneut erraten. Ergaenze fehlende Angaben und suche zusaetzliche Ansprechpartner.');
  } else {
    zeilen.push('- Im CRM sind zu dieser Organisation noch keine Ansprechpartner gefuehrt.');
  }
  return zeilen.join('\n');
}

async function createSellifyCompany(lead, workflowId, prompt) {
  const values = sellifyCompanyValues(lead);
  const result = await dispatchExternalSqlWrite('company_create', lead.id, values, workflowId, prompt);
  const contactId = returnedInteger(result, 'contact_id');
  if (!contactId) throw new Error('Sellify hat keine Organisations-ID bestätigt.');
  // Die Quellversion nach dem Create kommt aus der massgeblichen SQL-Antwort
  // (returned_rows.source_version), nicht aus der nachhinkenden Browserprojektion.
  // So kann der anschliessende company_update mit der echten Version starten.
  const authoritativeSourceVersion = returnedSourceVersion(result);
  const projected = await waitForProjectedRecord(
    sellifyReadCollection('sellify_companies'),
    `sellify-company-${contactId}`,
    (entry) => entry.contact_id === contactId && entry.name === values.name,
  );
  return withAuthoritativeSourceVersion(
    projected,
    authoritativeSourceVersion || Number(projected.updated_at_ms) || 0,
  );
}

async function updateSellifyCompany(company, lead, workflowId, prompt, options = {}) {
  const values = sellifyCompanyValues(lead);
  const mutableFields = ['name', 'short_name', 'number1', 'number2', 'email', 'phone', 'fax', 'website_url', 'address_line', 'postal_code', 'city'];
  const changedFields = mutableFields.filter((field) => field === 'name' || values[field] !== '');
  const expectedSourceVersion = await resolveExpectedSourceVersion({
    kind: 'company',
    entityId: company.contact_id,
    hintedVersion: options.expectedSourceVersion,
    record: company,
    workflowId,
    prompt,
  });
  const patch = {
    contact_id: company.contact_id,
    expected_source_version: expectedSourceVersion,
    changed_fields: changedFields,
    ...values,
    ...sellifyCommunicationIds(company),
  };
  delete patch.country_code;
  const result = await dispatchExternalSqlWriteWithSourceVersion(
    'company_update',
    company.id,
    patch,
    workflowId,
    prompt,
    { kind: 'company', entityId: company.contact_id },
  );
  const nextVersion = returnedSourceVersion(result);
  const employees = finiteNumber(firstValue(lead.data, ['employees', 'mitarbeiter']));
  const revenueMio = finiteNumber(firstValue(lead.data, ['revenue_mio', 'umsatz_mio']));
  if (employees || revenueMio) {
    await dispatchExternalSqlWrite('company_metrics_update', company.id, {
      contact_id: company.contact_id,
      cust_contact_id: Number(company.payload?.sql?.cust_contact_id) || undefined,
      employees,
      revenue_mio: revenueMio,
    }, workflowId, prompt);
  }
  const wzCode = String(firstValue(lead.data, ['wz_code', 'wzcode']) || '').trim();
  if (wzCode) {
    await dispatchExternalSqlWrite('company_wz_update', company.id, {
      contact_id: company.contact_id,
      contact_wzcode_id: Number(company.payload?.sql?.contact_wzcode_id) || undefined,
      wz_code: wzCode,
    }, workflowId, prompt);
  }
  const projected = await waitForProjectedRecord(
    sellifyReadCollection('sellify_companies'),
    company.id,
    (entry) => entry.contact_id === company.contact_id && entry.name === values.name,
  );
  return withAuthoritativeSourceVersion(
    projected,
    nextVersion || Number(projected.updated_at_ms) || expectedSourceVersion,
  );
}

// Adds one selected recipient to the Sellify campaign named after the lead's
// campaign. `campaign_create` is the source's own typed operation (it creates
// the selection and its member in one transaction and is idempotent through the
// write receipt), so no SQL is written from here directly.
async function addSellifyCampaignMember(company, personId, campaignName, workflowId, prompt) {
  const result = await dispatchExternalSqlWrite('campaign_create', `${company.id}:${personId}`, {
    contact_id: company.contact_id,
    person_id: personId,
    name: campaignName,
    note_text: '',
    // Sellify classifies a selection by the table it targets; 0 keeps the
    // source's own default instead of inventing a classification here.
    target_table_number: 0,
    search_category_id: 0,
    group_id: 0,
  }, workflowId, prompt);
  return returnedInteger(result, 'selection_id') || returnedInteger(result, 'selectionmember_id') || 0;
}

async function upsertSellifyPerson(company, contact, workflowId, prompt) {
  const values = sellifyPersonValues(contact);
  if (!values.first_name && !values.last_name) return null;
  const existing = await findSellifyPersonDuplicate(company.contact_id, values);
  if (!existing) {
    const result = await dispatchExternalSqlWrite('person_create', company.id, {
      contact_id: company.contact_id,
      ...values,
    }, workflowId, prompt);
    const personId = returnedInteger(result, 'person_id');
    if (!personId) throw new Error(`Sellify hat für ${[values.first_name, values.last_name].filter(Boolean).join(' ')} keine Personen-ID bestätigt.`);
    const authoritativeSourceVersion = returnedSourceVersion(result);
    const created = await waitForProjectedRecord(
      sellifyReadCollection('sellify_people'),
      `sellify-person-${personId}`,
      (entry) => entry.person_id === personId && entry.contact_id === company.contact_id,
    );
    return updateSellifyPersonCommunication(
      withAuthoritativeSourceVersion(
        created,
        authoritativeSourceVersion || Number(created.updated_at_ms) || 0,
      ),
      values,
      workflowId,
      prompt,
      { expectedSourceVersion: authoritativeSourceVersion },
    );
  }
  return updateSellifyPersonCommunication(existing, values, workflowId, prompt);
}

async function findSellifyPersonDuplicate(contactId, values) {
  const matches = values.email
    ? await sellifyReadCollection('sellify_people').find({ selector: { email: { $eq: values.email } } }).exec()
    : await sellifyReadCollection('sellify_people').find({ selector: { contact_id: { $eq: contactId } } }).exec();
  return matches.find((entry) => !entry.is_deleted
    && entry.contact_id === contactId
    && (values.email
      ? String(entry.email || '').trim().toLowerCase() === values.email.toLowerCase()
      : String(entry.first_name || '').trim().toLowerCase() === values.first_name.toLowerCase()
        && String(entry.last_name || '').trim().toLowerCase() === values.last_name.toLowerCase())) || null;
}

async function updateSellifyPersonCommunication(person, values, workflowId, prompt, options = {}) {
  const changedFields = ['salutation', 'title', 'first_name', 'last_name', 'department', 'function', 'number', 'email', 'phone', 'mobile', 'social_media']
    .filter((field) => values[field] !== '');
  if (!changedFields.length) return person;
  const expectedSourceVersion = await resolveExpectedSourceVersion({
    kind: 'person',
    entityId: person.person_id,
    hintedVersion: options.expectedSourceVersion,
    record: person,
    workflowId,
    prompt,
  });
  const result = await dispatchExternalSqlWriteWithSourceVersion(
    'person_update',
    person.id,
    {
      person_id: person.person_id,
      contact_id: person.contact_id,
      expected_source_version: expectedSourceVersion,
      changed_fields: changedFields,
      ...values,
      ...sellifyCommunicationIds(person),
    },
    workflowId,
    prompt,
    { kind: 'person', entityId: person.person_id },
  );
  const nextVersion = returnedSourceVersion(result);
  const projected = await waitForProjectedRecord(
    sellifyReadCollection('sellify_people'),
    person.id,
    (entry) => entry.person_id === person.person_id
      && (!values.email || String(entry.email || '').trim().toLowerCase() === values.email.toLowerCase()),
  );
  return withAuthoritativeSourceVersion(
    projected,
    nextVersion || Number(projected.updated_at_ms) || expectedSourceVersion,
  );
}

function isSourceVersionConflict(error) {
  return /source version changed/i.test(String(error?.message || error || ''));
}

// Die erwartete Quellversion darf NICHT aus der nachhinkenden Browserprojektion
// (sellify_companies / sellify_people) kommen. Massgeblich ist die SQL-Quelle:
// 1. vom vorherigen Schreibvorgang in returned_rows weitergereicht,
// 2. optional per company_source_version / person_source_version gelesen,
// 3. erst dann die Projektion als Notbehelf.
//
// Bei einem Versionskonflikt wird die Version EINMAL aus der massgeblichen
// Quelle gelesen und der Schreibvorgang mit dieser Version wiederholt. Scheitert
// auch das, ist es eine echte Fremdaenderung und wird als solche gemeldet.
// Die Sperre bleibt; blinde Wiederholungen und Projektions-Warte-Reparaturen
// sind absichtlich entfernt.
function withAuthoritativeSourceVersion(record, version) {
  const authoritative = Number(version) || 0;
  if (!record || !authoritative) return record;
  return { ...record, authoritative_source_version: authoritative };
}

async function resolveExpectedSourceVersion({
  kind, entityId, hintedVersion, record, workflowId, prompt,
}) {
  const hinted = Number(hintedVersion) || 0;
  if (hinted > 0) return hinted;
  const carried = Number(record?.authoritative_source_version) || 0;
  if (carried > 0) return carried;
  const authoritative = await fetchAuthoritativeSourceVersion(kind, entityId, workflowId, prompt);
  if (authoritative > 0) return authoritative;
  return Number(record?.updated_at_ms) || 0;
}

async function fetchAuthoritativeSourceVersion(kind, entityId, workflowId, prompt) {
  const id = Number(entityId) || 0;
  if (!id) return 0;
  const operationId = kind === 'person' ? 'person_source_version' : 'company_source_version';
  const idField = kind === 'person' ? 'person_id' : 'contact_id';
  try {
    const result = await dispatchExternalSqlWrite(
      operationId,
      String(id),
      { [idField]: id },
      workflowId,
      prompt,
    );
    return returnedSourceVersion(result);
  } catch {
    // Operation noch nicht auf dem Mandanten registriert, oder Quelle nicht
    // erreichbar: kein harter Abbruch, der Aufrufer greift auf den Hinweis oder
    // die Projektion zurueck.
    return 0;
  }
}

async function dispatchExternalSqlWriteWithSourceVersion(
  operationId, recordId, values, workflowId, prompt, entity,
) {
  try {
    return await dispatchExternalSqlWrite(operationId, recordId, values, workflowId, prompt);
  } catch (error) {
    if (!isSourceVersionConflict(error)) throw error;
    const actual = await fetchAuthoritativeSourceVersion(
      entity.kind, entity.entityId, workflowId, prompt,
    );
    if (!actual || actual === (Number(values.expected_source_version) || 0)) {
      throw new Error(
        'Sellify hat den Schreibvorgang wegen einer Versionspruefung abgelehnt. '
        + 'Die erwartete Quellversion weicht vom aktuellen Stand in Sellify ab '
        + '(echte Fremdaenderung oder Quellversion nicht massgeblich lesbar).',
      );
    }
    try {
      return await dispatchExternalSqlWrite(
        operationId,
        recordId,
        { ...values, expected_source_version: actual },
        workflowId,
        prompt,
      );
    } catch (second) {
      if (!isSourceVersionConflict(second)) throw second;
      throw new Error(
        'Sellify hat den Schreibvorgang wegen einer Versionspruefung abgelehnt. '
        + 'Die erwartete Quellversion weicht vom aktuellen Stand in Sellify ab '
        + '(echte Fremdaenderung).',
      );
    }
  }
}

async function dispatchExternalSqlWrite(operationId, recordId, values, workflowId, prompt) {
  const commandId = `cmd_${workflowId}_${operationId}_${crypto.randomUUID()}`;
  const result = await state.ctx.commandBus.dispatch({
    id: commandId,
    command_id: commandId,
    module: 'thesen-outbound',
    type: 'external_sql.write',
    command_type: 'external_sql.write',
    record_id: recordId,
    inbound_channel: 'business_os.thesen_outbound',
    payload: {
      source_id: 'primary-crm',
      operation_id: operationId,
      ...values,
      title: `Sellify-Übergabe: ${selectedLead()?.name || recordId}`,
      prompt,
      user_message: prompt,
      response_channel: 'business_os_chat',
      outbound_channel: 'business_os_chat',
      thread_key: `business-os/thesen-outbound/${workflowId}`,
    },
    client_context: {
      source_module: 'thesen-outbound',
      record_id: recordId,
      workflow_id: workflowId,
      response_channel: 'business_os_chat',
      writeback_required: true,
    },
  }, { until: 'terminal', timeoutMs: 90_000 });
  if (result?.status !== 'completed') {
    throw new Error(result?.error_message || result?.result?.error || `Sellify-Operation ${operationId} ist fehlgeschlagen.`);
  }
  return result;
}

// Der Schreibvorgang nach Sellify ist zu diesem Zeitpunkt bereits bestaetigt
// (`external_sql.write` -> completed). Was hier noch aussteht, ist allein die
// Rueckreplikation in die Browserkopie von `sellify_companies` — einer
// Collection mit ueber 17000 Zeilen. Ein Zeitueberlauf hier bedeutet also
// "noch nicht bestaetigt", nicht "fehlgeschlagen". Frueher wurde daraus ein
// harter Fehler, und der Nutzer sah "gescheitert" fuer einen Vorgang, der in
// Sellify erfolgreich war.
class SellifyReadbackPending extends Error {
  constructor(message) {
    super(message);
    this.name = 'SellifyReadbackPending';
    this.readbackPending = true;
  }
}

async function waitForProjectedRecord(collection, id, predicate, options = {}) {
  const deadline = Date.now() + Math.max(5_000, Number(options.timeoutMs) || 25_000);
  while (Date.now() < deadline) {
    const record = await collection.findOne(id).exec();
    if (record && predicate(record)) return record;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new SellifyReadbackPending(
    'Sellify hat den Schreibvorgang bestaetigt. Die Ruecksynchronisierung in die lokale Kopie steht noch aus.',
  );
}

function sellifyCompanyValues(lead) {
  return {
    name: String(lead.name || '').trim(),
    short_name: String(firstValue(lead.data, ['short_name', 'kurzname']) || '').trim(),
    number1: String(firstValue(lead.data, ['number1', 'nummer']) || '').trim(),
    number2: String(firstValue(lead.data, ['number2', 'nummer2']) || '').trim(),
    email: String(firstValue(lead.data, ['email', 'company_email', 'e_mail']) || '').trim(),
    phone: String(firstValue(lead.data, ['phone', 'company_phone', 'telefon']) || '').trim(),
    fax: String(firstValue(lead.data, ['fax']) || '').trim(),
    website_url: String(lead.website || firstValue(lead.data, ['website_url', 'website', 'internet']) || '').trim(),
    address_line: String(firstValue(lead.data, ['address_line', 'address', 'street', 'strasse']) || '').trim(),
    postal_code: String(firstValue(lead.data, ['postal_code', 'postcode', 'plz']) || '').trim(),
    city: String(lead.city || firstValue(lead.data, ['city', 'ort']) || '').trim(),
    country_code: normalizedResearchCountry(lead.country),
  };
}

function sellifyPersonValues(contact) {
  const names = String(contact.name || '').trim().split(/\s+/);
  return {
    salutation: String(contact.salutation || contact.person_anrede || '').trim(),
    title: String(contact.title || contact.person_titel || '').trim(),
    first_name: String(contact.first_name || contact.person_vorname || names.shift() || '').trim(),
    last_name: String(contact.last_name || contact.person_nachname || names.join(' ') || '').trim(),
    department: String(contact.department || contact.person_abteilung || '').trim(),
    function: String(contact.role || contact.function || contact.person_funktion || contact.position || '').trim(),
    number: String(contact.number || '').trim(),
    email: String(contact.email || contact.person_email || '').trim(),
    phone: String(contact.phone || contact.person_telefon || '').trim(),
    mobile: String(contact.mobile || contact.person_mobil || '').trim(),
    social_media: String(contact.social_media || contact.linkedin || contact.xing || '').trim(),
  };
}

function sellifyCommunicationIds(record) {
  const sql = record?.payload?.sql || {};
  return {
    email_id: Number(sql.primary_email_id || sql.email_id) || undefined,
    phone_id: Number(sql.primary_phone_id || sql.phone_id) || undefined,
    fax_id: Number(sql.primary_fax_id || sql.fax_id) || undefined,
    mobile_id: Number(sql.primary_mobile_id || sql.mobile_id) || undefined,
    url_id: Number(sql.primary_url_id || sql.url_id) || undefined,
    address_id: Number(sql.primary_address_id || sql.address_id) || undefined,
  };
}

function returnedInteger(result, field) {
  const values = [
    result?.result?.returned_rows?.[0]?.[field],
    result?.returned_rows?.[0]?.[field],
    result?.result?.[field],
    result?.[field],
  ];
  for (const value of values) {
    const number = Number(value);
    if (Number.isInteger(number) && number > 0) return number;
  }
  return 0;
}

// source_version kommt als BIGINT (ms seit Epoch) aus der SQL-Antwort. Im
// Gegensatz zu IDs kann 0 formal gueltig sein; massgeblich ist hier > 0.
function returnedSourceVersion(result) {
  const values = [
    result?.result?.returned_rows?.[0]?.source_version,
    result?.returned_rows?.[0]?.source_version,
    result?.result?.source_version,
    result?.source_version,
  ];
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return number;
  }
  return 0;
}

function firstValue(record, keys) {
  for (const key of keys) {
    const value = record?.[key];
    if (value !== undefined && value !== null && String(value).trim() !== '') return value;
  }
  return '';
}

function finiteNumber(value) {
  const number = Number(String(value ?? '').replace(',', '.'));
  return Number.isFinite(number) ? number : 0;
}

function normalizedDomain(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return '';
  try { return new URL(raw.includes('://') ? raw : `https://${raw}`).hostname.replace(/^www\./, ''); } catch { return raw.replace(/^www\./, '').split('/')[0]; }
}

function sellifyReadCollection(name) {
  const cached = name === 'sellify_companies' ? state.sellifyCompanies : state.sellifyPeople;
  const collection = cached || state.ctx?.db?.collection?.(name) || null;
  if (!collection) {
    throw new Error('Die synchronisierte Sellify-Projektion ist noch nicht verfügbar. Bitte Sellify einmal öffnen und die Übergabe erneut starten.');
  }
  if (name === 'sellify_companies') state.sellifyCompanies = collection;
  else state.sellifyPeople = collection;
  return collection;
}

function sellifyWritebackContract() {
  return {
    system: 'sellify-sqlite-sync', source_of_truth: 'sellify-original-sql', direct_sql_write_allowed: false, deduplicate_before_create: true,
    allowed_commands: ['external_sql.write'],
    allowed_operations: [
      'company_create', 'company_update', 'company_metrics_update', 'company_wz_update',
      'company_source_version',
      'person_create', 'person_update', 'person_source_version',
      'campaign_create',
    ],
    required_result: ['contact_id', 'person_ids', 'campaign_ids', 'command_ids', 'sync_status', 'deduplication_decision'],
  };
}

function enabledSourcePolicy() {
  return {
    skill: 'thesen-outbound-research', min_independent_sources: 2, verification_scope: 'per_field', validation_only_sources: ['experte.de'],
    sources: state.sources.filter((item) => item.enabled).map((item) => ({ id: item.id, url: item.url, field_keys: item.field_keys, target_key: item.target_key, credential_secret_name: item.credential_secret_name, secret_value_in_payload: false })),
  };
}

function normalizedResearchCountry(value) {
  const country = String(value || 'DE').trim().toUpperCase();
  if (['AT', 'AUT', 'AUSTRIA', 'ÖSTERREICH', 'OESTERREICH'].includes(country)) return 'AT';
  if (['CH', 'CHE', 'SCHWEIZ', 'SWITZERLAND'].includes(country)) return 'CH';
  return 'DE';
}

function researchOutcomeWriteback(lead, commandResult) {
  const outcome = commandResult?.result && typeof commandResult.result === 'object'
    ? commandResult.result
    : commandResult;
  const fields = outcome?.fields && typeof outcome.fields === 'object' ? outcome.fields : {};
  const data = { ...(lead.data || {}) };
  const contact = { ...(lead.contacts?.[0] || {}) };
  const evidence = [];
  const researchedFieldKeys = [];
  for (const [fieldKey, field] of Object.entries(fields)) {
    if (field?.value === null || field?.value === undefined || String(field.value).trim() === '') continue;
    researchedFieldKeys.push(fieldKey);
    if (fieldKey.startsWith('person_')) contact[fieldKey] = field.value;
    else data[fieldKey] = field.value;
    for (const candidate of field.candidates || []) {
      if (!candidate?.source_id && !candidate?.source_url) continue;
      evidence.push({
        field_key: fieldKey,
        value: candidate.value,
        confidence: candidate.confidence || '',
        source_id: candidate.source_id || '',
        source_url: candidate.source_url || '',
        tier: candidate.tier || '',
        via: candidate.via || '',
        label: candidate.source_id || candidate.source_url || 'Quelle',
      });
    }
  }
  const mergedEvidence = deduplicateEvidence([...(lead.evidence || []), ...evidence]);
  const normalizedContact = normalizeResearchedContact(contact);
  const contacts = withStableContactIds(
    lead.id || 'lead',
    normalizedContact ? [normalizedContact, ...(lead.contacts || []).slice(1)] : (lead.contacts || []),
  );
  const contactIds = new Set(contacts.map((entry) => entry.id));
  const selectedContactIds = (lead.selected_contact_ids || []).filter((id) => contactIds.has(id));
  const draft = {
    ...lead,
    data,
    contacts,
    selected_contact_ids: selectedContactIds,
    evidence: mergedEvidence,
    payload: {
      ...lead.payload,
      researched_field_keys: researchedFieldKeys,
      research_finished_at_ms: Date.now(),
      research_tool: outcome?.tool || '',
      browser_assist_tasks: outcome?.browser_assist_tasks || [],
    },
  };
  const unverifiedFieldKeys = researchedFieldKeys.filter((fieldKey) => independentFieldEvidenceCount(draft, fieldKey) < 2);
  return {
    data: draft.data,
    contacts: draft.contacts,
    selected_contact_ids: draft.selected_contact_ids,
    evidence: draft.evidence,
    research_status: researchedFieldKeys.length > 0 && unverifiedFieldKeys.length === 0 ? 'completed' : 'needs_review',
    payload: {
      ...draft.payload,
      verified_field_keys: researchedFieldKeys.filter((fieldKey) => !unverifiedFieldKeys.includes(fieldKey)),
      unverified_field_keys: unverifiedFieldKeys,
    },
  };
}

function researchCommandLeadPatch(lead, command) {
  if (command?.command_type !== 'web_stack.person_research') return null;
  const status = normalizedResearchCommandStatus(command);
  const commandId = String(command.command_id || command.id || lead.command_id || '').trim();
  if (['accepted', 'queued', 'running', 'leased', 'retry_wait', 'working'].includes(status)) {
    return {
      research_status: status === 'queued' ? 'queued' : 'running',
      command_id: commandId,
      task_id: String(command.task_id || lead.task_id || '').trim(),
      payload: {
        ...lead.payload,
        last_research_command_id: commandId,
        research_started_at_ms: Number(lead.payload?.research_started_at_ms || Date.now()),
      },
    };
  }
  if (['failed', 'blocked', 'cancelled', 'canceled', 'error'].includes(status)) {
    return {
      research_status: 'failed',
      payload: {
        ...lead.payload,
        reconciled_research_command_id: commandId,
        research_finished_at_ms: Date.now(),
        research_error: String(command.error_message || command.error || command.result?.error || 'Recherche fehlgeschlagen.'),
      },
    };
  }
  if (!['completed', 'handled', 'success', 'done', 'passed'].includes(status)) return null;
  if (!command.result || command.result.ok === false) return null;
  const writeback = researchOutcomeWriteback(lead, command);
  return {
    ...writeback,
    command_id: commandId,
    task_id: '',
    payload: {
      ...writeback.payload,
      reconciled_research_command_id: commandId,
      last_research_command_id: commandId,
      research_mode: String(command.result?.mode || ''),
    },
  };
}

function normalizedResearchCommandStatus(command) {
  const terminal = String(command?.terminal_status || '').trim().toLowerCase();
  if (['completed', 'handled', 'success', 'done', 'passed', 'failed', 'blocked', 'cancelled', 'canceled', 'error'].includes(terminal)) {
    return terminal;
  }
  return String(
    command?.status
    || command?.execution_phase
    || command?.result?.status
    || command?.task_status
    || '',
  ).trim().toLowerCase();
}

function normalizeResearchedContact(contact) {
  const firstName = String(contact.person_vorname || '').trim();
  const lastName = String(contact.person_nachname || '').trim();
  const name = [firstName, lastName].filter(Boolean).join(' ');
  if (!name && !contact.person_email && !contact.person_telefon && !contact.person_position) return null;
  return {
    ...contact,
    name: name || contact.name || '',
    role: contact.person_funktion || contact.person_position || contact.role || '',
    position: contact.person_position || contact.position || '',
    email: contact.person_email || contact.email || '',
    phone: contact.person_telefon || contact.phone || '',
  };
}

function deduplicateEvidence(entries) {
  const seen = new Set();
  return entries.filter((entry) => {
    const key = [
      entry?.field_key || entry?.field || '',
      entry?.source_id || '',
      entry?.source_url || entry?.url || '',
      String(entry?.value ?? ''),
    ].join('|').toLowerCase();
    if (!key.replace(/\|/g, '') || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

async function patchLead(id, patch) {
  const doc = await state.collections.leads.findOne(id).exec();
  if (!doc) return;
  const current = doc.toJSON?.() || doc;
  const normalized = normalizeLeadRecipientShape({ ...current, ...patch, id });
  await doc.incrementalPatch({
    ...patch,
    contacts: normalized.contacts,
    selected_contact_ids: normalized.selected_contact_ids,
    updated_at_ms: Date.now(),
  });
}

function selectedLead() { return state.leads.find((lead) => lead.id === state.selectedLeadId) || null; }
function adapterReady(value) { return ['adapter_ready', 'test_ok'].includes(value?.status || value?.adapter_status) && ['registered', 'test_executed'].includes(value?.scrape_status); }
function evidenceSourceKey(entry) {
  const rawKey = entry?.source_id || entry?.source_url || entry?.url || '';
  if (!rawKey) return '';
  let key = String(rawKey).trim().toLowerCase();
  try { key = new URL(key).hostname.replace(/^www\./, ''); } catch { /* Source IDs are valid evidence keys too. */ }
  return key;
}
function independentEvidenceCount(lead) {
  const sourceKeys = new Set();
  for (const entry of lead?.evidence || []) {
    const key = evidenceSourceKey(entry);
    if (key) sourceKeys.add(key);
  }
  return sourceKeys.size;
}
function sourceEvidenceGroups(lead) {
  const groups = new Map();
  for (const entry of lead?.evidence || []) {
    const rawUrl = String(entry?.source_url || entry?.url || '').trim();
    const rawId = String(entry?.source_id || '').trim();
    let host = '';
    try { host = new URL(rawUrl).hostname.replace(/^www\./, '').toLowerCase(); } catch { /* A source ID can be used without a URL. */ }
    const key = (rawId || host || rawUrl).toLowerCase();
    if (!key) continue;
    const current = groups.get(key) || {
      key,
      label: entry?.label || rawId || host || rawUrl,
      url: rawUrl,
      fields: new Set(),
    };
    if (!current.url && rawUrl) current.url = rawUrl;
    const fieldKey = entry?.field_key || entry?.field;
    if (fieldKey) current.fields.add(String(fieldKey));
    groups.set(key, current);
  }
  return [...groups.values()]
    .map((group) => ({ ...group, fieldCount: group.fields.size || 1 }))
    .sort((left, right) => left.label.localeCompare(right.label, 'de'));
}
function independentFieldEvidenceCount(lead, fieldKey) {
  return independentEvidenceCount({
    evidence: (lead?.evidence || []).filter((entry) => (entry?.field_key || entry?.field) === fieldKey),
  });
}
function leadReadyForValidation(lead) {
  // Owner-Entscheid 05.08.2026: solange Adapter fehlen, findet die Recherche
  // viele der 21 Felder nicht. Ein Lead ist freigebbar, sobald ein Firmenname
  // bekannt ist und keine Recherche mehr laeuft — alles Weitere ist Qualitaet,
  // keine Sperre. Die Feld-Ampeln bleiben sichtbar.
  if (['queued', 'running'].includes(String(lead?.research_status || ''))) return false;
  const name = researchFieldValue(lead, 'firma_name') || String(lead?.name || '').trim();
  return Boolean(name);
}
function researchLabel(lead) {
  if (lead.validation_status === 'validated') return 'Validiert';
  if (lead.research_status === 'queued') return 'Wartet';
  if (lead.research_status === 'running') return 'Läuft';
  if (lead.research_status === 'completed') return 'Geprüft';
  if (lead.research_status === 'needs_review') return 'Prüfung nötig';
  if (lead.research_status === 'failed') return 'Unvollständig';
  return 'Offen';
}
function sellifyLabel(lead) {
  if (lead.sellify_status === 'queued') return 'Übergabe läuft';
  if (lead.sellify_status === 'completed') return 'Übergeben';
  // Sellify hat geschrieben, nur die lokale Kopie steht noch aus.
  if (lead.sellify_status === 'pending_readback') return 'Übergeben · Bestätigung ausstehend';
  if (lead.sellify_status === 'failed') return 'Fehlgeschlagen';
  return '—';
}
function researchFieldValue(lead, fieldKey) {
  const aliases = RESEARCH_FIELD_VALUE_KEYS[fieldKey] || [fieldKey];
  if (fieldKey.startsWith('person_')) {
    return String(firstValue(lead?.contacts?.[0], aliases) || '').trim();
  }
  const value = String(firstValue(lead?.data, aliases) || '').trim();
  if (value) return value;
  if (fieldKey === 'firma_name') return String(lead?.name || '').trim();
  if (fieldKey === 'firma_domain') return String(lead?.domain || lead?.website || '').trim();
  if (fieldKey === 'firma_ort') return String(lead?.city || '').trim();
  return '';
}
function fieldEvidenceSources(lead, fieldKey) {
  const sourcesByKey = new Map();
  for (const entry of lead?.evidence || []) {
    if ((entry?.field_key || entry?.field) !== fieldKey) continue;
    const key = evidenceSourceKey(entry);
    if (!key) continue;
    const checkedAt = String(entry?.evidence_gate?.checked_at || '').trim();
    const current = sourcesByKey.get(key) || {
      key,
      label: String(entry?.label || entry?.source_id || key).trim(),
      url: String(entry?.source_url || entry?.url || '').trim(),
      confidence: String(entry?.confidence || '').trim(),
      note: String(entry?.note || '').trim(),
      checkedAt,
    };
    if (!current.url && entry?.source_url) current.url = String(entry.source_url).trim();
    if (!current.confidence && entry?.confidence) current.confidence = String(entry.confidence).trim();
    if (!current.note && entry?.note) current.note = String(entry.note).trim();
    if (checkedAt && (!current.checkedAt || checkedAt > current.checkedAt)) current.checkedAt = checkedAt;
    sourcesByKey.set(key, current);
  }
  return [...sourcesByKey.values()].sort((left, right) => left.label.localeCompare(right.label, 'de'));
}
function researchFieldReview(lead) {
  const groups = RESEARCH_FIELD_GROUPS.map((group) => ({
    id: group.id,
    label: group.label,
    fields: group.fields.map(([key, label]) => {
      const value = researchFieldValue(lead, key);
      const sources = fieldEvidenceSources(lead, key);
      return {
        key,
        label,
        value,
        filled: value !== '',
        sources,
        independentCount: sources.length,
        sufficient: sources.length >= 2,
      };
    }),
  }));
  const fields = groups.flatMap((group) => group.fields);
  return {
    groups,
    fields,
    total: fields.length,
    filledCount: fields.filter((field) => field.filled).length,
    sufficientCount: fields.filter((field) => field.filled && field.sufficient).length,
    missingValueKeys: fields.filter((field) => !field.filled).map((field) => field.key),
    missingEvidenceKeys: fields.filter((field) => field.filled && !field.sufficient).map((field) => field.key),
    // Traegt der Lead Belege, findet die Feldauswertung aber KEINE einzige Quelle,
    // ist der Datenstand unvollstaendig geladen — dann ist auch die Zusammenfassung
    // keine Messung. Siehe renderReviewFieldRow: am 11.08.2026 meldete die
    // Oberflaeche "0 mit mindestens zwei unabhaengigen Quellen belegt", waehrend
    // zwoelf Belege auf dem Server lagen.
    evidenceUnloaded: (Array.isArray(lead?.evidence) ? lead.evidence.length : 0) > 0
      && fields.every((field) => !field.sources.length),
  };
}
function validationBlockers(lead) {
  const blockers = [];
  if (['queued', 'running'].includes(String(lead?.research_status || ''))) blockers.push('Recherche läuft noch');
  const name = researchFieldValue(lead, 'firma_name') || String(lead?.name || '').trim();
  if (!name) blockers.push('Firmenname fehlt');
  return blockers;
}
function confidenceLabel(value) {
  const key = String(value || '').trim().toLowerCase();
  if (key === 'high') return 'hoch';
  if (key === 'medium') return 'mittel';
  if (key === 'low') return 'niedrig';
  return String(value || '').trim();
}
function formatEvidenceCheckedAt(value) {
  const time = Date.parse(String(value || ''));
  if (!Number.isFinite(time)) return '';
  return new Date(time).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}
// A lead that has never been researched has nothing to complain about yet.
// Red is reserved for "looked and found it wanting" — not for "not looked".
function leadIsUnresearched(lead) {
  if (!lead) return false;
  if (['queued', 'running', 'completed', 'needs_review', 'failed'].includes(lead.research_status)) return false;
  return !(lead.evidence || []).length;
}
function renderReviewFieldRow(field, untouched = false, lead = null) {
  const stateClass = untouched
    ? 'is-untouched'
    : !field.filled ? 'is-missing' : field.sufficient ? 'is-sufficient' : 'is-insufficient';
  // Owner-Vorgabe: Wert, Quellen, fertig. Ein Badge als einziges Signal,
  // Quellen als eine kompakte Zeile ohne Vertrauens-/Meta-Prosa. Die
  // Nutzerfreigabe erscheint als Haekchen-Chip, nicht als Pseudo-Quelle.
  const realSources = field.sources.filter((source) => source.key !== 'operator');
  const approved = field.sources.some((source) => source.key === 'operator');
  // "0 Quellen" ist eine AUSSAGE UEBER DIE WELT und darf nur fallen, wenn wir sie
  // treffen koennen. Am 11.08.2026 hing der Dienst 33 Stunden mit 4,2 GB fest,
  // die Replikation lieferte nichts mehr, und die Feldkarten meldeten fuer ANGUS
  // Chemie durchgaengig "0 Quellen" — waehrend serverseitig zwoelf Belege lagen.
  // Der Nutzer trifft auf solchen Zahlen Entscheidungen: er haette die Firma als
  // unbelegt verworfen. Ein stiller Anzeigefehler dieser Art ist schaedlicher als
  // ein sichtbarer Absturz.
  //
  // Solange der Lead nachweislich Belege traegt, die Feldauswertung aber keine
  // einzige Quelle findet, ist das ein unvollstaendiger Ladezustand und kein
  // Messergebnis. Dann sagt die Karte das auch.
  const belegeAmLead = Array.isArray(lead?.evidence) ? lead.evidence.length : 0;
  const auswertungLeer = belegeAmLead > 0 && !field.sources.length;
  const badge = auswertungLeer
    ? tr('sourcesNotLoaded', 'Quellen nicht geladen')
    : untouched && !field.filled
      ? tr('notResearchedYet', 'noch nicht recherchiert')
      : !field.filled
        ? tr('missing', 'fehlt')
        : `${field.independentCount} ${field.independentCount === 1 ? tr('source', 'Quelle') : tr('sources', 'Quellen')}`;
  const sourceLinks = realSources.map((source) => source.url
    ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.label)}</a>`
    : escapeHtml(source.label));
  if (approved) sourceLinks.push(`<span class="thesen-approved-chip">✓ ${tr('approved', 'freigegeben')}</span>`);
  // Frueher hatte ein GEFUELLTES Feld ueberhaupt keinen Bearbeitungsweg — nur
  // "freigeben". Ein falscher Wert liess sich nicht korrigieren. Jetzt ist der
  // Wert selbst anklickbar, und daneben steht der Weg ausdruecklich.
  const aendern = `<button class="thesen-approve-link" data-action="edit-lead" data-field="${escapeHtml(field.key)}">${tr('changeValue', 'ändern')}</button>`;
  const action = !field.filled
    ? `<button class="thesen-approve-link" data-action="edit-lead" data-id="">${tr('enterValue', 'eintragen')}</button>`
    : (!field.sufficient
      ? `${aendern}<button class="thesen-approve-link" data-action="approve-field" data-field="${escapeHtml(field.key)}">${tr('approveField', 'freigeben')}</button>`
      : aendern);
  return `<li class="thesen-review-field ${stateClass}" data-review-field="${escapeHtml(field.key)}">
    <div class="thesen-review-field-head">
      <span class="thesen-review-label">${escapeHtml(field.label)}</span>
      <button type="button" class="thesen-review-value${field.filled ? '' : ' is-empty'}" data-action="edit-lead" data-field="${escapeHtml(field.key)}" title="${escapeHtml(tr('clickToEdit', 'Klicken zum Bearbeiten'))}">${field.filled ? escapeHtml(field.value) : '—'}</button>
      <span class="thesen-review-badge">${escapeHtml(badge)}</span>
      ${action}
    </div>
    ${sourceLinks.length ? `<div class="thesen-review-sources-line">${sourceLinks.join(' · ')}</div>` : ''}
  </li>`;
}
function renderResearchReviewSummary(review, lead) {
  if (leadIsUnresearched(lead)) {
    // Der Zähler bleibt sichtbar (Wächtertest), aber ohne Alarmzeile: es gibt
    // noch nichts zu beanstanden, weil noch nichts gesucht wurde.
    return `<section class="thesen-detail-section thesen-review-summary">
      <h3>${tr('reviewTitle', 'Rechercheprüfung')}</h3>
      <p>${review.evidenceUnloaded
      ? `<strong>${tr('evidenceUnloadedTitle', 'Belege noch nicht geladen')}</strong> · ${tr('evidenceUnloadedHint', 'Der Lead traegt Belege, die Datenverbindung liefert sie gerade nicht. Bitte neu laden, bevor du entscheidest.')}`
      : `<strong>${review.filledCount} ${tr('ofFields', 'von')} ${review.total} ${tr('fieldsFilled', 'Feldern ermittelt')}</strong> · ${review.sufficientCount} ${tr('fieldsSufficient', 'mit mindestens zwei unabhängigen Quellen belegt')}`}</p>
      <p class="thesen-muted">${tr('researchNotStarted', 'Die Recherche wurde für diesen Lead noch nicht gestartet.')}</p>
      <p class="thesen-muted">${tr('status', 'Status')}: ${escapeHtml(researchLabel(lead))} · ${tr('campaign', 'Kampagne')}: ${escapeHtml(lead?.campaign || tr('annualResearch', 'Jahresrecherche'))}</p>
    </section>`;
  }
  const missingParts = [];
  if (review.missingValueKeys.length) {
    const count = review.missingValueKeys.length;
    missingParts.push(`${count} ${count === 1 ? 'Feld' : 'Felder'} nicht ermittelt`);
  }
  if (review.missingEvidenceKeys.length) {
    const count = review.missingEvidenceKeys.length;
    missingParts.push(`${count} ${count === 1 ? 'Feld' : 'Felder'} ohne zweiten unabhängigen Beleg`);
  }
  return `<section class="thesen-detail-section thesen-review-summary">
    <h3>${tr('reviewTitle', 'Rechercheprüfung')}</h3>
    <p>${review.evidenceUnloaded
      ? `<strong>${tr('evidenceUnloadedTitle', 'Belege noch nicht geladen')}</strong> · ${tr('evidenceUnloadedHint', 'Der Lead traegt Belege, die Datenverbindung liefert sie gerade nicht. Bitte neu laden, bevor du entscheidest.')}`
      : `<strong>${review.filledCount} ${tr('ofFields', 'von')} ${review.total} ${tr('fieldsFilled', 'Feldern ermittelt')}</strong> · ${review.sufficientCount} ${tr('fieldsSufficient', 'mit mindestens zwei unabhängigen Quellen belegt')}`}</p>
    <p class="${missingParts.length ? 'thesen-review-open' : 'thesen-muted'}">${missingParts.length
      ? `${tr('stillOpen', 'Noch offen')}: ${escapeHtml(missingParts.join(' · '))}`
      : escapeHtml(tr('allFieldsProven', 'Alle Pflichtfelder sind ermittelt und mit zwei unabhängigen Quellen belegt.'))}</p>
    <p class="thesen-muted">${tr('status', 'Status')}: ${escapeHtml(researchLabel(lead))} · ${tr('campaign', 'Kampagne')}: ${escapeHtml(lead?.campaign || tr('annualResearch', 'Jahresrecherche'))}</p>
  </section>`;
}
function renderContactRecipientSelection(lead) {
  const normalized = normalizeLeadRecipientShape(lead || {});
  const selectedIds = new Set(normalized.selected_contact_ids);
  // Doppelte Kontakte aus mehreren Rechercheläufen zusammenfassen: gleicher
  // Name = ein Empfänger. Sonst erscheint "Arndt Schlosser" mehrfach.
  const seen = new Map();
  for (const contact of normalized.contacts) {
    const name = (contact.name || [contact.first_name || contact.person_vorname, contact.last_name || contact.person_nachname].filter(Boolean).join(' ')).trim();
    const key = name.toLowerCase() || contact.id;
    if (!seen.has(key)) seen.set(key, { ...contact, _name: name || tr('contact', 'Kontakt') });
  }
  const contacts = [...seen.values()];
  const selectedCount = contacts.filter((contact) => (
    selectedIds.has(contact.id) && currentContactEligibility(normalized, contact).status === 'free'
  )).length;
  const countLabel = `${selectedCount} / ${contacts.length}`;
  const rows = contacts.map((contact) => {
    const role = contact.role || contact.function || contact.position || contact.person_funktion || '';
    const email = contact.email || contact.person_email || '';
    const detail = [role, email].filter(Boolean).join(' · ');
    const decision = currentContactEligibility(normalized, contact);
    const excluded = decision.status !== 'free';
    return `<label class="thesen-recipient ${excluded ? `is-${escapeHtml(decision.status)}` : ''}">
      <input type="checkbox" data-action="toggle-contact-recipient" data-id="${escapeHtml(normalized.id || '')}" data-contact-id="${escapeHtml(contact.id)}" ${selectedIds.has(contact.id) && !excluded ? 'checked' : ''} ${excluded ? 'disabled' : ''}>
      <span class="thesen-recipient-text">
        <span class="thesen-recipient-name">${escapeHtml(contact._name)}</span>
        ${detail ? `<span class="thesen-recipient-detail">${escapeHtml(detail)}</span>` : ''}
        ${excluded ? `<span class="thesen-recipient-protection"><strong>${escapeHtml(decision.label)}</strong>${decision.reason && decision.reason !== decision.label ? ` · ${escapeHtml(decision.reason)}` : ''}</span>` : ''}
        ${decision.originalRemark ? `<q class="thesen-recipient-remark">${escapeHtml(decision.originalRemark)}</q>` : ''}
      </span>
    </label>`;
  }).join('');
  return `<div class="thesen-recipient-selection">
    <div class="thesen-recipient-selection-head">
      <strong>${tr('sellifyRecipients', 'Empfänger für Sellify')}</strong>
      <span class="thesen-recipient-count" data-selected-contact-count>${escapeHtml(countLabel)}</span>
    </div>
    ${rows || `<p class="thesen-muted">${tr('noContacts', 'Noch keine Ansprechpartner.')}</p>`}
  </div>`;
}

function renderResearchReviewGroups(review, lead) {
  const untouched = leadIsUnresearched(lead);
  return review.groups.map((group) => {
    const recipientSelection = group.id === 'contact' ? renderContactRecipientSelection(lead) : '';
    return `<section class="thesen-detail-section thesen-review-group" data-review-group="${escapeHtml(group.id)}">
      <h3>${escapeHtml(group.label)}</h3>
      <ul class="thesen-review-fields">${group.fields.map((field) => renderReviewFieldRow(field, untouched, lead)).join('')}</ul>
      ${recipientSelection}
    </section>`;
  }).join('');
}
function renderResearchReview(lead) {
  const review = researchFieldReview(lead);
  return renderResearchReviewSummary(review, lead) + renderResearchReviewGroups(review, lead);
}
function campaignResearchProgress(campaign, leads, run = null) {
  const scoped = (leads || []).filter((lead) => String(lead.campaign || '').trim() === String(campaign || '').trim());
  const counts = { new: 0, queued: 0, running: 0, completed: 0, failed: 0, validated: 0 };
  for (const lead of scoped) {
    if (lead.validation_status === 'validated') counts.validated += 1;
    else if (lead.research_status === 'failed') counts.failed += 1;
    else if (lead.research_status === 'queued') counts.queued += 1;
    else if (lead.research_status === 'running') counts.running += 1;
    else if (['completed', 'needs_review'].includes(lead.research_status)) counts.completed += 1;
    else counts.new += 1;
  }
  const total = scoped.length;
  const actionable = campaignResearchQueue(scoped).length;
  const processed = counts.completed + counts.failed + counts.validated;
  const active = run?.status === 'running' && (counts.queued + counts.running) > 0;
  const running = active || counts.running > 0;
  const queued = counts.queued > 0;
  const trackedLead = scoped.find((lead) => (
    ['queued', 'running'].includes(lead.research_status)
    && String(lead.task_id || '').trim()
    && String(lead.command_id || '').trim()
  ));
  const campaignTrackingLead = scoped.find((lead) => (
    String(lead?.payload?.campaign_task_id || '').trim()
    && String(lead?.payload?.campaign_command_id || '').trim()
  ));
  let status = 'new';
  if (running) status = 'running';
  else if (queued) status = 'queued';
  else if (total > 0 && counts.validated === total) status = 'validated';
  else if (run?.status === 'failed' || (processed === total && counts.failed > 0)) status = 'failed';
  else if (total > 0 && processed === total) status = 'completed';
  return {
    status,
    active,
    queued,
    running,
    total,
    actionable,
    processed,
    percent: total > 0 ? Math.round((processed / total) * 100) : 0,
    counts,
    currentLeadId: run?.currentLeadId || '',
    currentLeadName: run?.currentLeadName || '',
    trackingTaskId: String(run?.taskId || campaignTrackingLead?.payload?.campaign_task_id || trackedLead?.task_id || '').trim(),
    trackingCommandId: String(run?.commandId || campaignTrackingLead?.payload?.campaign_command_id || trackedLead?.command_id || '').trim(),
    error: run?.error || '',
  };
}

async function openCtoxTask(taskId, commandId) {
  const normalizedTaskId = String(taskId || '').trim();
  const normalizedCommandId = String(commandId || '').trim();
  if (!normalizedTaskId || !normalizedCommandId) {
    showBusinessAlert('Für diese Automatisierung wurde kein verfolgbarer CTOX Task bestätigt.');
    return;
  }
  const focus = {
    taskId: normalizedTaskId,
    commandId: normalizedCommandId,
    taskStatus: 'queued',
    sourceModule: 'thesen-outbound',
    openDrawer: true,
  };
  try {
    sessionStorage.setItem('ctox.businessOs.focusTask', JSON.stringify(focus));
  } catch {}
  const params = new URLSearchParams({
    task_id: normalizedTaskId,
    command_id: normalizedCommandId,
    task_status: 'queued',
    source: 'thesen-outbound',
    drawer: '1',
  });
  await state.ctx?.businessChat?.open?.({
    title: 'Kampagnenrecherche',
    task_id: normalizedTaskId,
    command_id: normalizedCommandId,
    focus: {
      task_id: normalizedTaskId,
      command_id: normalizedCommandId,
    },
    source_module: 'thesen-outbound',
    reuseActive: false,
  });
  location.hash = `#ctox?${params.toString()}`;
  await state.ctx?.openApp?.('ctox');
  window.dispatchEvent(new CustomEvent('ctox-business-os-focus-task', { detail: focus }));
}
function campaignResearchStatusLabel(status) {
  const labels = {
    new: tr('statusNew', 'Neu'),
    queued: tr('statusQueued', 'Wartet'),
    running: tr('statusRunning', 'Laufend'),
    completed: tr('statusCompleted', 'Abgeschlossen'),
    failed: tr('statusFailed', 'Unvollständig'),
    validated: tr('statusValidated', 'Validiert'),
  };
  return labels[status] || labels.new;
}
function domainFromUrl(value) { try { return new URL(value).hostname.replace(/^www\./, ''); } catch { return ''; } }
function fingerprint(value) { let hash = 2166136261; for (const char of String(value).toLowerCase()) { hash ^= char.charCodeAt(0); hash = Math.imul(hash, 16777619); } return (hash >>> 0).toString(36); }
function sanitizeCommandResult(result) { return { status: result?.status || '', command_id: result?.command_id || '', task_id: result?.task_id || '', error: result?.error || '', secret_value_in_payload: false }; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]); }
function tr(key, fallback) { return String(state.messages?.[key] || fallback || key); }

function startResize(event) {
  const handle = event.target.closest('[data-resizer]');
  if (!handle) return;
  event.preventDefault();
  const layout = state.ctx.host.querySelector('.thesen-outbound-layout');
  const side = handle.dataset.resizer;
  const startX = event.clientX;
  const styles = getComputedStyle(layout);
  const start = Number.parseFloat(styles.getPropertyValue(side === 'left' ? '--thesen-left' : '--thesen-right')) || (side === 'left' ? 300 : 360);
  const move = (moveEvent) => {
    const delta = moveEvent.clientX - startX;
    const value = side === 'left' ? start + delta : start - delta;
    layout.style.setProperty(side === 'left' ? '--thesen-left' : '--thesen-right', `${Math.max(220, Math.min(520, value))}px`);
  };
  const stop = () => { globalThis.removeEventListener('pointermove', move); globalThis.removeEventListener('pointerup', stop); };
  globalThis.addEventListener('pointermove', move);
  globalThis.addEventListener('pointerup', stop, { once: true });
}

function icon(name) {
  const paths = {
    plus: '<path d="M12 5v14M5 12h14"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    settings: '<path d="M12 3a2 2 0 0 1 2 2v1.2a7 7 0 0 1 1.7 1l1-.6a2 2 0 0 1 2.7.7l.6 1a2 2 0 0 1-.7 2.7l-1 .6a7 7 0 0 1 0 2l1 .6a2 2 0 0 1 .7 2.7l-.6 1a2 2 0 0 1-2.7.7l-1-.6a7 7 0 0 1-1.7 1V19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-1.2a7 7 0 0 1-1.7-1l-1 .6a2 2 0 0 1-2.7-.7l-.6-1a2 2 0 0 1 .7-2.7l1-.6a7 7 0 0 1 0-2l-1-.6a2 2 0 0 1-.7-2.7l.6-1a2 2 0 0 1 2.7-.7l1 .6a7 7 0 0 1 1.7-1V5a2 2 0 0 1 2-2h1Z"/><circle cx="11.5" cy="12" r="2.5"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    wrench: '<path d="M14.7 6.3a4 4 0 0 0-5 5L3 18l3 3 6.7-6.7a4 4 0 0 0 5-5l-2.4 2.4-3-3 2.4-2.4Z"/>',
    test: '<path d="M9 3h6M10 3v5l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3M8 14h8"/>',
    login: '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4M10 17l5-5-5-5M15 12H3"/>',
    import: '<path d="M12 3v12m0 0 4-4m-4 4-4-4M5 19h14"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    spinner: '<path d="M20 12a8 8 0 1 1-2.3-5.7"/>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    external: '<path d="M14 3h7v7M10 14 21 3"/><path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/>',
    book: '<path d="M4 5a3 3 0 0 1 3-3h5v18H7a3 3 0 0 0-3 2V5Z"/><path d="M20 5a3 3 0 0 0-3-3h-5v18h5a3 3 0 0 1 3 2V5Z"/>',
    edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"/>',
    shards: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    table: '<path d="M4 5h16v14H4zM4 10h16M4 15h16M10 5v14"/>',
  };
  return `<svg class="thesen-icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.check}</svg>`;
}

export const __thesenOutboundTestHooks = {
  adapterCommandOperation,
  // Exported so the flow can be EXECUTED in tests, not only grepped:
  // every source-level assertion stayed green while renderDetail threw
  // and no recipient selection could persist.
  sellifyHandoffPrecondition,
  sendLeadToSellify,
  seriesEmailHandoffForLead,
  setContactRecipientSelection,
  buildCampaignRecipientList,
  classifySellifyPerson,
  deriveLeadRecipientEligibility,
  normalizeProtectionText,
  repairLeadRecipientSelections,
  refreshLeadRecipientEligibility,
  renderCampaignRecipientExclusions,
  adapterCommandRecordPatch,
  adapterReady,
  campaignResearchProgress,
  campaignResearchPrompt,
  campaignResearchQueue,
  enabledSourcePolicy,
  evidenceSourceKey,
  fieldEvidenceSources,
  independentEvidenceCount,
  sourceEvidenceGroups,
  independentFieldEvidenceCount,
  leadReadyForValidation,
  renderResearchReview,
  researchFieldReview,
  researchFieldValue,
  validationBlockers,
  RESEARCH_FIELD_GROUPS,
  leadResearchMode,
  normalizedResearchCountry,
  openCampaignResearchChat,
  requireTrackedSubmission,
  repairUntrackedResearchStatuses,
  researchPolicyInstructions,
  researchPolicyRecord,
  researchOutcomeWriteback,
  researchCommandLeadPatch,
  researchCommandForLead,
  researchCommandObservationKey,
  sellifyVorwissenAlsText,
  researchFieldReview,
  normalizedResearchCommandStatus,
  uniqueCommands,
  resetLeadForImport,
  rxdbIdSlug,
  sellifyCompanyValues,
  sellifyPersonValues,
  sellifyWritebackContract,
  sourceNeedsBrowserAuthorization,
  sourceStatus,
  SOURCE_DEFS,
  // Exposed so a test can actually RENDER and inspect the markup. A source-only
  // check cannot catch a render that produces structurally empty output — a
  // truncated template still parses and still passes every unit test, and the
  // app ships blank. See tests/render-smoke.
  __render: {
    setState: (next) => Object.assign(state, next),
    getState: () => state,
    render,
    renderCampaigns,
    renderCenter,
    renderDetail,
  },
};
