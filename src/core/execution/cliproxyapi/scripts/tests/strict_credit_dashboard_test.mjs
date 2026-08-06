import fs from 'node:fs';
import vm from 'node:vm';

const [dashboardPath, auditPath] = process.argv.slice(2);
if (!dashboardPath || !auditPath) {
  throw new Error('usage: strict_credit_dashboard_test.mjs DASHBOARD AUDIT');
}

const html = fs.readFileSync(dashboardPath, 'utf8');
const expected = JSON.parse(fs.readFileSync(auditPath, 'utf8'));
const match = html.match(/const strictAudit = decodeJson\('([^']+)'\);/);
if (!match) throw new Error('standalone dashboard does not embed strict-credit audit');
const embedded = JSON.parse(Buffer.from(match[1], 'base64').toString('utf8'));
const stateMatch = html.match(/const state = decodeJson\('([^']+)'\);/);
if (!stateMatch) throw new Error('standalone dashboard does not embed project state');
const state = JSON.parse(Buffer.from(stateMatch[1], 'base64').toString('utf8'));
const summaryMatch = html.match(/const summary = decodeJson\('([^']+)'\);/);
if (!summaryMatch) throw new Error('standalone dashboard does not embed port summary');
const summary = JSON.parse(Buffer.from(summaryMatch[1], 'base64').toString('utf8'));
const integrationMatch = html.match(/const integration = decodeJson\('([^']+)'\);/);
if (!integrationMatch) throw new Error('standalone dashboard does not embed CTOX integration state');
const integration = JSON.parse(Buffer.from(integrationMatch[1], 'base64').toString('utf8'));
const artifactMatch = html.match(/const artifact = decodeJson\('([^']+)'\);/);
if (!artifactMatch) throw new Error('standalone dashboard does not embed input integrity metadata');
const artifact = JSON.parse(Buffer.from(artifactMatch[1], 'base64').toString('utf8'));

if (JSON.stringify(embedded) !== JSON.stringify(expected)) {
  throw new Error('embedded strict-credit audit differs from validated source');
}
if (!html.includes('id="strict-credit-audit"')) {
  throw new Error('strict-credit audit section is missing');
}
if (!html.includes('strictAudit.umbrella_receipt')) {
  throw new Error('umbrella receipt closure is not rendered');
}
if (!html.includes('strictAudit.closed_reconstructed.map')) {
  throw new Error('dashboard renderer does not expose closed reconstructed clusters');
}
if (!html.includes('Track A · Reiner CLIProxy Rust Port') ||
    !html.includes('Track B · CTOX Provider Integration') ||
    !html.includes('Keine Gesamtprojekt-Prozentzahl')) {
  throw new Error('dashboard does not expose the strict port/integration boundary');
}

const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (scripts.length !== 1) throw new Error('standalone dashboard must contain one inline renderer');
const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) {
    elements.set(id, { textContent: '', innerHTML: '', style: {} });
  }
  return elements.get(id);
};
vm.runInNewContext(scripts[0][1], {
  document: { getElementById: element },
  requestAnimationFrame: (callback) => callback(),
  atob: (value) => Buffer.from(value, 'base64').toString('binary'),
  TextDecoder,
  Uint8Array,
  console,
});

const rendered = (id) => element(id).textContent;
if (rendered('verified-production') !== String(state.mirror_verification.verified_classified_production_files) ||
    rendered('closed-production') !== String(state.mirror_verification.strict_production_files)) {
  throw new Error('rendered production counters differ from project state');
}
if (rendered('verified-tests') !== String(state.mirror_verification.verified_test_files) ||
    rendered('production-total') !== String(summary.production_go_files) ||
    rendered('test-total') !== String(summary.test_go_files)) {
  throw new Error('rendered test or production total differs from its ledger');
}
if (rendered('project-status') !== (state.project_completion.complete ? 'ABGESCHLOSSEN' : 'IN ARBEIT') ||
    rendered('candidate-reviewed') !== String(state.upstream_candidate.reviewed) ||
    rendered('candidate-total') !== String(state.upstream_candidate.inventory_total) ||
    rendered('candidate-pending') !== String(state.upstream_candidate.pending)) {
  throw new Error('dashboard does not separate project state from candidate review fraction');
}
if (rendered('candidate-gates') !== String(state.upstream_candidate.gates_passed) ||
    rendered('candidate-gates-total') !== String(state.upstream_candidate.gates_total) ||
    rendered('candidate-promoted') !== (state.upstream_candidate.promoted ? 'JA' : 'NEIN')) {
  throw new Error('dashboard does not render candidate gate and promotion state');
}
const hasOpenCandidate = !(state.upstream_candidate.promoted &&
  state.project_completion.post_promotion_full_gate &&
  state.upstream_candidate.commit === state.upstream_commit);
if (hasOpenCandidate) {
  if (!rendered('target').includes(`${state.upstream_candidate.inventory_total}/${state.upstream_candidate.inventory_total} Candidate-Reviews`) ||
      !rendered('target').includes(`${state.upstream_candidate.gates_total}/${state.upstream_candidate.gates_total} Gates`) ||
      !rendered('candidate-state').includes('Offener Candidate')) {
    throw new Error('dashboard completion condition omits open candidate promotion work');
  }
} else if (!rendered('target').includes('Kein offener Upstream-Candidate') ||
           !rendered('candidate-state').includes('Kein offener Candidate') ||
           rendered('candidate-cycle-label') !== 'Letzter abgeschlossener Promotionszyklus') {
  throw new Error('dashboard presents the promoted historical cycle as an open candidate');
}
const completionUnitsTotal = state.upstream_candidate.inventory_total
  + state.upstream_candidate.gates_total + 2;
const completionUnitsDone = state.upstream_candidate.reviewed
  + state.upstream_candidate.gates_passed
  + (state.upstream_candidate.promoted ? 1 : 0)
  + (state.project_completion.post_promotion_full_gate ? 1 : 0);
const overallPercent = completionUnitsTotal === 0
  ? 0
  : (completionUnitsDone / completionUnitsTotal) * 100;
const expectedOverallPercent = overallPercent.toLocaleString('de-DE', {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
if (rendered('overall-percent') !== expectedOverallPercent ||
    rendered('completion-units-done') !== String(completionUnitsDone) ||
    rendered('completion-units-total') !== String(completionUnitsTotal) ||
    element('overall-progress-bar').style.width !== `${overallPercent}%`) {
  throw new Error('dashboard overall percentage is absent or differs from the closure formula');
}
if ((overallPercent === 100) !== state.project_completion.complete) {
  throw new Error('dashboard can render 100 percent without the complete project predicate');
}
if (!html.includes('Track A · Reiner CLIProxy Rust Port') ||
    !rendered('completion-formula').includes(`${completionUnitsDone}/${completionUnitsTotal}`)) {
  throw new Error('dashboard does not expose the complete Track-A progress formula');
}
const integrationEntries = integration.provider_modes.flatMap(mode => Object.values(mode.gates));
const integrationVerified = integrationEntries.filter(status => status === 'verified').length;
const integrationPending = integrationEntries.filter(status => status === 'pending').length;
if (rendered('integration-status') !== (integration.completion_rule.complete ? 'ABGESCHLOSSEN' : 'OFFENE INTEGRATIONSGATES') ||
    !element('integration-summary').innerHTML.includes(`>${integrationVerified}<`) ||
    !element('integration-summary').innerHTML.includes(`>${integrationPending}<`) ||
    !element('integration-grid').innerHTML.includes('Codex') ||
    !element('integration-grid').innerHTML.includes('Kimi') ||
    !element('integration-grid').innerHTML.includes('MiniMax')) {
  throw new Error('dashboard does not render the independent CTOX provider integration ledger');
}
if (element('integration-summary').innerHTML.includes('%') ||
    element('integration-grid').innerHTML.includes('%')) {
  throw new Error('CTOX provider integration lane must not synthesize a percentage');
}
if (artifact.schema !== 'ctox.cliproxyapi.dashboard-snapshot.v1' ||
    Object.keys(artifact.input_sha256).length !== 7 ||
    Object.values(artifact.input_sha256).some(hash => !/^[0-9a-f]{64}$/.test(hash)) ||
    !rendered('artifact-integrity').includes(artifact.schema)) {
  throw new Error('dashboard input integrity metadata is missing or malformed');
}
if (state.project_completion.complete) {
  if (state.work_items.some(item => item.status !== 'complete') ||
      element('work-list').innerHTML.trim() !== '') {
    throw new Error('completed dashboard still renders active work');
  }
} else if (!element('work-list').innerHTML.includes(`${state.upstream_candidate.reviewed}/${state.upstream_candidate.inventory_total} Candidate-Reviews`) ||
           !element('work-list').innerHTML.includes(`${state.upstream_candidate.gates_passed}/${state.upstream_candidate.gates_total} Promotion-Gates`)) {
  throw new Error('rendered active work list does not expose candidate completion requirements');
}
if (element('work-list').innerHTML.includes('596/605') || element('work-list').innerHTML.includes('403/418')) {
  throw new Error('rendered work list contains stale strict counters');
}

console.log('strict-credit dashboard embedding and renderer test passed');
