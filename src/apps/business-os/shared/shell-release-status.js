// SPDX-License-Identifier: MIT OR AGPL-3.0-only

const STATUS_COPY = Object.freeze({
  current: ['✓', 'Aktuell'],
  checking: ['…', 'Prüfung läuft'],
  available: ['↓', 'Update verfügbar'],
  download: ['↓', 'Download'],
  verify: ['◌', 'Verifikation'],
  ready: ['✓', 'Bereit zur Aktivierung'],
  restart: ['↻', 'Neustart erforderlich'],
  failed: ['!', 'Fehlgeschlagen'],
  incompatible: ['!', 'Inkompatibel'],
  blocked: ['×', 'Kein administrativer Zugriff'],
  rollback: ['↶', 'Rollback aktiv'],
  recovery: ['!', 'Recovery-Shell'],
});

export function shellChannel(version) {
  const value = String(version || '');
  if (value.includes('-nightly.')) return 'nightly';
  if (value.includes('-')) return 'beta';
  return 'stable';
}

export function normalizeShellVersion(value) {
  const version = String(value || '').trim().replace(/^v/, '');
  return /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(version) ? version : '';
}

export function normalizeShellUpdateStatus(value) {
  const state = String(value || '');
  return Object.hasOwn(STATUS_COPY, state) ? state : 'failed';
}

function render(root, detail) {
  const version = normalizeShellVersion(detail?.version);
  const state = normalizeShellUpdateStatus(detail?.state);
  const channel = detail?.channel || (version ? shellChannel(version) : 'recovery');
  const [icon, stateLabel] = STATUS_COPY[state];
  root.querySelector('[data-shell-version-label]').textContent = version ? `v${version}` : 'Recovery';
  const statusButton = root.querySelector('[data-shell-release-status]');
  statusButton.dataset.state = state;
  statusButton.title = stateLabel;
  statusButton.querySelector('[data-shell-release-icon]').textContent = icon;
  root.querySelector('[data-shell-release-panel-version]').textContent = version ? `v${version}` : 'Recovery-Shell';
  root.querySelector('[data-shell-release-panel-channel]').textContent = channel;
  root.querySelector('[data-shell-release-panel-state]').textContent = stateLabel;
}

async function loadEmbeddedIdentity() {
  const response = await fetch('./ctox-shell-manifest.json', { cache: 'no-store', credentials: 'same-origin' });
  if (!response.ok) throw new Error(`Shell manifest unavailable (${response.status})`);
  const manifest = await response.json();
  if (manifest?.schema !== 'ctox.business-os-shell.v1') throw new Error('Unknown shell manifest');
  const version = normalizeShellVersion(manifest.version);
  if (!version) throw new Error('Invalid shell version');
  return { version, channel: shellChannel(version), state: 'current' };
}

function start() {
  const root = document.querySelector('[data-shell-release]');
  if (!root) return;
  const button = root.querySelector('[data-shell-release-status]');
  const panel = root.querySelector('[data-shell-release-panel]');
  button.addEventListener('click', () => {
    panel.hidden = !panel.hidden;
    button.setAttribute('aria-expanded', String(!panel.hidden));
  });
  document.addEventListener('click', (event) => {
    if (root.contains(event.target)) return;
    panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  });
  window.addEventListener('workjet:shell-update-status', (event) => render(root, event.detail));
  loadEmbeddedIdentity().then((identity) => render(root, identity)).catch(() => {
    render(root, { version: '', channel: 'recovery', state: 'recovery' });
  });
}

if (typeof document !== 'undefined') start();
