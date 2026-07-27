function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function renderListOrState(rows, readiness, {
  renderRows = () => '',
  empty = 'Keine Einträge.',
  syncing = 'Daten werden synchronisiert.',
} = {}) {
  if (rows?.length > 0) return String(renderRows(rows) ?? '');
  if (readiness?.ready === false) {
    return `<div class="ctox-syncing" role="status" aria-live="polite">${escapeHtml(syncing)}</div>`;
  }
  return `<div class="ctox-empty">${escapeHtml(empty)}</div>`;
}
