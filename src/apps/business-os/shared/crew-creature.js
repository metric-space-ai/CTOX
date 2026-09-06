/** Durable Crew identity. No generated names, colours, IDs or random motion.
 * Consumers provide localized state labels; this component owns no UI copy.
 */
export const CREATURE_STATES = Object.freeze(['sleeping', 'queued', 'waking', 'thinking', 'tooling', 'reviewing', 'waiting', 'failed', 'done']);
export const CREATURE_SIZES = Object.freeze(['fab', 'dock', 'home', 'workplace']);
export const CREATURE_SHAPES = Object.freeze(['round', 'square', 'triangle', 'blob']);
const FAILED = new Set(['failed', 'error']);
const DONE = new Set(['done', 'completed', 'succeeded', 'sent', 'approved']);
const REVIEW = new Set(['review', 'reviewing', 'awaiting_review', 'validating']);
const ACTIVE = new Set(['leased', 'running', 'working', 'drafting']);

/** Terminal evidence outranks animation; stale or foreign events never animate. */
export function creatureState({ member, task, events = [], now = Date.now(), connected = true } = {}) {
  if (!connected) return 'sleeping';
  if (!task) return member?.state === 'resting_after_failure' ? 'failed' : member?.state === 'on_duty' ? 'waking' : 'sleeping';
  const status = String(task.status || task.task_status || '').toLowerCase();
  const route = String(task.route_status || task.routeStatus || '').toLowerCase();
  const phase = String(task.execution_phase || task.executionPhase || '').toLowerCase();
  const review = String(task.execution_progress?.review?.status || task.executionProgress?.reviewStatus || '').toLowerCase();
  if (FAILED.has(status) || (phase === 'terminal' && FAILED.has(task.terminal_status))) return 'failed';
  if (DONE.has(status) || (phase === 'terminal' && DONE.has(task.terminal_status))) return 'done';
  if (['blocked', 'cancelled', 'canceled'].includes(status) || ['blocked', 'deferred'].includes(route)
      || (task.retry_not_before && Date.parse(task.retry_not_before) > now) || task.wait_entity_id) return 'waiting';
  if ([status, route, phase, review].some(value => REVIEW.has(value))) return 'reviewing';
  const active = [status, route, phase].some(value => ACTIVE.has(value));
  if (!active) return 'queued';
  const id = task.task_id || task.taskId || task.id;
  const recent = events.filter(event => event.task_id === id && Number.isFinite(event.created_at_ms)
    && now >= event.created_at_ms && now - event.created_at_ms <= 10_000)
    .sort((a, b) => b.created_at_ms - a.created_at_ms || String(b.id).localeCompare(String(a.id)));
  for (const event of recent) {
    if (event.kind === 'tool_started') return 'tooling';
    if (['thinking', 'plan_updated', 'tool_completed'].includes(event.kind)) return 'thinking';
    if (event.kind === 'crew_selected') return 'waking';
  }
  return 'waking';
}

const bodies = Object.freeze({
  round: '<circle cx="32" cy="34" r="23"/>',
  square: '<rect x="10" y="12" width="44" height="44" rx="9"/>',
  triangle: '<path d="M27 10Q32 3 37 10L58 48Q62 57 51 57H13Q2 57 6 48Z"/>',
  blob: '<path d="M9 27C8 12 23 6 35 10C45 2 59 15 54 29C67 42 53 59 39 56C23 66 5 53 10 41C3 36 4 30 9 27Z"/>',
});
const eyes = Object.freeze({
  sleeping: '<path d="M19 36q5 5 10 0m7 0q5 5 10 0"/>',
  queued: '<path d="M21 36h7m10 0h7"/><path d="M28 46h8"/>',
  waking: '<path d="M25 30v9m15-9v9"/><ellipse cx="32" cy="47" rx="3" ry="4"/>',
  thinking: '<path d="M23 31v7m16-9v7m-13 10q5-3 10 0"/>',
  tooling: '<path d="M20 30l7 3m17-3l-7 3m-13 2v5m16-5v5m-12 6h8"/>',
  reviewing: '<circle cx="24" cy="35" r="6"/><circle cx="41" cy="35" r="6"/><path d="M30 35h5m-7 12h8"/>',
  waiting: '<path d="M24 32v5m16-5v5m-12 9h8"/>',
  failed: '<path d="M19 29l10 12m0-12L19 41m16-12l10 12m0-12L35 41m-7 8q4-5 8 0"/>',
  done: '<path d="M19 36q5-8 10 0m6 0q5-8 10 0m-19 8q6 8 12 0"/>',
});
const symbols = Object.freeze({
  sleeping: '<path d="M48 7h8l-8 8h8"/>',
  queued: '<path d="M51 10h10m-10 5h7m-7 5h4"/>',
  waking: '<path d="M6 6l5 6M20 1l1 7M1 20l7 1"/>',
  thinking: '<circle cx="52" cy="10" r="5"/><circle cx="45" cy="19" r="2"/>',
  tooling: '<path d="M51 12l8 8m-10 2l12-12m-12-1l4-4 8 8-4 4"/>',
  reviewing: '<path d="M49 6h11v16H49zM52 11h5m-5 5h5"/>',
  waiting: '<path d="M48 6h13m-13 19h13M49 7q0 7 6 9-6 2-6 8m11-17q0 7-5 9 5 2 5 8"/>',
  failed: '<path d="M51 7l9 9m0-9l-9 9"/>',
  done: '<path d="M48 12l5 5 9-11"/>',
});
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));

export function crewCreatureHtml(member, { state = 'sleeping', size = 'home', label = '', decorative = false } = {}) {
  if (!member?.id || !CREATURE_SHAPES.includes(member.shape) || !/^#[0-9a-f]{6}$/i.test(member.color || '')) return '';
  if (!CREATURE_STATES.includes(state)) state = 'sleeping';
  if (!CREATURE_SIZES.includes(size)) size = 'home';
  return `<span class="crew-creature crew-creature--${size} is-${state} shape-${member.shape}" data-crew-id="${escape(member.id)}" data-crew-state="${state}" style="--crew-color:${member.color}" ${decorative ? 'aria-hidden="true"' : `role="img" aria-label="${escape(label || member.name)}"`}><svg viewBox="0 0 68 68" focusable="false" aria-hidden="true"><g class="crew-creature__body">${bodies[member.shape]}</g><g class="crew-creature__face">${eyes[state]}</g><g class="crew-creature__signal">${symbols[state]}</g></svg></span>`;
}
