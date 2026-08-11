// Stable DOM writes for Business OS list wells and detail panes.
// Re-renders from data events must never yank the operator's scroll or steal
// focus from an active input. Prefer skipping the write entirely when the
// caller-supplied signature is unchanged; only fall back to a write that
// preserves scroll offsets when the content actually changed.

export function detailPaneHasActiveInput(root) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  if (!active || !root?.contains?.(active)) return false;
  if (active.isContentEditable) return true;
  const tag = String(active.tagName || '').toUpperCase();
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

export function preserveScrollDuring(target, fn) {
  if (!target || typeof fn !== 'function') {
    return typeof fn === 'function' ? fn() : undefined;
  }
  const top = Number(target.scrollTop) || 0;
  const left = Number(target.scrollLeft) || 0;
  const result = fn();
  if (typeof target.scrollTop === 'number') target.scrollTop = top;
  if (typeof target.scrollLeft === 'number') target.scrollLeft = left;
  return result;
}

export function renderHtmlIfChanged(target, html, {
  signature,
  preserveScroll = true,
} = {}) {
  if (!target) return false;
  const nextHtml = String(html ?? '');
  const sig = signature == null ? nextHtml : String(signature);
  if (target.dataset?.ctoxRenderSig === sig) return false;
  const write = () => {
    target.innerHTML = nextHtml;
    if (target.dataset) target.dataset.ctoxRenderSig = sig;
  };
  if (preserveScroll) preserveScrollDuring(target, write);
  else write();
  return true;
}

export function replaceChildrenIfChanged(target, nodes, {
  signature,
  preserveScroll = true,
} = {}) {
  if (!target) return false;
  const list = Array.isArray(nodes) ? nodes : (nodes == null ? [] : [nodes]);
  const sig = signature == null
    ? String(list.length)
    : String(signature);
  if (signature != null && target.dataset?.ctoxRenderSig === sig) return false;
  const write = () => {
    target.replaceChildren(...list);
    if (target.dataset) target.dataset.ctoxRenderSig = sig;
  };
  if (preserveScroll) preserveScrollDuring(target, write);
  else write();
  return true;
}
