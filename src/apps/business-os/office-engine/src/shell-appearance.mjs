// Only visual shell tokens cross the editor boundary. No stylesheet, selector,
// markup, account state or application data is passed into the editor.
export const SHELL_COLOR_TOKENS = Object.freeze([
  'bg', 'surface', 'surface-2', 'line', 'text', 'text-strong', 'muted',
  'accent', 'accent-soft', 'accent-foreground', 'danger', 'warning', 'success', 'focus-ring',
]);

export function readShellAppearance(host, fallback = 'system') {
  const view = host.ownerDocument.defaultView;
  const style = view.getComputedStyle(host);
  const shellTheme = host.ownerDocument.documentElement.dataset.theme;
  const theme = ['dark', 'light'].includes(shellTheme) ? shellTheme
    : ['dark', 'light'].includes(fallback) ? fallback
      : view.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const tokens = Object.fromEntries(SHELL_COLOR_TOKENS.map((name) => [name, style.getPropertyValue(`--${name}`).trim()]));
  return { theme, tokens, fontFamily: style.fontFamily };
}

export function applyShellAppearance(document, appearance = {}) {
  const root = document.documentElement;
  root.style.colorScheme = appearance.theme === 'dark' ? 'dark' : 'light';
  for (const name of SHELL_COLOR_TOKENS) {
    const value = appearance.tokens?.[name];
    const property = `--ctox-shell-${name}`;
    if (typeof value === 'string' && value && document.defaultView.CSS.supports('color', value)) {
      root.style.setProperty(property, value);
    } else {
      root.style.removeProperty(property);
    }
  }
  if (typeof appearance.fontFamily === 'string' && appearance.fontFamily) {
    root.style.setProperty('--ctox-shell-font', appearance.fontFamily);
  } else root.style.removeProperty('--ctox-shell-font');
}

export function observeShellAppearance(host, fallback, publish) {
  const view = host.ownerDocument.defaultView;
  let frame = 0;
  let previous = '';
  const update = () => {
    frame = 0;
    const appearance = readShellAppearance(host, fallback);
    const signature = JSON.stringify(appearance);
    if (signature !== previous) {
      previous = signature;
      publish(appearance);
    }
  };
  const schedule = () => { if (!frame) frame = view.requestAnimationFrame(update); };
  const observer = new view.MutationObserver(schedule);
  for (let node = host; node; node = node.parentElement) {
    observer.observe(node, { attributes: true, attributeFilter: ['style', 'class', 'data-theme'] });
  }
  // Custom-brand palettes may be replaced as stylesheet text rather than as
  // attributes. Observe the head, not the editor or the shell's business data.
  observer.observe(host.ownerDocument.head, { childList: true, subtree: true, characterData: true });
  const colorScheme = view.matchMedia('(prefers-color-scheme: dark)');
  colorScheme.addEventListener('change', schedule);
  update();
  return () => {
    observer.disconnect();
    colorScheme.removeEventListener('change', schedule);
    if (frame) view.cancelAnimationFrame(frame);
  };
}
