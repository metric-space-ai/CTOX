const root = document.documentElement;
const shell = document.querySelector('[data-qa-shell]');
const themeButton = document.querySelector('[data-qa-theme]');
const widthButton = document.querySelector('[data-qa-width]');

function setTheme(theme) {
  const isLight = theme === 'light';
  root.dataset.theme = isLight ? 'light' : 'dark';
  themeButton.setAttribute('aria-pressed', String(isLight));
  themeButton.textContent = isLight ? 'Dark' : 'Light';
}

function setNarrow(isNarrow) {
  shell.style.width = isNarrow ? '680px' : '100%';
  shell.style.marginInline = isNarrow ? 'auto' : '0';
  widthButton.setAttribute('aria-pressed', String(isNarrow));
  widthButton.textContent = isNarrow ? 'Full' : 'Narrow';
}

themeButton.addEventListener('click', () => {
  setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
});

widthButton.addEventListener('click', () => {
  setNarrow(widthButton.getAttribute('aria-pressed') !== 'true');
});

setTheme(root.dataset.theme);
setNarrow(false);
