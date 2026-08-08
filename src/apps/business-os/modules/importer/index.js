import { loadModuleMessages } from '../../shared/i18n.js';
import {
  transcodeApp,
  suggestedModuleId,
  scaffoldModule,
} from '../../shared/app-transcode.mjs?v=20260807-einheitliche-ladepfade-v97';

// The App Importer is the hand-over moment of the product story: a coding
// agent conceived the app, the importer raises it. Source (folder or public
// GitHub repo) -> transcode to plain ESM (vendored sucrase, once, at import
// time) -> honest report -> write into runtime/business-os/local-modules/
// through the user-picked directory (dropping is installing; the filesystem
// stays the installation truth — no new HTTP data path).

const MAX_FILES = 400;
const MAX_FILE_BYTES = 512 * 1024;
const TEXT_EXTENSIONS = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.css', '.json', '.html', '.svg', '.md', '.txt',
]);
const ASSET_EXTENSIONS = new Set([
  '.avif', '.gif', '.ico', '.jpeg', '.jpg', '.png', '.webp', '.woff', '.woff2',
]);

const FALLBACK_LABELS = {
  title: 'App Importer',
  subtitle: 'Conceived in a prompt. Born with the first deploy. Raised in CTOX.',
};

export function parseGitHubUrl(raw) {
  let url;
  try {
    url = new URL(String(raw || '').trim());
  } catch {
    return null;
  }
  if (url.hostname !== 'github.com') return null;
  const parts = url.pathname.split('/').filter(Boolean);
  if (parts.length < 2) return null;
  const [owner, repo, marker, ref, ...rest] = parts;
  return {
    owner,
    repo: repo.replace(/\.git$/, ''),
    ref: marker === 'tree' && ref ? ref : null,
    subdir: marker === 'tree' && rest.length ? rest.join('/') : '',
  };
}

export function shouldSkipPath(path) {
  return /(^|\/)(node_modules|\.git|dist|build|out|coverage|\.next|\.cache)(\/|$)/.test(path)
    || /(^|\/)\./.test(path) && !/^\.?[^/]*rc/.test(path.split('/').pop() || '')
    || path.endsWith('.lock')
    || path.endsWith('package-lock.json')
    || path.endsWith('.map');
}

export function isTextFile(path) {
  const dot = path.lastIndexOf('.');
  if (dot === -1) return false;
  return TEXT_EXTENSIONS.has(path.slice(dot).toLowerCase());
}

export function isImportableFile(path) {
  const dot = path.lastIndexOf('.');
  if (dot === -1) return false;
  const extension = path.slice(dot).toLowerCase();
  return TEXT_EXTENSIONS.has(extension) || ASSET_EXTENSIONS.has(extension);
}

export function validModuleId(id) {
  return /^[a-z0-9][a-z0-9-]{1,63}$/.test(id);
}

async function readDirectoryFiles(dirHandle) {
  const files = {};
  let count = 0;
  async function walk(handle, prefix) {
    for await (const [name, entry] of handle.entries()) {
      const path = prefix ? `${prefix}/${name}` : name;
      if (shouldSkipPath(path)) continue;
      if (entry.kind === 'directory') {
        await walk(entry, path);
      } else if (isImportableFile(path)) {
        count += 1;
        if (count > MAX_FILES) throw Object.assign(new Error('too_many_files'), { count });
        const file = await entry.getFile();
        if (file.size > MAX_FILE_BYTES) continue;
        files[path] = isTextFile(path)
          ? await file.text()
          : new Uint8Array(await file.arrayBuffer());
      }
    }
  }
  await walk(dirHandle, '');
  return files;
}

async function fetchGitHubFiles(ref) {
  const branch = ref.ref
    || (await (await fetch(`https://api.github.com/repos/${ref.owner}/${ref.repo}`)).json()).default_branch
    || 'main';
  const treeRes = await fetch(
    `https://api.github.com/repos/${ref.owner}/${ref.repo}/git/trees/${encodeURIComponent(branch)}?recursive=1`,
  );
  if (!treeRes.ok) throw new Error(`GitHub API ${treeRes.status}`);
  const tree = await treeRes.json();
  const wanted = (tree.tree || [])
    .filter((node) => node.type === 'blob')
    .map((node) => node.path)
    .filter((path) => (ref.subdir ? path.startsWith(`${ref.subdir}/`) : true))
    .filter((path) => !shouldSkipPath(path) && isImportableFile(path));
  if (wanted.length === 0) throw new Error('no importable files');
  if (wanted.length > MAX_FILES) throw Object.assign(new Error('too_many_files'), { count: wanted.length });
  const files = {};
  for (const path of wanted) {
    const rel = ref.subdir ? path.slice(ref.subdir.length + 1) : path;
    const res = await fetch(
      `https://raw.githubusercontent.com/${ref.owner}/${ref.repo}/${encodeURIComponent(branch)}/${path}`,
    );
    if (!res.ok) continue;
    files[rel] = isTextFile(rel)
      ? await res.text()
      : new Uint8Array(await res.arrayBuffer());
  }
  return files;
}

async function writeModuleToDirectory(rootHandle, moduleId, moduleFiles, sourceFiles) {
  const moduleDir = await rootHandle.getDirectoryHandle(moduleId, { create: true });
  async function writeFile(dir, path, content) {
    const parts = path.split('/');
    let current = dir;
    for (const part of parts.slice(0, -1)) {
      current = await current.getDirectoryHandle(part, { create: true });
    }
    const fileHandle = await current.getFileHandle(parts.at(-1), { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(content);
    await writable.close();
  }
  for (const [path, content] of Object.entries(moduleFiles)) {
    await writeFile(moduleDir, path, content);
  }
  for (const [path, content] of Object.entries(sourceFiles)) {
    await writeFile(moduleDir, `source/${path}`, content);
  }
}

async function loadModuleMarkup() {
  const response = await fetch(new URL('./index.html', import.meta.url));
  if (!response.ok) throw new Error(`importer markup unavailable: ${response.status}`);
  return response.text();
}

function ensureStyles() {
  if (document.getElementById('importer-module-styles')) return;
  const styleLink = document.createElement('link');
  styleLink.rel = 'stylesheet';
  styleLink.href = new URL('./index.css', import.meta.url).href;
  styleLink.id = 'importer-module-styles';
  document.head.appendChild(styleLink);
}

export async function mount(ctx) {
  ensureStyles();
  const host = ctx?.host || document.body;
  ctx?.left?.replaceChildren?.();
  ctx?.right?.replaceChildren?.();
  host.innerHTML = await loadModuleMarkup();
  const root = host.querySelector('[data-importer-root]') || host;
  const messages = await loadModuleMessages(import.meta.url, ctx?.locale, FALLBACK_LABELS);
  const t = (key, fallback, vars = {}) => {
    let text = messages?.[key] ?? fallback ?? key;
    for (const [name, value] of Object.entries(vars)) text = text.replaceAll(`{${name}}`, String(value));
    return text;
  };

  const refs = {
    title: root.querySelector('[data-imp-title]'),
    subtitle: root.querySelector('[data-imp-subtitle]'),
    sourceStep: root.querySelector('[data-imp-source-step]'),
    sourceStepNote: root.querySelector('[data-imp-source-step-note]'),
    reviewStep: root.querySelector('[data-imp-review-step]'),
    reviewStepNote: root.querySelector('[data-imp-review-step-note]'),
    doneStep: root.querySelector('[data-imp-done-step]'),
    doneStepNote: root.querySelector('[data-imp-done-step-note]'),
    notice: root.querySelector('[data-imp-notice]'),
    sourceSection: root.querySelector('[data-imp-source-section]'),
    sourceEyebrow: root.querySelector('[data-imp-source-eyebrow]'),
    sourceHeading: root.querySelector('[data-imp-source-heading]'),
    sourceIntro: root.querySelector('[data-imp-source-intro]'),
    githubLabel: root.querySelector('[data-imp-github-label]'),
    folderHeading: root.querySelector('[data-imp-folder-heading]'),
    pickFolder: root.querySelector('[data-imp-pick-folder]'),
    or: root.querySelector('[data-imp-or]'),
    githubForm: root.querySelector('[data-imp-github-form]'),
    githubUrl: root.querySelector('[data-imp-github-url]'),
    githubBtn: root.querySelector('[data-imp-github-btn]'),
    sourceHint: root.querySelector('[data-imp-source-hint]'),
    reportSection: root.querySelector('[data-imp-report-section]'),
    reviewEyebrow: root.querySelector('[data-imp-review-eyebrow]'),
    reviewHeading: root.querySelector('[data-imp-review-heading]'),
    reviewIntro: root.querySelector('[data-imp-review-intro]'),
    reportStatus: root.querySelector('[data-imp-report-status]'),
    reportStatusNote: root.querySelector('[data-imp-report-status-note]'),
    report: root.querySelector('[data-imp-report]'),
    reportHint: root.querySelector('[data-imp-report-hint]'),
    details: root.querySelector('[data-imp-details]'),
    idLabel: root.querySelector('[data-imp-id-label]'),
    titleLabel: root.querySelector('[data-imp-title-label]'),
    moduleId: root.querySelector('[data-imp-module-id]'),
    moduleTitle: root.querySelector('[data-imp-module-title]'),
    back: root.querySelector('[data-imp-back]'),
    install: root.querySelector('[data-imp-install]'),
    doneSection: root.querySelector('[data-imp-done-section]'),
    doneEyebrow: root.querySelector('[data-imp-done-eyebrow]'),
    doneHeading: root.querySelector('[data-imp-done-heading]'),
    done: root.querySelector('[data-imp-done]'),
    open: root.querySelector('[data-imp-open]'),
    openLabel: root.querySelector('[data-imp-open-label]'),
  };

  refs.title.textContent = t('title', FALLBACK_LABELS.title);
  refs.subtitle.textContent = t('subtitle', FALLBACK_LABELS.subtitle);
  refs.sourceStep.textContent = t('sourceStep', 'Source');
  refs.sourceStepNote.textContent = t('sourceStepNote', 'Choose an app');
  refs.reviewStep.textContent = t('reviewStep', 'Review');
  refs.reviewStepNote.textContent = t('reviewStepNote', 'Check portability');
  refs.doneStep.textContent = t('doneStep', 'Install');
  refs.doneStepNote.textContent = t('doneStepNote', 'Open in CTOX');
  refs.sourceEyebrow.textContent = t('sourceEyebrow', 'Import source');
  refs.sourceHeading.textContent = t('sourceHeading', 'Bring an existing app into CTOX.');
  refs.sourceIntro.textContent = t('sourceIntro', 'Use a public GitHub repository or choose the project folder from this computer.');
  refs.githubLabel.textContent = t('githubLabel', 'GitHub repository');
  refs.folderHeading.textContent = t('folderHeading', 'Choose project folder');
  refs.or.textContent = t('or', 'or');
  refs.githubUrl.placeholder = t('githubPlaceholder', 'https://github.com/owner/repo');
  refs.githubBtn.textContent = t('fetchGithub', 'Analyze repo');
  refs.sourceHint.textContent = t('sourceHint', '');
  refs.reviewEyebrow.textContent = t('reviewEyebrow', 'Import review');
  refs.reviewHeading.textContent = t('reviewHeading', 'Ready to become a CTOX app.');
  refs.reviewIntro.textContent = t('reviewIntro', 'Review the detected entry point and runtime dependencies before installation.');
  refs.reportStatus.textContent = t('reportStatus', 'Portable');
  refs.reportStatusNote.textContent = t('reportStatusNote', 'The app can run in CTOX.');
  refs.idLabel.textContent = t('idLabel', 'Module id');
  refs.titleLabel.textContent = t('titleLabel', 'Title');
  refs.back.textContent = t('back', 'Back');
  refs.install.textContent = t('install', 'Install app');
  refs.doneEyebrow.textContent = t('doneEyebrow', 'Import complete');
  refs.doneHeading.textContent = t('doneHeading', 'Your app is in CTOX.');
  refs.openLabel.textContent = t('openApp', 'Open app');

  const state = { files: null, result: null, installedModuleId: '' };
  let disposed = false;

  const setStage = (stage) => {
    root.dataset.impStage = stage;
    refs.sourceSection.hidden = stage !== 'source';
    refs.reportSection.hidden = stage !== 'review';
    refs.doneSection.hidden = stage !== 'done';
  };

  const notify = (text, isError = false) => {
    refs.notice.hidden = !text;
    refs.notice.textContent = text || '';
    refs.notice.classList.toggle('is-danger', isError);
  };

  // Report rows render into a .ctox-fields <dl>: key -> dt, value -> dd.
  const row = (key, value, cls = '') =>
    `<dt>${key}</dt><dd${cls ? ` class="${cls}"` : ''}>${value}</dd>`;
  const esc = (value) => String(value).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  const sucrase = await import('../../vendor/app-importer/sucrase.mjs');

  function renderReport() {
    const { files, result } = state;
    if (!files || !result) return;
    refs.reportHint.hidden = true;
    refs.reportStatus.textContent = t('reportStatus', 'Portable');
    refs.reportStatusNote.textContent = t('reportStatusNote', 'The app can run in CTOX.');
    refs.reportStatus.closest('.imp-report-status')?.classList.remove('is-danger');
    const lines = [row(t('filesRead', 'Files read'), Object.keys(files).length)];
    if (result.report?.error === 'entry_not_found') {
      lines.push(row(t('entry', 'Entry'), esc(t('entryMissing', 'No entry found.')), 'imp-bad'));
      refs.details.hidden = true;
      refs.reportStatus.textContent = t('reportBlocked', 'Needs attention');
      refs.reportStatusNote.textContent = t('reportBlockedNote', 'CTOX could not find an app entry point.');
      refs.reportStatus.closest('.imp-report-status')?.classList.add('is-danger');
    } else {
      lines.push(row(t('entry', 'Entry'), esc(result.entry), 'imp-good'));
      lines.push(row(t('bareDeps', 'Runtime dependencies'), esc(result.report.bareImports.join(', ') || '—')));
      lines.push(row(t('cssFiles', 'Stylesheets'), esc(result.cssFiles.join(', ') || '—')));
      if (result.report.unsupported.length) {
        lines.push(row(t('unsupported', 'Not portable'), esc(result.report.unsupported.join(', ')), 'imp-bad'));
        const hint = t('unsupportedHint', '');
        refs.reportHint.textContent = hint;
        refs.reportHint.hidden = !hint;
        refs.details.hidden = true;
        refs.reportStatus.textContent = t('reportBlocked', 'Needs attention');
        refs.reportStatusNote.textContent = t('reportBlockedNote', 'Some dependencies need to be replaced before installation.');
        refs.reportStatus.closest('.imp-report-status')?.classList.add('is-danger');
      } else {
        lines.push(row(t('unsupported', 'Not portable'), t('ok', 'ready'), 'imp-good'));
        refs.details.hidden = false;
        refs.moduleId.value = suggestedModuleId(files);
        refs.moduleTitle.value = refs.moduleId.value
          .split('-').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
      }
    }
    refs.report.innerHTML = lines.join('');
    setStage('review');
  }

  async function analyze(files) {
    notify(t('transcoding', 'Transcoding…'));
    state.files = files;
    state.result = transcodeApp(sucrase, files, { vendorBase: '../../vendor/app-importer' });
    notify('');
    renderReport();
  }

  refs.pickFolder.addEventListener('click', async () => {
    if (typeof globalThis.showDirectoryPicker !== 'function') {
      notify(t('noPicker', 'Folder dialog unsupported.'), true);
      return;
    }
    try {
      const handle = await globalThis.showDirectoryPicker({ mode: 'read' });
      notify(t('reading', 'Reading source…'));
      const files = await readDirectoryFiles(handle);
      await analyze(files);
    } catch (error) {
      if (error?.name === 'AbortError') { notify(''); return; }
      if (error?.message === 'too_many_files') notify(t('tooManyFiles', 'Too many files.', { count: error.count }), true);
      else notify(t('readFailed', 'Could not read the folder.', { error: error?.message || error }), true);
    }
  });

  refs.githubForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const ref = parseGitHubUrl(refs.githubUrl.value);
    if (!ref) {
      notify(t('fetchFailed', 'GitHub fetch failed.', { error: 'invalid URL' }), true);
      return;
    }
    try {
      notify(t('fetching', 'Fetching repo from GitHub…'));
      const files = await fetchGitHubFiles(ref);
      await analyze(files);
    } catch (error) {
      if (error?.message === 'too_many_files') notify(t('tooManyFiles', 'Too many files.', { count: error.count }), true);
      else notify(t('fetchFailed', 'GitHub fetch failed.', { error: error?.message || error }), true);
    }
  });

  refs.install.addEventListener('click', async () => {
    const moduleId = refs.moduleId.value.trim();
    if (!validModuleId(moduleId)) {
      notify(t('idInvalid', 'Invalid module id.'), true);
      return;
    }
    if (typeof globalThis.showDirectoryPicker !== 'function') {
      notify(t('noPicker', 'Folder dialog unsupported.'), true);
      return;
    }
    try {
      notify(t('installPickHint', 'Pick the local-modules folder.'));
      const rootHandle = await globalThis.showDirectoryPicker({ mode: 'readwrite' });
      notify(t('writing', 'Writing module…'));
      const moduleFiles = scaffoldModule(
        { id: moduleId, title: refs.moduleTitle.value.trim() || moduleId },
        state.result,
      );
      await writeModuleToDirectory(rootHandle, moduleId, moduleFiles, state.files);
      notify('');
      state.installedModuleId = moduleId;
      refs.done.innerHTML = `<p>${esc(t('doneNote', 'Module written.', { id: moduleId }))}</p>`;
      setStage('done');
    } catch (error) {
      if (error?.name === 'AbortError') { notify(''); return; }
      notify(t('writeFailed', 'Write failed.', { error: error?.message || error }), true);
    }
  });

  refs.back.addEventListener('click', () => {
    notify('');
    setStage('source');
  });

  refs.open.addEventListener('click', async () => {
    const moduleId = state.installedModuleId;
    if (!moduleId) return;
    // The install may replace an existing module manifest as well as create a
    // new one. Reload the catalog in both cases, then let the shell launch the
    // freshly written app contract rather than a stale in-memory descriptor.
    globalThis.location.hash = moduleId;
    globalThis.location.reload();
  });

  return () => { disposed = true; void disposed; };
}
