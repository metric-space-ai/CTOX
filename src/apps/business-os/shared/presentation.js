// Canonical Business OS app presentation contract with a bounded legacy reader.

const MODES = new Set(['window', 'maximized', 'focus']);

// The desktop is the Business OS shell surface itself. It owns the landing
// surface, taskbar/dock, and global chrome; every other module is an app and
// must be hosted by the shared shell-window manager, even when its manifest is
// legacy, runtime-installed, or imported from an older catalog.
export const SHELL_SURFACE_MODULE_ID = 'desktop';
export const SHELL_WINDOW_CONTRACT = 'v2';
export const SHELL_WINDOW_GEOMETRY_CONTRACT = 'business-os-v2-global-1';

function positiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function legacyIsWindowed(moduleDef) {
  return moduleDef?.launch_kind === 'desktop-app'
    || moduleDef?.layout?.launch_kind === 'desktop-app'
    || moduleDef?.layout?.shell === 'windowed'
    || moduleDef?.layout?.shell === 'desktop-window';
}

function legacyIsFullWorkspace(moduleDef) {
  return moduleDef?.layout?.shell === 'full-workspace'
    || moduleDef?.layout?.full_workspace === true
    || moduleDef?.layout?.fullFrame === true;
}

export function isShellSurfaceModule(moduleDef) {
  return String(moduleDef?.id || '').trim() === SHELL_SURFACE_MODULE_ID;
}

export function resolvePresentation(moduleDef = {}) {
  const source = moduleDef.presentation && typeof moduleDef.presentation === 'object'
    ? moduleDef.presentation
    : {};
  const explicitMode = MODES.has(source.default_mode) ? source.default_mode : '';
  const legacyWindowed = legacyIsWindowed(moduleDef);
  const legacyFullWorkspace = legacyIsFullWorkspace(moduleDef);
  const defaultMode = explicitMode || (isShellSurfaceModule(moduleDef) ? 'workspace' : 'window');
  const requestedModes = Array.isArray(source.supported_modes)
    ? source.supported_modes.filter((mode) => MODES.has(mode))
    : [];
  const supportedModes = Array.from(new Set([
    ...(defaultMode === 'workspace' ? [] : [defaultMode]),
    ...requestedModes,
  ]));
  if (defaultMode !== 'workspace' && !supportedModes.includes('window')) {
    supportedModes.unshift('window');
  }

  return Object.freeze({
    defaultMode,
    supportedModes: Object.freeze(supportedModes),
    initialSize: Object.freeze({
      width: positiveInt(source.initial_size?.width, positiveInt(moduleDef?.layout?.default_width, 1080)),
      height: positiveInt(source.initial_size?.height, positiveInt(moduleDef?.layout?.default_height, 720)),
    }),
    minimumSize: Object.freeze({
      width: positiveInt(source.minimum_size?.width, positiveInt(moduleDef?.layout?.min_width, 640)),
      height: positiveInt(source.minimum_size?.height, positiveInt(moduleDef?.layout?.min_height, 480)),
    }),
    multiInstance: source.multi_instance === true,
    autoRestore: source.auto_restore === true,
    legacy: Object.freeze({ windowed: legacyWindowed, fullWorkspace: legacyFullWorkspace }),
  });
}

export function launchesInWindow(moduleDef) {
  return !isShellSurfaceModule(moduleDef);
}

// Window chrome is owned by the tenant-delivered Business OS shell, not by a
// module catalog snapshot.  Runtime-installed and legacy module records may
// therefore describe content layout, but they cannot downgrade the active
// shell contract.  Keeping this resolution central also makes the remaining
// v1 renderer unreachable before its follow-up removal.
export function resolveShellWindowContract(moduleDef = {}) {
  if (isShellSurfaceModule(moduleDef)) return null;
  return Object.freeze({
    contract: SHELL_WINDOW_CONTRACT,
    geometryContract: String(moduleDef?.layout?.shell_geometry_contract || '').trim()
      || SHELL_WINDOW_GEOMETRY_CONTRACT,
  });
}

export function usesLegacyWorkspace(moduleDef) {
  // The shell surface is identified by its reserved id, not by a mutable
  // catalog field. This keeps a partially projected/runtime desktop record
  // from falling through into the pane mount path.
  return isShellSurfaceModule(moduleDef);
}
