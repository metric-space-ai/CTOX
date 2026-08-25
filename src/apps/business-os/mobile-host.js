const PROTOCOL = 'workjet.business-os-shell.v1';
const CATALOG_TYPE = 'workjet.business-os-mobile-apps.v1';
const MAX_MESSAGE_BYTES = 65_536;
const SAFE_ID = /^[a-z0-9][a-z0-9._-]{0,127}$/;

if (document.documentElement.dataset.workjetMobileHost === 'true') {
  let catalog = null;
  let activeAppId = null;

  const post = (message) => {
    const raw = JSON.stringify({ protocol: PROTOCOL, ...message });
    if (new TextEncoder().encode(raw).byteLength > MAX_MESSAGE_BYTES) return;
    globalThis.workjetBusinessOsPostMessage?.(JSON.parse(raw));
  };

  const descriptor = (appId) => catalog?.apps?.find((app) => app.id === appId) || null;

  const markActiveWindow = () => {
    document.querySelectorAll('.shell-window[data-workjet-mobile-active]').forEach((element) => {
      delete element.dataset.workjetMobileActive;
    });
    if (!activeAppId) return;
    const ownerIds = [`module:${activeAppId}`, `desktop-app:${activeAppId}`, activeAppId];
    const active = [...document.querySelectorAll('.shell-window')].find((element) =>
      ownerIds.includes(element.dataset.ownerId || ''),
    );
    if (active) active.dataset.workjetMobileActive = 'true';
  };

  const publishState = (state = 'active') => {
    if (!activeAppId) return;
    const app = descriptor(activeAppId);
    post({
      type: 'app.state',
      appId: activeAppId,
      title: app?.title || activeAppId,
      canGoBack: false,
      state,
      actions: [],
    });
  };

  const openApp = (appId) => {
    if (!SAFE_ID.test(appId)) return;
    if (appId === 'desktop') {
      post({ type: 'shell.error', code: 'native-home-route', retryable: false });
      return;
    }
    const app = descriptor(appId);
    if (!app || app.desktopOnly || !app.mobilePresentation) {
      post({ type: 'shell.error', code: 'mobile-presentation-unavailable', retryable: false });
      return;
    }
    activeAppId = appId;
    document.body.dataset.workjetMobileApp = appId;
    document.body.dataset.workjetMobilePresentation = app.mobilePresentation;
    post({
      type: 'app.state',
      appId,
      title: app.title,
      canGoBack: false,
      state: 'opening',
      actions: [],
    });
    if (location.hash.replace(/^#/, '').split('?')[0] !== appId) location.hash = appId;
    queueMicrotask(markActiveWindow);
    setTimeout(() => {
      markActiveWindow();
      publishState('active');
    }, 0);
  };

  const allowedKeys = (value, keys) =>
    value && typeof value === 'object' && Object.keys(value).every((key) => keys.includes(key));

  const receive = (event) => {
    const command = event.detail;
    if (!command || command.protocol !== PROTOCOL || typeof command.type !== 'string') return;
    if (JSON.stringify(command).length > MAX_MESSAGE_BYTES) return;
    if (command.type === 'host.configure' && allowedKeys(command, ['protocol', 'type', 'platform', 'windowClass', 'colorScheme', 'reducedMotion', 'locale'])) {
      document.documentElement.dataset.workjetWindowClass = command.windowClass;
      document.documentElement.dataset.workjetPlatform = command.platform;
      return;
    }
    if (command.type === 'catalog.request' && allowedKeys(command, ['protocol', 'type'])) {
      if (catalog) post({ type: 'catalog.replace', catalog });
      return;
    }
    if (command.type === 'app.open' && allowedKeys(command, ['protocol', 'type', 'appId'])) {
      openApp(command.appId);
      return;
    }
    if (['app.close', 'app.suspend', 'app.resume'].includes(command.type) && allowedKeys(command, ['protocol', 'type', 'appId'])) {
      if (command.appId !== activeAppId) return;
      document.documentElement.dataset.workjetLifecycle = command.type.split('.')[1];
      publishState(command.type === 'app.close' ? 'closed' : command.type === 'app.suspend' ? 'suspended' : 'active');
      if (command.type === 'app.close') {
        activeAppId = null;
        delete document.body.dataset.workjetMobileApp;
        delete document.body.dataset.workjetMobilePresentation;
        markActiveWindow();
      }
      return;
    }
    if (command.type === 'navigation.back' && allowedKeys(command, ['protocol', 'type'])) {
      history.back();
      return;
    }
    if (command.type === 'action.invoke' && allowedKeys(command, ['protocol', 'type', 'appId', 'actionId'])) {
      if (command.appId !== activeAppId || !SAFE_ID.test(command.actionId)) return;
      globalThis.dispatchEvent(new CustomEvent('workjet-business-os-action', {
        detail: { appId: command.appId, actionId: command.actionId },
      }));
    }
  };

  globalThis.addEventListener('workjet-business-os-host-command', receive);
  new MutationObserver(markActiveWindow).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  fetch('./mobile-apps.json', { cache: 'no-store', credentials: 'same-origin' })
    .then((response) => {
      if (!response.ok) throw new Error('catalog unavailable');
      return response.json();
    })
    .then((value) => {
      if (value?.type !== CATALOG_TYPE || !Array.isArray(value.apps) || value.apps.length > 256) {
        throw new Error('catalog invalid');
      }
      if (value.apps.some((app) => app?.id === 'desktop')) throw new Error('catalog contains native home');
      catalog = value;
      post({ type: 'shell.ready', revision: String(value.revision || 'unknown').slice(0, 256) });
      post({ type: 'catalog.replace', catalog });
    })
    .catch(() => post({ type: 'shell.error', code: 'catalog-unavailable', retryable: true }));
}
