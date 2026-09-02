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

  const boundedSessionTransferEvent = (event) => {
    if (!event || typeof event !== 'object' || event.type !== 'workjet.session.transfer') return null;
    const text = (value, maxLength) => {
      if (typeof value !== 'string') return null;
      const normalized = value.trim();
      if (!normalized || [...normalized].length > maxLength
        || /[\u0000-\u001f\u007f]/u.test(normalized)) return null;
      return normalized;
    };
    const integer = (value) => Number.isInteger(value) && value >= 0 ? value : null;
    const bounded = {
      type: 'workjet.session.transfer',
      transferId: text(event.transferId, 160),
      sessionId: text(event.sessionId, 160),
      state: text(event.state, 64),
      fenceEpoch: integer(event.fenceEpoch),
      sourceComputerId: text(event.sourceComputerId, 256),
      targetComputerId: text(event.targetComputerId, 256),
      deadlineAtMs: integer(event.deadlineAtMs),
      updatedAtMs: integer(event.updatedAtMs),
    };
    if (Object.values(bounded).some((value) => value === null)) return null;
    return Object.freeze(bounded);
  };

  const existingWorkjetHostBridge = globalThis.workjetHostBridge;
  globalThis.workjetHostBridge = Object.freeze({
    ...(existingWorkjetHostBridge && typeof existingWorkjetHostBridge === 'object'
      ? existingWorkjetHostBridge
      : {}),
    postSessionTransferEvent(event) {
      const bounded = boundedSessionTransferEvent(event);
      if (bounded) post({ type: 'session.transfer.event', event: bounded });
    },
  });

  const descriptor = (appId) => catalog?.apps?.find((app) => app.id === appId) || null;

  const appIdForWindow = (element) => {
    const ownerId = String(element?.dataset?.ownerId || '');
    const appId = ownerId.replace(/^(?:module|desktop-app):/, '');
    return SAFE_ID.test(appId) ? appId : null;
  };

  const applyActiveAppMetadata = (appId) => {
    activeAppId = appId;
    document.body.dataset.workjetMobileApp = appId;
    const presentation = descriptor(appId)?.mobilePresentation;
    if (presentation) document.body.dataset.workjetMobilePresentation = presentation;
    else delete document.body.dataset.workjetMobilePresentation;
  };

  const markActiveWindow = () => {
    document.querySelectorAll('.shell-window[data-workjet-mobile-active]').forEach((element) => {
      delete element.dataset.workjetMobileActive;
    });
    // A focused window is authoritative only while an app is already active.
    // When Home is active, a previously focused window may still exist in the
    // shared window manager. Promoting that stale focus here used to replace
    // the freshly opened Home desk with Threads immediately.
    const focused = document.querySelector('.shell-window.is-focused');
    const focusedAppId = appIdForWindow(focused);
    if (activeAppId !== 'desktop' && focusedAppId && focusedAppId !== activeAppId) {
      applyActiveAppMetadata(focusedAppId);
      publishState('active');
    }
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
      title: app?.title || (activeAppId === 'desktop' ? 'Business OS' : activeAppId),
      canGoBack: false,
      state,
      actions: [],
    });
  };

  const openApp = (appId) => {
    if (!SAFE_ID.test(appId)) return;
    if (appId === 'desktop') {
      applyActiveAppMetadata(appId);
      if (location.hash.replace(/^#/, '').split('?')[0] !== appId) location.hash = appId;
      publishState('active');
      return;
    }
    const app = descriptor(appId);
    if (!app || app.desktopOnly || !app.mobilePresentation) {
      post({ type: 'shell.error', code: 'mobile-presentation-unavailable', retryable: false });
      return;
    }
    applyActiveAppMetadata(appId);
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
      return;
    }
    if (command.type === 'device.control' && allowedKeys(command, ['protocol', 'type', 'requestId', 'request'])) {
      if (!SAFE_ID.test(command.requestId) || typeof globalThis.workjetBusinessOsDeviceControl !== 'function') return;
      Promise.resolve(globalThis.workjetBusinessOsDeviceControl(command.request))
        .then((result) => post({ type: 'device.control.result', requestId: command.requestId, result }))
        .catch((error) => post({
          type: 'device.control.result',
          requestId: command.requestId,
          error: {
            code: String(error?.code || 'device-control-failed').slice(0, 128),
            message: String(error?.message || 'Device control failed.').slice(0, 512),
          },
        }));
      return;
    }
    if (command.type === 'project.control' && allowedKeys(command, ['protocol', 'type', 'requestId', 'request'])) {
      if (!SAFE_ID.test(command.requestId) || typeof globalThis.workjetProjectControl !== 'function') return;
      Promise.resolve(globalThis.workjetProjectControl(command.request))
        .then((result) => post({ type: 'project.control.result', requestId: command.requestId, result }))
        .catch((error) => post({
          type: 'project.control.result',
          requestId: command.requestId,
          error: {
            code: String(error?.code || 'project-control-failed').slice(0, 128),
            message: String(error?.message || 'Project control failed.').slice(0, 512),
          },
        }));
    }
    if (command.type === 'session.control' && allowedKeys(command, ['protocol', 'type', 'requestId', 'request'])) {
      if (!SAFE_ID.test(command.requestId) || typeof globalThis.workjetSessionControl !== 'function') return;
      Promise.resolve(globalThis.workjetSessionControl(command.request))
        .then((result) => post({ type: 'session.control.result', requestId: command.requestId, result }))
        .catch((error) => post({
          type: 'session.control.result',
          requestId: command.requestId,
          error: {
            code: String(error?.code || 'session-control-failed').slice(0, 128),
            message: String(error?.message || 'Session control failed.').slice(0, 512),
          },
        }));
    }
  };

  globalThis.addEventListener('workjet-business-os-host-command', receive);
  // Desktop icons open apps through the canonical CTOX launcher rather than
  // through the host bridge. Capture the actual user activation so Home can
  // remain stable in the presence of stale window-manager focus while icon
  // clicks still promote the selected app to the responsive fullscreen view.
  document.addEventListener('click', (event) => {
    if (activeAppId !== 'desktop') return;
    const icon = event.target?.closest?.('.desktop-icon[data-target]');
    const appId = String(icon?.dataset?.target || '');
    if (!SAFE_ID.test(appId)) return;
    applyActiveAppMetadata(appId);
    queueMicrotask(() => {
      markActiveWindow();
      publishState('active');
    });
  }, true);
  document.addEventListener('keydown', (event) => {
    if (activeAppId !== 'desktop' || !['Enter', ' '].includes(event.key)) return;
    const icon = event.target?.closest?.('.desktop-icon[data-target]');
    const appId = String(icon?.dataset?.target || '');
    if (!SAFE_ID.test(appId)) return;
    applyActiveAppMetadata(appId);
    queueMicrotask(() => {
      markActiveWindow();
      publishState('active');
    });
  }, true);
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
