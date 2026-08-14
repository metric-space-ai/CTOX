import { loadModuleMessages } from '../../shared/i18n.js';

const STYLE_BUILD = '20260721-browser-ia-v1';

// Module-level translator; set from locales/<lang>.json during mount.
let t = (key, fallback) => fallback ?? key;
const DEFAULT_SESSION_ID = 'browser_session_default';
const DEFAULT_TAB_ID = 'browser_tab_default';
const VIEWPORT = { width: 1280, height: 720 };
const FRAME_SYNC_RECOVERY_MS = 12000;
const POINTER_CLICK_INTERVAL_MS = 500;
const POINTER_CLICK_SLOP_PX = 4;
const POINTER_HOVER_THROTTLE_MS = 50;
const BROWSER_NAV_COMMANDS = {
  navigate: 'browser.navigate',
  back: 'browser.back',
  forward: 'browser.forward',
  reload: 'browser.reload',
};
const BROWSER_SYNC_COLLECTIONS = [
  'browser_sessions',
  'browser_tabs',
  'browser_frames',
  'browser_input_events',
];

export async function mount(ctx) {
  await ensureStyles();
  const messages = await loadModuleMessages(import.meta.url, ctx.locale).catch(() => ({}));
  t = (key, fallback) => messages[key] ?? fallback ?? key;
  const moduleUrl = new URL(import.meta.url);
  const templateUrl = new URL('./index.html', moduleUrl);
  templateUrl.search = moduleUrl.search;
  templateUrl.searchParams.set('fragment', STYLE_BUILD);
  const html = await fetch(templateUrl, { cache: 'no-store' }).then((res) => res.text());
  ctx.host.innerHTML = html;

  const root = ctx.host.querySelector('[data-browser-root]');
  if (!root) throw new Error('browser: root element missing after fragment mount');
  applyTranslations(root);

  const refs = {
    root,
    sessionCard: root.querySelector('[data-browser-session-card]'),
    sessionList: root.querySelector('[data-browser-session-list]'),
    sessionsPane: root.querySelector('.browser-sessions'),
    sessions: root.querySelector('[data-browser-sessions]'),
    sessionsEmpty: root.querySelector('[data-browser-sessions-empty]'),
    sessionsImport: root.querySelector('[data-action="import"]'),
    sessionsExport: root.querySelector('[data-action="export"]'),
    start: root.querySelector('[data-browser-start]'),
    toggleAdvanced: root.querySelector('[data-browser-toggle-advanced]'),
    advanced: root.querySelector('[data-browser-advanced]'),
    privateMode: root.querySelector('[data-browser-private]'),
    viewport: root.querySelector('[data-browser-viewport]'),
    newTab: root.querySelector('[data-browser-new-tab]'),
    upload: root.querySelector('[data-browser-upload]'),
    controllerAcquire: root.querySelector('[data-browser-controller-acquire]'),
    controllerRelease: root.querySelector('[data-browser-controller-release]'),
    observerGrant: root.querySelector('[data-browser-observer-grant]'),
    observerRevoke: root.querySelector('[data-browser-observer-revoke]'),
    clipboardCopy: root.querySelector('[data-browser-clipboard-copy]'),
    clipboardPaste: root.querySelector('[data-browser-clipboard-paste]'),
    clipboardClear: root.querySelector('[data-browser-clipboard-clear]'),
    stop: root.querySelector('[data-browser-stop]'),
    back: root.querySelector('[data-browser-back]'),
    forward: root.querySelector('[data-browser-forward]'),
    reload: root.querySelector('[data-browser-reload]'),
    sendToCtox: root.querySelector('[data-browser-send-to-ctox]'),
    notice: root.querySelector('[data-browser-notice]'),
    form: root.querySelector('[data-browser-address-form]'),
    go: root.querySelector('[data-browser-go]'),
    address: root.querySelector('[data-browser-address]'),
    addressClear: null,
    statusChip: root.querySelector('[data-browser-status-chip]'),
    statusTitle: root.querySelector('[data-browser-status-title]'),
    statusMeta: root.querySelector('[data-browser-status-meta]'),
    downloads: root.querySelector('[data-browser-downloads]'),
    authAssist: root.querySelector('[data-browser-auth-assist]'),
    shell: root.querySelector('[data-browser-frame-shell]'),
    canvas: root.querySelector('[data-browser-canvas]'),
    empty: root.querySelector('[data-browser-empty]'),
    frameId: root.querySelector('[data-browser-frame-id]'),
    frameSeq: root.querySelector('[data-browser-frame-seq]'),
    frameSize: root.querySelector('[data-browser-frame-size]'),
    frameTime: root.querySelector('[data-browser-frame-time]'),
    inputState: root.querySelector('[data-browser-input-state]'),
    commandState: root.querySelector('[data-browser-command-state]'),
    commandHistory: root.querySelector('[data-browser-command-history]'),
    handoffHistory: root.querySelector('[data-browser-handoff-history]'),
  };
  refs.addressClear = installAddressClearControl(refs.form, refs.address, refs.go);

  const requestedSessionId = browserSessionIdFromArgs(ctx.args);
  const state = {
    selectedSessionId: requestedSessionId,
    requestedSessionId,
    latestFrame: null,
    latestSession: null,
    latestTab: null,
    latestCommand: null,
    browserCommands: [],
    handoffTasks: [],
    notice: '',
    drawing: false,
    lastInputSeq: 0,
    lastPointerMoveAt: 0,
    lastPointerClick: null,
    pointerIsDown: false,
    lastFrameSyncRecoveryAt: 0,
    leaseRenewInFlight: false,
    controllerLeaseId: '',
    addressDirty: false,
    requestedSessionStarts: new Set(),
    // LEFT grammar column (SHELL-wired data-pg-*). The module only mirrors the
    // shell state to filter/label the sessions list; it wires no chrome itself.
    leftView: { search: '', view: 'cards', band: 'all', filters: {} },
    // Imported sessions are a read-only local overlay for review (browser
    // sessions are a read-only projection; the app never persists them).
    importedSessions: [],
  };

  const cleanups = [];
  let mounted = true;
  const scheduleRefresh = debounce(safeLoadAndRender, 80);
  const requestedStartTimers = new Set();

  function openRequestedBrowserSession(args, attempt = 0) {
    state.notice = 'Browser-Anmeldung wird geöffnet.';
    scheduleRefresh();
    ensureRequestedBrowserSession(ctx, state, args)
      .then(scheduleRefresh)
      .catch((error) => {
        state.notice = browserStartErrorMessage(error);
        scheduleRefresh();
        if (!browserStartErrorIsRetryable(error) || attempt >= 3 || !mounted) return;
        const timer = globalThis.setTimeout(() => {
          requestedStartTimers.delete(timer);
          openRequestedBrowserSession(args, attempt + 1);
        }, Math.min(8_000, 1_000 * (2 ** attempt)));
        requestedStartTimers.add(timer);
      });
  }

  const sessionSelectionToken = ctx.eventBus?.on?.('browser:select-session', (detail = {}) => {
    const sessionId = browserSessionIdFromArgs(detail);
    if (!sessionId) return;
    if (sessionId !== state.selectedSessionId) state.controllerLeaseId = '';
    state.selectedSessionId = sessionId;
    state.requestedSessionId = sessionId;
    scheduleRefresh();
    openRequestedBrowserSession(detail);
  });
  if (sessionSelectionToken && ctx.eventBus?.off) {
    cleanups.push(() => ctx.eventBus.off('browser:select-session', sessionSelectionToken));
  }
  const handleFocusRefresh = () => {
    scheduleRefresh();
    renewControllerLeaseIfNeeded();
  };
  const focusRefreshToken = ctx.eventBus?.on?.('window:focused', handleFocusRefresh);
  if (focusRefreshToken && ctx.eventBus?.off) {
    cleanups.push(() => ctx.eventBus.off('window:focused', focusRefreshToken));
  }
  globalThis.addEventListener?.('focus', handleFocusRefresh);
  globalThis.addEventListener?.('blur', scheduleRefresh);
  globalThis.document?.addEventListener?.('visibilitychange', handleFocusRefresh);
  cleanups.push(() => {
    globalThis.removeEventListener?.('focus', handleFocusRefresh);
    globalThis.removeEventListener?.('blur', scheduleRefresh);
    globalThis.document?.removeEventListener?.('visibilitychange', handleFocusRefresh);
  });

  for (const collectionName of ['business_commands', ...BROWSER_SYNC_COLLECTIONS, 'ctox_queue_tasks']) {
    ctx.sync?.startCollection?.(collectionName)
      ?.catch?.((error) => console.warn(`[browser] ${collectionName} sync start failed`, error));
  }

  for (const collection of [
    browserCollection(ctx, 'business_commands'),
    browserCollection(ctx, 'browser_sessions'),
    browserCollection(ctx, 'browser_tabs'),
    browserCollection(ctx, 'browser_frames'),
    browserCollection(ctx, 'browser_input_events'),
    browserCollection(ctx, 'ctox_queue_tasks'),
  ]) {
    const sub = collection?.$?.subscribe?.(() => scheduleRefresh());
    if (sub?.unsubscribe) cleanups.push(() => sub.unsubscribe());
  }

  // LEFT column grammar is SHELL-wired (search / shard-list toggle / filter
  // tray / reset / active-dot / counted band). The module re-renders the
  // sessions list on the bubbling change event; no manual refresh button —
  // the sessions list is reactive (collection subscriptions above).
  const onLeftGrammarChange = (event) => {
    const detail = event?.detail || refs.sessionsPane?.__ctoxPaneGrammar?.state?.() || {};
    state.leftView = {
      search: String(detail.search || '').trim().toLowerCase(),
      view: detail.view === 'list' ? 'list' : 'cards',
      band: detail.band || 'all',
      filters: detail.filters || {},
    };
    renderSessions(refs, sessionRenderList(state), state.latestSession, state.leftView, sessionTabCounts(state.tabs), ctx);
  };
  root.addEventListener('ctox-pane-grammar-change', onLeftGrammarChange);
  cleanups.push(() => root.removeEventListener('ctox-pane-grammar-change', onLeftGrammarChange));
  refs.sessionsImport?.addEventListener('click', () => importBrowserSessions(ctx, state, refs));
  refs.sessionsExport?.addEventListener('click', () => exportBrowserSessions(state, refs));
  refs.toggleAdvanced?.addEventListener('click', () => {
    if (!refs.advanced) return;
    const hidden = refs.advanced.classList.toggle('is-advanced-hidden');
    refs.toggleAdvanced.setAttribute('aria-pressed', hidden ? 'false' : 'true');
  });
  const startNewBrowserSession = (url = refs.address?.value || 'https://example.com') => {
    const now = Date.now();
    const sessionId = `${userSessionPrefix(ctx.session)}_${now}`;
    const tabId = `browser_tab_${now}`;
    const viewport = selectedViewport(refs.viewport);
    console.info(`[browser] session start clicked session_id=${sessionId} tab_id=${tabId}`);
    state.addressDirty = false;
    state.selectedSessionId = sessionId;
    state.requestedSessionId = sessionId;
    state.controllerLeaseId = newBrowserControllerLeaseId();
    state.notice = 'Browser wird mit CTOX verbunden …';
    safeLoadAndRender();
    const command = () => dispatchBrowserCommand(ctx, state, 'browser.session.start', {
      session_id: sessionId,
      tab_id: tabId,
      url,
      viewport_w: viewport.width,
      viewport_h: viewport.height,
      profile_mode: refs.privateMode?.checked ? 'private' : 'persistent',
      lease_id: state.controllerLeaseId,
      new_session: true,
    });
    runBrowserCommand(command());
  };
  refs.start?.addEventListener('click', () => startNewBrowserSession());
  refs.stop?.addEventListener('click', () => dispatchBrowserCommand(ctx, state, 'browser.session.stop').then(safeLoadAndRender));
  refs.newTab?.addEventListener('click', () => {
    const now = Date.now();
    dispatchBrowserCommand(ctx, state, 'browser.tab.open', {
      tab_id: `browser_tab_${now}`,
      url: refs.address?.value || 'https://example.com',
    }).then(safeLoadAndRender);
  });
  refs.upload?.addEventListener('click', () => {
    const fileId = globalThis.prompt('CTOX Datei-ID für den Upload');
    if (!fileId) return;
    dispatchBrowserCommand(ctx, state, 'browser.upload.select', { file_id: fileId.trim() }).then(safeLoadAndRender);
  });
  refs.controllerAcquire?.addEventListener('click', () => {
    const leaseId = newBrowserControllerLeaseId();
    state.controllerLeaseId = leaseId;
    runBrowserCommand(
      dispatchBrowserCommand(ctx, state, 'browser.controller.acquire', { lease_id: leaseId })
        .catch((error) => {
          if (state.controllerLeaseId === leaseId) state.controllerLeaseId = '';
          throw error;
        }),
    );
  });
  refs.controllerRelease?.addEventListener('click', () => {
    const leaseId = state.controllerLeaseId;
    runBrowserCommand(
      dispatchBrowserCommand(ctx, state, 'browser.controller.release')
        .then((result) => {
          if (state.controllerLeaseId === leaseId) state.controllerLeaseId = '';
          return result;
        }),
    );
  });
  refs.observerGrant?.addEventListener('click', () => {
    const userId = globalThis.prompt('Benutzer-ID des Beobachters');
    if (!userId) return;
    dispatchBrowserCommand(ctx, state, 'browser.observer.grant', { user_id: userId.trim() }).then(safeLoadAndRender);
  });
  refs.observerRevoke?.addEventListener('click', () => {
    const userId = globalThis.prompt('Benutzer-ID des zu entfernenden Beobachters');
    if (!userId) return;
    dispatchBrowserCommand(ctx, state, 'browser.observer.revoke', { user_id: userId.trim() }).then(safeLoadAndRender);
  });
  refs.clipboardCopy?.addEventListener('click', () => dispatchBrowserCommand(ctx, state, 'browser.clipboard.copy', { confirmed: true }).then(safeLoadAndRender));
  refs.clipboardPaste?.addEventListener('click', () => dispatchBrowserCommand(ctx, state, 'browser.clipboard.paste', { confirmed: true }).then(safeLoadAndRender));
  refs.clipboardClear?.addEventListener('click', () => dispatchBrowserCommand(ctx, state, 'browser.clipboard.clear', { confirmed: true }).then(safeLoadAndRender));
  refs.back?.addEventListener('click', () => runBrowserCommand(submitBrowserNav(ctx, state, refs, 'back')));
  refs.forward?.addEventListener('click', () => runBrowserCommand(submitBrowserNav(ctx, state, refs, 'forward')));
  refs.reload?.addEventListener('click', () => runBrowserCommand(submitBrowserNav(ctx, state, refs, 'reload')));
  refs.address?.addEventListener('input', () => {
    state.addressDirty = true;
  });
  refs.address?.addEventListener('focus', (event) => {
    event.currentTarget?.select?.();
  });
  refs.addressClear?.addEventListener('click', () => {
    if (!refs.address) return;
    refs.address.value = '';
    state.addressDirty = true;
    refs.address.focus();
  });
  refs.sendToCtox?.addEventListener('click', () => sendBrowserContextToCtox(ctx, state).then(safeLoadAndRender));
  refs.authAssist?.addEventListener('click', (event) => {
    const permissionButton = event.target?.closest?.('[data-browser-permission-response]');
    if (permissionButton) {
      dispatchBrowserCommand(ctx, state, 'browser.permission.respond', {
        accept: permissionButton.dataset.browserPermissionResponse === 'accept',
        confirmed: true,
      }).then(safeLoadAndRender);
      return;
    }
    const httpAuthButton = event.target?.closest?.('[data-browser-http-auth-response]');
    if (httpAuthButton) {
      const accept = httpAuthButton.dataset.browserHttpAuthResponse === 'accept';
      const secretName = accept ? globalThis.prompt('CTOX Secret-Referenz für HTTP-Auth') : '';
      if (accept && !secretName) return;
      dispatchBrowserCommand(ctx, state, 'browser.http_auth.respond', {
        accept,
        confirmed: true,
        secret_name: secretName?.trim?.() || '',
      }).then(safeLoadAndRender);
      return;
    }
    const webAuthnButton = event.target?.closest?.('[data-browser-webauthn-response]');
    if (webAuthnButton) {
      dispatchBrowserCommand(ctx, state, 'browser.webauthn.respond', {
        accept: webAuthnButton.dataset.browserWebauthnResponse === 'accept',
        confirmed: true,
      }).then(safeLoadAndRender);
      return;
    }
    const dialogButton = event.target?.closest?.('[data-browser-dialog-response]');
    if (dialogButton) {
      const accept = dialogButton.dataset.browserDialogResponse === 'accept';
      const dialog = state.latestSession?.payload?.pending_dialog;
      const value = accept && dialog?.type === 'prompt'
        ? globalThis.prompt(dialog.message || 'Eingabe', dialog.default_value || '')
        : undefined;
      dispatchBrowserCommand(ctx, state, 'browser.dialog.respond', { accept, value }).then(safeLoadAndRender);
      return;
    }
    const fillButton = event.target?.closest?.('[data-browser-credential-fill]');
    if (fillButton) {
      fillWebStackCredential(ctx, state).then(safeLoadAndRender);
      return;
    }
    const completeButton = event.target?.closest?.('[data-browser-auth-complete]');
    if (completeButton) {
      completeWebStackAuthAssist(ctx, state).then(safeLoadAndRender);
      return;
    }
    const captureButton = event.target?.closest?.('[data-browser-web-stack-capture]');
    if (captureButton) sendBrowserContextToCtox(ctx, state, { webStack: true }).then(safeLoadAndRender);
    const extractButton = event.target?.closest?.('[data-browser-web-stack-extract]');
    if (extractButton) extractWebStackFields(ctx, state).then(safeLoadAndRender);
  });
  refs.downloads?.addEventListener('click', (event) => {
    const action = event.target?.closest?.('[data-browser-download-action]');
    if (!action) return;
    dispatchBrowserCommand(ctx, state, `browser.download.${action.dataset.browserDownloadAction}`, {
      download_id: action.dataset.browserDownloadId || '',
    }).then(safeLoadAndRender);
  });
  refs.sessionList?.addEventListener('click', (event) => {
    const tabItem = event.target?.closest?.('[data-browser-tab-id]');
    if (tabItem) {
      const tabId = tabItem.dataset.browserTabId || '';
      const commandType = event.target?.closest?.('[data-browser-tab-close]')
        ? 'browser.tab.close'
        : 'browser.tab.activate';
      dispatchBrowserCommand(ctx, state, commandType, { tab_id: tabId }).then(safeLoadAndRender);
      return;
    }
    const item = event.target?.closest?.('[data-browser-session-id]');
    if (!item) return;
    state.selectedSessionId = item.dataset.browserSessionId || '';
    safeLoadAndRender();
  });
  // LEFT sessions selector. Selecting a session is an in-place is-selected flip
  // across existing rows — it must NOT rebuild the list (a rebuild would clamp
  // the well's scrollTop to 0 and yank the operator to the top). Only the data
  // that drives the MAIN canvas changes, so the reactive load re-renders the
  // work surface while the signature guard leaves the list markup untouched.
  refs.sessions?.addEventListener('click', (event) => {
    const item = event.target?.closest?.('[data-browser-session-id]');
    if (!item) return;
    const sessionId = item.dataset.browserSessionId || '';
    if (sessionId === state.selectedSessionId) return;
    if (sessionId !== state.selectedSessionId) state.controllerLeaseId = '';
    state.selectedSessionId = sessionId;
    state.requestedSessionId = '';
    markActiveSession(refs, sessionId);
    safeLoadAndRender();
  });
  const submitAddress = () => {
    const canNavigate = browserSurfaceCanControl(ctx, state);
    const url = refs.address?.value || 'https://example.com';
    if (!canNavigate) {
      startNewBrowserSession(url);
      return;
    }
    state.addressDirty = false;
    state.notice = 'Browser wird mit CTOX verbunden …';
    safeLoadAndRender();
    runBrowserCommand(submitBrowserNav(ctx, state, refs, 'navigate', { url }));
  };
  refs.form?.addEventListener('submit', (event) => {
    event.preventDefault();
    submitAddress();
  });
  installInputHandlers(ctx, refs, state, scheduleRefresh);
  const leaseRenewTimer = globalThis.setInterval(renewControllerLeaseIfNeeded, 30_000);
  cleanups.push(() => globalThis.clearInterval(leaseRenewTimer));
  safeLoadAndRender();
  openRequestedBrowserSession(ctx.args);

  return () => {
    mounted = false;
    for (const timer of requestedStartTimers) globalThis.clearTimeout(timer);
    requestedStartTimers.clear();
    for (const cleanup of cleanups) {
      try { cleanup(); } catch (error) { console.error('[browser] cleanup failed', error); }
    }
    ctx.host.replaceChildren();
  };

  function renewControllerLeaseIfNeeded() {
    const session = state.latestSession;
    const actorIds = browserActorIds(ctx.session);
    const surface = ctx.host?.closest?.('.shell-window');
    if (!shouldRenewControllerLease(session, actorIds, Date.now(), {
      documentVisible: globalThis.document?.visibilityState !== 'hidden',
      documentFocused: globalThis.document?.hasFocus?.() !== false,
      surfaceFocused: Boolean(surface?.classList.contains('is-focused')),
      renewInFlight: state.leaseRenewInFlight,
      controllerLeaseId: state.controllerLeaseId,
    })) {
      // Abgelaufen? Dann nicht erneuern, sondern NEU HOLEN. Ohne diesen Zweig
      // bleibt die Sitzung fuer immer unbedienbar, waehrend sie "Bereit" meldet.
      versucheSteuerungZurueckzuholen();
      return;
    }
    state.leaseRenewInFlight = true;
    dispatchBrowserCommand(ctx, state, 'browser.controller.renew', {
      lease_id: state.controllerLeaseId,
    })
      .catch((error) => {
        console.warn('[browser] controller lease renewal failed', error);
      })
      .finally(() => {
        state.leaseRenewInFlight = false;
      });
  }

  function versucheSteuerungZurueckzuholen() {
    const session = state.latestSession;
    if (!shouldReacquireControllerLease(session, browserActorIds(ctx.session), Date.now(), {
      reacquireInFlight: state.leaseReacquireInFlight,
      lastReacquireAtMs: state.lastReacquireAtMs,
    })) return;
    state.leaseReacquireInFlight = true;
    state.lastReacquireAtMs = Date.now();
    const leaseId = newBrowserControllerLeaseId();
    state.controllerLeaseId = leaseId;
    console.info('[browser] Steuerung abgelaufen — hole sie zurück', { session: session?.id });
    dispatchBrowserCommand(ctx, state, 'browser.controller.acquire', { lease_id: leaseId })
      .then(() => {
        setzeEingabeHinweis(ctx, state, '');
        safeLoadAndRender();
      })
      .catch((error) => {
        if (state.controllerLeaseId === leaseId) state.controllerLeaseId = '';
        console.warn('[browser] Steuerung konnte nicht zurückgeholt werden', error);
        setzeEingabeHinweis(ctx, state, 'Steuerung abgelaufen und konnte nicht zurückgeholt werden.');
      })
      .finally(() => {
        state.leaseReacquireInFlight = false;
      });
  }

  function safeLoadAndRender() {
    loadAndRender().catch((error) => console.warn('[browser] refresh failed', error));
  }

  function runBrowserCommand(promise) {
    return promise
      .then((result) => {
        if (result?.opensNewSession && result.sessionId) {
          state.selectedSessionId = result.sessionId;
          state.requestedSessionId = result.sessionId;
        }
        state.notice = '';
        safeLoadAndRender();
      })
      .catch((error) => {
        state.notice = browserStartErrorMessage(error);
        console.warn('[browser] command failed', error);
        safeLoadAndRender();
      });
  }

  async function loadAndRender() {
    if (!mounted) return;
    const [commands, sessions, requestedSession, initialTabs, inputs, handoffTasks] = await Promise.all([
      readCollection(browserCollection(ctx, 'business_commands'), { limit: 50 }),
      readCollection(browserCollection(ctx, 'browser_sessions'), { limit: 200 }),
      readDocument(browserCollection(ctx, 'browser_sessions'), state.requestedSessionId),
      readCollection(browserCollection(ctx, 'browser_tabs'), { limit: 40 }),
      readCollection(browserCollection(ctx, 'browser_input_events'), { limit: 80 }),
      readCollection(browserCollection(ctx, 'ctox_queue_tasks'), { limit: 50 }),
    ]);
    const actorIds = browserActorIds(ctx.session);
    const visibleSessions = mergeRequestedSession(sessions, requestedSession)
      .filter((session) => actorIds.includes(String(session.owner_user_id || '')));
    if (state.selectedSessionId
      && state.selectedSessionId !== state.requestedSessionId
      && !visibleSessions.some((session) => session.id === state.selectedSessionId)) {
      state.selectedSessionId = '';
    }
    const selectedSession = state.selectedSessionId ? latestSession(visibleSessions, state.selectedSessionId) : null;
    const requestedTab = await readDocument(
      browserCollection(ctx, 'browser_tabs'),
      selectedSession?.current_tab_id || requestedSession?.current_tab_id || '',
    );
    const tabs = mergeRequestedDocument(initialTabs, requestedTab);
    const requestedSessionPending = Boolean(state.requestedSessionId && !selectedSession);
    const frameSessionId = selectedSession?.id || state.selectedSessionId || '';
    const frames = frameSessionId
      ? await readCollection(browserCollection(ctx, 'browser_frames'), {
        limit: 20,
        selector: { session_id: frameSessionId },
      })
      : [];
    if (!mounted) return;
    const newestFrame = latestFrame(frames);
    state.latestSession = requestedSessionPending
      ? null
      : selectedSession || latestSession(visibleSessions, newestFrame?.session_id) || latestSession(visibleSessions);
    if (!requestedSessionPending) {
      state.selectedSessionId = state.latestSession?.id || '';
      if (state.latestSession?.id === state.requestedSessionId
        && browserSessionIsLive(state.latestSession)) {
        state.requestedSessionId = '';
      }
    }
    state.latestFrame = latestFrame(frames, state.latestSession?.id);
    state.latestTab = latestTab(tabs, state.latestFrame?.tab_id || state.latestSession?.current_tab_id);
    state.latestCommand = latestBrowserCommand(commands, state.latestSession?.id || state.latestFrame?.session_id);
    applyLatestNavigationResult(state, commands);
    state.browserCommands = latestBrowserCommands(commands, state.latestSession?.id || state.latestFrame?.session_id, 5);
    state.handoffTasks = latestBrowserHandoffTasks(handoffTasks, state.latestSession?.id || state.latestFrame?.session_id, 5);
    state.lastInputSeq = Math.max(
      Number(state.lastInputSeq || 0),
      ...inputs.map((event) => Number(event.seq || 0)).filter(Number.isFinite),
    );
    const renderedTabs = state.latestTab?.id
      ? tabs.map((tab) => tab.id === state.latestTab.id ? { ...tab, ...state.latestTab } : tab)
      : tabs;
    state.visibleSessions = visibleSessions;
    state.tabs = tabs;
    // Left = persistent sessions selector; syncs from the shell grammar state
    // (may not be wired yet on the first paint — that is null-guarded).
    const grammar = refs.sessionsPane?.__ctoxPaneGrammar?.state?.();
    if (grammar) {
      state.leftView = {
        search: String(grammar.search || '').trim().toLowerCase(),
        view: grammar.view === 'list' ? 'list' : 'cards',
        band: grammar.band || 'all',
        filters: grammar.filters || {},
      };
    }
    renderSessions(refs, sessionRenderList(state), state.latestSession, state.leftView, sessionTabCounts(state.tabs), ctx);
    // Auto-reveal: the remote work surface is meaningful once a session is
    // selected (visible = hasSelection && !userCollapsed). No session -> the
    // canvas shows its empty state instead of stale chrome.
    refs.root.classList.toggle('is-session-active', browserWorkbenchVisible(Boolean(state.latestSession?.id)));
    renderSessionList(refs, visibleSessions, renderedTabs, state.latestSession);
    renderSession(refs, state.latestSession, state.latestTab, state.latestFrame, state.latestCommand, state);
    renderAuthAssist(refs, state.latestSession);
    renderStatus(refs, state.latestSession, state.latestTab, state.latestFrame, state.latestCommand);
    renderDownloads(refs, state.latestSession);
    renderDiagnostics(refs, state.latestFrame, inputs, state.latestCommand, state.browserCommands, state.handoffTasks);
    renderControls(ctx, refs, state);
    renderNotice(refs, state.notice);
    recoverFrameSyncIfNeeded(ctx, state);
    await renderFrame(refs, state.latestFrame, state);
  }
}

function recoverFrameSyncIfNeeded(ctx, state) {
  if (!ctx.sync) return;
  if (state.latestFrame?.data) return;
  const status = String(state.latestSession?.runtime_status || state.latestSession?.status || '').toLowerCase();
  const commandStatus = String(state.latestCommand?.status || state.latestCommand?.task_status || '').toLowerCase();
  const browserIsExpected = ['active', 'running', 'requested', 'pending_command', 'starting'].includes(status)
    || ['pending_sync', 'accepted', 'completed'].includes(commandStatus);
  if (!browserIsExpected) return;
  const now = Date.now();
  if (now - Number(state.lastFrameSyncRecoveryAt || 0) < FRAME_SYNC_RECOVERY_MS) return;
  state.lastFrameSyncRecoveryAt = now;
  for (const collectionName of BROWSER_SYNC_COLLECTIONS) {
    ctx.sync.restartCollection?.(collectionName)
      ?.catch?.((error) => console.warn(`[browser] ${collectionName} sync restart failed`, error));
  }
}

async function submitBrowserNav(ctx, state, refs, action, extra = {}) {
  const commandType = BROWSER_NAV_COMMANDS[action];
  if (!commandType) throw new Error('Unknown browser navigation action: ' + action);
  const payload = action === 'navigate'
    ? { url: extra.url || refs.address?.value || 'https://example.com' }
    : {};
  return dispatchBrowserCommand(ctx, state, commandType, payload);
}

// SPUR auf dem Befehlsweg. Am 13.08.2026 blieb der Sitzungsstart aus, ohne dass
// irgendwo etwas erschien: kein Chrome-Prozess, keine Protokollzeile im Dienst,
// alle Sitzungen 'disconnected' OHNE Fehlereintrag. Drei Reparaturen (Pacht,
// Startmerkliste, Meldungstrennung) haben das Symptom nicht aufgeloest.
// Statt einen vierten Fix zu raten: messen, wo die Kette reisst.
// Ein Vorgang, der nichts protokolliert, ist von einem Vorgang, der nie
// stattfindet, nicht zu unterscheiden.
function spurBefehl(schritt, commandType, extra = {}) {
  try {
    console.info(`[browser][SPUR] ${schritt} type=${commandType} ${JSON.stringify(extra)}`);
  } catch { console.info(`[browser][SPUR] ${schritt} type=${commandType}`); }
}

async function dispatchBrowserCommand(ctx, state, commandType, payloadPatch = {}) {
  spurBefehl('1-aufgerufen', commandType, {
    sitzung: state.latestSession?.id || '-',
    status: state.latestSession?.status || '-',
    neueSitzung: payloadPatch.new_session === true,
  });
  const requiresController = browserCommandRequiresController(commandType, state.latestSession);
  if (requiresController && !browserSurfaceCanControl(ctx, state)) {
    spurBefehl('2-ABBRUCH-keine-steuerung', commandType, {
      grund: eingabeSperrgrund(ctx, state) || '(unbekannt)',
    });
    throw new Error('Dieses Browser-Fenster ist nicht aktiv. Aktivieren Sie das Fenster oder übernehmen Sie die Steuerung.');
  }
  spurBefehl('2-steuerung-ok', commandType, { brauchtSteuerung: requiresController });
  const now = Date.now();
  const opensNewSession = payloadPatch.new_session === true;
  const requestedSessionId = browserSessionIdFromArgs(payloadPatch);
  const sessionId = requestedSessionId || (opensNewSession
    ? `${userSessionPrefix(ctx.session)}_${now}`
    : state.latestSession?.id || `${userSessionPrefix(ctx.session)}_default`);
  const tabId = String(payloadPatch.tab_id || (opensNewSession
    ? `browser_tab_${now}`
    : state.latestTab?.id || state.latestSession?.current_tab_id || DEFAULT_TAB_ID));
  const commandId = `browser_cmd_${now}_${Math.random().toString(36).slice(2, 10)}`;
  const payload = {
    session_id: sessionId,
    tab_id: tabId,
    viewport_w: VIEWPORT.width,
    viewport_h: VIEWPORT.height,
    ...payloadPatch,
  };
  if (requiresController) payload.lease_id = state.controllerLeaseId;
  delete payload.new_session;
  if (payload.url) payload.url = normalizeUrl(payload.url);
  // Browser commands must not race the command collection during a freshly
  // mounted window. Only await the command path here; awaiting all browser
  // projections can deadlock while an existing peer is resyncing.
  const command = {
    id: commandId,
    command_id: commandId,
    module: 'browser',
    command_type: commandType,
    type: commandType,
    record_id: sessionId,
    inbound_channel: 'browser',
    status: 'pending_sync',
    payload,
    client_context: {
      source: 'business-os.browser.runtime',
      module_id: 'browser',
      actor: actorContext(ctx.session),
      handled_by: 'native-rxdb-peer',
    },
    created_at_ms: now,
    updated_at_ms: now,
    sync_queue_tasks: false,
  };
  spurBefehl('3-befehl-gebaut', commandType, { commandId, sessionId, tabId });
  const commandBus = requireCommandBus(ctx);
  spurBefehl('4-befehlsbus-da', commandType, { busDa: !!commandBus });
  const waitsForRuntime = [
    'browser.session.start',
    'browser.navigate',
    'browser.reload',
    'browser.back',
    'browser.forward',
    'browser.reset',
  ].includes(commandType);
  let commandSyncPreflightError = null;
  try {
    await startCommandSync(ctx);
  } catch (error) {
    // Do not let a stale module-owned bridge drop the user action. The command
    // bus performs its own scoped readiness check and confirmed push below.
    commandSyncPreflightError = error;
    console.warn(
      `[browser] command sync preflight failed; continuing to command bus command_type=${commandType} command_id=${commandId} session_id=${sessionId}`,
      error,
    );
  }
  console.info(`[browser] submitting command command_type=${commandType} command_id=${commandId} session_id=${sessionId}`);
  try {
    // Compatibility contract for non-runtime commands:
    // commandBus.dispatch(command, { until: 'accepted' })
    // The command bus owns collection readiness, capability acquisition and
    // the confirmed RxDB push. A separate startCollection() preflight here
    // used to abort before dispatch whenever the mounted window's bridge was
    // stale, so the native service never saw the session-start command.
    await commandBus.dispatch(command, {
      until: waitsForRuntime ? 'terminal' : 'accepted',
      ...(waitsForRuntime ? { timeoutMs: 150_000 } : {}),
    });
  } catch (error) {
    console.error(
      `[browser] command submission failed command_type=${commandType} command_id=${commandId} session_id=${sessionId}`,
      error,
    );
    if (commandSyncPreflightError && !error.cause) {
      error.cause = commandSyncPreflightError;
    }
    throw error;
  }
  if (waitsForRuntime) await refreshBrowserProjections(ctx);
  return { commandId, sessionId, tabId, opensNewSession };
}

async function ensureRequestedBrowserSession(ctx, state, args = {}) {
  const request = browserAuthRequestFromArgs(args);
  if (!request) return false;
  state.selectedSessionId = request.session_id;
  state.requestedSessionId = request.session_id;
  // Die Merkliste verhindert, dass derselbe Start doppelt losgeschickt wird,
  // solange er noch laeuft. Sie darf aber nicht dauerhaft merken: eine Sitzung,
  // die einmal erfolgreich startete und spaeter wegbricht (Dienstneustart,
  // Prozessende), blieb bis 13.08.2026 fuer immer in der Liste — der Eintrag
  // wurde nur im FEHLERfall entfernt. Danach wurde jeder weitere Startversuch
  // stumm verworfen.
  // Auf der Kundeninstanz gemessen: 34 Sitzungen, davon 0 aktiv, kein
  // Chrome-Prozess, keine Protokollzeile in 20 Minuten — und weder der
  // Start-Knopf noch das Plus-Symbol bewirkten etwas. Es lief vorher; nach
  // einem Dienstneustart nie wieder.
  // Deshalb zuerst den Zustand lesen, dann die Merkliste bewerten: was nicht
  // mehr laeuft, ist auch nicht mehr "in Arbeit".
  const existing = await browserCollection(ctx, 'browser_sessions')?.findOne(request.session_id).exec();
  const existingData = existing?.toJSON?.() || existing;
  if (!browserSessionNeedsStart(existingData)) {
    // Laeuft bereits — Merkliste aufraeumen, damit ein spaeteres Wegbrechen
    // wieder startbar ist.
    state.requestedSessionStarts.delete(request.session_id);
    return false;
  }
  if (state.requestedSessionStarts.has(request.session_id)) {
    console.info('[browser] Start bereits angefordert, warte auf Ergebnis', request.session_id);
    return false;
  }

  state.requestedSessionStarts.add(request.session_id);
  try {
    state.controllerLeaseId = newBrowserControllerLeaseId();
    await dispatchBrowserCommand(ctx, state, 'browser.session.start', {
      ...request,
      lease_id: state.controllerLeaseId,
    });
    return true;
  } catch (error) {
    state.requestedSessionStarts.delete(request.session_id);
    throw error;
  }
}

function browserAuthRequestFromArgs(value) {
  const sessionId = browserSessionIdFromArgs(value);
  const purpose = String(value?.purpose || '').trim();
  const targetUrl = String(value?.target_url || value?.targetUrl || '').trim();
  if (!sessionId || purpose !== 'web_stack_auth' || !targetUrl) return null;
  const tabId = String(value?.tab_id || value?.tabId || `browser_tab_${sessionId}`).trim();
  const sourceId = String(value?.source_id || value?.sourceId || '').trim();
  const allowedDomains = Array.isArray(value?.allowed_domains)
    ? value.allowed_domains.map((entry) => String(entry || '').trim()).filter(Boolean)
    : [];
  const captureScript = String(value?.capture_script || value?.captureScript || '').trim();
  const secretName = String(value?.secret_name || value?.required_secret_name || '').trim();
  return {
    session_id: sessionId,
    tab_id: tabId,
    url: targetUrl,
    target_url: targetUrl,
    source_id: sourceId,
    purpose,
    allowed_domains: allowedDomains,
    capture_script: captureScript,
    secret_name: secretName,
    auth_assist_status: 'pending',
    profile_mode: 'persistent',
    secret_value_in_rxdb: false,
  };
}

function browserStartErrorMessage(error) {
  const code = String(error?.code || '').toLowerCase();
  if (code === 'auth_required') return 'Die Browser-Anmeldung benötigt eine neue Business-OS-Autorisierung.';
  if (code === 'native_unavailable') return 'Die Browser-Anmeldung konnte noch nicht mit CTOX verbunden werden. Bitte erneut versuchen.';
  // Zwei voellig verschiedene Fehler trugen bis 13.08.2026 dieselbe Meldung.
  // sync_unavailable heisst "keine Verbindung". projection_delayed heisst
  // "Verbindung steht, die Antwort kam nur nicht rechtzeitig zurueck".
  // Auf der Kundeninstanz gemessen: alle fuenf Browser-Collections
  // 'connected' — und trotzdem stand da "CTOX ist nicht mit dem
  // Browser-Datenkanal verbunden". Die Meldung behauptete das Gegenteil des
  // gemessenen Zustands und schickte den Nutzer auf die falsche Faehrte
  // ("Verbindung erneut aufbauen"), waehrend in Wahrheit nur die Antwort
  // ausstand. Eine Zeitueberschreitung ist kein Ausfall.
  if (code === 'sync_unavailable') {
    return 'CTOX ist nicht mit dem Browser-Datenkanal verbunden. Der Browser wurde nicht gestartet. Bitte die Verbindung erneut aufbauen und dann erneut versuchen.';
  }
  if (code === 'projection_delayed') {
    return 'CTOX hat den Befehl angenommen, aber noch nicht bestätigt. Die Verbindung steht — der Vorgang läuft möglicherweise noch. Bitte einen Moment warten und die Ansicht neu laden, bevor Sie es erneut versuchen.';
  }
  const message = String(error?.message || '').trim();
  return message
    ? `Die Browser-Anmeldung konnte nicht geöffnet werden: ${message}`
    : 'Die Browser-Anmeldung konnte nicht geöffnet werden.';
}

function browserSessionError(session) {
  const direct = String(session?.last_error || session?.error || '').trim();
  if (direct) return direct;
  const status = String(session?.runtime_status || session?.status || '').toLowerCase();
  if (['active', 'running', 'capturing'].includes(status)) return '';
  return String(session?.payload?.last_error || session?.payload?.error || '').trim();
}

function browserSessionIsLive(session) {
  const status = String(session?.runtime_status || session?.status || '').toLowerCase();
  return status === 'active';
}

function browserSessionNeedsStart(session) {
  if (!session?.id) return true;
  const status = String(session.runtime_status || session.status || '').toLowerCase();
  return !['active', 'starting'].includes(status);
}

function browserStartErrorIsRetryable(error) {
  const code = String(error?.code || '').toLowerCase();
  return ['projection_delayed', 'sync_unavailable', 'native_unavailable'].includes(code);
}

function userSessionPrefix(session) {
  const raw = String(session?.user?.id || session?.userId || 'browser-user');
  const safe = raw.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64);
  return `browser_session_${safe || 'user'}`;
}

function selectedViewport(select) {
  const [width, height] = String(select?.value || '1280x720').split('x').map(Number);
  return {
    width: Number.isFinite(width) ? Math.max(320, Math.min(3840, width)) : VIEWPORT.width,
    height: Number.isFinite(height) ? Math.max(240, Math.min(2160, height)) : VIEWPORT.height,
  };
}

async function sendBrowserContextToCtox(ctx, state, options = {}) {
  const session = state.latestSession;
  const tab = state.latestTab;
  const frame = state.latestFrame;
  if (!session?.id) {
    state.notice = 'Kein Browser-Fenster zum Senden geoeffnet.';
    return;
  }
  const payloadMetadata = session.payload || {};
  const now = Date.now();
  const url = tab?.url || session.current_url || '';
  const title = browserDisplayTitle(tab, session, url);
  const sourceId = payloadMetadata.source_id || '';
  const captureScript = payloadMetadata.capture_script || '';
  const verifySelector = payloadMetadata.verify_selector || '';
  const purpose = payloadMetadata.purpose || '';
  const sourceModule = options.webStack ? 'web_stack' : 'browser';
  const browserContext = {
    session_id: session.id,
    tab_id: tab?.id || session.current_tab_id || '',
    url,
    title,
    status: session.runtime_status || session.status || '',
    purpose,
    source_id: sourceId,
    capture_script: captureScript,
    verify_selector: verifySelector,
    frame_id: frame?.id || '',
    frame_seq: frame?.seq || 0,
    frame_captured_at_ms: frame?.captured_at_ms || 0,
    frame_expires_at_ms: frame?.expires_at_ms || 0,
    frame_mime_type: frame?.mime_type || '',
    frame_width: frame?.width || 0,
    frame_height: frame?.height || 0,
    frame_size_bytes: frame?.size_bytes || 0,
    frame_hash: frame?.frame_hash || '',
    frame_data_in_payload: false,
  };
  const commandId = `browser_context_${now}_${Math.random().toString(36).slice(2, 10)}`;
  const payload = {
    title: options.webStack && sourceId ? `Web Stack Browser: ${sourceId}` : `Browser: ${title}`,
    instruction: url
      ? `Use this browser context from ${url}. The screenshot is available as the referenced browser frame record.`
      : 'Use this browser context. The screenshot is available as the referenced browser frame record.',
    source_module: sourceModule,
    required_skills: options.webStack ? ['browser-context', 'web-stack', 'ctox'] : ['browser-context', 'ctox'],
    browser_context: browserContext,
    source_id: sourceId,
    capture_script: captureScript,
    secret_value_in_payload: false,
  };
  await startCommandSync(ctx);
  await requireCommandBus(ctx).dispatch({
      id: commandId,
      module: 'ctox',
      command_type: 'ctox.browser_context.capture',
      record_id: session.id,
      inbound_channel: 'browser',
      payload,
      client_context: {
        source: options.webStack ? 'business-os.browser.web-stack-capture' : 'business-os.browser.context',
        module_id: 'browser',
        actor: actorContext(ctx.session),
        browser_context: browserContext,
      },
    });
  const target = options.webStack ? 'Web Stack' : 'CTOX';
  state.notice = frame?.id
    ? `Browser-Kontext wurde an ${target} uebergeben.`
    : `Browser-Kontext wurde an ${target} uebergeben. Sobald die Seite geladen ist, wird auch die aktuelle Ansicht referenziert.`;
}

async function extractWebStackFields(ctx, state) {
  const session = state.latestSession;
  const tab = state.latestTab;
  const frame = state.latestFrame;
  const payload = session?.payload || {};
  if (!session?.id || !payload.capture_script) {
    state.notice = 'Keine Web-Stack-Uebergabe fuer dieses Browser-Fenster verfuegbar.';
    return;
  }
  const now = Date.now();
  const browserContext = {
    session_id: session.id,
    tab_id: tab?.id || session.current_tab_id || '',
    url: tab?.url || session.current_url || '',
    title: browserDisplayTitle(tab, session, tab?.url || session.current_url || ''),
    status: session.runtime_status || session.status || '',
    purpose: payload.purpose || '',
    source_id: payload.source_id || '',
    capture_script: payload.capture_script || '',
    verify_selector: payload.verify_selector || '',
    frame_id: frame?.id || '',
    frame_seq: frame?.seq || 0,
    frame_captured_at_ms: frame?.captured_at_ms || 0,
    frame_expires_at_ms: frame?.expires_at_ms || 0,
    frame_mime_type: frame?.mime_type || '',
    frame_width: frame?.width || 0,
    frame_height: frame?.height || 0,
    frame_size_bytes: frame?.size_bytes || 0,
    frame_hash: frame?.frame_hash || '',
    frame_data_in_payload: false,
  };
  const artifact = {
    kind: 'browser_context',
    schema_version: 1,
    stream: 'rxdb',
    source_module: 'web_stack',
    source_id: payload.source_id || '',
    capture_script: payload.capture_script || '',
    browser_context: browserContext,
    sensitivity: 'browser_context_reference',
    secret_value_in_payload: false,
    frame_data_in_payload: false,
  };
  const commandId = `browser_extract_${now}_${Math.random().toString(36).slice(2, 10)}`;
  const command = {
    id: commandId,
    command_id: commandId,
    module: 'ctox',
    command_type: 'browser.capture.extract',
    record_id: session.id,
    inbound_channel: 'browser',
    payload: {
      session_id: session.id,
      source_id: payload.source_id || '',
      capture_script: payload.capture_script || '',
      frame_id: frame?.id || '',
      browser_context_artifact: artifact,
      secret_value_in_payload: false,
      frame_data_in_payload: false,
    },
    client_context: {
      source: 'business-os.browser.web-stack-extract',
      module_id: 'browser',
      actor: actorContext(ctx.session),
    },
    created_at_ms: now,
    updated_at_ms: now,
  };
  await startCommandSync(ctx);
  await requireCommandBus(ctx).dispatch(command);
  state.notice = 'CTOX liest die Seite fuer den Web Stack aus.';
}

async function completeWebStackAuthAssist(ctx, state) {
  const session = state.latestSession;
  if (!session?.id) {
    state.notice = 'Kein Web-Stack-Browserfenster geoeffnet.';
    return;
  }
  if (session.payload?.purpose !== 'web_stack_auth') {
    state.notice = 'Dieses Browserfenster gehoert nicht zu einer Web-Stack-Anmeldung.';
    return;
  }
  const now = Date.now();
  const commandId = `web_stack_auth_complete_${now}_${Math.random().toString(36).slice(2, 10)}`;
  const payload = {
    session_id: session.id,
    tab_id: state.latestTab?.id || session.current_tab_id || '',
    source_id: session.payload?.source_id || '',
    secret_name: session.payload?.secret_name || '',
    completed_at_ms: now,
    browser_stream: 'rxdb',
    secret_value_in_rxdb: false,
  };
  await startCommandSync(ctx);
  await requireCommandBus(ctx).dispatch({
      id: commandId,
      module: 'browser',
      command_type: 'web_stack.auth_assist.complete',
      record_id: session.id,
      inbound_channel: 'browser',
      payload,
      client_context: {
        source: 'business-os.browser.auth-assist',
        module_id: 'browser',
        actor: actorContext(ctx.session),
      },
    });
  state.notice = 'Anmeldung wurde an CTOX uebergeben.';
}

async function fillWebStackCredential(ctx, state) {
  const session = state.latestSession;
  if (!session?.id) {
    state.notice = 'Kein Web-Stack-Browserfenster geoeffnet.';
    return;
  }
  const secretName = session.payload?.secret_name || '';
  if (!secretName) {
    state.notice = 'Fuer diese Quelle ist kein Zugang im CTOX Secret Store hinterlegt.';
    return;
  }
  const now = Date.now();
  const commandId = `browser_credential_fill_${now}_${Math.random().toString(36).slice(2, 10)}`;
  const payload = {
    session_id: session.id,
    tab_id: state.latestTab?.id || session.current_tab_id || '',
    lease_id: state.controllerLeaseId,
    source_id: session.payload?.source_id || '',
    secret_scope: 'credentials',
    secret_name: secretName,
    field_role: session.payload?.credential_field_role || 'both',
    confirmed: true,
    browser_stream: 'rxdb',
    secret_value_in_rxdb: false,
  };
  const selector = String(session.payload?.credential_selector || session.payload?.selector || '').trim();
  if (selector) payload.selector = selector;
  await startCommandSync(ctx);
  await requireCommandBus(ctx).dispatch({
      id: commandId,
      module: 'browser',
      command_type: 'browser.credential.fill',
      record_id: session.id,
      inbound_channel: 'browser',
      payload,
      client_context: {
        source: 'business-os.browser.credential-fill',
        module_id: 'browser',
        actor: actorContext(ctx.session),
      },
    }, { until: 'terminal', timeoutMs: 150_000 });
  await refreshBrowserProjections(ctx);
  state.notice = 'Zugangsdaten wurden eingesetzt.';
}

async function startCommandSync(ctx) {
  return startBrowserRuntimeSync(ctx, ['business_commands']);
}

async function refreshBrowserProjections(ctx) {
  if (!ctx.sync?.restartCollection) return;
  await Promise.all(
    ['browser_sessions', 'browser_tabs', 'browser_frames']
      .map((collection) => ctx.sync.restartCollection(collection, { forceDirect: true })),
  );
}

async function startBrowserRuntimeSync(ctx, collections = [
  'business_commands',
  'browser_sessions',
  'browser_tabs',
  'browser_frames',
  'browser_input_events',
]) {
  if (!ctx.sync?.startCollection) {
    const error = new Error('CTOX Browser-Datenkanal ist nicht verfügbar.');
    error.code = 'sync_unavailable';
    throw error;
  }
  await Promise.all(collections.map((collection) => ctx.sync.startCollection(collection)));
}

function actorContext(session) {
  const user = session?.user || {};
  return {
    user_id: user.id || session?.userId || '',
    display_name: user.display_name || user.name || '',
    role: user.role || '',
  };
}

function requireCommandBus(ctx) {
  if (!ctx?.commandBus?.dispatch) {
    throw new Error('CTOX command bus is unavailable. The action was not submitted.');
  }
  return ctx.commandBus;
}

function installInputHandlers(ctx, refs, state, scheduleRefresh) {
  refs.canvas?.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    refs.canvas.focus();
    refs.canvas.setPointerCapture?.(event.pointerId);
    state.pointerIsDown = true;
    writePointerInput(ctx, refs, state, 'mouseDown', event).then(scheduleRefresh);
  });
  refs.canvas?.addEventListener('pointerup', (event) => {
    event.preventDefault();
    state.pointerIsDown = false;
    writePointerInput(ctx, refs, state, 'mouseUp', event).then(scheduleRefresh);
  });
  refs.canvas?.addEventListener('pointercancel', (event) => {
    state.pointerIsDown = false;
    writePointerInput(ctx, refs, state, 'mouseUp', event).then(scheduleRefresh);
  });
  refs.canvas?.addEventListener('pointermove', (event) => {
    const now = Date.now();
    const dragging = state.pointerIsDown || Number(event.buttons || 0) > 0;
    if (!dragging && now - Number(state.lastPointerMoveAt || 0) < POINTER_HOVER_THROTTLE_MS) return;
    state.lastPointerMoveAt = now;
    writePointerInput(ctx, refs, state, 'mouseMove', event).then(scheduleRefresh);
  });
  refs.canvas?.addEventListener('wheel', (event) => {
    event.preventDefault();
    writePointerInput(ctx, refs, state, 'wheel', event).then(scheduleRefresh);
  }, { passive: false });
  refs.canvas?.addEventListener('keydown', (event) => {
    event.preventDefault();
    writeKeyboardInput(ctx, state, 'keyDown', event).then(scheduleRefresh);
  });
  refs.canvas?.addEventListener('keyup', (event) => {
    event.preventDefault();
    writeKeyboardInput(ctx, state, 'keyUp', event).then(scheduleRefresh);
  });
}

async function writePointerInput(ctx, refs, state, type, event, extra = {}) {
  const point = canvasPoint(refs.canvas, event);
  const clickCount = type === 'wheel' ? 0 : pointerClickCount(state, type, event, point);
  const buttons = pointerButtons(event, type);
  const patch = {
    x: point.x,
    y: point.y,
    detail: clickCount,
    clickCount,
    button: pointerButton(event.button),
    buttons,
    dx: type === 'wheel' ? Number(event.deltaX || 0) : 0,
    dy: type === 'wheel' ? Number(event.deltaY || 0) : 0,
    key: '',
    code: '',
    modifiers: eventModifiers(event),
    text: '',
    payload: {
      pointer_id: event.pointerId || 0,
      pointer_type: event.pointerType || 'mouse',
      click_count: clickCount,
      viewport_w: refs.canvas?.width || VIEWPORT.width,
      viewport_h: refs.canvas?.height || VIEWPORT.height,
      actor: actorContext(ctx.session),
    },
    ...extra,
  };
  await writeInputEvent(ctx, state, type, patch);
}

async function writeKeyboardInput(ctx, state, type, event) {
  const key = event.key || '';
  const code = event.code || '';
  const modifiers = eventModifiers(event);
  await writeInputEvent(ctx, state, type, {
    x: 0,
    y: 0,
    detail: 0,
    clickCount: 0,
    button: 'none',
    buttons: 0,
    dx: 0,
    dy: 0,
    key,
    code,
    modifiers,
    text: keyboardText(type, event),
    payload: {
      repeat: Boolean(event.repeat),
      location: Number(event.location || 0),
      actor: actorContext(ctx.session),
    },
  });
}

async function writeInputEvent(ctx, state, type, patch) {
  // Beide Abbruchgruende waren bis 13.08.2026 STUMM: kein Protokoll, keine
  // Meldung, kein Hinweis in der Oberflaeche. Auf der Kundeninstanz gemessen:
  // null Eingabe-Ereignisse in zehn Minuten aktiver Bedienung, waehrend beide
  // Sitzungen "active" meldeten und die Oberflaeche "Bereit" zeigte. Der
  // Nutzer sah einen laufenden Browser, den niemand bedienen durfte — und
  // nichts sagte ihm warum. Ein Abbruch ohne Grund ist von einem Absturz
  // nicht zu unterscheiden.
  if (!browserSurfaceCanControl(ctx, state)) {
    meldeEingabeGesperrt(ctx, state, type);
    return;
  }
  const session = state.latestSession;
  const frame = state.latestFrame;
  const sessionId = session?.id || frame?.session_id;
  if (!sessionId) {
    // Faellt der Bildstrom aus, gibt es kein latestFrame. Ohne latestSession
    // bricht es dann hier ab — der Bildstromdefekt blockiert die Eingabe mit.
    console.warn('[browser] Eingabe verworfen: keine Sitzung zuordenbar', { type });
    setzeEingabeHinweis(ctx, state, 'Keine Sitzung zugeordnet. Bitte die Sitzung neu starten.');
    return;
  }
  const now = Date.now();
  const seq = Math.max(now, Number(state.lastInputSeq || 0) + 1);
  state.lastInputSeq = seq;
  const event = {
    id: `${sessionId}:input:${seq}:${type}`,
    tenant_id: browserTenantId(ctx),
    owner_user_id: session?.owner_user_id || '',
    controller_user_id: session?.controller_user_id || browserActorIds(ctx.session)[0] || '',
    session_id: sessionId,
    tab_id: state.latestTab?.id || frame?.tab_id || '',
    seq,
    client_seq: seq,
    frame_seq: Number(frame?.seq || session?.last_frame_seq || 0),
    lease_id: state.controllerLeaseId || '',
    ack_status: 'pending',
    type,
    status: 'pending',
    created_at_ms: now,
    updated_at_ms: now,
    ...patch,
  };
  console.info('[browser] input sent', browserInputTrace(event));
  await upsertDoc(browserCollection(ctx, 'browser_input_events'), event);
}

function browserInputTrace(event) {
  const clickCount = Number(event.clickCount || event.detail || event.payload?.click_count || 0);
  return {
    session_id: event.session_id || '',
    tab_id: event.tab_id || '',
    seq: Number(event.seq || 0),
    type: event.type || '',
    x: Number(event.x || 0),
    y: Number(event.y || 0),
    detail: clickCount,
    clickCount,
    button: event.button || '',
    buttons: Number(event.buttons || 0),
    dx: Number(event.dx || 0),
    dy: Number(event.dy || 0),
    modifiers: Array.isArray(event.modifiers) ? event.modifiers : [],
    key: event.key || '',
    code: event.code || '',
    text: event.text || '',
  };
}

function installAddressClearControl(form, address, go) {
  if (!form || !address) return null;
  const existing = form.querySelector?.('[data-browser-address-clear]');
  if (existing) return existing;
  const button = globalThis.document?.createElement?.('button');
  if (!button) return null;
  button.type = 'button';
  button.className = 'ctox-icon-button';
  button.dataset.browserAddressClear = '';
  button.setAttribute('aria-label', 'Adresse löschen');
  button.title = 'Adresse löschen';
  button.textContent = '×';
  form.insertBefore(button, go || null);
  return button;
}

function browserTenantId(ctx) {
  return String(
    ctx?.sync?.config?.instance_id
      || ctx?.sync?.config?.instanceId
      || ctx?.config?.instance_id
      || ctx?.config?.instanceId
      || '',
  ).trim();
}

function canvasPoint(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / Math.max(1, rect.width);
  const scaleY = canvas.height / Math.max(1, rect.height);
  return {
    x: Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left) * scaleX))),
    y: Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top) * scaleY))),
  };
}

function eventModifiers(event) {
  return [
    event.altKey ? 'Alt' : '',
    event.ctrlKey ? 'Control' : '',
    event.metaKey ? 'Meta' : '',
    event.shiftKey ? 'Shift' : '',
  ].filter(Boolean);
}

function pointerButton(button) {
  if (button === 1) return 'middle';
  if (button === 2) return 'right';
  if (button === -1 || button == null) return 'none';
  return 'left';
}

function pointerButtons(event, type) {
  const reported = Number(event.buttons || 0);
  if (type === 'mouseDown' && reported === 0) {
    if (event.button === 1) return 4;
    if (event.button === 2) return 2;
    return 1;
  }
  if (type === 'mouseUp') return reported;
  return reported;
}

function pointerClickCount(state, type, event, point) {
  const fromDom = Number(event.detail || 0);
  if (fromDom > 0) {
    if (type === 'mouseDown' || type === 'mouseUp') {
      state.lastPointerClick = {
        count: fromDom,
        x: point.x,
        y: point.y,
        at: Date.now(),
        button: Number(event.button || 0),
      };
    }
    return fromDom;
  }
  if (type === 'mouseMove') return Number(state.lastPointerClick?.count || 0);
  if (type !== 'mouseDown' && type !== 'mouseUp') return 0;

  const now = Date.now();
  const previous = state.lastPointerClick;
  const sameButton = previous && Number(previous.button || 0) === Number(event.button || 0);
  const closeEnough = previous
    && Math.abs(previous.x - point.x) <= POINTER_CLICK_SLOP_PX
    && Math.abs(previous.y - point.y) <= POINTER_CLICK_SLOP_PX
    && now - Number(previous.at || 0) <= POINTER_CLICK_INTERVAL_MS;
  const nextCount = type === 'mouseDown'
    ? (sameButton && closeEnough ? Number(previous.count || 0) + 1 : 1)
    : (sameButton && closeEnough ? Number(previous.count || 1) : 1);
  state.lastPointerClick = {
    count: nextCount,
    x: point.x,
    y: point.y,
    at: now,
    button: Number(event.button || 0),
  };
  return nextCount;
}

function keyboardText(type, event) {
  if (type !== 'keyDown') return '';
  const key = event.key || '';
  if (!key || key.length !== 1) return '';
  if (event.ctrlKey || event.metaKey || event.altKey) return '';
  return key;
}

function browserInputPayload(event) {
  return {
    type: event.type || '',
    x: Number(event.x || 0),
    y: Number(event.y || 0),
    detail: Number(event.detail || event.clickCount || 0),
    clickCount: Number(event.clickCount || event.detail || 0),
    button: event.button || 'left',
    buttons: Number(event.buttons || 0),
    dx: Number(event.dx || 0),
    dy: Number(event.dy || 0),
    key: event.key || '',
    code: event.code || '',
    modifiers: Array.isArray(event.modifiers) ? event.modifiers : [],
    text: event.text || '',
  };
}

async function readCollection(collection, options = {}) {
  if (!collection?.find) return [];
  const limit = Number.isFinite(options.limit) ? options.limit : 100;
  const selector = options.selector || {};
  const sort = options.sort || [{ updated_at_ms: 'desc' }];
  const docs = await collection.find({ selector, sort, limit }).exec();
  return docs
    .map((doc) => doc?.toJSON?.() || doc)
    .filter((doc) => doc && doc._deleted !== true);
}

async function readDocument(collection, id) {
  if (!collection?.findOne || !id) return null;
  const doc = await collection.findOne(id).exec();
  const data = doc?.toJSON?.() || doc;
  return data && data._deleted !== true ? data : null;
}

function mergeRequestedSession(sessions, requestedSession) {
  return mergeRequestedDocument(sessions, requestedSession);
}

function mergeRequestedDocument(documents, requestedDocument) {
  if (!requestedDocument?.id) return documents;
  return [
    requestedDocument,
    ...documents.filter((document) => document.id !== requestedDocument.id),
  ];
}

function browserCollection(ctx, name) {
  return ctx?.db?.collection?.(name) || null;
}

function latestFrame(frames, sessionId = '') {
  return frames
    .filter((frame) => frame.data && (!sessionId || frame.session_id === sessionId) && Number(frame.expires_at_ms || 0) > Date.now())
    .sort((a, b) => Number(b.seq || 0) - Number(a.seq || 0) || Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0))[0] || null;
}

function latestSession(sessions, sessionId) {
  const candidates = sessionId
    ? sessions.filter((session) => session.id === sessionId)
    : sessions;
  return candidates.sort((a, b) => Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0))[0] || null;
}

function latestTab(tabs, tabId) {
  const candidates = tabId
    ? tabs.filter((tab) => tab.id === tabId)
    : tabs;
  return candidates.sort((a, b) => Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0))[0] || null;
}

function latestBrowserCommand(commands, sessionId) {
  return latestBrowserCommands(commands, sessionId, 1)[0] || null;
}

function applyLatestNavigationResult(state, commands) {
  const sessionId = state.latestSession?.id || state.latestFrame?.session_id;
  if (!sessionId) return;
  const command = commands
    .filter((candidate) => {
      const type = candidate.command_type || candidate.type || '';
      const candidateSessionId = candidate.payload?.session_id || candidate.record_id || '';
      return candidateSessionId === sessionId
        && ['browser.session.start', 'browser.navigate', 'browser.reload', 'browser.back', 'browser.forward', 'browser.reset'].includes(type)
        && candidate.status === 'completed'
        && candidate.result?.url;
    })
    .sort((a, b) => Number(b.updated_at_ms || b.created_at_ms || 0) - Number(a.updated_at_ms || a.created_at_ms || 0))[0];
  if (!command) return;
  const url = String(command.result.url || '');
  const title = String(command.result.title || state.latestTab?.title || state.latestSession?.title || 'Browser');
  state.latestSession = { ...state.latestSession, current_url: url, title };
  state.latestTab = { ...(state.latestTab || {}), url, title };
}

function latestBrowserCommands(commands, sessionId, limit = 5) {
  return commands
    .filter((command) => {
      const type = command.command_type || command.type || '';
      if (!type.startsWith('browser.')) return false;
      if (!sessionId) return true;
      const payloadSession = command.payload?.session_id;
      return command.record_id === sessionId || payloadSession === sessionId;
    })
    .sort((a, b) => Number(b.updated_at_ms || b.created_at_ms || 0) - Number(a.updated_at_ms || a.created_at_ms || 0))
    .slice(0, limit);
}

function latestBrowserHandoffTasks(tasks, sessionId, limit = 5) {
  return tasks
    .filter((task) => {
      if (task.command_type !== 'ctox.browser_context.capture') return false;
      if (task.inbound_channel !== 'browser') return false;
      if (!sessionId) return true;
      return String(task.prompt || '').includes(sessionId) || String(task.command_id || '').includes(sessionId);
    })
    .sort((a, b) => Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0))
    .slice(0, limit);
}

function renderSessionList(refs, sessions, tabs, activeSession) {
  if (!refs.sessionList) return;
  const sorted = [...sessions].sort((a, b) => Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0));
  if (!sorted.length) {
    refs.sessionList.innerHTML = '';
    return;
  }
  const activeTabs = tabs
    .filter((tab) => tab.session_id === activeSession?.id && tab.status !== 'closed')
    .sort((a, b) => Number(b.updated_at_ms || 0) - Number(a.updated_at_ms || 0));
  const tabMarkup = activeTabs.map((tab) => `
    <span class="ctox-pane-tab browser-tab" data-browser-tab-id="${escapeHtml(tab.id)}" aria-selected="${tab.id === activeSession?.current_tab_id ? 'true' : 'false'}">
      <span class="browser-tab-title">${escapeHtml(tab.title || tab.url || 'Tab')}</span>
      <button type="button" class="ctox-icon-button ctox-icon-button--sm" data-browser-tab-close aria-label="Tab schließen">×</button>
    </span>
  `).join('');
  refs.sessionList.innerHTML = tabMarkup;
}

// ----- LEFT sessions selector (canonical grammar column) -----

// Owned sessions plus the read-only import overlay (imported entries that are
// not already a real owned session). Imported sessions are marked and never
// persisted — browser sessions are a read-only projection.
function sessionRenderList(state) {
  const owned = Array.isArray(state?.visibleSessions) ? state.visibleSessions : [];
  const ownedIds = new Set(owned.map((session) => session.id));
  const imported = (Array.isArray(state?.importedSessions) ? state.importedSessions : [])
    .filter((session) => session?.id && !ownedIds.has(session.id));
  return [...owned, ...imported];
}

function sessionTabCounts(tabs) {
  const counts = {};
  for (const tab of Array.isArray(tabs) ? tabs : []) {
    if (tab.status === 'closed') continue;
    counts[tab.session_id] = Number(counts[tab.session_id] || 0) + 1;
  }
  return counts;
}

// Rebuild the sessions well ONLY when the rendered data changed (signature
// guard). Selection is applied in place via markActiveSession so a re-render
// never clamps the well scroll or moves the operator.
function renderSessions(refs, sessions, activeSession, view = {}, tabCounts = {}, ctx = null) {
  if (!refs.sessions) return;
  const all = Array.isArray(sessions) ? sessions : [];
  const filtered = filterSessionsForView(all, view);
  const listView = view?.view === 'list' ? 'list' : 'cards';
  refs.sessions.classList.toggle('is-list', listView === 'list');

  const signature = sessionListSignature(filtered, listView, tabCounts);
  if (refs.sessions.dataset.sig !== signature) {
    refs.sessions.dataset.sig = signature;
    refs.sessions.innerHTML = filtered
      .map((session) => sessionShardMarkup(session, tabCounts[session.id] || 0, ctx))
      .join('');
  }
  if (refs.sessionsEmpty) refs.sessionsEmpty.hidden = filtered.length > 0;
  markActiveSession(refs, activeSession?.id || '');

  const counts = browserSessionViewCounts(all, view);
  const pg = refs.sessionsPane?.__ctoxPaneGrammar;
  if (pg?.setCounts) pg.setCounts(counts);
  else writeSessionCounts(refs, counts);
  const noun = filtered.length === 1 ? 'Sitzung' : 'Sitzungen';
  const footer = `${filtered.length} ${noun} · ${bandLabel(view?.band)}`;
  if (pg?.setFooter) pg.setFooter(footer);
  else writeSessionFooter(refs, footer);
}

// In-place selection flip across existing rows — NEVER a rebuild.
function markActiveSession(refs, sessionId) {
  for (const node of refs.sessions?.querySelectorAll('[data-browser-session-id]') || []) {
    const active = node.dataset.browserSessionId === sessionId;
    node.classList.toggle('is-selected', active);
    node.classList.toggle('is-active', active);
    node.setAttribute('aria-selected', active ? 'true' : 'false');
  }
}

function sessionShardMarkup(session, tabCount, ctx = null) {
  const url = session.current_url || session.payload?.target_url || '';
  const title = browserDisplayTitle(null, session, url);
  const meta = browserSessionShardMeta(session, tabCount, ctx);
  const importedClass = session.__imported ? ' browser-session--imported' : '';
  // Das Symbol traegt den Zustand, nicht nur der Text: bei 34 Sitzungen findet
  // man "Eingriff noetig" sonst nicht, ohne jede Zeile zu lesen.
  const z = browserSitzungZustand(session, ctx);
  return `
    <div class="ctox-list-item browser-session${importedClass} ${z.klasse}" role="option" aria-selected="false" tabindex="0"
      data-browser-session-id="${escapeHtml(session.id)}"
      data-browser-zustand="${escapeHtml(z.klasse)}"
      data-context-record-id="${escapeHtml(session.id)}"
      data-context-record-type="browser_session"
      data-context-label="${escapeHtml(title)}">
      <span class="browser-session-zustand" title="${escapeHtml(z.text)}" aria-label="${escapeHtml(z.text)}">${z.symbol}</span>
      <span class="browser-session-title">${escapeHtml(title)}</span>
      <span class="browser-session-meta">${escapeHtml(meta)}</span>
    </div>`;
}

// WER STEUERT DIESE SITZUNG, und muss jemand eingreifen?
//
// Bis 13.08.2026 stand in jeder Zeile nur "Persoenlich · Bereit · 1 Tab". Bei
// 34 Sitzungen war die Liste damit wertlos: man sah nicht, ob eine Sitzung dem
// Nutzer gehoert oder von der Recherche gefahren wird, und vor allem nicht, wo
// jemand eingreifen muss. Eine Automatik, die auf eine Anmeldung oder eine
// Bot-Pruefung wartet, lief still in ihre Zeitueberschreitung — der Klick, der
// gereicht haette, wurde nie angefordert.
//
// Die vier Zustaende kommen ohne neue Felder aus; sie stehen alle schon im
// Sitzungsdatensatz.
const SITZUNG_ZUSTAENDE = Object.freeze({
  eingriff: { symbol: '⚠', klasse: 'is-eingriff', text: 'Eingriff nötig' },
  nutzer: { symbol: '👤', klasse: 'is-nutzer', text: 'Sie steuern' },
  automatik: { symbol: '⚙', klasse: 'is-automatik', text: 'Automatik' },
  ruhend: { symbol: '○', klasse: 'is-ruhend', text: 'Ruhend' },
});

function browserSitzungZustand(session, ctx, jetzt = Date.now()) {
  if (!session?.id) return SITZUNG_ZUSTAENDE.ruhend;
  // 1. Eingriff zuerst — er ist der einzige Zustand, der jemanden BRAUCHT.
  const fehler = String(browserSessionError(session) || '').toLowerCase();
  const titel = String(session.title || '').toLowerCase();
  const wartetAufMensch = /anmeld|login|sign in|captcha|verify|bot|zustimm|consent|just a moment/
    .test(`${fehler} ${titel}`);
  const lauft = ['active', 'running', 'ready', 'capturing'].includes(
    String(session.runtime_status || session.status || '').toLowerCase());
  if (lauft && wartetAufMensch) return SITZUNG_ZUSTAENDE.eingriff;
  if (!lauft) return SITZUNG_ZUSTAENDE.ruhend;

  // 2. Steuert der angemeldete Nutzer selbst — mit GUELTIGER Pacht?
  const meine = browserActorIds(ctx?.session) || [];
  const steuernder = String(session.controller_user_id || '');
  const pachtBis = Number(session.controller_lease_expires_at_ms || 0);
  if (steuernder && meine.includes(steuernder) && pachtBis > jetzt) {
    return SITZUNG_ZUSTAENDE.nutzer;
  }
  // 3. Sonst faehrt sie jemand anderes — Recherche, Adapter, anderer Nutzer.
  //    Diese Sitzungen NICHT versehentlich uebernehmen: ein Zugriff bricht
  //    einen laufenden Rechercheschritt ab.
  return SITZUNG_ZUSTAENDE.automatik;
}

function browserSessionShardMeta(session, tabCount = 0, ctx = null) {
  const status = browserStatusLabel(session);
  const profile = (session.profile_mode || session.payload?.profile_mode) === 'private' ? 'Privat' : 'Persönlich';
  const count = Number(tabCount || 0);
  const zustand = browserSitzungZustand(session, ctx);
  const parts = [zustand.text, profile, status, `${count} ${count === 1 ? 'Tab' : 'Tabs'}`];
  const error = browserSessionError(session);
  if (error) parts.push(error);
  if (session.__imported) parts.push('Import');
  return parts.join(' · ');
}

function browserSessionBand(session) {
  // "Aktiv" means a native runtime has confirmed a live Chromium-backed
  // session. Starting, blocked and disconnected rows do not count as active.
  return browserUiState(session) === 'ready' ? 'active' : 'closed';
}

function browserSessionMatchesBand(session, band) {
  if (!band || band === 'all') return true;
  return browserSessionBand(session) === band;
}

function filterSessionsForView(sessions, view = {}) {
  const search = String(view.search || '').trim().toLowerCase();
  const profile = view.filters?.profile && view.filters.profile !== 'all' ? view.filters.profile : '';
  const band = view.band || 'all';
  return (Array.isArray(sessions) ? sessions : []).filter((session) => {
    if (!browserSessionMatchesBand(session, band)) return false;
    if (profile) {
      const mode = session.profile_mode || session.payload?.profile_mode || 'persistent';
      if (mode !== profile) return false;
    }
    if (search) {
      const hay = `${session.id || ''} ${session.title || ''} ${session.current_url || ''} ${session.payload?.target_url || ''}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
}

// Band counts ignore the band selection (the band IS the selector) but honor
// search + profile filters; zeros are rendered, never hidden.
function browserSessionViewCounts(sessions, view = {}) {
  const base = filterSessionsForView(sessions, { ...view, band: 'all' });
  return {
    all: base.length,
    active: base.filter((session) => browserSessionBand(session) === 'active').length,
    closed: base.filter((session) => browserSessionBand(session) === 'closed').length,
  };
}

// Selection-independent signature: a pure selection change (not part of the
// signature) leaves it identical, so renderSessions skips the rebuild.
function sessionListSignature(filteredSessions, listView, tabCounts = {}) {
  const rows = (Array.isArray(filteredSessions) ? filteredSessions : []).map((session) => [
    session.id || '',
    browserUiState(session),
    String(session.title || session.current_url || ''),
    browserSessionError(session),
    session.profile_mode || session.payload?.profile_mode || '',
    Number(tabCounts[session.id] || 0),
    Number(session.updated_at_ms || 0),
    session.__imported ? 'i' : '',
    // Der Zustand haengt an Steuerndem und Pacht — ohne diese beiden
    // Werte in der Signatur bliebe das Symbol stehen, wenn eine Pacht
    // ablaeuft oder jemand anderes uebernimmt.
    String(session.controller_user_id || ''),
    Number(session.controller_lease_expires_at_ms || 0) > Date.now() ? 'p' : '-',
  ].join(':'));
  return `${listView}::${rows.join('|')}`;
}

// Auto-reveal model (design-guide "Progressive Disclosure"): the remote work
// surface is the browser's "detail" — revealed once a session is selected and
// not user-collapsed. Mirrors the outbound/conversations idiom.
function browserWorkbenchVisible(hasSession, userCollapsed = false) {
  return Boolean(hasSession) && !userCollapsed;
}

function bandLabel(band) {
  if (band === 'active') return 'Aktiv';
  if (band === 'closed') return 'Beendet';
  return 'Alle';
}

function writeSessionCounts(refs, counts) {
  for (const [key, value] of Object.entries(counts || {})) {
    const node = refs.sessionsPane?.querySelector(`[data-pg-count="${key}"]`);
    if (node) node.textContent = ` (${value})`;
  }
}

function writeSessionFooter(refs, text) {
  const node = refs.sessionsPane?.querySelector('[data-pg-footer]');
  if (node) node.textContent = text || '';
}

// Export/import are honest and small: export writes owned sessions as JSON via a
// Blob; import overlays sessions for local review only (never persisted).
function buildBrowserSessionsExport(sessions, nowMs = Date.now()) {
  return {
    kind: 'browser_sessions',
    schema_version: 1,
    exported_at_ms: Number(nowMs) || 0,
    sessions: (Array.isArray(sessions) ? sessions : []).map((session) => ({
      id: session.id || '',
      title: browserDisplayTitle(null, session, session.current_url || ''),
      url: session.current_url || session.payload?.target_url || '',
      status: session.runtime_status || session.status || '',
      profile_mode: session.profile_mode || session.payload?.profile_mode || 'persistent',
      owner_user_id: session.owner_user_id || '',
      controller_user_id: session.controller_user_id || '',
      updated_at_ms: Number(session.updated_at_ms || 0),
    })),
  };
}

function parseBrowserSessionsImport(parsed) {
  const rows = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.sessions) ? parsed.sessions : [];
  return rows
    .map((row) => {
      const id = String(row?.id || '').trim();
      if (!id) return null;
      const status = String(row.status || 'imported');
      return {
        id,
        title: String(row.title || ''),
        current_url: String(row.url || row.current_url || ''),
        status,
        runtime_status: status,
        profile_mode: String(row.profile_mode || 'persistent'),
        updated_at_ms: Number(row.updated_at_ms || 0),
        __imported: true,
      };
    })
    .filter(Boolean);
}

function downloadBrowserJson(payload, filename, root) {
  let url = '';
  try {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = 'noopener';
    (root || document.body)?.appendChild?.(anchor);
    anchor.click();
    anchor.remove?.();
  } catch (error) {
    console.error('[browser] session export failed', error);
  } finally {
    if (url) setTimeout(() => { try { URL.revokeObjectURL(url); } catch {} }, 4000);
  }
}

function exportBrowserSessions(state, refs) {
  const rows = sessionRenderList(state).filter((session) => !session.__imported);
  downloadBrowserJson(buildBrowserSessionsExport(rows, Date.now()), 'browser-sessions.json', refs?.root);
}

function importBrowserSessions(ctx, state, refs) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json,.json';
  input.addEventListener('change', async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    let parsed;
    try {
      parsed = JSON.parse(await file.text());
    } catch {
      ctx.notifications?.show?.({ type: 'error', title: 'Browser', message: 'Ungültige JSON-Datei.' });
      return;
    }
    const imported = parseBrowserSessionsImport(parsed);
    if (!imported.length) {
      ctx.notifications?.show?.({ type: 'warning', title: 'Browser', message: 'Keine Sitzungen in der Datei.' });
      return;
    }
    state.importedSessions = imported;
    renderSessions(refs, sessionRenderList(state), state.latestSession, state.leftView, sessionTabCounts(state.tabs), ctx);
    ctx.notifications?.show?.({ type: 'info', title: 'Browser', message: `${imported.length} Sitzungen geladen (nur lokal).` });
  });
  input.click();
}

function renderSession(refs, session, tab, frame, command, state) {
  if (!session) {
    refs.sessionCard.innerHTML = '<span class="browser-muted">Kein Browserfenster</span>';
    return;
  }
  const url = tab?.url || session.current_url || '';
  const commandError = commandErrorMessage(command);
  const runtimeError = browserSessionError(session);
  const sessionError = runtimeError
    ? `<div class="browser-error"><strong>Browser-Laufzeit:</strong> ${escapeHtml(runtimeError)}</div>`
    : '';
  const commandLine = command
    ? `<div class="browser-muted">Letzte Browseraktion: ${escapeHtml(browserActionLabel(command.command_type || command.type || 'browser'))}</div>`
    : '';
  const policy = session.payload?.control_policy || {};
  const owner = session.owner_user_id || policy.owner_user_id || '-';
  const controller = session.controller_user_id || policy.controller_user_id || '-';
  const leaseRemaining = Math.max(0, Number(session.controller_lease_expires_at_ms || 0) - Date.now());
  const profileMode = session.profile_mode || session.payload?.profile_mode || 'persistent';
  const status = browserStatusLabel(session);
  if (refs.address && url && !state?.addressDirty && document.activeElement !== refs.address) refs.address.value = url;
  refs.sessionCard.innerHTML = `
    <strong>${escapeHtml(browserDisplayTitle(tab, session, url))}</strong>
    <div class="browser-muted">${escapeHtml(url || 'about:blank')}</div>
    <div class="browser-meta-grid">
      <span>Status</span><span>${escapeHtml(status)}</span>
      <span>Owner</span><span>${escapeHtml(owner)}</span>
      <span>Control</span><span>${escapeHtml(controller)}</span>
      <span>Lease</span><span>${leaseRemaining ? `${Math.ceil(leaseRemaining / 60000)} min` : 'inaktiv'}</span>
      <span>Beobachter</span><span>${Array.isArray(session.allowed_observer_user_ids) ? session.allowed_observer_user_ids.length : 0}</span>
      <span>Profil</span><span>${profileMode === 'private' ? 'Privat – wird gelöscht' : 'Persönlich – persistent'}</span>
      <span>Frame</span><span>${escapeHtml(frame ? `${frame.width}x${frame.height}` : 'Kein Bild')}</span>
    </div>
    ${commandLine}
    ${sessionError}
    ${commandError ? `<div class="browser-error">${escapeHtml(commandError)}</div>` : ''}
  `;
}

function browserActionLabel(commandType) {
  const type = String(commandType || '');
  if (type === 'browser.session.start') return 'Fenster geoeffnet';
  if (type === 'browser.navigate') return 'Navigation';
  if (type === 'browser.reload') return 'Neu laden';
  if (type === 'browser.back') return 'Zurueck';
  if (type === 'browser.forward') return 'Vor';
  if (type === 'browser.reset') return 'Zurueckgesetzt';
  if (type === 'browser.session.stop') return 'Fenster geschlossen';
  return 'Aktualisiert';
}

function renderAuthAssist(refs, session) {
  if (!refs.authAssist) return;
  const payload = session?.payload || {};
  const permission = payload.pending_permission;
  if (permission && typeof permission === 'object') {
    refs.authAssist.hidden = false;
    refs.authAssist.innerHTML = `
      <div>
        <span class="ctox-pane-kicker">Website-Berechtigung</span>
        <strong>${escapeHtml(titleCase(permission.kind || 'permission'))}</strong>
        <small>${escapeHtml(permission.origin || '')}</small>
      </div>
      <div class="ctox-pane-tools">
        <button type="button" class="ctox-button" data-browser-permission-response="dismiss">Blockieren</button>
        <button type="button" class="ctox-button" data-browser-permission-response="accept">Einmal erlauben</button>
      </div>`;
    return;
  }
  const httpAuth = payload.pending_http_auth;
  if (httpAuth && typeof httpAuth === 'object') {
    refs.authAssist.hidden = false;
    refs.authAssist.innerHTML = `
      <div>
        <span class="ctox-pane-kicker">HTTP ${escapeHtml(httpAuth.scheme || 'Basic')} Authentifizierung</span>
        <strong>${escapeHtml(httpAuth.realm || httpAuth.origin || 'Geschützter Bereich')}</strong>
        <small>Zugangsdaten werden ausschließlich über eine CTOX Secret-Referenz eingesetzt.</small>
      </div>
      <div class="ctox-pane-tools">
        <button type="button" class="ctox-button" data-browser-http-auth-response="dismiss">Abbrechen</button>
        <button type="button" class="ctox-button" data-browser-http-auth-response="accept">Secret verwenden</button>
      </div>`;
    return;
  }
  const webAuthn = payload.pending_webauthn;
  if (webAuthn && typeof webAuthn === 'object') {
    refs.authAssist.hidden = false;
    refs.authAssist.innerHTML = `
      <div>
        <span class="ctox-pane-kicker">Passkey ${escapeHtml(webAuthn.type === 'create' ? 'registrieren' : 'verwenden')}</span>
        <strong>${escapeHtml(webAuthn.rp_id || 'Unbekannte Website')}</strong>
        <small>CTOX verwendet den verschlüsselten serverseitigen Passkey erst nach Ihrer Bestätigung.</small>
      </div>
      <div class="ctox-pane-tools">
        <button type="button" class="ctox-button" data-browser-webauthn-response="dismiss">Ablehnen</button>
        <button type="button" class="ctox-button" data-browser-webauthn-response="accept">Bestätigen</button>
      </div>`;
    return;
  }
  const dialog = payload.pending_dialog;
  if (dialog && typeof dialog === 'object') {
    refs.authAssist.hidden = false;
    refs.authAssist.innerHTML = `
      <div>
        <span class="ctox-pane-kicker">${escapeHtml(titleCase(dialog.type || 'dialog'))}</span>
        <strong>${escapeHtml(dialog.message || 'Die Webseite wartet auf eine Entscheidung.')}</strong>
      </div>
      <div class="ctox-pane-tools">
        <button type="button" class="ctox-button" data-browser-dialog-response="dismiss">Abbrechen</button>
        <button type="button" class="ctox-button" data-browser-dialog-response="accept">Bestätigen</button>
      </div>`;
    return;
  }
  const isAuthAssist = payload.purpose === 'web_stack_auth';
  refs.authAssist.hidden = !isAuthAssist;
  if (!isAuthAssist) {
    refs.authAssist.innerHTML = '';
    return;
  }
  const completed = payload.auth_assist_status === 'completed' || payload.authenticated === true;
  const fillStatus = payload.credential_fill_status || '';
  const extractStatus = payload.capture_extract_status || '';
  const canFill = Boolean(payload.secret_name) && !completed;
  const canCapture = Boolean(completed && payload.capture_script);
  const canExtract = Boolean(completed && payload.capture_script);
  const domains = Array.isArray(payload.allowed_domains) ? payload.allowed_domains.join(', ') : '';
  refs.authAssist.innerHTML = `
    <div>
      <span class="ctox-pane-kicker">Web Stack Anmeldung</span>
      <strong>${escapeHtml(payload.source_id || 'Anmeldung erforderlich')}</strong>
      <small>${escapeHtml(domains || payload.target_url || '')}</small>
      ${fillStatus ? `<small>${escapeHtml(authAssistStatusLabel(fillStatus, 'Zugangsdaten werden eingesetzt'))}</small>` : ''}
      ${extractStatus ? `<small>${escapeHtml(authAssistStatusLabel(extractStatus, 'Seitenauswertung laeuft'))}</small>` : ''}
    </div>
    <div class="ctox-pane-tools">
      <button type="button" class="ctox-button" data-browser-credential-fill ${canFill ? '' : 'disabled'}>
        Zugangsdaten einsetzen
      </button>
      <button type="button" class="ctox-button" data-browser-auth-complete ${completed ? 'disabled' : ''}>
        ${completed ? 'Angemeldet' : 'Ich bin angemeldet'}
      </button>
      <button type="button" class="ctox-button" data-browser-web-stack-capture ${canCapture ? '' : 'disabled'}>
        An CTOX uebergeben
      </button>
      <button type="button" class="ctox-button" data-browser-web-stack-extract ${canExtract ? '' : 'disabled'}>
        Seite auslesen
      </button>
    </div>
  `;
}

function authAssistStatusLabel(status, fallback) {
  const normalized = String(status || '').toLowerCase();
  if (['completed', 'done', 'ok', 'success'].includes(normalized)) return 'Abgeschlossen';
  if (['pending', 'pending_sync', 'accepted', 'running'].includes(normalized)) return fallback;
  if (['failed', 'error'].includes(normalized)) return 'Aktion fehlgeschlagen';
  return fallback;
}

function renderStatus(refs, session, tab, frame, command) {
  const state = browserUiState(session);
  if (refs.statusChip) {
    refs.statusChip.textContent = session ? browserStatusLabel(session) : t('statusDisconnected', 'Nicht verbunden');
    refs.statusChip.dataset.state = state;
    refs.statusChip.classList.toggle('is-success', state === 'ready');
    refs.statusChip.classList.toggle('is-warning', ['starting', 'waiting', 'blocked'].includes(state));
    refs.statusChip.classList.toggle('is-danger', state === 'error');
  }
  const url = tab?.url || session?.current_url || '';
  if (refs.statusTitle) {
    refs.statusTitle.textContent = browserDisplayTitle(tab, session, url) || t('noWindowOpen', 'Kein Browser-Fenster geoeffnet');
  }
  const bits = [];
  if (url) bits.push(url);
  if (frame?.seq != null) bits.push(`Frame ${frame.seq}`);
  if (session?.frame_rate_target) bits.push(`${session.frame_rate_target} fps`);
  const error = commandErrorMessage(command) || browserSessionError(session);
  if (error) bits.push(`Grund: ${error}`);
  if (refs.statusMeta) refs.statusMeta.textContent = bits.join(' - ') || '-';
}

function renderDownloads(refs, session) {
  if (!refs.downloads) return;
  const downloads = Array.isArray(session?.payload?.downloads) ? session.payload.downloads : [];
  refs.downloads.hidden = downloads.length === 0;
  refs.downloads.innerHTML = downloads.map((download) => `
    <span class="browser-download-item">
      <strong>${escapeHtml(download.filename || 'Download')}</strong>
      · ${escapeHtml(download.status || 'Unbekannt')}
      · ${escapeHtml(formatBytes(download.size_bytes || 0))}
      <button type="button" class="ctox-button ctox-button--sm" data-browser-download-action="release" data-browser-download-id="${escapeHtml(download.id || '')}" ${download.status === 'clean' ? '' : 'disabled'}>Freigeben</button>
      <button type="button" class="ctox-button ctox-button--sm" data-browser-download-action="rescan" data-browser-download-id="${escapeHtml(download.id || '')}" ${['infected', 'discarded', 'released'].includes(download.status) ? 'disabled' : ''}>Neu prüfen</button>
      <button type="button" class="ctox-button ctox-button--sm" data-browser-download-action="discard" data-browser-download-id="${escapeHtml(download.id || '')}" ${['discarded', 'released'].includes(download.status) ? 'disabled' : ''}>Verwerfen</button>
    </span>
  `).join('');
}

function renderControls(ctx, refs, state) {
  const hasSession = Boolean(state.latestSession?.id);
  const isStopped = ['stopped', 'closed'].includes(String(state.latestSession?.status || state.latestSession?.runtime_status || '').toLowerCase());
  const surfaceFocused = browserSurfaceIsFocused(ctx);
  const canControl = browserSurfaceCanControl(ctx, state);
  for (const button of [refs.go, refs.stop, refs.reload, refs.back, refs.forward, refs.sendToCtox, refs.upload, refs.newTab, refs.clipboardCopy, refs.clipboardPaste, refs.clipboardClear]) {
    if (!button) continue;
    button.disabled = !hasSession || isStopped || !canControl;
  }
  // The address bar is also the recovery path for a stale or disconnected
  // session. Keep it operable; submit starts a fresh leased session when the
  // current surface cannot safely navigate the existing one.
  if (refs.go) refs.go.disabled = false;
  if (refs.controllerAcquire) {
    refs.controllerAcquire.disabled = !hasSession || isStopped || !surfaceFocused || canControl;
  }
  if (refs.controllerRelease) {
    refs.controllerRelease.disabled = !hasSession || isStopped || !canControl;
  }
  for (const button of [refs.observerGrant, refs.observerRevoke]) {
    if (!button) continue;
    button.disabled = !hasSession || isStopped || !surfaceFocused;
  }
  refs.canvas?.setAttribute('aria-disabled', canControl ? 'false' : 'true');
}

function renderNotice(refs, notice) {
  if (!refs.notice) return;
  refs.notice.hidden = !notice;
  refs.notice.textContent = notice || '';
}

async function renderFrame(refs, frame, state) {
  if (!frame?.data || state.drawing) {
    refs.empty.hidden = Boolean(frame?.data);
    if (!frame?.data) refs.empty.textContent = frameEmptyText(state);
    return;
  }
  state.drawing = true;
  try {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = `data:${frame.mime_type || 'image/png'};base64,${frame.data}`;
    });
    refs.canvas.width = Number(frame.width || VIEWPORT.width);
    refs.canvas.height = Number(frame.height || VIEWPORT.height);
    const ctx = refs.canvas.getContext('2d');
    ctx.clearRect(0, 0, refs.canvas.width, refs.canvas.height);
    ctx.drawImage(img, 0, 0, refs.canvas.width, refs.canvas.height);
    refs.empty.hidden = true;
  } catch (error) {
    console.error('[browser] frame render failed', error);
    refs.empty.hidden = false;
    refs.empty.textContent = t('frameRenderFailed', 'Die Seite konnte nicht angezeigt werden.');
  } finally {
    state.drawing = false;
  }
}

function frameEmptyText(state) {
  const session = state.latestSession;
  const commandError = commandErrorMessage(state.latestCommand);
  if (commandError) return commandError;
  if (!session) return t('frameOpenNew', 'Oeffne ein neues Browser-Fenster');
  const status = String(session.runtime_status || session.status || '').toLowerCase();
  const runtimeError = browserSessionError(session);
  if (['failed', 'error', 'blocked'].includes(status)) {
    return runtimeError || t('frameStartFailed', 'Der Browser konnte nicht gestartet werden');
  }
  if (['disconnected', 'offline'].includes(status)) {
    return runtimeError || 'Kein laufender Browser-Prozess. Starten Sie die Sitzung erneut.';
  }
  if (status === 'stopped' || status === 'closed') return t('frameClosed', 'Browser-Fenster geschlossen');
  if (status === 'requested' || status === 'starting' || status === 'pending_command') return t('frameStarting', 'Browser wird gestartet');
  return t('frameLoading', 'Browser-Inhalt wird geladen');
}

function browserUiState(session) {
  if (!session) return 'offline';
  const raw = String(session.runtime_status || session.status || '').toLowerCase();
  if (['active', 'running', 'capturing'].includes(raw)) return 'ready';
  if (['requested', 'starting', 'pending_command', 'pending_sync'].includes(raw)) return 'starting';
  if (raw === 'blocked') return 'blocked';
  if (['stopped', 'closed', 'disconnected', 'offline'].includes(raw)) return 'offline';
  if (['failed', 'error'].includes(raw)) return 'error';
  return raw ? 'waiting' : 'offline';
}

function browserStatusLabel(session) {
  const state = browserUiState(session);
  if (state === 'ready') return t('statusReady', 'Bereit');
  if (state === 'starting') return t('statusStarting', 'Startet');
  if (state === 'waiting') return t('statusConnecting', 'Verbindet');
  if (state === 'blocked') return 'Aktion erforderlich';
  if (state === 'error') return t('statusError', 'Fehler');
  return t('statusDisconnected', 'Nicht verbunden');
}

function browserDisplayTitle(tab, session, url = '') {
  const raw = String(tab?.title || session?.title || '').trim();
  if (!raw || /^remote browser$/i.test(raw)) {
    return url ? browserUrlLabel(url) : 'Browser';
  }
  return raw;
}

function browserUrlLabel(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname || 'Browser';
  } catch {
    return 'Browser';
  }
}

function renderDiagnostics(refs, frame, inputs, command, commands = [], handoffTasks = []) {
  refs.frameId.textContent = frame?.id || '-';
  refs.frameSeq.textContent = frame?.seq == null ? '-' : String(frame.seq);
  refs.frameSize.textContent = frame?.size_bytes ? formatBytes(frame.size_bytes) : '-';
  refs.frameTime.textContent = frame?.captured_at_ms ? formatTime(frame.captured_at_ms) : '-';
  const pending = inputs.filter((event) => event.status === 'pending').length;
  const consumed = inputs.filter((event) => event.status === 'consumed').length;
  refs.inputState.textContent = `${pending} pending / ${consumed} consumed`;
  refs.commandState.textContent = command ? commandSummary(command) : '-';
  if (refs.commandHistory) {
    refs.commandHistory.innerHTML = commands.length
      ? commands.map((item) => `
          <div class="browser-command-row">
            <strong>${escapeHtml(item.command_type || item.type || 'browser')}</strong>
            <span>${escapeHtml(item.status || 'pending')}</span>
            <span>${escapeHtml(formatTime(item.updated_at_ms || item.created_at_ms))}</span>
            <span>${escapeHtml(commandErrorMessage(item) || item.record_id || '')}</span>
          </div>
        `).join('')
      : '<div class="browser-command-row"><span>No browser commands</span></div>';
  }
  if (refs.handoffHistory) {
    refs.handoffHistory.innerHTML = handoffTasks.length
      ? handoffTasks.map((item) => `
          <div class="browser-command-row">
            <strong>${escapeHtml(item.title || 'Browser context')}</strong>
            <span>${escapeHtml(item.status || item.route_status || 'queued')}</span>
            <span>${escapeHtml(formatTime(item.updated_at_ms))}</span>
            <span>${escapeHtml(item.id || item.command_id || '')}</span>
          </div>
        `).join('')
      : '<div class="browser-command-row"><span>No CTOX handoffs</span></div>';
  }
}

function commandSummary(command) {
  const type = command.command_type || command.type || 'browser';
  const status = command.status || 'pending';
  const error = commandErrorMessage(command);
  if (error) return `${type} failed: ${error}`;
  return `${type} ${status}`;
}

function commandErrorMessage(command) {
  if (!command) return '';
  const status = String(command.status || '').toLowerCase();
  const error = command.error || command.result?.error || command.payload?.error || '';
  if (status === 'failed' || error) return String(error || 'Die Browser-Aktion ist fehlgeschlagen.');
  return '';
}

async function upsertDoc(collection, doc) {
  if (!collection) throw new Error('Browser collection is not registered');
  const next = { ...doc };
  delete next._rev;
  delete next._meta;
  const existing = await collection.findOne(next.id).exec();
  if (existing?.incrementalPatch) {
    await existing.incrementalPatch(next);
    return;
  }
  if (existing) {
    await existing.patch(next);
  } else if (typeof collection.upsert === 'function') {
    await collection.upsert(next);
  } else {
    await collection.insert(next);
  }
}

async function patchDoc(collection, id, patch) {
  if (!collection || !id) return;
  const existing = await collection.findOne(id).exec();
  if (existing?.incrementalPatch) {
    await existing.incrementalPatch(patch);
  }
}

function normalizeUrl(value) {
  const trimmed = String(value || '').trim();
  if (!trimmed) return 'https://example.com';
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
}

function browserSessionIdFromArgs(value) {
  const sessionId = String(value?.session_id || value?.sessionId || '').trim();
  return /^browser_session_[a-z0-9_-]+$/i.test(sessionId) ? sessionId : '';
}

function browserSurfaceIsFocused(ctx) {
  if (globalThis.document?.visibilityState === 'hidden') return false;
  if (globalThis.document?.hasFocus?.() === false) return false;
  const surface = ctx?.host?.closest?.('.shell-window');
  return Boolean(surface?.classList.contains('is-focused'));
}

// Warum die Bedienung gesperrt ist — im Klartext, nicht als Schweigen.
// Die Reihenfolge folgt der Pruefung in browserSurfaceCanControl.
function eingabeSperrgrund(ctx, state, now = Date.now()) {
  if (!browserSurfaceIsFocused(ctx)) return 'Das Browserfenster hat nicht den Fokus.';
  const session = state?.latestSession;
  if (!session?.id) return 'Keine aktive Sitzung.';
  const actorIds = browserActorIds(ctx?.session);
  if (!actorIds.length) return 'Kein angemeldeter Nutzer.';
  if (!actorIds.includes(String(session.controller_user_id || ''))) {
    return 'Die Sitzung wird gerade von jemand anderem gesteuert.';
  }
  const leaseId = String(session.controller_lease_id || '').trim();
  if (!leaseId) return 'Keine Steuerungsberechtigung für diese Sitzung.';
  if (session.controller_lease_id !== state.controllerLeaseId) {
    return 'Die Steuerung wurde an ein anderes Fenster übergeben.';
  }
  const expiresAt = Number(session.controller_lease_expires_at_ms || 0);
  if (!Number.isFinite(expiresAt) || expiresAt <= now) {
    return 'Die Steuerung ist abgelaufen.';
  }
  return '';
}

function setzeEingabeHinweis(ctx, state, text) {
  if (!state) return;
  if (state.eingabeHinweis === text) return;
  state.eingabeHinweis = text;
  try { renderBrowserSurface?.(ctx, state); } catch {}
}

// Gedrosselt: bei gehaltener Maus entstehen sonst hunderte gleiche Zeilen.
function meldeEingabeGesperrt(ctx, state, type) {
  const grund = eingabeSperrgrund(ctx, state) || 'Bedienung gesperrt.';
  const jetzt = Date.now();
  if (state.letzteSperrmeldung !== grund || jetzt - (state.letzteSperrmeldungMs || 0) > 5000) {
    console.warn('[browser] Eingabe gesperrt', { grund, type });
    state.letzteSperrmeldung = grund;
    state.letzteSperrmeldungMs = jetzt;
  }
  setzeEingabeHinweis(ctx, state, grund);
}

function browserSurfaceCanControl(ctx, state, now = Date.now()) {
  if (!browserSurfaceIsFocused(ctx)) return false;
  const session = state?.latestSession;
  const actorIds = browserActorIds(ctx?.session);
  const expiresAt = Number(session?.controller_lease_expires_at_ms || 0);
  return Boolean(
    session?.id
      && actorIds.length
      && actorIds.includes(String(session.controller_user_id || ''))
      && String(session.controller_lease_id || '').trim()
      && session.controller_lease_id === state.controllerLeaseId
      && Number.isFinite(expiresAt)
      && expiresAt > now
  );
}

function browserCommandRequiresController(commandType, session) {
  if (!session?.id) return false;
  return ![
    'browser.session.start',
    'browser.controller.acquire',
    'browser.observer.grant',
    'browser.observer.revoke',
  ].includes(commandType);
}

function shouldRenewControllerLease(session, actorId, now = Date.now(), options = {}) {
  const {
    documentVisible = true,
    documentFocused = true,
    surfaceFocused = true,
    renewInFlight = false,
    controllerLeaseId = '',
  } = options;
  if (!documentVisible || !documentFocused || !surfaceFocused || renewInFlight) return false;
  const actorIds = Array.isArray(actorId)
    ? actorId.map((value) => String(value || '')).filter(Boolean)
    : [String(actorId || '')].filter(Boolean);
  if (!session?.id || !actorIds.includes(String(session.controller_user_id || ''))) return false;
  if (!String(session.controller_lease_id || '').trim()) return false;
  if (session.controller_lease_id !== controllerLeaseId) return false;
  const expiresAt = Number(session.controller_lease_expires_at_ms || 0);
  // Eine ABGELAUFENE Pacht wird bewusst NICHT erneuert — der Server weist die
  // Erneuerung ab, und alle 30 s ein Fehlschlag waere eine Endlosschleife.
  // Der Wächter darunter haelt das fest. Fuer diesen Fall ist NEU HOLEN
  // zustaendig, nicht Erneuern: siehe shouldReacquireControllerLease.
  if (!Number.isFinite(expiresAt) || expiresAt <= now) return false;
  return expiresAt - now <= 75_000;
}

// Abgelaufene Pacht: neu HOLEN statt zu erneuern.
//
// Bis 13.08.2026 gab es diesen Weg nicht. Die Erneuerung schloss den
// abgelaufenen Zustand zu Recht aus, aber niemand holte die Pacht danach
// zurueck — der Zustand, der die Reparatur am dringendsten braucht, hatte
// keine. Zusammen mit der Bedingung "nur bei sichtbarem, fokussiertem Fenster"
// ergab das eine Falle ohne Ausgang: Fenster wegklicken, Pacht laeuft
// planmaessig ab, zurueckkommen — und sie wird nie wieder angefasst.
// Auf der Kundeninstanz gemessen: beide Sitzungen "active", beide Pachten
// abgelaufen, null Eingabe-Ereignisse in zehn Minuten aktiver Bedienung.
//
// Anders als die Erneuerung braucht das Neuholen KEIN fokussiertes Fenster —
// sonst bliebe genau die Falle bestehen. Es verlangt aber, dass die Sitzung
// uns gehoert, und es laeuft nur, wenn nicht schon ein Versuch unterwegs ist.
function shouldReacquireControllerLease(session, actorId, now = Date.now(), options = {}) {
  const { reacquireInFlight = false, lastReacquireAtMs = 0 } = options;
  if (reacquireInFlight) return false;
  // Nach einem Fehlschlag nicht sofort wieder — sonst entsteht genau die
  // Endlosschleife, die der Wächter der Erneuerung verhindert.
  if (now - Number(lastReacquireAtMs || 0) < 10_000) return false;
  const actorIds = Array.isArray(actorId)
    ? actorId.map((value) => String(value || '')).filter(Boolean)
    : [String(actorId || '')].filter(Boolean);
  if (!session?.id || !actorIds.length) return false;
  // Fremd gesteuerte Sitzungen nicht an uns reissen.
  const controller = String(session.controller_user_id || '');
  if (controller && !actorIds.includes(controller)) return false;
  const expiresAt = Number(session.controller_lease_expires_at_ms || 0);
  return !Number.isFinite(expiresAt) || expiresAt <= now;
}

function newBrowserControllerLeaseId() {
  return globalThis.crypto?.randomUUID?.()
    || `browser-lease-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
}

function browserActorIds(session) {
  const user = session?.user && typeof session.user === 'object' ? session.user : {};
  return [...new Set([
    user.id,
    user.email,
    user.login,
    session?.userId,
  ].map((value) => String(value || '').trim()).filter(Boolean))];
}

function debounce(fn, delayMs) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delayMs);
  };
}

function formatTime(ms) {
  try {
    return new Date(Number(ms)).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch (_) {
    return '-';
  }
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function titleCase(value) {
  const text = String(value || '').replace(/[_-]+/g, ' ');
  if (!text) return '';
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export const __browserTestHooks = {
  normalizeUrl,
  browserSessionIdFromArgs,
  formatBytes,
  titleCase,
  userSessionPrefix,
  selectedViewport,
  browserAuthRequestFromArgs,
  shouldRenewControllerLease,
  shouldReacquireControllerLease,
  browserSitzungZustand,
  eingabeSperrgrund,
  browserCommandRequiresController,
  browserSurfaceIsFocused,
  browserSurfaceCanControl,
  browserActorIds,
  mergeRequestedSession,
  mergeRequestedDocument,
  browserSessionError,
  browserSessionIsLive,
  browserSessionNeedsStart,
  browserStartErrorIsRetryable,
  newBrowserControllerLeaseId,
  filterSessionsForView,
  browserSessionViewCounts,
  browserSessionBand,
  browserUiState,
  browserStatusLabel,
  frameEmptyText,
  sessionListSignature,
  browserWorkbenchVisible,
  browserSessionShardMeta,
  buildBrowserSessionsExport,
  parseBrowserSessionsImport,
  sessionRenderList,
  sessionTabCounts,
  browserInputTrace,
  browserInputPayload,
  eventModifiers,
  pointerButton,
  pointerButtons,
  pointerClickCount,
  keyboardText,
  submitBrowserNav,
  writePointerInput,
  writeKeyboardInput,
  writeInputEvent,
  installInputHandlers,
};

async function ensureStyles() {
  const href = new URL(`./index.css?v=${STYLE_BUILD}`, import.meta.url).href;
  if ([...document.querySelectorAll('link[rel="stylesheet"]')].some((link) => link.href === href)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  document.head.appendChild(link);
}

// Translate static markup: data-t (textContent) and data-t-aria (aria-label).
// German markup text is the fallback when a key is missing.
function applyTranslations(root) {
  root.querySelectorAll('[data-t]').forEach((el) => {
    el.textContent = t(el.dataset.t, el.textContent.trim());
  });
  root.querySelectorAll('[data-t-aria]').forEach((el) => {
    el.setAttribute('aria-label', t(el.dataset.tAria, el.getAttribute('aria-label') || ''));
  });
}
