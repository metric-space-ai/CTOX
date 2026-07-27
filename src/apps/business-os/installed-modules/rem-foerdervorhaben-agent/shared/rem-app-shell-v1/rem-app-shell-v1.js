(function () {
  "use strict";

  const DEFAULT_THEME_KEY = "rem.shell.v1.theme";
  const DEFAULT_RESIZE_KEY = "rem.shell.v1.resize";

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function numberFromCss(value, fallback) {
    const parsed = Number(String(value || "").replace("px", "").trim());
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function safeLocalStorage() {
    try {
      const storage = window.localStorage;
      const probe = "__rem_shell_probe__";
      storage.setItem(probe, "1");
      storage.removeItem(probe);
      return storage;
    } catch (_) {
      const memory = new Map();
      return {
        getItem(key) {
          key = String(key);
          return memory.has(key) ? memory.get(key) : null;
        },
        setItem(key, value) {
          memory.set(String(key), String(value));
        },
        removeItem(key) {
          memory.delete(String(key));
        },
        clear() {
          memory.clear();
        },
        key(index) {
          return Array.from(memory.keys())[index] || null;
        },
        get length() {
          return memory.size;
        },
      };
    }
  }

  const storage = safeLocalStorage();

  function initTheme(options = {}) {
    const root = options.root || document.documentElement;
    const storageKey = options.storageKey || DEFAULT_THEME_KEY;
    const lightButton = document.querySelector(options.lightSelector || "#lightTheme");
    const darkButton = document.querySelector(options.darkSelector || "#darkTheme");
    const initial = storage.getItem(storageKey) || root.dataset.theme || "light";

    function setTheme(theme) {
      const next = theme === "dark" ? "dark" : "light";
      root.dataset.theme = next;
      storage.setItem(storageKey, next);
      if (lightButton) lightButton.setAttribute("aria-pressed", String(next === "light"));
      if (darkButton) darkButton.setAttribute("aria-pressed", String(next === "dark"));
    }

    if (lightButton) lightButton.addEventListener("click", () => setTheme("light"));
    if (darkButton) darkButton.addEventListener("click", () => setTheme("dark"));
    setTheme(initial);
    return { setTheme };
  }

  function initColumnResizers(root = document, options = {}) {
    const frames = Array.from(root.querySelectorAll("[data-resize-frame]"));
    const storagePrefix = options.storagePrefix || DEFAULT_RESIZE_KEY;

    frames.forEach((frame, index) => {
      const frameKey = frame.dataset.resizeKey || frame.id || `frame-${index}`;
      const handles = Array.from(frame.querySelectorAll(".ctox-column-resizer[data-resizer-var]"));

      handles.forEach((handle) => {
        const cssVar = handle.dataset.resizerVar;
        const side = handle.dataset.resizer || "right";
        const min = Number(handle.dataset.resizerMin || 260);
        const max = Number(handle.dataset.resizerMax || 560);
        const fallback = Number(handle.dataset.resizerDefault || numberFromCss(getComputedStyle(frame).getPropertyValue(cssVar), side === "left" ? 300 : 360));
        const storageKey = `${storagePrefix}.${frameKey}.${cssVar}`;
        let startX = 0;
        let startValue = 0;

        function setValue(value, persist = false) {
          const next = clamp(Math.round(value), min, max);
          frame.style.setProperty(cssVar, `${next}px`);
          handle.setAttribute("aria-valuenow", String(next));
          if (persist) storage.setItem(storageKey, String(next));
        }

        handle.setAttribute("role", "separator");
        handle.setAttribute("tabindex", "0");
        handle.setAttribute("aria-orientation", "vertical");
        handle.setAttribute("aria-valuemin", String(min));
        handle.setAttribute("aria-valuemax", String(max));
        setValue(Number(storage.getItem(storageKey)) || fallback);

        handle.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          startX = event.clientX;
          startValue = numberFromCss(getComputedStyle(frame).getPropertyValue(cssVar), fallback);
          handle.classList.add("is-active");
          document.body.classList.add("is-resizing");
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp);
        });

        handle.addEventListener("keydown", (event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const current = numberFromCss(getComputedStyle(frame).getPropertyValue(cssVar), fallback);
          const sign = side === "left" ? 1 : -1;
          const next = event.key === "Home" ? min : event.key === "End" ? max : current + (event.key === "ArrowRight" ? 24 * sign : -24 * sign);
          setValue(next, true);
        });

        function onMove(event) {
          const delta = event.clientX - startX;
          setValue(startValue + (side === "left" ? delta : -delta));
        }

        function onUp() {
          setValue(numberFromCss(getComputedStyle(frame).getPropertyValue(cssVar), fallback), true);
          handle.classList.remove("is-active");
          document.body.classList.remove("is-resizing");
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
        }
      });
    });
  }

  function initDisclosure(toggle, panel, options = {}) {
    const toggleElement = typeof toggle === "string" ? document.querySelector(toggle) : toggle;
    const panelElement = typeof panel === "string" ? document.querySelector(panel) : panel;
    if (!toggleElement || !panelElement) return null;

    const className = options.className || "is-open";
    const initial = Boolean(options.initial);

    function setOpen(open) {
      panelElement.classList.toggle(className, open);
      toggleElement.setAttribute("aria-expanded", String(open));
      if (options.hiddenAttribute) panelElement.hidden = !open;
    }

    toggleElement.addEventListener("click", () => setOpen(!panelElement.classList.contains(className)));
    setOpen(initial);
    return { setOpen };
  }

  function textOf(node) {
    return (node?.textContent || "").replace(/\s+/g, " ").trim();
  }

  function findAgentPane(root = document) {
    return root.querySelector(".agent-pane, #agentPanelCard, [data-agent-pane], aside[aria-label*='Agent' i], .ctox-workspace--two-pane > aside.ctox-pane");
  }

  function findProgressPercent(pane) {
    const candidates = [
      "#progressPercent",
      "#progressPct",
      "[data-progress-value]",
      ".progress-percent",
      ".rem-progress-meta strong",
      ".progress-row strong",
    ];
    for (const selector of candidates) {
      const node = pane.querySelector(selector);
      const match = textOf(node).match(/(\d{1,3})\s*%/);
      if (match) return Number(match[1]);
    }
    const match = textOf(pane).match(/(\d{1,3})\s*%/);
    return match ? Number(match[1]) : null;
  }

  function inferAgentState(pane) {
    const explicit = pane.dataset.agentState;
    if (["idle", "ready", "running", "question", "done", "error"].includes(explicit)) return explicit;

    const text = textOf(pane).toLowerCase();
    if (/(fehler|error|abgebrochen|instabil)/.test(text)) return "error";
    if (/(rückfrage offen|reviewfragen|frage\s+\d|fragen offen|wartet auf antwort)/.test(text)) return "question";
    if (/(agent läuft|analyse läuft|läuft|wird erstellt|wird ermittelt|wird vorbereitet|recherche läuft|report wird erstellt|dokumentation wird erstellt)/.test(text)) return "running";
    if (/(bereit für export|export bereit|fertig)/.test(text)) return "done";
    if (/(bereit|wartet)/.test(text)) return "ready";
    return "idle";
  }

  function isVisibleElement(node) {
    if (!node || node.hidden || node.classList.contains("hidden")) return false;
    const styles = window.getComputedStyle ? window.getComputedStyle(node) : null;
    if (styles && (styles.display === "none" || styles.visibility === "hidden")) return false;
    const rect = node.getBoundingClientRect?.();
    return !rect || (rect.width > 0 && rect.height > 0);
  }

  function ensureFallbackProgress(pane, state) {
    const existing = pane.querySelector(".rem-agent-fallback-progress");
    const shouldShow = state === "running";
    const realProgress = Array.from(pane.querySelectorAll("#progressShell, #progressBlock, .progress-wrap, .progress-block, .rem-progress"))
      .filter((node) => !node.classList.contains("rem-agent-fallback-progress"))
      .some(isVisibleElement);

    if (!shouldShow || realProgress) {
      if (existing) existing.hidden = true;
      return;
    }

    const host = pane.querySelector(".agent-body, .ctox-pane-scroll, .ctox-pane-body") || pane;
    const status =
      host.querySelector(".agent-headline, .agent-title, .rem-agent-status, #agentStatus, #agentState, #agentTitle")?.closest(".agent-headline, .agent-title, .rem-agent-status, section, div")
      || host.firstElementChild;
    const block = existing || document.createElement("section");
    if (!existing) {
      block.className = "rem-agent-fallback-progress rem-progress is-visible";
      block.setAttribute("aria-label", "Fortschritt");
      block.innerHTML = `
        <div class="rem-progress-meta"><span>Analyse läuft</span><strong>läuft</strong></div>
        <div class="rem-progress-track"><span class="rem-progress-fill"></span></div>`;
      if (status?.parentNode === host) status.insertAdjacentElement("afterend", block);
      else host.prepend(block);
    }
    block.hidden = false;
    block.classList.add("is-running");
  }

  function normalizeProgressVisibility(pane) {
    const state = inferAgentState(pane);
    const isRunning = state === "running";
    const isQuestion = state === "question";
    const isDone = state === "done";
    pane.dataset.agentStateNormalized = state;

    pane.classList.toggle("is-running", isRunning);
    pane.classList.toggle("has-question", isQuestion);
    pane.classList.toggle("is-done", isDone);

    pane.querySelectorAll("#progressShell, #progressBlock, .progress-wrap, .progress-block, .rem-progress").forEach((block) => {
      block.classList.toggle("is-running", isRunning && !block.hidden);
    });
    ensureFallbackProgress(pane, state);
  }

  function setAgentState(paneOrState, maybeState) {
    const pane = typeof paneOrState === "string" ? findAgentPane(document) : paneOrState;
    const state = typeof paneOrState === "string" ? paneOrState : maybeState;
    if (!pane) return null;
    pane.dataset.agentState = ["idle", "ready", "running", "question", "done", "error"].includes(state) ? state : "idle";
    normalizeProgressVisibility(pane);
    return pane;
  }

  function normalizeAgentHeader(pane) {
    const header = pane.querySelector(".agent-header, .ctox-pane-header, .pane-header, header");
    if (!header) return;
    const titleContainer = header.querySelector(".ctox-pane-title, .pane-title");
    const title = titleContainer?.querySelector("h2, strong") || header.querySelector("h2, strong") || titleContainer;
    if (title && !titleContainer && !title.classList.contains("ctox-pane-title")) title.classList.add("ctox-pane-title");
    if (title && /agent/i.test(textOf(title)) && title.children.length === 0 && title.textContent !== "Agent") {
      title.textContent = "Agent";
    }
    const redundant = header.querySelectorAll("#agentStateLabel, .ctox-pane-kicker, .agent-kicker, .agent-state-label");
    redundant.forEach((node) => {
      if (!node.hidden) node.hidden = true;
      if (node.getAttribute("aria-hidden") !== "true") node.setAttribute("aria-hidden", "true");
    });
  }

  function normalizeLogItems(pane) {
    const lists = pane.querySelectorAll("#agentLog, #agentLogList, #activityLog, #activityList, .rem-activity-list, .activity-list");
    const timeRe = /^(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:·|-|:)?\s*(.*)$/;
    lists.forEach((list) => {
      list.classList.add("rem-activity-list");
      Array.from(list.children).forEach((item) => {
        if (item.dataset.remLogNormalized === "1") return;
        const raw = textOf(item);
        if (!raw) return;
        const match = raw.match(timeRe);
        if (!match) return;
        const message = match[2] || raw;
        item.textContent = "";
        const time = document.createElement("span");
        time.className = "rem-activity-time";
        time.textContent = match[1].length === 5 ? `${match[1]}:00` : match[1];
        const body = document.createElement("span");
        body.textContent = message.replace(/^·\s*/, "");
        item.append(time, body);
        item.dataset.remLogNormalized = "1";
      });
    });
  }

  function normalizeInternalStatusNodes(pane) {
    pane.querySelectorAll("#agentSummary, #runStatus, #currentStep, .agent-internal-status").forEach((node) => {
      node.classList.add("rem-agent-internal-status");
      node.setAttribute("aria-hidden", "true");
      node.hidden = true;
    });
  }

  function normalizeActivityVisibility(pane) {
    const lists = pane.querySelectorAll("#agentLog, #agentLogList, #activityLog, #activityList, .rem-activity-list, .activity-list");
    lists.forEach((list) => {
      const items = Array.from(list.children).filter((item) => textOf(item));
      const section =
        list.closest("[data-agent-activity]") ||
        list.closest(".activity") ||
        list.closest(".agent-activity") ||
        list.closest(".agent-section");
      const hasItems = items.length > 0;
      const looseHead = !section && list.previousElementSibling?.matches?.(".activity-head, .activity-header, .log-header")
        ? list.previousElementSibling
        : null;
      [section, looseHead, !section ? list : null].filter(Boolean).forEach((node) => {
        node.classList.toggle("rem-agent-empty-activity", !hasItems);
        if (!hasItems) node.setAttribute("aria-hidden", "true");
        else node.removeAttribute("aria-hidden");
      });
    });
  }

  function standardizeAgentPane(root = document) {
    const pane = findAgentPane(root);
    if (!pane) return null;
    pane.classList.add("agent-pane", "rem-agent-standard");
    normalizeAgentHeader(pane);
    normalizeProgressVisibility(pane);
    normalizeLogItems(pane);
    normalizeInternalStatusNodes(pane);
    normalizeActivityVisibility(pane);

    if (pane.dataset.remAgentObserver === "1") return pane;
    pane.dataset.remAgentObserver = "1";
    let scheduled = false;
    const observer = new MutationObserver(() => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        normalizeAgentHeader(pane);
        normalizeProgressVisibility(pane);
        normalizeLogItems(pane);
        normalizeInternalStatusNodes(pane);
        normalizeActivityVisibility(pane);
      });
    });
    observer.observe(pane, { childList: true, subtree: true, characterData: true });
    return pane;
  }

  function createAgentInstructionComposer(options = {}) {
    const pane = options.pane || findAgentPane(options.root || document);
    if (!pane) return null;
    if (pane._remInstructionComposer) return pane._remInstructionComposer;

    const host = options.host
      || pane.querySelector(".agent-body, .ctox-pane-scroll, .ctox-pane-body")
      || pane;
    const activity = host.querySelector(".activity, [data-agent-activity], #activityList, #agentLog, .rem-activity-list")?.closest(".activity, section, .agent-activity")
      || host.querySelector(".activity, [data-agent-activity]");
    const section = document.createElement("section");
    section.className = "rem-agent-instruction";
    section.hidden = true;
    section.setAttribute("aria-label", options.label || "Anweisung an den Agenten");
    section.innerHTML = `
      <div class="rem-agent-instruction-head">
        <strong>${options.title || "Anweisung"}</strong>
        <button class="rem-agent-instruction-clear" type="button" aria-label="Bezug entfernen" title="Bezug entfernen" hidden>&times;</button>
      </div>
      <div class="rem-agent-instruction-context" hidden></div>
      <div class="rem-agent-instruction-row">
        <textarea rows="2" placeholder="${options.placeholder || "Änderung kurz beschreiben ..."}" aria-label="Anweisung"></textarea>
        <button class="ctox-button ctox-button--primary rem-agent-instruction-submit" type="button">Anwenden</button>
      </div>`;
    if (activity && activity.parentNode === host) host.insertBefore(section, activity);
    else host.append(section);

    const textarea = section.querySelector("textarea");
    const submit = section.querySelector(".rem-agent-instruction-submit");
    const contextNode = section.querySelector(".rem-agent-instruction-context");
    const clear = section.querySelector(".rem-agent-instruction-clear");
    let context = null;
    let busy = false;

    function setContext(next) {
      context = next || null;
      const label = context?.label || context?.title || context?.section || "";
      contextNode.textContent = label ? `Bezug: ${label}` : "";
      contextNode.hidden = !label;
      clear.hidden = !context;
      section.hidden = false;
      textarea.focus({ preventScroll: false });
    }

    function clearContext() {
      context = null;
      contextNode.textContent = "";
      contextNode.hidden = true;
      clear.hidden = true;
    }

    function setBusy(next) {
      busy = Boolean(next);
      textarea.disabled = busy;
      submit.disabled = busy;
      submit.textContent = busy ? "Wird angewendet ..." : "Anwenden";
      section.classList.toggle("is-busy", busy);
    }

    async function send() {
      const instruction = textarea.value.trim();
      if (!instruction || busy || typeof options.onSubmit !== "function") return;
      setBusy(true);
      try {
        await options.onSubmit({ instruction, context });
        textarea.value = "";
        clearContext();
      } finally {
        setBusy(false);
      }
    }

    submit.addEventListener("click", send);
    textarea.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        send();
      }
    });
    clear.addEventListener("click", clearContext);

    const controller = {
      root: section,
      setVisible(visible = true) { section.hidden = !visible; },
      setBusy,
      setContext,
      clearContext,
      getContext() { return context; },
      focus() { section.hidden = false; textarea.focus({ preventScroll: false }); },
    };
    pane._remInstructionComposer = controller;
    return controller;
  }

  function attachAgentContextAction(element, context, composer, options = {}) {
    if (!element || !composer) return null;
    element.classList.add("rem-context-target");
    const existing = element.querySelector(":scope > .rem-context-action");
    if (existing) existing.remove();
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rem-context-action";
    button.setAttribute("aria-label", options.label || "Änderung an diesem Inhalt anweisen");
    button.title = options.label || "Änderung an diesem Inhalt anweisen";
    button.innerHTML = '<span class="rem-context-action-icon" aria-hidden="true"></span>';
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      composer.setContext(typeof context === "function" ? context(element) : context);
    });
    element.append(button);
    return button;
  }

  function initDefaultShell() {
    initTheme();
    initColumnResizers();
    standardizeAgentPane();
    document.querySelectorAll("[data-rem-options-toggle][data-rem-options-panel]").forEach((toggle) => {
      initDisclosure(toggle, toggle.dataset.remOptionsPanel, { hiddenAttribute: true });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDefaultShell, { once: true });
  } else {
    initDefaultShell();
  }

  window.RemAppShellV1 = {
    initTheme,
    initColumnResizers,
    initDisclosure,
    standardizeAgentPane,
    setAgentState,
    createAgentInstructionComposer,
    attachAgentContextAction,
    normalizeActivityVisibility,
    initDefaultShell,
  };
})();
