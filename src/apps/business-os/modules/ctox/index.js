import {
  COMMANDS,
  queries,
  readCollection,
  mergeBundleWithCommands,
  commandTaskFromProjection,
  normalizeExecutionProgress,
  mayReadPrivate,
  mayControl,
  taskActionReason,
} from "./model.js?v=20260906-crew-home-v1";
import {
  patchMarkup,
  statusMarkup,
  homeMarkup,
  workplaceMarkup,
  queueMarkup,
  profileMarkup,
  translate,
  escapeHtml as e,
  AXES,
} from "./view.js?v=20260906-crew-home-v1";
import { readPages } from "./paging.js?v=20260906-crew-home-v1";
const BUILD = "20260906-crew-home-v1";
const views = {
  home: homeMarkup,
  workplace: workplaceMarkup,
  queue: queueMarkup,
};
const appendUnique = (old, rows) => [
  ...new Map([...old, ...rows].map((row) => [row.id, row])).values(),
];

export async function mount(ctx) {
  const lang = ctx.locale === "en" ? "en" : "de";
  const [markup, messages] = await Promise.all([
    fetch(new URL(`./index.html?v=${BUILD}`, import.meta.url)).then((r) => {
      if (!r.ok) throw new Error(`markup: ${r.status}`);
      return r.text();
    }),
    fetch(new URL(`./locales/${lang}.json?v=${BUILD}`, import.meta.url)).then((r) => {
      if (!r.ok) throw new Error(`locale: ${r.status}`);
      return r.json();
    }),
  ]);
  ctx.host.innerHTML = markup;
  for (const file of ["./index.css", "../../shared/crew-creature.css"]) {
    const link = ctx.host.ownerDocument.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL(`${file}?v=${BUILD}`, import.meta.url).href;
    ctx.host.prepend(link);
  }
  const root = ctx.host.querySelector("[data-crew-home]");
  const s = {
    ctx,
    root,
    lang,
    messages,
    instanceId: crypto.randomUUID(),
    view: "home",
    members: [],
    tasks: [],
    events: [],
    runs: [],
    memberRuns: [],
    learnings: [],
    status: null,
    activeRows: [],
    terminalRows: [],
    commands: [],
    loading: true,
    connected: true,
    error: "",
    notice: "",
    selectedTaskId: String(ctx.args?.task_id || ctx.args?.taskId || ""),
    focusCommandId: String(ctx.args?.command_id || ctx.args?.commandId || ""),
    profileId: "",
    source: "",
    search: "",
    disposed: false,
    loadingData: false,
    rerun: false,
    pages: {
      members: 1,
      active: 1,
      terminal: 1,
      runs: 1,
      learnings: 1,
      memberRuns: 1,
    },
    commandSubscriptions: new Set(),
    moreMembers: false,
    moreActive: false,
    moreTerminal: false,
    moreRuns: false,
    moreLearnings: false,
    moreMemberRuns: false,
    cleanups: [],
    commandPending: false,
  };
  if (s.selectedTaskId || s.focusCommandId) s.view = "workplace";
  const t = (key, values) => translate(messages, key, values);
  root.querySelector("[data-app-title]").textContent = t("appTitle");
  root
    .querySelector("[data-navigation]")
    .setAttribute("aria-label", t("views"));
  const render = () => {
    if (s.disposed) return;
    patchMarkup(
      root.querySelector("[data-navigation]"),
      Object.keys(views)
        .map(
          (view) =>
            `<button type="button" data-view="${view}" data-key="nav-${view}" aria-current="${view === s.view ? "page" : "false"}">${e(t(view))}</button>`,
        )
        .join(""),
    );
    patchMarkup(root.querySelector("[data-status]"), statusMarkup(s));
    const notice = root.querySelector("[data-notice]");
    notice.hidden = !s.notice;
    notice.textContent = s.notice;
    patchMarkup(root.querySelector("[data-content]"), views[s.view](s));
    const profile = root.querySelector("[data-profile]");
    profile.hidden = !s.profileId;
    profile.setAttribute(
      "aria-labelledby",
      `crew-profile-title-${s.instanceId}`,
    );
    if (s.profileId) patchMarkup(profile, profileMarkup(s));
    clearTimeout(s.expiryTimer);
    const expires = s.events
      .map((event) => event.created_at_ms + 10001)
      .filter((value) => value > Date.now());
    if (expires.length)
      s.expiryTimer = setTimeout(render, Math.min(...expires) - Date.now() + 1);
  };
  const read = (name, query) => readCollection(ctx, name, query);
  const windowRead = (name, queryAt, kind) =>
    readPages(ctx, name, queryAt, s.pages[kind]);
  const readiness = () => {
    const names = [
      "ctox_crew_members",
      ...(mayReadPrivate(ctx) ? ["ctox_harness_status"] : []),
    ];
    const states = names
      .map((name) => ctx.sync?.collectionReadiness?.(name))
      .filter(Boolean);
    s.connected = !states.some(
      (state) =>
        state.state === "offline-pending" || state.state === "disconnected",
    );
    s.syncing = states.some(
      (state) =>
        state.syncing ||
        (state.ready === false && state.state !== "offline-pending"),
    );
    if (!s.connected) s.events = [];
  };
  async function refresh() {
    if (s.disposed) return;
    if (s.loadingData) {
      s.rerun = true;
      return;
    }
    s.loadingData = true;
    try {
      readiness();
      const filters = { source: s.source, search: s.search };
      const filterKey = JSON.stringify(filters);
      const [memberPage, status, activePage, terminalPage] = await Promise.all([
        windowRead("ctox_crew_members", queries.members, "members"),
        mayReadPrivate(ctx)
          ? read("ctox_harness_status", queries.status())
          : Promise.resolve([]),
        windowRead(
          "ctox_queue_tasks",
          (cursor) => queries.tasks({ ...filters, cursor }),
          "active",
        ),
        s.view === "queue"
          ? windowRead(
              "ctox_queue_tasks",
              (cursor) => queries.tasks({ ...filters, terminal: true, cursor }),
              "terminal",
            )
          : Promise.resolve({ rows: [], more: false }),
      ]);
      if (s.disposed) return;
      if (
        filterKey !== JSON.stringify({ source: s.source, search: s.search })
      ) {
        s.rerun = true;
        return;
      }
      const members = memberPage.rows,
        active = activePage.rows,
        terminal = terminalPage.rows;
      s.members = members;
      s.status = status[0] || null;
      s.activeRows = active;
      s.terminalRows = terminal;
      s.moreMembers = memberPage.more;
      s.moreActive = activePage.more;
      s.moreTerminal = terminalPage.more;
      s.memberCursor = members.at(-1);
      s.activeCursor = active.at(-1);
      s.terminalCursor = terminal.at(-1);
      const targetIds = [
        s.selectedTaskId,
        ...(s.status?.active_task_ids || []),
        ...members.map((m) => m.active_task_id),
      ].filter(Boolean);
      const visibleIds = new Set([...active, ...terminal].map((row) => row.id));
      const extra = (
        await Promise.all(
          [...new Set(targetIds)]
            .filter((id) => !visibleIds.has(id))
            .map((id) => read("ctox_queue_tasks", queries.task(id))),
        )
      ).flat();
      const ids = [
        ...new Set(
          [...active, ...terminal, ...extra]
            .map((row) => row.command_id)
            .filter(Boolean),
        ),
      ];
      if (s.focusCommandId && !ids.includes(s.focusCommandId))
        ids.push(s.focusCommandId);
      const [linkedPages, activeCommands] = await Promise.all([
        Promise.all(
          Array.from({ length: Math.ceil(ids.length / 200) }, (_, i) =>
            read(
              "business_commands",
              queries.commands(ids.slice(i * 200, (i + 1) * 200)),
            ),
          ),
        ),
        read("business_commands", queries.activeCommands(filters)),
      ]);
      const linked = linkedPages.flat();
      s.queueTaskIds = new Set([...active, ...terminal].map((row) => row.id));
      for (const command of activeCommands) {
        const task = commandTaskFromProjection(command);
        if (task) s.queueTaskIds.add(task.id);
      }
      s.commands = appendUnique(linked, activeCommands);
      s.tasks = mergeBundleWithCommands({}, s.commands, [
        ...active,
        ...terminal,
        ...extra,
      ]).queue;
      if (!s.selectedTaskId)
        s.selectedTaskId =
          s.tasks.find((task) => task.commandId === s.focusCommandId)?.id ||
          s.status?.active_task_ids?.[0] ||
          "";
      s.loading = false;
      s.error = "";
      render();
      await refreshDetails();
    } catch (error) {
      if (!s.disposed) {
        s.loading = false;
        s.error = String(error.message || error);
        render();
      }
    } finally {
      s.loadingData = false;
      if (s.rerun) {
        s.rerun = false;
        void refresh();
      }
    }
  }
  async function refreshDetails() {
    const taskId =
      s.view === "workplace"
        ? s.selectedTaskId
        : s.status?.active_task_ids?.[0];
    const profileId = s.profileId;
    if (!mayReadPrivate(ctx)) {
      s.events = [];
      s.runs = [];
      s.learnings = [];
      s.memberRuns = [];
      render();
      return;
    }
    const emptyPage = () => Promise.resolve({ rows: [], more: false });
    const [events, runPage, learningPage, memberRunPage] = await Promise.all([
      taskId
        ? read("ctox_harness_events", queries.events(taskId))
        : Promise.resolve([]),
      s.view === "workplace" && taskId
        ? windowRead(
            "ctox_runs",
            (cursor) => queries.runs({ taskId, cursor }),
            "runs",
          )
        : emptyPage(),
      profileId
        ? windowRead(
            "ctox_crew_learnings",
            (cursor) => queries.learnings(profileId, cursor),
            "learnings",
          )
        : emptyPage(),
      profileId
        ? windowRead(
            "ctox_runs",
            (cursor) => queries.runs({ memberId: profileId, cursor }),
            "memberRuns",
          )
        : emptyPage(),
    ]);
    if (
      s.disposed ||
      profileId !== s.profileId ||
      taskId !==
        (s.view === "workplace"
          ? s.selectedTaskId
          : s.status?.active_task_ids?.[0])
    )
      return;
    s.events = events;
    s.runs = runPage.rows;
    s.learnings = learningPage.rows;
    s.memberRuns = memberRunPage.rows;
    s.moreRuns = runPage.more;
    s.moreLearnings = learningPage.more;
    s.moreMemberRuns = memberRunPage.more;
    render();
  }
  function schedule() {
    clearTimeout(s.refreshTimer);
    s.refreshTimer = setTimeout(() => void refresh(), 80);
  }
  const collections = [
    "ctox_crew_members",
    "ctox_queue_tasks",
    "business_commands",
    ...(mayReadPrivate(ctx)
      ? [
          "ctox_harness_status",
          "ctox_harness_events",
          "ctox_runs",
          "ctox_crew_learnings",
        ]
      : []),
  ];
  for (const name of collections) {
    const sub = ctx.db?.collection?.(name)?.$?.subscribe?.(schedule);
    if (sub) s.cleanups.push(() => sub.unsubscribe?.());
    const unsubscribe = ctx.sync?.subscribeCollectionReadiness?.(
      name,
      schedule,
    );
    if (typeof unsubscribe === "function") s.cleanups.push(unsubscribe);
  }
  function closeDialog() {
    root.querySelector("[data-dialog]").hidden = true;
    patchMarkup(root.querySelector("[data-dialog]"), "");
    s.dialog = null;
    s.returnFocus?.focus();
  }
  function openControl(command, learningId) {
    const task = s.tasks.find((row) => row.id === s.selectedTaskId);
    const crew = command.startsWith("ctox.crew.");
    const workspace = ["ctox.queue.pause", "ctox.queue.capacity"].includes(
      command,
    );
    const reason =
      !workspace && !crew
        ? taskActionReason(ctx, command, task)
        : !mayControl(ctx, command, workspace ? null : task)
          ? "permissionDenied"
          : "";
    if (reason) {
      s.notice = t(reason);
      render();
      return;
    }
    if (
      command === "ctox.crew.assign" &&
      taskActionReason(ctx, command, task)
    ) {
      s.notice = t(taskActionReason(ctx, command, task));
      render();
      return;
    }
    s.dialog = {
      command,
      taskId: task?.id,
      learningId,
      memberId: s.profileId,
      paused: !s.status?.paused,
      archived: !s.members.find((m) => m.id === s.profileId)?.archived,
    };
    s.returnFocus = root.ownerDocument.activeElement;
    let fields = "";
    let key = "confirmControl";
    const field = (name, label, value = "", extra = "") =>
      `<label>${e(t(label))}<input name="${name}" value="${e(value)}" ${extra}></label>`;
    if (command === "ctox.queue.block") {
      key = "block";
      fields = field("reason", "reason", "", 'required maxlength="1000"');
    }
    if (command === "ctox.command.cancel") {
      key = "cancel";
      fields = field("reason", "reason", "", 'maxlength="1000"');
    }
    if (command === "ctox.queue.release") {
      key = "release";
      fields = field("note", "note", "", 'maxlength="1000"');
    }
    if (command === "ctox.queue.retry") key = "retry";
    if (command === "ctox.task.update") {
      key = "priority";
      fields = `<label>${e(t("priority"))}<select name="priority">${["low", "normal", "high", "urgent"].map((v) => `<option value="${v}" ${task?.priority === v ? "selected" : ""}>${e(t("priority_" + v))}</option>`).join("")}</select></label>`;
    }
    if (command === "ctox.crew.assign") {
      key = "assign";
      fields = `<label>${e(t("member"))}<select name="member_id" required>${s.members
        .filter((m) => !m.archived)
        .map((m) => `<option value="${e(m.id)}">${e(m.name)}</option>`)
        .join("")}</select></label>`;
    }
    if (command === "ctox.queue.capacity") {
      key = "capacity";
      fields = field(
        "workers",
        "capacity",
        s.status?.worker_capacity || 1,
        'type="number" min="1" max="8" required',
      );
    }
    if (command === "ctox.queue.pause") {
      key = s.status?.paused ? "resumeCrew" : "pauseCrew";
      fields = field("reason", "reason", "", 'maxlength="1000"');
    }
    if (command === "ctox.crew.member.update")
      key = s.members.find((m) => m.id === s.profileId)?.archived
        ? "restoreMember"
        : "archiveMember";
    if (command === "ctox.crew.learning.update") {
      key = "learningAction_update";
      fields = `<label>${e(t("learningText"))}<textarea name="text" required maxlength="400">${e(s.learnings.find((l) => l.id === learningId)?.text || "")}</textarea></label>`;
    }
    if (command === "ctox.crew.learning.confirm")
      key = "learningAction_confirm";
    if (command === "ctox.crew.learning.delete") key = "learningAction_delete";
    const layer = root.querySelector("[data-dialog]");
    layer.hidden = false;
    patchMarkup(
      layer,
      `<section class="crew-control-dialog" role="dialog" aria-modal="true" aria-labelledby="crew-control-title-${s.instanceId}"><form data-form="control"><h2 id="crew-control-title-${s.instanceId}">${e(t(key))}</h2>${fields}<p>${e(t("controlHint"))}</p><div class="crew-actions"><button type="submit">${e(t("confirmControl"))}</button><button type="button" data-action="dialog-close">${e(t("dismiss"))}</button></div></form></section>`,
    );
    layer.querySelector("input,textarea,select,button")?.focus();
  }
  async function dispatch(command, payload) {
    if (s.commandPending || s.disposed) return false;
    if (!COMMANDS.some((item) => item.command_type === command))
      throw new Error("unknown_crew_home_command");
    const id = `cmd_crew_home_${crypto.randomUUID()}`;
    s.commandPending = true;
    s.notice = t("commandSending");
    render();
    const sub = ctx.commandBus?.subscribe?.(id, (doc) => {
      if (s.disposed) return;
      if (doc?.status === "failed")
        s.notice = t("commandFailed", {
          reason: doc.error || doc.result?.error || t("unknown"),
        });
      schedule();
    });
    const unsubscribe = () => {
      sub?.unsubscribe?.();
      if (typeof sub === "function") sub();
      s.commandSubscriptions.delete(unsubscribe);
    };
    s.commandSubscriptions.add(unsubscribe);
    try {
      await ctx.commandBus.dispatch(
        {
          id,
          module: "ctox",
          command_type: command,
          record_id:
            payload.task_id ||
            payload.member_id ||
            payload.learning_id ||
            "harness",
          payload,
        },
        { until: "terminal" },
      );
      if (!s.disposed) {
        s.notice = t("commandApplied");
        closeDialog();
        await refresh();
      }
      return true;
    } catch (error) {
      if (!s.disposed)
        s.notice = t("commandFailed", { reason: error.message || error });
      return false;
    } finally {
      unsubscribe();
      s.commandPending = false;
      render();
    }
  }
  async function more(kind) {
    const key = { "member-runs": "memberRuns" }[kind] || kind;
    if (!(key in s.pages) || s.loadingData) return;
    s.pages[key] += 1;
    await refresh();
  }
  const onClick = (event) => {
    const button = event.target.closest("button");
    if (!button || !root.contains(button)) return;
    if (button.dataset.view) {
      s.view = button.dataset.view;
      render();
      void refresh();
    }
    if (button.dataset.task) {
      s.selectedTaskId = button.dataset.task;
      s.pages.runs = 1;
      s.view = "workplace";
      s.events = [];
      s.runs = [];
      render();
      void refresh();
    }
    if (button.dataset.member) {
      s.profileId = button.dataset.member;
      s.pages.learnings = 1;
      s.pages.memberRuns = 1;
      s.returnFocus = button;
      render();
      root.querySelector("[data-profile] button")?.focus();
      void refreshDetails().catch((error) => {
        s.notice = t("loadError", { reason: error.message });
        render();
      });
    }
    if (button.dataset.control)
      openControl(button.dataset.control, button.dataset.learning);
    const action = button.dataset.action;
    if (action === "profile-close") {
      s.profileId = "";
      s.learnings = [];
      s.memberRuns = [];
      render();
      s.returnFocus?.focus();
    }
    if (action === "dialog-close") closeDialog();
    if (action === "reload") void refresh();
    if (action?.endsWith("-more")) void more(action.slice(0, -5));
  };
  const onSubmit = (event) => {
    const form = event.target.closest("form");
    if (!form || !root.contains(form)) return;
    event.preventDefault();
    const data = Object.fromEntries(new FormData(form));
    if (form.dataset.form === "filter") {
      s.search = data.search.trim().slice(0, 60);
      s.source = data.source.trim();
      s.pages.active = 1;
      s.pages.terminal = 1;
      form.removeAttribute("data-dirty");
      void refresh();
      return;
    }
    if (form.dataset.form === "soul") {
      if (!mayControl(ctx, "ctox.crew.member.update")) return;
      const member = s.members.find((m) => m.id === s.profileId);
      const soul = { ...member.soul, sketch: data.sketch, voice: data.voice };
      for (const axis of AXES) soul[axis] = Number(data[axis]);
      void dispatch("ctox.crew.member.update", {
        member_id: member.id,
        name: data.name,
        soul,
      }).then((applied) => {
        if (applied) {
          form.removeAttribute("data-dirty");
          root.querySelector("[data-profile]").__crewMarkup = null;
          void refresh();
        }
      });
      return;
    }
    if (form.dataset.form === "control" && s.dialog) {
      const { command, taskId, learningId, memberId, paused, archived } =
        s.dialog;
      const task = s.tasks.find((t) => t.id === taskId);
      const taskCommand =
        command === "ctox.crew.assign" ||
        (!command.startsWith("ctox.crew.") &&
          !["ctox.queue.pause", "ctox.queue.capacity"].includes(command));
      const reason = taskCommand
        ? taskActionReason(ctx, command, task)
        : mayControl(ctx, command)
          ? ""
          : "permissionDenied";
      if (reason) {
        s.notice = t(reason);
        closeDialog();
        render();
        return;
      }
      let payload = { task_id: taskId };
      if (command === "ctox.command.cancel")
        payload = {
          target_command_id: task?.commandId,
          reason: data.reason || t("ownerCancelled"),
        };
      if (command === "ctox.queue.block") payload.reason = data.reason;
      if (command === "ctox.queue.release" && data.note)
        payload.note = data.note;
      if (command === "ctox.task.update") payload.priority = data.priority;
      if (command === "ctox.crew.assign") payload.member_id = data.member_id;
      if (command === "ctox.queue.capacity")
        payload = { workers: Number(data.workers) };
      if (command === "ctox.queue.pause")
        payload = { paused, ...(data.reason ? { reason: data.reason } : {}) };
      if (command === "ctox.crew.member.update")
        payload = { member_id: memberId, archived };
      if (command.startsWith("ctox.crew.learning."))
        payload = {
          learning_id: learningId,
          ...(data.text ? { text: data.text } : {}),
        };
      void dispatch(command, payload);
    }
  };
  const onInput = (event) =>
    event.target.closest("form")?.setAttribute("data-dirty", "true");
  const onKey = (event) => {
    if (event.key === "Escape") {
      if (s.dialog) closeDialog();
      else if (s.profileId) {
        s.profileId = "";
        render();
        s.returnFocus?.focus();
      }
      return;
    }
    if (event.key === "Tab" && s.dialog) {
      const nodes = [
        ...root
          .querySelector("[data-dialog]")
          .querySelectorAll("button,input,select,textarea"),
      ].filter((n) => !n.disabled);
      const first = nodes[0],
        last = nodes.at(-1);
      if (event.shiftKey && event.target === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && event.target === last) {
        event.preventDefault();
        first.focus();
      }
    }
  };
  root.addEventListener("click", onClick);
  root.addEventListener("submit", onSubmit);
  root.addEventListener("input", onInput);
  root.addEventListener("keydown", onKey);
  render();
  void refresh();
  return () => {
    s.disposed = true;
    clearTimeout(s.refreshTimer);
    clearTimeout(s.expiryTimer);
    s.cleanups.forEach((fn) => fn());
    s.commandSubscriptions.forEach((fn) => fn());
    root.removeEventListener("click", onClick);
    root.removeEventListener("submit", onSubmit);
    root.removeEventListener("input", onInput);
    root.removeEventListener("keydown", onKey);
  };
}
export const __ctoxTestHooks = {
  mergeBundleWithCommands,
  commandTaskFromProjection,
  normalizeExecutionProgress,
};
