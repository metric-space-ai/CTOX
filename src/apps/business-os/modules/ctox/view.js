import { crewCreatureHtml, creatureState } from "../../shared/crew-creature.js?v=20260906-crew-home-v1";
import {
  COMMANDS,
  taskGroup,
  taskActionReason,
  mayControl,
  mayReadPrivate,
} from "./model.js?v=20260906-crew-home-v1";
export const escapeHtml = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const e = escapeHtml;
export const AXES = [
  "gruendlichkeit_vs_tempo",
  "vorsicht_vs_mut",
  "knapp_vs_ausfuehrlich",
  "regeltreu_vs_kreativ",
  "nachfragen_vs_annehmen",
];
export function translate(messages, key, values = {}) {
  return String(messages[key] ?? messages.unknown).replace(
    /\{(\w+)\}/g,
    (_, name) => String(values[name] ?? ""),
  );
}
const tr = (s, key, values) => translate(s.messages, key, values);
export const time = (s, value) => {
  const n = typeof value === "number" ? value : Date.parse(value);
  return Number.isFinite(n)
    ? new Intl.DateTimeFormat(s.lang, {
        dateStyle: "short",
        timeStyle: "short",
      }).format(n)
    : tr(s, "unknown");
};
const number = (s, value, maximumFractionDigits = 2) =>
  typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(s.lang, { maximumFractionDigits }).format(value)
    : tr(s, "unknown");
const button = (s, key, attrs = "", label) =>
  `<button type="button" ${attrs}>${e(label ?? tr(s, key))}</button>`;
const title = (s, task) => task.title || tr(s, "untitledTask");
const eventTitle = (s, event) => {
  const selection =
    event.kind === "crew_selected" &&
    /^(assigned|continuity|selected):\s*(.*)$/s.exec(event.title || "");
  return selection
    ? tr(s, "selection_" + selection[1], { reason: selection[2] })
    : event.title;
};
const holdText = (s, reason) =>
  ["missing_review_evidence", "missing_artifact"].includes(reason)
    ? tr(s, "failure_" + reason)
    : reason?.startsWith("technical:")
      ? tr(s, "technicalHold", { reason: reason.slice(10) })
      : reason;
const creature = (s, member, task, size = "home", decorative = true) =>
  crewCreatureHtml(member, {
    size,
    decorative,
    state: creatureState({
      member,
      task,
      events: s.events,
      connected: s.connected,
    }),
    label: `${member?.name}: ${tr(s, "creature_" + creatureState({ member, task, events: s.events, connected: s.connected }))}`,
  });

export function taskSentence(s, task) {
  if (!task) return tr(s, "noTask");
  if (["cancelled", "canceled"].includes(task.status))
    return tr(s, "taskCancelled");
  if (task.status === "failed")
    return tr(s, "taskFailed", {
      reason: task.error || tr(s, "failure_" + task.failure_class),
    });
  if (taskGroup(task) === "done") return tr(s, "group_done");
  if (task.retry_not_before && Date.parse(task.retry_not_before) > Date.now())
    return tr(s, "taskRetryAt", {
      count: task.failure_attempt_count || 0,
      time: time(s, task.retry_not_before),
    });
  if (task.hold_reason)
    return tr(s, "taskHeld", { reason: holdText(s, task.hold_reason) });
  if (task.wait_entity_id)
    return tr(s, "taskWaitsFor", {
      entity: tr(s, "wait_" + task.wait_entity_type),
      id: task.wait_entity_id,
    });
  if (taskGroup(task) === "running" && task.lease_expires_at)
    return tr(s, "taskWorkingUntil", { time: time(s, task.lease_expires_at) });
  return tr(s, "group_" + taskGroup(task));
}
export function statusMarkup(s) {
  if (s.error)
    return `<span class="crew-error">${e(tr(s, "loadError", { reason: s.error }))}</span>${button(s, "retryLoad", 'data-action="reload"')}`;
  if (!s.connected) return `<span>${e(tr(s, "disconnected"))}</span>`;
  if (s.loading || s.syncing) return `<span>${e(tr(s, "loading"))}</span>`;
  if (!mayReadPrivate(s.ctx)) return `<span>${e(tr(s, "publicView"))}</span>`;
  const h = s.status;
  if (!h) return `<span>${e(tr(s, "awaitingStatus"))}</span>`;
  const member = s.members.find((m) => m.id === h.active_crew_member_id);
  return `<span class="crew-status-dot ${h.paused ? "is-paused" : ""}" aria-hidden="true"></span><strong>${e(tr(s, h.paused ? "paused" : h.service_running ? "running" : "stopped"))}</strong>
    ${h.paused && h.pause_reason ? `<span>${e(h.pause_reason)}</span>` : ""}
    ${member ? `<span>${e(tr(s, "memberWorking", { name: member.name }))}</span>` : ""}
    <span>${e(tr(s, "capacityCount", { count: h.worker_capacity ?? tr(s, "unknown") }))}</span>
    <span>${e(tr(s, "queueCounts", { pending: h.pending_count ?? 0, leased: h.leased_count ?? 0, blocked: h.blocked_count ?? 0 }))}</span>
    ${h.pressure_active ? `<span class="crew-error">${e(tr(s, "pressure", { count: h.pressure_threshold }))}</span>` : ""}
    ${h.work_hours?.enabled ? `<span>${e(tr(s, h.work_hours.inside_window ? "workHours" : "outsideHours", { start: h.work_hours.start, end: h.work_hours.end }))}</span>` : ""}
    ${h.last_error ? `<span class="crew-error">${e(tr(s, "statusError", { reason: h.last_error }))}</span>` : ""}`;
}
export function homeMarkup(s) {
  const active =
    s.members.find((m) => m.id === s.status?.active_crew_member_id) ||
    s.members.find((m) => m.state === "on_duty");
  const task = s.tasks.find(
    (t) => t.id === (active?.active_task_id || s.status?.active_task_ids?.[0]),
  );
  const atWork = Boolean(active || task);
  const atHome = s.members.filter((m) => m.id !== active?.id);
  const memberButton = (member, size = "home", work = null) =>
    `<button type="button" class="crew-member" data-key="member-${e(member.id)}" data-member="${e(member.id)}">${creature(s, member, work, size)}<strong>${e(member.name)}</strong><span>${e(tr(s, member.archived ? "archived" : "creature_" + creatureState({ member, task: work, connected: s.connected, events: s.events })))}</span></button>`;
  return `<header class="crew-view-heading ${atWork ? "crew-home-active-heading" : ""}"><h1>${e(tr(s, atWork ? "homeWorking" : "homeTitle"))}</h1>${atWork ? "" : `<p>${e(tr(s, "homeHint"))}</p>`}</header>
    ${atWork ? `<section class="crew-current" data-key="current"><div>${active ? memberButton(active, "workplace", task) : `<p class="crew-identity-missing">${e(tr(s, task?.command_type === "ctox.coding.turn" ? "codingNoMember" : "unassignedTask"))}</p>`}</div><div><p class="crew-eyebrow">${e(tr(s, "currentTask"))}</p><h2>${e(task ? title(s, task) : tr(s, "awaitingTask"))}</h2><p>${e(task ? taskSentence(s, task) : tr(s, "taskLoading"))}</p>${task ? button(s, "openWorkplace", `data-task="${e(task.id)}" class="crew-primary"`) : ""}</div></section>` : ""}
    <section class="crew-home-room" aria-label="${e(tr(s, "home"))}">${(active ? atHome : s.members).map((m) => memberButton(m)).join("")}</section>
    ${!s.members.length ? `<p class="crew-empty">${e(tr(s, s.loading || s.syncing ? "loading" : s.connected ? "noMembers" : "disconnected"))}</p>` : ""}
    ${s.moreMembers ? button(s, "loadMore", 'data-action="members-more"') : ""}`;
}
const controls = COMMANDS.filter((command) => command.scope === "task").map(
  (command) => [command.label, command.command_type],
);
export function actionMarkup(s, task) {
  const denied = [];
  const buttons = controls
    .map(([key, command]) => {
      const reason = taskActionReason(s.ctx, command, task);
      if (reason) {
        denied.push(`${tr(s, key)}: ${tr(s, reason)}`);
        return "";
      }
      return button(
        s,
        key,
        `data-control="${command}" data-key="action-${key}" ${s.commandPending ? "disabled" : ""}`,
      );
    })
    .join("");
  return `<div class="crew-actions">${buttons}${denied.length ? `<span tabindex="0" class="crew-action-hint" title="${e(denied.join("\n"))}" aria-label="${e(denied.join(". "))}">${e(tr(s, "restrictedActions"))}</span>` : ""}</div>`;
}
function planMarkup(s, task) {
  const progress = task.executionProgress;
  if (!progress?.steps?.length)
    return `<p class="crew-empty">${e(tr(s, "planPending"))}</p>`;
  return `<ol class="crew-plan">${progress.steps.map((step) => `<li data-key="step-${step.position}" class="is-${["completed", "in_progress", "failed"].includes(step.status) ? step.status : "pending"}"><span class="crew-step-number">${e(step.position)}</span><div><strong>${e(step.label)}</strong><span>${e(tr(s, "step_" + step.status))}</span></div></li>`).join("")}</ol>`;
}
export function runsMarkup(s, runs) {
  return (
    runs
      .map(
        (
          run,
        ) => `<article class="crew-run" data-key="run-${e(run.id)}"><header><strong>${e(tr(s, "runStatus_" + run.status))}</strong><time>${e(time(s, run.finished_at_ms))}</time></header>
    <div class="crew-run-metrics"><span>${e(run.metrics?.model || tr(s, "unknownModel"))}</span><span>${e(tr(s, "tokens", { input: number(s, run.metrics?.input_tokens), output: number(s, run.metrics?.output_tokens), reasoning: number(s, run.metrics?.reasoning_tokens) }))}</span><span>${e(tr(s, "cost", { value: number(s, run.metrics?.cost_usd, 6) }))}</span><span>${e(tr(s, "duration", { value: typeof run.metrics?.elapsed_ms === "number" ? number(s, run.metrics.elapsed_ms / 1000) : tr(s, "unknown") }))}</span></div>
    ${run.review?.disposition ? `<p>${e(tr(s, "reviewVerdict", { value: tr(s, "review_" + run.review.disposition) }))}</p>` : ""}
    ${run.review?.hold_reason ? `<p>${e(run.review.hold_reason)}</p>` : ""}${run.error_text ? `<p class="crew-error">${e(run.error_text)}</p>` : ""}
    ${run.retrospective ? `<blockquote>${e(run.retrospective)}</blockquote>` : ""}
    ${!run.crew_member_id ? `<small>${e(tr(s, "unassignedRun"))}</small>` : ""}</article>`,
      )
      .join("") || `<p class="crew-empty">${e(tr(s, "noRuns"))}</p>`
  );
}
export function workplaceMarkup(s) {
  const task = s.tasks.find((t) => t.id === s.selectedTaskId);
  if (!task)
    return `<header class="crew-view-heading"><h1>${e(tr(s, "workplace"))}</h1><p>${e(tr(s, "chooseTask"))}</p>${button(s, "queue", 'data-view="queue"')}</header>`;
  const member = s.members.find((m) => m.id === task.crew_member_id);
  return `<header class="crew-task-heading" data-key="task-header">${member ? `<button type="button" class="crew-member" data-member="${e(member.id)}">${creature(s, member, task, "home")}<strong>${e(member.name)}</strong></button>` : `<span class="crew-identity-missing">${e(tr(s, task.command_type === "ctox.coding.turn" ? "codingNoMember" : "unassignedTask"))}</span>`}<div><p class="crew-eyebrow">${e(task.source || tr(s, "unknownSource"))}</p><h1>${e(title(s, task))}</h1><p>${e(taskSentence(s, task))}</p></div></header>
    ${actionMarkup(s, task)}<p class="crew-prompt">${e(task.prompt || "")}</p>
    <section class="crew-section"><h2>${e(tr(s, "plan"))}</h2>${planMarkup(s, task)}</section>
    ${
      mayReadPrivate(s.ctx)
        ? `<section class="crew-section"><h2>${e(tr(s, "activity"))}</h2><ol class="crew-events">${s.events.map((event) => `<li data-key="event-${e(event.id)}"><time>${e(time(s, event.created_at_ms))}</time><div><strong>${e(tr(s, "event_" + event.kind, { tool: event.tool_name || tr(s, "tool") }))}</strong>${event.title ? `<p>${e(eventTitle(s, event))}</p>` : ""}</div></li>`).join("")}</ol>${!s.events.length ? `<p>${e(tr(s, "noEvents"))}</p>` : ""}</section>
    <section class="crew-section"><h2>${e(tr(s, "runs"))}</h2>${runsMarkup(s, s.runs)}${s.moreRuns ? button(s, "loadMore", 'data-action="runs-more"') : ""}</section>`
        : `<p>${e(tr(s, "privateDetails"))}</p>`
    }`;
}
export function queueMarkup(s) {
  const groups = ["running", "waiting", "review", "blocked", "done", "failed"];
  return `<header class="crew-view-heading"><h1>${e(tr(s, "queue"))}</h1><p>${e(tr(s, "queueHint"))}</p></header>
    <form class="crew-filters" data-form="filter" data-key="filters"><label>${e(tr(s, "search"))}<input name="search" type="search" maxlength="60" value="${e(s.search)}"></label><label>${e(tr(s, "source"))}<input name="source" value="${e(s.source)}" list="crew-sources-${s.instanceId}"></label><datalist id="crew-sources-${s.instanceId}">${[...new Set(s.tasks.map((t) => t.source).filter(Boolean))].map((source) => `<option value="${e(source)}"></option>`).join("")}</datalist><button type="submit">${e(tr(s, "applyFilter"))}</button></form>
    ${mayControl(s.ctx, "ctox.queue.pause") ? `<div class="crew-owner-controls">${button(s, s.status?.paused ? "resumeCrew" : "pauseCrew", 'data-control="ctox.queue.pause"')}${button(s, "capacity", 'data-control="ctox.queue.capacity"')}<span>${e(tr(s, "serialHint"))}</span></div>` : ""}
    <div class="crew-queue-groups">${groups
      .map((group) => {
        const tasks = s.tasks.filter(
          (t) =>
            (!s.queueTaskIds || s.queueTaskIds.has(t.id)) &&
            taskGroup(t) === group,
        );
        return `<section class="crew-queue-group" data-key="group-${group}"><h2>${e(tr(s, "group_" + group))}<span>${tasks.length}</span></h2><div>${tasks.map((task) => `<button type="button" class="crew-task-row" data-task="${e(task.id)}" data-key="task-${e(task.id)}"><span><strong>${e(title(s, task))}</strong><small>${e(task.source || tr(s, "unknownSource"))}</small></span><span>${e(taskSentence(s, task))}</span><time>${e(time(s, task.updated_at_ms))}</time></button>`).join("")}</div></section>`;
      })
      .join("")}</div>
    <footer class="crew-paging">${s.moreActive ? button(s, "moreActive", 'data-action="active-more"') : ""}${s.moreTerminal ? button(s, "moreTerminal", 'data-action="terminal-more"') : ""}<small>${e(tr(s, "retentionHint"))}</small></footer>`;
}
export function profileMarkup(s) {
  const member = s.members.find((m) => m.id === s.profileId);
  if (!member) return "";
  const task = s.tasks.find((t) => t.id === member.active_task_id);
  const soulReady =
    member.soul && AXES.every((axis) => Number.isFinite(member.soul[axis]));
  const edit = mayControl(s.ctx, "ctox.crew.member.update");
  return `<header class="crew-profile-heading">${button(s, "close", 'data-action="profile-close" class="crew-close"')}${creature(s, member, task, "home", false)}<h2 id="crew-profile-title-${s.instanceId}">${e(member.name)}</h2><p>${e(tr(s, "shape_" + member.shape))} · <span class="crew-color-swatch" style="background:${/^#[0-9a-f]{6}$/i.test(member.color) ? member.color : "transparent"}"></span>${e(member.color)}</p></header>
    ${
      mayReadPrivate(s.ctx)
        ? `${soulReady ? `<form data-form="soul" data-key="soul-${e(member.id)}"><h3>${e(tr(s, "soul"))}</h3><label>${e(tr(s, "name"))}<input name="name" maxlength="60" value="${e(member.name)}" ${edit ? "" : "readonly"}></label>${AXES.map((axis) => `<label class="crew-axis">${e(tr(s, "axis_" + axis))}<input type="range" name="${axis}" min="0" max="100" value="${e(member.soul[axis])}" ${edit ? "" : "disabled"}></label>`).join("")}<label>${e(tr(s, "sketch"))}<textarea name="sketch" maxlength="600" ${edit ? "" : "readonly"}>${e(member.soul.sketch || "")}</textarea></label><label>${e(tr(s, "voice"))}<input name="voice" maxlength="200" value="${e(member.soul.voice || "")}" ${edit ? "" : "readonly"}></label>${edit ? `<button type="submit">${e(tr(s, "saveSoul"))}</button>` : ""}</form>` : `<p>${e(tr(s, "profileLoading"))}</p>`}
    <section class="crew-section"><h3>${e(tr(s, "specialties"))}</h3>${["modules", "command_types", "skills", "tags"].map((key) => (member.specialties?.[key]?.length ? `<p><strong>${e(tr(s, "specialty_" + key))}</strong> ${e(member.specialties[key].join(", "))}</p>` : "")).join("") || `<p>${e(tr(s, "noSpecialties"))}</p>`}</section>
    <section class="crew-section"><h3>${e(tr(s, "life"))}</h3><dl class="crew-stats">${["tasks_total", "succeeded", "failed", "review_passed", "review_rejected"].map((key) => `<div><dt>${e(tr(s, "stat_" + key))}</dt><dd>${e(number(s, member.stats?.[key]))}</dd></div>`).join("")}</dl><p>${e(tr(s, "stat_avg_elapsed_ms"))}: ${e(typeof member.stats?.avg_elapsed_ms === "number" ? tr(s, "duration", { value: number(s, member.stats.avg_elapsed_ms / 1000) }) : tr(s, "unknown"))}</p><p>${e(tr(s, "stat_last_active_at"))}: ${e(time(s, member.stats?.last_active_at))}</p></section>
    <section class="crew-section"><h3>${e(tr(s, "learnings"))}</h3>${
      s.learnings
        .map(
          (learning) =>
            `<article class="crew-learning" data-key="learning-${e(learning.id)}"><small>${e(tr(s, "learning_" + learning.kind))} · ${e(tr(s, learning.confirmed_by_owner ? "confirmed" : "unconfirmed"))}</small><p>${e(learning.text)}</p><div class="crew-actions">${[
              "confirm",
              "update",
              "delete",
            ]
              .filter(
                (action) =>
                  mayControl(s.ctx, "ctox.crew.learning." + action) &&
                  !(action === "confirm" && learning.confirmed_by_owner),
              )
              .map((action) =>
                button(
                  s,
                  "learningAction_" + action,
                  `data-learning="${e(learning.id)}" data-control="ctox.crew.learning.${action}"`,
                ),
              )
              .join("")}</div></article>`,
        )
        .join("") || `<p>${e(tr(s, "noLearnings"))}</p>`
    }${s.moreLearnings ? button(s, "loadMore", 'data-action="learnings-more"') : ""}</section>
    <section class="crew-section"><h3>${e(tr(s, "timesheet"))}</h3>${runsMarkup(s, s.memberRuns)}${s.moreMemberRuns ? button(s, "loadMore", 'data-action="member-runs-more"') : ""}</section>
    ${edit ? button(s, member.archived ? "restoreMember" : "archiveMember", 'data-control="ctox.crew.member.update"') : ""}`
        : `<p>${e(tr(s, "privateProfile"))}</p>`
    }`;
}

/** Keyed DOM reconciliation preserves focus, scroll and dirty form fields. */
export function patchMarkup(root, markup) {
  if (root.__crewMarkup === markup) return;
  const template = root.ownerDocument.createElement("template");
  template.innerHTML = markup;
  patchChildren(root, template.content);
  root.__crewMarkup = markup;
}
const key = (node) =>
  node.nodeType === 1 ? node.getAttribute("data-key") || node.id || "" : "";
function patchChildren(parent, source) {
  const old = [...parent.childNodes];
  const keyed = new Map(old.filter(key).map((node) => [key(node), node]));
  let cursor = parent.firstChild;
  for (const desired of [...source.childNodes]) {
    let node = key(desired) ? keyed.get(key(desired)) : cursor;
    if (
      !node ||
      node.nodeType !== desired.nodeType ||
      node.nodeName !== desired.nodeName ||
      (key(node) && key(node) !== key(desired))
    ) {
      node = desired.cloneNode(true);
      parent.insertBefore(node, cursor);
    } else {
      if (node !== cursor) parent.insertBefore(node, cursor);
      if (node.nodeType === 3) {
        if (node.textContent !== desired.textContent)
          node.textContent = desired.textContent;
      } else if (node.nodeType === 1) {
        const editing =
          node.matches("input,textarea,select") &&
          (node.closest('[data-dirty="true"]') ||
            node === node.ownerDocument.activeElement);
        for (const attr of [...node.attributes])
          if (!desired.hasAttribute(attr.name) && attr.name !== "data-dirty")
            node.removeAttribute(attr.name);
        for (const attr of [...desired.attributes])
          if (
            !(editing && attr.name === "value") &&
            node.getAttribute(attr.name) !== attr.value
          )
            node.setAttribute(attr.name, attr.value);
        if (!editing) {
          patchChildren(node, desired);
          if (
            node.matches("input,textarea,select") &&
            node.value !== desired.value
          )
            node.value = desired.value;
        }
      }
    }
    cursor = node.nextSibling;
  }
  while (cursor) {
    const next = cursor.nextSibling;
    cursor.remove();
    cursor = next;
  }
}
