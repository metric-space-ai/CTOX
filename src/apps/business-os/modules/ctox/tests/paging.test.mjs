import assert from "node:assert/strict";
import { test } from "node:test";
import { ctoxRxdbTestInternals } from "../../../rxdb/dist/ctox-rxdb-js.mjs";
import { readPages } from "../paging.js";
import {
  queries,
  mergeBundleWithCommands,
  taskActionReason,
  taskGroup,
} from "../model.js";
import { creatureState } from "../../../shared/crew-creature.js";
const { matchesSelector, sortDocuments, normalizeSort } = ctoxRxdbTestInternals;

test("loaded keyset pages survive refresh and tied timestamps without duplicates", async () => {
  let rows = Array.from({ length: 85 }, (_, i) => ({
    id: String(i).padStart(3, "0"),
    status: "pending",
    module: "ctox",
    updated_at_ms: 100,
  }));
  const requests = [];
  const ctx = {
    db: {
      collection() {
        return {
          find(query) {
            requests.push(query);
            return {
              exec: async () =>
                sortDocuments(
                  rows.filter((row) => matchesSelector(row, query.selector)),
                  normalizeSort(query.sort),
                ).slice(0, query.limit),
            };
          },
        };
      },
    },
  };
  const before = await readPages(
    ctx,
    "ctox_queue_tasks",
    (cursor) => queries.tasks({ cursor }),
    2,
  );
  assert.equal(before.rows.length, 80);
  assert.equal(new Set(before.rows.map((r) => r.id)).size, 80);
  assert.equal(before.more, true);
  rows.push({
    id: "new",
    status: "pending",
    module: "ctox",
    updated_at_ms: 200,
  });
  const after = await readPages(
    ctx,
    "ctox_queue_tasks",
    (cursor) => queries.tasks({ cursor }),
    2,
  );
  assert.equal(after.rows[0].id, "new");
  assert.equal(after.rows.length, 80);
  assert.equal(new Set(after.rows.map((r) => r.id)).size, 80);
  const all = await readPages(
    ctx,
    "ctox_queue_tasks",
    (cursor) => queries.tasks({ cursor }),
    3,
  );
  assert.equal(all.rows.length, 86);
  assert.equal(all.more, false);
  assert.ok(
    requests.every(
      (query) => query.limit === 40 && Object.keys(query.selector).length,
    ),
  );
});

test("task and command title filters use the actual runtime literal regex subset", () => {
  const filters = { source: "ctox", search: "a.b [x]" };
  const task = {
    id: "t",
    status: "pending",
    module: "ctox",
    title: "Read a.b [x] now",
  };
  const command = {
    id: "c",
    module: "ctox",
    execution_mode: "queue",
    execution_phase: "running",
    payload: { title: task.title },
  };
  assert.equal(matchesSelector(task, queries.tasks(filters).selector), true);
  assert.equal(
    matchesSelector(command, queries.activeCommands(filters).selector),
    true,
  );
  assert.equal(
    matchesSelector(
      { ...task, title: "Read axb x now" },
      queries.tasks(filters).selector,
    ),
    false,
  );
  assert.equal(
    matchesSelector(
      { ...command, module: "another" },
      queries.activeCommands(filters).selector,
    ),
    false,
  );
});

test("retry permits failed standalone tasks but denies terminal linked commands", () => {
  const ctx = { session: { user: { role: "admin" } } };
  assert.equal(
    taskActionReason(ctx, "ctox.queue.retry", { id: "t", status: "failed" }),
    "",
  );
  assert.equal(
    taskActionReason(ctx, "ctox.queue.retry", {
      id: "t",
      status: "failed",
      executionPhase: "terminal",
    }),
    "alreadyTerminal",
  );
});

test("running command clears a stale blocked route and old attempts cannot animate this attempt", () => {
  const task = mergeBundleWithCommands(
    {},
    [
      {
        id: "c",
        execution_task_id: "t",
        execution_mode: "queue",
        execution_phase: "running",
      },
    ],
    [{ id: "t", status: "blocked", route_status: "blocked" }],
  ).queue[0];
  assert.equal(taskGroup(task), "running");
  assert.equal(task.route_status, "running");
  const now = Date.now(),
    member = { id: "m", state: "on_duty" };
  const active = { id: "t", status: "running", attempt: 2 };
  assert.equal(
    creatureState({
      member,
      task: active,
      now,
      events: [
        { task_id: "t", attempt: 1, kind: "tool_started", created_at_ms: now },
      ],
    }),
    "waking",
  );
  assert.equal(
    creatureState({
      member,
      task: active,
      now,
      events: [
        { task_id: "t", attempt: 2, kind: "tool_started", created_at_ms: now },
      ],
    }),
    "tooling",
  );
});
