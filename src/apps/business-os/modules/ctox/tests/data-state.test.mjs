import assert from "node:assert/strict";
import { test } from "node:test";
import { workspaceDataState } from "../data-state.js";

test("missing data plane, cold sync and known idle are distinct", () => {
  assert.deepEqual(workspaceDataState({ available: false }), {
    kind: "offline",
  });
  assert.deepEqual(
    workspaceDataState({ readiness: { ready: false, state: "catching-up" } }),
    { kind: "loading" },
  );
  assert.deepEqual(
    workspaceDataState({ loaded: true, readiness: { ready: true } }),
    { kind: "idle" },
  );
});
test("a failed read never becomes idle and retains the reason", () => {
  assert.deepEqual(
    workspaceDataState({ loaded: true, error: "IndexedDB read failed" }),
    { kind: "error", reason: "IndexedDB read failed" },
  );
  assert.equal(
    workspaceDataState({ error: "x".repeat(500) }).reason.length,
    300,
  );
});
test("cached rows are not evidence of a live connection", () => {
  const tasks = [{ id: "cached-task" }];
  assert.equal(
    workspaceDataState({
      tasks,
      loaded: true,
      readiness: { ready: false, state: "offline-pending" },
    }).kind,
    "offline",
  );
  assert.equal(
    workspaceDataState({
      tasks,
      loaded: true,
      readiness: { ready: false, state: "catching-up" },
    }).kind,
    "loading",
  );
  assert.equal(
    workspaceDataState({
      tasks,
      loaded: true,
      readiness: { ready: true, state: "live" },
    }).kind,
    "ready",
  );
  assert.deepEqual(tasks, [{ id: "cached-task" }]);
});
