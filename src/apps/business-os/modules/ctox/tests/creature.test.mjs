import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import {
  CREATURE_STATES,
  CREATURE_SIZES,
  CREATURE_SHAPES,
  crewCreatureHtml,
  creatureState,
} from "../../../shared/crew-creature.js";
const member = {
  id: "crew-milo",
  name: "Milo",
  shape: "round",
  color: "#1685ee",
  state: "home",
};
const now = 100_000;
const task = { id: "t", status: "running", route_status: "leased" };
const event = (kind, age = 0, task_id = "t") => ({
  id: kind,
  kind,
  created_at_ms: now - age,
  task_id,
});
test("all nine states have deterministic class and distinct face/signal snapshots", () => {
  const rendered = new Set();
  for (const state of CREATURE_STATES) {
    const html = crewCreatureHtml(member, { state, label: `Milo: ${state}` });
    assert.match(
      html,
      new RegExp(
        `class="crew-creature crew-creature--home is-${state} shape-round"`,
      ),
    );
    assert.match(html, new RegExp(`data-crew-state="${state}"`));
    assert.equal(
      html,
      crewCreatureHtml(member, { state, label: `Milo: ${state}` }),
    );
    rendered.add(html.split("crew-creature__face")[1]);
  }
  assert.equal(rendered.size, 9);
  for (const shape of CREATURE_SHAPES)
    for (const size of CREATURE_SIZES)
      assert.match(
        crewCreatureHtml({ ...member, shape }, { size }),
        new RegExp(`crew-creature--${size}.*shape-${shape}`),
      );
});
test("identity cannot be invented or inject attributes; labels are escaped", () => {
  assert.equal(crewCreatureHtml(null), "");
  assert.equal(crewCreatureHtml({ ...member, color: "red;display:none" }), "");
  assert.equal(crewCreatureHtml({ ...member, shape: "random" }), "");
  assert.match(
    crewCreatureHtml({ ...member, name: "<Milo>" }),
    /aria-label="&lt;Milo&gt;"/,
  );
});
test("durable terminal/wait/review evidence wins over recent tool activity", () => {
  const events = [event("tool_started")];
  for (const [status, state] of [
    ["failed", "failed"],
    ["completed", "done"],
    ["blocked", "waiting"],
    ["review", "reviewing"],
    ["pending", "queued"],
  ])
    assert.equal(
      creatureState({ task: { id: "t", status }, events, now }),
      state,
    );
  assert.equal(
    creatureState({
      task: { ...task, retry_not_before: new Date(now + 1000).toISOString() },
      events,
      now,
    }),
    "waiting",
  );
  assert.equal(
    creatureState({
      task: {
        ...task,
        execution_progress: { review: { status: "validating" } },
      },
      events,
      now,
    }),
    "reviewing",
  );
});
test("recent task-bound events map to activity and expire after ten seconds", () => {
  assert.equal(
    creatureState({ task, events: [event("tool_started")], now }),
    "tooling",
  );
  assert.equal(
    creatureState({ task, events: [event("thinking")], now }),
    "thinking",
  );
  assert.equal(
    creatureState({ task, events: [event("crew_selected")], now }),
    "waking",
  );
  assert.equal(
    creatureState({
      task,
      events: [event("tool_started", 10001), event("thinking", 0, "other")],
      now,
    }),
    "waking",
  );
  assert.equal(
    creatureState({ task, events: [event("tool_started", -1)], now }),
    "waking",
  );
  assert.equal(
    creatureState({
      task,
      events: [event("tool_started", 1000), event("tool_completed")],
      now,
    }),
    "thinking",
  );
});
test("idle, unavailable, failure resting and disconnected remain distinct", () => {
  assert.equal(creatureState({ member }), "sleeping");
  assert.equal(
    creatureState({ member: { ...member, state: "on_duty" } }),
    "waking",
  );
  assert.equal(
    creatureState({ member: { ...member, state: "resting_after_failure" } }),
    "failed",
  );
  assert.equal(creatureState({ member, task, connected: false }), "sleeping");
  const css = readFileSync(
    new URL("../../../shared/crew-creature.css", import.meta.url),
    "utf8",
  );
  assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(css, /animation:\s*none\s*!important/);
});
