/** Read failures and an unavailable data plane must never look like idle work. */
export function workspaceDataState({
  error = "",
  available = true,
  readiness,
  loaded = false,
  tasks = [],
} = {}) {
  if (error) return { kind: "error", reason: String(error).slice(0, 300) };
  if (!available || readiness?.state === "offline-pending")
    return { kind: "offline" };
  if (!loaded || readiness?.ready === false) return { kind: "loading" };
  return { kind: tasks.length ? "ready" : "idle" };
}
