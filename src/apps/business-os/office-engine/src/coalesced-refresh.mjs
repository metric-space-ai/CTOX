// One in-flight refresh per app. Changes arriving during a read are retained
// for one following batch rather than spawning overlapping remote queries.
export function createCoalescedRefresh({ refresh, onError = () => {}, delayMs = 80,
  setTimer = setTimeout, clearTimer = clearTimeout }) {
  let timer = null;
  let running = false;
  let disposed = false;
  const pending = new Set();
  const isActive = () => !disposed;
  const arm = () => {
    if (disposed || running || timer !== null || !pending.size) return;
    timer = setTimer(async () => {
      timer = null;
      if (disposed) return;
      running = true;
      const changed = new Set(pending);
      pending.clear();
      try {
        await refresh(changed, isActive);
      } catch (error) {
        if (!disposed) {
          try { onError(error); } catch { /* Diagnostics must not wedge refreshes. */ }
        }
      } finally {
        running = false;
        arm();
      }
    }, delayMs);
  };
  return {
    notify(collection) {
      if (disposed) return;
      pending.add(collection);
      arm();
    },
    dispose() {
      disposed = true;
      pending.clear();
      if (timer !== null) clearTimer(timer);
      timer = null;
    },
  };
}
