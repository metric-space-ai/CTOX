// Keep the DOM and replication lease alive until an app explicitly accepts
// closing. Failed saves veto the request; repeated close clicks share it.
export function createWindowCloseGate(onError = () => {}) {
  const guards = new Set();
  let pending = null;
  return {
    add(guard) {
      if (typeof guard !== 'function') throw new TypeError('Close guard must be a function');
      guards.add(guard);
      return () => guards.delete(guard);
    },
    get size() { return guards.size; },
    request(close) {
      if (pending) return pending;
      pending = Promise.resolve().then(async () => {
        for (const guard of [...guards]) {
          if (await guard() === false) return false;
        }
        close();
        return true;
      }).catch(error => {
        try { onError(error); } catch { /* A reporting failure must not close the app. */ }
        return false;
      }).finally(() => { pending = null; });
      return pending;
    },
  };
}
