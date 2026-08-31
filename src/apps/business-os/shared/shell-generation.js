export function isShellGenerationMismatchResponse(response) {
  return response?.status === 409
    && response.headers?.get?.('x-ctox-shell-generation-mismatch') === '1';
}

export function createShellGenerationReloadGuard(options = {}) {
  const readMarker = typeof options.readMarker === 'function' ? options.readMarker : () => null;
  const writeMarker = typeof options.writeMarker === 'function' ? options.writeMarker : () => {};
  const defer = typeof options.defer === 'function' ? options.defer : (callback) => setTimeout(callback, 0);
  const reload = typeof options.reload === 'function' ? options.reload : () => {};
  let scheduled = false;

  return Object.freeze({
    inspect(response) {
      if (scheduled || !isShellGenerationMismatchResponse(response)) return false;
      scheduled = true;
      const active = response.headers.get('x-ctox-shell-generation') || 'unknown';
      const markerKey = `ctox.businessOs.shellGenerationReload.${active}`;
      let alreadyReloaded = false;
      try {
        alreadyReloaded = readMarker(markerKey) === '1';
        if (!alreadyReloaded) writeMarker(markerKey, '1');
      } catch {
        // Restricted storage must not prevent a one-shot recovery reload.
      }
      if (!alreadyReloaded) defer(reload);
      return true;
    },
    get scheduled() {
      return scheduled;
    },
  });
}
