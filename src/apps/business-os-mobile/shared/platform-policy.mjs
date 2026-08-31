export function requireAndroidMultiProfile(featureSupported) {
  if (!featureSupported) throw new Error("Android WebView MULTI_PROFILE is required");
  return true;
}

export function officeRequestState(event, state = { status: "idle", progress: 0 }) {
  switch (event.type) {
    case "request": return { status: "awaiting-consent", progress: 0, totalBytes: event.totalBytes };
    case "download": return { ...state, status: "downloading" };
    case "progress": return { ...state, status: "downloading", progress: Math.max(0, Math.min(1, event.value)) };
    case "complete": return { ...state, status: "active", progress: 1 };
    case "cancel": return { ...state, status: "canceled", progress: 0, retryable: true };
    case "offline": return { ...state, status: "offline", progress: 0, retryable: true };
    case "error": return { ...state, status: "error", progress: 0, retryable: true, message: event.message };
    default: throw new Error("unsupported office pack event");
  }
}
