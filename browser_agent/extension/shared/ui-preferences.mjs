export const UI_PREFERENCES_KEY = "sinepanel.ui.preferences.v1";

export function createDefaultUiPreferences() {
  return {
    showDevHeader: false,
  };
}

export function normalizeUiPreferences(rawValue) {
  const value = rawValue && typeof rawValue === "object" ? rawValue : {};
  return {
    showDevHeader: value.showDevHeader === true,
  };
}

export async function readUiPreferences(storageApi) {
  if (!storageApi?.getValue) {
    return createDefaultUiPreferences();
  }
  return normalizeUiPreferences(await storageApi.getValue(UI_PREFERENCES_KEY, null));
}

export async function writeUiPreferences(storageApi, rawValue) {
  const nextValue = normalizeUiPreferences(rawValue);
  if (storageApi?.setValue) {
    await storageApi.setValue(UI_PREFERENCES_KEY, nextValue);
  }
  return nextValue;
}
