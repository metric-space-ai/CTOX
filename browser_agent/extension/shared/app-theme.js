(function registerSinepanelAppTheme(globalScope) {
  const STORAGE_KEY = "sinepanel.app.theme.v1";
  const THEME_CACHE_KEY = "sinepanel.app.theme.cache.v1";
  const STORAGE_TIMEOUT_MS = 1200;
  const DEFAULT_THEME_ID = "copyshop";
  const THEMES = Object.freeze([
    {
      id: "copyshop",
      label: "Copyshop",
      title: "Berlin Copyshop Punk",
      shortNote: "Xerox red",
      description: "Off-white paper, signal red accents, poster typography.",
      colorScheme: "light",
    },
    {
      id: "flyer",
      label: "Flyer",
      title: "Underground Techno Flyer",
      shortNote: "Acid lime",
      description: "Dark brutalist panels, acid lime accents, tighter geometry.",
      colorScheme: "dark",
    },
    {
      id: "garage",
      label: "Garage",
      title: "Garage Tool",
      shortNote: "Safety orange",
      description: "Warm industrial surfaces, stencil cues, workshop energy.",
      colorScheme: "dark",
    },
  ]);

  function getStorageApi() {
    return globalScope.SinepanelCraftSync || null;
  }

  function readCachedThemeId() {
    try {
      const storage = globalScope.localStorage || null;
      if (!storage?.getItem) return DEFAULT_THEME_ID;
      return normalizeThemeId(storage.getItem(THEME_CACHE_KEY));
    } catch (_error) {
      return DEFAULT_THEME_ID;
    }
  }

  function writeCachedThemeId(themeId) {
    try {
      const storage = globalScope.localStorage || null;
      storage?.setItem?.(THEME_CACHE_KEY, normalizeThemeId(themeId));
    } catch (_error) {}
  }

  async function withStorageTimeout(promiseLike, timeoutMs, timeoutMessage) {
    let timeoutId = 0;
    try {
      return await Promise.race([
        Promise.resolve(promiseLike),
        new Promise((_, reject) => {
          timeoutId = globalScope.setTimeout(() => {
            reject(new Error(timeoutMessage));
          }, Math.max(0, Number(timeoutMs) || 0));
        }),
      ]);
    } finally {
      if (timeoutId) {
        globalScope.clearTimeout(timeoutId);
      }
    }
  }

  function normalizeThemeId(value) {
    const candidate = String(value || "").trim().toLowerCase();
    return THEMES.some((theme) => theme.id === candidate) ? candidate : DEFAULT_THEME_ID;
  }

  function getTheme(themeId) {
    const normalizedId = normalizeThemeId(themeId);
    return THEMES.find((theme) => theme.id === normalizedId) || THEMES[0];
  }

  function applyTheme(themeId) {
    const normalizedId = normalizeThemeId(themeId);
    const theme = getTheme(normalizedId);
    const documentRef = globalScope.document;
    const root = documentRef?.documentElement || null;
    const body = documentRef?.body || null;

    if (root?.dataset) {
      root.dataset.theme = normalizedId;
      root.style.colorScheme = theme.colorScheme || "light";
    }

    if (body?.dataset) {
      body.dataset.theme = normalizedId;
    }

    return normalizedId;
  }

  async function readThemeId() {
    const cachedThemeId = readCachedThemeId();
    const storageApi = getStorageApi();
    if (!storageApi?.getValue) {
      return cachedThemeId;
    }
    try {
      const storedTheme = await withStorageTimeout(
        storageApi.getValue(STORAGE_KEY, cachedThemeId),
        STORAGE_TIMEOUT_MS,
        "Reading theme from craft sync timed out.",
      );
      const normalizedThemeId = normalizeThemeId(storedTheme);
      writeCachedThemeId(normalizedThemeId);
      return normalizedThemeId;
    } catch (_error) {
      return cachedThemeId;
    }
  }

  async function writeTheme(themeId) {
    const normalizedId = applyTheme(themeId);
    writeCachedThemeId(normalizedId);
    const storageApi = getStorageApi();
    if (storageApi?.setValue) {
      try {
        await withStorageTimeout(
          storageApi.setValue(STORAGE_KEY, normalizedId),
          STORAGE_TIMEOUT_MS,
          "Saving theme into craft sync timed out.",
        );
      } catch (error) {
        console.warn("[app-theme] failed to persist theme in craft sync", error);
      }
    }
    return normalizedId;
  }

  async function hydrateTheme() {
    const themeId = await readThemeId();
    return applyTheme(themeId);
  }

  globalScope.SinepanelAppTheme = {
    STORAGE_KEY,
    THEME_CACHE_KEY,
    DEFAULT_THEME_ID,
    THEMES,
    normalizeThemeId,
    getTheme,
    applyTheme,
    readThemeId,
    writeTheme,
    hydrateTheme,
  };
})(globalThis);
