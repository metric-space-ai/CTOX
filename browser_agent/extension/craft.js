import "./vendor/react-libs.js";
import {
  CraftWorkspacePage,
  normalizeCraftViewTab,
} from "./options-workspace.js";

const __g =
  typeof globalThis !== "undefined"
    ? globalThis
    : typeof window !== "undefined"
      ? window
      : self;

var react = __g.react || { exports: {} };
var reactDom = __g.reactDom || { exports: {} };
var client = __g.client || {};
var hasRequiredReact;
var hasRequiredReactDom;
var hasRequiredClient;
var requireReact_production_min = __g.requireReact_production_min;
var requireReactDom_production_min = __g.requireReactDom_production_min;
var getDefaultExportFromCjs =
  __g.getDefaultExportFromCjs ||
  function getDefaultExportFromCjsLocal(module) {
    return module && module.__esModule ? module.default : module;
  };

var requireReact =
  typeof requireReact !== "undefined"
    ? requireReact
    : function requireReactLocal() {
        if (hasRequiredReact) return react.exports;
        hasRequiredReact = 1;
        react.exports = requireReact_production_min();
        return react.exports;
      };

function requireReactDom() {
  if (hasRequiredReactDom) return reactDom.exports;
  hasRequiredReactDom = 1;
  reactDom.exports = requireReactDom_production_min();
  return reactDom.exports;
}

function requireClient() {
  if (hasRequiredClient) return client;
  hasRequiredClient = 1;
  const module = requireReactDom();
  client.createRoot = module.createRoot;
  client.hydrateRoot = module.hydrateRoot;
  return client;
}

const clientExports = requireClient();
const reactExports = requireReact();
const React = getDefaultExportFromCjs(reactExports);
const h = React.createElement;
const themeApi = globalThis.SinepanelAppTheme || null;
let craftSyncLoadPromise = null;

async function ensureCraftSyncLoaded() {
  if (globalThis.SinepanelCraftSync) return globalThis.SinepanelCraftSync;
  if (craftSyncLoadPromise) return craftSyncLoadPromise;

  craftSyncLoadPromise = import("./shared/craft-sync.js")
    .then(() => {
      const loaded = globalThis.SinepanelCraftSync || null;
      if (!loaded) {
        throw new Error("Craft sync runtime did not register on globalThis.");
      }
      return loaded;
    })
    .catch((error) => {
      console.warn("[craft] craft sync bootstrap failed", error);
      return null;
    })
    .finally(() => {
      craftSyncLoadPromise = null;
    });

  return craftSyncLoadPromise;
}

function readCraftRoute() {
  try {
    const url = new URL(location.href);
    return {
      craftId: String(url.searchParams.get("craft") || "").trim(),
      tab:
        normalizeCraftViewTab(url.searchParams.get("tab")) ||
        normalizeCraftViewTab(String(location.hash || "").replace(/^#/, "").split("/")[0]),
    };
  } catch (_error) {
    return {
      craftId: "",
      tab: "model",
    };
  }
}

function writeCraftRoute(route = {}) {
  const url = new URL(location.href);
  const craftId = String(route.craftId || "").trim();
  const tab = normalizeCraftViewTab(route.tab);

  if (craftId) {
    url.searchParams.set("craft", craftId);
  } else {
    url.searchParams.delete("craft");
  }
  url.searchParams.set("tab", tab);
  url.hash = tab;
  history.replaceState(null, "", url.toString());
}

function useCraftRoute() {
  const [route, setRoute] = React.useState(readCraftRoute());

  React.useEffect(() => {
    const handleLocationChange = () => {
      setRoute(readCraftRoute());
    };
    window.addEventListener("hashchange", handleLocationChange);
    window.addEventListener("popstate", handleLocationChange);
    return () => {
      window.removeEventListener("hashchange", handleLocationChange);
      window.removeEventListener("popstate", handleLocationChange);
    };
  }, []);

  const updateRoute = React.useCallback((patch = {}) => {
    const current = readCraftRoute();
    const nextRoute = {
      ...current,
      ...patch,
    };
    nextRoute.tab = normalizeCraftViewTab(nextRoute.tab);
    writeCraftRoute(nextRoute);
    setRoute(readCraftRoute());
  }, []);

  return [route, updateRoute];
}

function CraftApp() {
  const [route, updateRoute] = useCraftRoute();
  return h("div", { className: "craft-app" }, [
    h(CraftWorkspacePage, {
      route,
      onRouteChange: updateRoute,
      key: route.craftId || route.tab,
    }),
  ]);
}

async function init() {
  let initialThemeId = themeApi?.DEFAULT_THEME_ID || "copyshop";
  try {
    initialThemeId = (await themeApi?.hydrateTheme?.()) || initialThemeId;
  } catch (_error) {
    themeApi?.applyTheme?.(initialThemeId);
  }
  const craftSync = await ensureCraftSyncLoaded();
  try {
    await craftSync?.ensureStartedFromSettings?.({ pageName: "Craft Workspace" });
  } catch (error) {
    console.warn("[craft] craft sync warmup failed", error);
  }

  const craftRoot = document.querySelector("#craft-root");
  if (!craftRoot) {
    throw new Error("Can not find #craft-root");
  }
  const root = clientExports.createRoot(craftRoot);
  root.render(h(CraftApp));
}

void init();
