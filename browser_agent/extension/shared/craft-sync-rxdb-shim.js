import * as rx from "../vendor/rxdb-bundle.js"

const ENABLE_RXDB_DEV_MODE = false

;(function ensureNextTick() {
  const scope =
    typeof globalThis !== "undefined"
      ? globalThis
      : typeof self !== "undefined"
        ? self
        : window

  if (!scope.process) scope.process = {}
  if (typeof scope.process.nextTick === "function") return

  scope.process.nextTick = function nextTick(callback, ...args) {
    Promise.resolve().then(() => {
      callback(...args)
    })
  }
})()

let validatedDexieStorage = null
let pluginsInstalled = false

function installPlugin(plugin) {
  if (!plugin) return
  const addRxPlugin = rx.addRxPlugin || rx.addRxDBPlugin || rx.addPlugin
  if (typeof addRxPlugin !== "function") return
  try {
    addRxPlugin(plugin)
  } catch (_error) {
    // RxDB throws when the same plugin is added twice across extension pages.
  }
}

function ensurePlugins() {
  if (pluginsInstalled) return
  pluginsInstalled = true

  if (ENABLE_RXDB_DEV_MODE) {
    installPlugin(rx.RxDBDevModePlugin || null)
  }
  installPlugin(rx.RxDBMigrationSchemaPlugin || rx.RxDBMigrationPlugin || null)
  installPlugin(rx.RxDBQueryBuilderPlugin || rx.QueryBuilderPlugin || rx.RxDBQueryBuilder || null)
}

function getValidatedDexieStorage() {
  if (validatedDexieStorage) return validatedDexieStorage
  if (typeof rx.getRxStorageDexie !== "function") {
    throw new Error("[craft-sync] RxDB Dexie storage is unavailable in the vendor bundle.")
  }

  const baseStorage = rx.getRxStorageDexie()
  if (typeof rx.wrappedValidateZSchemaStorage === "function") {
    validatedDexieStorage = rx.wrappedValidateZSchemaStorage({ storage: baseStorage })
    return validatedDexieStorage
  }
  if (typeof rx.wrappedValidateAjvStorage === "function") {
    validatedDexieStorage = rx.wrappedValidateAjvStorage({ storage: baseStorage })
    return validatedDexieStorage
  }

  validatedDexieStorage = baseStorage
  return validatedDexieStorage
}

export function rxdbCore() {
  ensurePlugins()
  return {
    createRxDatabase: rx.createRxDatabase,
    getRxStorageDexie: getValidatedDexieStorage,
  }
}

export function rxdbWebRTC() {
  return {
    replicateWebRTC: rx.replicateWebRTC,
    getConnectionHandlerSimplePeer: rx.getConnectionHandlerSimplePeer,
  }
}
