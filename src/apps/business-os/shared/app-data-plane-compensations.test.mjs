// Origin: CTOX
// License: Apache-2.0
//
// I-006: Quelltext-Vertrag, kein Verhaltenstest — und das ist Absicht.
//
// app.js hat rund 12.000 Zeilen und kein Testgerüst, das den Startpfad der
// Datenebene ausführen könnte. Was hier geprüft wird, ist deshalb die Form:
// die vier Kompensationen sind fort, und die Ursachen, die sie nötig machten,
// sind an ihrer Stelle behoben. Das Repo kennt diese Art Wächter bereits
// (`module_boundaries_hold`, `text_matching_stays_within_its_declared_budget`).
//
// Ein Mustertest ist schwächer als ein Verhaltenstest. Er ist aber stärker als
// gar keiner — und ohne ihn wäre der Beweis für diese Reparatur eine Datei in
// /tmp geblieben, die niemand ausführt.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const app = readFileSync(join(here, '..', 'app.js'), 'utf8');

test('die vier Reparaturen der Datenebene sind fort', () => {
  for (const name of [
    'repairBusinessDataPlane',
    'scheduleSyncRecoveryRepairIfNeeded',
    'repairRecoveringDataPlane',
    'recoverFromLocalRxDbSchemaDrift',
  ]) {
    assert.ok(
      !new RegExp(`function\\s+${name}\\b`).test(app),
      `${name} existiert noch — die Kompensation ist zurück`,
    );
  }
});

test('die Startkonfiguration wird normalisiert statt nachträglich repariert', () => {
  assert.ok(/normalizeBusinessOsLaunchConfig\(launch, firstObject\(/.test(app));
  assert.ok(/declaredIceServers\.length \? declaredIceServers : fallbackIceServers/.test(app));
  assert.ok(/transportFallback\?\.ice_servers_refresh_url/.test(app));
});

test('ein lokaler Startfehler wird behandelt, nicht mit einem Reset überdeckt', () => {
  const openDb =
    app.match(/async function openBusinessDbAndRegisterCoreCollections\(dbName\) \{([\s\S]*?)\n\}/)?.[1] || '';
  assert.ok(openDb.length > 0, 'openBusinessDbAndRegisterCoreCollections nicht gefunden');
  assert.ok(/try \{\s*state\.db = await createBusinessDb/.test(openDb));
  assert.ok(/isLocalRxDbStartupError\(error\)/.test(openDb));
  assert.ok(!/resetBusinessDb/.test(openDb), 'der Startpfad greift wieder zum Reset');
});

test('ein Modul-Rollback lädt neu, statt Schemaregistrierungen zu löschen', () => {
  const rollback = app.match(/async function rollbackModuleSelection\([\s\S]*?\n\}/)?.[0] || '';
  assert.ok(rollback.length > 0, 'rollbackModuleSelection nicht gefunden');
  assert.ok(/await reloadAfterModuleSchemaChange\(/.test(rollback));
  assert.ok(!/schemaRegistrations\.delete/.test(rollback));
});

test('resetBusinessDb bleibt der eine ausdrückliche Notausgang', () => {
  // Mehr als ein Aufruf hiesse, dass der Reset wieder zum Reflex geworden ist.
  assert.equal([...app.matchAll(/resetBusinessDb\s*\(/g)].length, 1);
});
