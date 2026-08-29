import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

await import('../registry-launch-smoke.mjs');
await import('./app-commands.test.mjs');
await import('./responsive.test.mjs');

// --- Wartungs-Schreibschutz darf den Mount nicht abbrechen (Befund 29.08.2026)
// Der Desktop saet beim Mounten Icons und Layout. Waehrend eines Upgrades sind
// diese Sammlungen schreibgeschuetzt; der Fehler wurde durchgereicht und brach
// den Mount ab. Damit entstand eine Verklemmung: der Wartungs-Lease endet erst,
// wenn der Browser seine Sammlungen bestaetigt, und diese Bestaetigung setzt
// einen erfolgreichen Mount voraus. Gemessen auf thesen.ctox.dev: Desktop ohne
// jedes Symbol, Sync dauerhaft bei "0/3 - Wartet auf Icons, Layout", Lease bei
// gesundem Peer endlos verlaengert.
{
  const quelle = await readFile(new URL('../index.js', import.meta.url), 'utf8');

  assert.ok(
    /function isMaintenanceReadOnlyError/.test(quelle),
    'der Wartungs-Schreibschutz muss als eigener, erwarteter Zustand erkannt werden',
  );
  assert.ok(
    /CTOX_MAINTENANCE_READ_ONLY/.test(quelle),
    'die Erkennung muss den Fehlercode selbst pruefen, nicht nur den Meldungstext',
  );

  // Beide Saatpfade muessen den Fehler tolerieren, nicht nur einer.
  const treffer = quelle.match(/if \(!isTransientSeedError\(error\)\) throw error;/g) || [];
  assert.equal(treffer.length, 3, 'alle drei Saatpfade muessen tolerant sein: Icons saeen, Layout saeen, Icons lesen');
  assert.ok(
    !/if \(!isDatabaseClosingError\(error\)\) throw error;/.test(quelle),
    'kein Saatpfad darf weiterhin nur den Datenbank-Neustart abfangen',
  );

  // Die Klassifizierung selbst pruefen.
  const fn = new Function(`
    ${quelle.match(/function isDatabaseClosingError[\s\S]*?\n  \}/)[0]}
    ${quelle.match(/function isMaintenanceReadOnlyError[\s\S]*?\n  \}/)[0]}
    return isMaintenanceReadOnlyError;
  `.replace(/^  /gm, ''))();
  const wartung = new Error('CTOX wird aktualisiert. desktop_icons bleibt vorübergehend schreibgeschützt.');
  wartung.code = 'CTOX_MAINTENANCE_READ_ONLY';
  assert.equal(fn(wartung), true, 'der echte Wartungsfehler muss erkannt werden');
  assert.equal(fn(new Error('irgendein anderer Fehler')), false, 'fremde Fehler duerfen nicht verschluckt werden');

  console.log('  ok: Desktop mountet auch im Wartungs-Schreibschutz');
}
