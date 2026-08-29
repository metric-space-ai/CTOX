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
// einen erfolgreichen Mount voraus. Auf einer Produktionsinstanz trat der
// Desktop deshalb ohne jedes Symbol auf; Sync blieb dauerhaft bei
// "0/3 - Wartet auf Icons, Layout" und der Lease wurde trotz gesundem Peer
// endlos verlaengert.
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

// --- Symbole duerfen einander nicht verdecken (E2E-01-Befund 29.08.2026) -----
// normalizeIconLayoutIfNeeded prueste nur auf exakt gleiche Koordinaten. Auf
// In der Referenzinstanz lagen "Outbound Lead Generation" (128, 520) und
// "Bugs & Features" (99, 556) mit 7455 px2 uebereinander: Beschriftungen
// unlesbar, und das obere Symbol fing die Klicks des unteren ab, sodass die
// Outbound-App nicht zu oeffnen war. Verschiedene Schluessel, also keine
// erkannte Kollision.
{
  const quelle = await readFile(new URL('../index.js', import.meta.url), 'utf8');

  assert.ok(
    !/const key = `\$\{Math\.round\(doc\.x \|\| 0\)\}:\$\{Math\.round\(doc\.y \|\| 0\)\}`/.test(quelle),
    'die Kollisionspruefung darf nicht auf exakte Koordinatengleichheit zurueckfallen',
  );
  assert.ok(
    /zelle\.w - Math\.abs/.test(quelle) && /zelle\.h - Math\.abs/.test(quelle),
    'geprueft werden muss die tatsaechliche Ueberdeckung der Rasterzellen',
  );

  // Die Erkennung isoliert nachrechnen, mit den realen Mindestmassen.
  const zelle = { w: 104, h: 120 };
  const schwelle = zelle.w * zelle.h * 0.1;
  const kollidiert = (docs) => {
    for (let i = 0; i < docs.length; i += 1) {
      for (let j = i + 1; j < docs.length; j += 1) {
        const dx = Math.max(0, zelle.w - Math.abs(docs[i].x - docs[j].x));
        const dy = Math.max(0, zelle.h - Math.abs(docs[i].y - docs[j].y));
        if (dx * dy > schwelle) return true;
      }
    }
    return false;
  };

  // Der echte Produktivfall muss anschlagen.
  assert.equal(kollidiert([{ x: 128, y: 520 }, { x: 99, y: 556 }]), true,
    'die gemessene Ueberlappung von Outbound und Bugs & Features muss erkannt werden');

  // Ein sauber besetztes Raster darf NICHT staendig neu angeordnet werden.
  const sauber = [];
  for (let c = 0; c < 4; c += 1) for (let r = 0; r < 5; r += 1) sauber.push({ x: 24 + c * zelle.w, y: 24 + r * zelle.h });
  assert.equal(kollidiert(sauber), false, 'ein sauberes Raster ist keine Kollision');

  // Exakte Deckung muss weiterhin erkannt werden (alte Faehigkeit).
  assert.equal(kollidiert([{ x: 24, y: 24 }, { x: 24, y: 24 }]), true, 'exakte Deckung bleibt eine Kollision');

  // Knappe Beruehrung benachbarter Zellen ist zulaessig.
  assert.equal(kollidiert([{ x: 24, y: 24 }, { x: 123, y: 139 }]), false, 'knappe Beruehrung ist keine Kollision');

  console.log('  ok: einander verdeckende Symbole werden neu angeordnet');
}
