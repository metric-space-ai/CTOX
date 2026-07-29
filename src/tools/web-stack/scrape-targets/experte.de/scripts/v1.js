// experte.de — prospect.v1 extractor (Phase B initial revision).
//
// Reads CTOX_SCRAPE_INPUT_JSON for the e-mail address, drives the CTOX web
// stack (`ctox web browser-automation`) through the live
// https://www.experte.de/email-pruefen form and parses the verdict table
// for the field set documented in `tools/web-stack/src/sources/EXCEL_MATRIX.md`
// (experte.de emits exactly one field: person_email_validation).
//
// Flow proven solo-first against the live site on 2026-07-29 with
// scrape-targets/experte.de/solo/probe.mjs ("info@bnt-chemicals.de" →
// "Gültig", a nonsense address → "Ungültig"):
//   - the checker is an Angular widget on the "Einzelne E-Mail" tab,
//     a single visible <input type="url" placeholder="E-Mail eingeben">
//     plus a sibling <button class="btn btn-primary mt-2">E-Mail prüfen</button>
//     (no enclosing <form>),
//   - the verdict renders asynchronously as
//     <table><thead><tr><th>E-Mail</th><th>Ergebnis</th>…</thead>
//     <tbody><tr><td>{email}</td><td class="font-bold text-green">Gültig</td>…
//   - substring trap: "Ungültig" contains "Gültig" — verdicts are anchored.
//
// Drift contract: if the selectors below stop matching but the checker page
// loads successfully, this script returns an empty records array (failure
// mode portal_drift), never a crash. `ctox scrape execute --allow-heal` then
// enqueues a `universal-scraping` repair task that will revise this very file.

"use strict";

const { execFileSync } = require("child_process");

const SOURCE_ID = "experte.de";
const ALLOWED_HOST = "experte.de";
const START_URL = "https://www.experte.de/email-pruefen";
const CONCLUSIVE_STATUSES = new Set(["valid", "invalid", "unknown"]);

function readInput() {
  const raw = process.env.CTOX_SCRAPE_INPUT_JSON;
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch (err) {
    process.stderr.write("invalid CTOX_SCRAPE_INPUT_JSON: " + err.message + "\n");
    return {};
  }
}

function ctoxBin() {
  return process.env.CTOX_BIN || "ctox";
}

function runCtox(args, input, timeout = 90_000) {
  try {
    const out = execFileSync(ctoxBin(), args, {
      encoding: "utf8",
      input,
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
      maxBuffer: 32 * 1024 * 1024,
      timeout,
    });
    return JSON.parse(out);
  } catch (_err) {
    // Stay silent on per-attempt failures: `classify_outcome` in
    // src/capabilities/scrape.rs runs a substring search for "temporary",
    // "timeout", "429", … on stderr and would misclassify the whole run.
    // Fatal-only stderr stays in main().
    return null;
  }
}

function normalized(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("de-DE")
    .replace(/\u00df/g, "ss")
    .replace(/\s+/g, " ")
    .trim();
}

function isPortalOrLoginTitle(title) {
  const text = String(title || "").replace(/\s+/g, " ").trim();
  if (!text) return false;
  return /\b(?:log[ -]?in|sign[ -]?in|anmeld(?:en|ung)|authentication|authentifizierung|kundenportal|customer portal)\b/i.test(text)
    || /^(?:portal|startseite|home|willkommen)(?:\s*[-|:]\s*.*)?$/i.test(text);
}

function isAllowedUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname.toLowerCase().replace(/^www\./, "") === ALLOWED_HOST;
  } catch (_err) {
    return false;
  }
}

function hasBlockedDetection(page) {
  const markers = Array.isArray(page?.detection?.markers) ? page.detection.markers.join(" ") : "";
  return /captcha|cloudflare|challenge|turnstile|access[_ -]?denied|request[_ -]?blocked|rate[_ -]?limit/i.test(markers);
}

function isBlockedPage(page) {
  const corpus = normalized([page?.title, page?.evidence].filter(Boolean).join(" "));
  return /captcha|cloudflare|verify you are human|access denied|zugriff verweigert|sicherheitsuberprufung/.test(corpus);
}

// Browser-automation source: replay of the solo probe inside the CTOX
// browser sandbox. Returns {email, status, evidence, url, title, reason?}.
// Selector drift is reported via reason ("selector_drift: …") instead of an
// exception so main() can classify portal_drift deterministically.
function experteBrowserSource(email) {
  return `
    const email = ${JSON.stringify(email)};
    const startUrl = ${JSON.stringify(START_URL)};

    await ctoxBrowser.goto(startUrl, { timeoutMs: 45000 });
    await page.waitForLoadState("networkidle", { timeout: 10000 }).catch(() => null);

    const bodyStart = await page.locator("body").innerText().catch(() => "");
    if (/captcha|cloudflare|verify you are human|access denied|zugriff verweigert/i.test(bodyStart.slice(0, 6000))) {
      return { email, status: "blocked", evidence: bodyStart.slice(0, 400), url: page.url(), title: await page.title().catch(() => "") };
    }

    // Ordinary consent dialog (Cookiebot & co.): click the visible accept
    // button. Never touch CAPTCHAs or bot challenges.
    const consentCandidates = [
      page.locator("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"),
      page.locator("#CybotCookiebotDialogBodyButtonAccept"),
      page.getByRole("button", { name: /alle akzeptieren|akzeptieren|zustimmen|accept all|agree/i }),
    ];
    for (const locator of consentCandidates) {
      try {
        const button = locator.first();
        if (await button.count() && await button.isVisible({ timeout: 1500 }).catch(() => false)) {
          await button.click({ timeout: 3000 });
          await page.waitForTimeout(800);
          break;
        }
      } catch (_err) { /* next candidate */ }
    }

    const field = page.locator('input[type="url"][placeholder*="E-Mail" i]:visible').first();
    if ((await field.count()) < 1) {
      return { email, status: "failed", reason: "selector_drift: email input not found on loaded page", url: page.url(), title: await page.title().catch(() => "") };
    }
    await field.fill(email);

    const submit = page.locator('button.btn-primary:has-text("prüfen"), button:has-text("E-Mail prüfen")').first();
    if ((await submit.count()) < 1) {
      return { email, status: "failed", reason: "selector_drift: submit button not found on loaded page", url: page.url(), title: await page.title().catch(() => "") };
    }
    await submit.click();

    const rowReady = await page.waitForFunction((needle) => {
      for (const row of Array.from(document.querySelectorAll("table tbody tr"))) {
        const cells = Array.from(row.querySelectorAll("td"));
        if (cells.length >= 2
            && (cells[0].innerText || "").trim().toLowerCase() === needle
            && /gültig|unbekannt|riskant|fehler/i.test((cells[1].innerText || "").trim())) {
          return true;
        }
      }
      return false;
    }, email, { timeout: 60000 }).then(() => true).catch(() => false);
    if (!rowReady) {
      return { email, status: "failed", reason: "verdict_timeout: no verdict row within 60s after submit", url: page.url(), title: await page.title().catch(() => "") };
    }

    const rows = await page.evaluate(() =>
      Array.from(document.querySelectorAll("table tbody tr")).map((row) =>
        Array.from(row.querySelectorAll("td")).map((cell) => (cell.innerText || "").replace(/\\s+/g, " ").trim())),
    );
    const row = rows.find((cells) => cells.length >= 2 && cells[0].toLowerCase() === email);
    if (!row) {
      return { email, status: "failed", reason: "verdict_unparseable: no result row for the input address", url: page.url(), title: await page.title().catch(() => "") };
    }
    const verdictText = row[1];
    const status = /^ungültig/i.test(verdictText) ? "invalid"
      : /^gültig/i.test(verdictText) ? "valid"
      : /unbekannt|riskant|catch[ -]?all|fehler/i.test(verdictText) ? "unknown"
      : null;
    if (!status) {
      return { email, status: "failed", reason: "verdict_unparseable: unrecognized verdict text " + verdictText, url: page.url(), title: await page.title().catch(() => "") };
    }
    return {
      email,
      status,
      evidence: row.filter(Boolean).join(" | ").slice(0, 700),
      url: page.url(),
      title: await page.title().catch(() => ""),
    };
  `;
}

function validateEmail(email) {
  const payload = runCtox(
    ["web", "browser-automation", "--timeout-ms", "120000"],
    experteBrowserSource(email),
    130_000,
  );
  if (!payload) return null;
  return { ...(payload.result || {}), ok: payload.ok === true, detection: payload.detection };
}

function recordUnlockSignal(url, markers) {
  return runCtox([
    "web", "unlock", "signals", "record",
    "--source", "scrape-target:experte.de",
    "--url", isAllowedUrl(url) ? url : START_URL,
    "--evidence", JSON.stringify({
      source_id: SOURCE_ID,
      detection: "access_challenge",
      markers: [...new Set((markers || []).map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ], undefined, 20_000);
}

function main() {
  const input = readInput();
  const email = String(input.email || "").trim().toLowerCase();
  if (!email) {
    process.stdout.write(JSON.stringify({
      records: [],
      failure_mode: "portal_drift",
      detail: "CTOX_SCRAPE_INPUT_JSON.email missing",
    }));
    return;
  }

  const validation = validateEmail(email);
  const blocked = hasBlockedDetection(validation)
    || isBlockedPage(validation)
    || validation?.status === "blocked";
  if (blocked) {
    recordUnlockSignal(
      isAllowedUrl(validation?.url) ? validation.url : START_URL,
      validation?.detection?.markers || ["access_challenge"],
    );
    process.stdout.write(JSON.stringify({
      records: [],
      failure_mode: "blocked",
      detail: "experte.de access challenge recorded by CTOX browser automation for web-unlock",
    }));
    return;
  }

  // Identity + origin gate: only conclusive provider evidence for the exact
  // requested address, rendered on the provider's own origin (never a
  // portal/login landing), is accepted.
  const accepted = validation?.ok
    && validation?.email === email
    && CONCLUSIVE_STATUSES.has(validation?.status)
    && isAllowedUrl(validation?.url)
    && !isPortalOrLoginTitle(validation?.title);
  if (accepted) {
    const sourceUrl = new URL(validation.url).href;
    process.stdout.write(JSON.stringify({
      records: [{
        field: "person_email_validation",
        value: validation.status,
        confidence: validation.status === "unknown" ? "medium" : "high",
        source_url: sourceUrl,
        note: `EXPERTE.de verdict: ${String(validation.evidence || `${email} ${validation.status}`).slice(0, 300)}`,
      }],
    }));
    return;
  }

  const reason = String(validation?.reason || "");
  const drift = reason.startsWith("selector_drift")
    || reason.startsWith("verdict_unparseable");
  process.stdout.write(JSON.stringify({
    records: [],
    failure_mode: drift ? "portal_drift" : "temporary_unreachable",
    detail: reason
      || (validation?.ok === false ? "experte.de browser automation failed before provider evidence was available"
        : "experte.de did not return conclusive provider evidence"),
  }));
}

if (require.main === module) {
  main();
}

module.exports = {
  experteBrowserSource,
  hasBlockedDetection,
  isAllowedUrl,
  isBlockedPage,
  isPortalOrLoginTitle,
};
