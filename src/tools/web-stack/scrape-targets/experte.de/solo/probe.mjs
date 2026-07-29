// experte.de — SOLO-FIRST live probe (plain Playwright, no CTOX stack).
//
// Usage: node probe.mjs "info@bnt-chemicals.de"
//
// Drives https://www.experte.de/email-pruefen, submits the e-mail address,
// waits for the provider verdict and prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only on real extraction; non-zero with {reason} otherwise.

import { chromium } from "playwright";

const TARGET = "experte.de";
const START_URL = "https://www.experte.de/email-pruefen";
const MIN_NAV_GAP_MS = 2000;

const email = String(process.argv[2] || "").trim().toLowerCase();

function fail(reason, extra = {}) {
  process.stdout.write(JSON.stringify({
    target: TARGET,
    input: email,
    fetched_at: new Date().toISOString(),
    fields: {},
    reason,
    ...extra,
  }) + "\n");
  process.exit(1);
}

if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)) {
  fail("invalid_email_argument", { usage: "node probe.mjs <email-address>" });
}

let lastNav = 0;
async function politeGoto(page, url, options) {
  const wait = MIN_NAV_GAP_MS - (Date.now() - lastNav);
  if (wait > 0) await page.waitForTimeout(wait);
  const response = await page.goto(url, options);
  lastNav = Date.now();
  return response;
}

async function dismissConsent(page) {
  // Ordinary consent dialogs (Cookiebot & co.): click the visible accept
  // button. Never touch CAPTCHAs or bot challenges.
  const candidates = [
    page.locator("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"),
    page.locator("#CybotCookiebotDialogBodyButtonAccept"),
    page.getByRole("button", { name: /alle akzeptieren|akzeptieren|zustimmen|accept all|agree/i }),
  ];
  for (const locator of candidates) {
    try {
      const button = locator.first();
      if (await button.count() && await button.isVisible({ timeout: 1500 }).catch(() => false)) {
        await button.click({ timeout: 3000 });
        await page.waitForTimeout(800);
        return true;
      }
    } catch (_err) {
      // keep trying the next candidate
    }
  }
  return false;
}

const VERDICT_LABELS = { valid: "valid", invalid: "invalid", unknown: "unknown" };

function hostOf(value) {
  try {
    return new URL(value).hostname.toLowerCase().replace(/^www\./, "");
  } catch (_err) {
    return "";
  }
}

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ locale: "de-DE" });
  const page = await context.newPage();

  const response = await politeGoto(page, START_URL, {
    waitUntil: "domcontentloaded",
    timeout: 45000,
  });
  if (!response || !response.ok()) {
    fail("navigation_failed", { http_status: response?.status() ?? null });
  }
  await page.waitForLoadState("networkidle", { timeout: 10000 }).catch(() => null);

  const bodyText = (await page.locator("body").innerText().catch(() => "")).slice(0, 6000);
  if (/captcha|cloudflare|verify you are human|access denied|zugriff verweigert/i.test(bodyText)) {
    fail("blocked", { detail: "access challenge detected; not attempting to bypass" });
  }

  await dismissConsent(page);

  // Proven against the live site (2026-07-29): the checker is an Angular
  // widget on the "Einzelne E-Mail" tab — a single visible
  //   <input type="url" placeholder="E-Mail eingeben">
  // plus a sibling
  //   <button class="btn btn-primary mt-2">E-Mail prüfen</button>
  // (no enclosing <form>).
  const field = page.locator('input[type="url"][placeholder*="E-Mail" i]:visible').first();
  if ((await field.count()) < 1) {
    fail("selector_drift", { detail: "email input not found on loaded page" });
  }
  await field.fill(email);

  const submit = page.locator('button.btn-primary:has-text("prüfen"), button:has-text("E-Mail prüfen")').first();
  if ((await submit.count()) < 1) {
    fail("selector_drift", { detail: "submit button not found on loaded page" });
  }

  lastNav = Date.now();
  await submit.click();

  // Proven result contract: the verdict renders asynchronously as a table
  //   <table><thead><tr><th>E-Mail</th><th>Ergebnis</th>…</tr></thead>
  //   <tbody><tr><td>{email}</td><td class="font-bold text-green">Gültig</td>…
  // Wait until the row for OUR address shows a verdict word in its 2nd cell.
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
    fail("verdict_timeout", { detail: "no verdict row within 60s after submit" });
  }

  const rows = await page.evaluate(() =>
    Array.from(document.querySelectorAll("table tbody tr")).map((row) =>
      Array.from(row.querySelectorAll("td")).map((cell) => (cell.innerText || "").replace(/\s+/g, " ").trim())),
  );
  const row = rows.find((cells) => cells.length >= 2 && cells[0].toLowerCase() === email);
  if (!row) {
    fail("verdict_unparseable", { detail: "result table has no row for the input address" });
  }
  // Mind the substring trap: "Ungültig" contains "Gültig" — anchor at start.
  const verdictText = row[1];
  const status = /^ungültig/i.test(verdictText) ? "invalid"
    : /^gültig/i.test(verdictText) ? "valid"
    : /unbekannt|riskant|catch[ -]?all|fehler/i.test(verdictText) ? "unknown"
    : null;
  if (!status) {
    fail("verdict_unparseable", {
      detail: `unrecognized verdict text: ${verdictText}`,
      row: row.join(" | ").slice(0, 400),
    });
  }
  const evidence = row.filter(Boolean).join(" | ");

  const sourceUrl = page.url() || START_URL;
  if (hostOf(sourceUrl) !== "experte.de") {
    fail("wrong_origin", { detail: `unexpected result origin: ${sourceUrl}` });
  }
  process.stdout.write(JSON.stringify({
    target: TARGET,
    input: email,
    fetched_at: new Date().toISOString(),
    fields: {
      person_email_validation: {
        value: VERDICT_LABELS[status],
        source_url: sourceUrl,
      },
      person_email: {
        value: email,
        source_url: sourceUrl,
      },
      firma_domain: {
        value: email.split("@").pop(),
        source_url: sourceUrl,
      },
    },
    evidence: evidence.slice(0, 700),
    page_title: await page.title().catch(() => ""),
  }, null, 2) + "\n");
  process.exit(0);
} catch (error) {
  fail("probe_error", { detail: String(error?.message || error).slice(0, 400) });
} finally {
  await browser.close().catch(() => null);
}
