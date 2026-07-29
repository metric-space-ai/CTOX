// bundesanzeiger.de - direct Playwright extractor for prospect.v1.
//
// Repaired 2026-07-29 against the LIVE site (see solo/probe.mjs for the
// standalone plain-Playwright proof):
//  - consent banner is clicked via role/name patterns (exact label first),
//    tolerating late-rendering banners instead of requiring count === 1;
//  - the search submit is honeypot-aware: the form contains a hidden
//    decoy input[name="search-button"] (tabindex -1, no value); only the
//    visible input[name="search-button"][value="Suchen"] is clicked;
//  - an explicit "keine passenden Daten gefunden" page is recognised as a
//    real (empty) results page instead of being misread as drift;
//  - one resubmit retry covers Wicket session bounces back to the form;
//  - the per-document "Sicherheitsabfrage" (image CAPTCHA) that gates every
//    Rechnungslegung document is treated as an access challenge and is
//    NEVER opened or solved; extraction stays on the public results table.
//
// Drift contract: if the selectors below stop matching but a result page
// loads successfully, this script returns an empty records array with
// failure_mode "portal_drift" - never a crash.

"use strict";

const { execFileSync } = require("child_process");

const ALLOWED_HOST = "bundesanzeiger.de";
const SEARCH_URL = "https://www.bundesanzeiger.de/pub/de/suche?0";
const BROWSER_TIMEOUT_MS = 120_000;

function readInput() {
  const raw = process.env.CTOX_SCRAPE_INPUT_JSON;
  if (!raw) return { company: "", country: "" };
  try {
    return JSON.parse(raw);
  } catch (err) {
    process.stderr.write("invalid CTOX_SCRAPE_INPUT_JSON: " + err.message + "\n");
    return { company: "", country: "" };
  }
}

function ctoxBin() {
  return process.env.CTOX_BIN || "ctox";
}

function runCtox(args, input) {
  try {
    const out = execFileSync(ctoxBin(), args, {
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
      input,
      timeout: BROWSER_TIMEOUT_MS + 10_000,
      maxBuffer: 32 * 1024 * 1024,
    });
    return JSON.parse(out);
  } catch {
    return null;
  }
}

function recordUnlockSignal(url, markers) {
  const safeUrl = allowedSourceUrl(url);
  return runCtox([
    "web", "unlock", "signals", "record",
    "--source", "scrape-target:bundesanzeiger.de",
    "--url", safeUrl?.href || SEARCH_URL,
    "--evidence", JSON.stringify({
      source_id: "bundesanzeiger.de",
      detection: "access_challenge",
      markers: [...new Set((markers || []).map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ]);
}

function allowedSourceUrl(raw) {
  try {
    const url = new URL(raw);
    const host = url.hostname.toLowerCase().replace(/\.$/, "");
    if (url.protocol !== "https:" || url.username || url.password) return null;
    if (host !== ALLOWED_HOST && !host.endsWith(`.${ALLOWED_HOST}`)) return null;
    return url;
  } catch {
    return null;
  }
}

function normalizeCompanyName(value) {
  return String(value || "")
    .toLocaleLowerCase("de-DE")
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/\s*\((?:vormals|ehemals|fruher|früher)\s*:[^)]+\)\s*$/i, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function entryMatchesCompany(company, entry) {
  const expected = normalizeCompanyName(company);
  const actual = normalizeCompanyName(entry && entry.name);
  return expected.length > 0 && actual === expected;
}

function selectMatchingEntry(company, entries) {
  return (Array.isArray(entries) ? entries : []).find((entry) =>
    entryMatchesCompany(company, entry),
  ) || null;
}

function buildRecords(entry, sourceUrl) {
  const safeUrl = allowedSourceUrl(sourceUrl);
  if (!entry || !safeUrl || !entry.name) return [];
  const noteParts = ["Bundesanzeiger Suchergebnis"];
  if (entry.section) noteParts.push(entry.section);
  if (entry.information) noteParts.push(entry.information);
  if (entry.date) noteParts.push(`Veroeffentlicht: ${entry.date}`);
  const note = noteParts.join(" - ");
  const records = [{
    field: "firma_name",
    value: entry.name,
    confidence: "medium",
    source_url: safeUrl.href,
    note,
  }];
  if (entry.city) {
    records.push({
      field: "firma_ort",
      value: entry.city,
      confidence: "medium",
      source_url: safeUrl.href,
      note,
    });
  }
  return records;
}

function classifyBrowserResult(company, result) {
  if (!result) {
    return {
      records: [],
      failure_mode: "temporary_unreachable",
      detail: "bundesanzeiger.de browser automation did not return a result",
    };
  }
  if (result.blocked === true) {
    return {
      records: [],
      failure_mode: "blocked",
      detail: "bundesanzeiger.de requires web unlock after a visible access challenge",
    };
  }
  const sourceUrl = allowedSourceUrl(result.url);
  if (!sourceUrl) {
    return {
      records: [],
      failure_mode: "temporary_unreachable",
      detail: "bundesanzeiger.de browser left the allowed origin",
    };
  }
  const entry = selectMatchingEntry(company, result.entries);
  if (entry) return { records: buildRecords(entry, sourceUrl.href) };
  if (result.no_results === true) {
    return {
      records: [],
      failure_mode: "temporary_unreachable",
      detail: "bundesanzeiger.de found no publications for the exact company name",
    };
  }
  if (result.results_page === true) {
    return {
      records: [],
      failure_mode: Array.isArray(result.entries) && result.entries.length > 0
        ? "temporary_unreachable"
        : "portal_drift",
      detail: Array.isArray(result.entries) && result.entries.length > 0
        ? "bundesanzeiger.de returned no exact company match"
        : "bundesanzeiger.de result page did not match known result selectors",
    };
  }
  return {
    records: [],
    failure_mode: "temporary_unreachable",
    detail: "bundesanzeiger.de did not reach a readable result page",
  };
}

function browserSearch(company) {
  const source = `// ctox-browser: timeout_ms=${BROWSER_TIMEOUT_MS}
const searchUrl = ${JSON.stringify(SEARCH_URL)};
const query = ${JSON.stringify(company)};
await page.goto(searchUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForTimeout(1500);

const challengeState = async () => await page.evaluate(() => {
  const text = document.body ? document.body.innerText : "";
  const challenge = document.querySelector(
    'iframe[src*="captcha" i], iframe[src*="challenge" i], .g-recaptcha, [data-sitekey]',
  );
  return {
    blocked: Boolean(challenge) || /schutzma(?:ß|ss)nahme|sicherheitsabfrage|to_nlp_start|cf-chl-|turnstile|captcha|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert|zu viele anfragen/i.test(text),
    noResults: /keine passenden Daten gefunden/i.test(text),
  };
});

if ((await challengeState()).blocked) {
  return { url: page.url(), blocked: true, results_page: false, entries: [] };
}

// Consent: exact label first, then generic patterns; tolerate late banners.
for (const pattern of [
  "Nur technisch notwendige Cookies akzeptieren",
  /alle akzeptieren|akzeptieren|zustimmen|verstanden/i,
]) {
  const button = page.getByRole("button", { name: pattern }).first();
  if (await button.count()) {
    await button.click({ timeout: 2500 }).catch(() => null);
    break;
  }
}
await page.waitForTimeout(1200);

const searchInput = page.locator('input[name="fulltext"]');
if (await searchInput.count() !== 1) {
  return { url: page.url(), blocked: false, results_page: false, entries: [] };
}

let onResults = false;
for (let attempt = 0; attempt < 2 && !onResults; attempt += 1) {
  await searchInput.fill(query);
  // Honeypot-aware: a hidden decoy input[name="search-button"] exists; only
  // the visible valued submit triggers the search.
  const searchButton = page.locator('input[name="search-button"][value="Suchen"]');
  if (await searchButton.count() === 1) {
    await searchButton.click();
  } else {
    await searchInput.press("Enter");
  }
  await page.waitForLoadState("domcontentloaded", { timeout: 30000 }).catch(() => {});
  onResults = await page.waitForSelector(".result_container", { timeout: 20000 })
    .then(() => true)
    .catch(() => false);
  if (!onResults) {
    const state = await challengeState();
    if (state.blocked) {
      return { url: page.url(), blocked: true, results_page: false, entries: [] };
    }
    if (state.noResults) {
      return { url: page.url(), blocked: false, results_page: true, entries: [], no_results: true };
    }
    // Wicket sometimes bounces back to the empty form; retry once.
    if (attempt === 0 && (await searchInput.count()) === 1) {
      await page.waitForTimeout(1500);
      continue;
    }
    return { url: page.url(), blocked: false, results_page: false, entries: [] };
  }
}

return await page.evaluate(() => {
  const text = document.body ? document.body.innerText : "";
  const challenge = document.querySelector(
    'iframe[src*="captcha" i], iframe[src*="challenge" i], .g-recaptcha, [data-sitekey]',
  );
  const blocked = Boolean(challenge) || /schutzma(?:ß|ss)nahme|sicherheitsabfrage|to_nlp_start|cf-chl-|turnstile|captcha|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert|zu viele anfragen/i.test(text);
  const container = document.querySelector(".result_container");
  const entries = container
    ? [...container.querySelectorAll(":scope > .row")].map((row) => {
        if (row.classList.contains("result_header")
            || row.classList.contains("concern_list")
            || row.classList.contains("subsidiary_list")) return null;
        const first = row.querySelector(":scope > .col-md-3 .first");
        if (!first) return null;
        const lines = first.innerText.split("\\n").map((part) => part.trim()).filter(Boolean);
        return {
          name: lines[0] || "",
          city: lines[1] || "",
          section: (row.querySelector(":scope > .col-md-2 .part")?.innerText || "").replace(/\\s+/g, " ").trim(),
          information: (row.querySelector(":scope > .col-md-5 .info > a")?.innerText || "").replace(/\\s+/g, " ").trim(),
          date: (row.querySelector(":scope > .col-md-2 .date")?.innerText || "").replace(/\\s+/g, " ").trim(),
        };
      }).filter((entry) => entry && entry.name)
    : [];
  return {
    url: location.href,
    blocked,
    results_page: Boolean(container) || /Suchergebnis/i.test(document.title),
    entries,
  };
});
`;
  const payload = runCtox(
    ["web", "browser-automation", "--timeout-ms", String(BROWSER_TIMEOUT_MS)],
    source,
  );
  if (payload && payload.ok === true) return payload.result;
  if (Array.isArray(payload?.detection?.markers) && payload.detection.markers.length > 0) {
    return { url: SEARCH_URL, blocked: true, results_page: false, entries: [] };
  }
  return null;
}

async function main() {
  const input = readInput();
  const company = String(input.company || "").trim();
  if (!company) {
    process.stdout.write(JSON.stringify({
      records: [],
      failure_mode: "portal_drift",
      detail: "CTOX_SCRAPE_INPUT_JSON.company missing",
    }));
    return;
  }
  const browserResult = browserSearch(company);
  const output = classifyBrowserResult(company, browserResult);
  if (output.failure_mode === "blocked") {
    recordUnlockSignal(browserResult?.url, ["access_challenge"]);
  }
  process.stdout.write(JSON.stringify(output));
}

if (require.main === module) {
  main().catch((err) => {
    process.stdout.write(JSON.stringify({
      records: [],
      failure_mode: "temporary_unreachable",
      detail: `bundesanzeiger.de browser flow failed: ${err.message}`,
    }));
  });
}

module.exports = {
  allowedSourceUrl,
  buildRecords,
  classifyBrowserResult,
  entryMatchesCompany,
  normalizeCompanyName,
  selectMatchingEntry,
};
