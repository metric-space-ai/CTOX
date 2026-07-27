// handelsregister.de - direct Playwright extractor for prospect.v1.

"use strict";

const { execFileSync } = require("child_process");

const ALLOWED_HOST = "handelsregister.de";
const SEARCH_URL = "https://www.handelsregister.de/rp_web/normalesuche/welcome.xhtml";
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
    "--source", "scrape-target:handelsregister.de",
    "--url", safeUrl?.href || SEARCH_URL,
    "--evidence", JSON.stringify({
      source_id: "handelsregister.de",
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
  const note = entry.registry
    ? `Handelsregister Suchergebnis - ${entry.registry}`
    : "Handelsregister Suchergebnis";
  const records = [{
    field: "firma_name",
    value: entry.name,
    confidence: "high",
    source_url: safeUrl.href,
    note,
  }];
  if (entry.city) {
    records.push({
      field: "firma_ort",
      value: entry.city,
      confidence: "high",
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
      detail: "handelsregister.de browser automation did not return a result",
    };
  }
  if (result.blocked === true) {
    return {
      records: [],
      failure_mode: "blocked",
      detail: "handelsregister.de requires web unlock after a visible access challenge",
    };
  }
  const sourceUrl = allowedSourceUrl(result.url);
  if (!sourceUrl) {
    return {
      records: [],
      failure_mode: "temporary_unreachable",
      detail: "handelsregister.de browser left the allowed origin",
    };
  }
  const entry = selectMatchingEntry(company, result.entries);
  if (entry) return { records: buildRecords(entry, sourceUrl.href) };
  if (result.results_page === true) {
    return {
      records: [],
      failure_mode: Array.isArray(result.entries) && result.entries.length > 0
        ? "temporary_unreachable"
        : "portal_drift",
      detail: Array.isArray(result.entries) && result.entries.length > 0
        ? "handelsregister.de returned no exact company match"
        : "handelsregister.de result page did not match known result selectors",
    };
  }
  return {
    records: [],
    failure_mode: "temporary_unreachable",
    detail: "handelsregister.de did not reach a readable result page",
  };
}

function browserSearch(company) {
  const source = `// ctox-browser: timeout_ms=${BROWSER_TIMEOUT_MS}
const searchUrl = ${JSON.stringify(SEARCH_URL)};
const query = ${JSON.stringify(company)};
await page.goto(searchUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForTimeout(800);

const inspectAccess = async () => page.evaluate(() => {
  const text = document.body ? document.body.innerText : "";
  const challenge = document.querySelector(
    'iframe[src*="recaptcha" i], iframe[src*="captcha" i], iframe[src*="challenge" i], .g-recaptcha, [data-sitekey]',
  );
  return Boolean(challenge) || /captcha|bitte beweisen sie|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert/i.test(text);
});
if (await inspectAccess()) {
  return { url: page.url(), blocked: true, results_page: false, entries: [] };
}

const companyInput = page.locator('[id="form:schlagwoerter"]');
const exactRadioLabel = page.locator('label[for="form:schlagwortOptionen:2"]');
const searchButton = page.locator('[id="form:btnSuche"]');
if (await companyInput.count() !== 1
    || await exactRadioLabel.count() !== 1
    || await searchButton.count() !== 1) {
  return { url: page.url(), blocked: false, results_page: false, entries: [] };
}

await companyInput.fill(query);
await exactRadioLabel.click();
await searchButton.click();
await page.waitForLoadState("domcontentloaded", { timeout: 30000 }).catch(() => {});
await page.waitForSelector(
  '[id="ergebnissForm:selectedSuchErgebnisFormTable_data"]',
  { timeout: 15000 },
).catch(() => {});

return await page.evaluate(() => {
  const text = document.body ? document.body.innerText : "";
  const challenge = document.querySelector(
    'iframe[src*="recaptcha" i], iframe[src*="captcha" i], iframe[src*="challenge" i], .g-recaptcha, [data-sitekey]',
  );
  const blocked = Boolean(challenge) || /captcha|bitte beweisen sie|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert/i.test(text);
  const resultBody = document.querySelector(
    '[id="ergebnissForm:selectedSuchErgebnisFormTable_data"]',
  );
  const tables = resultBody
    ? [...resultBody.querySelectorAll(":scope > tr > td > table.ui-panelgrid")]
    : [];
  const entries = tables.map((table) => {
    const rows = table.querySelectorAll(":scope > tbody > tr");
    const header = rows[0]?.querySelector(".fontTableNameSize");
    const companyRow = rows[1];
    if (!header || !companyRow) return null;
    const name = companyRow.querySelector("td:first-child > .marginLeft20")?.innerText || "";
    const city = companyRow.querySelector(".sitzSuchErgebnisse")?.innerText || "";
    return {
      name: name.replace(/\\s+/g, " ").trim(),
      city: city.replace(/\\s+/g, " ").trim(),
      registry: header.innerText.replace(/\\s+/g, " ").trim(),
    };
  }).filter((entry) => entry && entry.name);
  return {
    url: location.href,
    blocked,
    results_page: Boolean(resultBody) || /Suchergebnis/i.test(document.title),
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
      detail: `handelsregister.de browser flow failed: ${err.message}`,
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
