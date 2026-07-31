// firmenabc.at - prospect.v1 extractor with browser/unlock fallback.
//
// Page-content acquisition uses `ctox web browser-capture` (full page.html);
// the truncated `ctox web read` path only remains as a compatibility
// fallback for runtimes that do not expose the subcommand yet.

"use strict";

const { execFileSync } = require("child_process");
const { mkdtempSync, mkdirSync, readFileSync, rmSync } = require("fs");
const { tmpdir } = require("os");
const path = require("path");

const SOURCE_ID = "firmenabc.at";
const ALLOWED_HOST = "firmenabc.at";
const MAX_HITS = 6;
const BROWSER_TIMEOUT_MS = 45_000;
const UNLOCK_TIMEOUT_MS = 90_000;

function readInput() {
  try {
    return JSON.parse(process.env.CTOX_SCRAPE_INPUT_JSON || "{}");
  } catch (err) {
    process.stderr.write(`invalid CTOX_SCRAPE_INPUT_JSON: ${err.message}\n`);
    return {};
  }
}

function runCtox(args, input) {
  try {
    const stdout = execFileSync(process.env.CTOX_BIN || "ctox", args, {
      encoding: "utf8",
      input,
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
      maxBuffer: 32 * 1024 * 1024,
      timeout: 125_000,
    });
    return JSON.parse(stdout);
  } catch (err) {
    const detail = String(err?.stderr || err?.message || "");
    const status = detail.match(/status code\s+(\d{3})/i)?.[1];
    return {
      ok: false,
      command_error: detail.slice(0, 4000),
      http_status: status ? Number(status) : null,
    };
  }
}

function normalized(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("de-DE")
    .replace(/ß/g, "ss")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

const LEGAL_TOKENS = new Set([
  "ag", "co", "gmbh", "kg", "mbh", "og", "se", "und", "company",
]);

function identityTokens(company) {
  return normalized(company)
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !LEGAL_TOKENS.has(token));
}

function identityMatches(company, corpus) {
  const tokens = identityTokens(company);
  const haystack = normalized(corpus);
  if (tokens.length === 0 || !haystack) return false;
  const matches = tokens.filter((token) => haystack.includes(token)).length;
  return matches >= Math.max(1, Math.ceil(tokens.length * 0.75));
}

function legalForm(value) {
  const tokens = new Set(normalized(value).split(/\s+/));
  if (tokens.has("gmbh") && tokens.has("kg")) return "gmbh-kg";
  for (const form of ["kgaa", "gmbh", "sarl", "srl", "se", "ag", "kg", "og", "sa"]) {
    if (tokens.has(form)) return form;
  }
  return null;
}

function legalFormMatches(company, title) {
  const expected = legalForm(company);
  return expected === null || legalForm(title) === expected;
}

function isAllowedUrl(value) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase().replace(/^www\./, "");
    return url.protocol === "https:" && host === ALLOWED_HOST;
  } catch (_err) {
    return false;
  }
}

function blockingMarkers(page) {
  const detection = Array.isArray(page?.detection?.markers)
    ? page.detection.markers.map(String)
    : [];
  const corpus = normalized([
    page?.title,
    page?.body_text,
    page?.page_text_excerpt,
    page?.raw_html_excerpt,
    page?.command_error,
    detection.join(" "),
  ].filter(Boolean).join(" "));
  const markers = detection.filter((marker) =>
    /captcha|cloudflare|challenge|human|access.?denied|blocked|rate.?limit|too.?many/i.test(marker)
  );
  if ([401, 403, 429].includes(Number(page?.http_status))) {
    markers.push(`http-${page.http_status}`);
  }
  for (const phrase of [
    "einen moment", "one moment please", "captcha", "cloudflare", "challenge",
    "verify you are human", "access denied", "request blocked", "too many requests",
  ]) {
    if (corpus.includes(phrase)) markers.push(phrase.replace(/\s+/g, "-"));
  }
  return [...new Set(markers)];
}

function isBlockedPage(page) {
  return blockingMarkers(page).length > 0;
}

function isPortalPage(page) {
  const title = normalized(page?.title);
  return /^(login|log in|anmelden|anmeldung|portal|startseite|home|willkommen)( |$)/.test(title);
}

function pageCorpus(page) {
  const fieldValues = page?.extracted_fields?.fields?.map((item) => item?.value) || [];
  return [
    page?.title,
    page?.summary,
    page?.body_text,
    page?.page_text_excerpt,
    page?.raw_html_excerpt,
    page?.raw_html,
    ...fieldValues,
  ].filter(Boolean).join(" ");
}

function validatedPage(company, page, fallbackUrl) {
  if (!page || page.ok === false || isBlockedPage(page) || isPortalPage(page)) return null;
  const finalUrl = page.url || fallbackUrl;
  if (!isAllowedUrl(finalUrl)) return null;
  if (!identityMatches(company, page.title) || !legalFormMatches(company, page.title)) return null;
  if (!identityMatches(company, pageCorpus(page))) return null;
  return { ...page, url: finalUrl };
}

function searchHits(company, country) {
  const variants = [
    ["web", "search", "--query", company, "--source", SOURCE_ID, "--domain", ALLOWED_HOST, "--include-sources"],
    ["web", "search", "--query", `site:${ALLOWED_HOST} ${company}`, "--domain", ALLOWED_HOST, "--include-sources"],
  ];
  const hits = [];
  for (const args of variants) {
    if (country) args.push("--country", country);
    const payload = runCtox(args);
    for (const hit of payload?.results || []) {
      if (isAllowedUrl(hit?.url)
          && identityMatches(company, hit?.title)
          && legalFormMatches(company, hit?.title)) {
        hits.push(hit.url);
      }
    }
    if (hits.length > 0) break;
  }
  return [...new Set(hits)].slice(0, MAX_HITS);
}

function candidateUrls(input, company, country) {
  const explicit = [input.url, input.source_url, input.profile_url].filter(isAllowedUrl);
  if (explicit.length > 0) return [...new Set(explicit)];
  return searchHits(company, country);
}

function readPage(url, country) {
  const args = ["web", "read", "--url", url];
  if (country) args.push("--country", country);
  return runCtox(args);
}

// ---------------------------------------------------------------------------
// Full-page capture (mirrors scrape-targets/northdata.de/scripts/v1.js).
// `ctox web read` truncates the page (page_text null, empty excerpts), which
// drops the address/register fields below the cutoff and ends runs in
// temporary_unreachable. `ctox web browser-capture` writes a full page.html.
// ---------------------------------------------------------------------------

function titleFromHtml(html) {
  const match = String(html || "").match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  return match ? match[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "";
}

function captureBodyText(html) {
  // Preserve block-element line breaks: bodyProfile() parses the profile
  // line-by-line (street line, then "<plz> <ort>").
  const text = String(html || "")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<br\b[^>]*>/gi, "\n")
    .replace(/<\/(?:p|div|li|tr|td|th|dd|dt|h[1-6]|section|article|header|footer|ul|ol|table)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;|&#160;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'");
  return text.split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
}

function captureJsonLd(html) {
  const blocks = [];
  const re = /<script\b[^>]*type\s*=\s*(["'])application\/ld\+json\1[^>]*>([\s\S]*?)<\/script>/gi;
  for (const match of String(html || "").matchAll(re)) {
    if (blocks.length >= 20) break;
    blocks.push(match[2]);
  }
  return blocks;
}

function browserCapturePage(url) {
  if (!isAllowedUrl(url)) return { page: null, commandUnavailable: false };
  const captureRoot = process.env.CTOX_SCRAPE_OUTPUT_DIR || tmpdir();
  mkdirSync(captureRoot, { recursive: true });
  const outDir = mkdtempSync(path.join(captureRoot, "firmenabc-browser-capture-"));
  try {
    const args = [
      "web", "browser-capture",
      "--url", url,
      "--out-dir", outDir,
      "--timeout-ms", String(BROWSER_TIMEOUT_MS),
    ];
    let payload;
    try {
      const out = execFileSync(process.env.CTOX_BIN || "ctox", args, {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
        maxBuffer: 32 * 1024 * 1024,
        timeout: (BROWSER_TIMEOUT_MS * 2) + 20_000,
      });
      payload = JSON.parse(out);
    } catch (err) {
      const detail = String(err?.stderr || err?.stdout || err?.message || "");
      return {
        page: null,
        commandUnavailable: /unsupported|unknown|unrecognized|usage:/i.test(detail),
      };
    }

    const markerMap = payload?.markers && typeof payload.markers === "object"
      ? payload.markers
      : {};
    const markers = Object.entries(markerMap)
      .filter(([, detected]) => detected === true)
      .map(([marker]) => marker);
    const finalUrl = payload?.finalUrl || payload?.targetUrl || url;
    if (!payload?.ok) {
      // Surface the capture's challenge markers so blockingMarkers() classifies
      // a captcha/sorry/enablejs page as blocked, never as an empty success.
      return {
        page: {
          ok: false,
          url: finalUrl,
          title: payload?.title || "",
          capture_markers: markerMap,
          detection: { markers },
        },
        commandUnavailable: false,
      };
    }

    const html = readFileSync(path.join(outDir, "page.html"), "utf8");
    return {
      page: {
        ok: true,
        url: finalUrl,
        final_url: payload?.finalUrl || null,
        title: payload?.title || titleFromHtml(html),
        html,
        raw_html: html,
        body_text: captureBodyText(html),
        json_ld: captureJsonLd(html),
        capture_markers: markerMap,
        detection: { markers },
      },
      commandUnavailable: false,
    };
  } finally {
    // Mandatory: every capture writes a full Chrome profile and a >12 MB
    // netlog; the production tenant's disk must not fill up.
    rmSync(outDir, { recursive: true, force: true });
  }
}

function providerBrowserSource(url, unlockMode = false) {
  if (!isAllowedUrl(url)) return null;
  return `// ctox-browser: timeout_ms=${unlockMode ? UNLOCK_TIMEOUT_MS : BROWSER_TIMEOUT_MS}
const targetUrl = ${JSON.stringify(url)};
const homeUrl = "https://www.firmenabc.at/";
const unlockMode = ${JSON.stringify(unlockMode)};
const challenge = async () => {
  const title = await page.title().catch(() => "");
  const text = await page.locator("body").innerText({ timeout: 3000 }).catch(() => "");
  const html = await page.content().catch(() => "");
  const corpus = (title + " " + text + " " + html.slice(0, 64000)).toLowerCase();
  return /einen moment|one moment please|captcha|cloudflare|cf-chl-|challenge-platform|turnstile|verify (?:that )?you are human|access denied|request blocked|too many requests|zu viele anfragen/.test(corpus);
};
const dismissConsent = async () => {
  const button = page.getByRole("button", { name: /^(alle akzeptieren|akzeptieren|accept all|zustimmen)$/i }).first();
  if (await button.isVisible({ timeout: 1500 }).catch(() => false)) {
    await button.click({ timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(600);
  }
};
const settleChallenge = async () => {
  for (const delay of [3000, 5000]) {
    if (!(await challenge())) return true;
    await page.waitForTimeout(delay);
    if (!(await challenge())) return true;
    await page.reload({ waitUntil: "domcontentloaded", timeout: 12000 }).catch(() => null);
  }
  return !(await challenge());
};
if (unlockMode) {
  await page.goto(homeUrl, { waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => null);
  await page.waitForTimeout(1800);
  await dismissConsent();
  if (globalThis.humanlike?.humanScroll) {
    await globalThis.humanlike.humanScroll(page, 360, { scrollOvershootChance: 0 }).catch(() => {});
  }
}
let response = await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => null);
await page.waitForTimeout(unlockMode ? 3000 : 2500);
await dismissConsent();
if (unlockMode && await challenge()) {
  await settleChallenge();
  response = await page.waitForLoadState("domcontentloaded", { timeout: 5000 })
    .then(() => response).catch(() => response);
}
return await page.evaluate(() => ({
  url: location.href,
  title: document.title,
  body_text: (document.body?.innerText || "").slice(0, 120000),
  json_ld: Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
    .map((node) => node.textContent || "").slice(0, 20),
})).then(async (result) => ({
  ...result,
  http_status: response?.status?.() || null,
  blocked: await challenge(),
  unlock_attempted: unlockMode,
}));
`;
}

function browserSessionId(value) {
  const candidate = String(value || "").trim();
  return /^[A-Za-z0-9._:-]{1,180}$/.test(candidate) ? candidate : null;
}

function browserPage(url, unlockMode = false, sessionId = null) {
  const source = providerBrowserSource(url, unlockMode);
  if (!source) return null;
  const args = [
    "web", "browser-automation", "--timeout-ms",
    String(unlockMode ? UNLOCK_TIMEOUT_MS : BROWSER_TIMEOUT_MS),
  ];
  const safeSessionId = browserSessionId(sessionId);
  if (safeSessionId) args.push("--session-id", safeSessionId);
  const payload = runCtox(args, source);
  if (!payload) return null;
  return {
    ...(payload.result || {}),
    ok: payload.ok === true,
    detection: payload.detection,
    unlock_attempted: unlockMode,
  };
}

function recordUnlockSignal(url, markers) {
  return runCtox([
    "web", "unlock", "signals", "record",
    "--source", `scrape-target:${SOURCE_ID}`,
    "--url", isAllowedUrl(url) ? url : `https://www.${ALLOWED_HOST}/`,
    "--evidence", JSON.stringify({
      source_id: SOURCE_ID,
      detection: "access_challenge",
      markers: [...new Set(markers.map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ]);
}

function failureResult(markers, matchingPageSeen = false) {
  const blocked = markers.length > 0;
  return {
    records: [],
    failure_mode: blocked ? "blocked" : matchingPageSeen ? "portal_drift" : "temporary_unreachable",
    detail: blocked
      ? "FirmenABC access challenge persisted after provider browser unlock retry"
      : matchingPageSeen
        ? "company-matching FirmenABC page did not match current provider selectors"
        : "no origin- and identity-verified FirmenABC profile data",
  };
}

function organizationObjects(page) {
  const values = [];
  for (const raw of page?.json_ld || []) {
    try {
      const parsed = JSON.parse(raw);
      const queue = Array.isArray(parsed) ? parsed : [parsed];
      for (const item of queue) {
        if (item?.["@graph"] && Array.isArray(item["@graph"])) queue.push(...item["@graph"]);
        const type = Array.isArray(item?.["@type"]) ? item["@type"] : [item?.["@type"]];
        if (type.some((value) => /organization|localbusiness/i.test(String(value)))) values.push(item);
      }
    } catch (_err) {
      // Invalid third-party JSON-LD is ignored; no record is synthesized from it.
    }
  }
  return values;
}

function bodyProfile(page) {
  const text = String(page?.body_text || page?.page_text_excerpt || "");
  const lines = text.split(/\r?\n/).map((line) => line.replace(/\s+/g, " ").trim()).filter(Boolean);
  const anchor = lines.findIndex((line) => normalized(line) === "informationen zur firmenstruktur");
  const profileLines = anchor >= 0 ? lines.slice(anchor + 1, anchor + 16) : [];
  const postalIndex = profileLines.findIndex((line) => /^\d{4}\s+\S/.test(line));
  const postal = postalIndex >= 0 ? profileLines[postalIndex].match(/^(\d{4})\s+(.+)$/) : null;
  const titleName = String(page?.title || "").replace(/\s+in\s+[^|]+(?:\|.*)?$/i, "").trim();
  const contact = (prefix) => profileLines.find((line) => line.startsWith(prefix))?.slice(prefix.length).trim();
  return {
    name: titleName,
    street: postalIndex > 0 ? profileLines[postalIndex - 1] : null,
    postalCode: postal?.[1],
    locality: postal?.[2],
    telephone: contact("T:"),
    email: contact("M:"),
    website: contact("W:"),
  };
}

function recordsFromPage(page) {
  const sourceUrl = page.url;
  const records = [];
  const seen = new Set();
  const push = (field, value, confidence, note) => {
    const clean = String(value || "").replace(/\s+/g, " ").trim();
    if (!clean || seen.has(`${field}\u0000${clean}`)) return;
    seen.add(`${field}\u0000${clean}`);
    records.push({ field, value: clean, confidence, source_url: sourceUrl, note });
  };

  for (const item of page?.extracted_fields?.fields || []) {
    if (typeof item?.field === "string" && typeof item?.value === "string") {
      push(item.field, item.value, item.confidence || "medium", item.note || "CTOX Web Read");
    }
  }
  for (const org of organizationObjects(page)) {
    const address = org.address || {};
    push("firma_name", org.legalName || org.name, "high", "FirmenABC JSON-LD");
    push("firma_anschrift", address.streetAddress, "high", "FirmenABC JSON-LD");
    push("firma_plz", address.postalCode, "high", "FirmenABC JSON-LD");
    push("firma_ort", address.addressLocality, "high", "FirmenABC JSON-LD");
    push("firma_telefon", org.telephone, "medium", "FirmenABC JSON-LD");
    push("firma_email", org.email, "medium", "FirmenABC JSON-LD");
    if (org.url) {
      try {
        const domain = new URL(org.url, sourceUrl).hostname.replace(/^www\./, "");
        if (domain !== ALLOWED_HOST) push("firma_domain", domain, "medium", "FirmenABC JSON-LD");
      } catch (_err) {}
    }
  }
  const body = bodyProfile(page);
  push("firma_name", body.name, "high", "FirmenABC company heading");
  push("firma_anschrift", body.street, "high", "FirmenABC company profile");
  push("firma_plz", body.postalCode, "high", "FirmenABC company profile");
  push("firma_ort", body.locality, "high", "FirmenABC company profile");
  push("firma_telefon", body.telephone, "medium", "FirmenABC company profile");
  push("firma_email", body.email, "medium", "FirmenABC company profile");
  if (body.website) {
    try {
      const absolute = /^https?:\/\//i.test(body.website) ? body.website : `https://${body.website}`;
      push("firma_domain", new URL(absolute).hostname.replace(/^www\./, ""), "medium", "FirmenABC company profile");
    } catch (_err) {}
  }
  return records;
}

function main() {
  const input = readInput();
  const company = String(input.company || "").trim();
  const country = String(input.country || "AT").trim() || "AT";
  const persistentSessionId = browserSessionId(input.browser_session_id || input.session_id);
  if (!company) {
    process.stdout.write(JSON.stringify({ records: [], failure_mode: "portal_drift", detail: "company missing" }));
    return;
  }

  const candidates = candidateUrls(input, company, country);
  let blockedUrl = candidates[0] || `https://www.${ALLOWED_HOST}/`;
  const blockedMarkers = [];
  let matchingPageSeen = false;
  let unlockSignalRecorded = false;
  for (const url of candidates) {
    const browser = browserPage(url);
    const browserMarkers = blockingMarkers(browser);
    if (browserMarkers.length > 0) {
      blockedUrl = browser?.url || url;
      blockedMarkers.push(...browserMarkers);
    }
    const validBrowser = validatedPage(company, browser, url);
    matchingPageSeen ||= Boolean(validBrowser);
    const browserRecords = validBrowser ? recordsFromPage(validBrowser) : [];
    if (browserRecords.length > 0) {
      process.stdout.write(JSON.stringify({ records: browserRecords }));
      return;
    }

    const capture = browserCapturePage(url);
    // Compatibility only for runtimes that do not expose browser-capture yet.
    // A capture that ran and failed is never retried through another transport.
    const direct = capture.commandUnavailable ? readPage(url, country) : capture.page;
    const directMarkers = blockingMarkers(direct);
    if (directMarkers.length > 0) {
      blockedUrl = direct?.url || url;
      blockedMarkers.push(...directMarkers);
    }
    const validDirect = validatedPage(company, direct, url);
    matchingPageSeen ||= Boolean(validDirect);
    const directRecords = validDirect ? recordsFromPage(validDirect) : [];
    if (directRecords.length > 0) {
      process.stdout.write(JSON.stringify({ records: directRecords }));
      return;
    }

    if (browserMarkers.length > 0 || directMarkers.length > 0) {
      if (!unlockSignalRecorded) {
        recordUnlockSignal(blockedUrl, [...browserMarkers, ...directMarkers]);
        unlockSignalRecorded = true;
      }
      const unlocked = browserPage(url, true);
      const unlockMarkers = blockingMarkers(unlocked);
      blockedMarkers.push(...unlockMarkers);
      const validUnlocked = validatedPage(company, unlocked, url);
      matchingPageSeen ||= Boolean(validUnlocked);
      const unlockedRecords = validUnlocked ? recordsFromPage(validUnlocked) : [];
      if (unlockedRecords.length > 0) {
        process.stdout.write(JSON.stringify({ records: unlockedRecords }));
        return;
      }
      if (persistentSessionId) {
        const persistent = browserPage(url, true, persistentSessionId);
        const persistentMarkers = blockingMarkers(persistent);
        blockedMarkers.push(...persistentMarkers);
        const validPersistent = validatedPage(company, persistent, url);
        matchingPageSeen ||= Boolean(validPersistent);
        const persistentRecords = validPersistent ? recordsFromPage(validPersistent) : [];
        if (persistentRecords.length > 0) {
          process.stdout.write(JSON.stringify({ records: persistentRecords }));
          return;
        }
      }
    }
  }

  process.stdout.write(JSON.stringify(failureResult(blockedMarkers, matchingPageSeen)));
}

if (require.main === module) main();

module.exports = {
  blockingMarkers,
  browserCapturePage,
  browserSessionId,
  bodyProfile,
  failureResult,
  identityMatches,
  isAllowedUrl,
  legalFormMatches,
  organizationObjects,
  providerBrowserSource,
  recordsFromPage,
  validatedPage,
};
