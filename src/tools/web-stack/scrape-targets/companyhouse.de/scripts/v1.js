// companyhouse.de — prospect.v1 extractor (Phase B initial revision).
//
// Reads CTOX_SCRAPE_INPUT_JSON for the company + country, drives the
// CTOX web stack (`ctox web search` + `ctox web read`) to find a profile
// page, then parses the page HTML for the field set documented in
// `tools/web-stack/src/sources/EXCEL_MATRIX.md`.
//
// Companyhouse exposes two profile types under the bare host root:
//   * Person profiles at `/person/<Vorname>-<Nachname>` — the `<h1>`
//     contains the academic title prefix plus first + last name as one
//     string, e.g. "Dr. Manfred Schneider" or
//     "Prof. Dr.-Ing. Anna Müller". This script peels off the title
//     tokens, splits the remaining name on DE conventions (first name
//     first, last name last, nobility particles attach to surname) and
//     emits `person_titel` / `person_vorname` / `person_nachname` at
//     Confidence::medium — the Excel matrix marks `person_titel` with an
//     asterisk because the title heuristic is regex-based.
//   * Company profiles at `/<Firmenname>-<Ort>` — the `<h1>` is the
//     canonical Handelsregister name. We emit `firma_name` at
//     Confidence::high.
//
// Cloudflare interstitials are detected and skipped per-hit so the
// run is classified `temporary_unreachable` (or `portal_drift` if no
// hit produced any record), not silently failed.
//
// Drift contract: if the selectors below stop matching but a profile
// page loads successfully, this script returns an empty records array.
// `ctox scrape execute --allow-heal` then classifies the run as
// `portal_drift` and enqueues a `universal-scraping` repair task that
// will revise this very file.

"use strict";

const { execFileSync } = require("child_process");

const MAX_HITS = 3;
const ALLOWED_HOST = "companyhouse.de";
const BROWSER_TIMEOUT_MS = 45_000;
const UNLOCK_TIMEOUT_MS = 90_000;

function readInput() {
  const raw = process.env.CTOX_SCRAPE_INPUT_JSON;
  if (!raw) {
    return { company: "", country: "" };
  }
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
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
      input,
      timeout: UNLOCK_TIMEOUT_MS + 35_000,
      maxBuffer: 32 * 1024 * 1024,
    });
    return JSON.parse(out);
  } catch (err) {
    // Stay silent on per-hit failures: `classify_outcome` in
    // src/capabilities/scrape.rs runs a substring search for "temporary",
    // "timeout", "429", … on stderr and would misclassify the whole run
    // as temporary_unreachable if one Companyhouse page returned 429 while
    // others succeeded. Fatal-only stderr stays in main().
    return {
      ok: false,
      ctox_error: String(err?.stderr || err?.message || "").slice(0, 4000),
    };
  }
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

function canonicalProviderUrl(raw) {
  const url = allowedSourceUrl(raw);
  if (!url) return null;
  url.hash = "";
  for (const key of [...url.searchParams.keys()]) {
    if (key.startsWith("__cf_chl_")) url.searchParams.delete(key);
  }
  return url;
}

function providerSearchUrl(company) {
  const query = String(company || "").trim().replace(/\s+/g, "+");
  return query
    ? `https://www.companyhouse.de/s/${encodeURIComponent(query)}`
    : "https://www.companyhouse.de/";
}

function providerBrowserSource(rawUrl, unlockMode = false, company = "") {
  const safeUrl = allowedSourceUrl(rawUrl);
  if (!safeUrl) return null;
  return `// ctox-browser: timeout_ms=${unlockMode ? UNLOCK_TIMEOUT_MS : BROWSER_TIMEOUT_MS}
const targetUrl = ${JSON.stringify(safeUrl.href)};
const homeUrl = "https://www.companyhouse.de/";
const allowedHost = ${JSON.stringify(ALLOWED_HOST)};
const unlockMode = ${JSON.stringify(unlockMode)};
const requestedCompany = ${JSON.stringify(String(company || "").trim())};
const alignBrowserIdentity = async () => {
  const currentUa = await page.evaluate(() => navigator.userAgent);
  let runtimeVersion = "";
  if (browser && typeof browser.version === "function") {
    try { runtimeVersion = String(await browser.version() || ""); } catch {}
  }
  const version = runtimeVersion || currentUa.match(/Chrome\/(\d+(?:\.\d+){0,3})/)?.[1] || "";
  const major = String(version || "").match(/^(\\d+)/)?.[1];
  if (!major) return;
  const userAgent = currentUa.replace(/Chrome\\/\\d+(?:\\.\\d+){0,3}/, "Chrome/" + version);
  const platformName = process.platform === "darwin" ? "macOS"
    : process.platform === "win32" ? "Windows" : "Linux";
  const navigatorPlatform = process.platform === "darwin" ? "MacIntel"
    : process.platform === "win32" ? "Win32" : "Linux x86_64";
  const brands = [
    { brand: "Chromium", version: major },
    { brand: "Google Chrome", version: major },
    { brand: "Not.A/Brand", version: "24" },
  ];
  await context.setExtraHTTPHeaders({
    "Sec-CH-UA": '"Chromium";v="' + major + '", "Google Chrome";v="' + major + '", "Not.A/Brand";v="24"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"' + platformName + '"',
  });
  const client = await context.newCDPSession(page);
  await client.send("Network.setUserAgentOverride", {
    userAgent,
    acceptLanguage: "de-DE,de;q=0.9,en;q=0.8",
    platform: navigatorPlatform,
    userAgentMetadata: {
      brands,
      fullVersionList: brands.map((item) => ({
        brand: item.brand,
        version: item.brand === "Not.A/Brand" ? "24.0.0.0" : version,
      })),
      fullVersion: version,
      platform: platformName,
      platformVersion: "",
      architecture: process.arch === "arm64" ? "arm" : "x86",
      model: "",
      mobile: false,
      bitness: "64",
      wow64: false,
    },
  });
};
const challenge = async () => {
  const title = await page.title().catch(() => "");
  const text = await page.locator("body").innerText({ timeout: 3000 }).catch(() => "");
  const html = await page.content().catch(() => "");
  const corpus = (title + " " + text + " " + html.slice(0, 64000)).toLowerCase();
  return /cloudflare|cf-chl-|cf-mitigated|challenge-platform|turnstile|sicherheits(?:ü|u)berpr(?:ü|u)fung|security verification|noch einen schritt|nur einen moment|just a moment|captcha|verify (?:that )?you are human|nat(?:ü|u)rlichen zugriff|access denied|request blocked|wurden gesperrt|zugriff.{0,40}gesperrt/.test(corpus);
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
const normalizedIdentity = (value) => String(value || "")
  .normalize("NFKD").replace(/\\p{M}/gu, "")
  .toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
const followCompanySearchResult = async () => {
  const path = new URL(page.url()).pathname.toLowerCase();
  if (!path.startsWith("/s/") || !requestedCompany) return false;
  const expected = normalizedIdentity(requestedCompany);
  const candidate = await page.locator("a[href]").evaluateAll((anchors, expectedValue) => {
    const normalize = (value) => String(value || "")
      .normalize("NFKD").replace(/\\p{M}/gu, "")
      .toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    return anchors.map((anchor) => {
      try {
        const url = new URL(anchor.href, document.baseURI);
        const pathName = url.pathname.toLowerCase();
        const host = url.hostname.toLowerCase().replace(/\\.$/, "");
        const text = normalize(anchor.textContent);
        const profilePath = pathName.split("/").filter(Boolean).length === 1;
        return {
          href: url.href,
          exact: text === expectedValue,
          matching: text.includes(expectedValue) || expectedValue.includes(text),
          providerOwned: url.protocol === "https:"
            && (host === "companyhouse.de" || host.endsWith(".companyhouse.de"))
            && profilePath,
        };
      } catch {
        return null;
      }
    }).filter((item) => item?.providerOwned && item.matching)
      .sort((left, right) => Number(right.exact) - Number(left.exact))[0]?.href || null;
  }, expected).catch(() => null);
  if (!candidate) return false;
  await page.goto(candidate, { waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => null);
  await page.waitForTimeout(1800);
  await dismissConsent();
  return true;
};
if (unlockMode) {
  await alignBrowserIdentity();
  await page.goto(homeUrl, { waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => null);
  await page.waitForTimeout(1800);
  await dismissConsent();
  if (globalThis.humanlike?.humanScroll) {
    await globalThis.humanlike.humanScroll(page, 360, { scrollOvershootChance: 0 }).catch(() => {});
  }
}
let response = await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 15000 }).catch(() => null);
await page.waitForTimeout(unlockMode ? 3000 : 5000);
await dismissConsent();
if (unlockMode && await challenge()) {
  await settleChallenge();
  response = await page.waitForLoadState("domcontentloaded", { timeout: 5000 })
    .then(() => response).catch(() => response);
}
if (!(await challenge()) && await followCompanySearchResult()) {
  response = await page.waitForLoadState("domcontentloaded", { timeout: 5000 })
    .then(() => response).catch(() => response);
  if (unlockMode && await challenge()) await settleChallenge();
}
const finalUrl = page.url();
const parsed = new URL(finalUrl);
const host = parsed.hostname.toLowerCase().replace(/\\.$/, "");
const originOk = parsed.protocol === "https:"
  && !parsed.username && !parsed.password
  && (host === allowedHost || host.endsWith("." + allowedHost));
const title = await page.title();
const text = (await page.locator("body").innerText({ timeout: 5000 }).catch(() => "")).slice(0, 160000);
const html = (await page.content()).slice(0, 500000);
const blocked = await challenge();
const providerProfile = await page.evaluate(() => {
  const clean = (value) => String(value || "").replace(/\\s+/g, " ").trim() || null;
  const heading = clean(document.querySelector("h1")?.textContent);
  const locationIcon = document.querySelector('[class*="ch-ico-location"]');
  const address = clean(locationIcon?.closest("div")?.querySelector("p")?.textContent
    || locationIcon?.parentElement?.nextElementSibling?.textContent);
  const detail = (label) => {
    const headers = Array.from(document.querySelectorAll('[class*="tile-table-header"]'));
    const header = headers.find((node) => clean(node.textContent) === label);
    if (!header) return null;
    const row = header.parentElement;
    return clean(row?.querySelector('[class*="mb-3"]')?.textContent
      || header.nextElementSibling?.textContent);
  };
  return {
    heading,
    address,
    telephone: detail("Telefonnummer"),
    email: detail("E-Mail"),
    website: detail("Webseite"),
  };
});
return {
  url: finalUrl,
  title,
  text,
  html,
  origin_ok: originOk,
  blocked,
  http_status: response?.status?.() || null,
  provider_profile: providerProfile,
  unlock_attempted: unlockMode,
};
`;
}

function browserSessionId(value) {
  const candidate = String(value || "").trim();
  return /^[A-Za-z0-9._:-]{1,180}$/.test(candidate) ? candidate : null;
}

function browserRead(rawUrl, unlockMode = false, sessionId = null, company = "") {
  const safeUrl = allowedSourceUrl(rawUrl);
  if (!safeUrl) return null;
  const source = providerBrowserSource(safeUrl.href, unlockMode, company);
  const args = [
    "web", "browser-automation", "--timeout-ms",
    String(unlockMode ? UNLOCK_TIMEOUT_MS : BROWSER_TIMEOUT_MS),
  ];
  const safeSessionId = browserSessionId(sessionId);
  if (safeSessionId) args.push("--session-id", safeSessionId);
  const payload = runCtox(args, source);
  const result = payload && payload.ok === true ? payload.result : null;
  if (!result && Array.isArray(payload?.detection?.markers) && payload.detection.markers.length > 0) {
    return {
      ok: false,
      url: safeUrl.href,
      blocked: true,
      transport: "browser",
      detection: payload.detection,
      unlock_attempted: unlockMode,
    };
  }
  if (!result || result.origin_ok !== true || !allowedSourceUrl(result.url)) return null;
  return {
    ok: true,
    url: result.url,
    title: result.title,
    page_text_excerpt: result.text || result.page_text_excerpt,
    raw_html: result.html || result.raw_html,
    provider_profile: result.provider_profile,
    blocked: result.blocked === true || /(?:^|[-|:]\s*)(?:login|anmeldung|anmelden|portal)\b/i.test(String(result.title || "")),
    transport: "browser",
    detection: payload.detection,
    unlock_attempted: unlockMode,
  };
}

function blockingMarkers(page) {
  const markers = Array.isArray(page?.detection?.markers)
    ? page.detection.markers.map(String).filter((marker) =>
      /captcha|cloudflare|challenge|human|access.?denied|blocked|turnstile/i.test(marker))
    : [];
  const statusMatch = String(page?.ctox_error || "").match(/status code\s+(401|403|429)\b/i);
  if (statusMatch) markers.push(`http-${statusMatch[1]}`);
  if ([401, 403, 429].includes(Number(page?.http_status))) markers.push(`http-${page.http_status}`);
  if (page?.blocked === true || isCloudflareBlock([
    page?.title,
    page?.page_text_excerpt,
    page?.raw_html,
    page?.ctox_error,
  ].filter(Boolean).join(" "))) {
    markers.push("access_challenge");
  }
  return [...new Set(markers)];
}

function searchHits(company, country) {
  const args = [
    "web",
    "search",
    "--query",
    company,
    "--source",
    "companyhouse.de",
    "--include-sources",
  ];
  if (country) {
    args.push("--country", country);
  }
  const payload = runCtox(args);
  if (!payload || !Array.isArray(payload.results)) {
    return [];
  }
  return payload.results
    .map((hit) => ({ ...hit, url: canonicalProviderUrl(hit.url)?.href || "" }))
    .filter((hit) => hit.url)
    .slice(0, MAX_HITS);
}

function candidateHits(input, company, country) {
  const explicit = [input.url, input.source_url, input.profile_url]
    .map(canonicalProviderUrl)
    .filter(Boolean)
    .map((url) => ({ url: url.href }));
  if (explicit.length > 0) return explicit;
  const discovered = searchHits(company, country);
  return discovered.length > 0
    ? discovered
    : [{ url: providerSearchUrl(company) }];
}

function readPage(url, country) {
  const args = ["web", "read", "--url", url];
  if (country) {
    args.push("--country", country);
  }
  return runCtox(args);
}

function recordUnlockSignal(url, markers) {
  const safeUrl = allowedSourceUrl(url);
  return runCtox([
    "web", "unlock", "signals", "record",
    "--source", "scrape-target:companyhouse.de",
    "--url", safeUrl?.href || "https://www.companyhouse.de/",
    "--evidence", JSON.stringify({
      source_id: "companyhouse.de",
      detection: "access_challenge",
      markers: [...new Set((markers || []).map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ]);
}

function pageMatchesCompany(company, page) {
  const title = String(page?.title || "").replace(/\s+/g, " ").trim();
  if (/\b(?:log[ -]?in|sign[ -]?in|anmeld(?:en|ung)|authentication|authentifizierung|kundenportal|customer portal)\b/i.test(title)
      || /^(?:portal|startseite|home|willkommen)(?:\s*[-|:]\s*.*)?$/i.test(title)) {
    return false;
  }
  const legalForms = new Set(["ag", "gmbh", "mbh", "se", "kg", "kgaa", "ohg", "ug", "sa", "sarl"]);
  const tokens = String(company || "").toLocaleLowerCase("de-DE").normalize("NFKD")
    .replace(/\p{M}/gu, "").replace(/[^a-z0-9äöüß]+/gi, " ").split(/\s+/)
    .filter((token) => token.length >= 3 && !legalForms.has(token));
  const corpus = [
    page?.title,
    page?.summary,
    page?.text,
    page?.page_text_excerpt,
    page?.html,
    page?.raw_html_excerpt,
    page?.raw_html,
    ...Object.values(page?.provider_profile || {}),
  ]
    .filter(Boolean).join(" ").toLocaleLowerCase("de-DE").normalize("NFKD").replace(/\p{M}/gu, "");
  return tokens.length > 0 && tokens.every((token) => corpus.includes(token));
}

// ---------------------------------------------------------------------------
// URL classification — mirrors src/sources/companyhouse.rs is_person_url /
// is_company_url. Person profiles live under `/person/`, company profiles
// live at the bare host root (`/<Firmenname>-<Ort>`); known non-profile
// segments (login, agb, suche, …) are skipped so we don't hallucinate a
// firma_name from a search-results page heading.
// ---------------------------------------------------------------------------

const NON_PROFILE_SEGMENTS = [
  "/login",
  "/register",
  "/suche",
  "/search",
  "/impressum",
  "/agb",
  "/datenschutz",
  "/faq",
  "/preise",
  "/kontakt",
  "/s/",
  "/l/",
];

function isPersonUrl(url) {
  const lower = (url || "").toLowerCase();
  return lower.includes("/person/") || lower.includes("/personen/");
}

function isCompanyUrl(url) {
  const lower = (url || "").toLowerCase();
  if (!lower.includes("companyhouse.de")) return false;
  if (isPersonUrl(lower)) return false;
  return !NON_PROFILE_SEGMENTS.some((seg) => lower.includes(seg));
}

// ---------------------------------------------------------------------------
// Cloudflare interstitial heuristic.
//
// Companyhouse fronts the site with Cloudflare and frequently returns either
// the classic "Just a moment…" challenge page or a localized block page
// ("Zugriff … gesperrt"). Both keep the response status at 200, so we can't
// rely on HTTP codes alone — we sniff the body. When matched, the page is
// skipped silently and the executor will classify the whole run as
// `temporary_unreachable` if no other hit produced records.
// ---------------------------------------------------------------------------

function isCloudflareBlock(html) {
  if (!html) return false;
  return /cloudflare|cf-chl-|cf-mitigated|challenge-platform|turnstile|sicherheits(?:ü|u)berpr(?:ü|u)fung|security verification|noch einen schritt|nur einen moment|just a moment|captcha|verify (?:that )?you are human|nat(?:ü|u)rlichen zugriff|access denied|request blocked|wurden gesperrt|zugriff.{0,40}gesperrt/i.test(html);
}

function isBlockedFailure(page) {
  return /\b(?:403|forbidden)\b|access denied|request blocked|zugriff.{0,40}gesperrt|wurden gesperrt/i
    .test(String(page?.ctox_error || ""));
}

// ---------------------------------------------------------------------------
// Parsing — mirrors src/sources/companyhouse.rs extract_from_html. Regex is
// intentionally permissive; the unit tests on the Rust side gate the
// selector logic. JS-side drift fixes happen by revising this file.
// ---------------------------------------------------------------------------

function parseHeading(html) {
  const m = html.match(/<h1[^>]*>([\s\S]*?)<\/h1>/i);
  if (!m) return null;
  return m[1].replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

function htmlText(value) {
  return String(value || "")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;|&#160;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function parseAddressLine(value) {
  const clean = htmlText(value);
  const match = clean.match(/^(.+?),\s*(\d{5})\s+(.+)$/u);
  if (!match) return { street: clean || null, plz: null, city: null };
  return { street: match[1].trim(), plz: match[2], city: match[3].trim() };
}

function parseProfileAddress(html) {
  const match = String(html || "").match(
    /class=["'][^"']*ch-ico-location[^"']*["'][\s\S]{0,600}?<p[^>]*>([\s\S]*?)<\/p>/i,
  );
  return match ? parseAddressLine(match[1]) : null;
}

function parseDetailValue(html, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = String(html || "").match(new RegExp(
    `class=["'][^"']*tile-table-header[^"']*["'][^>]*>\\s*${escaped}\\s*<\\/div>`
      + `[\\s\\S]{0,500}?<div[^>]*class=["'][^"']*mb-3[^"']*["'][^>]*>([\\s\\S]*?)<\\/div>`,
    "iu",
  ));
  return match ? htmlText(match[1]) : null;
}

// Conservative whitelist of common DE academic / professional titles plus
// the generic "ends in '.'" rule for combos like `Dr.-Ing.` or `Dipl.-Kfm.`
// that the whitelist may not cover.
const TITLE_WHITELIST = new Set([
  "Prof",
  "Dr",
  "Mag",
  "Dipl",
  "Ing",
  "MBA",
  "MSc",
  "MA",
  "BA",
  "LL.M",
  "PhD",
  "DDr",
]);

function isTitleToken(token) {
  if (!token) return false;
  if (token.endsWith(".")) return true;
  return TITLE_WHITELIST.has(token);
}

const NOBILITY_PARTICLES = new Set([
  "von",
  "vom",
  "zu",
  "zur",
  "der",
  "den",
  "de",
  "del",
  "di",
  "van",
  "ten",
  "ter",
]);

function isNobilityParticle(token) {
  return NOBILITY_PARTICLES.has((token || "").toLowerCase());
}

// Walk the remaining name tokens from the right to find where the surname
// starts. The surname is the last token, plus any preceding nobility
// particles ("von", "von der", "zu", "de", "van", "van der", "del", "di").
function surnameStart(tokens) {
  if (tokens.length <= 1) return 0;
  let start = tokens.length - 1;
  while (start > 0 && isNobilityParticle(tokens[start - 1])) {
    start -= 1;
  }
  if (start === 0) {
    // Particles consumed everything — fall back to "last token is surname".
    return tokens.length - 1;
  }
  return start;
}

function parsePersonHeading(heading) {
  const tokens = (heading || "").split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return null;

  let idx = 0;
  const titleParts = [];
  while (idx < tokens.length && isTitleToken(tokens[idx])) {
    titleParts.push(tokens[idx]);
    idx += 1;
  }
  const remaining = tokens.slice(idx);
  if (remaining.length === 0) {
    // Heading is *only* titles — defensive, almost never happens.
    return null;
  }

  const result = { title: null, first: null, last: null };
  if (titleParts.length > 0) {
    result.title = titleParts.join(" ");
  }

  if (remaining.length === 1) {
    result.last = remaining[0];
  } else {
    const split = surnameStart(remaining);
    const firstTokens = remaining.slice(0, split);
    const lastTokens = remaining.slice(split);
    if (firstTokens.length > 0) result.first = firstTokens.join(" ");
    if (lastTokens.length > 0) result.last = lastTokens.join(" ");
  }
  return result;
}

function extractRecords(url, html, providerProfile = null) {
  const canonicalUrl = canonicalProviderUrl(url);
  if (!canonicalUrl) return [];
  const records = [];
  const push = (field, value, confidence, note) => {
    const v = (value || "").trim();
    if (!v) return;
    records.push({
      field,
      value: v,
      confidence,
      source_url: canonicalUrl.href,
      note,
    });
  };

  if (isCloudflareBlock(html)) {
    // Skip — the executor will pick this up as temporary_unreachable
    // if no other hit succeeds.
    return records;
  }

  const heading = providerProfile?.heading || parseHeading(html);
  if (!heading) return records;

  if (isPersonUrl(canonicalUrl.href)) {
    const parsed = parsePersonHeading(heading);
    if (parsed) {
      if (parsed.title) {
        push(
          "person_titel",
          parsed.title,
          "medium",
          "companyhouse person <h1> title prefix",
        );
      }
      if (parsed.first) {
        push(
          "person_vorname",
          parsed.first,
          "medium",
          "companyhouse person <h1> first name",
        );
      }
      if (parsed.last) {
        push(
          "person_nachname",
          parsed.last,
          "medium",
          "companyhouse person <h1> last name",
        );
      }
    }
  } else if (isCompanyUrl(canonicalUrl.href)) {
    push("firma_name", heading, "high", "companyhouse company <h1>");
    const address = providerProfile?.address
      ? parseAddressLine(providerProfile.address)
      : parseProfileAddress(html);
    if (address) {
      push("firma_anschrift", address.street, "high", "companyhouse profile address");
      push("firma_plz", address.plz, "high", "companyhouse profile address");
      push("firma_ort", address.city, "high", "companyhouse profile address");
    }
    push("firma_telefon", providerProfile?.telephone || parseDetailValue(html, "Telefonnummer"), "high", "companyhouse Details");
    push("firma_email", providerProfile?.email || parseDetailValue(html, "E-Mail"), "high", "companyhouse Details");
    push("firma_domain", providerProfile?.website || parseDetailValue(html, "Webseite"), "high", "companyhouse Details");
  }
  // Else: neither person nor company path (search hit, status page, …).
  // Empty records → drift loop will classify accordingly.

  return records;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main() {
  const input = readInput();
  const company = (input.company || "").trim();
  const country = (input.country || "").trim();
  const persistentSessionId = browserSessionId(input.browser_session_id || input.session_id);
  if (!company) {
    process.stdout.write(
      JSON.stringify({
        records: [],
        failure_mode: "portal_drift",
        detail: "CTOX_SCRAPE_INPUT_JSON.company missing",
      }),
    );
    return;
  }

  const hits = candidateHits(input, company, country);
  if (hits.length === 0) {
    process.stdout.write(
      JSON.stringify({
        records: [],
        failure_mode: "temporary_unreachable",
        detail: "ctox web search returned no companyhouse.de hits",
      }),
    );
    return;
  }

  const aggregated = [];
  let blockedSeen = false;
  let matchingPageSeen = false;
  let unlockSignalRecorded = false;
  for (const hit of hits) {
    const safeHit = allowedSourceUrl(hit.url);
    if (!safeHit) continue;
    let page = readPage(safeHit.href, country);
    let html = page && page.ok
      ? page.raw_html_excerpt || page.raw_html || page.page_text_excerpt || ""
      : "";
    const hitMarkers = blockingMarkers(page);
    let usablePage = page && page.ok && html && !isCloudflareBlock(html) ? page : null;
    if (!usablePage && hitMarkers.length === 0) {
      const browser = browserRead(safeHit.href, false, null, company);
      hitMarkers.push(...blockingMarkers(browser));
      const browserHtml = browser?.raw_html || browser?.page_text_excerpt || "";
      if (browser && browser.ok && browserHtml && !browser.blocked && !isCloudflareBlock(browserHtml)) {
        page = browser;
        html = browserHtml;
        usablePage = browser;
      }
    }
    if (!usablePage && hitMarkers.length > 0) {
      blockedSeen = true;
      if (!unlockSignalRecorded) {
        recordUnlockSignal(safeHit.href, hitMarkers);
        unlockSignalRecorded = true;
      }
      const unlocked = browserRead(safeHit.href, true, null, company);
      hitMarkers.push(...blockingMarkers(unlocked));
      const unlockedHtml = unlocked?.raw_html || unlocked?.page_text_excerpt || "";
      if (unlocked && unlocked.ok && unlockedHtml && !unlocked.blocked && !isCloudflareBlock(unlockedHtml)) {
        page = unlocked;
        html = unlockedHtml;
        usablePage = unlocked;
      }
      if (!usablePage && persistentSessionId) {
        const persistent = browserRead(safeHit.href, true, persistentSessionId, company);
        hitMarkers.push(...blockingMarkers(persistent));
        const persistentHtml = persistent?.raw_html || persistent?.page_text_excerpt || "";
        if (persistent && persistent.ok && persistentHtml
            && !persistent.blocked && !isCloudflareBlock(persistentHtml)) {
          page = persistent;
          html = persistentHtml;
          usablePage = persistent;
        }
      }
    }
    if (!usablePage) continue;
    const evidenceUrl = canonicalProviderUrl(page.url || safeHit.href);
    if (!evidenceUrl) continue;
    if (!pageMatchesCompany(company, page)) continue;
    matchingPageSeen = true;
    const records = extractRecords(evidenceUrl.href, html, page.provider_profile);
    const names = records.filter((record) => record.field === "firma_name");
    if (names.length > 0 && !names.some((record) => pageMatchesCompany(company, { raw_html: record.value }))) {
      continue;
    }
    aggregated.push(...records);
  }

  if (aggregated.length > 0) {
    process.stdout.write(JSON.stringify({ records: aggregated }));
    return;
  }
  process.stdout.write(JSON.stringify({
    records: [],
    failure_mode: blockedSeen ? "blocked" : matchingPageSeen ? "portal_drift" : "temporary_unreachable",
    detail: blockedSeen
      ? "companyhouse.de access challenge persisted after provider browser unlock retry"
      : matchingPageSeen
        ? "company-matching companyhouse.de evidence did not match known profile selectors"
        : "companyhouse.de returned no readable evidence for the requested company",
  }));
}

if (require.main === module) {
  main();
}

module.exports = {
  candidateHits,
  blockingMarkers,
  browserSessionId,
  canonicalProviderUrl,
  extractRecords,
  isBlockedFailure,
  isCloudflareBlock,
  parseAddressLine,
  parseDetailValue,
  parseProfileAddress,
  pageMatchesCompany,
  providerBrowserSource,
  providerSearchUrl,
};
