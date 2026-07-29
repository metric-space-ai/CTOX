// northdata.de — prospect.v1 extractor (Phase B, hardened 2026-07-29).
//
// Reads CTOX_SCRAPE_INPUT_JSON for the company + country, drives the
// CTOX web stack (`ctox web read` + `ctox web browser-automation`) to
// load a profile page, then parses the page HTML for the field set
// documented in `tools/web-stack/src/sources/EXCEL_MATRIX.md`.
//
// Hardening vs. the initial revision (live-verified with
// `scrape-targets/northdata.de/solo/probe.mjs` on 2026-07-29 against
// "BNT Chemicals GmbH" and "AKEMI chemisch technische Spezialfabrik
// GmbH"): navigation timeouts raised to 45 s, one in-browser retry on
// goto failure, and one outer retry that fires ONLY when no page
// loaded at all (the 23.07 acceptance flakiness was
// `temporary_unreachable`, i.e. failed loads, not drift).
//
// Drift contract: if the selectors below stop matching but a profile
// page loads successfully, this script returns an empty records array
// and does NOT retry. `ctox scrape execute --allow-heal` then
// classifies the run as `portal_drift` and enqueues a
// `universal-scraping` repair task that will revise this very file.

"use strict";

const { execFileSync } = require("child_process");

const SOURCE_ID = "northdata.de";
const ALLOWED_HOST = "northdata.de";
const MAX_HITS = 2;
// Transient "temporarily unreachable" loads get one second chance; a
// successfully loaded page with drifting selectors does NOT (drift
// contract — empty records, portal_drift, heal task).
const MAX_LOAD_ATTEMPTS = 2;
const NAVIGATION_TIMEOUT_MS = 45_000;

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

function runCtox(args, input, timeout = 35_000) {
  try {
    const out = execFileSync(ctoxBin(), args, {
      encoding: "utf8",
      input,
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
      maxBuffer: 32 * 1024 * 1024,
      timeout,
    });
    return JSON.parse(out);
  } catch (err) {
    // Stay silent on per-hit failures: `classify_outcome` in
    // src/capabilities/scrape.rs runs a substring search for "temporary",
    // "timeout", "429", … on stderr and would misclassify the whole run
    // as temporary_unreachable if one Northdata page returned 429 while
    // others succeeded. Fatal-only stderr stays in main().
    return null;
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

const LEGAL_TOKENS = new Set(["ag", "gmbh", "kg", "mbh", "se", "und"]);

function identityTokens(company) {
  return normalized(company).split(/\s+/).filter((token) => token.length >= 3 && !LEGAL_TOKENS.has(token));
}

function identityMatches(company, corpus) {
  const tokens = identityTokens(company);
  const haystack = normalized(corpus);
  if (tokens.length === 0 || !haystack) return false;
  return tokens.filter((token) => haystack.includes(token)).length >= Math.max(1, Math.ceil(tokens.length * 0.75));
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
    return url.protocol === "https:" && url.hostname.toLowerCase().replace(/^www\./, "") === ALLOWED_HOST;
  } catch (_err) {
    return false;
  }
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
      if (isAllowedUrl(hit?.url)) hits.push(hit.url);
    }
    if (hits.length > 0) break;
  }
  const unique = [...new Set(hits)];
  const exactRoutes = unique.filter((url) => requestedPathMatches(company, url));
  return [...exactRoutes, ...unique.filter((url) => !exactRoutes.includes(url))].slice(0, MAX_HITS);
}

function candidateUrls(input, company, country) {
  const explicit = [input.url, input.source_url, input.profile_url].filter(isAllowedUrl);
  if (explicit.length > 0) return [...new Set(explicit)];
  const portalSearchUrl = `https://www.northdata.de/${encodeURIComponent(company).replace(/%20/g, "+")}`;
  return [portalSearchUrl];
}

function readPage(url, country) {
  const args = ["web", "read", "--url", url];
  if (country) {
    args.push("--country", country);
  }
  return runCtox(args, undefined, NAVIGATION_TIMEOUT_MS + 5_000);
}

function browserSessionId(value) {
  const candidate = String(value || "").trim();
  return /^[A-Za-z0-9._:-]{1,180}$/.test(candidate) ? candidate : null;
}

function browserAutomationArgs(timeoutMs, sessionId = null) {
  const args = ["web", "browser-automation", "--timeout-ms", String(timeoutMs)];
  const safeSessionId = browserSessionId(sessionId);
  if (safeSessionId) args.push("--session-id", safeSessionId);
  return args;
}

function northdataBrowserSource(url, company) {
  if (!isAllowedUrl(url)) return null;
  return `
    const targetUrl = ${JSON.stringify(url)};
    const expectedCompany = ${JSON.stringify(company)};
    const installPageHelpers = async () => page.evaluate((companyName) => {
      const normalize = (value) => String(value || "").normalize("NFKD")
        .replace(/[\\u0300-\\u036f]/g, "").toLowerCase().replace(/ß/g, "ss")
        .replace(/[^a-z0-9]+/g, " ").trim();
      const legalTokens = new Set(["ag", "gmbh", "kg", "mbh", "se", "und"]);
      const identityTokens = (value) => normalize(value).split(/\\s+/)
        .filter((token) => token.length >= 3 && !legalTokens.has(token));
      const identityMatches = (value) => {
        const tokens = identityTokens(companyName);
        const corpus = normalize(value);
        return tokens.length > 0 && corpus.length > 0
          && tokens.filter((token) => corpus.includes(token)).length >= Math.max(1, Math.ceil(tokens.length * 0.75));
      };
      const legalForm = (value) => {
        const tokens = new Set(normalize(value).split(/\\s+/));
        if (tokens.has("gmbh") && tokens.has("kg")) return "gmbh-kg";
        return ["kgaa", "gmbh", "sarl", "srl", "se", "ag", "kg", "og", "sa"]
          .find((form) => tokens.has(form)) || null;
      };
      const legalFormMatches = (value) => {
        const expected = legalForm(companyName);
        return expected === null || legalForm(value) === expected;
      };
      const canonicalProfileRoute = (value) => {
        try {
          const candidate = new URL(value, location.href);
          const segments = candidate.pathname.split("/").filter(Boolean).map(decodeURIComponent);
          return candidate.protocol === "https:"
            && candidate.hostname.toLowerCase().replace(/^www\\./, "") === "northdata.de"
            && segments.length >= 2
            && identityMatches(segments[0])
            && legalFormMatches(segments[0]);
        } catch (_err) {
          return false;
        }
      };
      const nextElement = (start) => {
        if (start?.firstElementChild) return start.firstElementChild;
        let node = start;
        while (node) {
          if (node.nextElementSibling) return node.nextElementSibling;
          node = node.parentElement;
        }
        return null;
      };
      const normalizedLabel = (value) => normalize(value).replace(/\\s+/g, " ");
      const ribbonValue = (label) => {
        const expectedLabel = normalizedLabel(label);
        const headings = Array.from(document.querySelectorAll("h3.ribbon, h3[class*='ribbon'], dt, [data-label]"));
        const heading = headings.find((node) => normalizedLabel(
          node.getAttribute("data-label") || node.textContent || "",
        ) === expectedLabel);
        if (!heading) return null;

        const controlledId = heading.getAttribute("aria-controls");
        if (controlledId) {
          const controlled = document.getElementById(controlledId);
          const controlledValue = controlled?.querySelector(".content, dd, li, [data-value]")?.textContent
            || controlled?.textContent;
          if (controlledValue?.trim()) return controlledValue.replace(/\\s+/g, " ").trim();
        }

        let node = nextElement(heading);
        for (let inspected = 0; node && inspected < 80; inspected += 1) {
          if (node.matches("h3.ribbon, h3[class*='ribbon'], dt, [data-label]")) break;
          let valueNode = null;
          if (node.matches(".content, dd, [data-value]")) valueNode = node;
          if (node.matches(".general-information, li")) {
            valueNode = node.querySelector(".content, dd, [data-value]") || node;
          }
          const value = valueNode?.getAttribute?.("data-value") || valueNode?.textContent;
          if (value?.trim()) return value.replace(/\\s+/g, " ").trim();
          node = nextElement(node);
        }
        return null;
      };
      const heading = () => document.querySelector("h1.qualified")?.textContent
        ?.replace(/\\s+/g, " ").trim() || null;
      const exactProfileLink = () => Array.from(document.querySelectorAll("a[href]"))
        .map((anchor) => anchor.href)
        .find((href) => canonicalProfileRoute(href)) || null;
      const canonicalProfileUrl = () => {
        const declared = document.querySelector('link[rel~="canonical"]')?.href
          || document.querySelector('meta[property="og:url"]')?.content
          || null;
        if (declared && canonicalProfileRoute(declared)) return declared;
        if (canonicalProfileRoute(location.href)) return location.href;
        return exactProfileLink();
      };
      const snapshot = () => {
        const profileHeading = heading();
        const ribbonName = ribbonValue("Name");
        const name = ribbonName || profileHeading?.split(",")[0]?.trim() || null;
        const canonicalUrl = canonicalProfileUrl();
        const canonicalRoute = Boolean(canonicalUrl);
        return {
          url: location.href,
          canonical_url: canonicalUrl,
          title: document.title,
          body_text: (document.body?.innerText || "").slice(0, 140000),
          html: document.documentElement.outerHTML.slice(0, 300000),
          profile: {
            heading: profileHeading,
            name,
            address: ribbonValue("Adresse") || ribbonValue("Anschrift"),
            canonical_route: canonicalRoute,
            identity_matches: canonicalRoute
              && identityMatches(name || profileHeading)
              && legalFormMatches(name || profileHeading),
          },
        };
      };
      globalThis.__ctoxNorthdata = {
        canonicalProfileRoute,
        canonicalProfileUrl,
        profileMarkerReady: () => Boolean(heading() || ribbonValue("Name")),
        snapshot,
      };
    }, expectedCompany);

    // One in-browser retry per navigation: the 23.07 acceptance failure was
    // flaky first loads ("temporarily unreachable"), not selector drift.
    const gotoWithRetry = async (url) => {
      try {
        return await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
      } catch (_firstError) {
        await page.waitForTimeout(2000);
        return await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
      }
    };
    await gotoWithRetry(targetUrl);
    await installPageHelpers();
    await page.waitForFunction(() => {
      const helper = globalThis.__ctoxNorthdata;
      return Boolean(helper?.canonicalProfileUrl());
    }, null, { timeout: 12000 }).catch(() => null);
    const resolvedProfileUrl = await page.evaluate(() => {
      const helper = globalThis.__ctoxNorthdata;
      return helper?.canonicalProfileUrl() || null;
    });
    if (resolvedProfileUrl && resolvedProfileUrl !== page.url()) {
      await gotoWithRetry(resolvedProfileUrl);
      await installPageHelpers();
    }
    await page.waitForFunction(
      () => Boolean(globalThis.__ctoxNorthdata?.profileMarkerReady()),
      null,
      { timeout: 15000 },
    ).catch(() => null);
    return await page.evaluate(() => globalThis.__ctoxNorthdata?.snapshot() || {
      url: location.href,
      title: document.title,
      body_text: (document.body?.innerText || "").slice(0, 140000),
      html: document.documentElement.outerHTML.slice(0, 300000),
      profile: null,
    });
  `;
}

function browserPage(url, company, sessionId = null) {
  const source = northdataBrowserSource(url, company);
  // Budget covers the in-browser goto retry: up to two 45 s navigations
  // plus the marker waits.
  const payload = runCtox(browserAutomationArgs(150_000, sessionId), source, 160_000);
  if (!payload) return null;
  return { ...(payload.result || {}), ok: payload.ok === true, detection: payload.detection };
}

function recordUnlockSignal(url, markers) {
  return runCtox([
    "web", "unlock", "signals", "record",
    "--source", "scrape-target:northdata.de",
    "--url", isAllowedUrl(url) ? url : "https://www.northdata.de/",
    "--evidence", JSON.stringify({
      source_id: "northdata.de",
      detection: "access_challenge",
      markers: [...new Set((markers || []).map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ]);
}

function isBlockedPage(page) {
  const markers = Array.isArray(page?.detection?.markers) ? page.detection.markers.join(" ") : "";
  const corpus = normalized([
    page?.title, page?.body_text, page?.page_text_excerpt, page?.raw_html_excerpt,
    page?.raw_html, page?.html, markers,
  ].filter(Boolean).join(" "));
  return /captcha|cloudflare|challenge|turnstile|verify you are human|access denied|request blocked|too many requests|wurden gesperrt|sicherheitsuberprufung/.test(corpus);
}

function hasBlockedDetection(page) {
  const markers = Array.isArray(page?.detection?.markers) ? page.detection.markers.join(" ") : "";
  return /captcha|cloudflare|challenge|turnstile|access[_ -]?denied|request[_ -]?blocked|rate[_ -]?limit/i.test(markers);
}

function htmlToText(value) {
  return String(value || "")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;|&#160;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function requestedPathMatches(company, value) {
  try {
    const url = new URL(value);
    const segments = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    if (!isAllowedUrl(value) || segments.length < 2) return false;
    const firstSegment = segments[0];
    return identityMatches(company, firstSegment) && legalFormMatches(company, firstSegment);
  } catch (_err) {
    return false;
  }
}

function tagAttribute(tag, name) {
  const escaped = name.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
  const match = String(tag || "").match(new RegExp(`\\b${escaped}\\s*=\\s*(["'])(.*?)\\1`, "i"));
  return match?.[2] || null;
}

function declaredCanonicalUrls(page) {
  const candidates = [
    page?.canonical_url,
    page?.profile?.canonical_url,
  ];
  const html = String(page?.raw_html_excerpt || page?.raw_html || page?.html || "");
  for (const match of html.matchAll(/<link\b[^>]*>/gi)) {
    const rel = normalized(tagAttribute(match[0], "rel"));
    if (rel.split(/\s+/).includes("canonical")) candidates.push(tagAttribute(match[0], "href"));
  }
  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    if (normalized(tagAttribute(match[0], "property")) === "og url") {
      candidates.push(tagAttribute(match[0], "content"));
    }
  }
  return [...new Set(candidates.filter(isAllowedUrl))];
}

function verifiedProfileUrl(company, page) {
  const declared = declaredCanonicalUrls(page)
    .find((url) => requestedPathMatches(company, url));
  if (declared) return new URL(declared).href;
  return requestedPathMatches(company, page?.url) ? new URL(page.url).href : null;
}

function publishedIdentityName(company, country, page) {
  if (!verifiedProfileUrl(company, page)) return null;
  const corpus = htmlToText([
    page?.body_text, page?.page_text_excerpt, page?.raw_html_excerpt,
    page?.raw_html, page?.html,
  ].filter(Boolean).join(" "));
  if (!corpus) return null;

  const companyPattern = String(company || "").trim().split(/\s+/)
    .map((part) => part.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&"))
    .join("\\s+");
  if (!companyPattern) return null;
  const countryNames = { DE: "Deutschland", AT: "Österreich", CH: "Schweiz" };
  const countryPattern = countryNames[String(country || "").toUpperCase()]
    || "(?:Deutschland|Österreich|Schweiz)";
  const match = corpus.match(new RegExp(`\\bals\\s+(${companyPattern})\\s+in\\s+${countryPattern}\\b`, "iu"));
  return match ? match[1].replace(/\s+/g, " ").trim() : null;
}

function pageMatchesCompany(company, page, country = "") {
  const title = String(page?.title || "").replace(/\s+/g, " ").trim();
  if (/\b(?:log[ -]?in|sign[ -]?in|anmeld(?:en|ung)|authentication|authentifizierung|kundenportal|customer portal)\b/i.test(title)
      || /^(?:portal|startseite|home|willkommen)(?:\s*[-|:]\s*.*)?$/i.test(title)) {
    return false;
  }
  if (/^suche nach\b/i.test(title) || /^search for\b/i.test(title) || isBlockedPage(page)) return false;
  const finalUrl = page?.url;
  if (!isAllowedUrl(finalUrl)) return false;
  const corpus = [page?.title, page?.summary, page?.body_text, page?.page_text_excerpt,
    page?.raw_html_excerpt, page?.raw_html, page?.html, page?.profile?.heading, page?.profile?.name,
    page?.profile?.address].filter(Boolean).join(" ");
  const sourceHtml = page?.raw_html_excerpt || page?.raw_html || page?.html || "";
  const profileIdentity = page?.profile?.name || page?.profile?.heading || parseHeading(sourceHtml) || title;
  const exactProfile = Boolean(verifiedProfileUrl(company, page))
    && identityMatches(company, profileIdentity)
    && legalFormMatches(company, profileIdentity)
    && identityMatches(company, corpus);
  return exactProfile || publishedIdentityName(company, country, page) !== null;
}

function recordsFromBrowserProfile(page, sourceUrl = page?.url) {
  const records = [];
  const push = (field, value, confidence, note) => {
    const clean = String(value || "").replace(/\s+/g, " ").trim();
    if (clean) records.push({ field, value: clean, confidence, source_url: sourceUrl, note });
  };
  push("firma_name", page?.profile?.name, "high", "Northdata profile: Name");
  if (page?.profile?.address) {
    const address = parseAddressLine(page.profile.address);
    push("firma_anschrift", address.street, "high", "Northdata profile: Adresse");
    push("firma_plz", address.plz, "high", "Northdata profile: Adresse");
    push("firma_ort", address.ort, "high", "Northdata profile: Adresse");
  }
  return records;
}

const PROVIDER_FIELD_KEYS = new Set([
  "firma_name",
  "firma_anschrift",
  "firma_plz",
  "firma_ort",
  "person_position",
  "person_vorname",
  "person_nachname",
]);

function recordsFromProviderFields(page, sourceUrl = null) {
  if (page?.extracted_fields?.source_id !== SOURCE_ID) return [];
  const pageUrl = String(sourceUrl || page?.canonical_url || page?.final_url || page?.url || "");
  if (!isAllowedUrl(pageUrl)) return [];
  const allowedEvidenceUrls = new Set([
    pageUrl,
    page?.url,
    page?.canonical_url,
    page?.final_url,
  ].filter(isAllowedUrl).map((url) => new URL(url).href));
  const records = [];
  for (const record of page.extracted_fields.fields || []) {
    const sourceUrl = String(record?.source_url || "");
    const value = String(record?.value || "").replace(/\s+/g, " ").trim();
    if (!PROVIDER_FIELD_KEYS.has(record?.field) || !value || !isAllowedUrl(sourceUrl)) continue;
    if (!allowedEvidenceUrls.has(new URL(sourceUrl).href)) continue;
    records.push({
      field: record.field,
      value,
      confidence: ["low", "medium", "high", "user_provided"].includes(record.confidence)
        ? record.confidence
        : "medium",
      source_url: pageUrl,
      note: String(record.note || "Northdata provider extraction").replace(/\s+/g, " ").trim(),
    });
  }
  return records;
}

function recordsForPage(page, company, country) {
  const sourceUrl = verifiedProfileUrl(company, page);
  if (!sourceUrl) return [];
  const sourceHtml = page?.raw_html_excerpt || page?.raw_html || page?.html || "";
  const candidates = [
    sourceHtml ? extractRecords(sourceUrl, sourceHtml) : [],
    recordsFromProviderFields(page, sourceUrl),
    page?.profile?.name ? recordsFromBrowserProfile(page, sourceUrl) : [],
  ];
  const profileRecords = candidates.find((records) => records.some((record) =>
    record.field === "firma_name"
      && identityMatches(company, record.value)
      && legalFormMatches(company, record.value)
  ));
  if (profileRecords) return profileRecords;

  const publishedName = publishedIdentityName(company, country, page);
  if (!publishedName) return [];
  return [{
    field: "firma_name",
    value: publishedName,
    confidence: "high",
    source_url: sourceUrl,
    note: `Northdata publication: exact company identity in ${country || "DACH"}`,
  }];
}

// ---------------------------------------------------------------------------
// Parsing — mirrors src/sources/northdata.rs extract_from_html. Regex is
// intentionally permissive; the unit tests on the Rust side gate the
// selector logic. JS-side drift fixes happen by revising this file.
// ---------------------------------------------------------------------------

function parseGeneralInfoItem(html, label) {
  // Northdata renders ribbon sections as
  //   <h3 class="... ribbon ... label">Adresse</h3>
  //   <div class="general-information"><ul><li><div class="content">Grenzacherstrasse 124, 4058 Basel</div></li></ul>
  const ribbons = [];
  const headingRe = /<h3\b([^>]*)>([\s\S]*?)<\/h3>/gi;
  for (const match of html.matchAll(headingRe)) {
    const classValue = match[1].match(/\bclass\s*=\s*(["'])(.*?)\1/i)?.[2] || "";
    if (!classValue.split(/\s+/).some((token) => token.toLowerCase() === "ribbon")) continue;
    ribbons.push({ index: match.index, end: match.index + match[0].length, label: htmlToText(match[2]) });
  }
  const ribbonIndex = ribbons.findIndex((ribbon) => normalized(ribbon.label) === normalized(label));
  if (ribbonIndex < 0) return null;

  const ribbon = ribbons[ribbonIndex];
  const blockEnd = ribbons[ribbonIndex + 1]?.index ?? html.length;
  const block = html.slice(ribbon.end, blockEnd);
  const openingTagRe = /<([a-z0-9:-]+)\b([^>]*)>/gi;
  for (const match of block.matchAll(openingTagRe)) {
    const classValue = match[2].match(/\bclass\s*=\s*(["'])(.*?)\1/i)?.[2] || "";
    if (!classValue.split(/\s+/).some((token) => token.toLowerCase() === "content")) continue;
    const contentStart = match.index + match[0].length;
    const closingTag = new RegExp(`<\\/${match[1]}\\s*>`, "i");
    const close = closingTag.exec(block.slice(contentStart));
    if (!close) continue;
    const value = htmlToText(block.slice(contentStart, contentStart + close.index));
    if (value) return value;
  }

  const fallback = block.match(/<(?:dd|li)\b[^>]*>([\s\S]*?)<\/(?:dd|li)>/i);
  return fallback ? htmlToText(fallback[1]) : null;
}

function parseHeading(html) {
  const headingRe = /<h1\b([^>]*)>([\s\S]*?)<\/h1>/gi;
  for (const match of html.matchAll(headingRe)) {
    const classValue = match[1].match(/\bclass\s*=\s*(["'])(.*?)\1/i)?.[2] || "";
    if (!classValue.split(/\s+/).some((token) => token.toLowerCase() === "qualified")) continue;
    const value = htmlToText(match[2]);
    if (value) return value;
  }
  return null;
}

function parseAddressLine(line) {
  // "Grenzacherstrasse 124, 4058 Basel" → {street, plz, ort}
  const parts = line.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length < 2) return { street: line, plz: null, ort: null };
  const street = parts[0];
  const last = parts[parts.length - 1];
  const plzMatch = last.match(/\b(\d{4,5})\b\s*(.*)/);
  if (plzMatch) {
    return { street, plz: plzMatch[1], ort: plzMatch[2].trim() || null };
  }
  return { street, plz: null, ort: last };
}

function parseBizqPersons(html) {
  // <figure class="bizq" data-data='[{...}]'> with persons.
  const figures = [
    ...html.matchAll(/<figure[^>]*class=\"[^\"]*bizq[^\"]*\"[^>]*data-data=\"([^\"]+)\"/gi),
  ];
  const out = [];
  for (const fig of figures) {
    let dataStr = fig[1]
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, "&")
      .replace(/&#39;/g, "'");
    try {
      const data = JSON.parse(dataStr);
      const items = Array.isArray(data) ? data : data.items || [];
      for (const item of items) {
        if (item && !item.old && typeof item.text === "string") {
          out.push(item.text);
        }
      }
    } catch (err) {
      // Selector drifted; let the empty-records path trigger portal_drift.
    }
  }
  return out;
}

function splitPersonClause(text) {
  // "Vorstand Anna Müller" → {position: "Vorstand", first: "Anna", last: "Müller"}
  const trimmed = text.trim();
  const m = trimmed.match(/^([A-Za-zÄÖÜäöü\-\s\.]+?)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü\-]+)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü\-]+)*)$/);
  if (!m) return null;
  return { position: m[1].trim(), first: m[2].trim(), last: m[3].trim() };
}

function extractRecords(url, html) {
  const records = [];
  const push = (field, value, confidence, note) => {
    const v = (value || "").trim();
    if (!v) return;
    records.push({
      field,
      value: v,
      confidence,
      source_url: url,
      note,
    });
  };

  const name = parseGeneralInfoItem(html, "Name");
  if (name) {
    push("firma_name", name, "high", "ribbon section: Name");
  } else {
    const h1 = parseHeading(html);
    if (h1) {
      const cleaned = h1.split(",")[0].trim();
      push("firma_name", cleaned, "medium", "h1 fallback");
    }
  }

  let addressLine = parseGeneralInfoItem(html, "Adresse");
  if (!addressLine) {
    addressLine = parseGeneralInfoItem(html, "Anschrift");
  }
  if (addressLine) {
    const parsed = parseAddressLine(addressLine);
    if (parsed.street) push("firma_anschrift", parsed.street, "high", "ribbon section: Adresse");
    if (parsed.plz) push("firma_plz", parsed.plz, "high", "ribbon section: Adresse");
    if (parsed.ort) push("firma_ort", parsed.ort, "high", "ribbon section: Adresse");
  }

  for (const clause of parseBizqPersons(html)) {
    const parsed = splitPersonClause(clause);
    if (!parsed) continue;
    push("person_position", parsed.position, "medium", "bizq figure: position");
    push("person_vorname", parsed.first, "medium", "bizq figure: first name");
    push("person_nachname", parsed.last, "medium", "bizq figure: last name");
    // First clause is enough for an aggregated record set.
    break;
  }

  return records;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main() {
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

  let blocked = false;
  let blockedUrl = "";
  for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt += 1) {
    // A page that loaded but yields no identity-verified records is portal
    // drift (or an identity mismatch), never a reason to hammer the origin.
    let loadedAnyPage = false;
    for (const url of candidateUrls(input, company, country)) {
      const direct = readPage(url, country);
      const directBlocked = isBlockedPage(direct) || hasBlockedDetection(direct);
      blocked ||= directBlocked;
      if (directBlocked) blockedUrl ||= url;
      if (direct?.ok) {
        loadedAnyPage = true;
        if (!direct.url) direct.url = url;
      }
      if (pageMatchesCompany(company, direct, country)) {
        const records = recordsForPage(direct, company, country);
        if (records.length > 0) {
          process.stdout.write(JSON.stringify({ records }));
          return;
        }
      }

      const browser = browserPage(url, company, persistentSessionId);
      const browserBlocked = isBlockedPage(browser) || hasBlockedDetection(browser);
      blocked ||= browserBlocked;
      if (browserBlocked) blockedUrl ||= browser?.url || url;
      if (browser?.ok) loadedAnyPage = true;
      if (pageMatchesCompany(company, browser, country)) {
        const records = recordsForPage(browser, company, country);
        if (records.length > 0) {
          process.stdout.write(JSON.stringify({ records }));
          return;
        }
      }
    }
    if (loadedAnyPage || blocked) break;
    // Nothing loaded at all (transient network/origin failure): wait once,
    // then give the candidate URLs a single second chance.
    if (attempt + 1 < MAX_LOAD_ATTEMPTS) {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2_000);
    }
  }

  if (blocked) recordUnlockSignal(blockedUrl, ["access_challenge"]);

  process.stdout.write(JSON.stringify({
    records: [],
    failure_mode: blocked ? "blocked" : "temporary_unreachable",
    detail: blocked
      ? "Northdata challenge recorded by CTOX browser automation for web-unlock"
      : "no origin- and identity-verified Northdata profile data",
  }));
}

if (require.main === module) {
  main();
}

module.exports = {
  browserAutomationArgs,
  browserSessionId,
  candidateUrls,
  extractRecords,
  hasBlockedDetection,
  htmlToText,
  northdataBrowserSource,
  pageMatchesCompany,
  publishedIdentityName,
  recordsFromProviderFields,
  recordsForPage,
  requestedPathMatches,
  verifiedProfileUrl,
};
