// moneyhouse.ch - prospect.v1 extractor with browser/unlock fallback.

"use strict";

const { execFileSync } = require("child_process");

const SOURCE_ID = "moneyhouse.ch";
const ALLOWED_HOST = "moneyhouse.ch";
const MAX_HITS = 6;

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
      timeout: 95_000,
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

const LEGAL_TOKENS = new Set(["ag", "gmbh", "kg", "sa", "sarl", "srl", "se", "und"]);

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

function blockingMarkers(page) {
  const detection = Array.isArray(page?.detection?.markers)
    ? page.detection.markers.map(String)
    : [];
  const corpus = normalized([
    page?.title, page?.body_text, page?.page_text_excerpt, page?.raw_html_excerpt,
    page?.html, page?.command_error, detection.join(" "),
  ].filter(Boolean).join(" "));
  const markers = detection.filter((marker) =>
    /captcha|cloudflare|challenge|human|access.?denied|blocked|rate.?limit|too.?many/i.test(marker)
  );
  if ([401, 403, 429].includes(Number(page?.http_status))) {
    markers.push(`http-${page.http_status}`);
  }
  for (const phrase of [
    "captcha", "cloudflare", "challenge", "verify you are human", "access denied",
    "request blocked", "too many requests",
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
  const fields = page?.extracted_fields?.fields?.map((item) => item?.value) || [];
  return [page?.title, page?.summary, page?.body_text, page?.page_text_excerpt,
    page?.raw_html_excerpt, page?.raw_html, page?.html, ...fields].filter(Boolean).join(" ");
}

function validatedPage(company, page, fallbackUrl) {
  if (!page || page.ok === false || isBlockedPage(page) || isPortalPage(page)) return null;
  const finalUrl = page.url || fallbackUrl;
  if (!isAllowedUrl(finalUrl) || !identityMatches(company, page.title)
      || !legalFormMatches(company, page.title)
      || !identityMatches(company, pageCorpus(page))) return null;
  return { ...page, url: finalUrl };
}

function managementUrl(value) {
  try {
    const url = new URL(value);
    if (!isAllowedUrl(url.href)) return null;
    const match = url.pathname.match(/^\/(de|en|fr|it)\/company\/([^/]+)(?:\/[^/]+)?\/?$/i);
    if (!match) return null;
    url.pathname = `/${match[1].toLowerCase()}/company/${match[2]}/management`;
    url.search = "";
    url.hash = "";
    return url.href;
  } catch (_err) {
    return null;
  }
}

function searchHits(company, country) {
  const variants = [
    ["web", "search", "--query", company, "--domain", ALLOWED_HOST, "--include-sources"],
    ["web", "search", "--query", `site:${ALLOWED_HOST} ${company}`, "--domain", ALLOWED_HOST, "--include-sources"],
  ];
  const hits = [];
  for (const args of variants) {
    if (country) args.push("--country", country);
    const payload = runCtox(args);
    for (const hit of payload?.results || []) {
      const url = managementUrl(hit?.url);
      if (url
          && identityMatches(company, hit?.title)
          && legalFormMatches(company, hit?.title)) {
        hits.push(url);
      }
    }
    if (hits.length > 0) break;
  }
  return [...new Set(hits)].slice(0, MAX_HITS);
}

function searchHitsFromHtml(html) {
  const hits = [];
  const pattern = /<a[^>]+href=(?:"|')([^"']*\/company\/[^"'?#]+)(?:"|')[^>]*>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = pattern.exec(html || "")) !== null) {
    const name = match[2]
      .replace(/<[^>]+>/g, " ")
      .replace(/&amp;/gi, "&")
      .replace(/&nbsp;/gi, " ")
      .replace(/&#39;|&apos;/gi, "'")
      .replace(/&quot;/gi, "\"")
      .replace(/\s+/g, " ")
      .trim();
    try {
      hits.push({ name, url: new URL(match[1], `https://www.${ALLOWED_HOST}/`).href });
    } catch (_err) {}
  }
  return hits;
}

function directPortalSearchResult(direct, searchUrl) {
  if (!direct?.ok) return null;
  const hits = searchHitsFromHtml(direct.raw_html || direct.raw_html_excerpt || "");
  if (hits.length === 0) return null;
  return {
    ...direct,
    url: direct.url || direct.final_url || searchUrl,
    hits,
  };
}

function portalSearch(company, country = "CH") {
  const searchUrl = `https://www.${ALLOWED_HOST}/de/search?q=${encodeURIComponent(company)}`;
  const direct = readPage(searchUrl, country);
  const directResult = directPortalSearchResult(direct, searchUrl);
  if (directResult) return directResult;
  const source = `
    const response = await page.goto(${JSON.stringify(searchUrl)}, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.locator('a[href*="/company/"]').first()
      .waitFor({ state: "attached", timeout: 12000 }).catch(() => null);
    return await page.evaluate(() => ({
      url: location.href,
      title: document.title,
      body_text: (document.body?.innerText || "").slice(0, 12000),
      hits: Array.from(document.querySelectorAll('a[href*="/company/"]'))
        .map((node) => ({ name: (node.textContent || "").trim(), url: node.href }))
        .filter((hit) => hit.name && hit.url)
        .slice(0, 40),
    })).then((result) => ({ ...result, http_status: response?.status() || null }));
  `;
  const payload = runCtox(
    ["web", "browser-automation", "--timeout-ms", "90000"],
    source,
  );
  if (!payload) return null;
  return { ...(payload.result || {}), ok: payload.ok === true, detection: payload.detection };
}

function candidateUrls(input, company, country) {
  const explicit = [input.url, input.source_url, input.profile_url]
    .map(managementUrl)
    .filter(Boolean);
  if (explicit.length > 0) return { urls: [...new Set(explicit)], discovery: null };

  const discovery = portalSearch(company, country);
  const portalHits = (discovery?.hits || [])
    .filter((hit) => identityMatches(company, hit.name) && legalFormMatches(company, hit.name))
    .map((hit) => managementUrl(hit.url))
    .filter(Boolean);
  const fallbackHits = portalHits.length > 0 ? [] : searchHits(company, country);
  const urls = [...new Set([...portalHits, ...fallbackHits])].slice(0, MAX_HITS);
  return { urls, discovery };
}

function readPage(url, country) {
  const args = ["web", "read", "--url", url];
  if (country) args.push("--country", country);
  return runCtox(args);
}

function browserPage(url) {
  const source = `
    const response = await page.goto(${JSON.stringify(url)}, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForTimeout(1800);
    return await page.evaluate(() => ({
      url: location.href,
      title: document.title,
      body_text: (document.body?.innerText || "").slice(0, 80000),
      html: Array.from(document.querySelectorAll('td.entity-name'))
        .map((cell) => cell.parentElement?.outerHTML || "").join("\\n").slice(0, 250000),
      management: Array.from(document.querySelectorAll("table tbody tr")).map((row) => {
        const person = row.querySelector("td.entity-name a.name-link");
        const roles = Array.from(row.querySelectorAll("td.entity-relation-sticky .role, td.entity-relation .role"))
          .map((node) => (node.textContent || "").trim()).filter(Boolean);
        return person && roles.length > 0
          ? { name: (person.textContent || "").trim(), url: person.href, roles }
          : null;
      }).filter(Boolean),
      profile: (() => {
        for (const node of document.querySelectorAll('script[type="application/ld+json"]')) {
          try {
            const parsed = JSON.parse(node.textContent || "null");
            const queue = Array.isArray(parsed) ? [...parsed] : [parsed];
            for (const item of queue) {
              if (Array.isArray(item?.["@graph"])) queue.push(...item["@graph"]);
              const types = Array.isArray(item?.["@type"]) ? item["@type"] : [item?.["@type"]];
              if (types.some((type) => /organization|localbusiness/i.test(String(type)))) {
                return {
                  name: item.legalName || item.name || null,
                  street: item.address?.streetAddress || null,
                  postalCode: item.address?.postalCode || null,
                  locality: item.address?.addressLocality || null,
                };
              }
            }
          } catch (_err) {}
        }
        return null;
      })(),
      json_ld: Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
        .map((node) => node.textContent || "").slice(0, 20),
    })).then((result) => ({ ...result, http_status: response?.status() || null }));
  `;
  const payload = runCtox(["web", "browser-automation", "--timeout-ms", "90000"], source);
  if (!payload) return null;
  return { ...(payload.result || {}), ok: payload.ok === true, detection: payload.detection };
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

function failureResult(markers, matchingPageSeen) {
  const blocked = markers.length > 0;
  return {
    records: [],
    failure_mode: blocked ? "blocked" : matchingPageSeen ? "portal_drift" : "temporary_unreachable",
    detail: blocked
      ? "Moneyhouse challenge recorded by CTOX browser automation for web-unlock"
      : matchingPageSeen
        ? "company-matching Moneyhouse page did not match current provider selectors"
        : "no origin- and identity-verified Moneyhouse profile data",
  };
}

function jsonLdScriptsFromHtml(html) {
  const scripts = [];
  const pattern = /<script[^>]*type=(?:"|')application\/ld\+json(?:"|')[^>]*>([\s\S]*?)<\/script>/gi;
  let match;
  while ((match = pattern.exec(html || "")) !== null) scripts.push(match[1]);
  return scripts;
}

function jsonLdOrganizations(page) {
  const result = [];
  const scripts = [
    ...(page?.json_ld || []),
    ...jsonLdScriptsFromHtml(page?.raw_html || page?.html || ""),
  ];
  for (const raw of scripts) {
    try {
      const parsed = JSON.parse(raw);
      const queue = Array.isArray(parsed) ? [...parsed] : [parsed];
      for (const item of queue) {
        if (Array.isArray(item?.["@graph"])) queue.push(...item["@graph"]);
        const types = Array.isArray(item?.["@type"]) ? item["@type"] : [item?.["@type"]];
        if (types.some((type) => /organization|localbusiness/i.test(String(type)))) result.push(item);
      }
    } catch (_err) {
      // Ignore malformed third-party JSON-LD.
    }
  }
  return result;
}

function managementFromHtml(html) {
  const people = [];
  const rowPattern = /<td[^>]*class=(?:"|')[^"']*entity-name[^"']*(?:"|')[^>]*>([\s\S]*?)<\/td>\s*<td[^>]*class=(?:"|')[^"']*entity-relation[^"']*(?:"|')[^>]*>([\s\S]*?)<\/td>/gi;
  let match;
  while ((match = rowPattern.exec(html || "")) !== null) {
    const href = match[1].match(/<a[^>]+href=(?:"|')([^"']+)(?:"|')/i)?.[1];
    const name = match[1].replace(/<[^>]+>/g, " ").replace(/Exklusiv für registrierte Mitglieder/gi, " ").replace(/\s+/g, " ").trim();
    const role = match[2].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    const parts = name.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      let sourceUrl = null;
      try {
        sourceUrl = href ? new URL(href, `https://www.${ALLOWED_HOST}/`).href : null;
      } catch (_err) {}
      people.push({ first: parts.slice(0, -1).join(" "), last: parts.at(-1), role, sourceUrl });
    }
  }
  return people;
}

function managementPeople(page) {
  const people = [];
  for (const person of page?.management || []) {
    const parts = String(person?.name || "").trim().split(/\s+/).filter(Boolean);
    if (parts.length < 2 || !isAllowedUrl(person?.url)) continue;
    people.push({
      first: parts.slice(0, -1).join(" "),
      last: parts.at(-1),
      role: (person.roles || []).join(" "),
      sourceUrl: person.url,
    });
  }
  return people.length > 0 ? people : managementFromHtml(page?.html || page?.raw_html || "");
}

function recordsFromPage(page) {
  const records = [];
  const seen = new Set();
  const push = (field, value, confidence, note, sourceUrl = page.url) => {
    const clean = String(value || "").replace(/\s+/g, " ").trim();
    const key = `${field}\u0000${clean}\u0000${sourceUrl}`;
    if (!clean || seen.has(key)) return;
    seen.add(key);
    records.push({ field, value: clean, confidence, source_url: sourceUrl, note });
  };

  for (const item of page?.extracted_fields?.fields || []) {
    if (typeof item?.field === "string" && typeof item?.value === "string") {
      push(item.field, item.value, item.confidence || "medium", item.note || "CTOX Web Read");
    }
  }
  for (const org of jsonLdOrganizations(page)) {
    const address = org.address || {};
    push("firma_name", org.legalName || org.name, "high", "Moneyhouse JSON-LD");
    push("firma_anschrift", address.streetAddress, "high", "Moneyhouse JSON-LD");
    push("firma_plz", address.postalCode, "high", "Moneyhouse JSON-LD");
    push("firma_ort", address.addressLocality, "high", "Moneyhouse JSON-LD");
  }
  push("firma_name", page?.profile?.name, "high", "Moneyhouse JSON-LD");
  push("firma_anschrift", page?.profile?.street, "high", "Moneyhouse JSON-LD");
  push("firma_plz", page?.profile?.postalCode, "high", "Moneyhouse JSON-LD");
  push("firma_ort", page?.profile?.locality, "high", "Moneyhouse JSON-LD");
  for (const person of managementPeople(page)) {
    const slug = normalized(`${person.first} ${person.last}`).replace(/\s+/g, "-");
    const personUrl = isAllowedUrl(person.sourceUrl)
      ? person.sourceUrl
      : `${page.url.split("#")[0]}#person-${slug}`;
    push("person_vorname", person.first, "medium", "Moneyhouse management table", personUrl);
    push("person_nachname", person.last, "medium", "Moneyhouse management table", personUrl);
    push("person_position", person.role, "medium", "Moneyhouse management table", personUrl);
  }
  return records;
}

function main() {
  const input = readInput();
  const company = String(input.company || "").trim();
  const country = String(input.country || "CH").trim() || "CH";
  if (!company) {
    process.stdout.write(JSON.stringify({ records: [], failure_mode: "portal_drift", detail: "company missing" }));
    return;
  }

  const candidates = candidateUrls(input, company, country);
  let blockedUrl = candidates.discovery?.url || candidates.urls[0] || `https://www.${ALLOWED_HOST}/`;
  const blockedMarkers = blockingMarkers(candidates.discovery);
  let matchingPageSeen = false;
  for (const url of candidates.urls) {
    const direct = readPage(url, country);
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
  }

  const blocked = blockedMarkers.length > 0;
  if (blocked) recordUnlockSignal(blockedUrl, blockedMarkers);
  process.stdout.write(JSON.stringify(failureResult(blockedMarkers, matchingPageSeen)));
}

if (require.main === module) main();

module.exports = {
  blockingMarkers,
  candidateUrls,
  directPortalSearchResult,
  failureResult,
  identityMatches,
  jsonLdOrganizations,
  managementFromHtml,
  managementPeople,
  managementUrl,
  portalSearch,
  searchHitsFromHtml,
  recordsFromPage,
  validatedPage,
};
