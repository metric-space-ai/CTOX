// impressum — prospect.v1 extractor (Phase B, solo-verified 2026-07-31).
//
// First-party source: the company legal notice page (Impressum/Imprint)
// that every German company must publish. The target is INPUT-DRIVEN:
// CTOX_SCRAPE_INPUT_JSON must carry the company URL/domain (url, website
// or domain key); there is no fixed portal host.
//
// Live-verified with scrape-targets/impressum/solo/probe.mjs on 2026-07-31
// against destilla.com, bnt-chemicals.de and akemi.de: all seven policy
// fields (firma_name, firma_anschrift, firma_plz, firma_ort, firma_telefon,
// firma_email, firma_domain) extracted from each live Impressum.
//
// Capture pattern mirrors northdata.de/scripts/v1.js: mkdtempSync out-dir,
// CTOX_BIN web browser-capture --url … --out-dir … --timeout-ms …, read
// page.html, surface capture markers for honest blocked classification,
// fall back to web browser-automation only when the runtime lacks the
// browser-capture subcommand, and remove the out-dir in a finally block.
//
// Honesty rules: only what the page states is extracted (an Impressum
// without a phone number yields no phone number). Contact details are
// attributed only from the window around the address block, so agency
// credits elsewhere on the page are never reported as company data.
//
// Drift contract: if the extraction stops matching on a loaded page this
// script returns an empty records array (portal_drift), never a crash and
// never a fabricated value.

"use strict";

const { execFileSync } = require("child_process");
const { mkdtempSync, mkdirSync, readFileSync, rmSync } = require("fs");
const { tmpdir } = require("os");
const path = require("path");

const SOURCE_ID = "impressum";
const NAVIGATION_TIMEOUT_MS = 45_000;
// Transient load failures get one second chance; a loaded page whose
// extraction yields nothing does NOT (drift contract).
const MAX_LOAD_ATTEMPTS = 2;
const CANDIDATE_PATHS = ["/impressum", "/de/impressum", "/impressum.html"];

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
  } catch (_err) {
    // Silent on per-URL failures (see northdata.de/scripts/v1.js: stderr
    // substrings would misclassify the whole run).
    return null;
  }
}

// ---------------------------------------------------------------------------
// Identity + safety helpers
// ---------------------------------------------------------------------------

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
  for (const form of ["kgaa", "gmbh", "se", "ag", "kg", "og"]) {
    if (tokens.has(form)) return form;
  }
  return null;
}

function legalFormMatches(company, candidate) {
  const expected = legalForm(company);
  return expected === null || legalForm(candidate) === expected;
}

function isPortalOrLoginTitle(title) {
  const text = String(title || "").replace(/\s+/g, " ").trim();
  if (!text) return false;
  return /\b(?:log[ -]?in|sign[ -]?in|anmeld(?:en|ung)|authentication|authentifizierung|kundenportal|customer portal)\b/i.test(text)
    || /^(?:portal|startseite|home|willkommen)(?:\s*[-|:]\s*.*)?$/i.test(text);
}

function safePublicHttpUrl(value) {
  try {
    const parsed = new URL(value);
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
      return false;
    }
    const host = parsed.hostname.toLowerCase();
    return Boolean(host)
      && host !== "localhost"
      && !host.endsWith(".localhost")
      && !host.endsWith(".local")
      && !/^(?:127\.|10\.|169\.254\.|192\.168\.)/.test(host)
      && !/^172\.(?:1[6-9]|2\d|3[01])\./.test(host)
      && host !== "::1";
  } catch (_err) {
    return false;
  }
}

// The start URL is per-company: it comes from the input, never from config.
function originFromInput(input) {
  const candidate = [input.url, input.website, input.domain, input.firma_domain]
    .map((value) => String(value || "").trim())
    .find(Boolean);
  if (!candidate) return null;
  const withScheme = /^https?:\/\//i.test(candidate) ? candidate : "https://" + candidate;
  if (!safePublicHttpUrl(withScheme)) return null;
  try {
    return new URL(withScheme).origin;
  } catch (_err) {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Extraction — identical to the solo-verified probe logic (pure over HTML).
// ---------------------------------------------------------------------------

function decodeEntities(value) {
  const named = {
    amp: "&", quot: String.fromCharCode(34), apos: String.fromCharCode(39), nbsp: " ",
    lt: "<", gt: ">",
    auml: "ä", ouml: "ö", uuml: "ü", Auml: "Ä", Ouml: "Ö", Uuml: "Ü", szlig: "ß",
    ndash: "–", mdash: "—", hellip: "…", copy: "©", reg: "®", eacute: "é",
    agrave: "à", ccedil: "ç", bull: "•", middot: "·",
  };
  return String(value || "")
    .replace(/&#(\d+);/g, (_m, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_m, code) => String.fromCodePoint(parseInt(code, 16)))
    .replace(/&([a-zA-Z]+);/g, (m, name) => (name in named ? named[name] : m));
}

function htmlToLines(html) {
  let text = String(html || "")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(?:br|hr)\b[^>]*>/gi, "\n")
    .replace(/<\/(?:p|div|li|tr|td|th|h[1-6]|section|article|header|footer|address|table|ul|ol|dl|dt|dd|blockquote|main|aside|figure|figcaption|option)>/gi, "\n")
    .replace(/<(?:p|div|li|tr|h[1-6]|section|article|header|footer|address|table|ul|ol|dl|dt|dd|blockquote|main|aside|figure|figcaption)\b[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ");
  text = decodeEntities(text);
  return text.split(/\n+/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);
}

function htmlTitle(html) {
  const match = String(html || "").match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  return match ? decodeEntities(match[1]).replace(/\s+/g, " ").trim() : "";
}

const LEGAL_FORM_RE = /\b(?:gmbh|mbh|ag|se|kg|kgaa|ohg|gbr|ug|e\.\s?k\.|ltd|llc|inc|sarl|sàrl|bv|b\.v\.|nv|n\.v\.|oy|ab|aps|sro|s\.r\.o\.|d\.o\.o\.)\b/i;

const GENERIC_LINE_RE = /^(?:impressum|imprint|angaben|anbieter|diensteanbieter|anbieterkennzeichnung|verantwortlich|verantwortliche|vertreten|inhaltlich|kontakt|contact|firma|company|unternehmen|betreiber|herausgeber|gemäß|gemaess|§|tmg|ddg|mstg|umsatzsteuer|handelsregister|register|aufsicht|geschäftsführung|geschaeftsfuehrung|vorstand|telefon|telefax|fax|e-?mail|internet|web|vertretungsberechtigt|sitz|ladungsfähige|ladungsfaehige|anschrift|adresse|address|postanschrift)\b/i;

const STREET_RE = /^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß."()\/-]*(?:[ ][A-Za-zÄÖÜäöüß."()\/-]+){0,5}[ ]?\d+\s*[a-zA-Z]?\s*(?:[\/-]\s*\d+\s*[a-zA-Z]?)?$/;

const PLZ_RE = /\b(?:[Dd]-|D )?(\d{5})[ ]([A-ZÄÖÜ][A-Za-zÄÖÜäöüß."-]*(?:[ ][A-Za-zÄÖÜäöüß."()-]+){0,3})/;

const PHONE_LABEL_RE = /(?:telefon|tel\.?|phone|zentrale)\b\s*[:.]?\s*(\+?\d[\d\s().\/-]{5,22}\d)/i;

const EMAIL_TEXT_RE = /\b([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b/g;

function looksLikeStreet(line) {
  if (line.length > 80 || PLZ_RE.test(line)) return false;
  return STREET_RE.test(line);
}

function cleanOrt(value) {
  return String(value || "")
    .replace(/[.,;:]+$/g, "")
    .replace(/\s+/g, " ")
    .trim() || null;
}

function parseAddress(lines) {
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const match = line.match(PLZ_RE);
    if (!match) continue;
    const beforePlz = line.slice(0, match.index).replace(/[,\s]+$/g, "").trim();
    let street = null;
    if (beforePlz && looksLikeStreet(beforePlz)) {
      street = beforePlz;
    } else {
      for (let back = index - 1; back >= Math.max(0, index - 4); back -= 1) {
        if (looksLikeStreet(lines[back])) { street = lines[back]; break; }
        if (PLZ_RE.test(lines[back])) break;
      }
    }
    const ort = cleanOrt(match[2]);
    if (street && ort && !/^\d/.test(ort)) {
      return { street, plz: match[1], ort, addressIndex: index };
    }
  }
  return null;
}

function extractName(lines, address, title) {
  if (address) {
    for (let back = address.addressIndex - 1; back >= Math.max(0, address.addressIndex - 8); back -= 1) {
      const line = lines[back];
      if (LEGAL_FORM_RE.test(line) && line.length <= 120 && !GENERIC_LINE_RE.test(line)) {
        return line;
      }
    }
    for (let back = address.addressIndex - 1; back >= Math.max(0, address.addressIndex - 4); back -= 1) {
      const line = lines[back];
      if (looksLikeStreet(line) || PLZ_RE.test(line)) continue;
      if (GENERIC_LINE_RE.test(line)) continue;
      if (line.length <= 90) return line;
    }
  }
  for (const part of String(title || "").split(/\s*[–—|-]\s*/)) {
    const candidate = part.trim();
    if (LEGAL_FORM_RE.test(candidate) && !/impressum|imprint/i.test(candidate) && candidate.length <= 120) {
      return candidate;
    }
  }
  return null;
}

function deobfuscateLine(line) {
  // Common Impressum obfuscation the page itself states: (at) / [dot].
  return String(line || "")
    .replace(/\s*\(\s*at\s*\)\s*/gi, "@")
    .replace(/\s*\[\s*at\s*\]\s*/gi, "@")
    .replace(/\s*\(\s*dot\s*\)\s*/gi, ".")
    .replace(/\s*\[\s*dot\s*\]\s*/gi, ".");
}

function hostBase(host) {
  const parts = String(host || "").split(".").filter(Boolean);
  return parts.slice(-2).join(".");
}

function emailMatchesHost(email, host) {
  const base = hostBase(host);
  const domain = String(email || "").split("@")[1] || "";
  return Boolean(base) && hostBase(domain) === base;
}

function extractPhones(windowLines, html) {
  const labeled = [];
  for (const line of windowLines) {
    if (/\bfax\b|telefax/i.test(line)) continue;
    const match = line.match(PHONE_LABEL_RE);
    if (match) labeled.push(match[1].replace(/\s+/g, " ").trim());
  }
  const dedup = [...new Set(labeled)].filter((value) => value.replace(/\D/g, "").length >= 6);
  if (dedup.length > 0) return dedup;
  const hrefs = [];
  for (const match of String(html || "").matchAll(/href\s*=\s*["]tel:([^"]+)["]/gi)) {
    hrefs.push(decodeEntities(match[1]).replace(/\s+/g, " ").trim());
  }
  return [...new Set(hrefs)].filter((value) => value.replace(/\D/g, "").length >= 6);
}

function extractEmails(windowLines, html, host) {
  // Only the contact window around the address block counts: impressum pages
  // routinely credit a web agency with its own email further down, and that
  // address belongs to the agency, not to the company.
  const textEmails = [];
  for (const rawLine of windowLines) {
    const line = deobfuscateLine(rawLine);
    for (const match of line.matchAll(EMAIL_TEXT_RE)) {
      const email = match[1].toLowerCase();
      if (/\.(?:png|jpe?g|gif|webp|svg|css|js)$/.test(email)) continue;
      if (/^(?:example|mustermann|noreply@(?:wixpress|github))/.test(email)) continue;
      textEmails.push(email);
    }
  }
  const mailtoEmails = [];
  for (const match of String(html || "").matchAll(/href\s*=\s*["]mailto:([^">?]+)/gi)) {
    const email = decodeEntities(match[1]).trim().toLowerCase();
    if (/^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$/.test(email)) mailtoEmails.push(email);
  }
  const text = [...new Set(textEmails)];
  const mailto = [...new Set(mailtoEmails)];
  // A company impressum states the company email on the company domain;
  // off-domain mailtos (agency credits) are never the firma_email.
  return text.find((email) => emailMatchesHost(email, host))
    || mailto.find((email) => emailMatchesHost(email, host))
    || text[0]
    || null;
}

function isBlockedText(lines, title) {
  const corpus = title + " " + lines.slice(0, 30).join(" ");
  return /captcha|cloudflare|verify you are human|access denied|zugriff verweigert|sicherheitsüberprüfung|just a moment/i.test(corpus);
}

function extractImpressum(html, finalUrl) {
  const lines = htmlToLines(html);
  const title = htmlTitle(html);
  if (isBlockedText(lines, title)) return { blocked: true, fields: {} };

  const headingIndex = lines.findIndex((line) => /^(?:impressum|imprint|anbieterkennzeichnung)\b/i.test(line));
  const region = headingIndex >= 0 ? lines.slice(headingIndex) : lines;

  const address = parseAddress(region);
  const name = extractName(region, address, title);

  let host = "";
  try { host = new URL(finalUrl).hostname.replace(/^www\./, "").toLowerCase(); } catch (_err) { /* keep empty */ }

  // Contact details belong to the address block: search a window around it
  // so credits/footers (agency phone, agency email) are never attributed
  // to the company.
  const windowStart = address ? Math.max(0, address.addressIndex - 10) : 0;
  const windowEnd = address ? address.addressIndex + 16 : region.length;
  const windowLines = region.slice(windowStart, windowEnd);
  const phones = extractPhones(windowLines, html);
  const emails = extractEmails(windowLines, html, host);

  const fields = {};
  const put = (key, value) => {
    const clean = String(value || "").replace(/\s+/g, " ").trim();
    if (clean) fields[key] = { value: clean, source_url: finalUrl };
  };
  put("firma_name", name);
  if (address) {
    put("firma_anschrift", address.street);
    put("firma_plz", address.plz);
    put("firma_ort", address.ort);
  }
  put("firma_telefon", phones[0]);
  put("firma_email", emails);
  put("firma_domain", host);
  return { blocked: false, fields, title, lineCount: lines.length };
}

function looksLikeImpressumHtml(html) {
  const text = htmlToLines(html).join(" ");
  return /impressum|imprint|angaben gemäß|anbieterkennzeichnung/i.test(text) || PLZ_RE.test(text);
}

// Start-page fallback: an anchor whose text or href mentions Impressum/Imprint.
function discoverImpressumLink(html, origin) {
  const anchors = String(html || "").matchAll(/<a\b[^>]*href\s*=\s*["]([^"]+)["][^>]*>([\s\S]*?)<\/a>/gi);
  for (const match of anchors) {
    const href = decodeEntities(match[1]);
    const text = htmlToLines(match[2]).join(" ");
    if (!/impressum|imprint/i.test(href) && !/impressum|imprint/i.test(text)) continue;
    try {
      const target = new URL(href, origin);
      const sameOrigin = target.hostname.replace(/^www\./, "") === new URL(origin).hostname.replace(/^www\./, "");
      if (sameOrigin && safePublicHttpUrl(target.href)) return target.href;
    } catch (_err) { /* try the next anchor */ }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Capture — northdata.de pattern: browser-capture, page.html, markers,
// finally-rmSync, browser-automation only when the subcommand is missing.
// ---------------------------------------------------------------------------

function browserCapturePage(url) {
  if (!safePublicHttpUrl(url)) return { page: null, commandUnavailable: false };
  const captureRoot = process.env.CTOX_SCRAPE_OUTPUT_DIR || tmpdir();
  mkdirSync(captureRoot, { recursive: true });
  const outDir = mkdtempSync(path.join(captureRoot, "impressum-browser-capture-"));
  try {
    const args = [
      "web", "browser-capture",
      "--url", url,
      "--out-dir", outDir,
      "--timeout-ms", String(NAVIGATION_TIMEOUT_MS),
    ];
    let payload;
    try {
      const out = execFileSync(ctoxBin(), args, {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
        maxBuffer: 32 * 1024 * 1024,
        timeout: (NAVIGATION_TIMEOUT_MS * 2) + 20_000,
      });
      payload = JSON.parse(out);
    } catch (err) {
      const detail = String((err && (err.stderr || err.stdout || err.message)) || "");
      return {
        page: null,
        commandUnavailable: /unsupported|unknown|unrecognized|usage:/i.test(detail),
      };
    }

    const markerMap = payload && payload.markers && typeof payload.markers === "object"
      ? payload.markers
      : {};
    const markers = Object.entries(markerMap)
      .filter(([, detected]) => detected === true)
      .map(([marker]) => marker);
    if (!payload || !payload.ok) {
      return {
        page: {
          ok: false,
          url: (payload && (payload.finalUrl || payload.targetUrl)) || url,
          title: (payload && payload.title) || "",
          capture_markers: markerMap,
          detection: { markers },
        },
        commandUnavailable: false,
      };
    }

    let html;
    try {
      html = readFileSync(path.join(outDir, "page.html"), "utf8");
    } catch (_err) {
      return { page: null, commandUnavailable: false };
    }
    return {
      page: {
        ok: true,
        url: payload.finalUrl || payload.targetUrl || url,
        final_url: payload.finalUrl || null,
        title: payload.title || "",
        html,
        raw_html: html,
        capture_markers: markerMap,
        detection: { markers },
      },
      commandUnavailable: false,
    };
  } finally {
    rmSync(outDir, { recursive: true, force: true });
  }
}

// Fallback for runtimes that do not expose browser-capture yet. Consent
// overlays do not remove the legal notice from the DOM, so a plain load
// plus one ordinary consent click is enough.
function impressumBrowserSource(url) {
  return [
    "const targetUrl = " + JSON.stringify(url) + ";",
    "await page.goto(targetUrl, { waitUntil: \"domcontentloaded\", timeout: 45000 }).catch(async () => {",
    "  await page.waitForTimeout(2000);",
    "  await page.goto(targetUrl, { waitUntil: \"domcontentloaded\", timeout: 45000 });",
    "});",
    "const consentPatterns = [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i, /^accept( all)?$/i, /verstanden/i];",
    "for (const pattern of consentPatterns) {",
    "  const button = page.getByRole(\"button\", { name: pattern }).first();",
    "  if (await button.count()) { await button.click({ timeout: 3000 }).catch(() => null); break; }",
    "}",
    "await page.waitForTimeout(1500);",
    "const html = document.documentElement.outerHTML;",
    "return { url: page.url(), title: document.title, html, body_text: ((document.body && document.body.innerText) || \"\").slice(0, 120000) };",
  ].join("\n");
}

function browserAutomationPage(url) {
  if (!safePublicHttpUrl(url)) return null;
  const source = impressumBrowserSource(url);
  const payload = runCtox(["web", "browser-automation", "--timeout-ms", "150000"], source, 160_000);
  if (!payload) return null;
  const result = payload.result || {};
  return {
    ...result,
    ok: payload.ok === true && Boolean(result.html || result.body_text),
    detection: payload.detection,
  };
}

function isBlockedPage(page) {
  const markers = Array.isArray(page && page.detection && page.detection.markers)
    ? page.detection.markers.join(" ")
    : "";
  const corpus = normalized([
    page && page.title,
    page && page.body_text,
    page && page.raw_html,
    page && page.html,
    markers,
  ].filter(Boolean).join(" "));
  return /captcha|cloudflare|challenge|turnstile|verify you are human|access denied|request blocked|too many requests|wurden gesperrt|sicherheitsuberprufung/.test(corpus);
}

function recordUnlockSignal(url, markers) {
  const args = [
    "web", "unlock", "signals", "record",
    "--source", "scrape-target:impressum",
    "--evidence", JSON.stringify({
      source_id: SOURCE_ID,
      detection: "access_challenge",
      markers: [...new Set((markers || []).map(String))].slice(0, 12),
      secret_value_in_payload: false,
    }),
  ];
  if (safePublicHttpUrl(url)) args.push("--url", url);
  return runCtox(args);
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

const RECORD_NOTES = {
  firma_name: "Impressum: company name stated above the address block",
  firma_anschrift: "Impressum: street line of the address block",
  firma_plz: "Impressum: postal code of the address block",
  firma_ort: "Impressum: city of the address block",
  firma_telefon: "Impressum: labelled phone number near the address block",
  firma_email: "Impressum: email stated near the address block",
  firma_domain: "company origin hosting its own Impressum",
};

function recordsFromFields(fields, host) {
  const records = [];
  for (const [field, entry] of Object.entries(fields || {})) {
    const value = String((entry && entry.value) || "").replace(/\s+/g, " ").trim();
    const sourceUrl = String((entry && entry.source_url) || "").trim();
    if (!value || !safePublicHttpUrl(sourceUrl)) continue;
    const confidence = field === "firma_email" && !emailMatchesHost(value, host)
      ? "medium"
      : "high";
    records.push({
      field,
      value,
      confidence,
      source_url: sourceUrl,
      note: RECORD_NOTES[field] || "Impressum extraction",
    });
  }
  return records;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main() {
  const input = readInput();
  const company = String(input.company || "").trim();
  const origin = originFromInput(input);
  if (!origin) {
    process.stdout.write(JSON.stringify({
      records: [],
      failure_mode: "portal_drift",
      detail: "impressum target is input-driven: url/website/domain input required",
    }));
    return;
  }
  const host = new URL(origin).hostname.replace(/^www\./, "").toLowerCase();

  let blocked = false;
  let identityMismatch = false;

  const loadPage = (url) => {
    const capture = browserCapturePage(url);
    // Compatibility only for runtimes that do not expose browser-capture
    // yet. A capture that ran and failed is never retried through another
    // transport.
    return capture.commandUnavailable ? browserAutomationPage(url) : capture.page;
  };

  // Returns the verified fields for a page, or null. Sets blocked /
  // identityMismatch as side channels for honest failure classification.
  const consider = (url, loadedRef) => {
    const page = loadPage(url);
    if (!page) return null;
    if (isBlockedPage(page)) { blocked = true; return null; }
    if (!page.ok) return null;
    loadedRef.loaded = true;
    const html = page.html || page.raw_html || "";
    const title = String(page.title || htmlTitle(html));
    if (isPortalOrLoginTitle(title)) return null;
    if (!html || !looksLikeImpressumHtml(html)) return null;
    const result = extractImpressum(html, page.url || url);
    if (result.blocked) { blocked = true; return null; }
    const fields = result.fields || {};
    const hasAddress = fields.firma_anschrift && fields.firma_plz && fields.firma_ort;
    if (!hasAddress || !fields.firma_name) return null; // drift contract
    if (company && !(
      identityMatches(company, fields.firma_name.value)
        && legalFormMatches(company, fields.firma_name.value)
    )) {
      identityMismatch = true;
      return null;
    }
    return fields;
  };

  for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt += 1) {
    const loadedRef = { loaded: false };
    let fields = null;
    for (const candidate of CANDIDATE_PATHS) {
      fields = consider(origin + candidate, loadedRef);
      if (blocked || fields) break;
    }
    if (!fields && !blocked) {
      // Start-page fallback: follow the Impressum/Imprint link.
      const startPage = loadPage(origin + "/");
      if (startPage && isBlockedPage(startPage)) {
        blocked = true;
      } else if (startPage && startPage.ok) {
        loadedRef.loaded = true;
        const html = startPage.html || startPage.raw_html || "";
        const link = html ? discoverImpressumLink(html, origin) : null;
        if (link) fields = consider(link, loadedRef);
      }
    }
    if (fields) {
      process.stdout.write(JSON.stringify({ records: recordsFromFields(fields, host) }));
      return;
    }
    if (loadedRef.loaded || blocked || identityMismatch) break;
    // Nothing loaded at all (transient failure): one second chance.
    if (attempt + 1 < MAX_LOAD_ATTEMPTS) {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2_000);
    }
  }

  if (blocked) recordUnlockSignal(origin, ["access_challenge"]);

  process.stdout.write(JSON.stringify({
    records: [],
    failure_mode: blocked ? "blocked" : "portal_drift",
    detail: blocked
      ? "access challenge on the company site recorded for web-unlock"
      : identityMismatch
        ? "an impressum-like page loaded but its company identity does not match the input"
        : "no impressum page with extractable prospect fields (loaded pages yield empty records, never fabricated ones)",
  }));
}

if (require.main === module) {
  main();
}

module.exports = {
  browserAutomationPage,
  browserCapturePage,
  discoverImpressumLink,
  extractImpressum,
  identityMatches,
  isBlockedPage,
  isPortalOrLoginTitle,
  legalFormMatches,
  looksLikeImpressumHtml,
  originFromInput,
  recordsFromFields,
  safePublicHttpUrl,
};
