// impressum — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/impressum/solo/probe.mjs <domain-or-url>
//
// Drives the LIVE company site headless, locates its legal notice page
// (Impressum / Imprint) and extracts the prospect.v1 contact fields the
// research policy expects from this source:
//   firma_name, firma_anschrift, firma_plz, firma_ort, firma_telefon,
//   firma_email, firma_domain
// Prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when a real address (anschrift + plz + ort) plus a name were
// extracted from the page itself; otherwise non-zero with a reason.
// Only what the page states is extracted — never inferred, never guessed.

import { chromium } from "playwright";

const TARGET = "impressum";
const rawInput = (process.argv[2] || "").trim();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fail(reason, fields) {
  console.log(JSON.stringify({
    target: TARGET,
    input: rawInput,
    fetched_at: new Date().toISOString(),
    fields: fields || {},
    reason,
  }));
  process.exit(reason.startsWith("usage") ? 2 : 1);
}

if (!rawInput) fail("usage: probe.mjs <domain-or-url>");

function toOrigin(value) {
  const withScheme = /^https?:\/\//i.test(value) ? value : "https://" + value;
  try {
    const url = new URL(withScheme);
    if (!url.hostname.includes(".")) return null;
    return url.origin;
  } catch (_err) {
    return null;
  }
}

const origin = toOrigin(rawInput);
if (!origin) fail("cannot derive an https origin from " + JSON.stringify(rawInput));

// ---------------------------------------------------------------------------
// Extraction — pure functions over HTML. scripts/v1.js reuses them verbatim.
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
  // 1. Nearest line above the address block that carries a legal form.
  if (address) {
    for (let back = address.addressIndex - 1; back >= Math.max(0, address.addressIndex - 8); back -= 1) {
      const line = lines[back];
      if (LEGAL_FORM_RE.test(line) && line.length <= 120 && !GENERIC_LINE_RE.test(line)) {
        return line;
      }
    }
    // 2. The nearest usable line above the street, unless it is a generic label.
    for (let back = address.addressIndex - 1; back >= Math.max(0, address.addressIndex - 4); back -= 1) {
      const line = lines[back];
      if (looksLikeStreet(line) || PLZ_RE.test(line)) continue;
      if (GENERIC_LINE_RE.test(line)) continue;
      if (line.length <= 90) return line;
    }
  }
  // 3. Title segment carrying a legal form (e.g. Impressum - Beispiel GmbH).
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

  // Narrow to the region after the Impressum/Imprint heading when one exists.
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

// ---------------------------------------------------------------------------
// Live drive
// ---------------------------------------------------------------------------

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ locale: "de-DE" }); // default user agent
const page = await context.newPage();

async function goto(url) {
  try {
    return await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
  } catch (_err) {
    return null;
  }
}

async function clickConsent() {
  const patterns = [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i, /^accept( all)?$/i, /verstanden/i, /^okay?$|^ok$/i];
  for (const pattern of patterns) {
    const button = page.getByRole("button", { name: pattern }).first();
    if (await button.count()) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await sleep(1000);
      return true;
    }
  }
  return false;
}

function looksLikeImpressumHtml(html) {
  const text = htmlToLines(html).join(" ");
  return /impressum|imprint|angaben gemäß|anbieterkennzeichnung/i.test(text) || PLZ_RE.test(text);
}

try {
  const candidatePaths = ["/impressum", "/de/impressum", "/impressum.html"];
  let impressumUrl = null;
  let html = null;

  for (const candidate of candidatePaths) {
    const response = await goto(origin + candidate);
    await sleep(2000); // politeness between navigations
    if (!response || !response.ok()) continue;
    const content = await page.content();
    if (looksLikeImpressumHtml(content)) {
      impressumUrl = page.url();
      html = content;
      break;
    }
  }

  if (!html) {
    // Fall back to the start page and follow an Impressum/Imprint link.
    const response = await goto(origin + "/");
    await sleep(2000);
    if (!response || !response.ok()) {
      fail("start page unreachable from " + origin);
    }
    await clickConsent();
    const link = await page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll("a[href]"));
      const match = anchors.find((anchor) =>
        /impressum|imprint/i.test(anchor.textContent || "")
          || /impressum|imprint/i.test(anchor.getAttribute("href") || ""));
      return match ? match.href : null;
    }).catch(() => null);
    if (!link) fail("no Impressum link found on " + origin);
    const target = new URL(link, origin);
    const sameOrigin = target.hostname.replace(/^www\./, "") === new URL(origin).hostname.replace(/^www\./, "");
    if (!sameOrigin) fail("impressum link points off-origin: " + target.href);
    const linked = await goto(target.href);
    await sleep(2000);
    if (!linked || !linked.ok()) fail("impressum link unreachable: " + target.href);
    impressumUrl = page.url();
    html = await page.content();
  }

  await clickConsent();

  const result = extractImpressum(html, impressumUrl || page.url());
  if (result.blocked) fail("blocked: anti-bot page detected");
  const fields = result.fields;
  const hasAddress = fields.firma_anschrift && fields.firma_plz && fields.firma_ort;
  if (!hasAddress) {
    fail("no full address extracted (title: " + JSON.stringify(result.title || "") + ")", fields);
  }
  if (!fields.firma_name) {
    fail("no company name extracted (title: " + JSON.stringify(result.title || "") + ")", fields);
  }
  console.log(JSON.stringify({
    target: TARGET,
    input: rawInput,
    fetched_at: new Date().toISOString(),
    fields,
  }, null, 2));
  process.exit(0);
} finally {
  await browser.close().catch(() => null);
}
