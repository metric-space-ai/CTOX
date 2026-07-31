// impressum — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/impressum/solo/probe.mjs <domain-or-url | company name>
//
// Drives the LIVE company site headless, locates its legal notice page
// (Impressum / Imprint) and extracts the prospect.v1 contact fields the
// research policy expects from this source:
//   firma_name, firma_anschrift, firma_plz, firma_ort, firma_telefon,
//   firma_email, firma_domain
// plus the legally required representatives (§ 5 DDG/TMG), one record set
// per named person:
//   person_vorname, person_nachname, person_funktion, person_titel
//
// When the input is a company name rather than a domain, the probe derives
// candidate hosts from the name (same derivation as scripts/v1.js) and keeps
// the first candidate whose own legal notice passes the same identity +
// legal-form gate the adapter applies. A wrong guess is discarded silently.
// Prints ONE JSON object:
//   {target, input, fetched_at, verified_host?, fields: {<field_key>: {value, source_url}},
//    persons: [{person_vorname, person_nachname, person_funktion,
//               person_titel?, source_url}]}
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

if (!rawInput) fail("usage: probe.mjs <domain-or-url | company name>");

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

// A domain/URL input drives one origin; anything else is treated as a
// company name and goes through domain discovery.
const origin = toOrigin(rawInput);

// ---------------------------------------------------------------------------
// Identity gate + candidate derivation — identical to scripts/v1.js. A
// guessed candidate host verifies only through its own legal notice: the
// notice must name the company (identity) with the same legal form. A wrong
// guess (a parent group's domain, a namesake) is discarded silently.
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

const MAX_CANDIDATE_HOSTS = 6;
const CANDIDATE_POLITENESS_MS = 1500;

function candidateHostsFromCompany(company) {
  const withoutLegalForm = String(company || "")
    .replace(/&\s*Co\.?/gi, " ")
    .replace(/\b(?:gmbh|mbh|kgaa|ag|se|kg|ohg|gbr|ug|ltd|llc|inc|co)\b\.?/gi, " ");
  const transliterated = withoutLegalForm
    .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue")
    .replace(/Ä/g, "Ae").replace(/Ö/g, "Oe").replace(/Ü/g, "Ue")
    .replace(/ß/g, "ss");
  const words = transliterated.toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 0) return [];
  // Keep this list and its ORDER identical to scripts/v1.js. The probe is the
  // solo-first proving ground — if the two drift apart, a probe run stops
  // saying anything about what the deployed adapter will do. Ordered by how
  // German companies actually name their sites: leading word (aeroxon.de),
  // hyphenated full name (bnt-chemicals.de), concatenation, then .com.
  const joined = words.join("");
  const hyphenated = words.join("-");
  const hosts = [];
  if (words.length > 1) hosts.push(words[0] + ".de", hyphenated + ".de");
  hosts.push(joined + ".de");
  if (words.length > 1) hosts.push(words[0] + ".com", hyphenated + ".com");
  hosts.push(joined + ".com");
  if (words.length > 1) {
    const initials = words.map((word) => word[0]).join("");
    if (initials.length >= 2) hosts.push(initials + ".de", initials + ".com");
  }
  return [...new Set(hosts)].slice(0, MAX_CANDIDATE_HOSTS);
}

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

// ---------------------------------------------------------------------------
// Representatives (§ 5 DDG/TMG): the notice names the authorised
// representatives (Geschäftsführer, Vorstand, Inhaber, Vertretungsberechtigte).
// Only what the notice itself states is extracted — never inferred from an
// email address, and never attributed from a web agency's credit block (the
// same discipline the contact window applies to phones and emails).
// ---------------------------------------------------------------------------

const PERSON_LABEL_RE = /^(geschäftsführer(?:in)?|geschäftsführung|vertreten\s+durch|vertretungsberechtigte?r?(?:\(r\))?|vorstand|vorstände|inhaber(?:in)?)(?:\s*:\s*|\s+)(.*)$/i;

const ROLE_PREFIX_RE = /^(geschäftsführer(?:in)?|geschäftsführung|vorstand|inhaber(?:in)?|prokurist(?:in)?)\s*:\s*(.*)$/i;

const ROLE_EXACT_RE = /^(?:geschäftsführer(?:in)?|geschäftsführung|vorstand|vorstandsmitglied|inhaber(?:in)?|prokurist(?:in)?|gesellschafter(?:in)?)$/i;

const TITLE_PAREN_RE = /(?:dipl|dr|prof|mag|ing|kaufmann|kauffrau|betriebswirt|fachwirt|meister|techniker|ökonom|oekonom|med|rer|nat|jur|mba|msc|bsc|wirtschafts)/i;

const TITLE_FIRST_RE = /^(?:prof\.?|pd|dr\.?|habil\.?|dipl\.?-[a-zäöü]+\.?|dipl\.?|mag\.?|ing\.?|mba|msc|m\.sc\.|bsc|b\.sc\.|ll\.?m\.?)$/i;

const TITLE_CONT_RE = /^(?:med|rer|nat|jur|phil|dent|habil|techn|oec|pol|agr|ing|sc)\.?$/i;

const NAME_PARTICLES = new Set(["von", "van", "de", "der", "den", "zu", "vom", "da", "di", "del", "la", "le", "ten"]);

// Lines that end a person list: structural labels of the legal notice, the
// agency credit block, or another person label.
const PERSON_STOP_RE = /(?:handelsregister|registereintrag|registergericht|registernummer|umsatzsteuer|ust\.?-?id|steuernummer|telefon|telefax|fax\b|e-?mail|homepage|amtsgericht|anschrift|postfach|impressum|datenschutz|kontakt|haftungs|urheber|bildquellen|quellenangaben|konzeption|design|umsetzung|programmierung|agentur|verantwortlich|redaktion|betreiber|anbieter|ladungsfähig|ladungsfaehig|öffnungszeiten|geschäftsführer|geschaeftsfuehrer|vorstand|inhaber|vertretungsberechtigt)/i;

// A person label standing next to an agency credit ("Umsetzung", "Webdesign",
// "Betreuende Agentur" …) names the agency's staff, not the company's.
const AGENCY_CONTEXT_RE = /(?:konzeption|screendesign|webdesign|webentwicklung|gestaltung|programmierung|realisierung|umsetzung|betreuende\s+agentur|\bagentur\b|erstellt\s+(?:von|durch)|design\s+by|made\s+by|fotograf|bildquellen|quellenangaben|webmaster)/i;

function validNameTokens(tokens) {
  if (tokens.length < 2 || tokens.length > 5) return false;
  return tokens.every((token) =>
    /^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'.-]*$/.test(token) || NAME_PARTICLES.has(token.toLowerCase()));
}

// One segment -> one person. Titles stated on the page (leading "Dr."/"Prof."
// or a parenthetical like "(Dipl. Kaufmann)") become person_titel; a
// parenthetical role word ("(Vorstand)") becomes person_funktion; anything
// else in parentheses is a clause, not person data, and is dropped.
function parsePerson(segment, funktion) {
  let text = String(segment || "").replace(/\s+/g, " ").trim().replace(/[.,;:]+$/, "").trim();
  if (!text || LEGAL_FORM_RE.test(text)) return null;
  let titel = null;
  let role = null;
  text = text.replace(/\(([^)]{1,50})\)/g, (_m, inner) => {
    const content = inner.replace(/\s+/g, " ").trim();
    if (ROLE_EXACT_RE.test(content)) role = content;
    else if (TITLE_PAREN_RE.test(content)) titel = titel ? titel + " " + content : content;
    return " ";
  }).replace(/\s+/g, " ").trim();
  const tokens = text.split(/\s+/).filter(Boolean);
  const leading = [];
  while (tokens.length > 2 && TITLE_FIRST_RE.test(tokens[0])) {
    leading.push(tokens.shift());
    while (tokens.length > 2 && TITLE_CONT_RE.test(tokens[0])) leading.push(tokens.shift());
  }
  if (leading.length > 0) titel = titel ? leading.join(" ") + " " + titel : leading.join(" ");
  if (!validNameTokens(tokens)) return null;
  return {
    vorname: tokens.slice(0, -1).join(" "),
    nachname: tokens[tokens.length - 1],
    funktion: role || funktion,
    titel,
  };
}

// Several people are separated by commas, semicolons, "und"/"sowie" or line
// breaks. Commas inside parentheses belong to the parenthetical.
function splitPersonSegments(text) {
  const flattened = String(text || "").replace(/\(([^)]*)\)/g, (m) => m.replace(/[,;]/g, " "));
  return flattened
    .split(/[,;]|\s+und\s+|\s+sowie\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function isPersonContinuationLine(line) {
  const text = String(line || "").trim();
  if (!text || text.length > 90) return false;
  if (/[:@]/.test(text) || /\d/.test(text)) return false;
  if (PERSON_STOP_RE.test(text) || LEGAL_FORM_RE.test(text)) return false;
  return splitPersonSegments(text).some((segment) => parsePerson(segment, "x") !== null);
}

function isAgencyContext(lines, index) {
  // Only what precedes (or heads) the label line: an agency credit introduces
  // its own staff below it; a credit AFTER a representative label does not
  // make the company's representative agency staff.
  for (let at = Math.max(0, index - 2); at <= index; at += 1) {
    if (AGENCY_CONTEXT_RE.test(lines[at])) return true;
  }
  return false;
}

function personKey(person) {
  return (person.vorname + " " + person.nachname).toLocaleLowerCase("de-DE");
}

// Every person the notice names, each as their own record set — never
// collapsed into one.
function extractPersons(region, finalUrl) {
  const persons = [];
  const seen = new Set();
  for (let index = 0; index < region.length; index += 1) {
    const match = region[index].match(PERSON_LABEL_RE);
    if (!match) continue;
    if (isAgencyContext(region, index)) continue;
    let funktion = match[1].replace(/\s+/g, " ").trim();
    let rest = String(match[2] || "").trim();
    const rolePrefix = rest.match(ROLE_PREFIX_RE);
    if (rolePrefix) {
      funktion = rolePrefix[1].replace(/\s+/g, " ").trim();
      rest = rolePrefix[2].trim();
    }
    const segments = [];
    if (rest) segments.push(...splitPersonSegments(rest));
    for (let next = index + 1; next < Math.min(region.length, index + 9); next += 1) {
      if (!isPersonContinuationLine(region[next])) break;
      segments.push(...splitPersonSegments(region[next]));
    }
    for (const segment of segments) {
      const person = parsePerson(segment, funktion);
      if (!person) continue;
      const key = personKey(person);
      if (seen.has(key)) continue;
      seen.add(key);
      persons.push({ ...person, source_url: finalUrl });
    }
  }
  return persons;
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
  const persons = extractPersons(region, finalUrl);

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
  return { blocked: false, fields, persons, title, lineCount: lines.length };
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

const candidatePaths = ["/impressum", "/de/impressum", "/impressum.html"];

// Returns { url, html } for the notice page of an origin, or { reason }.
// A reason means the origin is unreachable or shows no notice — callers
// doing domain discovery skip such a candidate, never retry it.
async function locateImpressum(originUrl) {
  for (const candidate of candidatePaths) {
    const response = await goto(originUrl + candidate);
    await sleep(2000); // politeness between navigations
    if (!response || !response.ok()) continue;
    const content = await page.content();
    if (looksLikeImpressumHtml(content)) {
      return { url: page.url(), html: content };
    }
  }

  // Fall back to the start page and follow an Impressum/Imprint link.
  const response = await goto(originUrl + "/");
  await sleep(2000);
  if (!response || !response.ok()) {
    return { reason: "start page unreachable from " + originUrl };
  }
  await clickConsent();
  const link = await page.evaluate(() => {
    const anchors = Array.from(document.querySelectorAll("a[href]"));
    const match = anchors.find((anchor) =>
      /impressum|imprint/i.test(anchor.textContent || "")
        || /impressum|imprint/i.test(anchor.getAttribute("href") || ""));
    return match ? match.href : null;
  }).catch(() => null);
  if (!link) return { reason: "no Impressum link found on " + originUrl };
  const target = new URL(link, originUrl);
  const sameOrigin = target.hostname.replace(/^www\./, "") === new URL(originUrl).hostname.replace(/^www\./, "");
  if (!sameOrigin) return { reason: "impressum link points off-origin: " + target.href };
  const linked = await goto(target.href);
  await sleep(2000);
  if (!linked || !linked.ok()) return { reason: "impressum link unreachable: " + target.href };
  return { url: page.url(), html: await page.content() };
}

function printSuccess(result, extra) {
  console.log(JSON.stringify({
    target: TARGET,
    input: rawInput,
    fetched_at: new Date().toISOString(),
    ...(extra || {}),
    fields: result.fields,
    persons: (result.persons || []).map((person) => ({
      person_vorname: person.vorname,
      person_nachname: person.nachname,
      person_funktion: person.funktion,
      ...(person.titel ? { person_titel: person.titel } : {}),
      source_url: person.source_url,
    })),
  }, null, 2));
  process.exit(0);
}

try {
  if (origin) {
    const found = await locateImpressum(origin);
    if (found.reason) fail(found.reason);

    await clickConsent();

    const result = extractImpressum(found.html, found.url);
    if (result.blocked) fail("blocked: anti-bot page detected");
    const fields = result.fields;
    const hasAddress = fields.firma_anschrift && fields.firma_plz && fields.firma_ort;
    if (!hasAddress) {
      fail("no full address extracted (title: " + JSON.stringify(result.title || "") + ")", fields);
    }
    if (!fields.firma_name) {
      fail("no company name extracted (title: " + JSON.stringify(result.title || "") + ")", fields);
    }
    printSuccess(result);
  }

  // Company name only — domain discovery: derive candidate hosts from the
  // name and keep the first whose own legal notice verifies the company.
  const company = rawInput;
  const candidates = candidateHostsFromCompany(company);
  if (candidates.length === 0) {
    fail("cannot derive candidate hosts from " + JSON.stringify(rawInput));
  }
  const tried = [];
  for (const candidateHost of candidates) {
    if (tried.length > 0) await sleep(CANDIDATE_POLITENESS_MS); // politeness between candidate hosts
    tried.push(candidateHost);
    const candidateOrigin = "https://" + candidateHost;
    // DNS/connection failure or no notice on this host: skip the candidate,
    // never retry it.
    const found = await locateImpressum(candidateOrigin);
    if (found.reason) continue;

    await clickConsent();

    const result = extractImpressum(found.html, found.url);
    if (result.blocked) continue;
    const fields = result.fields || {};
    const hasAddress = fields.firma_anschrift && fields.firma_plz && fields.firma_ort;
    if (!hasAddress || !fields.firma_name) continue;
    // Verification is unchanged from the adapter's gate: the notice on this
    // host must name the company with the same legal form. No "close
    // enough" — a wrong guess reports no domain rather than a plausible one.
    if (!(
      identityMatches(company, fields.firma_name.value)
        && legalFormMatches(company, fields.firma_name.value)
    )) {
      continue;
    }
    printSuccess(result, { verified_host: candidateHost });
  }
  fail("domain discovery found no host whose impressum verifies "
    + JSON.stringify(company) + " (tried: " + tried.join(", ") + ")");
} finally {
  await browser.close().catch(() => null);
}
