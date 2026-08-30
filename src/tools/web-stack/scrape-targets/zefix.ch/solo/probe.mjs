#!/usr/bin/env node
// zefix.ch — standalone live probe against the PUBLIC Zefix REST API.
//
// Usage: node scrape-targets/zefix.ch/solo/probe.mjs "<company name>"
//
// Prints one JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when at least one real prospect field was extracted from the
// LIVE API. Otherwise exit 1 with {reason: "..."}.
//
// No Playwright: zefix.ch is access_mode "public_native_api" (target.json), so
// the adapter path is the documented REST endpoint, not the HTML portal.
// A block/HTTP error is a finding and is reported, never evaded.

const TARGET = 'zefix.ch';
const API_BASE = 'https://www.zefix.admin.ch/ZefixREST/api/v1';
const SEARCH_URL = `${API_BASE}/firm/search.json`;
const TIMEOUT_MS = 25000;

const company = (process.argv[2] || '').trim();
if (!company) {
  console.log(JSON.stringify({ target: TARGET, input: '', reason: 'missing company CLI argument' }));
  process.exit(1);
}

function normalizedIdentity(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

async function main() {
  const fields = {};
  let reason = null;

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    const response = await fetch(SEARCH_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ name: company, activeOnly: true }),
      signal: controller.signal,
    }).finally(() => clearTimeout(timer));

    const bodyText = await response.text();

    // Zefix answers an empty result set with HTTP 404 + code API.ZFR.SEARCH.NORESULT.
    // That is a legitimate "no match", NOT a block — keep the two apart.
    if (response.status === 404 && /SEARCH\.NORESULT/.test(bodyText)) {
      reason = `no result: Zefix returned zero firms for "${company}"`;
    } else if (!response.ok) {
      reason = `blocked or api error: HTTP ${response.status} ${response.statusText} — ${bodyText.slice(0, 200)}`;
    } else {
      let payload = null;
      try {
        payload = JSON.parse(bodyText);
      } catch {
        reason = `portal drift: response is not JSON — ${bodyText.slice(0, 200)}`;
      }

      if (!reason) {
        const list = Array.isArray(payload?.list) ? payload.list : [];
        if (list.length === 0) {
          reason = `no result: Zefix returned zero firms for "${company}"`;
        } else {
          const expected = normalizedIdentity(company);
          const hit =
            list.find((entry) => {
              const got = normalizedIdentity(entry?.name);
              return got && (got === expected || got.includes(expected) || expected.includes(got));
            }) || null;

          if (!hit) {
            reason = `identity mismatch: no returned firm name matches "${company}" (first: "${list[0]?.name}")`;
          } else {
            const sourceUrl = hit.ehraid
              ? `${API_BASE}/firm/${hit.ehraid}.json`
              : SEARCH_URL;
            const put = (key, value) => {
              if (value === null || value === undefined || value === '') return;
              fields[key] = { value: String(value), source_url: sourceUrl };
            };
            put('firma_name', hit.name);
            put('firma_ort', hit.legalSeat);
            put('firma_status', hit.status);
            put('firma_uid', hit.uidFormatted || hit.uid);

            if (Object.keys(fields).length === 0) {
              reason = 'portal drift: firm entry returned but no known API keys present';
            }
          }
        }
      }
    }
  } catch (error) {
    const message = String(error?.name === 'AbortError' ? 'request timeout' : error?.message || error);
    reason = `fatal: ${message.slice(0, 300)}`;
  }

  const output = {
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields,
  };
  if (reason) output.reason = reason;
  console.log(JSON.stringify(output, null, 2));
  process.exit(reason ? 1 : 0);
}

main();
