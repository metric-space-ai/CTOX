export const FEW_SHOT_PLAYWRIGHT_CODE_EXAMPLES = Object.freeze([
  {
    id: "open-and-scan",
    title: "Open And Scan",
    intent: "Extract core page evidence from the currently loaded page.",
    setupUrl: "https://example.com/",
    code:
`await page.waitForTimeout(200);
const evidence = await page.evaluate(() => {
  const clean = (s) => String(s || "").replace(/\\s+/g, " ").trim();
  const h1 = clean(document.querySelector("h1")?.textContent || "");
  const links = Array.from(document.querySelectorAll("a[href]"))
    .slice(0, 5)
    .map((a) => ({ text: clean(a.textContent), url: a.href || "" }));
  return {
    url: location.href,
    title: document.title || "",
    h1,
    links,
  };
});
return { ok: true, evidence };`,
  },
  {
    id: "list-extraction",
    title: "List Extraction",
    intent: "Extract ranked candidate items from current page without tab switching.",
    code:
`await page.waitForTimeout(150);
const items = await page.$$eval("a[href]", (anchors) => {
  const clean = (s) => String(s || "").replace(/\\s+/g, " ").trim();
  const out = [];
  const seen = new Set();
  for (const a of anchors) {
    const title = clean(a.textContent || a.innerText || "");
    const url = String(a.href || "").trim();
    if (!title || !url) continue;
    const key = title + "|" + url;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ title, url });
    if (out.length >= 10) break;
  }
  return out;
});
return {
  ok: items.length > 0,
  evidence: { count: items.length, items }
};`,
  },
  {
    id: "interaction-check",
    title: "Interaction Check",
    intent: "Verify a user interaction and return before/after evidence.",
    setupUrl: "https://the-internet.herokuapp.com/dropdown",
    code:
`const before = await page.locator("#dropdown").inputValue();
await page.locator("#dropdown").selectOption("2");
const after = await page.locator("#dropdown").inputValue();
return {
  ok: after === "2",
  evidence: {
    url: page.url(),
    before,
    after
  }
};`,
  },
  {
    id: "error-aware-step",
    title: "Error Aware Step",
    intent: "Use try/catch and return structured error evidence instead of throwing.",
    setupUrl: "https://the-internet.herokuapp.com/login",
    code:
`try {
  const userCount = await page.locator("#username").count();
  const passCount = await page.locator("#password").count();
  const submitCount = await page.getByRole("button", { name: /login/i }).count();
  return {
    ok: userCount > 0 && passCount > 0 && submitCount > 0,
    evidence: { url: page.url(), userCount, passCount, submitCount }
  };
} catch (err) {
  return {
    ok: false,
    evidence: {
      url: page.url(),
      error: String(err && err.message ? err.message : err)
    }
  };
}`,
  },
]);
