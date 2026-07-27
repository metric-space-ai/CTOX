import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  parseGitHubUrl,
  shouldSkipPath,
  isTextFile,
  isImportableFile,
  validModuleId,
} from "./index.js";

test("parseGitHubUrl handles repo, tree refs and subdirs", () => {
  assert.deepEqual(parseGitHubUrl("https://github.com/acme/po-tracker"), {
    owner: "acme", repo: "po-tracker", ref: null, subdir: "",
  });
  assert.deepEqual(parseGitHubUrl("https://github.com/acme/po-tracker.git"), {
    owner: "acme", repo: "po-tracker", ref: null, subdir: "",
  });
  assert.deepEqual(parseGitHubUrl("https://github.com/acme/mono/tree/main/apps/tracker"), {
    owner: "acme", repo: "mono", ref: "main", subdir: "apps/tracker",
  });
  assert.equal(parseGitHubUrl("https://gitlab.com/acme/x"), null);
  assert.equal(parseGitHubUrl("not a url"), null);
  assert.equal(parseGitHubUrl("https://github.com/onlyowner"), null);
});

test("shouldSkipPath drops build artifacts and vcs noise", () => {
  for (const path of [
    "node_modules/react/index.js",
    "src/node_modules/x.js",
    ".git/HEAD",
    "dist/bundle.js",
    "build/main.js",
    ".next/app.js",
    "yarn.lock",
    "package-lock.json",
    "src/main.js.map",
  ]) {
    assert.equal(shouldSkipPath(path), true, path);
  }
  for (const path of ["src/main.tsx", "src/App.tsx", "index.html", "src/lib/format.ts"]) {
    assert.equal(shouldSkipPath(path), false, path);
  }
});

test("isTextFile allows source and asset text, rejects binaries", () => {
  assert.equal(isTextFile("src/main.tsx"), true);
  assert.equal(isTextFile("styles/app.css"), true);
  assert.equal(isTextFile("logo.svg"), true);
  assert.equal(isTextFile("logo.png"), false);
  assert.equal(isTextFile("font.woff2"), false);
  assert.equal(isTextFile("Makefile"), false);
});

test("isImportableFile keeps browser image and font assets", () => {
  assert.equal(isImportableFile("src/assets/hero.png"), true);
  assert.equal(isImportableFile("src/fonts/ui.woff2"), true);
  assert.equal(isImportableFile("archive.zip"), false);
});

test("validModuleId enforces the launcher slug contract", () => {
  assert.equal(validModuleId("po-tracker"), true);
  assert.equal(validModuleId("a1"), true);
  assert.equal(validModuleId("-bad"), false);
  assert.equal(validModuleId("Bad"), false);
  assert.equal(validModuleId("x"), false);
  assert.equal(validModuleId("with space"), false);
});

test("presentation is a focused windowed import flow", async () => {
  const html = await readFile(new URL("./index.html", import.meta.url), "utf8");
  const css = await readFile(new URL("./index.css", import.meta.url), "utf8");
  const js = await readFile(new URL("./index.js", import.meta.url), "utf8");
  const manifest = JSON.parse(await readFile(new URL("./module.json", import.meta.url), "utf8"));
  const de = JSON.parse(await readFile(new URL("./locales/de.json", import.meta.url), "utf8"));
  const en = JSON.parse(await readFile(new URL("./locales/en.json", import.meta.url), "utf8"));

  assert.match(html, /class="imp-rail"/);
  assert.match(html, /data-imp-step="source"/);
  assert.match(html, /data-imp-step="review"/);
  assert.match(html, /data-imp-step="done"/);
  assert.match(html, /data-imp-open/);
  assert.doesNotMatch(html, /class="ctox-card"/);

  assert.match(css, /\.importer-module\s*\{[^}]*grid-template-columns:\s*230px/);
  assert.match(css, /\.imp-stage\s*\{[^}]*width:\s*min\(100%,\s*650px\)/);
  assert.match(css, /\.imp-done-stage/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);

  assert.match(js, /const setStage = \(stage\)/);
  assert.match(js, /setStage\('review'\)/);
  assert.match(js, /setStage\('done'\)/);
  assert.match(js, /globalThis\.location\.hash = moduleId/);
  assert.match(js, /globalThis\.location\.reload\(\)/);
  assert.match(js, /globalThis\.location\.reload\(\)/);

  assert.equal(manifest.layout.shell, "windowed");
  assert.deepEqual(manifest.presentation.initial_size, { width: 860, height: 600 });
  assert.deepEqual(manifest.presentation.minimum_size, { width: 640, height: 480 });

  for (const locale of [de, en]) {
    assert.equal(typeof locale.title, "string");
    assert.equal(typeof locale.sourceHeading, "string");
    assert.equal(typeof locale.reviewHeading, "string");
    assert.equal(typeof locale.doneHeading, "string");
    assert.equal(typeof locale.openApp, "string");
  }
});
