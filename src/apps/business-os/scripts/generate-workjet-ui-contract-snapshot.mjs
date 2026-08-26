import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const CONTRACT_DIR = resolve(dirname(SCRIPT_PATH), "../ui-contract/v1");
const SOURCE_PATH = resolve(CONTRACT_DIR, "workjet-ui-contract.source.json");
const JSON_PATH = resolve(CONTRACT_DIR, "workjet-ui-contract.json");
const CSS_PATH = resolve(CONTRACT_DIR, "workjet-ui-contract.css");
const GENERATOR_PATH = "src/apps/business-os/scripts/generate-workjet-ui-contract-snapshot.mjs";

const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

export const canonicalJson = (value) => {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (!isRecord(value)) return JSON.stringify(value);
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
};

export const provenanceHash = (value) => createHash("sha256").update(canonicalJson(value)).digest("hex");

const assertSourceSnapshot = (source) => {
  if (!isRecord(source) || source.schema !== "workjet.ui.contract.snapshot-source.v1") {
    throw new Error(`Invalid Workjet UI source snapshot schema in ${SOURCE_PATH}.`);
  }
  if (!isRecord(source.payload)) {
    throw new Error(`Workjet UI source snapshot has no payload in ${SOURCE_PATH}.`);
  }

  const actualHash = provenanceHash(source.payload);
  if (source.sourceSha256 !== actualHash) {
    throw new Error(
      `Workjet UI source provenance mismatch: expected ${source.sourceSha256}, calculated ${actualHash}.`,
    );
  }
  if (source.payload.schema !== "workjet.ui.contract.v1" || source.payload.version !== 1) {
    throw new Error("Workjet UI source payload is not the v1 contract.");
  }
  if (typeof source.sourcePath !== "string" || source.sourcePath.length === 0) {
    throw new Error("Workjet UI source snapshot is missing sourcePath provenance.");
  }
};

export const readSourceSnapshot = async (sourcePath = SOURCE_PATH) => {
  const source = JSON.parse(await readFile(sourcePath, "utf8"));
  assertSourceSnapshot(source);
  return source;
};

const cssSlug = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
const cssPixels = (value) => (value === 0 ? "0" : `${value}px`);

const themeCss = (selector, themeName, theme) => [
  `${selector} {`,
  `  --workjet-theme: ${themeName};`,
  `  --workjet-surface-canvas: ${theme.surfaces.canvas};`,
  `  --workjet-surface-chrome: ${theme.surfaces.chrome};`,
  `  --workjet-surface: ${theme.surfaces.surface};`,
  `  --workjet-surface-raised: ${theme.surfaces.raised};`,
  `  --workjet-surface-overlay: ${theme.surfaces.overlay};`,
  `  --workjet-surface-sunken: ${theme.surfaces.sunken};`,
  `  --workjet-text-primary: ${theme.text.primary};`,
  `  --workjet-text-secondary: ${theme.text.secondary};`,
  `  --workjet-text-muted: ${theme.text.muted};`,
  `  --workjet-text-on-accent: ${theme.text.onAccent};`,
  `  --workjet-border-subtle: ${theme.borders.subtle};`,
  `  --workjet-border-default: ${theme.borders.default};`,
  `  --workjet-border-strong: ${theme.borders.strong};`,
  `  --workjet-accent: ${theme.accent.value};`,
  `  --workjet-accent-foreground: ${theme.accent.foreground};`,
  `  --workjet-accent-soft: ${theme.accent.soft};`,
  `  --workjet-focus: ${theme.accent.focus};`,
  "}",
].join("\n");

const sharedCss = (payload) => {
  const lines = [
    "  --workjet-font-sans: " + payload.typography.fontFamily.sans + ";",
    "  --workjet-font-mono: " + payload.typography.fontFamily.mono + ";",
  ];

  for (const [name, step] of Object.entries(payload.typography.scale)) {
    lines.push(`  --workjet-font-size-${name}: ${cssPixels(step.fontSize)};`);
    lines.push(`  --workjet-line-height-${name}: ${cssPixels(step.lineHeight)};`);
  }
  for (const [name, weight] of Object.entries(payload.typography.weights)) {
    lines.push(`  --workjet-font-weight-${name}: ${weight};`);
  }
  for (const [name, tracking] of Object.entries(payload.typography.tracking)) {
    lines.push(`  --workjet-tracking-${name}: ${tracking};`);
  }
  for (const [name, value] of Object.entries(payload.spacing)) {
    lines.push(`  --workjet-space-${name}: ${cssPixels(value)};`);
  }
  for (const [name, value] of Object.entries(payload.radii)) {
    lines.push(`  --workjet-radius-${name}: ${cssPixels(value)};`);
  }

  lines.push(`  --workjet-focus-width: ${cssPixels(payload.focus.outlineWidth)};`);
  lines.push(`  --workjet-focus-offset: ${cssPixels(payload.focus.outlineOffset)};`);
  lines.push(`  --workjet-focus-ring-alpha: ${payload.focus.ringAlpha};`);
  lines.push(`  --workjet-focus-min-contrast: ${payload.focus.minContrastRatio};`);
  for (const [name, shadow] of Object.entries(payload.elevation)) {
    lines.push(`  --workjet-shadow-${name}: ${shadow};`);
  }

  for (const [category, accent] of Object.entries(payload.categories)) {
    const slug = cssSlug(category);
    lines.push(`  --workjet-category-${slug}-accent: ${accent.accent};`);
    lines.push(`  --workjet-category-${slug}-accent-foreground: ${accent.foreground};`);
  }
  return lines.join("\n");
};

export const renderSnapshotCss = (payload, sourceSha256) => [
  `/* Workjet UI contract v${payload.version}; provenance-sha256: ${sourceSha256}; generated-by: ${GENERATOR_PATH} */`,
  ":root,",
  ':root[data-theme="light"] {',
  sharedCss(payload),
  "}",
  themeCss(':root, :root[data-theme="light"]', "light", payload.themes.light),
  themeCss(':root[data-theme="dark"]', "dark", payload.themes.dark),
  "",
].join("\n");

export const renderSnapshotJson = (source) => JSON.stringify(
  {
    ...source.payload,
    provenance: {
      sourcePath: source.sourcePath,
      sourceSha256: source.sourceSha256,
      generator: GENERATOR_PATH,
    },
  },
  null,
  2,
) + "\n";

export const expectedArtifacts = async (sourcePath = SOURCE_PATH) => {
  const source = await readSourceSnapshot(sourcePath);
  return {
    source,
    json: renderSnapshotJson(source),
    css: renderSnapshotCss(source.payload, source.sourceSha256),
  };
};

export const generateArtifacts = async ({ check = false } = {}) => {
  const artifacts = await expectedArtifacts();
  const current = await Promise.all([
    readFile(JSON_PATH, "utf8").catch(() => undefined),
    readFile(CSS_PATH, "utf8").catch(() => undefined),
  ]);
  const [currentJson, currentCss] = current;
  const mismatches = [];
  if (currentJson !== artifacts.json) mismatches.push(JSON_PATH);
  if (currentCss !== artifacts.css) mismatches.push(CSS_PATH);

  if (check) {
    if (mismatches.length > 0) {
      throw new Error(`Workjet UI snapshot drift detected:\n${mismatches.join("\n")}`);
    }
    return { ...artifacts, changed: false };
  }

  if (mismatches.includes(JSON_PATH)) await writeFile(JSON_PATH, artifacts.json, "utf8");
  if (mismatches.includes(CSS_PATH)) await writeFile(CSS_PATH, artifacts.css, "utf8");
  return { ...artifacts, changed: mismatches.length > 0 };
};

const main = async () => {
  const args = new Set(process.argv.slice(2));
  const unknown = [...args].filter((arg) => arg !== "--check");
  if (unknown.length > 0) throw new Error(`Unknown option: ${unknown.join(", ")}`);

  const result = await generateArtifacts({ check: args.has("--check") });
  if (args.has("--check")) {
    console.log(`workjet_ui_contract_snapshot ok sha256=${result.source.sourceSha256}`);
  } else {
    console.log(`workjet_ui_contract_snapshot generated sha256=${result.source.sourceSha256}`);
  }
};

if (process.argv[1] && resolve(process.argv[1]) === SCRIPT_PATH) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
