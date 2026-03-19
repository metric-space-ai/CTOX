function asText(value) {
  return String(value == null ? "" : value).trim();
}

function normalizeArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeToolName(value, fallback = "") {
  const normalized = asText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || asText(fallback);
}

function indentBlock(source, prefix = "  ") {
  return String(source || "")
    .split("\n")
    .map((line) => `${prefix}${line}`)
    .join("\n");
}

function isSafeJsIdentifier(value) {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(asText(value));
}

function canUseBareIdentifierReference(value) {
  const identifier = asText(value);
  if (!isSafeJsIdentifier(identifier)) return false;
  return !new Set([
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "null",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
  ]).has(identifier);
}

function getToolBindingNames(entry = {}) {
  const names = [
    normalizeToolName(entry?.id),
    asText(entry?.id),
    normalizeToolName(entry?.name),
  ];
  return Array.from(new Set(names.filter(Boolean))).slice(0, 8);
}

function getToolResolverCandidateNames(entry = {}, toolScriptsPayload = null) {
  return Array.from(new Set([
    ...getToolBindingNames(entry),
    ...normalizeArray(toolScriptsPayload?.declaredTools).flatMap((value) => {
      const raw = asText(value);
      const normalized = normalizeToolName(value);
      return [raw, normalized];
    }),
    "main",
    "default",
  ].filter(Boolean))).slice(0, 32);
}

function pushNamedExport(exportsList, localName = "", exportedName = "") {
  const normalizedLocalName = asText(localName);
  const normalizedExportedName = asText(exportedName || localName);
  if (!normalizedLocalName || !normalizedExportedName) return;
  if (!isSafeJsIdentifier(normalizedLocalName)) return;
  if (!isSafeJsIdentifier(normalizedExportedName)) return;
  if (
    exportsList.some((entry) =>
      entry.localName === normalizedLocalName && entry.exportedName === normalizedExportedName
    )
  ) {
    return;
  }
  exportsList.push({
    localName: normalizedLocalName,
    exportedName: normalizedExportedName,
  });
}

function parseNamedExportSpecifiers(rawSpecifiers = "", exportsList = []) {
  normalizeArray(String(rawSpecifiers || "").split(","))
    .map((entry) => asText(entry))
    .filter(Boolean)
    .forEach((entry) => {
      const aliasMatch = entry.match(
        /^([A-Za-z_$][A-Za-z0-9_$]*)\s+as\s+([A-Za-z_$][A-Za-z0-9_$]*)$/i,
      );
      if (aliasMatch) {
        pushNamedExport(exportsList, aliasMatch[1], aliasMatch[2]);
        return;
      }
      pushNamedExport(exportsList, entry, entry);
    });
}

function transformReviewedToolSource(source = "") {
  const original = String(source || "").trim();
  if (!/^\s*export\b/m.test(original)) {
    return original;
  }

  const namedExports = [];
  let transformed = original;
  let hasDefaultExport = false;

  transformed = transformed.replace(
    /^\s*export\s+default\s+/gm,
    () => {
      hasDefaultExport = true;
      return "const __reviewedToolDefaultExport = ";
    },
  );

  transformed = transformed.replace(
    /^\s*export\s+(async\s+function|function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm,
    (match, declarationKind, exportedName) => {
      pushNamedExport(namedExports, exportedName, exportedName);
      return `${declarationKind} ${exportedName}`;
    },
  );

  transformed = transformed.replace(
    /^\s*export\s+(const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b/gm,
    (match, declarationKind, exportedName) => {
      pushNamedExport(namedExports, exportedName, exportedName);
      return `${declarationKind} ${exportedName}`;
    },
  );

  transformed = transformed.replace(
    /^\s*export\s*\{([^}]*)\}\s*;?\s*$/gm,
    (match, specifiers) => {
      parseNamedExportSpecifiers(specifiers, namedExports);
      return "";
    },
  );

  const exportAssignments = namedExports.flatMap((entry) => [
    `module.exports[${JSON.stringify(entry.exportedName)}] = ${entry.localName};`,
    `exports[${JSON.stringify(entry.exportedName)}] = ${entry.localName};`,
  ]);
  if (hasDefaultExport) {
    exportAssignments.push(
      "module.exports.default = __reviewedToolDefaultExport;",
      "exports.default = __reviewedToolDefaultExport;",
    );
  }
  if (!exportAssignments.length) {
    return transformed;
  }
  return [
    transformed,
    ...exportAssignments,
  ].filter((entry) => asText(entry)).join("\n");
}

function buildEntrypointResolverSource(entrypoint = "main", candidateNames = []) {
  const normalizedEntrypoint = asText(entrypoint) || "main";
  const encodedEntrypoint = JSON.stringify(normalizedEntrypoint);
  const prioritizedCandidateNames = Array.from(new Set([
    normalizedEntrypoint,
    ...normalizeArray(candidateNames),
  ].map((value) => asText(value)).filter(Boolean))).slice(0, 32);

  const callableRegistrationLines = prioritizedCandidateNames.flatMap((candidateName) => {
    const encodedName = JSON.stringify(candidateName);
    const safeCandidate = canUseBareIdentifierReference(candidateName) ? candidateName : "";
    return [
      `  __reviewedToolMaybeRegisterCallable(${encodedName}, __reviewedToolExport && typeof __reviewedToolExport[${encodedName}] === 'function' ? __reviewedToolExport[${encodedName}] : null, __reviewedToolExport);`,
      `  __reviewedToolMaybeRegisterCallable(${encodedName}, __reviewedToolExports && typeof __reviewedToolExports[${encodedName}] === 'function' ? __reviewedToolExports[${encodedName}] : null, __reviewedToolExports);`,
      safeCandidate
        ? `  __reviewedToolMaybeRegisterCallable(${encodedName}, typeof ${safeCandidate} === 'function' ? ${safeCandidate} : null);`
        : "",
      `  __reviewedToolMaybeRegisterCallable(${encodedName}, typeof globalThis[${encodedName}] === 'function' ? globalThis[${encodedName}] : null);`,
    ].filter(Boolean);
  });

  const prioritizedCallableLookups = prioritizedCandidateNames
    .map((candidateName) => `__reviewedToolCallableExports[${JSON.stringify(candidateName)}] || null`);

  return [
    "  const __reviewedToolExport = module.exports;",
    "  const __reviewedToolExports = exports;",
    "  const __reviewedToolCallableExports = Object.create(null);",
    "  const __reviewedToolMaybeRegisterCallable = (name, value, owner = null) => {",
    "    const key = String(name || '').trim();",
    "    if (!key || typeof value !== 'function' || typeof __reviewedToolCallableExports[key] === 'function') return null;",
    "    const boundValue = owner && owner !== globalThis ? value.bind(owner) : value;",
    "    __reviewedToolCallableExports[key] = boundValue;",
    "    return boundValue;",
    "  };",
    "  if (__reviewedToolExport && typeof __reviewedToolExport === 'object') {",
    "    for (const [key, value] of Object.entries(__reviewedToolExport)) {",
    "      __reviewedToolMaybeRegisterCallable(key, value, __reviewedToolExport);",
    "    }",
    "  }",
    "  if (__reviewedToolExports && typeof __reviewedToolExports === 'object') {",
    "    for (const [key, value] of Object.entries(__reviewedToolExports)) {",
    "      __reviewedToolMaybeRegisterCallable(key, value, __reviewedToolExports);",
    "    }",
    "  }",
    ...callableRegistrationLines,
    "  const __reviewedToolResolvedCandidates = [",
    "    typeof __reviewedToolExport === 'function' ? __reviewedToolExport : null,",
    "    typeof __reviewedToolExports === 'function' ? __reviewedToolExports : null,",
    ...prioritizedCallableLookups.map((line) => `    ${line},`),
    "    ...Object.values(__reviewedToolCallableExports),",
    "  ].filter((value, index, list) => typeof value === 'function' && list.indexOf(value) === index);",
    "  const __reviewedToolResolved = __reviewedToolResolvedCandidates[0] || null;",
  ].join("\n");
}

export function buildDefaultReviewedToolExecuteScript(toolName = "") {
  const normalizedToolName = normalizeToolName(toolName, toolName);
  if (!normalizedToolName) return "";
  return `return await __callReviewedTool(${JSON.stringify(normalizedToolName)}, args);`;
}

export function buildReviewedToolScriptsPrelude(toolScriptsPayload = null) {
  return normalizeArray(toolScriptsPayload?.scripts)
    .filter((entry) => {
      const source = asText(entry?.source);
      const language = asText(entry?.language).toLowerCase();
      return source && (!language || language === "javascript" || language === "js" || language === "mjs");
    })
    .map((entry, index) => {
      const label = asText(entry?.name || entry?.id) || `tool_script_${index + 1}`;
      const bindingVar = `__reviewedToolBinding${index + 1}`;
      const entrypoint = asText(entry?.entrypoint) || "main";
      const bindingNames = getToolBindingNames(entry);
      const candidateNames = getToolResolverCandidateNames(entry, toolScriptsPayload);
      const registrations = bindingNames.map((bindingName) =>
        `__registerReviewedTool(${JSON.stringify(bindingName)}, ${bindingVar});`
      );
      return [
        `// Reviewed tool script: ${label}`,
        `const ${bindingVar} = (() => {`,
        "  const module = { exports: {} };",
        "  const exports = module.exports;",
        indentBlock(transformReviewedToolSource(entry.source), "  "),
        buildEntrypointResolverSource(entrypoint, candidateNames),
        "  for (const [toolName, handler] of Object.entries(__reviewedToolCallableExports)) {",
        "    __registerReviewedTool(toolName, handler);",
        "  }",
        "  if (typeof __reviewedToolResolved !== 'function') {",
        `    throw new Error(${JSON.stringify(`Reviewed tool script "${label}" does not expose a callable entrypoint "${entrypoint}".`)});`,
        "  }",
        "  return __reviewedToolResolved;",
        "})();",
        ...registrations,
      ].join("\n");
    })
    .join("\n\n");
}
