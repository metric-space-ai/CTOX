function asText(value) {
  return String(value == null ? "" : value).trim();
}

function cloneJson(value, fallback = null) {
  try {
    if (typeof globalThis.structuredClone === "function") {
      return globalThis.structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

function isPlainObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function parseJsonLoose(raw) {
  const text = asText(raw);
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fenced?.[1]) {
      try {
        return JSON.parse(fenced[1]);
      } catch {}
    }
    const objMatch = text.match(/\{[\s\S]*\}/);
    if (objMatch?.[0]) {
      try {
        return JSON.parse(objMatch[0]);
      } catch {}
    }
    return null;
  }
}

function toFiniteInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : fallback;
}

function getSchemaVariants(schema) {
  if (!schema || typeof schema !== "object") return [];
  const variants = [];
  if (Array.isArray(schema.anyOf)) variants.push(...schema.anyOf.filter(Boolean));
  if (Array.isArray(schema.oneOf)) variants.push(...schema.oneOf.filter(Boolean));
  if (Array.isArray(schema.type)) {
    for (const type of schema.type) {
      variants.push({ ...schema, type });
    }
  }
  return variants;
}

function schemaAllowsNull(schema) {
  if (!schema || typeof schema !== "object") return false;
  if (schema.type === "null") return true;
  if (Array.isArray(schema.type) && schema.type.includes("null")) return true;
  if (schema.const === null) return true;
  if (Array.isArray(schema.enum) && schema.enum.includes(null)) return true;
  return getSchemaVariants(schema).some((variant) => schemaAllowsNull(variant));
}

function matchesSchemaType(value, schema) {
  if (!schema || typeof schema !== "object") return true;
  if (schema.const !== undefined) return value === schema.const;
  if (Array.isArray(schema.enum) && schema.enum.length) return schema.enum.includes(value);
  if (schema.properties || schema.type === "object") return isPlainObject(value);
  if (schema.items || schema.type === "array") return Array.isArray(value);
  switch (schema.type) {
    case "string":
      return typeof value === "string";
    case "integer":
      return Number.isInteger(value);
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return true;
  }
}

function pickSchemaVariant(value, schema) {
  const variants = getSchemaVariants(schema);
  if (!variants.length) return schema;
  if (value == null) {
    return variants.find((variant) => schemaAllowsNull(variant)) || variants[0];
  }
  return (
    variants.find((variant) => matchesSchemaType(value, variant)) ||
    variants.find((variant) => !schemaAllowsNull(variant)) ||
    variants[0]
  );
}

function coerceString(value, schema) {
  let text = typeof value === "string" ? value : value == null ? "" : String(value);
  if (!text) {
    if (typeof schema?.const === "string") {
      text = schema.const;
    } else if (Array.isArray(schema?.enum) && schema.enum.length) {
      const preferred = schema.enum.find((entry) => typeof entry === "string");
      text = typeof preferred === "string" ? preferred : "";
    }
  }
  if (!text) {
    text = "Pending details";
  }
  const maxLength = Math.max(0, toFiniteInteger(schema?.maxLength));
  if (maxLength && text.length > maxLength) {
    text = text.slice(0, maxLength);
  }
  const minLength = Math.max(0, toFiniteInteger(schema?.minLength));
  if (minLength && text.length < minLength) {
    const fallback = "Pending details";
    if (
      fallback.length >= minLength &&
      (!maxLength || fallback.length <= maxLength)
    ) {
      text = fallback;
    }
    if (text.length < minLength) {
      text = text.padEnd(minLength, ".");
    }
    if (maxLength && text.length > maxLength) {
      text = text.slice(0, maxLength);
    }
  }
  return text;
}

function coerceNumber(value, schema, forceInteger = false) {
  let number = typeof value === "number" && Number.isFinite(value) ? value : Number(value);
  if (!Number.isFinite(number)) {
    number = Number.isFinite(Number(schema?.minimum))
      ? Number(schema.minimum)
      : Number.isFinite(Number(schema?.exclusiveMinimum))
        ? Number(schema.exclusiveMinimum) + 1
        : 0;
  }
  if (Number.isFinite(Number(schema?.minimum))) {
    number = Math.max(number, Number(schema.minimum));
  }
  if (Number.isFinite(Number(schema?.exclusiveMinimum))) {
    number = Math.max(number, Number(schema.exclusiveMinimum) + 1);
  }
  if (Number.isFinite(Number(schema?.maximum))) {
    number = Math.min(number, Number(schema.maximum));
  }
  if (Number.isFinite(Number(schema?.exclusiveMaximum))) {
    number = Math.min(number, Number(schema.exclusiveMaximum) - 1);
  }
  return forceInteger ? Math.trunc(number) : number;
}

export function coerceJsonSchemaValue(value, rawSchema) {
  const schema = pickSchemaVariant(value, rawSchema);
  if (!schema || typeof schema !== "object") {
    return cloneJson(value, value ?? null);
  }
  if (value == null && schemaAllowsNull(schema)) {
    return null;
  }
  if (schema.const !== undefined) {
    return cloneJson(schema.const, schema.const);
  }
  if (Array.isArray(schema.enum) && schema.enum.length) {
    if (schema.enum.includes(value)) return cloneJson(value, value);
    const preferred = schema.enum.find((entry) => entry !== null);
    return cloneJson(preferred, preferred ?? null);
  }
  if (schema.properties || schema.type === "object") {
    const source = isPlainObject(value) ? value : {};
    const properties = isPlainObject(schema.properties) ? schema.properties : {};
    const required = new Set(Array.isArray(schema.required) ? schema.required : []);
    const output = {};
    for (const [key, propertySchema] of Object.entries(properties)) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        output[key] = coerceJsonSchemaValue(source[key], propertySchema);
      } else if (required.has(key)) {
        output[key] = coerceJsonSchemaValue(undefined, propertySchema);
      }
    }
    if (schema.additionalProperties !== false) {
      const additionalSchema = isPlainObject(schema.additionalProperties)
        ? schema.additionalProperties
        : null;
      for (const [key, entryValue] of Object.entries(source)) {
        if (Object.prototype.hasOwnProperty.call(properties, key)) continue;
        output[key] = additionalSchema
          ? coerceJsonSchemaValue(entryValue, additionalSchema)
          : cloneJson(entryValue, entryValue);
      }
    }
    return output;
  }
  if (schema.items || schema.type === "array") {
    const itemsSchema = Array.isArray(schema.items) ? schema.items[0] : schema.items;
    const maxItems = Math.max(0, toFiniteInteger(schema.maxItems));
    const minItems = Math.max(0, toFiniteInteger(schema.minItems));
    const source = Array.isArray(value) ? value : [];
    const output = source.map((entry) => coerceJsonSchemaValue(entry, itemsSchema));
    if (maxItems) {
      output.splice(maxItems);
    }
    while (itemsSchema && output.length < minItems) {
      output.push(coerceJsonSchemaValue(undefined, itemsSchema));
    }
    return output;
  }
  switch (schema.type) {
    case "string":
      return coerceString(value, schema);
    case "integer":
      return coerceNumber(value, schema, true);
    case "number":
      return coerceNumber(value, schema, false);
    case "boolean":
      return typeof value === "boolean" ? value : Boolean(value);
    case "null":
      return null;
    default:
      if (value == null && schemaAllowsNull(schema)) return null;
      return cloneJson(value, value ?? null);
  }
}

const STRUCTURED_OUTPUT_WRAPPER_KEYS = [
  "output",
  "final",
  "text",
  "result",
  "data",
  "response",
  "payload",
  "object",
  "plan",
];

const STRUCTURED_OUTPUT_CONTROL_KEYS = new Set([
  "kind",
  "type",
  "mode",
  ...STRUCTURED_OUTPUT_WRAPPER_KEYS,
]);

function looksLikeWrapperObject(value) {
  if (!isPlainObject(value)) return false;
  const keys = Object.keys(value);
  if (!keys.length) return false;
  const nonWrapperKeys = keys.filter((key) => !STRUCTURED_OUTPUT_CONTROL_KEYS.has(key));
  return (
    nonWrapperKeys.length === 0 &&
    keys.some((key) => STRUCTURED_OUTPUT_WRAPPER_KEYS.includes(key))
  );
}

export function extractStructuredOutputObject(rawValue, seen = new Set()) {
  const source =
    typeof rawValue === "string"
      ? parseJsonLoose(rawValue)
      : rawValue;
  if (!isPlainObject(source)) return null;
  if (seen.has(source)) return null;
  seen.add(source);

  for (const key of STRUCTURED_OUTPUT_WRAPPER_KEYS) {
    const nested = extractStructuredOutputObject(source[key], seen);
    if (nested) return nested;
  }
  if (!looksLikeWrapperObject(source)) {
    return source;
  }
  for (const nestedValue of Object.values(source)) {
    const nested = extractStructuredOutputObject(nestedValue, seen);
    if (nested) return nested;
  }
  return null;
}

export function repairStructuredOutputText(text, schema = null) {
  const candidate = extractStructuredOutputObject(text);
  if (!candidate) return asText(text);
  const repaired = schema && typeof schema === "object"
    ? coerceJsonSchemaValue(candidate, schema)
    : candidate;
  try {
    return JSON.stringify(repaired);
  } catch {
    return asText(text);
  }
}
