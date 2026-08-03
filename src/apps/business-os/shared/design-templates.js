const BUILTIN_SHELL_STYLES = Object.freeze([
  Object.freeze({ value: 'ctox', label: 'CTOX', shellStyle: 'ctox', templateId: '' }),
  Object.freeze({ value: 'windows', label: 'Windows', shellStyle: 'windows', templateId: '' }),
  Object.freeze({ value: 'macos', label: 'macOS', shellStyle: 'macos', templateId: '' }),
]);

const TEMPLATE_ID_RE = /^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$/;
const STYLESHEET_RE = /^design-templates\/[a-z0-9][a-z0-9._-]*\/[a-zA-Z0-9][a-zA-Z0-9._-]*\.css$/;

export function normalizeDesignTemplates(input = globalThis.CTOX_BUSINESS_OS_DESIGN_TEMPLATES) {
  if (!Array.isArray(input)) return [];
  const seen = new Set();
  const templates = [];
  for (const candidate of input) {
    const id = String(candidate?.id || '').trim().toLowerCase();
    const title = String(candidate?.title || '').trim();
    const baseStyle = normalizeBuiltinShellStyle(candidate?.base_style);
    const stylesheet = String(candidate?.stylesheet_href || '').trim();
    if (
      !TEMPLATE_ID_RE.test(id)
      || !title
      || title.length > 80
      || !STYLESHEET_RE.test(stylesheet)
      || seen.has(id)
    ) {
      continue;
    }
    seen.add(id);
    templates.push(Object.freeze({
      value: `custom:${id}`,
      id,
      label: title,
      description: String(candidate?.description || '').trim().slice(0, 240),
      shellStyle: baseStyle,
      stylesheet,
      templateId: id,
    }));
  }
  return templates.sort((left, right) => (
    left.label.localeCompare(right.label) || left.id.localeCompare(right.id)
  ));
}

export function shellDesignOptions(input = globalThis.CTOX_BUSINESS_OS_DESIGN_TEMPLATES) {
  return [...BUILTIN_SHELL_STYLES, ...normalizeDesignTemplates(input)];
}

export function resolveShellDesign(
  value,
  input = globalThis.CTOX_BUSINESS_OS_DESIGN_TEMPLATES,
) {
  const requested = String(value || '').trim().toLowerCase();
  return shellDesignOptions(input).find((option) => option.value === requested)
    || BUILTIN_SHELL_STYLES[0];
}

export function currentShellDesignValue(
  root = globalThis.document?.documentElement,
  input = globalThis.CTOX_BUSINESS_OS_DESIGN_TEMPLATES,
) {
  const templateId = String(root?.dataset?.designTemplate || '').trim().toLowerCase();
  if (templateId) {
    const customValue = `custom:${templateId}`;
    if (shellDesignOptions(input).some((option) => option.value === customValue)) {
      return customValue;
    }
  }
  return normalizeBuiltinShellStyle(root?.dataset?.shellStyle);
}

export function normalizeBuiltinShellStyle(value) {
  const style = String(value || '').trim().toLowerCase();
  return style === 'windows' || style === 'macos' ? style : 'ctox';
}

