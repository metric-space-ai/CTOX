/*
 * CTOX iframe shell around the pinned Easy Email React source.
 * Compiled locally by build-upstream.mjs; no runtime bare imports remain.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider } from '@arco-design/web-react';
import enUS from '@arco-design/web-react/es/locale/en-US';
import mjml from 'mjml-browser';
import { cloneDeep } from 'lodash';
import {
  AdvancedType,
  BasicType,
  BlockManager,
  JsonToMjml,
  getPageIdx,
} from 'easy-email-core';
import {
  ActiveTabKeys,
  EmailEditor,
  EmailEditorProvider,
  getBlockNodeByIdx,
  getPluginElement,
  useActiveTab,
  useBlock,
  useFocusIdx,
} from 'easy-email-editor';
import {
  AttributePanel,
  BlockLayer,
  InteractivePrompt,
  MergeTagBadgePrompt,
  SourceCodePanel,
} from 'easy-email-extensions';
import { DragIcon } from '@extensions/ShortcutToolbar/components/DragIcon';

import '@arco-design/web-react/dist/css/arco.css';
import '@extensions/index.scss';
import './frame.css';

type EmailTemplate = {
  subject: string;
  subTitle: string;
  content: any;
};

type BridgeMessage = {
  source: 'ctox-easy-email-parent';
  instanceId: string;
  type: string;
  requestId?: string;
  payload?: any;
};

const INSTANCE_ID = new URLSearchParams(window.location.search).get('instance') || 'default';
const PARENT_ORIGIN = window.location.origin;
const ALLOWED_THEME_TOKENS = new Set([
  '--bg', '--surface', '--surface-2', '--surface-3', '--line', '--text',
  '--text-strong', '--muted', '--accent', '--accent-soft',
  '--accent-foreground', '--danger', '--warning', '--success', '--focus-ring',
]);

const RICH_TEXT_TOOLBAR_THEME = `
  #easy-email-rich-text-bar {
    display: var(--ctox-rich-text-toolbar-display, block) !important;
    left: 12px !important;
    top: 10px !important;
    width: max-content !important;
    max-width: calc(100% - 24px) !important;
    padding: 4px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    border: 1px solid var(--line, ButtonBorder) !important;
    border-radius: 5px !important;
    color: var(--text, CanvasText) !important;
    background: var(--surface, Canvas) !important;
    box-shadow: 0 8px 24px color-mix(in srgb, var(--text, CanvasText) 14%, transparent) !important;
    scrollbar-width: thin;
  }

  #easy-email-rich-text-bar > div:first-child {
    display: none !important;
  }

  #Tools {
    width: max-content;
    min-width: 0;
    align-items: center;
  }

  #Tools > div:first-child {
    display: inline-flex;
    align-items: center;
    margin-right: 5px !important;
  }

  #Tools > div:first-child > span:first-child {
    margin: 0 4px 0 3px !important;
    color: var(--muted, CanvasText) !important;
    font-size: 11px;
    font-weight: 650;
  }

  .easy-email-extensions-emailToolItem {
    width: 26px !important;
    height: 26px !important;
    border-radius: 3px !important;
    color: var(--muted, CanvasText) !important;
  }

  .easy-email-extensions-emailToolItem:hover,
  .easy-email-extensions-emailToolItem-active {
    color: var(--text-strong, CanvasText) !important;
    background: var(--surface-2, Canvas) !important;
  }

  .easy-email-extensions-divider {
    height: 14px !important;
    margin: 0 2px;
    background: var(--line, ButtonBorder) !important;
  }
`;

function RichTextToolbarTheme() {
  useEffect(() => {
    let animationFrame = 0;
    let style: HTMLStyleElement | null = null;
    let cancelled = false;
    const install = () => {
      const pluginRoot = getPluginElement();
      if (!pluginRoot) {
        if (!cancelled) animationFrame = window.requestAnimationFrame(install);
        return;
      }
      style = document.createElement('style');
      style.dataset.ctoxRichTextToolbarTheme = 'true';
      style.textContent = RICH_TEXT_TOOLBAR_THEME;
      pluginRoot.appendChild(style);
    };
    install();
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(animationFrame);
      style?.remove();
    };
  }, []);
  return null;
}

function applyBusinessOsTheme(payload: any) {
  const theme = payload?.theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  document.documentElement.setAttribute('arco-theme', theme === 'dark' ? 'dark' : 'light');
  const tokens = payload?.tokens && typeof payload.tokens === 'object' ? payload.tokens : {};
  for (const [name, value] of Object.entries(tokens)) {
    if (!ALLOWED_THEME_TOKENS.has(name) || typeof value !== 'string' || !value.trim()) continue;
    document.documentElement.style.setProperty(name, value.trim());
  }
}

function send(type: string, payload?: any, requestId?: string) {
  window.parent.postMessage(
    { source: 'ctox-easy-email-frame', instanceId: INSTANCE_ID, type, payload, requestId },
    PARENT_ORIGIN,
  );
}

function createDefaultTemplate(): EmailTemplate {
  const page = BlockManager.getBlockByType(BasicType.PAGE)?.create({});
  if (!page) throw new Error('Easy Email page block is unavailable');
  return { subject: '', subTitle: '', content: page };
}

function isEmailTemplate(value: any): value is EmailTemplate {
  return Boolean(value && typeof value === 'object' && value.content?.type === BasicType.PAGE);
}

function resolveLogicValue(data: any, path: string) {
  return String(path || '').split('.').filter(Boolean).reduce((value, key) => value?.[key], data);
}

function logicValueType(rule: any) {
  if (['string', 'number', 'boolean', 'null'].includes(rule?.valueType)) return rule.valueType;
  if (rule?.value === null) return 'null';
  if (typeof rule?.value === 'number') return 'number';
  if (typeof rule?.value === 'boolean') return 'boolean';
  return 'string';
}

function coerceLogicValue(value: any, type: string) {
  if (type === 'null') return value == null ? null : value;
  if (type === 'number') {
    const number = Number(value);
    return Number.isFinite(number) ? number : Number.NaN;
  }
  if (type === 'boolean') {
    if (value === true || value === 'true' || value === 1 || value === '1') return true;
    if (value === false || value === 'false' || value === 0 || value === '0') return false;
    return Boolean(value);
  }
  return String(value ?? '');
}

function evaluateRule(rule: any, data: any): boolean {
  const actual = resolveLogicValue(data, rule.field);
  const type = logicValueType(rule);
  const expected = coerceLogicValue(rule.value, type);
  const actualTyped = coerceLogicValue(actual, type);
  switch (rule.operator) {
    case 'equals': return Object.is(actualTyped, expected);
    case 'not-equals': return !Object.is(actualTyped, expected);
    case 'contains': return Array.isArray(actual)
      ? actual.some((entry) => Object.is(coerceLogicValue(entry, type), expected))
      : String(actual ?? '').includes(String(expected ?? ''));
    case 'not-contains': return Array.isArray(actual)
      ? !actual.some((entry) => Object.is(coerceLogicValue(entry, type), expected))
      : !String(actual ?? '').includes(String(expected ?? ''));
    case 'exists': return actual !== undefined && actual !== null;
    case 'empty': return actual === undefined || actual === null || actual === '' || (Array.isArray(actual) && actual.length === 0);
    case 'greater': return !(typeof actualTyped === 'number' && Number.isNaN(actualTyped)) && actualTyped > expected;
    case 'less': return !(typeof actualTyped === 'number' && Number.isNaN(actualTyped)) && actualTyped < expected;
    default: return false;
  }
}

function evaluateLogicNode(node: any, data: any): boolean {
  if (!node) return true;
  if (node.kind === 'rule') return evaluateRule(node, data);
  const results = (node.children || []).map((child: any) => evaluateLogicNode(child, data));
  return node.combinator === 'or' ? results.some(Boolean) : results.every(Boolean);
}

function filterDocumentForPreview(template: EmailTemplate, mergeTags: any): EmailTemplate {
  const copy = cloneDeep(template);
  const visit = (block: any) => {
    if (!Array.isArray(block?.children)) return;
    block.children = block.children.filter((child: any) => {
      const logic = child?.data?.value?.logic;
      if (logic?.version === 1 && logic.root) {
        return evaluateLogicNode(logic.root, mergeLogicData(logic.testData, mergeTags));
      }
      return true;
    });
    block.children.forEach(visit);
  };
  visit(copy.content);
  return copy;
}

function mergeLogicData(base: any, override: any) {
  const output = cloneDeep(base && typeof base === 'object' ? base : {});
  const mergeInto = (target: any, source: any) => {
    if (!source || typeof source !== 'object' || Array.isArray(source)) return target;
    for (const [key, value] of Object.entries(source)) {
      if (key === '__proto__' || key === 'prototype' || key === 'constructor') continue;
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const existing = target[key];
        target[key] = mergeInto(existing && typeof existing === 'object' && !Array.isArray(existing) ? existing : {}, value);
      } else {
        target[key] = cloneDeep(value);
      }
    }
    return target;
  };
  return mergeInto(output, override);
}

type LogicPreview = {
  blockId?: string;
  matched?: boolean;
  testData?: Record<string, any>;
  logic?: any;
} | null;

function BlockLibrary() {
  const blocks = [
    { type: AdvancedType.TEXT, label: 'Text', hint: 'Überschrift oder Absatz', payload: { attributes: { padding: '0px 25px', align: 'left' } } },
    { type: AdvancedType.IMAGE, label: 'Bild', hint: 'Foto oder Grafik', payload: { attributes: { padding: '0px' } } },
    { type: AdvancedType.BUTTON, label: 'Button', hint: 'Handlungslink' },
    { type: AdvancedType.SOCIAL, label: 'Social', hint: 'Soziale Links' },
    { type: AdvancedType.NAVBAR, label: 'Navigation', hint: 'Link-Leiste' },
    { type: AdvancedType.DIVIDER, label: 'Trennlinie', hint: 'Inhalte gliedern' },
    { type: AdvancedType.SPACER, label: 'Abstand', hint: 'Freiraum setzen' },
    {
      type: AdvancedType.SECTION,
      label: '2 Spalten',
      hint: 'Zweispaltiger Abschnitt',
      payload: {
        children: [0, 1].map(() => ({
          type: AdvancedType.COLUMN,
          data: { value: {} },
          attributes: { padding: '0px', border: 'none', 'vertical-align': 'top' },
          children: [],
        })),
      },
    },
  ];
  return (
    <div className='ctox-frame-block-library'>
      <p>Baustein in die E-Mail ziehen</p>
      <div className='ctox-frame-block-grid'>
        {blocks.map((block) => (
          <div className='ctox-frame-block-item' key={block.label} title={`${block.label}: ${block.hint}`}>
            <DragIcon type={block.type} color='var(--muted)' payload={block.payload || {}} />
            <span><strong>{block.label}</strong><small>{block.hint}</small></span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DesignPanel() {
  const [section, setSection] = useState<'element' | 'structure'>('element');
  const { activeTab, setActiveTab } = useActiveTab();
  const { undo, redo, undoable, redoable } = useBlock();
  return (
    <div className='ctox-frame-design-panel'>
      <label className='ctox-frame-view-select'>
        <span>Canvas</span>
        <select value={activeTab} onChange={(event) => setActiveTab(event.target.value as ActiveTabKeys)}>
          <option value={ActiveTabKeys.EDIT}>Bearbeiten</option>
          <option value={ActiveTabKeys.PC}>Desktop-Vorschau</option>
          <option value={ActiveTabKeys.MOBILE}>Mobil-Vorschau</option>
        </select>
      </label>
      <div className='ctox-frame-history' role='group' aria-label='Verlauf'>
        <button type='button' disabled={!undoable} onClick={undo}>Rückgängig</button>
        <button type='button' disabled={!redoable} onClick={redo}>Wiederholen</button>
      </div>
      <div className='ctox-frame-segmented' role='tablist' aria-label='Designbereich'>
        <button type='button' role='tab' aria-selected={section === 'element'} onClick={() => setSection('element')}>Element</button>
        <button type='button' role='tab' aria-selected={section === 'structure'} onClick={() => setSection('structure')}>Struktur</button>
      </div>
      <section hidden={section !== 'element'}><AttributePanel /></section>
      <section hidden={section !== 'structure'}><BlockLayer /></section>
    </div>
  );
}

function PanelDrawer({ panel, onClose }: { panel: string | null; onClose(): void }) {
  const visible = panel === 'blocks' || panel === 'design' || panel === 'source';
  const title = panel === 'blocks' ? 'Bausteine' : panel === 'design' ? 'Design' : 'HTML & MJML';
  return (
    <aside
      className={`ctox-frame-drawer${visible ? '' : ' is-closed'}`}
      aria-hidden={!visible}
      aria-label={panel || 'Editor-Werkzeuge'}
    >
      <header className='ctox-frame-drawer-header'>
        <strong>{title}</strong>
        <button type='button' className='ctox-frame-icon-button' onClick={onClose} aria-label='Schließen' title='Schließen'>×</button>
      </header>
      <div className='ctox-frame-drawer-body'>
        <section hidden={panel !== 'blocks'}><BlockLibrary /></section>
        {/* AttributePanel stays mounted because RichTextField owns the
            DOM-to-form synchronization for the selected block. */}
        <section hidden={panel !== 'design'}><DesignPanel /></section>
        <section hidden={panel !== 'source'}>
          <SourceCodePanel jsonReadOnly={false} mjmlReadOnly={false} />
        </section>
      </div>
    </aside>
  );
}

function LogicPreviewEffect({ preview, values }: { preview: LogicPreview; values: EmailTemplate }) {
  useEffect(() => {
    if (!preview?.blockId) return;
    let node: HTMLElement | null = null;
    const animationFrame = window.requestAnimationFrame(() => {
      node = getBlockNodeByIdx(preview.blockId!);
      if (!node) return;
      node.dataset.ctoxLogicPreview = preview.matched === false ? 'excluded' : 'included';
      if (preview.matched === false) {
        const warning = getComputedStyle(document.documentElement).getPropertyValue('--warning').trim() || 'currentColor';
        node.style.setProperty('opacity', '0.28', 'important');
        node.style.setProperty('filter', 'grayscale(0.85)', 'important');
        node.style.setProperty('outline', `2px dashed ${warning}`, 'important');
        node.style.setProperty('outline-offset', '2px', 'important');
      }
    });
    return () => {
      window.cancelAnimationFrame(animationFrame);
      if (!node) return;
      delete node.dataset.ctoxLogicPreview;
      node.style.removeProperty('opacity');
      node.style.removeProperty('filter');
      node.style.removeProperty('outline');
      node.style.removeProperty('outline-offset');
    };
  }, [preview, values]);

  if (!preview?.blockId) return null;
  return (
    <output
      className={`ctox-logic-preview${preview.matched === false ? ' is-excluded' : ' is-included'}`}
      aria-live='polite'
    >
      <span aria-hidden='true'>{preview.matched === false ? '○' : '✓'}</span>
      {preview.matched === false ? 'Logik: Block wird ausgeblendet' : 'Logik: Block wird angezeigt'}
    </output>
  );
}

function RuntimeController({
  form,
  values,
  mergeTags,
  setMergeTags,
  panel,
  setPanel,
  logicPreview,
  setLogicPreview,
}: any) {
  const { focusIdx, setFocusIdx } = useFocusIdx();
  const { setActiveTab } = useActiveTab();
  const { undo, redo } = useBlock();
  const valuesRef = useRef(values);
  const mergeTagsRef = useRef(mergeTags);
  valuesRef.current = values;
  mergeTagsRef.current = mergeTags;

  useEffect(() => {
    let animationFrame = 0;
    let cancelled = false;
    let pluginRoot: HTMLElement | null = null;
    const sync = () => {
      pluginRoot = getPluginElement();
      if (!pluginRoot) {
        if (!cancelled) animationFrame = window.requestAnimationFrame(sync);
        return;
      }
      pluginRoot.style.setProperty('--ctox-rich-text-toolbar-display', panel ? 'none' : 'block');
    };
    sync();
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(animationFrame);
      pluginRoot?.style.removeProperty('--ctox-rich-text-toolbar-display');
    };
  }, [panel]);

  useEffect(() => {
    send('selection', { blockId: focusIdx || getPageIdx() });
  }, [focusIdx]);

  useEffect(() => {
    send('change', { document: values });
  }, [values]);

  useEffect(() => {
    const onMessage = (event: MessageEvent<BridgeMessage>) => {
      if (event.source !== window.parent || event.origin !== PARENT_ORIGIN) return;
      const message = event.data;
      if (message?.source !== 'ctox-easy-email-parent' || message.instanceId !== INSTANCE_ID) return;
      try {
        if (message.type === 'set-document' || message.type === 'init') {
          const nextDocument = isEmailTemplate(message.payload?.document)
            ? message.payload.document
            : createDefaultTemplate();
          const documentCopy = cloneDeep(nextDocument);
          // form.initialize schedules a React update. Publish the transactional
          // bridge snapshot first so an immediately following getDocument or
          // getHtml can never read the previous template.
          valuesRef.current = documentCopy;
          form.initialize(cloneDeep(documentCopy));
          if (message.payload?.mergeTags) setMergeTags(cloneDeep(message.payload.mergeTags));
          if (message.type === 'init') applyBusinessOsTheme(message.payload);
          window.setTimeout(() => {
            if (message.type === 'init') send('initialized');
            else send('response', { document: cloneDeep(valuesRef.current) }, message.requestId);
          }, 0);
        } else if (message.type === 'set-theme') {
          applyBusinessOsTheme(message.payload);
        } else if (message.type === 'set-merge-tags') {
          const nextMergeTags = cloneDeep(message.payload || {});
          mergeTagsRef.current = nextMergeTags;
          setMergeTags(nextMergeTags);
          window.setTimeout(() => send('response', { applied: true }, message.requestId), 0);
        } else if (message.type === 'set-panel') {
          setPanel(message.payload?.name || null);
        } else if (message.type === 'set-viewport') {
          const viewport = message.payload?.name;
          setActiveTab(viewport === 'desktop' ? ActiveTabKeys.PC : viewport === 'mobile' ? ActiveTabKeys.MOBILE : ActiveTabKeys.EDIT);
        } else if (message.type === 'history') {
          if (message.payload?.action === 'undo') undo();
          else if (message.payload?.action === 'redo') redo();
          window.setTimeout(() => send('response', { applied: true }, message.requestId), 0);
        } else if (message.type === 'set-selection') {
          setFocusIdx(message.payload?.blockId || getPageIdx());
        } else if (message.type === 'set-logic-preview') {
          setLogicPreview(message.payload?.blockId ? cloneDeep(message.payload) : null);
          window.requestAnimationFrame(() => send('response', { applied: true }, message.requestId));
        } else if (message.type === 'get-document') {
          send('response', { document: cloneDeep(valuesRef.current) }, message.requestId);
        } else if (message.type === 'get-html') {
          const filtered = filterDocumentForPreview(valuesRef.current, mergeTagsRef.current);
          const mjmlSource = JsonToMjml({ data: filtered.content, mode: 'production', context: filtered.content });
          const result = mjml(mjmlSource, {});
          send('response', { html: result.html, mjml: mjmlSource, errors: result.errors || [] }, message.requestId);
        } else if (message.type === 'focus') {
          document.getElementById('easy-email-editor')?.focus();
        }
      } catch (error: any) {
        send('error', { message: error?.message || String(error) }, message.requestId);
      }
    };
    window.addEventListener('message', onMessage);
    send('ready', { upstreamCommit: '16bb02926a20af20dc6dc473c72619f4a0b4f64b' });
    return () => window.removeEventListener('message', onMessage);
  }, [form, redo, setActiveTab, setFocusIdx, setLogicPreview, setMergeTags, setPanel, undo]);

  return (
    <>
      <RichTextToolbarTheme />
      <LogicPreviewEffect preview={logicPreview} values={values} />
      <PanelDrawer panel={panel} onClose={() => setPanel(null)} />
    </>
  );
}

function App() {
  const [template] = useState<EmailTemplate>(() => createDefaultTemplate());
  const [mergeTags, setMergeTags] = useState<Record<string, any>>({});
  const [panel, setPanel] = useState<string | null>(null);
  const [logicPreview, setLogicPreview] = useState<LogicPreview>(null);

  const onUploadImage = useCallback(async (blob: Blob) => {
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  }, []);

  return (
    <ConfigProvider locale={enUS}>
      <EmailEditorProvider
        height='100vh'
        data={template}
        mergeTags={mergeTags}
        previewInjectData={mergeTags}
        enabledMergeTagsBadge
        enabledLogic
        autoComplete
        dashed={false}
        compact
        onUploadImage={onUploadImage}
      >
        {(formState, form) => (
          <div className='ctox-frame-shell'>
            <main className='ctox-frame-canvas'>
              <EmailEditor />
            </main>
            <RuntimeController
              form={form}
              values={formState.values}
              mergeTags={mergeTags}
              setMergeTags={setMergeTags}
              panel={panel}
              setPanel={setPanel}
              logicPreview={logicPreview}
              setLogicPreview={setLogicPreview}
            />
            <InteractivePrompt />
            <MergeTagBadgePrompt />
          </div>
        )}
      </EmailEditorProvider>
    </ConfigProvider>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
