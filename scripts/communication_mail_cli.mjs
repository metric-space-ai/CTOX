#!/usr/bin/env node

import * as childProcess from "child_process";
import * as crypto from "crypto";
import * as events from "events";
import * as fs from "fs";
import * as path from "path";
import * as tls from "tls";
import * as url from "url";

/*
 * Node-oriented transplant of the working Exchange/Outlook transport layer from Thesen_AI_Tool.
 *
 * Active scope in CTOX:
 *  - EWS
 *  - Microsoft Graph with caller-supplied bearer token
 *  - Exchange ActiveSync
 *  - Unified MailboxClient facade
 */

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function normalizeArray(x) {
  return (Array.isArray(x) ? x : [x]).filter(Boolean);
}

function stableUnique(arr) {
  const seen = new Set();
  const out = [];
  for (const v of arr) {
    if (v == null) continue;
    const s = String(v);
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(v);
  }
  return out;
}

//------------------------------------------------------------------------------
// EWS SOAP Client (legacy)
//------------------------------------------------------------------------------

const SOAP_NS = 'http://schemas.xmlsoap.org/soap/envelope/';
const TYPES_NS = 'http://schemas.microsoft.com/exchange/services/2006/types';
const MSG_NS = 'http://schemas.microsoft.com/exchange/services/2006/messages';

function escapeXml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function jsonToXml(obj, { omitNull = true } = {}) {
  if (obj == null) return '';
  if (typeof obj !== 'object') return escapeXml(obj);

  let xml = '';
  for (const tag of Object.keys(obj)) {
    const value = obj[tag];
    if (value == null && omitNull) continue;
    if (Array.isArray(value)) {
      for (const v of value) {
        if (v == null && omitNull) continue;
        xml += buildElement(tag, v, { omitNull });
      }
    } else {
      xml += buildElement(tag, value, { omitNull });
    }
  }
  return xml;
}

function buildElement(tag, value, { omitNull } = {}) {
  if (value == null) return omitNull ? '' : `<${tag}/>`;

  if (typeof value !== 'object') {
    return `<${tag}>${escapeXml(value)}</${tag}>`;
  }

  const attrs = value.attributes || {};
  let attrStr = '';
  for (const a of Object.keys(attrs)) {
    const v = attrs[a];
    if (v == null && omitNull) continue;
    if (v == null) continue;
    attrStr += ` ${a}="${escapeXml(String(v))}"`;
  }

  const raw = Object.prototype.hasOwnProperty.call(value, '$raw') ? String(value.$raw) : null;
  const text = Object.prototype.hasOwnProperty.call(value, '$value') ? escapeXml(value.$value) : '';

  let children = '';
  for (const childName of Object.keys(value)) {
    if (childName === 'attributes' || childName === '$value' || childName === '$raw') continue;
    const childVal = value[childName];
    if (childVal == null && omitNull) continue;

    if (Array.isArray(childVal)) {
      for (const v of childVal) {
        if (v == null && omitNull) continue;
        children += buildElement(childName, v, { omitNull });
      }
    } else {
      children += buildElement(childName, childVal, { omitNull });
    }
  }

  const inner = (raw || '') + text + children;
  if (!inner) return `<${tag}${attrStr}/>`;
  return `<${tag}${attrStr}>${inner}</${tag}>`;
}

function xmlElementToObject(node) {
  if (!node || node.nodeType !== 1) return null;
  const obj = {};

  if (node.attributes && node.attributes.length) {
    obj.attributes = {};
    for (let i = 0; i < node.attributes.length; i++) {
      const a = node.attributes[i];
      obj.attributes[a.name] = a.value;
    }
  }

  let hasElementChildren = false;
  const textParts = [];

  for (let i = 0; i < node.childNodes.length; i++) {
    const c = node.childNodes[i];
    if (c.nodeType === 1) {
      hasElementChildren = true;
      const name = c.nodeName;
      const childObj = xmlElementToObject(c);
      if (Object.prototype.hasOwnProperty.call(obj, name)) {
        if (!Array.isArray(obj[name])) obj[name] = [obj[name]];
        obj[name].push(childObj);
      } else {
        obj[name] = childObj;
      }
    } else if (c.nodeType === 3 || c.nodeType === 4) {
      const t = c.nodeValue;
      if (t && t.trim()) textParts.push(t.trim());
    }
  }

  if (!hasElementChildren && textParts.length) {
    obj.$value = textParts.join(' ');
    if (Object.keys(obj).length === 1) return obj.$value;
  }

  return obj;
}

function decodeXmlEntities(text) {
  if (!text || text.indexOf('&') < 0) return text || '';

  return String(text).replace(/&(#x?[0-9a-fA-F]+|amp|lt|gt|quot|apos);/g, (m, entity) => {
    const e = String(entity || '').toLowerCase();
    switch (e) {
      case 'amp':
        return '&';
      case 'lt':
        return '<';
      case 'gt':
        return '>';
      case 'quot':
        return '"';
      case 'apos':
        return "'";
      default:
        break;
    }

    if (e.startsWith('#x')) {
      const cp = Number.parseInt(e.slice(2), 16);
      if (Number.isFinite(cp)) return String.fromCodePoint(cp);
      return m;
    }
    if (e.startsWith('#')) {
      const cp = Number.parseInt(e.slice(1), 10);
      if (Number.isFinite(cp)) return String.fromCodePoint(cp);
      return m;
    }
    return m;
  });
}

function parseXmlAttributes(attrText) {
  const attrs = {};
  if (!attrText) return attrs;

  const re = /([^\s=/>]+)\s*=\s*("([^"]*)"|'([^']*)')/g;
  let m;
  while ((m = re.exec(attrText))) {
    const key = m[1];
    const rawVal = m[3] ?? m[4] ?? '';
    attrs[key] = decodeXmlEntities(rawVal);
  }
  return attrs;
}

function parseXmlLight(rawXml) {
  const xml = String(rawXml || '');
  const tokenRe = /<!\[CDATA\[[\s\S]*?\]\]>|<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!DOCTYPE[\s\S]*?>|<\/?[^>]+>|[^<]+/g;
  const root = { name: '#document', attributes: {}, children: [], textParts: [] };
  const stack = [root];
  let match;

  while ((match = tokenRe.exec(xml))) {
    const token = match[0];
    if (!token) continue;

    if (token.startsWith('<?') || token.startsWith('<!--') || token.startsWith('<!DOCTYPE')) {
      continue;
    }

    if (token.startsWith('<![CDATA[')) {
      const text = token.slice(9, -3);
      if (text && text.trim()) {
        stack[stack.length - 1].textParts.push(text.trim());
      }
      continue;
    }

    if (token.startsWith('</')) {
      const tagName = token.slice(2, -1).trim().split(/\s+/, 1)[0];
      if (!tagName) throw new Error('XML parse error: invalid closing tag');
      if (stack.length <= 1) {
        throw new Error(`XML parse error: unexpected closing tag </${tagName}>`);
      }
      const current = stack[stack.length - 1];
      if (current.name !== tagName) {
        throw new Error(`XML parse error: closing </${tagName}> does not match <${current.name}>`);
      }
      stack.pop();
      continue;
    }

    if (token.startsWith('<')) {
      const isSelfClosing = /\/\s*>$/.test(token);
      const raw = token.slice(1, token.length - (isSelfClosing ? 2 : 1)).trim();
      if (!raw) continue;

      const wsIndex = raw.search(/\s/);
      const tagName = wsIndex < 0 ? raw : raw.slice(0, wsIndex);
      const attrText = wsIndex < 0 ? '' : raw.slice(wsIndex + 1);
      if (!tagName) throw new Error('XML parse error: missing tag name');

      const node = {
        name: tagName,
        attributes: parseXmlAttributes(attrText),
        children: [],
        textParts: [],
      };

      stack[stack.length - 1].children.push(node);
      if (!isSelfClosing) stack.push(node);
      continue;
    }

    const text = decodeXmlEntities(token);
    if (text && text.trim()) {
      stack[stack.length - 1].textParts.push(text.trim());
    }
  }

  if (stack.length !== 1) {
    const openTag = stack[stack.length - 1]?.name || 'unknown';
    throw new Error(`XML parse error: unclosed tag <${openTag}>`);
  }

  const docElement = root.children.find((n) => n && typeof n.name === 'string') || null;
  if (!docElement) throw new Error('XML parse error: no root element');

  return { root: docElement };
}

function xmlNodeToObject(node) {
  if (!node) return null;

  const obj = {};
  if (node.attributes && Object.keys(node.attributes).length) {
    obj.attributes = { ...node.attributes };
  }

  let hasElementChildren = false;
  for (const child of node.children || []) {
    hasElementChildren = true;
    const name = child.name;
    const childObj = xmlNodeToObject(child);
    if (Object.prototype.hasOwnProperty.call(obj, name)) {
      if (!Array.isArray(obj[name])) obj[name] = [obj[name]];
      obj[name].push(childObj);
    } else {
      obj[name] = childObj;
    }
  }

  const textParts = [];
  for (const t of node.textParts || []) {
    if (t && t.trim()) textParts.push(t.trim());
  }

  if (!hasElementChildren && textParts.length) {
    obj.$value = textParts.join(' ');
    if (Object.keys(obj).length === 1) return obj.$value;
  }

  return obj;
}

function localName(name) {
  const s = String(name || '');
  const idx = s.indexOf(':');
  return idx >= 0 ? s.slice(idx + 1) : s;
}

function findFirstNode(node, predicate) {
  if (!node) return null;
  if (predicate(node)) return node;
  for (const child of node.children || []) {
    const found = findFirstNode(child, predicate);
    if (found) return found;
  }
  return null;
}

function parseSoapResponse(xml) {
  const rawXml = typeof xml === 'string' ? xml : String(xml ?? '');
  const hasDomParser = typeof DOMParser === 'function';

  if (hasDomParser) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(rawXml, 'text/xml');

    const parserError = doc.getElementsByTagName('parsererror')[0] || null;
    if (parserError) {
      const msg = (parserError.textContent || '').trim() || 'Unknown XML parse error';
      const err = new Error(`EWS: XML parse error: ${msg}`);
      err.rawXml = rawXml;
      err.document = doc;
      throw err;
    }

    const body = doc.getElementsByTagNameNS(SOAP_NS, 'Body')[0] || null;
    const fault = doc.getElementsByTagNameNS(SOAP_NS, 'Fault')[0] || null;

    let responseNode = null;
    if (body) {
      for (let i = 0; i < body.childNodes.length; i++) {
        const child = body.childNodes[i];
        if (child.nodeType === 1) {
          responseNode = child;
          break;
        }
      }
    }

    return {
      rawXml,
      document: doc,
      fault: fault ? xmlElementToObject(fault) : null,
      response: responseNode ? xmlElementToObject(responseNode) : null,
    };
  }

  let parsed;
  try {
    parsed = parseXmlLight(rawXml);
  } catch (e) {
    const msg = String(e?.message || e || 'Unknown XML parse error');
    const err = new Error(`EWS: XML parse error: ${msg}`);
    err.rawXml = rawXml;
    err.document = null;
    throw err;
  }
  const bodyNode = findFirstNode(parsed.root, (n) => localName(n.name) === 'Body');
  const faultNode = bodyNode
    ? findFirstNode(bodyNode, (n) => localName(n.name) === 'Fault')
    : findFirstNode(parsed.root, (n) => localName(n.name) === 'Fault');

  let responseNode = null;
  if (bodyNode) {
    for (const child of bodyNode.children || []) {
      if (localName(child.name) !== 'Fault') {
        responseNode = child;
        break;
      }
    }
  }

  return {
    rawXml,
    document: null,
    fault: faultNode ? xmlNodeToObject(faultNode) : null,
    response: responseNode ? xmlNodeToObject(responseNode) : null,
  };
}

export class EwsError extends Error {
  constructor(message, { code, responseClass, httpStatus, httpStatusText, details, rawXml } = {}) {
    super(message);
    this.name = 'EwsError';
    this.code = code || null;
    this.responseClass = responseClass || null;
    this.httpStatus = httpStatus || null;
    this.httpStatusText = httpStatusText || null;
    this.details = details || null;
    this.rawXml = rawXml || null;
  }
}

function assertEwsSuccess(parsed, { httpStatus = null, httpStatusText = null } = {}) {
  if (!parsed) throw new EwsError('EWS: No response');

  if (parsed.fault) {
    const fs =
      (parsed.fault['faultstring'] && parsed.fault['faultstring'].$value) ||
      parsed.fault['faultstring'] ||
      (typeof parsed.fault === 'string' ? parsed.fault : null) ||
      'SOAP Fault';
    throw new EwsError(`EWS SOAP Fault: ${fs}`, {
      httpStatus,
      httpStatusText,
      details: parsed.fault,
      rawXml: parsed.rawXml,
    });
  }

  const resp = parsed.response;
  if (!resp || typeof resp !== 'object') return parsed;

  const responseMessages = resp['m:ResponseMessages'] || resp['ResponseMessages'] || resp['s:ResponseMessages'] || null;
  if (!responseMessages || typeof responseMessages !== 'object') return parsed;

  const msgNodes = [];
  for (const k of Object.keys(responseMessages)) {
    if (k.toLowerCase().includes('responsemessage')) {
      const v = responseMessages[k];
      if (Array.isArray(v)) msgNodes.push(...v);
      else if (v) msgNodes.push(v);
    }
  }

  for (const m of msgNodes) {
    const rc = m?.attributes?.ResponseClass || null;
    if (rc && rc !== 'Success') {
      const code = m['m:ResponseCode'] || m['ResponseCode'] || m['t:ResponseCode'] || 'Error';
      const text = m['m:MessageText'] || m['MessageText'] || m['t:MessageText'] || 'EWS error';
      throw new EwsError(`EWS ${rc}: ${code} - ${text}`, {
        code,
        responseClass: rc,
        httpStatus,
        httpStatusText,
        details: m,
        rawXml: parsed.rawXml,
      });
    }
  }

  return parsed;
}

export class ExchangeCredentials {
  /**
   * opts:
   *  - username, password
   *  - token
   *  - authType: 'basic' | 'bearer' | 'ntlm'
   */
  constructor(opts = {}) {
    const { username, password, token, authType } = opts;
    this.username = username || null;
    this.password = password || null;
    this.token = token || null;

    if (authType) this.authType = authType.toLowerCase();
    else if (token) this.authType = 'bearer';
    else if (username && password) this.authType = 'basic';
    else this.authType = 'ntlm';
  }
}

export class EwsClient {
  constructor(opts = {}) {
    const {
      url,
      version,
      credentials,
      extraHeaders,
      userAgent,
      maxRetries,
      retryBaseDelayMs,
      retryMaxDelayMs,
    } = opts;

    if (!url) throw new Error('EwsClient: url is required');
    this.url = url;
    this.version = version || 'Exchange2013';
    this.credentials = credentials || null;
    this.extraHeaders = extraHeaders || {};
    this.userAgent = userAgent || 'ews-lite-client/3.0';
    this.maxRetries = Number.isFinite(maxRetries) ? maxRetries : 2;
    this.retryBaseDelayMs = Number.isFinite(retryBaseDelayMs) ? retryBaseDelayMs : 400;
    this.retryMaxDelayMs = Number.isFinite(retryMaxDelayMs) ? retryMaxDelayMs : 5000;
  }

  _buildEnvelope(opName, innerXml, headerXml, opAttributes) {
    const header =
      headerXml || `<t:RequestServerVersion Version="${escapeXml(this.version)}" xmlns:t="${TYPES_NS}"/>`;

    let attrStr = '';
    if (opAttributes && typeof opAttributes === 'object') {
      for (const [k, v] of Object.entries(opAttributes)) {
        if (v == null) continue;
        attrStr += ` ${k}="${escapeXml(String(v))}"`;
      }
    }

    const body = `<m:${opName}${attrStr}>${innerXml || ''}</m:${opName}>`;

    return (
      `<?xml version="1.0" encoding="utf-8"?>` +
      `<soap:Envelope xmlns:soap="${SOAP_NS}" xmlns:m="${MSG_NS}" xmlns:t="${TYPES_NS}">` +
      `<soap:Header>${header}</soap:Header>` +
      `<soap:Body>${body}</soap:Body>` +
      `</soap:Envelope>`
    );
  }

  _getHeaders(soapAction) {
    const headers = {
      'Content-Type': 'text/xml; charset=utf-8',
      Accept: 'text/xml',
      'X-ClientInfo': this.userAgent,
      ...this.extraHeaders,
    };
    if (soapAction) headers.SOAPAction = soapAction;

    if (!this.credentials) return headers;

    switch (this.credentials.authType) {
      case 'basic': {
        if (!this.credentials.username || !this.credentials.password) {
          throw new Error('Basic auth requires username and password');
        }
        const token =
          typeof btoa === 'function'
            ? btoa(`${this.credentials.username}:${this.credentials.password}`)
            : Buffer.from(`${this.credentials.username}:${this.credentials.password}`, 'utf8').toString('base64');
        headers.Authorization = `Basic ${token}`;
        break;
      }
      case 'bearer': {
        if (!this.credentials.token) throw new Error('Bearer auth requires token');
        headers.Authorization = `Bearer ${this.credentials.token}`;
        break;
      }
      case 'ntlm':
        // Integrated auth via browser session.
        break;
      default:
        throw new Error(`Unsupported auth type: ${this.credentials.authType}`);
    }

    return headers;
  }

  _isRetryableHttp(status) {
    return status === 408 || status === 429 || (status >= 500 && status <= 504);
  }

  _parseRetryAfterMs(res) {
    const ra = res?.headers?.get?.('Retry-After');
    if (!ra) return null;
    const secs = Number(ra);
    if (Number.isFinite(secs)) return Math.max(0, secs * 1000);
    const dt = Date.parse(ra);
    if (!Number.isNaN(dt)) return Math.max(0, dt - Date.now());
    return null;
  }

  async _send(xml, options = {}) {
    const timeoutMs = typeof options.timeoutMs === 'number' ? options.timeoutMs : 30000;
    const soapAction = options.soapAction || null;

    let attempt = 0;
    let lastErr = null;

    while (attempt <= this.maxRetries) {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const res = await fetch(this.url, {
          method: 'POST',
          headers: this._getHeaders(soapAction),
          body: xml,
          credentials: 'include',
          signal: controller.signal,
        });

        const text = await res.text().catch(() => '');

        const contentType = (res.headers.get('Content-Type') || '').toLowerCase();
        const looksLikeXml = contentType.includes('xml') || /^\s*</.test(text);

        if (!res.ok) {
          if (looksLikeXml && text) {
            try {
              const parsed = parseSoapResponse(text);
              assertEwsSuccess(parsed, { httpStatus: res.status, httpStatusText: res.statusText });
              return text;
            } catch (e) {
              const retryable = this._isRetryableHttp(res.status);
              if (!retryable) throw e;
              lastErr = e;
            }
          } else {
            const err = new EwsError(`HTTP ${res.status} ${res.statusText}`, {
              httpStatus: res.status,
              httpStatusText: res.statusText,
              details: { raw: text },
              rawXml: looksLikeXml ? text : null,
            });
            const retryable = this._isRetryableHttp(res.status);
            if (!retryable) throw err;
            lastErr = err;
          }

          const raMs = this._parseRetryAfterMs(res);
          const base = Math.min(this.retryMaxDelayMs, this.retryBaseDelayMs * Math.pow(2, attempt));
          const jitter = Math.floor(Math.random() * 150);
          await sleep(Math.max(raMs ?? 0, base + jitter));
          attempt++;
          continue;
        }

        return text;
      } catch (err) {
        lastErr = err;
        const isAbort = err?.name === 'AbortError';
        const retryable = isAbort || err instanceof TypeError;

        if (!retryable || attempt >= this.maxRetries) throw err;

        const base = Math.min(this.retryMaxDelayMs, this.retryBaseDelayMs * Math.pow(2, attempt));
        const jitter = Math.floor(Math.random() * 150);
        await sleep(base + jitter);
        attempt++;
      } finally {
        clearTimeout(t);
      }
    }

    throw lastErr || new Error('EWS: request failed');
  }

  async call(opName, bodyArgs = null, options = {}) {
    const innerXml = typeof bodyArgs === 'string' ? bodyArgs : bodyArgs ? jsonToXml(bodyArgs) : '';
    const envelope = this._buildEnvelope(opName, innerXml, options.headerXml, options.opAttributes);
    const rawXml = await this._send(envelope, options);

    if (options.rawXml) return { rawXml };

    const parsed = parseSoapResponse(rawXml);
    if (options.noThrow) return parsed;

    return assertEwsSuccess(parsed);
  }
}

//------------------------------------------------------------------------------
// EWS Domain Objects + Account
//------------------------------------------------------------------------------

export class Folder {
  constructor(opts = {}) {
    const {
      account,
      id = null,
      changeKey = null,
      distinguishedId = null,
      displayName = null,
      totalCount = null,
      childFolderCount = null,
      unreadCount = null,
      parentFolderId = null,
    } = opts;

    if (!(account instanceof ExchangeAccount)) throw new Error('Folder: account must be ExchangeAccount');

    this.account = account;
    this.id = id;
    this.changeKey = changeKey;
    this.distinguishedId = distinguishedId;
    this.displayName = displayName;

    this.totalCount = typeof totalCount === 'number' ? totalCount : null;
    this.childFolderCount = typeof childFolderCount === 'number' ? childFolderCount : null;
    this.unreadCount = typeof unreadCount === 'number' ? unreadCount : null;
    this.parentFolderId = parentFolderId;
  }

  get isDistinguished() {
    return !!this.distinguishedId;
  }

  toFolderIdArg() {
    return ExchangeAccount.toFolderIdArg(this);
  }

  async listItems(opts = {}) {
    return this.account.findItems({ ...opts, folderId: this });
  }

  async listSubfolders(opts = {}) {
    return this.account.findFolders({ parentFolderId: this, ...opts });
  }

  static fromEwsFolder(account, ewsFolder) {
    if (!ewsFolder) return null;

    let id = null;
    let changeKey = null;
    let distinguishedId = null;

    const folderIdNode = ewsFolder['t:FolderId'] || ewsFolder.FolderId || null;
    if (folderIdNode?.attributes) {
      id = folderIdNode.attributes.Id || null;
      changeKey = folderIdNode.attributes.ChangeKey || null;
    }

    const distinguishedNode = ewsFolder['t:DistinguishedFolderId'] || ewsFolder.DistinguishedFolderId || null;
    if (distinguishedNode?.attributes) {
      distinguishedId = distinguishedNode.attributes.Id || null;
    }

    const displayNameRaw = ewsFolder['t:DisplayName'] || ewsFolder.DisplayName;
    const displayName = typeof displayNameRaw === 'string' ? displayNameRaw : displayNameRaw?.$value || null;

    const totalCountRaw = ewsFolder['t:TotalCount'] || ewsFolder.TotalCount;
    const childCountRaw = ewsFolder['t:ChildFolderCount'] || ewsFolder.ChildFolderCount;
    const unreadCountRaw = ewsFolder['t:UnreadCount'] || ewsFolder.UnreadCount;

    return new Folder({
      account,
      id,
      changeKey,
      distinguishedId,
      displayName,
      totalCount: totalCountRaw != null ? Number(totalCountRaw) : null,
      childFolderCount: childCountRaw != null ? Number(childCountRaw) : null,
      unreadCount: unreadCountRaw != null ? Number(unreadCountRaw) : null,
    });
  }
}

export class Message {
  /**
   * Minimal message constructor for sending/saving (EWS).
   */
  constructor(opts = {}) {
    const {
      account,
      subject,
      body,
      bodyType = 'HTML',
      toRecipients = [],
      ccRecipients = [],
      bccRecipients = [],
      categories = null,
      importance = null,
    } = opts;

    if (!(account instanceof ExchangeAccount)) throw new Error('Message: account required');
    if (!subject) throw new Error('Message: subject required');
    if (body == null) throw new Error('Message: body required');

    this.account = account;
    this.subject = subject;
    this.body = body;
    this.bodyType = bodyType;

    this.toRecipients = normalizeArray(toRecipients);
    this.ccRecipients = normalizeArray(ccRecipients);
    this.bccRecipients = normalizeArray(bccRecipients);

    this.categories = categories ? normalizeArray(categories) : null;
    this.importance = importance || null;
  }

  toEwsCreateItemArgs({ savedFolderId = 'sentitems' } = {}) {
    const recipients = {};
    const mk = (addr) => ({ 't:Mailbox': { 't:EmailAddress': { $value: addr } } });

    if (this.toRecipients.length) recipients['t:ToRecipients'] = this.toRecipients.map(mk);
    if (this.ccRecipients.length) recipients['t:CcRecipients'] = this.ccRecipients.map(mk);
    if (this.bccRecipients.length) recipients['t:BccRecipients'] = this.bccRecipients.map(mk);

    const msg = {
      't:ItemClass': { $value: 'IPM.Note' },
      't:Subject': { $value: this.subject },
      't:Body': { attributes: { BodyType: this.bodyType }, $value: this.body },
      ...recipients,
    };

    if (this.categories?.length) {
      msg['t:Categories'] = { 't:String': this.categories.map((c) => ({ $value: c })) };
    }
    if (this.importance) {
      msg['t:Importance'] = { $value: this.importance };
    }

    return {
      'm:SavedItemFolderId': ExchangeAccount.toFolderIdArg(savedFolderId),
      'm:Items': { 't:Message': msg },
    };
  }
}

function firstEwsResponseNode(node) {
  if (Array.isArray(node)) return node[0] || null;
  return node || null;
}

function extractEwsCreatedItemNode(parsed) {
  const resp = parsed?.response;
  const rm = resp?.['m:ResponseMessages'] || resp?.ResponseMessages || resp?.['s:ResponseMessages'] || null;
  if (!rm || typeof rm !== 'object') return null;
  const msgKey = Object.keys(rm).find((key) => key.toLowerCase().includes('createitemresponsemessage'));
  const msg = msgKey ? firstEwsResponseNode(rm[msgKey]) : null;
  const itemsNode = msg?.['m:Items'] || msg?.Items || null;
  if (!itemsNode || typeof itemsNode !== 'object') return null;
  for (const value of Object.values(itemsNode)) {
    const itemNode = firstEwsResponseNode(value);
    if (itemNode) return itemNode;
  }
  return null;
}

function extractEwsItemIdentity(itemNode) {
  const itemIdNode = itemNode?.['t:ItemId'] || itemNode?.ItemId || null;
  const attrs = itemIdNode?.attributes || {};
  return {
    id: attrs.Id || itemIdNode?.Id || itemNode?.Id || null,
    changeKey: attrs.ChangeKey || itemIdNode?.ChangeKey || null,
  };
}

export class ExchangeAccount {
  constructor(opts = {}) {
    const { primarySmtpAddress, client } = opts;
    if (!primarySmtpAddress || !String(primarySmtpAddress).includes('@')) {
      throw new Error('ExchangeAccount: valid primarySmtpAddress required');
    }
    if (!(client instanceof EwsClient)) throw new Error('ExchangeAccount: client must be EwsClient');

    this.primarySmtpAddress = primarySmtpAddress;
    this.client = client;
  }

  static toItemId(item) {
    if (typeof item === 'string') return { attributes: { Id: item } };
    if (item && typeof item === 'object') {
      const id = item.Id || item.id;
      const changeKey = item.ChangeKey || item.changeKey;
      if (!id) throw new Error('Invalid item identifier: missing Id');
      const attrs = { Id: id };
      if (changeKey) attrs.ChangeKey = changeKey;
      return { attributes: attrs };
    }
    throw new Error('Invalid item identifier');
  }

  static distinguishedFolder(id) {
    return { 't:DistinguishedFolderId': { attributes: { Id: id } } };
  }

  static toFolderIdArg(folder) {
    if (!folder) throw new Error('Invalid folder identifier');

    if (typeof folder === 'string') {
      return ExchangeAccount.distinguishedFolder(folder);
    }

    if (folder instanceof Folder) {
      if (folder.isDistinguished && folder.distinguishedId) return ExchangeAccount.distinguishedFolder(folder.distinguishedId);
      if (folder.id) {
        const attrs = { Id: folder.id };
        if (folder.changeKey) attrs.ChangeKey = folder.changeKey;
        return { 't:FolderId': { attributes: attrs } };
      }
      throw new Error('Folder must have either distinguishedId or id');
    }

    if (typeof folder === 'object') {
      const id = folder.Id || folder.id;
      const changeKey = folder.ChangeKey || folder.changeKey;
      if (!id) throw new Error('Invalid folder identifier object');
      const attrs = { Id: id };
      if (changeKey) attrs.ChangeKey = changeKey;
      return { 't:FolderId': { attributes: attrs } };
    }

    throw new Error('Unsupported folder identifier');
  }

  static _buildParentFolderIds(folderId) {
    const parentFolderIds = {};
    const addOne = (f) => {
      const arg = ExchangeAccount.toFolderIdArg(f);
      for (const key of Object.keys(arg)) {
        if (Object.prototype.hasOwnProperty.call(parentFolderIds, key)) {
          const existing = parentFolderIds[key];
          parentFolderIds[key] = Array.isArray(existing) ? [...existing, arg[key]] : [existing, arg[key]];
        } else {
          parentFolderIds[key] = arg[key];
        }
      }
    };

    if (Array.isArray(folderId)) folderId.forEach(addOne);
    else addOne(folderId);

    return parentFolderIds;
  }

  folder(distinguishedId) {
    return new Folder({ account: this, distinguishedId });
  }
  get inboxFolder() { return this.folder('inbox'); }
  get sentFolder() { return this.folder('sentitems'); }
  get draftsFolder() { return this.folder('drafts'); }
  get trashFolder() { return this.folder('deleteditems'); }
  get rootFolder() { return this.folder('msgfolderroot'); }

  // ---- Mail listing/search ----
  async findItems({
    folderId = 'inbox',
    maxEntriesReturned = 50,
    offset = 0,
    baseShape = 'IdOnly',
    traversal = 'Shallow',
    additionalProperties = null,
    queryString = null,
    sortBy = null,
  } = {}) {
    const itemShape = { 't:BaseShape': { $value: baseShape } };

    if (additionalProperties?.length) {
      itemShape['t:AdditionalProperties'] = {
        't:FieldURI': additionalProperties.map((f) => ({ attributes: { FieldURI: f } })),
      };
    }

    const args = {
      'm:ItemShape': itemShape,
      'm:ParentFolderIds': ExchangeAccount._buildParentFolderIds(folderId),
      'm:IndexedPageItemView': {
        attributes: {
          MaxEntriesReturned: String(maxEntriesReturned),
          Offset: String(offset),
          BasePoint: 'Beginning',
        },
      },
    };

    if (queryString) args['m:QueryString'] = { $value: queryString };

    if (sortBy?.fieldURI) {
      args['m:SortOrder'] = {
        't:FieldOrder': {
          attributes: { Order: sortBy.order || 'Descending' },
          't:FieldURI': { attributes: { FieldURI: sortBy.fieldURI } },
        },
      };
    }

    const parsed = await this.client.call('FindItem', args, { opAttributes: { Traversal: traversal } });

    const resp = parsed.response;
    const rm = resp?.['m:ResponseMessages'];
    const msgKey = rm ? Object.keys(rm).find((k) => k.toLowerCase().includes('finditemresponsemessage')) : null;
    const msg = msgKey ? rm[msgKey] : null;

    const rootFolder = msg?.['m:RootFolder'] || null;
    const rfAttrs = rootFolder?.attributes || {};
    const includesLastItemInRange = rfAttrs.IncludesLastItemInRange === 'true';

    const nextOffsetRaw = rfAttrs.IndexedPagingOffset;
    let nextOffset = null;
    if (nextOffsetRaw != null) {
      const n = Number(nextOffsetRaw);
      nextOffset = Number.isFinite(n) ? n : null;
    } else if (!includesLastItemInRange) {
      nextOffset = offset + maxEntriesReturned;
    }

    const itemsNode = rootFolder?.['t:Items'] || null;
    const items = [];
    if (itemsNode && typeof itemsNode === 'object') {
      for (const k of Object.keys(itemsNode)) {
        const v = itemsNode[k];
        if (Array.isArray(v)) items.push(...v);
        else if (v) items.push(v);
      }
    }

    return { parsed, items, nextOffset, includesLastItemInRange };
  }

  async listInbox({ pageSize = 50, offset = 0, queryString = null } = {}) {
    return this.findItems({
      folderId: 'inbox',
      maxEntriesReturned: pageSize,
      offset,
      baseShape: 'IdOnly',
      additionalProperties: [
        'item:Subject',
        'message:From',
        'item:DateTimeReceived',
        'message:IsRead',
        'item:ItemClass',
        'item:Categories',
        'item:Importance',
        'item:Flag',
        'item:HasAttachments',
      ],
      sortBy: { fieldURI: 'item:DateTimeReceived', order: 'Descending' },
      queryString,
    });
  }

  async getItem(item, { baseShape = 'Default', includeMimeContent = false, additionalProperties = null } = {}) {
    const itemId = ExchangeAccount.toItemId(item);

    const itemShape = { 't:BaseShape': { $value: baseShape } };
    if (includeMimeContent) itemShape['t:IncludeMimeContent'] = { $value: 'true' };

    if (additionalProperties?.length) {
      itemShape['t:AdditionalProperties'] = {
        't:FieldURI': additionalProperties.map((f) => ({ attributes: { FieldURI: f } })),
      };
    }

    const args = {
      'm:ItemShape': itemShape,
      'm:ItemIds': { 't:ItemId': itemId },
    };

    const parsed = await this.client.call('GetItem', args);

    const resp = parsed.response;
    const rm = resp?.['m:ResponseMessages'];
    const msgKey = rm ? Object.keys(rm).find((k) => k.toLowerCase().includes('getitemresponsemessage')) : null;
    const msg = msgKey ? rm[msgKey] : null;

    const itemsNode = msg?.['m:Items'] || null;
    const message = itemsNode?.['t:Message'] || itemsNode?.Message || null;

    return { parsed, message };
  }

  async send(message, { saveCopyFolderId = 'sentitems' } = {}) {
    if (!(message instanceof Message)) throw new Error('send() expects Message');
    const ewsArgs = message.toEwsCreateItemArgs({ savedFolderId: saveCopyFolderId });
    return this.client.call('CreateItem', ewsArgs, {
      opAttributes: { MessageDisposition: 'SendAndSaveCopy' },
    });
  }

  async saveDraft(message, { folderId = 'drafts' } = {}) {
    if (!(message instanceof Message)) throw new Error('saveDraft() expects Message');
    const ewsArgs = message.toEwsCreateItemArgs({ savedFolderId: folderId });
    return this.client.call('CreateItem', ewsArgs, {
      opAttributes: { MessageDisposition: 'SaveOnly' },
    });
  }

  async createReplyDraft(item, { body = '', replyAll = false, bodyType = 'HTML', folderId = 'drafts' } = {}) {
    const idAttrs = ExchangeAccount.toItemId(item).attributes;
    const tag = replyAll ? 't:ReplyAllToItem' : 't:ReplyToItem';

    const replyObj = { 't:ReferenceItemId': { attributes: idAttrs } };
    if (body) {
      replyObj['t:NewBodyContent'] = { attributes: { BodyType: bodyType }, $value: body };
    }

    const args = {
      'm:SavedItemFolderId': ExchangeAccount.toFolderIdArg(folderId),
      'm:Items': { [tag]: replyObj },
    };
    const parsed = await this.client.call('CreateItem', args, {
      opAttributes: { MessageDisposition: 'SaveOnly' },
    });
    const itemNode = extractEwsCreatedItemNode(parsed);
    const identity = extractEwsItemIdentity(itemNode);
    return {
      parsed,
      item: itemNode,
      id: identity.id,
      draftId: identity.id,
      changeKey: identity.changeKey,
    };
  }

  async sendExisting(items, { saveCopy = true, saveFolderId = 'sentitems' } = {}) {
    const list = Array.isArray(items) ? items : [items];
    if (!list.length) throw new Error('sendExisting() expects at least one item');

    const ids = list.map(ExchangeAccount.toItemId);
    const args = { 'm:ItemIds': { 't:ItemId': ids } };
    if (saveCopy) args['m:SavedItemFolderId'] = ExchangeAccount.toFolderIdArg(saveFolderId);

    return this.client.call('SendItem', args, {
      opAttributes: { SaveItemToFolder: saveCopy ? 'true' : 'false' },
    });
  }

  async deleteItems(items, deleteType = 'MoveToDeletedItems') {
    const list = Array.isArray(items) ? items : [items];
    if (!list.length) throw new Error('deleteItems() expects at least one item');
    const itemIds = list.map(ExchangeAccount.toItemId);
    const args = { 'm:ItemIds': { 't:ItemId': itemIds } };
    return this.client.call('DeleteItem', args, { opAttributes: { DeleteType: deleteType } });
  }

  async moveItems(items, destFolderId) {
    const list = Array.isArray(items) ? items : [items];
    if (!list.length) throw new Error('moveItems() expects at least one item');
    const args = {
      'm:ToFolderId': ExchangeAccount.toFolderIdArg(destFolderId),
      'm:ItemIds': { 't:ItemId': list.map(ExchangeAccount.toItemId) },
    };
    return this.client.call('MoveItem', args);
  }

  async copyItems(items, destFolderId) {
    const list = Array.isArray(items) ? items : [items];
    if (!list.length) throw new Error('copyItems() expects at least one item');
    const args = {
      'm:ToFolderId': ExchangeAccount.toFolderIdArg(destFolderId),
      'm:ItemIds': { 't:ItemId': list.map(ExchangeAccount.toItemId) },
    };
    return this.client.call('CopyItem', args);
  }

  async setReadFlag(items, isRead = true) {
    const list = Array.isArray(items) ? items : [items];
    if (!list.length) throw new Error('setReadFlag() expects at least one item');

    const itemChanges = list.map((item) => ({
      't:ItemId': ExchangeAccount.toItemId(item),
      't:Updates': {
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'message:IsRead' } },
          't:Message': { 't:IsRead': { $value: isRead ? 'true' : 'false' } },
        },
      },
    }));

    const args = { 'm:ItemChanges': { 't:ItemChange': itemChanges } };

    return this.client.call('UpdateItem', args, {
      opAttributes: { ConflictResolution: 'AutoResolve', MessageDisposition: 'SaveOnly' },
    });
  }

  /**
   * Update fields on message (drafts safe).
   * Supports: subject, body, categories, importance, flag
   * flag: { status:"Flagged"|"Complete"|"NotFlagged", dueDate?:ISO, startDate?:ISO, completeDate?:ISO }
   */
  async updateMessage(item, { subject = null, body = null, bodyType = 'HTML', categories = null, importance = null, flag = null } = {}) {
    if (!item) throw new Error('updateMessage() requires item');

    const updates = [];

    if (subject != null) {
      updates.push({
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'item:Subject' } },
          't:Message': { 't:Subject': { $value: subject } },
        },
      });
    }

    if (body != null) {
      updates.push({
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'item:Body' } },
          't:Message': { 't:Body': { attributes: { BodyType: bodyType }, $value: body } },
        },
      });
    }

    if (categories != null) {
      const cats = normalizeArray(categories);
      updates.push({
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'item:Categories' } },
          't:Message': { 't:Categories': { 't:String': cats.map((c) => ({ $value: c })) } },
        },
      });
    }

    if (importance != null) {
      updates.push({
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'item:Importance' } },
          't:Message': { 't:Importance': { $value: importance } },
        },
      });
    }

    if (flag != null) {
      const status = String(flag.status || 'NotFlagged');
      const flagNode = { 't:FlagStatus': { $value: status } };
      if (flag.startDate) flagNode['t:StartDate'] = { $value: flag.startDate };
      if (flag.dueDate) flagNode['t:DueDate'] = { $value: flag.dueDate };
      if (flag.completeDate) flagNode['t:CompleteDate'] = { $value: flag.completeDate };

      updates.push({
        't:SetItemField': {
          't:FieldURI': { attributes: { FieldURI: 'item:Flag' } },
          't:Message': { 't:Flag': flagNode },
        },
      });
    }

    if (!updates.length) return { skipped: true };

    const args = {
      'm:ItemChanges': {
        't:ItemChange': {
          't:ItemId': ExchangeAccount.toItemId(item),
          't:Updates': updates,
        },
      },
    };

    return this.client.call('UpdateItem', args, {
      opAttributes: { ConflictResolution: 'AutoResolve', MessageDisposition: 'SaveOnly' },
    });
  }

  async replyTo(item, { body = '', replyAll = false, bodyType = 'HTML', saveCopy = true } = {}) {
    const idAttrs = ExchangeAccount.toItemId(item).attributes;
    const tag = replyAll ? 't:ReplyAllToItem' : 't:ReplyToItem';

    const replyObj = { 't:ReferenceItemId': { attributes: idAttrs } };
    if (body) {
      replyObj['t:NewBodyContent'] = { attributes: { BodyType: bodyType }, $value: body };
    }

    const args = { 'm:Items': { [tag]: replyObj } };
    return this.client.call('CreateItem', args, {
      opAttributes: { MessageDisposition: saveCopy ? 'SendAndSaveCopy' : 'SendOnly' },
    });
  }

  async forward(item, { toRecipients, body = '', bodyType = 'HTML', saveCopy = true } = {}) {
    const recipients = normalizeArray(toRecipients);
    if (!recipients.length) throw new Error('forward() requires toRecipients');

    const idAttrs = ExchangeAccount.toItemId(item).attributes;

    const toRecipientNodes = recipients.map((addr) => ({
      't:Mailbox': { 't:EmailAddress': { $value: addr } },
    }));

    const forwardObj = {
      't:ToRecipients': toRecipientNodes,
      't:ReferenceItemId': { attributes: idAttrs },
    };

    if (body) {
      forwardObj['t:NewBodyContent'] = { attributes: { BodyType: bodyType }, $value: body };
    }

    const args = { 'm:Items': { 't:ForwardItem': forwardObj } };
    return this.client.call('CreateItem', args, {
      opAttributes: { MessageDisposition: saveCopy ? 'SendAndSaveCopy' : 'SendOnly' },
    });
  }

  // ---- Folder functions ----
  async findFolders({
    parentFolderId = 'msgfolderroot',
    traversal = 'Shallow',
    baseShape = 'Default',
    additionalProperties = null,
  } = {}) {
    const folderShape = { 't:BaseShape': { $value: baseShape } };

    if (additionalProperties?.length) {
      folderShape['t:AdditionalProperties'] = {
        't:FieldURI': additionalProperties.map((f) => ({ attributes: { FieldURI: f } })),
      };
    }

    const args = {
      'm:FolderShape': folderShape,
      'm:ParentFolderIds': ExchangeAccount._buildParentFolderIds(parentFolderId),
    };

    const parsed = await this.client.call('FindFolder', args, { opAttributes: { Traversal: traversal } });

    const resp = parsed.response;
    const rm = resp?.['m:ResponseMessages'];
    const msgKey = rm ? Object.keys(rm).find((k) => k.toLowerCase().includes('findfolderresponsemessage')) : null;
    const msg = msgKey ? rm[msgKey] : null;

    const rootFolder = msg?.['m:RootFolder'] || null;
    const foldersNode = rootFolder?.['t:Folders'] || null;

    const folders = [];
    if (foldersNode && typeof foldersNode === 'object') {
      for (const k of Object.keys(foldersNode)) {
        const v = foldersNode[k];
        if (Array.isArray(v)) folders.push(...v);
        else if (v) folders.push(v);
      }
    }

    return { parsed, folders: folders.map((f) => Folder.fromEwsFolder(this, f)).filter(Boolean) };
  }

  async createFolder({ parentFolderId = 'msgfolderroot', displayName } = {}) {
    if (!displayName) throw new Error('createFolder: displayName required');
    const args = {
      'm:ParentFolderId': ExchangeAccount.toFolderIdArg(parentFolderId),
      'm:Folders': {
        't:Folder': {
          't:DisplayName': { $value: displayName },
          't:FolderClass': { $value: 'IPF.Note' },
        },
      },
    };
    return this.client.call('CreateFolder', args);
  }

  async renameFolder(folder, displayName) {
    if (!displayName) throw new Error('renameFolder: displayName required');

    const updates = [
      {
        't:SetFolderField': {
          't:FieldURI': { attributes: { FieldURI: 'folder:DisplayName' } },
          't:Folder': { 't:DisplayName': { $value: displayName } },
        },
      },
    ];

    const fidArg = ExchangeAccount.toFolderIdArg(folder);
    const change = { 't:Updates': updates };
    if (fidArg['t:FolderId']) change['t:FolderId'] = fidArg['t:FolderId'];
    if (fidArg['t:DistinguishedFolderId']) change['t:DistinguishedFolderId'] = fidArg['t:DistinguishedFolderId'];

    const args = { 'm:FolderChanges': { 't:FolderChange': change } };
    return this.client.call('UpdateFolder', args);
  }

  async deleteFolder(folder, deleteType = 'HardDelete') {
    const fidArg = ExchangeAccount.toFolderIdArg(folder);
    const args = { 'm:FolderIds': fidArg };
    return this.client.call('DeleteFolder', args, { opAttributes: { DeleteType: deleteType } });
  }

  // ---- Attachments (EWS) ----
  async listAttachments(item) {
    const { message } = await this.getItem(item, {
      baseShape: 'Default',
      additionalProperties: ['item:Attachments'],
    });

    const atts = message?.['t:Attachments'] || message?.Attachments || null;
    if (!atts) return [];

    const out = [];
    for (const k of Object.keys(atts)) {
      const v = atts[k];
      if (Array.isArray(v)) out.push(...v);
      else if (v) out.push(v);
    }
    return out;
  }

  async getAttachment(attachmentId) {
    if (!attachmentId) throw new Error('getAttachment: attachmentId required');
    const args = { 'm:AttachmentIds': { 't:AttachmentId': { attributes: { Id: attachmentId } } } };
    const parsed = await this.client.call('GetAttachment', args);

    const resp = parsed.response;
    const rm = resp?.['m:ResponseMessages'];
    const msgKey = rm ? Object.keys(rm).find((k) => k.toLowerCase().includes('getattachmentresponsemessage')) : null;
    const msg = msgKey ? rm[msgKey] : null;

    const attachmentsNode = msg?.['m:Attachments'] || null;
    const att =
      attachmentsNode?.['t:FileAttachment'] ||
      attachmentsNode?.FileAttachment ||
      attachmentsNode?.['t:ItemAttachment'] ||
      attachmentsNode?.ItemAttachment ||
      null;

    return { parsed, attachment: att };
  }

  async addFileAttachmentToDraft(item, { name, contentBytes, contentType } = {}) {
    if (!item) throw new Error('addFileAttachmentToDraft: item required');
    if (!name) throw new Error('addFileAttachmentToDraft: name required');
    if (!contentBytes) throw new Error('addFileAttachmentToDraft: contentBytes (base64) required');

    const args = {
      'm:ParentItemId': { 't:ItemId': ExchangeAccount.toItemId(item) },
      'm:Attachments': {
        't:FileAttachment': {
          't:Name': { $value: name },
          ...(contentType ? { 't:ContentType': { $value: contentType } } : {}),
          't:Content': { $value: contentBytes },
        },
      },
    };

    return this.client.call('CreateAttachment', args);
  }

  async deleteAttachment(attachmentId) {
    if (!attachmentId) throw new Error('deleteAttachment: attachmentId required');
    const args = { 'm:AttachmentIds': { 't:AttachmentId': { attributes: { Id: attachmentId } } } };
    return this.client.call('DeleteAttachment', args);
  }

  // ---- Path resolving (EWS best-effort) ----
  async resolveFolderPath(path, { startFolderId = 'inbox' } = {}) {
    if (!path) throw new Error('resolveFolderPath(EWS): path required');
    const parts = String(path)
      .split('/')
      .map((p) => p.trim())
      .filter(Boolean);

    let current = startFolderId;
    for (const part of parts) {
      const { folders } = await this.findFolders({
        parentFolderId: current,
        baseShape: 'Default',
        additionalProperties: ['folder:DisplayName'],
      });
      const found = folders.find((f) => (f.displayName || '').toLowerCase() === part.toLowerCase());
      if (!found) throw new Error(`resolveFolderPath(EWS): folder not found: ${part}`);
      current = found;
    }
    return current;
  }
}

// Convenience factory
export function createExchangeAccount({
  ewsUrl,
  primarySmtpAddress,
  version = 'Exchange2013',
  credentials = null,
  extraHeaders = {},
} = {}) {
  const client = new EwsClient({ url: ewsUrl, version, credentials, extraHeaders });
  return new ExchangeAccount({ primarySmtpAddress, client });
}

//------------------------------------------------------------------------------
// Graph Client (new)
//------------------------------------------------------------------------------

export class GraphError extends Error {
  constructor(message, { status, data, headers } = {}) {
    super(message);
    this.name = 'GraphError';
    this.status = status || null;
    this.data = data ?? null;
    this.headers = headers || null;
  }
}

export class GraphClient {
  constructor({
    accessToken,
    baseUrl = 'https://graph.microsoft.com/v1.0',
    user = 'me',
    extraHeaders = {},
    maxRetries = 2,
    retryBaseDelayMs = 400,
    retryMaxDelayMs = 5000,
  } = {}) {
    if (!accessToken) throw new Error('GraphClient: accessToken is required');
    this.accessToken = accessToken;
    this.baseUrl = baseUrl;
    this.user = user;
    this.extraHeaders = extraHeaders;
    this.maxRetries = Number.isFinite(maxRetries) ? maxRetries : 2;
    this.retryBaseDelayMs = Number.isFinite(retryBaseDelayMs) ? retryBaseDelayMs : 400;
    this.retryMaxDelayMs = Number.isFinite(retryMaxDelayMs) ? retryMaxDelayMs : 5000;
  }

  setAccessToken(accessToken) {
    if (!accessToken) throw new Error('setAccessToken: accessToken required');
    this.accessToken = accessToken;
  }

  _headers(additional = {}) {
    return {
      Authorization: `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
      ...this.extraHeaders,
      ...additional,
    };
  }

  _isRetryableHttp(status) {
    return status === 408 || status === 429 || (status >= 500 && status <= 504);
  }

  _parseRetryAfterMs(res) {
    const ra = res?.headers?.get?.('Retry-After');
    if (!ra) return null;
    const secs = Number(ra);
    if (Number.isFinite(secs)) return Math.max(0, secs * 1000);
    const dt = Date.parse(ra);
    if (!Number.isNaN(dt)) return Math.max(0, dt - Date.now());
    return null;
  }

  async request(method, path, { query, body, headers } = {}) {
    let attempt = 0;
    let lastErr = null;

    while (attempt <= this.maxRetries) {
      const url = new URL(this.baseUrl + path);

      if (query && typeof query === 'object') {
        for (const [k, v] of Object.entries(query)) {
          if (v === undefined || v === null) continue;
          url.searchParams.set(k, String(v));
        }
      }

      try {
        const res = await fetch(url.toString(), {
          method,
          headers: this._headers(headers),
          body: body != null ? JSON.stringify(body) : undefined,
        });

        const text = await res.text().catch(() => '');
        let data = null;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          data = text || null;
        }

        if (!res.ok) {
          const err = new GraphError(`Graph HTTP ${res.status}: ${res.statusText}`, {
            status: res.status,
            data,
            headers: Object.fromEntries(res.headers.entries()),
          });

          if (!this._isRetryableHttp(res.status) || attempt >= this.maxRetries) throw err;

          const raMs = this._parseRetryAfterMs(res);
          const base = Math.min(this.retryMaxDelayMs, this.retryBaseDelayMs * Math.pow(2, attempt));
          const jitter = Math.floor(Math.random() * 150);
          await sleep(Math.max(raMs ?? 0, base + jitter));
          attempt++;
          lastErr = err;
          continue;
        }

        return data;
      } catch (e) {
        lastErr = e;
        const retryable = e instanceof TypeError;
        if (!retryable || attempt >= this.maxRetries) throw e;

        const base = Math.min(this.retryMaxDelayMs, this.retryBaseDelayMs * Math.pow(2, attempt));
        const jitter = Math.floor(Math.random() * 150);
        await sleep(base + jitter);
        attempt++;
      }
    }

    throw lastErr || new Error('Graph: request failed');
  }

  async requestAll(path, { query, headers } = {}) {
    const out = [];
    let nextUrl = null;

    let data = await this.request('GET', path, { query, headers });
    if (data?.value && Array.isArray(data.value)) out.push(...data.value);
    nextUrl = data?.['@odata.nextLink'] || null;

    while (nextUrl) {
      const res = await fetch(nextUrl, { method: 'GET', headers: this._headers(headers) });
      const text = await res.text().catch(() => '');
      let chunk = null;
      try {
        chunk = text ? JSON.parse(text) : null;
      } catch {
        chunk = text || null;
      }

      if (!res.ok) {
        throw new GraphError(`Graph HTTP ${res.status}: ${res.statusText}`, {
          status: res.status,
          data: chunk,
          headers: Object.fromEntries(res.headers.entries()),
        });
      }

      if (chunk?.value && Array.isArray(chunk.value)) out.push(...chunk.value);
      nextUrl = chunk?.['@odata.nextLink'] || null;
    }

    return out;
  }

  // ---- Folders ----
  async listFolders({ top = 100 } = {}) {
    return this.request('GET', `/${this.user}/mailFolders`, { query: { $top: top } });
  }

  async listFoldersAll() {
    return this.requestAll(`/${this.user}/mailFolders`);
  }

  async getFolder(folderId) {
    if (!folderId) throw new Error('getFolder: folderId required');
    return this.request('GET', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}`);
  }

  async createFolder({ parentFolderId = null, displayName } = {}) {
    if (!displayName) throw new Error('createFolder: displayName required');
    const base = parentFolderId
      ? `/${this.user}/mailFolders/${encodeURIComponent(parentFolderId)}/childFolders`
      : `/${this.user}/mailFolders`;
    return this.request('POST', base, { body: { displayName } });
  }

  async renameFolder(folderId, displayName) {
    if (!folderId) throw new Error('renameFolder: folderId required');
    if (!displayName) throw new Error('renameFolder: displayName required');
    return this.request('PATCH', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}`, {
      body: { displayName },
    });
  }

  async deleteFolder(folderId) {
    if (!folderId) throw new Error('deleteFolder: folderId required');
    return this.request('DELETE', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}`);
  }

  async resolveFolderPath(path, { startFolderId = 'inbox' } = {}) {
    if (!path) throw new Error('resolveFolderPath: path required');
    const parts = String(path)
      .split('/')
      .map((p) => p.trim())
      .filter(Boolean);

    let currentId = startFolderId;
    for (const part of parts) {
      const data = await this.request(
        'GET',
        `/${this.user}/mailFolders/${encodeURIComponent(currentId)}/childFolders`,
        { query: { $top: 200 } }
      );
      const found = (data?.value || []).find((f) => (f.displayName || '').toLowerCase() === part.toLowerCase());
      if (!found?.id) throw new Error(`resolveFolderPath: folder not found: ${part}`);
      currentId = found.id;
    }
    return currentId;
  }

  // ---- Messages ----
  async listMessages({
    folderId = 'inbox',
    top = 50,
    orderBy = 'receivedDateTime desc',
    select =
      'id,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview,parentFolderId,categories,importance,flag,conversationId,internetMessageId',
  } = {}) {
    return this.request('GET', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}/messages`, {
      query: { $top: top, $orderby: orderBy, $select: select },
    });
  }

  async searchMessages({
    folderId = 'inbox',
    search,
    top = 50,
    select =
      'id,subject,from,toRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview,parentFolderId,categories,importance,flag,conversationId',
  } = {}) {
    if (!search) throw new Error('searchMessages: search is required');

    return this.request('GET', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}/messages`, {
      query: {
        $search: `"${String(search).replace(/\"/g, '\\"')}"`,
        $top: top,
        $select: select,
        $count: 'true',
      },
      headers: { ConsistencyLevel: 'eventual' },
    });
  }

  async filterMessages({
    folderId = 'inbox',
    filter,
    top = 50,
    orderBy = 'receivedDateTime desc',
    select =
      'id,subject,from,toRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview,parentFolderId,categories,importance,flag,conversationId',
  } = {}) {
    if (!filter) throw new Error('filterMessages: filter is required');

    return this.request('GET', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}/messages`, {
      query: { $filter: filter, $top: top, $orderby: orderBy, $select: select },
    });
  }

  async getMessage(
    messageId,
    {
      select =
        'id,subject,from,toRecipients,ccRecipients,bccRecipients,receivedDateTime,sentDateTime,isRead,hasAttachments,body,parentFolderId,conversationId,internetMessageId,categories,importance,flag',
    } = {}
  ) {
    if (!messageId) throw new Error('getMessage: messageId required');
    return this.request('GET', `/${this.user}/messages/${encodeURIComponent(messageId)}`, {
      query: { $select: select },
    });
  }

  async updateMessage(messageId, patch = {}) {
    if (!messageId) throw new Error('updateMessage: messageId required');
    if (!patch || typeof patch !== 'object') throw new Error('updateMessage: patch object required');
    return this.request('PATCH', `/${this.user}/messages/${encodeURIComponent(messageId)}`, { body: patch });
  }

  async markRead(messageId, isRead = true) {
    return this.updateMessage(messageId, { isRead: !!isRead });
  }

  async moveMessage(messageId, destinationFolderId) {
    if (!messageId) throw new Error('moveMessage: messageId required');
    if (!destinationFolderId) throw new Error('moveMessage: destinationFolderId required');
    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/move`, {
      body: { destinationId: destinationFolderId },
    });
  }

  async copyMessage(messageId, destinationFolderId) {
    if (!messageId) throw new Error('copyMessage: messageId required');
    if (!destinationFolderId) throw new Error('copyMessage: destinationFolderId required');
    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/copy`, {
      body: { destinationId: destinationFolderId },
    });
  }

  async deleteMessage(messageId) {
    if (!messageId) throw new Error('deleteMessage: messageId required');
    return this.request('DELETE', `/${this.user}/messages/${encodeURIComponent(messageId)}`);
  }

  // ---- Drafts ----
  async createDraft({
    subject = '',
    body = '',
    bodyType = 'HTML',
    to = [],
    cc = [],
    bcc = [],
    categories = undefined,
    importance = undefined,
    flag = undefined,
  } = {}) {
    const mkRecipients = (arr) => normalizeArray(arr).map((addr) => ({ emailAddress: { address: addr } }));

    const msg = {
      subject: subject ?? '',
      body: { contentType: bodyType, content: body ?? '' },
      toRecipients: mkRecipients(to),
      ccRecipients: mkRecipients(cc),
      bccRecipients: mkRecipients(bcc),
    };
    if (categories !== undefined) msg.categories = normalizeArray(categories);
    if (importance !== undefined) msg.importance = importance;
    if (flag !== undefined) msg.flag = flag;

    return this.request('POST', `/${this.user}/messages`, { body: msg });
  }

  async sendDraft(messageId) {
    if (!messageId) throw new Error('sendDraft: messageId required');
    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/send`);
  }

  async sendMail({ subject, body, bodyType = 'HTML', to = [], cc = [], bcc = [], saveToSentItems = true } = {}) {
    if (!subject) throw new Error('sendMail: subject required');
    if (body == null) throw new Error('sendMail: body required');

    const mkRecipients = (arr) => normalizeArray(arr).map((addr) => ({ emailAddress: { address: addr } }));

    return this.request('POST', `/${this.user}/sendMail`, {
      body: {
        message: {
          subject,
          body: { contentType: bodyType, content: body },
          toRecipients: mkRecipients(to),
          ccRecipients: mkRecipients(cc),
          bccRecipients: mkRecipients(bcc),
        },
        saveToSentItems: !!saveToSentItems,
      },
    });
  }

  // ---- Rich reply/forward via draft ----
  async createReplyDraft(messageId, { replyAll = false } = {}) {
    if (!messageId) throw new Error('createReplyDraft: messageId required');
    const endpoint = replyAll ? 'createReplyAll' : 'createReply';
    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/${endpoint}`, { body: {} });
  }

  async createForwardDraft(messageId, { toRecipients = [] } = {}) {
    if (!messageId) throw new Error('createForwardDraft: messageId required');
    const mk = (arr) => normalizeArray(arr).map((addr) => ({ emailAddress: { address: addr } }));
    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/createForward`, {
      body: { toRecipients: mk(toRecipients) },
    });
  }

  async replyRich(messageId, { replyAll = false, body = '', bodyType = 'HTML' } = {}) {
    const draft = await this.createReplyDraft(messageId, { replyAll });
    const draftId = draft?.id;
    if (!draftId) throw new Error('replyRich: Graph did not return draft id');

    if (body != null) {
      await this.updateMessage(draftId, { body: { contentType: bodyType, content: String(body) } });
    }

    await this.sendDraft(draftId);
    return { draftId };
  }

  async forwardRich(messageId, { to = [], body = '', bodyType = 'HTML' } = {}) {
    const draft = await this.createForwardDraft(messageId, { toRecipients: to });
    const draftId = draft?.id;
    if (!draftId) throw new Error('forwardRich: Graph did not return draft id');

    const patch = {};
    if (body != null) patch.body = { contentType: bodyType, content: String(body) };
    if (to != null) {
      patch.toRecipients = normalizeArray(to).map((addr) => ({ emailAddress: { address: addr } }));
    }
    if (Object.keys(patch).length) await this.updateMessage(draftId, patch);

    await this.sendDraft(draftId);
    return { draftId };
  }

  // ---- Categories / flag ----
  async getCategories(messageId) {
    const msg = await this.getMessage(messageId, { select: 'id,categories' });
    return msg?.categories || [];
  }

  async setCategories(messageId, categories = []) {
    return this.updateMessage(messageId, { categories: normalizeArray(categories) });
  }

  async addCategories(messageId, categories = []) {
    const current = await this.getCategories(messageId);
    const merged = stableUnique([...current, ...normalizeArray(categories)]);
    return this.setCategories(messageId, merged);
  }

  async removeCategories(messageId, categories = []) {
    const current = await this.getCategories(messageId);
    const rem = new Set(normalizeArray(categories));
    return this.setCategories(
      messageId,
      current.filter((c) => !rem.has(c))
    );
  }

  async setFlag(messageId, flagObj) {
    if (!flagObj || typeof flagObj !== 'object') throw new Error('setFlag: flag object required');
    return this.updateMessage(messageId, { flag: flagObj });
  }

  async clearFlag(messageId) {
    return this.updateMessage(messageId, { flag: { flagStatus: 'notFlagged' } });
  }

  // ---- Attachments ----
  async listAttachments(messageId) {
    if (!messageId) throw new Error('listAttachments: messageId required');
    return this.request('GET', `/${this.user}/messages/${encodeURIComponent(messageId)}/attachments`, {
      query: { $top: 200 },
    });
  }

  async getAttachment(messageId, attachmentId) {
    if (!messageId) throw new Error('getAttachment: messageId required');
    if (!attachmentId) throw new Error('getAttachment: attachmentId required');
    return this.request(
      'GET',
      `/${this.user}/messages/${encodeURIComponent(messageId)}/attachments/${encodeURIComponent(attachmentId)}`
    );
  }

  async addFileAttachmentToDraft(messageId, { name, contentBytes, contentType } = {}) {
    if (!messageId) throw new Error('addFileAttachmentToDraft: messageId required');
    if (!name) throw new Error('addFileAttachmentToDraft: name required');
    if (!contentBytes) throw new Error('addFileAttachmentToDraft: contentBytes (base64) required');

    const body = {
      '@odata.type': '#microsoft.graph.fileAttachment',
      name,
      contentBytes,
    };
    if (contentType) body.contentType = contentType;

    return this.request('POST', `/${this.user}/messages/${encodeURIComponent(messageId)}/attachments`, { body });
  }

  async deleteAttachment(messageId, attachmentId) {
    if (!messageId) throw new Error('deleteAttachment: messageId required');
    if (!attachmentId) throw new Error('deleteAttachment: attachmentId required');
    return this.request(
      'DELETE',
      `/${this.user}/messages/${encodeURIComponent(messageId)}/attachments/${encodeURIComponent(attachmentId)}`
    );
  }

  // Large attachment upload session + chunk upload
  async createAttachmentUploadSession(messageId, { name, size, contentType } = {}) {
    if (!messageId) throw new Error('createAttachmentUploadSession: messageId required');
    if (!name) throw new Error('createAttachmentUploadSession: name required');
    if (!Number.isFinite(size) || size <= 0) throw new Error('createAttachmentUploadSession: valid size required');

    const body = {
      AttachmentItem: {
        attachmentType: 'file',
        name,
        size,
      },
    };
    if (contentType) body.AttachmentItem.contentType = contentType;

    return this.request(
      'POST',
      `/${this.user}/messages/${encodeURIComponent(messageId)}/attachments/createUploadSession`,
      { body }
    );
  }

  async uploadToSession(uploadUrl, data, { chunkSize = 320 * 1024 } = {}) {
    if (!uploadUrl) throw new Error('uploadToSession: uploadUrl required');
    if (!data) throw new Error('uploadToSession: data required');

    const bytes =
      data instanceof Uint8Array
        ? data
        : data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : (() => {
            throw new Error('uploadToSession: data must be Uint8Array or ArrayBuffer');
          })();

    const total = bytes.byteLength;
    let start = 0;
    let lastResponse = null;

    while (start < total) {
      const end = Math.min(start + chunkSize, total) - 1;
      const chunk = bytes.slice(start, end + 1);

      const res = await fetch(uploadUrl, {
        method: 'PUT',
        headers: {
          'Content-Length': String(chunk.byteLength),
          'Content-Range': `bytes ${start}-${end}/${total}`,
        },
        body: chunk,
      });

      const text = await res.text().catch(() => '');
      let json = null;
      try {
        json = text ? JSON.parse(text) : null;
      } catch {
        json = text || null;
      }

      if (!res.ok) {
        throw new GraphError(`UploadSession HTTP ${res.status}: ${res.statusText}`, {
          status: res.status,
          data: json,
          headers: Object.fromEntries(res.headers.entries()),
        });
      }

      lastResponse = json;
      start = end + 1;
    }

    return lastResponse;
  }

  async addLargeAttachmentToDraft(messageId, { name, bytes, contentType } = {}) {
    if (!messageId) throw new Error('addLargeAttachmentToDraft: messageId required');
    if (!name) throw new Error('addLargeAttachmentToDraft: name required');
    if (!bytes) throw new Error('addLargeAttachmentToDraft: bytes required');

    const size = bytes instanceof Uint8Array ? bytes.byteLength : bytes instanceof ArrayBuffer ? bytes.byteLength : null;
    if (!Number.isFinite(size)) throw new Error('addLargeAttachmentToDraft: bytes must be Uint8Array or ArrayBuffer');

    const session = await this.createAttachmentUploadSession(messageId, { name, size, contentType });
    const uploadUrl = session?.uploadUrl;
    if (!uploadUrl) throw new Error('addLargeAttachmentToDraft: Graph did not return uploadUrl');

    return this.uploadToSession(uploadUrl, bytes);
  }

  // MIME (.eml)
  async getMessageMime(messageId) {
    if (!messageId) throw new Error('getMessageMime: messageId required');

    const url = new URL(this.baseUrl + `/${this.user}/messages/${encodeURIComponent(messageId)}/$value`);
    const res = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${this.accessToken}`,
        Accept: 'message/rfc822',
        ...this.extraHeaders,
      },
    });

    const text = await res.text().catch(() => '');
    if (!res.ok) {
      throw new GraphError(`Graph MIME HTTP ${res.status}: ${res.statusText}`, {
        status: res.status,
        data: text,
        headers: Object.fromEntries(res.headers.entries()),
      });
    }

    return text;
  }

  // People/Contacts (autocomplete)
  async findPeople(query, { top = 10 } = {}) {
    if (!query) throw new Error('findPeople: query required');
    return this.request('GET', `/${this.user}/people`, {
      query: { $search: `"${String(query).replace(/\"/g, '\\"')}"`, $top: top },
      headers: { ConsistencyLevel: 'eventual' },
    });
  }

  async searchContacts(query, { top = 10 } = {}) {
    if (!query) throw new Error('searchContacts: query required');
    return this.request('GET', `/${this.user}/contacts`, {
      query: { $search: `"${String(query).replace(/\"/g, '\\"')}"`, $top: top },
      headers: { ConsistencyLevel: 'eventual' },
    });
  }

  // Delta
  async deltaMessages({ folderId = 'inbox', deltaLink = null, nextLink = null, select = null } = {}) {
    if (nextLink) {
      const res = await fetch(nextLink, { method: 'GET', headers: this._headers() });
      const text = await res.text().catch(() => '');
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = text || null;
      }
      if (!res.ok) throw new GraphError(`Graph HTTP ${res.status}: ${res.statusText}`, { status: res.status, data });
      return {
        value: data?.value || [],
        deltaLink: data?.['@odata.deltaLink'] || null,
        nextLink: data?.['@odata.nextLink'] || null,
        raw: data,
      };
    }

    if (deltaLink) {
      const res = await fetch(deltaLink, { method: 'GET', headers: this._headers() });
      const text = await res.text().catch(() => '');
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = text || null;
      }
      if (!res.ok) throw new GraphError(`Graph HTTP ${res.status}: ${res.statusText}`, { status: res.status, data });
      return {
        value: data?.value || [],
        deltaLink: data?.['@odata.deltaLink'] || null,
        nextLink: data?.['@odata.nextLink'] || null,
        raw: data,
      };
    }

    const query = {};
    if (select) query.$select = select;

    const data = await this.request('GET', `/${this.user}/mailFolders/${encodeURIComponent(folderId)}/messages/delta`, {
      query,
    });

    return {
      value: data?.value || [],
      deltaLink: data?.['@odata.deltaLink'] || null,
      nextLink: data?.['@odata.nextLink'] || null,
      raw: data,
    };
  }
}

//------------------------------------------------------------------------------
// ActiveSync client (EAS / Mobile Sync)
//------------------------------------------------------------------------------

function base64EncodeUtf8(str) {
  const s = String(str || '');
  if (typeof btoa === 'function') {
    const bytes = new TextEncoder().encode(s);
    let bin = '';
    for (const b of bytes) bin += String.fromCharCode(b);
    return btoa(bin);
  }
  return Buffer.from(s, 'utf8').toString('base64');
}

function asString(v, fallback = '') {
  const s = String(v ?? '').trim();
  return s || fallback;
}

function asBool(v, fallback = false) {
  if (typeof v === 'boolean') return v;
  if (v == null) return !!fallback;
  const s = String(v).trim().toLowerCase();
  if (!s) return !!fallback;
  return s === '1' || s === 'true' || s === 'yes' || s === 'on';
}

function asInt(v, fallback = 0, { min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER } = {}) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  const out = Math.round(n);
  return Math.min(max, Math.max(min, out));
}

function easMbUIntEncode(value) {
  let n = Number(value);
  if (!Number.isFinite(n) || n < 0) n = 0;
  n = Math.floor(n);
  const out = [];
  do {
    out.unshift(n & 0x7f);
    n >>= 7;
  } while (n > 0);
  for (let i = 0; i < out.length - 1; i++) out[i] |= 0x80;
  return out;
}

function easMbUIntDecode(bytes, cursor) {
  let value = 0;
  let b = 0;
  do {
    if (cursor.i >= bytes.length) throw new Error('WBXML mb_u_int32 overflow');
    b = bytes[cursor.i++];
    value = (value << 7) | (b & 0x7f);
  } while (b & 0x80);
  return value >>> 0;
}

const EAS_CODE_PAGES = {
  AirSync: {
    page: 0,
    tags: {
      Sync: 0x05,
      Responses: 0x06,
      Add: 0x07,
      Change: 0x08,
      Delete: 0x09,
      Fetch: 0x0a,
      SyncKey: 0x0b,
      ClientId: 0x0c,
      ServerId: 0x0d,
      Status: 0x0e,
      Collection: 0x0f,
      Class: 0x10,
      Version: 0x11,
      CollectionId: 0x12,
      GetChanges: 0x13,
      MoreAvailable: 0x14,
      WindowSize: 0x15,
      Commands: 0x16,
      Options: 0x17,
      FilterType: 0x18,
      Truncation: 0x19,
      RtfTruncation: 0x1a,
      Conflict: 0x1b,
      Collections: 0x1c,
      ApplicationData: 0x1d,
      DeletesAsMoves: 0x1e,
      NotifyGUID: 0x1f,
      Supported: 0x20,
      SoftDelete: 0x21,
      MIMESupport: 0x22,
      MIMETruncation: 0x23,
      Wait: 0x24,
      Limit: 0x25,
      Partial: 0x26,
      ConversationMode: 0x27,
      MaxItems: 0x28,
      HeartbeatInterval: 0x29,
    },
  },
  Email: {
    page: 2,
    tags: {
      Attachment: 0x05,
      Attachments: 0x06,
      AttName: 0x07,
      AttSize: 0x08,
      AttOid: 0x09,
      AttMethod: 0x0a,
      AttRemoved: 0x0b,
      Body: 0x0c,
      BodySize: 0x0d,
      BodyTruncated: 0x0e,
      DateReceived: 0x0f,
      DisplayName: 0x10,
      DisplayTo: 0x11,
      Importance: 0x12,
      MessageClass: 0x13,
      Subject: 0x14,
      Read: 0x15,
      To: 0x16,
      Cc: 0x17,
      From: 0x18,
      ReplyTo: 0x19,
      AllDayEvent: 0x1a,
      Categories: 0x1b,
      Category: 0x1c,
      DtStamp: 0x1d,
      EndTime: 0x1e,
      InstanceType: 0x1f,
      BusyStatus: 0x20,
      Location: 0x21,
      MeetingRequest: 0x22,
      Organizer: 0x23,
      RecurrenceId: 0x24,
      Reminder: 0x25,
      ResponseRequested: 0x26,
      Recurrences: 0x27,
      Recurrence: 0x28,
      Type: 0x29,
      Until: 0x2a,
      Occurrences: 0x2b,
      Interval: 0x2c,
      DayOfWeek: 0x2d,
      DayOfMonth: 0x2e,
      WeekOfMonth: 0x2f,
      MonthOfYear: 0x30,
      StartTime: 0x31,
      Sensitivity: 0x32,
      TimeZone: 0x33,
      GlobalObjId: 0x34,
      ThreadTopic: 0x35,
      MIMEDATA: 0x36,
      MIMETruncated: 0x37,
      MIMESize: 0x38,
      InternetCPID: 0x39,
      Flag: 0x3a,
      FlagStatus: 0x3b,
      ContentClass: 0x3c,
      FlagType: 0x3d,
      CompleteTime: 0x3e,
      DisallowNewTimeProposal: 0x3f,
    },
  },
  FolderHierarchy: {
    page: 7,
    tags: {
      Folders: 0x05,
      Folder: 0x06,
      DisplayName: 0x07,
      ServerId: 0x08,
      ParentId: 0x09,
      Type: 0x0a,
      Response: 0x0b,
      Status: 0x0c,
      ContentClass: 0x0d,
      Changes: 0x0e,
      Add: 0x0f,
      Delete: 0x10,
      Update: 0x11,
      SyncKey: 0x12,
      FolderCreate: 0x13,
      FolderDelete: 0x14,
      FolderUpdate: 0x15,
      FolderSync: 0x16,
      Count: 0x17,
    },
  },
  AirSyncBase: {
    page: 17,
    tags: {
      BodyPreference: 0x05,
      Type: 0x06,
      TruncationSize: 0x07,
      AllOrNone: 0x08,
      Body: 0x0a,
      Data: 0x0b,
      EstimatedDataSize: 0x0d,
      Truncated: 0x0e,
      Attachments: 0x0f,
      Attachment: 0x10,
      DisplayName: 0x11,
      FileReference: 0x12,
      Method: 0x13,
      ContentId: 0x14,
      ContentLocation: 0x15,
      IsInline: 0x16,
      NativeBodyType: 0x17,
      ContentType: 0x18,
      Preview: 0x19,
      BodyPartPreference: 0x1a,
      BodyPart: 0x1b,
      Status: 0x1c,
    },
  },
};

const EAS_TAG_TO_TOKEN = {};
const EAS_TOKEN_TO_TAG = {};
for (const [ns, cfg] of Object.entries(EAS_CODE_PAGES)) {
  EAS_TOKEN_TO_TAG[cfg.page] = EAS_TOKEN_TO_TAG[cfg.page] || {};
  for (const [tagName, token] of Object.entries(cfg.tags)) {
    const fq = `${ns}:${tagName}`;
    EAS_TAG_TO_TOKEN[fq] = { page: cfg.page, token };
    EAS_TOKEN_TO_TAG[cfg.page][token] = fq;
  }
}

function easElem(name, children = []) {
  const arr = Array.isArray(children) ? children : [children];
  return { name, children: arr.filter((x) => x != null) };
}

function easChild(node, name) {
  if (!node || !Array.isArray(node.children)) return null;
  return node.children.find((c) => c && typeof c === 'object' && c.name === name) || null;
}

function easChildren(node, name) {
  if (!node || !Array.isArray(node.children)) return [];
  return node.children.filter((c) => c && typeof c === 'object' && c.name === name);
}

function easNodeText(node) {
  if (!node || !Array.isArray(node.children)) return '';
  const parts = [];
  for (const c of node.children) {
    if (typeof c === 'string') parts.push(c);
  }
  return parts.join('').trim();
}

function easText(node, childName, fallback = '') {
  const c = easChild(node, childName);
  if (!c) return fallback;
  const t = easNodeText(c);
  return t || fallback;
}

function easEncode(rootNode) {
  if (!rootNode || typeof rootNode !== 'object' || !rootNode.name) {
    throw new Error('ActiveSync WBXML encode: invalid root node');
  }

  const enc = new TextEncoder();
  const out = [0x03, 0x01, 0x6a, 0x00]; // WBXML 1.3, unknown public id, UTF-8, empty string table
  let currentPage = 0;

  const writeText = (text) => {
    out.push(0x03); // STR_I
    const bytes = enc.encode(String(text ?? ''));
    for (const b of bytes) out.push(b);
    out.push(0x00);
  };

  const writeNode = (node) => {
    const def = EAS_TAG_TO_TOKEN[node.name];
    if (!def) throw new Error(`ActiveSync WBXML encode: unknown tag ${node.name}`);

    if (def.page !== currentPage) {
      out.push(0x00); // SWITCH_PAGE
      out.push(def.page & 0xff);
      currentPage = def.page;
    }

    const kids = Array.isArray(node.children) ? node.children : [];
    const hasContent = kids.length > 0;
    let token = def.token;
    if (hasContent) token |= 0x40;
    out.push(token);

    if (!hasContent) return;
    for (const child of kids) {
      if (typeof child === 'string') writeText(child);
      else writeNode(child);
    }
    out.push(0x01); // END
  };

  writeNode(rootNode);
  return new Uint8Array(out);
}

function easDecode(bytesInput) {
  const bytes = bytesInput instanceof Uint8Array ? bytesInput : new Uint8Array(bytesInput || []);
  if (!bytes.length) return null;

  const dec = new TextDecoder('utf-8');
  const cur = { i: 0 };

  // header
  cur.i += 1; // version
  easMbUIntDecode(bytes, cur); // public id
  easMbUIntDecode(bytes, cur); // charset
  const strTblLen = easMbUIntDecode(bytes, cur);
  cur.i += strTblLen;

  let currentPage = 0;
  const roots = [];
  const stack = [];

  const append = (value) => {
    if (!stack.length) return;
    stack[stack.length - 1].children.push(value);
  };

  const readInlineString = () => {
    const start = cur.i;
    while (cur.i < bytes.length && bytes[cur.i] !== 0x00) cur.i += 1;
    const view = bytes.slice(start, cur.i);
    cur.i += 1;
    return dec.decode(view);
  };

  const skipAttributes = () => {
    while (cur.i < bytes.length) {
      const t = bytes[cur.i++];
      if (t === 0x01) return; // END
      if (t === 0x03) {
        // STR_I
        while (cur.i < bytes.length && bytes[cur.i] !== 0x00) cur.i += 1;
        cur.i += 1;
      } else if (t === 0x83) {
        // STR_T
        easMbUIntDecode(bytes, cur);
      } else if (t === 0x00) {
        cur.i += 1; // page index
      } else if (t === 0xc3) {
        // OPAQUE
        const l = easMbUIntDecode(bytes, cur);
        cur.i += l;
      }
    }
  };

  while (cur.i < bytes.length) {
    const token = bytes[cur.i++];

    if (token === 0x00) {
      currentPage = bytes[cur.i++] || 0;
      continue;
    }
    if (token === 0x01) {
      stack.pop();
      continue;
    }
    if (token === 0x03) {
      append(readInlineString());
      continue;
    }
    if (token === 0xc3) {
      const len = easMbUIntDecode(bytes, cur);
      const data = bytes.slice(cur.i, cur.i + len);
      cur.i += len;
      append(dec.decode(data));
      continue;
    }

    const hasAttrs = !!(token & 0x80);
    const hasContent = !!(token & 0x40);
    const tag = token & 0x3f;
    const name = EAS_TOKEN_TO_TAG[currentPage]?.[tag] || `p${currentPage}:${tag}`;
    const node = { name, children: [] };

    if (stack.length) append(node);
    else roots.push(node);

    if (hasAttrs) skipAttributes();
    if (hasContent) stack.push(node);
  }

  return roots[0] || null;
}

function parseMailboxAddress(raw) {
  const text = asString(raw || '');
  if (!text) return { name: '', address: '' };

  const angle = text.match(/^(.*)<([^>]+)>$/);
  if (angle) {
    const name = asString(angle[1].replace(/"/g, ''), '');
    const address = asString(angle[2], '');
    return { name: name || address, address };
  }

  const mail = text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i);
  if (mail) {
    const address = asString(mail[0], '');
    const name = asString(text.replace(address, '').replace(/[<>()"]/g, ' '), '');
    return { name: name || address, address };
  }

  return { name: text, address: '' };
}

function makeActiveSyncDeviceId(seed = '') {
  const clean = String(seed || '').replace(/[^A-Za-z0-9]/g, '').slice(0, 24);
  if (clean.length >= 8) return clean;
  const rand = (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function')
    ? crypto.randomUUID().replace(/-/g, '')
    : `${Date.now()}${Math.floor(Math.random() * 1e9)}`;
  return `NWT${rand.replace(/[^A-Za-z0-9]/g, '').slice(0, 21)}`;
}

function mapFolderAliasToType(alias) {
  const a = asString(alias || '').toLowerCase();
  if (a === 'inbox' || a === 'important') return 2;
  if (a === 'sentitems' || a === 'sent') return 5;
  if (a === 'drafts' || a === 'draft') return 3;
  if (a === 'deleteditems' || a === 'trash') return 4;
  if (a === 'outbox') return 6;
  return 0;
}

function maybeIsoDate(s) {
  const text = asString(s || '');
  if (!text) return null;
  const ms = Date.parse(text);
  return Number.isFinite(ms) ? new Date(ms).toISOString() : text;
}

export class ActiveSyncError extends Error {
  constructor(message, { status = null, data = null, headers = null } = {}) {
    super(message);
    this.name = 'ActiveSyncError';
    this.status = status;
    this.data = data;
    this.headers = headers;
  }
}

export class ActiveSyncClient {
  constructor({
    server,
    username,
    password,
    useSSL = true,
    path = 'Microsoft-Server-ActiveSync',
    protocolVersion = '14.1',
    policyKey = '0',
    deviceId = '',
    deviceType = 'Chrome',
    timeoutMs = 30000,
    maxCachedMessages = 400,
  } = {}) {
    const srv = asString(server || '');
    if (!srv) throw new Error('ActiveSyncClient: server required');

    this.server = srv.replace(/^https?:\/\//i, '').replace(/\/+$/, '');
    this.username = asString(username || '');
    this.password = String(password || '');
    this.useSSL = asBool(useSSL, true);
    this.path = asString(path || 'Microsoft-Server-ActiveSync', 'Microsoft-Server-ActiveSync').replace(/^\/+/, '');
    this.protocolVersion = asString(protocolVersion || '14.1', '14.1');
    this.policyKey = asString(policyKey || '0', '0');
    this.deviceId = makeActiveSyncDeviceId(deviceId || this.username || this.server);
    this.deviceType = asString(deviceType || 'Chrome', 'Chrome');
    this.timeoutMs = asInt(timeoutMs, 30000, { min: 5000, max: 120000 });
    this.maxCachedMessages = asInt(maxCachedMessages, 400, { min: 50, max: 2000 });

    if (!this.username || !this.password) {
      throw new Error('ActiveSyncClient: username and password required');
    }

    this._folderSyncKey = '0';
    this._foldersById = new Map();
    this._mailSyncKeys = new Map(); // folderId -> sync key
    this._messageCache = new Map(); // messageId -> message
  }

  get baseUrl() {
    const protocol = this.useSSL ? 'https' : 'http';
    return `${protocol}://${this.server}/${this.path}`;
  }

  _authHeader() {
    return `Basic ${base64EncodeUtf8(`${this.username}:${this.password}`)}`;
  }

  _commandUrl(cmd) {
    const u = new URL(this.baseUrl);
    u.searchParams.set('Cmd', asString(cmd || 'Sync'));
    u.searchParams.set('User', this.username);
    u.searchParams.set('DeviceId', this.deviceId);
    u.searchParams.set('DeviceType', this.deviceType);
    return u.toString();
  }

  _headers({ includePolicy = true, contentType = true } = {}) {
    const h = {
      Authorization: this._authHeader(),
      'MS-ASProtocolVersion': this.protocolVersion,
      'User-Agent': 'ThesenAITools-ActiveSync/1.0',
      Accept: 'application/vnd.ms-sync.wbxml, */*',
    };
    if (includePolicy && this.policyKey) h['X-MS-PolicyKey'] = this.policyKey;
    if (contentType) h['Content-Type'] = 'application/vnd.ms-sync.wbxml';
    return h;
  }

  _rememberMessage(msg) {
    if (!msg?.id) return;
    if (this._messageCache.has(msg.id)) this._messageCache.delete(msg.id);
    this._messageCache.set(msg.id, msg);
    while (this._messageCache.size > this.maxCachedMessages) {
      const oldest = this._messageCache.keys().next();
      if (oldest?.done) break;
      this._messageCache.delete(oldest.value);
    }
  }

  _rememberMessages(list = []) {
    for (const m of list) this._rememberMessage(m);
  }

  _folderRecordFromNode(node) {
    const serverId = easText(node, 'FolderHierarchy:ServerId', '');
    if (!serverId) return null;
    const type = asInt(easText(node, 'FolderHierarchy:Type', '0'), 0);
    return {
      serverId,
      parentId: easText(node, 'FolderHierarchy:ParentId', '0'),
      displayName: easText(node, 'FolderHierarchy:DisplayName', serverId),
      type,
      contentClass: easText(node, 'FolderHierarchy:ContentClass', ''),
    };
  }

  _normalizeMessage(appNode, { serverId = '', folderId = '' } = {}) {
    if (!appNode) return null;
    const subject = easText(appNode, 'Email:Subject', '(ohne Betreff)');
    const fromRaw = easText(appNode, 'Email:From', '');
    const parsedFrom = parseMailboxAddress(fromRaw);
    const bodyNode = easChild(appNode, 'AirSyncBase:Body');
    const bodyData = easText(bodyNode, 'AirSyncBase:Data', easText(appNode, 'Email:Body', ''));
    const preview = easText(bodyNode, 'AirSyncBase:Preview', bodyData.slice(0, 280));
    const received = maybeIsoDate(easText(appNode, 'Email:DateReceived', ''));
    const read = easText(appNode, 'Email:Read', '0') === '1';
    const id = asString(serverId || easText(appNode, 'AirSync:ServerId', ''), '');
    if (!id) return null;

    return {
      id,
      serverId: id,
      folderId: folderId || '',
      subject: subject || '(ohne Betreff)',
      bodyPreview: preview || '',
      receivedDateTime: received,
      sentDateTime: received,
      isRead: !!read,
      from: {
        emailAddress: {
          name: parsedFrom.name || parsedFrom.address || fromRaw || '—',
          address: parsedFrom.address || '',
        },
      },
      body: {
        contentType: 'text',
        content: bodyData || preview || '',
      },
    };
  }

  async options() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(this.baseUrl, {
        method: 'OPTIONS',
        headers: this._headers({ includePolicy: false, contentType: false }),
        signal: controller.signal,
      });
      if (res.status === 401) {
        throw new ActiveSyncError('ActiveSync auth failed (401). Username/Passwort pruefen.', {
          status: res.status,
          headers: Object.fromEntries(res.headers.entries()),
        });
      }
      if (!res.ok) {
        throw new ActiveSyncError(`ActiveSync OPTIONS failed: HTTP ${res.status}`, {
          status: res.status,
          headers: Object.fromEntries(res.headers.entries()),
        });
      }
      const hdr = Object.fromEntries(res.headers.entries());
      return {
        ok: true,
        status: res.status,
        protocolVersions: hdr['ms-asprotocolversions'] || '',
        commands: hdr['ms-asprotocolcommands'] || '',
        headers: hdr,
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async _command(cmd, node) {
    const body = node ? easEncode(node) : new Uint8Array([0x03, 0x01, 0x6a, 0x00]);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(this._commandUrl(cmd), {
        method: 'POST',
        headers: this._headers({ includePolicy: true, contentType: true }),
        body,
        signal: controller.signal,
      });

      if (res.status === 401) {
        throw new ActiveSyncError('ActiveSync auth failed (401). Username/Passwort pruefen.', {
          status: res.status,
          headers: Object.fromEntries(res.headers.entries()),
        });
      }

      if (res.status === 449) {
        throw new ActiveSyncError(
          'ActiveSync server verlangt Device Provisioning (HTTP 449). Diese Server-Richtlinie wird aktuell nicht unterstuetzt.',
          { status: res.status, headers: Object.fromEntries(res.headers.entries()) }
        );
      }

      const buf = await res.arrayBuffer().catch(() => new ArrayBuffer(0));
      const bytes = new Uint8Array(buf);
      if (!res.ok) {
        let data = null;
        try { data = new TextDecoder('utf-8').decode(bytes); } catch {}
        throw new ActiveSyncError(`ActiveSync ${cmd} failed: HTTP ${res.status}`, {
          status: res.status,
          headers: Object.fromEntries(res.headers.entries()),
          data,
        });
      }

      const root = bytes.length ? easDecode(bytes) : null;
      return {
        status: res.status,
        headers: Object.fromEntries(res.headers.entries()),
        root,
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async folderSync({ force = false } = {}) {
    if (!force && this._foldersById.size) return Array.from(this._foldersById.values());

    const req = easElem('FolderHierarchy:FolderSync', [
      easElem('FolderHierarchy:SyncKey', [this._folderSyncKey || '0']),
    ]);
    const res = await this._command('FolderSync', req);
    const root = res.root;
    if (!root || root.name !== 'FolderHierarchy:FolderSync') {
      throw new ActiveSyncError('ActiveSync FolderSync: ungueltige Antwort');
    }

    const status = easText(root, 'FolderHierarchy:Status', '');
    if (status && status !== '1') {
      throw new ActiveSyncError(`ActiveSync FolderSync status ${status}`);
    }

    const nextSyncKey = easText(root, 'FolderHierarchy:SyncKey', this._folderSyncKey || '0');
    if (nextSyncKey) this._folderSyncKey = nextSyncKey;

    const changes = easChild(root, 'FolderHierarchy:Changes');
    if (changes) {
      for (const n of easChildren(changes, 'FolderHierarchy:Add')) {
        const rec = this._folderRecordFromNode(n);
        if (rec) this._foldersById.set(rec.serverId, rec);
      }
      for (const n of easChildren(changes, 'FolderHierarchy:Update')) {
        const rec = this._folderRecordFromNode(n);
        if (rec) this._foldersById.set(rec.serverId, rec);
      }
      for (const n of easChildren(changes, 'FolderHierarchy:Delete')) {
        const id = easText(n, 'FolderHierarchy:ServerId', '');
        if (id) this._foldersById.delete(id);
      }
    }

    if (!this._foldersById.size) {
      const foldersNode = easChild(root, 'FolderHierarchy:Folders');
      if (foldersNode) {
        for (const f of easChildren(foldersNode, 'FolderHierarchy:Folder')) {
          const rec = this._folderRecordFromNode(f);
          if (rec) this._foldersById.set(rec.serverId, rec);
        }
      }
    }

    return Array.from(this._foldersById.values());
  }

  async listFolders() {
    await this.folderSync({ force: !this._foldersById.size });
    return {
      items: Array.from(this._foldersById.values()),
    };
  }

  async resolveFolderId(folderHint = 'inbox') {
    const hint = asString(folderHint || 'inbox', 'inbox').toLowerCase();
    await this.folderSync();

    if (this._foldersById.has(folderHint)) return folderHint;

    const folders = Array.from(this._foldersById.values());
    const wantedType = mapFolderAliasToType(hint);
    if (wantedType) {
      const byType = folders.find((f) => f.type === wantedType);
      if (byType?.serverId) return byType.serverId;
    }

    const nameAliases = {
      inbox: ['inbox', 'eingang', 'posteingang'],
      sentitems: ['sent', 'gesendet'],
      drafts: ['draft', 'entw'],
      deleteditems: ['deleted', 'trash', 'papierkorb'],
      outbox: ['outbox', 'ausgang'],
    };
    const aliases = nameAliases[hint] || [hint];
    const byName = folders.find((f) => {
      const n = asString(f.displayName || '').toLowerCase();
      return aliases.some((a) => n.includes(a));
    });
    if (byName?.serverId) return byName.serverId;

    if (hint && this._foldersById.has(hint)) return hint;
    throw new ActiveSyncError(`ActiveSync folder not found: ${folderHint}`);
  }

  async _sync({
    folderId,
    syncKey = '0',
    windowSize = 50,
    getChanges = true,
    commands = null,
    truncationSize = 8192,
  } = {}) {
    const collectionChildren = [
      easElem('AirSync:Class', ['Email']),
      easElem('AirSync:SyncKey', [syncKey || '0']),
      easElem('AirSync:CollectionId', [folderId]),
    ];

    if (getChanges) {
      collectionChildren.push(easElem('AirSync:GetChanges', ['1']));
      collectionChildren.push(easElem('AirSync:WindowSize', [String(asInt(windowSize, 50, { min: 1, max: 200 }))]));
    }

    collectionChildren.push(
      easElem('AirSync:Options', [
        easElem('AirSyncBase:BodyPreference', [
          easElem('AirSyncBase:Type', ['1']),
          easElem('AirSyncBase:TruncationSize', [String(asInt(truncationSize, 8192, { min: 256, max: 200000 }))]),
        ]),
      ])
    );

    if (commands) collectionChildren.push(commands);

    const req = easElem('AirSync:Sync', [
      easElem('AirSync:Collections', [
        easElem('AirSync:Collection', collectionChildren),
      ]),
    ]);

    const res = await this._command('Sync', req);
    const root = res.root;
    if (!root || root.name !== 'AirSync:Sync') throw new ActiveSyncError('ActiveSync Sync: ungueltige Antwort');

    const collectionsNode = easChild(root, 'AirSync:Collections');
    const collectionNode = collectionsNode ? easChild(collectionsNode, 'AirSync:Collection') : null;
    if (!collectionNode) throw new ActiveSyncError('ActiveSync Sync: Collection fehlt');

    const status = easText(collectionNode, 'AirSync:Status', '');
    if (status && status !== '1') throw new ActiveSyncError(`ActiveSync Sync status ${status}`);

    const nextSyncKey = easText(collectionNode, 'AirSync:SyncKey', syncKey || '0');
    const moreAvailable = !!easChild(collectionNode, 'AirSync:MoreAvailable');

    const messages = [];
    const cmdNode = easChild(collectionNode, 'AirSync:Commands');
    if (cmdNode) {
      for (const tag of ['AirSync:Add', 'AirSync:Change', 'AirSync:Fetch']) {
        for (const c of easChildren(cmdNode, tag)) {
          const serverId = easText(c, 'AirSync:ServerId', '');
          const app = easChild(c, 'AirSync:ApplicationData');
          const item = this._normalizeMessage(app, { serverId, folderId });
          if (item) messages.push(item);
        }
      }
    }

    const respNode = easChild(collectionNode, 'AirSync:Responses');
    if (respNode) {
      for (const f of easChildren(respNode, 'AirSync:Fetch')) {
        const serverId = easText(f, 'AirSync:ServerId', '');
        const app = easChild(f, 'AirSync:ApplicationData');
        const item = this._normalizeMessage(app, { serverId, folderId });
        if (item) messages.push(item);
      }
    }

    return {
      status: status || '1',
      syncKey: nextSyncKey || syncKey || '0',
      moreAvailable,
      items: messages,
      root,
    };
  }

  async listFolder({ folderId = 'inbox', pageSize = 50, query = null } = {}) {
    const resolved = await this.resolveFolderId(folderId);

    let syncKey = '0';
    let pass = 0;
    let more = false;
    const items = [];
    do {
      const res = await this._sync({
        folderId: resolved,
        syncKey,
        windowSize: pageSize,
        getChanges: true,
        truncationSize: 8192,
      });
      syncKey = res.syncKey || syncKey;
      more = !!res.moreAvailable;
      if (Array.isArray(res.items) && res.items.length) items.push(...res.items);
      pass += 1;
      if (syncKey && syncKey !== '0' && pass === 1 && !items.length && !more) {
        // Initial key negotiation often returns empty once with new SyncKey.
        more = true;
      }
    } while (more && pass < 8 && items.length < pageSize);

    this._mailSyncKeys.set(resolved, syncKey || '0');

    const dedup = [];
    const seen = new Set();
    for (const it of items) {
      const id = asString(it?.id || '');
      if (!id || seen.has(id)) continue;
      seen.add(id);
      dedup.push(it);
    }

    let out = dedup;
    if (query) {
      const q = String(query).trim().toLowerCase();
      out = out.filter((m) => {
        const fromName = asString(m?.from?.emailAddress?.name || '').toLowerCase();
        const fromEmail = asString(m?.from?.emailAddress?.address || '').toLowerCase();
        const subject = asString(m?.subject || '').toLowerCase();
        const prev = asString(m?.bodyPreview || '').toLowerCase();
        return fromName.includes(q) || fromEmail.includes(q) || subject.includes(q) || prev.includes(q);
      });
    }

    out = out.slice(0, asInt(pageSize, 50, { min: 1, max: 250 }));
    this._rememberMessages(out);
    return { items: out, folderId: resolved, syncKey };
  }

  async listInbox({ pageSize = 50, query = null } = {}) {
    return this.listFolder({ folderId: 'inbox', pageSize, query });
  }

  async listSent({ pageSize = 50, query = null } = {}) {
    return this.listFolder({ folderId: 'sentitems', pageSize, query });
  }

  async listDrafts({ pageSize = 50, query = null } = {}) {
    return this.listFolder({ folderId: 'drafts', pageSize, query });
  }

  async search({ folderId = 'inbox', query, top = 50 } = {}) {
    if (!query) throw new Error('ActiveSync search: query required');
    return this.listFolder({ folderId, pageSize: top, query });
  }

  async getMessage(messageId, { folderId = 'inbox' } = {}) {
    const id = asString(messageId || '');
    if (!id) throw new Error('ActiveSync getMessage: messageId required');

    const cached = this._messageCache.get(id);
    if (cached?.body?.content) return cached;

    const resolved = cached?.folderId
      ? cached.folderId
      : await this.resolveFolderId(folderId || 'inbox');

    let syncKey = this._mailSyncKeys.get(resolved) || '0';
    if (!syncKey || syncKey === '0') {
      await this.listFolder({ folderId: resolved, pageSize: 20, query: null });
      syncKey = this._mailSyncKeys.get(resolved) || '0';
    }

    const cmd = easElem('AirSync:Commands', [
      easElem('AirSync:Fetch', [
        easElem('AirSync:ServerId', [id]),
      ]),
    ]);
    const res = await this._sync({
      folderId: resolved,
      syncKey: syncKey || '0',
      getChanges: false,
      commands: cmd,
      windowSize: 1,
      truncationSize: 120000,
    });
    this._mailSyncKeys.set(resolved, res.syncKey || syncKey || '0');

    const found = (res.items || []).find((m) => m.id === id) || cached || null;
    if (!found) throw new ActiveSyncError(`ActiveSync message not found: ${id}`);

    this._rememberMessage(found);
    return found;
  }

  async markRead(messageId, isRead = true, { folderId = 'inbox' } = {}) {
    const id = asString(messageId || '');
    if (!id) throw new Error('ActiveSync markRead: messageId required');

    const cached = this._messageCache.get(id);
    const resolved = cached?.folderId
      ? cached.folderId
      : await this.resolveFolderId(folderId || 'inbox');

    let syncKey = this._mailSyncKeys.get(resolved) || '0';
    if (!syncKey || syncKey === '0') {
      await this.listFolder({ folderId: resolved, pageSize: 20, query: null });
      syncKey = this._mailSyncKeys.get(resolved) || '0';
    }

    const cmd = easElem('AirSync:Commands', [
      easElem('AirSync:Change', [
        easElem('AirSync:ServerId', [id]),
        easElem('AirSync:ApplicationData', [
          easElem('Email:Read', [isRead ? '1' : '0']),
        ]),
      ]),
    ]);

    const res = await this._sync({
      folderId: resolved,
      syncKey: syncKey || '0',
      getChanges: false,
      commands: cmd,
      windowSize: 1,
      truncationSize: 1024,
    });
    this._mailSyncKeys.set(resolved, res.syncKey || syncKey || '0');

    const nowCached = this._messageCache.get(id);
    if (nowCached) {
      nowCached.isRead = !!isRead;
      this._rememberMessage(nowCached);
    }
    return { ok: true, status: res.status || '1' };
  }

  async verifyConnection() {
    const info = await this.options();
    const folders = await this.folderSync({ force: true });
    return {
      options: info,
      folderCount: folders.length,
      deviceId: this.deviceId,
    };
  }
}

//------------------------------------------------------------------------------
// Unified Facade: MailboxClient (ews | graph)
//------------------------------------------------------------------------------

export class MailboxClient {
  constructor({ mode, ewsAccount = null, graphClient = null, activeSyncClient = null, messageClass = null } = {}) {
    this.mode = mode;
    this.ews = ewsAccount;
    this.graph = graphClient;
    this.activeSync = activeSyncClient;
    this.MessageClass = messageClass || Message;

    if (mode === 'ews' && !ewsAccount) throw new Error('MailboxClient: ewsAccount required for mode=ews');
    if (mode === 'graph' && !graphClient) throw new Error('MailboxClient: graphClient required for mode=graph');
    if (mode === 'activesync' && !activeSyncClient) {
      throw new Error('MailboxClient: activeSyncClient required for mode=activesync');
    }
    if (mode !== 'ews' && mode !== 'graph' && mode !== 'activesync') {
      throw new Error("MailboxClient: mode must be 'ews' | 'graph' | 'activesync'");
    }
  }

  // ---- LIST/SEARCH ----
  async listInbox({ pageSize = 50, offset = 0, query = null } = {}) {
    if (this.mode === 'activesync') return this.activeSync.listInbox({ pageSize, query });
    if (this.mode === 'ews') return this.ews.listInbox({ pageSize, offset, queryString: query });
    if (query) return this.graph.searchMessages({ folderId: 'inbox', search: query, top: pageSize });
    return this.graph.listMessages({ folderId: 'inbox', top: pageSize });
  }

  async listSent({ pageSize = 50, offset = 0, query = null } = {}) {
    if (this.mode === 'activesync') return this.activeSync.listSent({ pageSize, query });
    if (this.mode === 'ews') {
      return this.ews.findItems({ folderId: 'sentitems', maxEntriesReturned: pageSize, offset, queryString: query });
    }
    if (query) return this.graph.searchMessages({ folderId: 'sentitems', search: query, top: pageSize });
    return this.graph.listMessages({ folderId: 'sentitems', top: pageSize });
  }

  async listDrafts({ pageSize = 50, offset = 0, query = null } = {}) {
    if (this.mode === 'activesync') return this.activeSync.listDrafts({ pageSize, query });
    if (this.mode === 'ews') {
      return this.ews.findItems({ folderId: 'drafts', maxEntriesReturned: pageSize, offset, queryString: query });
    }
    if (query) return this.graph.searchMessages({ folderId: 'drafts', search: query, top: pageSize });
    return this.graph.listMessages({ folderId: 'drafts', top: pageSize });
  }

  async search({ folderId = 'inbox', query, top = 50 } = {}) {
    if (!query) throw new Error('search: query required');
    if (this.mode === 'activesync') return this.activeSync.search({ folderId, query, top });
    if (this.mode === 'ews') return this.ews.findItems({ folderId, maxEntriesReturned: top, offset: 0, queryString: query });
    return this.graph.searchMessages({ folderId, search: query, top });
  }

  async filter({ folderId = 'inbox', filter, top = 50 } = {}) {
    if (!filter) throw new Error('filter: filter required');
    if (this.mode === 'activesync') return this.activeSync.search({ folderId, query: filter, top });
    if (this.mode === 'ews') {
      // best-effort: treat filter as AQS
      return this.ews.findItems({ folderId, maxEntriesReturned: top, offset: 0, queryString: filter });
    }
    return this.graph.filterMessages({ folderId, filter, top });
  }

  // ---- READ ----
  async getMessage(idOrItem, { includeMime = false } = {}) {
    if (this.mode === 'activesync') {
      return this.activeSync.getMessage(idOrItem, { includeMime });
    }
    if (this.mode === 'ews') {
      return this.ews.getItem(idOrItem, {
        baseShape: 'Default',
        includeMimeContent: !!includeMime,
        additionalProperties: ['item:Categories', 'item:Importance', 'item:Flag', 'item:Attachments', 'item:HasAttachments'],
      });
    }
    return this.graph.getMessage(idOrItem);
  }

  async getMessageMime(messageId) {
    if (this.mode === 'activesync') {
      throw new Error('ActiveSync getMessageMime not implemented');
    }
    if (this.mode === 'ews') {
      return this.ews.getItem(messageId, { baseShape: 'Default', includeMimeContent: true });
    }
    return this.graph.getMessageMime(messageId);
  }

  async _resolveEwsItemRef(item, { refreshChangeKey = false } = {}) {
    if (this.mode !== 'ews') return item;

    const src = typeof item === 'string'
      ? { Id: item }
      : (item && typeof item === 'object'
        ? {
            Id: item.Id || item.id || item.draftId || item.messageId || null,
            ChangeKey: item.ChangeKey || item.changeKey || null,
          }
        : null);

    const id = src?.Id ? String(src.Id).trim() : '';
    const changeKey = src?.ChangeKey ? String(src.ChangeKey).trim() : '';
    if (!id) throw new Error('EWS item reference requires Id');
    if (!refreshChangeKey && changeKey) {
      return changeKey ? { Id: id, ChangeKey: changeKey } : { Id: id };
    }

    const fetched = await this.ews.getItem({ Id: id, ...(changeKey ? { ChangeKey: changeKey } : {}) }, {
      baseShape: 'IdOnly',
    });
    const identity = extractEwsItemIdentity(fetched?.message || null);
    const resolvedId = String(identity?.id || id).trim();
    const resolvedChangeKey = String(identity?.changeKey || changeKey || '').trim();
    return resolvedChangeKey
      ? { Id: resolvedId, ChangeKey: resolvedChangeKey }
      : { Id: resolvedId };
  }

  // ---- SEND/WRITE ----
  async sendMail({ subject, body, bodyType = 'HTML', to = [], cc = [], bcc = [], saveCopyFolderId = 'sentitems' } = {}) {
    if (this.mode === 'ews') {
      const msg = new this.MessageClass({
        account: this.ews,
        subject,
        body,
        bodyType,
        toRecipients: to,
        ccRecipients: cc,
        bccRecipients: bcc,
      });
      return this.ews.send(msg, { saveCopyFolderId });
    }
    return this.graph.sendMail({ subject, body, bodyType, to, cc, bcc });
  }

  // Draft flow unified
  async createReplyDraft(messageIdOrItem, { replyAll = false, body = '', bodyType = 'HTML', folderId = 'drafts' } = {}) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageIdOrItem, { refreshChangeKey: true });
      const draft = await this.ews.createReplyDraft(itemRef, { replyAll, body, bodyType, folderId });
      return {
        ...draft,
        draftId: draft?.draftId || draft?.id || null,
        providerMode: 'ews',
        threaded: true,
      };
    }
    const draft = await this.graph.createReplyDraft(messageIdOrItem, { replyAll });
    return {
      ...draft,
      draftId: draft?.id || null,
      providerMode: 'graph',
      threaded: true,
    };
  }

  async createDraft({ subject = '', body = '', bodyType = 'HTML', to = [], cc = [], bcc = [], categories = null, importance = null, flag = null } = {}) {
    if (this.mode === 'ews') {
      const msg = new this.MessageClass({
        account: this.ews,
        subject: subject || '(no subject)',
        body: body ?? '',
        bodyType,
        toRecipients: to,
        ccRecipients: cc,
        bccRecipients: bcc,
        categories,
        importance,
      });
      return this.ews.saveDraft(msg, { folderId: 'drafts' });
    }
    return this.graph.createDraft({ subject, body, bodyType, to, cc, bcc, categories, importance, flag });
  }

  async updateDraft(draftId, { subject = null, body = null, bodyType = 'HTML', to = null, cc = null, bcc = null, categories = null, importance = null, flag = null, isRead = null } = {}) {
    if (!draftId) throw new Error('updateDraft: draftId required');

    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(draftId, { refreshChangeKey: true });
      return this.ews.updateMessage(itemRef, { subject, body, bodyType, categories, importance, flag });
    }

    const patch = {};
    if (subject !== null) patch.subject = subject;
    if (body !== null) patch.body = { contentType: bodyType, content: body };

    const mkRecipients = (arr) => normalizeArray(arr).map((addr) => ({ emailAddress: { address: addr } }));
    if (to !== null) patch.toRecipients = mkRecipients(to);
    if (cc !== null) patch.ccRecipients = mkRecipients(cc);
    if (bcc !== null) patch.bccRecipients = mkRecipients(bcc);

    if (categories !== null) patch.categories = normalizeArray(categories);
    if (importance !== null) patch.importance = importance;
    if (flag !== null) patch.flag = flag;
    if (isRead !== null) patch.isRead = !!isRead;

    return this.graph.updateMessage(draftId, patch);
  }

  async sendDraft(draftId, { saveCopy = true, saveFolderId = 'sentitems' } = {}) {
    if (!draftId) throw new Error('sendDraft: draftId required');
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(draftId, { refreshChangeKey: true });
      return this.ews.sendExisting([itemRef], { saveCopy, saveFolderId });
    }
    return this.graph.sendDraft(draftId);
  }

  async deleteDraft(draftId) {
    if (!draftId) throw new Error('deleteDraft: draftId required');
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(draftId, { refreshChangeKey: true });
      return this.ews.deleteItems([itemRef], 'MoveToDeletedItems');
    }
    return this.graph.deleteMessage(draftId);
  }

  // Rich reply/forward
  async reply(messageIdOrItem, { body = '', replyAll = false, bodyType = 'HTML', saveCopy = true } = {}) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageIdOrItem, { refreshChangeKey: true });
      return this.ews.replyTo(itemRef, { body, replyAll, bodyType, saveCopy });
    }
    return this.graph.replyRich(messageIdOrItem, { replyAll, body, bodyType });
  }

  async forward(messageIdOrItem, { toRecipients, body = '', bodyType = 'HTML', saveCopy = true } = {}) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageIdOrItem, { refreshChangeKey: true });
      return this.ews.forward(itemRef, { toRecipients, body, bodyType, saveCopy });
    }
    return this.graph.forwardRich(messageIdOrItem, { to: toRecipients, body, bodyType });
  }

  // ---- MOVE/COPY/DELETE ----
  async moveMessage(messageId, destinationFolder) {
    if (this.mode === 'ews') return this.ews.moveItems([{ Id: messageId }], destinationFolder);
    return this.graph.moveMessage(messageId, destinationFolder);
  }

  async copyMessage(messageId, destinationFolder) {
    if (this.mode === 'ews') return this.ews.copyItems([{ Id: messageId }], destinationFolder);
    return this.graph.copyMessage(messageId, destinationFolder);
  }

  async deleteMessage(messageId, deleteType = 'MoveToDeletedItems') {
    if (this.mode === 'ews') return this.ews.deleteItems([{ Id: messageId }], deleteType);
    return this.graph.deleteMessage(messageId);
  }

  // ---- UPDATE ----
  async updateMessage(messageId, { subject = null, body = null, bodyType = 'HTML', isRead = null, categories = null, importance = null, flag = null } = {}) {
    if (this.mode === 'activesync') {
      const wantsOnlyRead = subject === null && body === null && categories === null && importance === null && flag === null;
      if (!wantsOnlyRead) throw new Error('ActiveSync updateMessage currently supports only isRead');
      if (isRead === null) return { ok: true };
      return this.activeSync.markRead(messageId, !!isRead);
    }
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageId, { refreshChangeKey: true });
      const ops = [];
      if (subject !== null || body !== null || categories !== null || importance !== null || flag !== null) {
        ops.push(this.ews.updateMessage(itemRef, { subject, body, bodyType, categories, importance, flag }));
      }
      if (isRead !== null) ops.push(this.ews.setReadFlag([itemRef], !!isRead));
      return { results: await Promise.all(ops) };
    }

    const patch = {};
    if (subject !== null) patch.subject = subject;
    if (body !== null) patch.body = { contentType: bodyType, content: body };
    if (isRead !== null) patch.isRead = !!isRead;
    if (categories !== null) patch.categories = normalizeArray(categories);
    if (importance !== null) patch.importance = importance;
    if (flag !== null) patch.flag = flag;

    return this.graph.updateMessage(messageId, patch);
  }

  async markRead(messageId, isRead = true) {
    if (this.mode === 'activesync') return this.activeSync.markRead(messageId, isRead);
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageId, { refreshChangeKey: true });
      return this.ews.setReadFlag([itemRef], isRead);
    }
    return this.graph.markRead(messageId, isRead);
  }

  // ---- TAGS / CATEGORIES ----
  async getCategories(messageId) {
    if (this.mode === 'ews') {
      const { message } = await this.ews.getItem(messageId, {
        baseShape: 'Default',
        additionalProperties: ['item:Categories'],
      });
      const cats = message?.['t:Categories'] || message?.Categories || null;
      if (!cats) return [];
      const strings = cats?.['t:String'] || cats?.String || [];
      const arr = Array.isArray(strings) ? strings : [strings];
      return arr.map((x) => (typeof x === 'string' ? x : x?.$value)).filter(Boolean);
    }
    return this.graph.getCategories(messageId);
  }

  async setCategories(messageId, categories = []) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageId, { refreshChangeKey: true });
      return this.ews.updateMessage(itemRef, { categories });
    }
    return this.graph.setCategories(messageId, categories);
  }

  async addCategories(messageId, categories = []) {
    const current = await this.getCategories(messageId);
    const merged = stableUnique([...current, ...normalizeArray(categories)]);
    return this.setCategories(messageId, merged);
  }

  async removeCategories(messageId, categories = []) {
    const current = await this.getCategories(messageId);
    const rem = new Set(normalizeArray(categories));
    return this.setCategories(messageId, current.filter((c) => !rem.has(c)));
  }

  // ---- FLAG / FOLLOW-UP ----
  async setFlag(messageId, flagObj) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageId, { refreshChangeKey: true });
      return this.ews.updateMessage(itemRef, { flag: flagObj });
    }
    return this.graph.setFlag(messageId, flagObj);
  }

  async clearFlag(messageId) {
    if (this.mode === 'ews') {
      const itemRef = await this._resolveEwsItemRef(messageId, { refreshChangeKey: true });
      return this.ews.updateMessage(itemRef, { flag: { status: 'NotFlagged' } });
    }
    return this.graph.clearFlag(messageId);
  }

  // ---- ATTACHMENTS ----
  async listAttachments(messageIdOrItem) {
    if (this.mode === 'ews') return this.ews.listAttachments(messageIdOrItem);
    return this.graph.listAttachments(messageIdOrItem);
  }

  async getAttachment(messageIdOrAttachmentId, attachmentId = null) {
    if (this.mode === 'ews') {
      const attId = attachmentId || messageIdOrAttachmentId;
      return this.ews.getAttachment(attId);
    }
    if (!attachmentId) throw new Error('Graph getAttachment: requires (messageId, attachmentId)');
    return this.graph.getAttachment(messageIdOrAttachmentId, attachmentId);
  }

  async addFileAttachmentToDraft(draftId, { name, contentBytes, contentType } = {}) {
    if (this.mode === 'ews') return this.ews.addFileAttachmentToDraft({ Id: draftId }, { name, contentBytes, contentType });
    return this.graph.addFileAttachmentToDraft(draftId, { name, contentBytes, contentType });
  }

  async addLargeAttachmentToDraft(draftId, { name, bytes, contentType } = {}) {
    if (this.mode === 'ews') {
      throw new Error('EWS large attachment streaming not implemented; use addFileAttachmentToDraft(base64)');
    }
    return this.graph.addLargeAttachmentToDraft(draftId, { name, bytes, contentType });
  }

  async deleteAttachment(messageIdOrAttachmentId, attachmentId = null) {
    if (this.mode === 'ews') {
      const attId = attachmentId || messageIdOrAttachmentId;
      return this.ews.deleteAttachment(attId);
    }
    if (!attachmentId) throw new Error('Graph deleteAttachment: requires (messageId, attachmentId)');
    return this.graph.deleteAttachment(messageIdOrAttachmentId, attachmentId);
  }

  // ---- FOLDERS ----
  async listFolders() {
    if (this.mode === 'activesync') return this.activeSync.listFolders();
    if (this.mode === 'ews') return this.ews.findFolders({ parentFolderId: 'msgfolderroot' });
    return this.graph.listFolders();
  }

  async createFolder({ parentFolderId = null, displayName } = {}) {
    if (this.mode === 'ews') return this.ews.createFolder({ parentFolderId: parentFolderId || 'msgfolderroot', displayName });
    return this.graph.createFolder({ parentFolderId, displayName });
  }

  async renameFolder(folderIdOrObj, displayName) {
    if (this.mode === 'ews') return this.ews.renameFolder(folderIdOrObj, displayName);
    return this.graph.renameFolder(folderIdOrObj, displayName);
  }

  async deleteFolder(folderIdOrObj, deleteType = 'HardDelete') {
    if (this.mode === 'ews') return this.ews.deleteFolder(folderIdOrObj, deleteType);
    return this.graph.deleteFolder(folderIdOrObj);
  }

  async resolveFolderPath(path, { startFolderId = 'inbox' } = {}) {
    if (this.mode === 'activesync') return this.activeSync.resolveFolderId(path || startFolderId);
    if (this.mode === 'ews') return this.ews.resolveFolderPath(path, { startFolderId });
    return this.graph.resolveFolderPath(path, { startFolderId });
  }

  // ---- SYNC ----
  async deltaMessages({ folderId = 'inbox', deltaLink = null, nextLink = null, select = null } = {}) {
    if (this.mode === 'activesync') {
      throw new Error('ActiveSync deltaMessages not implemented (server SyncKey strategy differs).');
    }
    if (this.mode === 'ews') throw new Error('EWS deltaMessages not implemented (use FindItem paging / notifications).');
    return this.graph.deltaMessages({ folderId, deltaLink, nextLink, select });
  }

  // ---- AUTOCOMPLETE ----
  async autocompletePeople(query, { top = 10 } = {}) {
    if (this.mode === 'activesync') {
      throw new Error('autocompletePeople(ActiveSync) not implemented');
    }
    if (this.mode === 'ews') throw new Error('autocompletePeople(EWS) not implemented (use ResolveNames/GAL if needed).');
    return this.graph.findPeople(query, { top });
  }

  async autocompleteContacts(query, { top = 10 } = {}) {
    if (this.mode === 'activesync') {
      throw new Error('autocompleteContacts(ActiveSync) not implemented');
    }
    if (this.mode === 'ews') throw new Error('autocompleteContacts(EWS) not implemented (use ResolveNames/GAL if needed).');
    return this.graph.searchContacts(query, { top });
  }
}

//------------------------------------------------------------------------------
// Delta Poller helper (Graph)
//------------------------------------------------------------------------------

/**
 * createDeltaPoller
 * - Uses Graph delta to keep mailbox in sync without webhooks.
 * - You provide loadState/saveState via your host runtime.
 */
export function createDeltaPoller({
  graphClient,
  folderId = 'inbox',
  intervalMs = 30_000,
  select = 'id,subject,from,receivedDateTime,isRead,parentFolderId,categories,hasAttachments,conversationId',
  loadState,
  saveState,
  onChanges,
  onError,
} = {}) {
  if (!graphClient) throw new Error('createDeltaPoller: graphClient required');
  if (typeof loadState !== 'function') throw new Error('createDeltaPoller: loadState required');
  if (typeof saveState !== 'function') throw new Error('createDeltaPoller: saveState required');
  if (typeof onChanges !== 'function') throw new Error('createDeltaPoller: onChanges required');

  let timer = null;
  let running = false;
  let stopped = false;

  const tick = async () => {
    if (running || stopped) return;
    running = true;
    try {
      const state = (await loadState()) || {};
      let deltaLink = state.deltaLink || null;

      let page = await graphClient.deltaMessages({ folderId, deltaLink, select });
      const all = [];
      all.push(...(page.value || []));

      while (page.nextLink) {
        page = await graphClient.deltaMessages({ nextLink: page.nextLink });
        all.push(...(page.value || []));
      }

      if (page.deltaLink) {
        deltaLink = page.deltaLink;
        await saveState({ deltaLink });
      }

      await onChanges(all, { folderId, deltaLink });
    } catch (err) {
      if (onError) onError(err);
    } finally {
      running = false;
    }
  };

  const start = () => {
    stopped = false;
    if (timer) clearInterval(timer);
    timer = setInterval(tick, intervalMs);
    tick();
  };

  const stop = () => {
    stopped = true;
    if (timer) clearInterval(timer);
    timer = null;
  };

  return { start, stop, tick };
}

export default {
  // EWS
  ExchangeCredentials,
  EwsClient,
  EwsError,
  ExchangeAccount,
  Folder,
  Message,
  createExchangeAccount,

  // Graph
  GraphClient,
  GraphError,

  // ActiveSync
  ActiveSyncClient,
  ActiveSyncError,

  // Facade
  MailboxClient,

  // Sync
  createDeltaPoller,

  // Helpers
  jsonToXml,
  parseSoapResponse,
  assertEwsSuccess,
};

const execFileSync = childProcess.execFileSync;
const once = events.once;
const mkdir = fs.promises.mkdir;
const readFile = fs.promises.readFile;
const writeFile = fs.promises.writeFile;
const dirname = path.dirname;
const join = path.join;
const resolve = path.resolve;
const tlsConnect = tls.connect;
const fileURLToPath = url.fileURLToPath;
const randomUUID = crypto.randomUUID
  ? function randomUuidCompat() {
      return crypto.randomUUID();
    }
  : function randomUuidCompat() {
      const bytes = crypto.randomBytes(16);
      bytes[6] = (bytes[6] & 0x0f) | 0x40;
      bytes[8] = (bytes[8] & 0x3f) | 0x80;
      const hex = bytes.toString("hex");
      return [
        hex.slice(0, 8),
        hex.slice(8, 12),
        hex.slice(12, 16),
        hex.slice(16, 20),
        hex.slice(20, 32),
      ].join("-");
    };

const __dirname = dirname(fileURLToPath(import.meta.url));

function defaultAgentBinary() {
  if (process.env.CTO_AGENT_BINARY) return process.env.CTO_AGENT_BINARY;
  const releasePath = resolve(process.cwd(), "target/release/cto-agent");
  if (fs.existsSync(releasePath)) return releasePath;
  return resolve(process.cwd(), "target/debug/cto-agent");
}

const DEFAULTS = {
  channel: "email",
  provider: "imap",
  db: resolve(process.cwd(), "runtime/cto_agent.db"),
  rawDir: resolve(process.cwd(), "runtime/communication/raw"),
  schema: join(__dirname, "communication_schema.sql"),
  agentBinary: defaultAgentBinary(),
  imapHost: "imap.one.com",
  imapPort: 993,
  smtpHost: "send.one.com",
  smtpPort: 465,
  folder: "INBOX",
  limit: 20,
  trustLevel: "low",
  emitInterrupts: "false",
  interruptChannel: "email",
  graphBaseUrl: "https://graph.microsoft.com/v1.0",
  graphUser: "me",
  ewsVersion: "Exchange2013",
  ewsAuthType: "basic",
  activeSyncPath: "Microsoft-Server-ActiveSync",
  activeSyncDeviceType: "CodexCLI",
  activeSyncProtocolVersion: "14.1",
  activeSyncPolicyKey: "0",
  verifySend: "true",
  sentVerifyWindowSeconds: 90,
};

const DEBUG = process.env.COMM_DEBUG === "1";

function nowIso() {
  return new Date().toISOString();
}

function fail(message) {
  throw new Error(message);
}

function debugLog(...args) {
  if (!DEBUG) return;
  process.stderr.write(`[comm-debug] ${args.join(" ")}\n`);
}

function sqlValue(value) {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "NULL";
  if (typeof value === "boolean") return value ? "1" : "0";
  const text = String(value).replace(/\u0000/g, "").replace(/'/g, "''");
  return `'${text}'`;
}

function parseJsonOutput(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return [];
  return JSON.parse(trimmed);
}

function toBool(value) {
  const normalized = String(value == null ? "" : value)
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function runSql(dbPath, sql, { json = false } = {}) {
  const input = `.timeout 5000\n${json ? ".mode json\n" : ""}${sql.trim().endsWith(";") ? sql.trim() : `${sql.trim()};`}\n`;
  return execFileSync("sqlite3", [dbPath], {
    input,
    encoding: "utf8",
    stdio: ["pipe", "pipe", "pipe"],
  });
}

function firstRow(dbPath, sql) {
  return parseJsonOutput(runSql(dbPath, sql, { json: true }))[0] || null;
}

async function ensureSchema(dbPath, schemaPath) {
  await mkdir(dirname(dbPath), { recursive: true });
  const sql = await readFile(schemaPath, "utf8");
  runSql(dbPath, sql);
}

function decodeMimeHeader(value = "") {
  return String(value).replace(/=\?([^?]+)\?([bqBQ])\?([^?]*)\?=/g, (_full, _charset, encoding, payload) => {
    try {
      if (String(encoding).toUpperCase() === "B") {
        return Buffer.from(payload, "base64").toString("utf8");
      }
      const qp = payload
        .replace(/_/g, " ")
        .replace(/=([0-9A-Fa-f]{2})/g, (_match, hex) => String.fromCharCode(Number.parseInt(hex, 16)));
      return Buffer.from(qp, "binary").toString("utf8");
    } catch {
      return payload;
    }
  });
}

function unfoldHeaders(headerText) {
  return headerText.replace(/\r?\n[ \t]+/g, " ");
}

function parseHeaders(headerText) {
  const unfolded = unfoldHeaders(headerText);
  const headers = {};
  for (const line of unfolded.split(/\r?\n/)) {
    const index = line.indexOf(":");
    if (index <= 0) continue;
    const name = line.slice(0, index).trim().toLowerCase();
    const value = decodeMimeHeader(line.slice(index + 1).trim());
    headers[name] = value;
  }
  return headers;
}

function extractAddress(token = "") {
  const bracket = token.match(/<([^>]+)>/);
  if (bracket && bracket[1]) return bracket[1].trim().toLowerCase();
  const naked = token.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i);
  return (naked && naked[0] ? naked[0].trim().toLowerCase() : "") || "";
}

function extractDisplayName(token = "") {
  const bracket = token.match(/^(.*)<[^>]+>/);
  if (bracket && bracket[1]) return decodeMimeHeader(bracket[1].replace(/"/g, "").trim());
  return "";
}

function extractAddresses(raw = "") {
  const seen = new Set();
  const out = [];
  for (const token of String(raw || "").split(",")) {
    const address = extractAddress(token);
    if (!address || seen.has(address)) continue;
    seen.add(address);
    out.push(address);
  }
  return out;
}

function previewText(input = "") {
  return String(input || "").replace(/\s+/g, " ").trim().slice(0, 280);
}

function decodeQuotedPrintable(input = "") {
  const normalized = String(input || "")
    .replace(/=\r?\n/g, "")
    .replace(/=([0-9A-Fa-f]{2})/g, (_match, hex) =>
      String.fromCharCode(Number.parseInt(hex, 16))
    );
  return Buffer.from(normalized, "binary").toString("utf8");
}

function decodeBase64Body(input = "") {
  let sanitized = String(input || "").replace(/[^A-Za-z0-9+/=]/g, "");
  while (sanitized.length % 4 === 1) {
    sanitized = sanitized.slice(0, -1);
  }
  if (!sanitized) return "";
  sanitized += "=".repeat((4 - (sanitized.length % 4)) % 4);
  return Buffer.from(sanitized, "base64").toString("utf8");
}

function decodeTransferEncodedBody(body = "", transferEncoding = "") {
  const normalizedEncoding = String(transferEncoding || "").trim().toLowerCase();
  if (normalizedEncoding === "base64") {
    return decodeBase64Body(body);
  }
  if (normalizedEncoding === "quoted-printable") {
    return decodeQuotedPrintable(body);
  }
  return String(body || "");
}

function stripHtml(input = "") {
  return String(input || "")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|li|tr|h[1-6])>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#39;/gi, "'")
    .replace(/&quot;/gi, "\"")
    .replace(/\r/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function boundaryFromContentType(contentType = "") {
  const match = String(contentType || "").match(/boundary="?([^";]+)"?/i);
  return match && match[1] ? match[1] : "";
}

function splitMultipartBody(body = "", boundary = "") {
  if (!boundary) return [];
  const normalized = String(body || "").replace(/\r\n/g, "\n");
  const marker = `--${boundary}`;
  return normalized
    .split(marker)
    .slice(1)
    .map((segment) => segment.replace(/^\n/, "").replace(/\n$/, ""))
    .filter((segment) => segment && !segment.startsWith("--"));
}

function parseMimeEntity(rawText) {
  const separator = String(rawText || "").search(/\r?\n\r?\n/);
  const headerText = separator === -1 ? String(rawText || "") : String(rawText || "").slice(0, separator);
  const body = separator === -1 ? "" : String(rawText || "").slice(separator).replace(/^\r?\n\r?\n/, "");
  const headers = parseHeaders(headerText);
  const contentType = headers["content-type"] || "text/plain; charset=utf-8";
  const contentDisposition = headers["content-disposition"] || "";
  const transferEncoding = headers["content-transfer-encoding"] || "";
  const mediaType = String(contentType).split(";")[0].trim().toLowerCase();

  if (mediaType.startsWith("multipart/")) {
    const boundary = boundaryFromContentType(contentType);
    let bodyText = "";
    let bodyHtml = "";
    let hasAttachments = /attachment/i.test(contentDisposition);
    for (const part of splitMultipartBody(body, boundary)) {
      const parsed = parseMimeEntity(part);
      if (!bodyText && parsed.bodyText) bodyText = parsed.bodyText;
      if (!bodyHtml && parsed.bodyHtml) bodyHtml = parsed.bodyHtml;
      hasAttachments = hasAttachments || parsed.hasAttachments;
    }
    if (!bodyText && bodyHtml) {
      bodyText = stripHtml(bodyHtml);
    }
    return { headers, bodyText: bodyText.trim(), bodyHtml, hasAttachments };
  }

  const decodedBody = decodeTransferEncodedBody(body, transferEncoding);
  const hasAttachments = /attachment/i.test(contentDisposition);
  if (mediaType === "text/html") {
    return {
      headers,
      bodyText: stripHtml(decodedBody),
      bodyHtml: decodedBody,
      hasAttachments,
    };
  }
  return {
    headers,
    bodyText: decodedBody.trim(),
    bodyHtml: "",
    hasAttachments,
  };
}

function parseRfc822(rawBuffer) {
  const rawText = rawBuffer.toString("utf8");
  const separator = rawText.search(/\r?\n\r?\n/);
  const headerText = separator === -1 ? rawText : rawText.slice(0, separator);
  const headers = parseHeaders(headerText);
  const parsedEntity = parseMimeEntity(rawText);
  return {
    headers,
    bodyText: parsedEntity.bodyText,
    bodyHtml: parsedEntity.bodyHtml,
    subject: headers.subject || "(ohne Betreff)",
    fromHeader: headers.from || "",
    toHeader: headers.to || "",
    ccHeader: headers.cc || "",
    messageId: headers["message-id"] || "",
    references: headers.references || "",
    inReplyTo: headers["in-reply-to"] || "",
    sentAt: headers.date || "",
    hasAttachments:
      parsedEntity.hasAttachments ||
      /content-disposition:\s*attachment/i.test(rawText) ||
      /multipart\/mixed/i.test(rawText),
  };
}

function threadKeyFromEmail(parsed, fallback) {
  const references = String(parsed.references || "")
    .split(/\s+/)
    .map((value) => value.trim())
    .filter(Boolean);
  return references[0] || parsed.inReplyTo || parsed.messageId || fallback;
}

function accountKeyFromEmail(address) {
  return `email:${String(address || "").trim().toLowerCase()}`;
}

function messageKeyFromRemote(accountKey, folder, remoteId) {
  return `${accountKey}::${folder}::${remoteId}`;
}

async function writeRawPayload(rawDir, remoteId, rawBuffer) {
  await mkdir(rawDir, { recursive: true });
  const safeId = String(remoteId || randomUUID()).replace(/[^A-Za-z0-9_.-]/g, "_");
  const fullPath = join(rawDir, `${safeId}.eml`);
  await writeFile(fullPath, rawBuffer);
  return fullPath;
}

function upsertAccount(dbPath, account) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_accounts (
      account_key, channel, address, provider, profile_json, created_at, updated_at, last_inbound_ok_at, last_outbound_ok_at
    ) VALUES (
      ${sqlValue(account.accountKey)},
      ${sqlValue(account.channel)},
      ${sqlValue(account.address)},
      ${sqlValue(account.provider)},
      ${sqlValue(account.profileJson)},
      ${sqlValue(account.createdAt)},
      ${sqlValue(account.updatedAt)},
      ${sqlValue(account.lastInboundOkAt)},
      ${sqlValue(account.lastOutboundOkAt)}
    )
    ON CONFLICT(account_key) DO UPDATE SET
      channel=excluded.channel,
      address=excluded.address,
      provider=excluded.provider,
      profile_json=excluded.profile_json,
      updated_at=excluded.updated_at,
      last_inbound_ok_at=COALESCE(excluded.last_inbound_ok_at, communication_accounts.last_inbound_ok_at),
      last_outbound_ok_at=COALESCE(excluded.last_outbound_ok_at, communication_accounts.last_outbound_ok_at)
    `
  );
}

function upsertMessage(dbPath, message) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_messages (
      message_key, channel, account_key, thread_key, remote_id, direction, folder_hint, sender_display, sender_address,
      recipient_addresses_json, cc_addresses_json, bcc_addresses_json, subject, preview, body_text, body_html,
      raw_payload_ref, trust_level, status, seen, has_attachments, external_created_at, observed_at, metadata_json
    ) VALUES (
      ${sqlValue(message.messageKey)},
      ${sqlValue(message.channel)},
      ${sqlValue(message.accountKey)},
      ${sqlValue(message.threadKey)},
      ${sqlValue(message.remoteId)},
      ${sqlValue(message.direction)},
      ${sqlValue(message.folderHint)},
      ${sqlValue(message.senderDisplay)},
      ${sqlValue(message.senderAddress)},
      ${sqlValue(message.recipientAddressesJson)},
      ${sqlValue(message.ccAddressesJson)},
      ${sqlValue(message.bccAddressesJson)},
      ${sqlValue(message.subject)},
      ${sqlValue(message.preview)},
      ${sqlValue(message.bodyText)},
      ${sqlValue(message.bodyHtml)},
      ${sqlValue(message.rawPayloadRef)},
      ${sqlValue(message.trustLevel)},
      ${sqlValue(message.status)},
      ${sqlValue(message.seen)},
      ${sqlValue(message.hasAttachments)},
      ${sqlValue(message.externalCreatedAt)},
      ${sqlValue(message.observedAt)},
      ${sqlValue(message.metadataJson)}
    )
    ON CONFLICT(message_key) DO UPDATE SET
      thread_key=excluded.thread_key,
      sender_display=excluded.sender_display,
      sender_address=excluded.sender_address,
      recipient_addresses_json=excluded.recipient_addresses_json,
      cc_addresses_json=excluded.cc_addresses_json,
      bcc_addresses_json=excluded.bcc_addresses_json,
      subject=excluded.subject,
      preview=excluded.preview,
      body_text=excluded.body_text,
      body_html=excluded.body_html,
      raw_payload_ref=excluded.raw_payload_ref,
      trust_level=excluded.trust_level,
      status=excluded.status,
      seen=excluded.seen,
      has_attachments=excluded.has_attachments,
      external_created_at=excluded.external_created_at,
      observed_at=excluded.observed_at,
      metadata_json=excluded.metadata_json
    `
  );
}

function messageExists(dbPath, messageKey) {
  const row = firstRow(
    dbPath,
    `
    SELECT message_key
    FROM communication_messages
    WHERE message_key = ${sqlValue(messageKey)}
    LIMIT 1
    `
  );
  return !!(row && row.message_key);
}

function refreshThread(dbPath, threadKey) {
  const latestRows = parseJsonOutput(
    runSql(
      dbPath,
      `
      SELECT subject, message_key, external_created_at, sender_address, recipient_addresses_json
      FROM communication_messages
      WHERE thread_key = ${sqlValue(threadKey)}
      ORDER BY external_created_at DESC, observed_at DESC
      LIMIT 1
      `,
      { json: true }
    )
  );
  if (!latestRows.length) return;

  const latest = latestRows[0];
  const counts = parseJsonOutput(
    runSql(
      dbPath,
      `
      SELECT
        COUNT(*) AS message_count,
        SUM(CASE WHEN seen = 0 THEN 1 ELSE 0 END) AS unread_count
      FROM communication_messages
      WHERE thread_key = ${sqlValue(threadKey)}
      `,
      { json: true }
    )
  )[0];

  const participants = Array.from(
    new Set([
      String(latest.sender_address || "").trim().toLowerCase(),
      ...JSON.parse(latest.recipient_addresses_json || "[]"),
    ].filter(Boolean))
  );

  runSql(
    dbPath,
    `
    INSERT INTO communication_threads (
      thread_key, channel, account_key, subject, participant_keys_json, last_message_key,
      last_message_at, message_count, unread_count, metadata_json, updated_at
    )
    SELECT
      ${sqlValue(threadKey)},
      channel,
      account_key,
      ${sqlValue(latest.subject || "(ohne Betreff)")},
      ${sqlValue(JSON.stringify(participants))},
      ${sqlValue(latest.message_key || "")},
      ${sqlValue(latest.external_created_at || nowIso())},
      ${sqlValue(Number((counts && counts.message_count) || 0))},
      ${sqlValue(Number((counts && counts.unread_count) || 0))},
      ${sqlValue("{}")},
      ${sqlValue(nowIso())}
    FROM communication_messages
    WHERE thread_key = ${sqlValue(threadKey)}
    LIMIT 1
    ON CONFLICT(thread_key) DO UPDATE SET
      subject=excluded.subject,
      participant_keys_json=excluded.participant_keys_json,
      last_message_key=excluded.last_message_key,
      last_message_at=excluded.last_message_at,
      message_count=excluded.message_count,
      unread_count=excluded.unread_count,
      metadata_json=excluded.metadata_json,
      updated_at=excluded.updated_at
    `
  );
}

function recordSyncRun(dbPath, run) {
  runSql(
    dbPath,
    `
    INSERT INTO communication_sync_runs (
      run_key, channel, account_key, folder_hint, started_at, finished_at,
      ok, fetched_count, stored_count, error_text, metadata_json
    ) VALUES (
      ${sqlValue(run.runKey)},
      ${sqlValue(run.channel)},
      ${sqlValue(run.accountKey)},
      ${sqlValue(run.folderHint)},
      ${sqlValue(run.startedAt)},
      ${sqlValue(run.finishedAt)},
      ${sqlValue(run.ok)},
      ${sqlValue(run.fetchedCount)},
      ${sqlValue(run.storedCount)},
      ${sqlValue(run.errorText)},
      ${sqlValue(run.metadataJson)}
    )
    `
  );
}

class BufferSocket {
  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.closed = false;
    this.waiters = [];
    socket.on("data", (chunk) => {
      this.buffer = Buffer.concat([this.buffer, chunk]);
      this.flush();
    });
    socket.on("close", () => {
      this.closed = true;
      this.flush();
    });
  }

  flush() {
    const waiters = [...this.waiters];
    this.waiters = [];
    for (const waiter of waiters) {
      if (!waiter()) this.waiters.push(waiter);
    }
  }

  waitFor(test, timeoutMs = 20000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((entry) => entry !== runner);
        reject(new Error("socket timeout"));
      }, timeoutMs);

      const runner = () => {
        try {
          const result = test(this.buffer);
          if (result === false || result === null || result === undefined) {
            if (this.closed) {
              clearTimeout(timer);
              reject(new Error("socket closed before response"));
              return true;
            }
            return false;
          }
          clearTimeout(timer);
          resolve(result);
          return true;
        } catch (error) {
          clearTimeout(timer);
          reject(error);
          return true;
        }
      };

      this.waiters.push(runner);
      runner();
    });
  }

  async readLine(timeoutMs = 20000) {
    return this.waitFor((buffer) => {
      const index = buffer.indexOf("\r\n");
      if (index === -1) return false;
      const line = buffer.subarray(0, index).toString("utf8");
      this.buffer = buffer.subarray(index + 2);
      return line;
    }, timeoutMs);
  }

  async readUntilTagged(tag, timeoutMs = 20000) {
    return this.waitFor((buffer) => {
      const startMarker = Buffer.from(`${tag} `);
      const inline = buffer.indexOf(startMarker) === 0 ? 0 : -1;
      const tagged = inline === 0 ? 0 : buffer.indexOf(Buffer.from(`\r\n${tag} `));
      const lineStart = inline === 0 ? 0 : tagged === -1 ? -1 : tagged + 2;
      if (lineStart === -1) return false;
      const lineEnd = buffer.indexOf(Buffer.from("\r\n"), lineStart);
      if (lineEnd === -1) return false;
      const out = buffer.subarray(0, lineEnd + 2);
      this.buffer = buffer.subarray(lineEnd + 2);
      return out;
    }, timeoutMs);
  }
}

async function writeSocket(socket, data) {
  debugLog("write", JSON.stringify(String(data).slice(0, 400)));
  if (socket.write(data)) return;
  await once(socket, "drain");
}

function imapQuote(value) {
  return `"${String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

function extractFetchLiteral(buffer) {
  const latin1 = buffer.toString("latin1");
  const match = /\{(\d+)\}\r\n/.exec(latin1);
  if (!match) {
    return { prefix: latin1, literal: Buffer.alloc(0) };
  }
  const literalLength = Number.parseInt(match[1], 10);
  const literalStart = match.index + match[0].length;
  return {
    prefix: latin1.slice(0, literalStart),
    literal: buffer.subarray(literalStart, literalStart + literalLength),
  };
}

class ImapClient {
  constructor(config) {
    this.config = config;
    this.tagCounter = 0;
    this.socket = null;
    this.reader = null;
  }

  nextTag() {
    this.tagCounter += 1;
    return `A${String(this.tagCounter).padStart(4, "0")}`;
  }

  async connect() {
    this.socket = await new Promise((resolveSocket, rejectSocket) => {
      const socket = tlsConnect(
        {
          host: this.config.imapHost,
          port: this.config.imapPort,
          servername: this.config.imapHost,
        },
        () => resolveSocket(socket)
      );
      socket.setTimeout(20000, () => socket.destroy(new Error("IMAP socket timeout")));
      socket.once("error", rejectSocket);
    });
    this.reader = new BufferSocket(this.socket);
    const greeting = await this.reader.readLine();
    debugLog("imap-greeting", greeting);
    if (!/\bOK\b/i.test(greeting)) fail(`IMAP greeting failed: ${greeting}`);
  }

  async command(commandText) {
    const tag = this.nextTag();
    await writeSocket(this.socket, `${tag} ${commandText}\r\n`);
    const response = await this.reader.readUntilTagged(tag);
    const text = response.toString("utf8");
    debugLog("imap-response", commandText, JSON.stringify(text.slice(0, 400)));
    const statusLine = text.match(new RegExp(`(?:^|\\r\\n)${tag} (OK|NO|BAD)`, "i"));
    if (!statusLine || statusLine[1].toUpperCase() !== "OK") {
      fail(`IMAP command failed: ${commandText}`);
    }
    return { tag, buffer: response, text };
  }

  async login(emailAddress, password) {
    await this.command(`LOGIN ${imapQuote(emailAddress)} ${imapQuote(password)}`);
  }

  async select(folder) {
    await this.command(`SELECT ${imapQuote(folder)}`);
  }

  async searchAllUids() {
    const response = await this.command("UID SEARCH ALL");
    const match = response.text.match(/\* SEARCH ?([0-9 ]*)/);
    return String((match && match[1]) || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
  }

  async fetchRaw(uid) {
    const response = await this.command(`UID FETCH ${uid} (UID FLAGS RFC822)`);
    const { prefix, literal } = extractFetchLiteral(response.buffer);
    const flagsMatch = prefix.match(/FLAGS \(([^)]*)\)/i);
    const flags = String((flagsMatch && flagsMatch[1]) || "")
      .split(/\s+/)
      .map((value) => value.trim())
      .filter(Boolean);
    return { uid, flags, raw: literal };
  }

  async logout() {
    if (!this.socket) return;
    try {
      await this.command("LOGOUT");
    } catch {}
    this.socket.end();
  }
}

class SmtpClient {
  constructor(config) {
    this.config = config;
    this.socket = null;
    this.reader = null;
  }

  async connect() {
    this.socket = await new Promise((resolveSocket, rejectSocket) => {
      const socket = tlsConnect(
        {
          host: this.config.smtpHost,
          port: this.config.smtpPort,
          servername: this.config.smtpHost,
        },
        () => resolveSocket(socket)
      );
      socket.setTimeout(20000, () => socket.destroy(new Error("SMTP socket timeout")));
      socket.once("error", rejectSocket);
    });
    this.reader = new BufferSocket(this.socket);
    await this.expect([220]);
  }

  async expect(allowedCodes) {
    const first = await this.reader.readLine();
    const code = Number.parseInt(first.slice(0, 3), 10);
    const lines = [first];
    debugLog("smtp-response", JSON.stringify(first));
    let current = first;
    while (current[3] === "-") {
      current = await this.reader.readLine();
      debugLog("smtp-response", JSON.stringify(current));
      lines.push(current);
    }
    if (!allowedCodes.includes(code)) {
      fail(`SMTP failed: ${lines.join(" | ")}`);
    }
    return { code, lines };
  }

  async sendCommand(commandText, allowedCodes) {
    debugLog("smtp-command", JSON.stringify(commandText));
    await writeSocket(this.socket, `${commandText}\r\n`);
    return this.expect(allowedCodes);
  }

  async login(emailAddress, password) {
    await this.sendCommand(`EHLO localhost`, [250]);
    const payload = Buffer.from(`\u0000${emailAddress}\u0000${password}`, "utf8").toString("base64");
    await this.sendCommand(`AUTH PLAIN ${payload}`, [235]);
  }

  async sendMail(message) {
    await this.sendCommand(`MAIL FROM:<${message.from}>`, [250]);
    for (const recipient of [...message.to, ...message.cc, ...message.bcc]) {
      await this.sendCommand(`RCPT TO:<${recipient}>`, [250, 251]);
    }
    await this.sendCommand("DATA", [354]);

    const lines = [
      `From: ${message.from}`,
      `To: ${message.to.join(", ")}`,
      ...(message.cc.length ? [`Cc: ${message.cc.join(", ")}`] : []),
      `Subject: ${message.subject}`,
      `Message-ID: ${message.messageId}`,
      `Date: ${new Date().toUTCString()}`,
      "Content-Type: text/plain; charset=utf-8",
      "",
      ...String(message.body || "").split(/\r?\n/).map((line) => (line.startsWith(".") ? `.${line}` : line)),
      ".",
    ];
    await writeSocket(this.socket, `${lines.join("\r\n")}\r\n`);
    await this.expect([250]);
  }

  async close() {
    if (!this.socket) return;
    try {
      this.socket.end("QUIT\r\n");
    } catch {}
  }
}

function normalizeOptions(rawArgv) {
  const argv = [...rawArgv];
  const command = argv.shift();
  if (!command) fail("Usage: communication_mail_cli.mjs <sync|send|test|list> [options]");

  const options = {
    command,
    to: [],
    cc: [],
    bcc: [],
    db: DEFAULTS.db,
    rawDir: DEFAULTS.rawDir,
    schema: DEFAULTS.schema,
    agentBinary: DEFAULTS.agentBinary,
    provider: DEFAULTS.provider,
    folder: DEFAULTS.folder,
    limit: DEFAULTS.limit,
    trustLevel: DEFAULTS.trustLevel,
    emitInterrupts: DEFAULTS.emitInterrupts,
    interruptChannel: DEFAULTS.interruptChannel,
    imapHost: DEFAULTS.imapHost,
    imapPort: DEFAULTS.imapPort,
    smtpHost: DEFAULTS.smtpHost,
    smtpPort: DEFAULTS.smtpPort,
    email: process.env.CTO_EMAIL_ADDRESS || "",
    passwordEnv: "CTO_EMAIL_PASSWORD",
    graphAccessToken: process.env.CTO_EMAIL_GRAPH_ACCESS_TOKEN || "",
    graphBaseUrl: process.env.CTO_EMAIL_GRAPH_BASE_URL || DEFAULTS.graphBaseUrl,
    graphUser: process.env.CTO_EMAIL_GRAPH_USER || DEFAULTS.graphUser,
    ewsUrl: process.env.CTO_EMAIL_EWS_URL || "",
    owaUrl: process.env.CTO_EMAIL_OWA_URL || "",
    ewsVersion: process.env.CTO_EMAIL_EWS_VERSION || DEFAULTS.ewsVersion,
    ewsAuthType: process.env.CTO_EMAIL_EWS_AUTH_TYPE || DEFAULTS.ewsAuthType,
    ewsUsername: process.env.CTO_EMAIL_EWS_USERNAME || "",
    ewsBearerToken: process.env.CTO_EMAIL_EWS_BEARER_TOKEN || "",
    activeSyncServer: process.env.CTO_EMAIL_ACTIVESYNC_SERVER || "",
    activeSyncUsername: process.env.CTO_EMAIL_ACTIVESYNC_USERNAME || "",
    activeSyncPath: process.env.CTO_EMAIL_ACTIVESYNC_PATH || DEFAULTS.activeSyncPath,
    activeSyncDeviceId: process.env.CTO_EMAIL_ACTIVESYNC_DEVICE_ID || "",
    activeSyncDeviceType: process.env.CTO_EMAIL_ACTIVESYNC_DEVICE_TYPE || DEFAULTS.activeSyncDeviceType,
    activeSyncProtocolVersion:
      process.env.CTO_EMAIL_ACTIVESYNC_PROTOCOL_VERSION || DEFAULTS.activeSyncProtocolVersion,
    activeSyncPolicyKey:
      process.env.CTO_EMAIL_ACTIVESYNC_POLICY_KEY || DEFAULTS.activeSyncPolicyKey,
    verifySend: process.env.CTO_EMAIL_VERIFY_SEND || DEFAULTS.verifySend,
    sentVerifyWindowSeconds: process.env.CTO_EMAIL_SENT_VERIFY_WINDOW_SECONDS || DEFAULTS.sentVerifyWindowSeconds,
  };

  while (argv.length) {
    const token = argv.shift();
    if (!token.startsWith("--")) fail(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[0] && !argv[0].startsWith("--") ? argv.shift() : "true";
    if (["to", "cc", "bcc"].includes(key)) {
      options[key].push(value);
      continue;
    }
    options[key.replace(/-([a-z])/g, (_m, char) => char.toUpperCase())] = value;
  }

  options.limit = Number.parseInt(options.limit, 10) || DEFAULTS.limit;
  options.imapPort = Number.parseInt(options.imapPort, 10) || DEFAULTS.imapPort;
  options.smtpPort = Number.parseInt(options.smtpPort, 10) || DEFAULTS.smtpPort;
  options.sentVerifyWindowSeconds =
    Number.parseInt(options.sentVerifyWindowSeconds, 10) || DEFAULTS.sentVerifyWindowSeconds;
  options.password = process.env[options.passwordEnv] || "";
  options.provider = normalizeProvider(options.provider);
  return options;
}

function normalizeProvider(value) {
  const normalized = String(value || DEFAULTS.provider)
    .trim()
    .toLowerCase();
  if (!normalized || normalized === "classic" || normalized === "smtp" || normalized === "imap-smtp") {
    return "imap";
  }
  if (normalized === "m365" || normalized === "graph-cloud" || normalized === "exchange-online") {
    return "graph";
  }
  if (normalized === "outlook" || normalized === "exchange") {
    return "ews";
  }
  if (normalized === "owa") {
    return "owa";
  }
  if (normalized === "eas") {
    return "activesync";
  }
  return normalized;
}

function requireCommonIdentity(options) {
  if (!options.email) fail("Missing --email or CTO_EMAIL_ADDRESS.");
}

function requirePassword(options, label) {
  if (!options.password) fail(`Missing password in env ${options.passwordEnv} for ${label}.`);
}

function requireProviderCredentials(options) {
  requireCommonIdentity(options);
  if (options.provider === "imap") {
    requirePassword(options, "IMAP/SMTP");
    return;
  }
  if (options.provider === "graph") {
    if (!options.graphAccessToken) {
      fail("Missing --graph-access-token or CTO_EMAIL_GRAPH_ACCESS_TOKEN for Graph.");
    }
    return;
  }
  if (options.provider === "ews" || options.provider === "owa") {
    if (!resolveEwsUrl(options)) {
      fail("Missing --ews-url or --owa-url / CTO_EMAIL_EWS_URL / CTO_EMAIL_OWA_URL for EWS/OWA.");
    }
    const authType = String(options.ewsAuthType || DEFAULTS.ewsAuthType).trim().toLowerCase();
    if (authType === "basic") {
      if (!options.ewsUsername) options.ewsUsername = options.email;
      requirePassword(options, "EWS basic auth");
    } else if (authType === "bearer") {
      if (!options.ewsBearerToken) fail("Missing --ews-bearer-token or CTO_EMAIL_EWS_BEARER_TOKEN.");
    } else if (authType !== "ntlm") {
      fail(`Unsupported EWS auth type: ${authType}`);
    }
    return;
  }
  if (options.provider === "activesync") {
    if (!options.activeSyncServer) {
      fail("Missing --active-sync-server or CTO_EMAIL_ACTIVESYNC_SERVER.");
    }
    if (!options.activeSyncUsername) options.activeSyncUsername = options.email;
    requirePassword(options, "ActiveSync");
    return;
  }
  fail(`Unsupported email provider: ${options.provider}`);
}

function buildProfileJson(options) {
  return JSON.stringify({
    provider: options.provider,
    imapHost: options.imapHost,
    imapPort: options.imapPort,
    smtpHost: options.smtpHost,
    smtpPort: options.smtpPort,
    folder: options.folder,
    graphBaseUrl: options.graphBaseUrl,
    graphUser: options.graphUser,
    ewsUrl: resolveEwsUrl(options),
    owaUrl: options.owaUrl || "",
    ewsVersion: options.ewsVersion,
    ewsAuthType: options.ewsAuthType,
    activeSyncServer: options.activeSyncServer,
    activeSyncUsername: options.activeSyncUsername,
    activeSyncPath: options.activeSyncPath,
    activeSyncDeviceId: options.activeSyncDeviceId,
    activeSyncDeviceType: options.activeSyncDeviceType,
    activeSyncProtocolVersion: options.activeSyncProtocolVersion,
    activeSyncPolicyKey: options.activeSyncPolicyKey,
  });
}

function deriveEwsUrlFromOwaUrl(raw) {
  const text = String(raw || "").trim();
  if (!text) return "";
  try {
    const parsed = new URL(text);
    parsed.pathname = "/EWS/Exchange.asmx";
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return "";
  }
}

function resolveEwsUrl(options) {
  return String(options.ewsUrl || deriveEwsUrlFromOwaUrl(options.owaUrl || "") || "").trim();
}

function folderHintToMailboxFolder(folderHint) {
  const normalized = String(folderHint || "inbox").trim().toLowerCase();
  if (normalized === "sent" || normalized === "sentitems") return "sentitems";
  if (normalized === "drafts") return "drafts";
  return "inbox";
}

function mailboxFolderToHint(folderId) {
  const normalized = String(folderId || "inbox").trim().toLowerCase();
  if (normalized === "sentitems" || normalized === "sent") return "sent";
  if (normalized === "drafts") return "drafts";
  return "inbox";
}

function jsonArrayOrEmpty(value) {
  if (Array.isArray(value)) return value;
  return [];
}

function normalizeGraphMailItem(raw, folderIdFallback = "inbox") {
  if (!raw || !raw.id) return null;
  if (raw["@removed"]) return null;
  const from = raw.from?.emailAddress || {};
  const receivedAt = raw.receivedDateTime || raw.sentDateTime || null;
  const sentAt = raw.sentDateTime || raw.receivedDateTime || null;
  return {
    remoteId: String(raw.id),
    threadKey: String(raw.conversationId || raw.id),
    folderHint: mailboxFolderToHint(raw.parentFolderId || folderIdFallback),
    subject: String(raw.subject || "(ohne Betreff)"),
    senderDisplay: String(from.name || from.address || "unknown"),
    senderAddress: String(from.address || ""),
    recipientAddresses: jsonArrayOrEmpty(raw.toRecipients).map((entry) =>
      String(entry?.emailAddress?.address || "").trim().toLowerCase()
    ),
    ccAddresses: jsonArrayOrEmpty(raw.ccRecipients).map((entry) =>
      String(entry?.emailAddress?.address || "").trim().toLowerCase()
    ),
    bodyText: String(raw.body?.contentType || "").toLowerCase() === "html"
      ? stripHtml(raw.body?.content || "")
      : String(raw.body?.content || ""),
    bodyHtml: String(raw.body?.contentType || "").toLowerCase() === "html" ? String(raw.body?.content || "") : "",
    preview: previewText(raw.bodyPreview || raw.body?.content || raw.subject),
    seen: raw.isRead === false ? 0 : 1,
    hasAttachments: raw.hasAttachments ? 1 : 0,
    externalCreatedAt: receivedAt || sentAt || nowIso(),
    metadata: {
      internetMessageId: String(raw.internetMessageId || ""),
      conversationId: String(raw.conversationId || raw.id),
      graphFolderId: String(raw.parentFolderId || folderIdFallback || "inbox"),
      sourceUpdatedAt: raw.lastModifiedDateTime || receivedAt || null,
    },
  };
}

function normalizeActiveSyncMailItem(raw, folderIdFallback = "inbox") {
  const remoteId = raw?.id || raw?.serverId || raw?.remote_id || null;
  if (!remoteId) return null;
  const from = raw?.from?.emailAddress || {};
  const receivedAt = raw?.receivedDateTime || raw?.received_at || raw?.sentDateTime || null;
  return {
    remoteId: String(remoteId),
    threadKey: String(raw?.conversationId || remoteId),
    folderHint: mailboxFolderToHint(raw?.folderId || folderIdFallback),
    subject: String(raw?.subject || "(ohne Betreff)"),
    senderDisplay: String(from.name || from.address || raw?.from_name || "unknown"),
    senderAddress: String(from.address || raw?.from_email || ""),
    recipientAddresses: jsonArrayOrEmpty(raw?.toRecipients).map((entry) =>
      String(entry?.emailAddress?.address || "").trim().toLowerCase()
    ),
    ccAddresses: jsonArrayOrEmpty(raw?.ccRecipients).map((entry) =>
      String(entry?.emailAddress?.address || "").trim().toLowerCase()
    ),
    bodyText: String(raw?.body?.content || raw?.bodyText || ""),
    bodyHtml: String(raw?.body?.contentType || "").toLowerCase() === "html" ? String(raw?.body?.content || "") : "",
    preview: previewText(raw?.bodyPreview || raw?.preview || raw?.subject),
    seen: raw?.isRead === false || raw?.unread === true ? 0 : 1,
    hasAttachments: raw?.hasAttachments || raw?.has_attachments ? 1 : 0,
    externalCreatedAt: receivedAt || nowIso(),
    metadata: {
      conversationId: String(raw?.conversationId || remoteId),
      activeSyncFolderId: String(raw?.folderId || folderIdFallback || "inbox"),
      sourceUpdatedAt: raw?.lastModifiedDateTime || raw?.source_updated_at || receivedAt || null,
    },
  };
}

function normalizeEwsMailItem(raw, folderIdFallback = "inbox") {
  const remoteId =
    raw?.["t:ItemId"]?.attributes?.Id ||
    raw?.ItemId?.attributes?.Id ||
    raw?.id ||
    raw?.Id ||
    null;
  if (!remoteId) return null;
  const subject =
    raw?.["t:Subject"]?.$value ||
    raw?.Subject ||
    raw?.["t:Subject"] ||
    "(ohne Betreff)";
  const fromNode = raw?.["t:From"]?.["t:Mailbox"] || raw?.From?.Mailbox || raw?.From || {};
  const senderAddress =
    fromNode?.["t:EmailAddress"]?.$value ||
    fromNode?.EmailAddress?.$value ||
    fromNode?.EmailAddress ||
    "";
  const senderDisplay =
    fromNode?.["t:Name"]?.$value ||
    fromNode?.Name?.$value ||
    fromNode?.Name ||
    senderAddress ||
    "unknown";
  const isReadRaw =
    raw?.["t:IsRead"]?.$value ??
    raw?.IsRead?.$value ??
    raw?.IsRead ??
    null;
  const receivedAt =
    raw?.["t:DateTimeReceived"]?.$value ||
    raw?.DateTimeReceived?.$value ||
    raw?.DateTimeReceived ||
    null;
  const recipientsNode = raw?.["t:ToRecipients"] || raw?.ToRecipients || {};
  const ccNode = raw?.["t:CcRecipients"] || raw?.CcRecipients || {};
  const collectMailboxAddresses = (node) => {
    const mailboxes = []
      .concat(node?.["t:Mailbox"] || [])
      .concat(node?.Mailbox || [])
      .filter(Boolean);
    return mailboxes
      .map((entry) =>
        String(
          entry?.["t:EmailAddress"]?.$value || entry?.EmailAddress?.$value || entry?.EmailAddress || ""
        )
          .trim()
          .toLowerCase()
      )
      .filter(Boolean);
  };
  return {
    remoteId: String(remoteId),
    threadKey: String(
      raw?.["t:ConversationId"]?.attributes?.Id || raw?.ConversationId?.attributes?.Id || remoteId
    ),
    folderHint: mailboxFolderToHint(folderIdFallback),
    subject: String(subject || "(ohne Betreff)"),
    senderDisplay: String(senderDisplay || "unknown"),
    senderAddress: String(senderAddress || ""),
    recipientAddresses: collectMailboxAddresses(recipientsNode),
    ccAddresses: collectMailboxAddresses(ccNode),
    bodyText: "",
    bodyHtml: "",
    preview: previewText(subject),
    seen: String(isReadRaw).toLowerCase() === "false" ? 0 : 1,
    hasAttachments:
      String(raw?.["t:HasAttachments"]?.$value || raw?.HasAttachments || "").toLowerCase() === "true" ? 1 : 0,
    externalCreatedAt: receivedAt || nowIso(),
    metadata: {
      conversationId: String(
        raw?.["t:ConversationId"]?.attributes?.Id || raw?.ConversationId?.attributes?.Id || remoteId
      ),
      ewsFolderId: String(folderIdFallback || "inbox"),
    },
  };
}

async function createMailboxClient(options) {
  if (options.provider === "graph") {
    const graphClient = new GraphClient({
      accessToken: options.graphAccessToken,
      baseUrl: options.graphBaseUrl,
      user: options.graphUser,
    });
    return {
      provider: "graph",
      client: new MailboxClient({ mode: "graph", graphClient }),
    };
  }
  if (options.provider === "activesync") {
    const activeSyncClient = new ActiveSyncClient({
      server: options.activeSyncServer,
      username: options.activeSyncUsername,
      password: options.password,
      useSSL: true,
      path: options.activeSyncPath,
      deviceId: options.activeSyncDeviceId,
      deviceType: options.activeSyncDeviceType,
      protocolVersion: options.activeSyncProtocolVersion,
      policyKey: options.activeSyncPolicyKey,
    });
    return {
      provider: "activesync",
      client: new MailboxClient({ mode: "activesync", activeSyncClient }),
    };
  }
  if (options.provider === "ews" || options.provider === "owa") {
    const authType = String(options.ewsAuthType || DEFAULTS.ewsAuthType).trim().toLowerCase();
    const credentials =
      authType === "basic"
        ? new ExchangeCredentials({
            authType: "basic",
            username: options.ewsUsername || options.email,
            password: options.password,
          })
        : authType === "bearer"
          ? new ExchangeCredentials({
              authType: "bearer",
              token: options.ewsBearerToken,
            })
          : new ExchangeCredentials({ authType: "ntlm" });
    const ewsAccount = createExchangeAccount({
      ewsUrl: resolveEwsUrl(options),
      primarySmtpAddress: options.email,
      version: options.ewsVersion,
      credentials,
      extraHeaders: {},
    });
    return {
      provider: options.provider,
      client: new MailboxClient({ mode: "ews", ewsAccount }),
    };
  }
  fail(`Unsupported mailbox client provider: ${options.provider}`);
}

function emitAgentInterrupt(options, speaker, summary) {
  const output = execFileSync(
    options.agentBinary,
    ["channel-interrupt", options.interruptChannel, speaker, summary],
    {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  return String(output || "").trim();
}

async function listMailboxMessages(client, folderHint, limit) {
  const folderId = folderHintToMailboxFolder(folderHint);
  if (folderId === "sentitems") return client.listSent({ pageSize: limit, query: null });
  if (folderId === "drafts") return client.listDrafts({ pageSize: limit, query: null });
  return client.listInbox({ pageSize: limit, query: null });
}

function normalizeMailboxMessage(provider, raw, folderHint) {
  if (provider === "graph") return normalizeGraphMailItem(raw, folderHintToMailboxFolder(folderHint));
  if (provider === "activesync") return normalizeActiveSyncMailItem(raw, folderHintToMailboxFolder(folderHint));
  return normalizeEwsMailItem(raw, folderHintToMailboxFolder(folderHint));
}

async function verifyImapSentCopy(options, messageId) {
  const sentFolders = [
    "Sent",
    "Sent Items",
    "Sent Messages",
    "Gesendet",
    "INBOX.Sent",
    "INBOX.Sent Items",
  ];
  const verifyWindowSeconds = Math.max(1, Number.parseInt(options.sentVerifyWindowSeconds, 10) || 1);
  const attempts = Math.max(1, Math.min(verifyWindowSeconds, 30));
  const imap = new ImapClient(options);
  try {
    await imap.connect();
    await imap.login(options.email, options.password);
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      for (const folder of sentFolders) {
        try {
          await imap.select(folder);
        } catch {
          continue;
        }
        const uids = await imap.searchAllUids();
        for (const uid of uids.slice(-15).reverse()) {
          const fetched = await imap.fetchRaw(uid);
          const parsed = parseRfc822(fetched.raw.toString("utf8"));
          if (String(parsed.messageId || "").trim() === String(messageId || "").trim()) {
            return {
              confirmed: true,
              method: "imap-sent-folder",
              folder,
              remoteId: messageId,
            };
          }
        }
      }
      if (attempt + 1 < attempts) {
        await sleep(1000);
      }
    }
    return {
      confirmed: false,
      method: "imap-sent-folder",
      detail: "message-id not found in checked sent folders",
    };
  } finally {
    await imap.logout();
  }
}

function mailboxRecipientSet(normalized) {
  return new Set([...(normalized.recipientAddresses || []), ...(normalized.ccAddresses || [])]);
}

function verificationRecipientSet(options) {
  return new Set([
    ...options.to.map((value) => String(value || "").trim().toLowerCase()),
    ...options.cc.map((value) => String(value || "").trim().toLowerCase()),
  ]);
}

function messageLooksLikeSentCopy(normalized, options, earliestIso) {
  if (!normalized) return false;
  if (String(normalized.subject || "").trim() !== String(options.subject || "(ohne Betreff)").trim()) {
    return false;
  }
  const candidateTime = Date.parse(normalized.externalCreatedAt || "");
  const earliest = Date.parse(earliestIso || "");
  if (Number.isFinite(candidateTime) && Number.isFinite(earliest) && candidateTime + 1000 < earliest) {
    return false;
  }
  const expectedRecipients = verificationRecipientSet(options);
  const actualRecipients = mailboxRecipientSet(normalized);
  for (const recipient of expectedRecipients) {
    if (!actualRecipients.has(recipient)) {
      return false;
    }
  }
  return true;
}

async function verifyProviderSentCopy(options, earliestIso) {
  const { provider, client } = await createMailboxClient(options);
  const verifyWindowSeconds = Math.max(1, Number.parseInt(options.sentVerifyWindowSeconds, 10) || 1);
  const attempts = Math.max(1, Math.min(verifyWindowSeconds, 30));
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const rawResult = await client.listSent({ pageSize: 25, query: options.subject || null });
    const items = Array.isArray(rawResult?.value)
      ? rawResult.value
      : Array.isArray(rawResult?.items)
        ? rawResult.items
        : [];
    for (const raw of items) {
      const normalized = normalizeMailboxMessage(provider, raw, "sentitems");
      if (!messageLooksLikeSentCopy(normalized, options, earliestIso)) continue;
      return {
        confirmed: true,
        method: `${provider}-sent-folder`,
        remoteId: normalized.remoteId,
        threadKey: normalized.threadKey,
        observedAt: normalized.externalCreatedAt || nowIso(),
      };
    }
    if (attempt + 1 < attempts) {
      await sleep(1000);
    }
  }
  return {
    confirmed: false,
    method: `${provider}-sent-folder`,
    detail: "matching sent copy not found",
  };
}

async function verifySentDelivery(options, messageId, earliestIso) {
  if (!toBool(options.verifySend)) {
    return { confirmed: false, skipped: true, method: "disabled" };
  }
  if (options.provider === "imap") {
    return verifyImapSentCopy(options, messageId);
  }
  return verifyProviderSentCopy(options, earliestIso);
}

async function syncProviderMail(options, accountKey, startedAt) {
  const { provider, client } = await createMailboxClient(options);
  const observedAt = nowIso();
  const rawResult = await listMailboxMessages(client, options.folder, options.limit);
  const items = Array.isArray(rawResult?.value)
    ? rawResult.value
    : Array.isArray(rawResult?.items)
      ? rawResult.items
      : [];

  let storedCount = 0;
  for (const raw of items) {
    const normalized = normalizeMailboxMessage(provider, raw, options.folder);
    if (!normalized) continue;
    const messageKey = messageKeyFromRemote(accountKey, normalized.folderHint, normalized.remoteId);
    const alreadyKnown = messageExists(options.db, messageKey);
    upsertMessage(options.db, {
      messageKey,
      channel: DEFAULTS.channel,
      accountKey,
      threadKey: normalized.threadKey,
      remoteId: normalized.remoteId,
      direction: "inbound",
      folderHint: normalized.folderHint,
      senderDisplay: normalized.senderDisplay,
      senderAddress: normalized.senderAddress,
      recipientAddressesJson: JSON.stringify(normalized.recipientAddresses || []),
      ccAddressesJson: JSON.stringify(normalized.ccAddresses || []),
      bccAddressesJson: JSON.stringify([]),
      subject: normalized.subject,
      preview: normalized.preview,
      bodyText: normalized.bodyText,
      bodyHtml: normalized.bodyHtml,
      rawPayloadRef: "",
      trustLevel: options.trustLevel,
      status: "received",
      seen: normalized.seen,
      hasAttachments: normalized.hasAttachments,
      externalCreatedAt: normalized.externalCreatedAt || observedAt,
      observedAt,
      metadataJson: JSON.stringify(normalized.metadata || {}),
    });
    refreshThread(options.db, normalized.threadKey);
    if (!alreadyKnown && toBool(options.emitInterrupts)) {
      const speaker = normalized.senderAddress
        ? `${normalized.senderDisplay} <${normalized.senderAddress}>`
        : normalized.senderDisplay;
      const summary = [
        `E-Mail eingegangen von ${speaker || "unknown sender"}.`,
        `Betreff: ${normalized.subject || "(ohne Betreff)"}`,
        normalized.preview ? `Vorschau: ${normalized.preview}` : "",
      ]
        .filter(Boolean)
        .join("\n");
      emitAgentInterrupt(options, speaker || "unknown sender", summary);
    }
    storedCount += 1;
  }

  return {
    provider,
    fetchedCount: items.length,
    storedCount,
    metadata: {
      adapter: "exchange-mail-core",
      provider,
      startedAt,
    },
  };
}

function outboundMessageRecord(options, body, messageId, delivery) {
  const accountKey = accountKeyFromEmail(options.email);
  const observedAt = nowIso();
  const remoteId = messageId;
  const threadKey = String(delivery?.threadKey || remoteId);
  const status = delivery?.confirmed ? "confirmed" : "accepted";
  return {
    accountKey,
    message: {
      messageKey: messageKeyFromRemote(accountKey, "sent", remoteId),
      channel: DEFAULTS.channel,
      accountKey,
      threadKey,
      remoteId,
      direction: "outbound",
      folderHint: "sent",
      senderDisplay: "",
      senderAddress: options.email.toLowerCase(),
      recipientAddressesJson: JSON.stringify(options.to.map((value) => value.toLowerCase())),
      ccAddressesJson: JSON.stringify(options.cc.map((value) => value.toLowerCase())),
      bccAddressesJson: JSON.stringify(options.bcc.map((value) => value.toLowerCase())),
      subject: options.subject || "(ohne Betreff)",
      preview: previewText(body),
      bodyText: body,
      bodyHtml: "",
      rawPayloadRef: "",
      trustLevel: options.trustLevel,
      status,
      seen: 1,
      hasAttachments: 0,
      externalCreatedAt: observedAt,
      observedAt,
      metadataJson: JSON.stringify({
        messageId,
        delivery: delivery || { confirmed: false, method: "none" },
      }),
    },
  };
}

async function sendMail(options) {
  requireProviderCredentials(options);
  await ensureSchema(options.db, options.schema);
  const accountKey = accountKeyFromEmail(options.email);
  const timestamp = nowIso();
  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.email.toLowerCase(),
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: timestamp,
    updatedAt: timestamp,
    lastInboundOkAt: null,
    lastOutboundOkAt: timestamp,
  });

  const emailParts = options.email.split("@");
  const messageId = `<${randomUUID()}@${emailParts[emailParts.length - 1]}>`;
  const sendStartedAt = nowIso();
  if (options.provider === "imap") {
    const smtp = new SmtpClient(options);
    try {
      await smtp.connect();
      await smtp.login(options.email, options.password);
      await smtp.sendMail({
        from: options.email,
        to: options.to.map((value) => value.toLowerCase()),
        cc: options.cc.map((value) => value.toLowerCase()),
        bcc: options.bcc.map((value) => value.toLowerCase()),
        subject: options.subject || "(ohne Betreff)",
        body: options.body || "",
        messageId,
      });
    } finally {
      await smtp.close();
    }
  } else {
    const { provider, client } = await createMailboxClient(options);
    if (provider === "activesync") {
      fail("ActiveSync outbound send is not implemented in the transplanted client.");
    }
    await client.sendMail({
      subject: options.subject || "(ohne Betreff)",
      body: options.body || "",
      bodyType: "Text",
      to: options.to.map((value) => value.toLowerCase()),
      cc: options.cc.map((value) => value.toLowerCase()),
      bcc: options.bcc.map((value) => value.toLowerCase()),
      saveCopyFolderId: "sentitems",
    });
  }

  const delivery = await verifySentDelivery(options, messageId, sendStartedAt);
  const record = outboundMessageRecord(options, options.body || "", messageId, delivery);
  upsertMessage(options.db, record.message);
  refreshThread(options.db, record.message.threadKey);
  return {
    ok: true,
    accountKey,
    to: options.to,
    subject: options.subject || "(ohne Betreff)",
    messageId,
    status: delivery.confirmed ? "confirmed" : "accepted",
    delivery,
    dbPath: options.db,
  };
}

async function testMailSetup(options) {
  requireProviderCredentials(options);
  await ensureSchema(options.db, options.schema);
  const accountKey = accountKeyFromEmail(options.email);
  const timestamp = nowIso();
  const result = {
    ok: true,
    channel: DEFAULTS.channel,
    provider: options.provider,
    accountKey,
    checks: [],
    dbPath: options.db,
  };

  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.email.toLowerCase(),
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: timestamp,
    updatedAt: timestamp,
    lastInboundOkAt: null,
    lastOutboundOkAt: null,
  });

  if (options.provider === "imap") {
    const imap = new ImapClient(options);
    try {
      await imap.connect();
      result.checks.push({ name: "imap_connect", ok: true });
      await imap.login(options.email, options.password);
      result.checks.push({ name: "imap_login", ok: true });
      await imap.select("INBOX");
      result.checks.push({ name: "imap_inbox_select", ok: true });
      const sentCheck = await verifyImapSentCopy(options, "<ctox-test-probe@invalid>");
      result.checks.push({
        name: "imap_sent_folder_probe",
        ok: sentCheck.confirmed || sentCheck.detail === "message-id not found in checked sent folders",
        detail: sentCheck.detail || sentCheck.folder || sentCheck.method,
      });
    } finally {
      await imap.logout();
    }
  } else {
    const { provider, client } = await createMailboxClient(options);
    result.checks.push({ name: `${provider}_client`, ok: true });
    await client.listInbox({ pageSize: 1, query: null });
    result.checks.push({ name: `${provider}_inbox_probe`, ok: true });
    await client.listSent({ pageSize: 1, query: null });
    result.checks.push({ name: `${provider}_sent_probe`, ok: true });
  }

  return result;
}

async function syncMail(options) {
  requireProviderCredentials(options);
  await ensureSchema(options.db, options.schema);
  const accountKey = accountKeyFromEmail(options.email);
  const startedAt = nowIso();
  let fetchedCount = 0;
  let storedCount = 0;

  upsertAccount(options.db, {
    accountKey,
    channel: DEFAULTS.channel,
    address: options.email.toLowerCase(),
    provider: options.provider,
    profileJson: buildProfileJson(options),
    createdAt: startedAt,
    updatedAt: startedAt,
    lastInboundOkAt: null,
    lastOutboundOkAt: null,
  });

  try {
    if (options.provider === "imap") {
      const imap = new ImapClient(options);
      try {
        await imap.connect();
        await imap.login(options.email, options.password);
        await imap.select(options.folder);
        const allUids = await imap.searchAllUids();
        const selected = allUids.slice(-options.limit).reverse();
        fetchedCount = selected.length;

        for (const uid of selected) {
          const fetched = await imap.fetchRaw(uid);
          const parsed = parseRfc822(fetched.raw);
          const senderAddress = extractAddress(parsed.fromHeader);
          const senderDisplay = extractDisplayName(parsed.fromHeader) || senderAddress || "unknown";
          const remoteId = uid;
          const threadKey = threadKeyFromEmail(parsed, `${accountKey}::${remoteId}`);
          const messageKey = messageKeyFromRemote(accountKey, options.folder, remoteId);
          const alreadyKnown = messageExists(options.db, messageKey);
          const rawPayloadRef = await writeRawPayload(options.rawDir, remoteId, fetched.raw);
          const observedAt = nowIso();
          const preview = previewText(parsed.bodyText || parsed.subject);

          upsertMessage(options.db, {
            messageKey,
            channel: DEFAULTS.channel,
            accountKey,
            threadKey,
            remoteId,
            direction: "inbound",
            folderHint: options.folder,
            senderDisplay,
            senderAddress,
            recipientAddressesJson: JSON.stringify(extractAddresses(parsed.toHeader)),
            ccAddressesJson: JSON.stringify(extractAddresses(parsed.ccHeader)),
            bccAddressesJson: JSON.stringify([]),
            subject: parsed.subject,
            preview,
            bodyText: parsed.bodyText,
            bodyHtml: parsed.bodyHtml || "",
            rawPayloadRef,
            trustLevel: options.trustLevel,
            status: "received",
            seen: fetched.flags.includes("\\Seen") ? 1 : 0,
            hasAttachments: parsed.hasAttachments ? 1 : 0,
            externalCreatedAt: parsed.sentAt || observedAt,
            observedAt,
            metadataJson: JSON.stringify({
              messageId: parsed.messageId,
              references: parsed.references,
              inReplyTo: parsed.inReplyTo,
              imapFlags: fetched.flags,
            }),
          });
          refreshThread(options.db, threadKey);
          if (!alreadyKnown && toBool(options.emitInterrupts)) {
            const speaker = senderAddress ? `${senderDisplay} <${senderAddress}>` : senderDisplay;
            const summary = [
              `E-Mail eingegangen von ${speaker || "unknown sender"}.`,
              `Betreff: ${parsed.subject || "(ohne Betreff)"}`,
              preview ? `Vorschau: ${preview}` : "",
            ]
              .filter(Boolean)
              .join("\n");
            emitAgentInterrupt(options, speaker || "unknown sender", summary);
          }
          storedCount += 1;
        }
      } finally {
        await imap.logout();
      }
    } else {
      const synced = await syncProviderMail(options, accountKey, startedAt);
      fetchedCount = synced.fetchedCount;
      storedCount = synced.storedCount;
    }

    const finishedAt = nowIso();
    upsertAccount(options.db, {
      accountKey,
      channel: DEFAULTS.channel,
      address: options.email.toLowerCase(),
      provider: options.provider,
      profileJson: buildProfileJson(options),
      createdAt: startedAt,
      updatedAt: finishedAt,
      lastInboundOkAt: finishedAt,
      lastOutboundOkAt: null,
    });
    recordSyncRun(options.db, {
      runKey: randomUUID(),
      channel: DEFAULTS.channel,
      accountKey,
      folderHint: options.folder,
      startedAt,
      finishedAt,
      ok: true,
      fetchedCount,
      storedCount,
      errorText: "",
      metadataJson: JSON.stringify({
        adapter: options.provider === "imap" ? "js-mail-template" : "exchange-mail-core",
        provider: options.provider,
      }),
    });
    return {
      ok: true,
      accountKey,
      folder: options.folder,
      fetchedCount,
      storedCount,
      dbPath: options.db,
    };
  } catch (error) {
    const finishedAt = nowIso();
    recordSyncRun(options.db, {
      runKey: randomUUID(),
      channel: DEFAULTS.channel,
      accountKey,
      folderHint: options.folder,
      startedAt,
      finishedAt,
      ok: false,
      fetchedCount,
      storedCount,
      errorText: String((error && error.message) || error),
      metadataJson: JSON.stringify({
        adapter: options.provider === "imap" ? "js-mail-template" : "exchange-mail-core",
        provider: options.provider,
      }),
    });
    throw error;
  }
}

async function listMessages(options) {
  await ensureSchema(options.db, options.schema);
  const rows = parseJsonOutput(
    runSql(
      options.db,
      `
      SELECT channel, account_key, folder_hint, direction, subject, sender_address, external_created_at, preview
      FROM communication_messages
      ORDER BY external_created_at DESC, observed_at DESC
      LIMIT ${sqlValue(options.limit)}
      `,
      { json: true }
    )
  );
  return {
    ok: true,
    count: rows.length,
    dbPath: options.db,
    messages: rows,
  };
}

async function main() {
  const options = normalizeOptions(process.argv.slice(2));
  let result;
  if (options.command === "send") {
    if (!options.subject) fail("Missing --subject for send.");
    if (!options.body) fail("Missing --body for send.");
    if (!options.to.length) fail("Need at least one --to recipient.");
    result = await sendMail(options);
  } else if (options.command === "sync") {
    result = await syncMail(options);
  } else if (options.command === "test") {
    result = await testMailSetup(options);
  } else if (options.command === "list") {
    result = await listMessages(options);
  } else {
    fail(`Unsupported command: ${options.command}`);
  }

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

export { decodeTransferEncodedBody, main, parseMimeEntity, parseRfc822, previewText, stripHtml };

const executedPath = process.argv[1] ? resolve(process.argv[1]) : "";
const currentModulePath = fileURLToPath(import.meta.url);

if (executedPath === currentModulePath) {
  main().catch((error) => {
    process.stdout.write(
      `${JSON.stringify({ ok: false, error: String((error && error.message) || error) }, null, 2)}\n`
    );
    process.exitCode = 1;
  });
}
