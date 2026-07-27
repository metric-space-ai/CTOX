#!/usr/bin/env node
// Statischer Shell-Server fuer die Snap-Repro: umgeht die haengende
// Index-Route der Live-Instanz. Liefert das bereits injizierte index.html
// (Session + Sync-Config) und alle Assets aus dem State-Mirror. Read-only.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const stateRoot = '/Users/michaelwelsch/.local/state/ctox/business-os';
const injectedIndex = path.join(scriptDir, 'serve/index.html');
const port = Number(process.env.SNAP_SERVE_PORT || 8901);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json',
  '.wasm': 'application/wasm',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.ico': 'image/x-icon',
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/' || rel === '/business-os' || rel === '/business-os/') {
    res.writeHead(200, { 'content-type': MIME['.html'], 'cache-control': 'no-store' });
    fs.createReadStream(injectedIndex).pipe(res);
    return;
  }
  rel = rel.replace(/^\/business-os\//, '/').replace(/^\//, '');
  if (rel.split('/').some((part) => part === '..' || part.startsWith('.'))) {
    res.writeHead(403).end('forbidden');
    return;
  }
  let file = path.join(stateRoot, rel);
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404).end('not found: ' + rel);
    return;
  }
  res.writeHead(200, { 'content-type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});

server.listen(port, '127.0.0.1', () => console.log(`snap-proof shell server on http://127.0.0.1:${port}/`));
