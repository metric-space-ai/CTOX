import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const bridgeDir = path.dirname(fileURLToPath(import.meta.url));
const upstreamDir = path.resolve(bridgeDir, '..');

export default defineConfig({
  root: bridgeDir,
  base: './',
  resolve: {
    dedupe: ['react', 'react-dom', 'react-final-form'],
    alias: [
      {
        find: /^react-dom\/server$/,
        replacement: path.resolve(bridgeDir, 'node_modules/react-dom/server.browser.js'),
      },
      {
        find: /^react-dom\/client$/,
        replacement: path.resolve(bridgeDir, 'node_modules/react-dom/client.js'),
      },
      {
        find: /^react-dom$/,
        replacement: path.resolve(bridgeDir, 'node_modules/react-dom/index.js'),
      },
      {
        find: /^react$/,
        replacement: path.resolve(bridgeDir, 'node_modules/react/index.js'),
      },
      {
        find: /^react-final-form$/,
        replacement: path.resolve(bridgeDir, 'node_modules/react-final-form'),
      },
      {
        find: 'easy-email-editor/lib/locales.json',
        replacement: path.resolve(
        upstreamDir,
        'packages/easy-email-editor/public/locales.json',
        ),
      },
      {
        find: 'easy-email-localization',
        replacement: path.resolve(
        upstreamDir,
        'packages/easy-email-localization',
        ),
      },
      {
        find: 'easy-email-core',
        replacement: path.resolve(
        upstreamDir,
        'packages/easy-email-core/src/index.tsx',
        ),
      },
      {
        find: 'easy-email-editor',
        replacement: path.resolve(
        upstreamDir,
        'packages/easy-email-editor/src/index.tsx',
        ),
      },
      {
        find: 'easy-email-extensions',
        replacement: path.resolve(
        upstreamDir,
        'packages/easy-email-extensions/src/index.tsx',
        ),
      },
      {
        find: '@extensions',
        replacement: path.resolve(upstreamDir, 'packages/easy-email-extensions/src'),
      },
      {
        find: '@core',
        replacement: path.resolve(upstreamDir, 'packages/easy-email-core/src'),
      },
      {
        find: '@',
        replacement: path.resolve(upstreamDir, 'packages/easy-email-editor/src'),
      },
    ],
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
    'process.env': '{}',
    global: 'globalThis',
  },
  css: {
    modules: { localsConvention: 'dashes' },
    preprocessorOptions: { less: { javascriptEnabled: true } },
  },
  plugins: [react()],
  build: {
    outDir: path.resolve(bridgeDir, 'dist'),
    emptyOutDir: true,
    target: 'es2020',
    sourcemap: false,
    minify: 'esbuild',
    chunkSizeWarningLimit: 6000,
    rollupOptions: {
      input: path.resolve(bridgeDir, 'frame.html'),
    },
  },
});
