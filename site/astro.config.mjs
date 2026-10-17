// The Wobbl website: static pages for GitHub Pages under /wobbl/.
// `npm run build` writes dist/ and then indexes it for search with Pagefind.
import { defineConfig } from 'astro/config';
import { fileURLToPath } from 'node:url';
import { syncScenes } from './scripts/scenes.mjs';
import { drawPosters } from './scripts/posters.mjs';

// Every example's scene.wobbl and the viewer that plays it, copied into public/data/scenes before each dev or build,
// with the stills of every example page's live view and a card of each scene for the gallery.
const scenes = {
  name: 'wobbl-scenes',
  hooks: {
    'astro:config:setup': async ({ config }) => {
      const site = fileURLToPath(new URL('.', import.meta.url));
      syncScenes(site);
      await drawPosters(site, config.base);
    },
  },
};

export default defineConfig({
  site: 'https://sohan-bhat.github.io',
  base: '/wobbl',
  trailingSlash: 'always',
  build: { format: 'directory', inlineStylesheets: 'never' },
  // Links load on hover, the main nav and previous and next links as soon as they are on screen.
  // Astro skips prefetching on Save-Data and 2G connections.
  prefetch: { prefetchAll: true, defaultStrategy: 'hover' },
  integrations: [scenes],
  devToolbar: { enabled: false },
  server: { port: 4321, host: '127.0.0.1' },
});
