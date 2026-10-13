// The Wobbl website: static pages for GitHub Pages under /wobbl/.
// `npm run build` writes dist/ and then indexes it for search with Pagefind.
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sohan-bhat.github.io',
  base: '/wobbl',
  trailingSlash: 'always',
  build: { format: 'directory', inlineStylesheets: 'never' },
  // Links load on hover, the main nav and previous and next links as soon as they are on screen.
  // Astro skips prefetching on Save-Data and 2G connections.
  prefetch: { prefetchAll: true, defaultStrategy: 'hover' },
  devToolbar: { enabled: false },
  server: { port: 4321, host: '127.0.0.1' },
});
