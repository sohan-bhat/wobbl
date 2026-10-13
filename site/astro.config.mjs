// The Wobbl website: static pages for GitHub Pages under /wobbl/.
// `npm run build` writes dist/ and then indexes it for search with Pagefind.
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sohan-bhat.github.io',
  base: '/wobbl',
  trailingSlash: 'always',
  build: { format: 'directory', inlineStylesheets: 'never' },
  prefetch: { prefetchAll: false, defaultStrategy: 'hover' },
  devToolbar: { enabled: false },
  server: { port: 4321, host: '127.0.0.1' },
});
