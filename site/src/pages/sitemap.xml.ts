// sitemap.xml with every page, built from the docs collection and the static routes.
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { examples } from '../lib/examples';
import { fullUrl } from '../lib/site';

export const GET: APIRoute = async () => {
  const docs = (await getCollection('docs')).map((e) => `${e.id}/`);
  const pages = ['', 'examples/', ...examples().map((e) => `examples/${e.name}/`), 'get-started/changelog/', ...docs];
  const rows = [...new Set(pages)].sort().map((p) => `  <url><loc>${fullUrl(p)}</loc></url>`).join('\n');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${rows}\n</urlset>\n`,
    { headers: { 'Content-Type': 'application/xml' } });
};
