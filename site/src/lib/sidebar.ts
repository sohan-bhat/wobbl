// The docs sidebar, built from the docs collection: every page, in groups like MUI's docs.
// A page's folder picks its group and its `order` field its place. Previous and next links,
// breadcrumbs and the phone drawer all come from this list.
import { getCollection } from 'astro:content';

export interface Item { label: string; href: string }
export interface Group { label: string; items: Item[] }

/** Groups in sidebar order, with the folder each one reads its pages from. */
const GROUPS: { label: string; prefix: string }[] = [
  { label: 'Getting started', prefix: 'get-started' },
  { label: 'Simulations', prefix: 'guides/simulations' },
  { label: 'Inputs', prefix: 'guides/inputs' },
  { label: 'Results', prefix: 'guides/results' },
  { label: 'Fixing parts', prefix: 'guides/fixing' },
  { label: 'Agents', prefix: 'guides/agents' },
  { label: 'Command line', prefix: 'guides/cli' },
  { label: 'Validation', prefix: 'validation' },
];

/** Pages that are not Markdown files but belong in a group: [group, label, href, after this route]. */
const EXTRA: [string, string, string, string][] = [
  ['get-started', 'Examples', '/examples/', '/get-started/first-run/'],
  ['get-started', 'Changelog', '/get-started/changelog/', ''],
];

let cache: Group[] | null = null;

export async function sidebar(): Promise<Group[]> {
  if (cache) return cache;
  const entries = await getCollection('docs');
  cache = GROUPS.map(({ label, prefix }) => {
    const items = entries
      .filter((e) => e.id === prefix || e.id.startsWith(`${prefix}/`))
      .sort((a, b) => (a.id === prefix ? -1 : b.id === prefix ? 1 : 0)
        || (a.data.order ?? 99) - (b.data.order ?? 99) || a.data.title.localeCompare(b.data.title))
      .map((e) => ({ label: e.data.title, href: `/${e.id}/` }));
    for (const [g, l, href, after] of EXTRA) {
      if (g !== prefix) continue;
      const i = items.findIndex((it) => it.href === after);
      items.splice(i >= 0 ? i + 1 : items.length, 0, { label: l, href });
    }
    return { label, items };
  }).filter((g) => g.items.length);
  return cache;
}

/** Where a page sits: its group, and the pages before and after it in reading order. */
export async function locate(path: string) {
  const groups = await sidebar();
  const pages = groups.flatMap((g) => g.items.map((it) => ({ ...it, group: g })));
  const i = pages.findIndex((p) => p.href === path);
  if (i < 0) return null;
  return { group: pages[i].group, item: pages[i], prev: i > 0 ? pages[i - 1] : null, next: pages[i + 1] ?? null };
}
