// Settings shared by every page: links under the base path, the repository and the version.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/** The repository root. Astro runs with site/ as its working directory. */
export const ROOT = resolve(process.cwd(), '..');
export const REPO = 'https://github.com/sohan-bhat/wobbl';
export const SITE = 'https://sohan-bhat.github.io';

const BASE = import.meta.env.BASE_URL.endsWith('/') ? import.meta.env.BASE_URL : `${import.meta.env.BASE_URL}/`;

/** A link to a page or file on the site, under the base path. */
export function url(path = ''): string {
  return BASE + path.replace(/^\//, '');
}

/** The public address of a page, for canonical and Open Graph tags. */
export function fullUrl(path = ''): string {
  return SITE + url(path);
}

export function repoUrl(rel: string, folder = false): string {
  return `${REPO}/${folder ? 'tree' : 'blob'}/main/${rel}`;
}

/** The version in pyproject.toml, shown in the top bar. */
export const VERSION = (() => {
  const toml = readFileSync(resolve(ROOT, 'pyproject.toml'), 'utf-8');
  return toml.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? '';
})();
