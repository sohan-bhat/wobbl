// Markdown for the docs pages, with the site's own blocks.
//
// Code fences become labeled terminal blocks with a copy button, tables scroll
// on their own, links and images get the base path, and every h2 and h3 is
// collected for the page's Contents list. Links to #fragments are refused:
// no page on this site puts a hash in the address bar.
import MarkdownIt from 'markdown-it';
import Token from 'markdown-it/lib/token.mjs';
import { url } from './site';
import { picture, publicSize } from './images';

const LABELS: Record<string, string> = {
  sh: 'Terminal', console: 'Terminal', python: 'Python', text: 'Output', toml: 'TOML', json: 'JSON',
};

export const esc = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Split a shell line into the command and a trailing # comment, ignoring # inside quotes. */
function splitComment(line: string): [string, string] {
  let quote = '';
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quote) { if (ch === quote) quote = ''; }
    else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '#' && (i === 0 || line[i - 1] === ' ' || line[i - 1] === '\t')) {
      return [line.slice(0, i).trimEnd(), line.slice(i)];
    }
  }
  return [line, ''];
}

/**
 * A terminal block: a label bar with a copy button over the code. Shell lines get
 * a $ prompt, comments are dimmed, and the button copies only the commands, so the
 * text pastes cleanly into any shell (zsh does not treat # as a comment by default).
 */
export function term(code: string, lang = 'sh', label?: string): string {
  const name = label || LABELS[lang] || lang.toUpperCase() || 'Code';
  const lines: string[] = [];
  const copy: string[] = [];
  const raw = code.replace(/\n+$/, '').split('\n');
  if (lang === 'sh' || lang === 'console') {
    let cont = false;
    for (const r of raw) {
      if (lang === 'console' && !cont && !r.startsWith('$ ')) {
        lines.push(`<span class="ln o">${esc(r) || ' '}</span>`);
        continue;
      }
      const text = lang === 'console' && !cont ? r.slice(2) : r;
      const [cmd, comment] = splitComment(text);
      if (!cmd.trim()) {
        lines.push(comment ? `<span class="ln"><span class="c">${esc(comment)}</span></span>` : '<span class="ln"> </span>');
        continue;
      }
      const prompt = cont ? '' : '<span class="p" data-pagefind-ignore>$ </span>';
      const tail = comment ? ` <span class="c">${esc(comment)}</span>` : '';
      lines.push(`<span class="ln">${prompt}${esc(cmd)}${tail}</span>`);
      copy.push(cmd);
      cont = cmd.endsWith('\\');
    }
  } else {
    for (const r of raw) lines.push(`<span class="ln">${esc(r) || ' '}</span>`);
    copy.push(...raw);
  }
  const out = lang === 'text';
  const button = out ? '' :
    `<button class="copy" type="button" data-copy="${esc(copy.join('\n')).replace(/\n/g, '&#10;')}" aria-label="Copy the ${esc(name)} block">Copy</button>`;
  return `<div class="term${out ? ' term-out' : ''}" data-copy-block><div class="term-bar"><span>${esc(name)}</span>${button}</div>`
    + `<pre translate="no"${out ? ' tabindex="0"' : ''}><code>${lines.join('')}</code></pre></div>\n`;
}

/** GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens. */
export function slugify(text: string): string {
  return text.trim().toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu, '').replace(/ /g, '-');
}

/** A link on the site: /guides/x/ gets the base path, outside links stay as they are. */
export function siteHref(href: string, where: string): string {
  if (/^[a-z]+:/i.test(href)) return href;
  if (href.startsWith('#') || href.includes('#')) {
    throw new Error(`${where}: links with a #fragment are not allowed (${href}); link to a page instead`);
  }
  if (!href.startsWith('/')) throw new Error(`${where}: use a site path such as /guides/inputs/loads/ (${href})`);
  return url(href);
}

// A table cell that is a number with an optional unit: set in mono and kept on one line
const NUMBER = /^-?[\d.,]+(e-?\d+)?\s?(%|[A-Za-z/^\d]{1,6})?$/;

export interface Heading { depth: number; slug: string; text: string }

const md = new MarkdownIt({ html: true, typographer: true });
md.disable('replacements'); // keeps -- as it is: no dashes on this site
md.enable(['table', 'smartquotes']);

const inlineText = (t: Token | undefined) =>
  (t?.children ?? []).filter((c) => c.type === 'text' || c.type === 'code_inline').map((c) => c.content).join('');

md.renderer.rules.fence = (tokens, idx) => {
  const [lang = '', ...rest] = tokens[idx].info.trim().split(/\s+/);
  return term(tokens[idx].content, lang, rest.join(' ') || undefined);
};
md.renderer.rules.code_inline = (tokens, idx) => `<code translate="no">${esc(tokens[idx].content)}</code>`;
md.renderer.rules.table_open = (tokens, idx) =>
  `<div class="table-scroll" role="region" aria-label="${esc(tokens[idx].meta?.label ?? 'Table')}" tabindex="0"><table class="data">\n`;
md.renderer.rules.table_close = () => '</table></div>\n';

/** Render a Markdown body. `where` names the file in error messages. */
export function render(body: string, where: string): { html: string; headings: Heading[] } {
  const tokens = md.parse(body, {});
  const headings: Heading[] = [];
  const used = new Set<string>();
  let lastHeading = 'Table';
  const out: Token[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (tok.type === 'heading_open') {
      const text = inlineText(tokens[i + 1]);
      let slug = slugify(text) || 'section';
      for (let n = 1; used.has(slug); n++) slug = `${slugify(text)}-${n}`;
      used.add(slug);
      tok.attrSet('id', slug);
      tok.attrSet('tabindex', '-1');
      lastHeading = text;
      const depth = Number(tok.tag.slice(1));
      if (depth === 2 || depth === 3) headings.push({ depth, slug, text });
    } else if (tok.type === 'table_open') {
      tok.meta = { label: lastHeading };
    } else if (tok.type === 'td_open' && NUMBER.test(inlineText(tokens[i + 1]).trim())) {
      tok.attrSet('class', 'num');
    } else if (tok.type === 'inline') {
      const kids = tok.children ?? [];
      // a paragraph that is only an image becomes a figure, with its title as the caption
      if (tokens[i - 1]?.type === 'paragraph_open' && kids.length === 1 && kids[0].type === 'image') {
        const img = kids[0];
        const src = img.attrGet('src') ?? '';
        const size = publicSize(src);
        const dims = size ? ` width="${size[0]}" height="${size[1]}"` : '';
        // phones get the smaller copy of a site image when there is one (name-800.webp and so on)
        const name = src.match(/^\/img\/([\w-]+)\.webp$/)?.[1];
        const set = name ? picture(name) : null;
        const responsive = set && set.srcset.includes(',') ? ` srcset="${set.srcset}" sizes="(min-width: 1000px) 720px, 100vw"` : '';
        const cap = img.attrGet('title');
        const html = new Token('html_block', '', 0);
        html.content = `<figure><img src="${esc(siteHref(src, where))}"${responsive}${dims} loading="lazy" decoding="async" alt="${esc(inlineText(img) || img.content)}">`
          + (cap ? `<figcaption>${esc(cap)}</figcaption>` : '') + '</figure>\n';
        out.pop();
        out.push(html);
        i += 1; // skip paragraph_close
        continue;
      }
      for (const c of kids) {
        if (c.type !== 'link_open') continue;
        const href = siteHref(c.attrGet('href') ?? '', where);
        c.attrSet('href', href);
        // a file on the site, such as a report.html, loads as itself and is not prefetched
        if (href.startsWith('/') && /\.[a-z0-9]+$/i.test(href)) {
          c.attrSet('data-astro-reload', '');
          c.attrSet('data-astro-prefetch', 'false');
        }
      }
    } else if (tok.type === 'html_block') {
      tok.content = tok.content.replace(/\b(src|href)="(\/[^"]*)"/g, (_, a, v) => {
        const size = a === 'src' ? publicSize(v) : null;
        return `${a}="${url(v)}"` + (size ? ` width="${size[0]}" height="${size[1]}" loading="lazy" decoding="async"` : '');
      });
    }
    out.push(tok);
  }
  return { html: md.renderer.render(sections(out), md.options, {}), headings };
}

/**
 * On a long page, wrap each h2 section after the second in <section class="cv">, which the
 * stylesheet gives content-visibility: auto, so sections far below the screen cost nothing
 * until they get close. Short pages are left as they are.
 */
function sections(tokens: Token[]): Token[] {
  const starts = tokens.map((t, i) => (t.type === 'heading_open' && t.tag === 'h2' ? i : -1)).filter((i) => i >= 0);
  if (starts.length < 4) return tokens;
  const html = (content: string) => Object.assign(new Token('html_block', '', 0), { content });
  const out: Token[] = [];
  let open = false;
  tokens.forEach((t, i) => {
    const k = starts.indexOf(i);
    if (k >= 2) {
      if (open) out.push(html('</section>\n'));
      out.push(html('<section class="cv">\n'));
      open = true;
    }
    out.push(t);
  });
  if (open) out.push(html('</section>\n'));
  return out;
}
