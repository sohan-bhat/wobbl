// Page behaviors for every page. Pages change without a reload (Astro's ClientRouter),
// so anything tied to a page's elements runs on astro:page-load and cleans up on
// astro:before-swap. Document level listeners are added once.
//
// No behavior here writes a #fragment to the address bar: the skip link and the
// Contents list move focus and scroll with JS, and state lives in the query string.
const $ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => r.querySelector<T>(s);
const $$ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => [...r.querySelectorAll<T>(s)];
const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// A polite live region for short announcements such as "Copied"
const say = (() => {
  let el: HTMLElement | null = null;
  return (text: string) => {
    if (!el || !el.isConnected) {
      el = document.createElement('p');
      el.className = 'visually-hidden';
      el.setAttribute('aria-live', 'polite');
      document.body.append(el);
    }
    el.textContent = text;
  };
})();

// ---- once per visit: listeners on the document ---------------------------------

// The skip link moves focus without a #main in the address bar. It listens in the capture
// phase so it runs before the router, which would otherwise treat it as a same page link.
document.addEventListener('click', (e) => {
  if (!(e.target as Element).closest('a[data-skip]')) return;
  e.preventDefault();
  const main = $('#main');
  main?.focus();
  main?.scrollIntoView();
}, true);

document.addEventListener('click', async (e) => {
  const t = e.target as Element;

  const copy = t.closest<HTMLButtonElement>('button.copy');
  if (copy) {
    try {
      await navigator.clipboard.writeText(copy.dataset.copy ?? '');
    } catch {
      const pre = copy.closest('[data-copy-block]')?.querySelector('pre');
      if (pre) getSelection()?.selectAllChildren(pre);
      say('Selected. Press Command C or Control C to copy.');
      return;
    }
    copy.textContent = 'Copied';
    copy.dataset.done = '';
    say('Copied to the clipboard');
    setTimeout(() => { copy.textContent = 'Copy'; delete copy.dataset.done; }, 1600);
    return;
  }

  const to = t.closest<HTMLElement>('[data-scroll-to]');
  if (to) {
    const target = document.getElementById(to.dataset.scrollTo ?? '');
    if (target) {
      target.scrollIntoView({ behavior: reduceMotion() ? 'auto' : 'smooth', block: 'start' });
      target.focus({ preventScroll: true });
    }
    return;
  }

  if (t.closest('[data-search-open]')) { openSearch(); return; }
  if (t instanceof HTMLDialogElement && t.id === 'search-dialog') { t.close(); return; } // a click on the backdrop
  if (t.closest('[data-search-close]')) { $<HTMLDialogElement>('#search-dialog')?.close(); return; }
  if (t.closest('[data-docnav-open]')) { docnav(true); return; }
  if (t.closest('[data-docnav-close]')) { docnav(false); return; }
});

document.addEventListener('keydown', (e) => {
  const typing = (e.target as Element).closest('input, textarea, [contenteditable]');
  if (!typing && (e.key === '/' || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k'))) {
    e.preventDefault();
    openSearch();
  }
  const nav = $('#docnav');
  if (nav?.dataset.open !== 'true') return;
  if (e.key === 'Escape') docnav(false);
  if (e.key === 'Tab') { // keep focus inside the open drawer
    const items = $$<HTMLElement>('a, button, summary', nav).filter((el) => el.offsetParent !== null);
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});

// ---- search: Pagefind, loaded the first time the dialog opens ---------------------

let searchReady = false;
function openSearch() {
  const dialog = $<HTMLDialogElement>('#search-dialog');
  if (!dialog) return;
  dialog.showModal();
  const mount = $('#search-ui')!;
  if (!searchReady) {
    searchReady = true;
    const bundle = mount.dataset.bundle!;
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = `${bundle}pagefind-ui.css`;
    document.head.append(css);
    const js = document.createElement('script');
    js.src = `${bundle}pagefind-ui.js`;
    js.onload = () => {
      // @ts-expect-error: PagefindUI is a global from the script above
      new PagefindUI({ element: '#search-ui', baseUrl: mount.dataset.base, bundlePath: bundle, showImages: false,
        resetStyles: false, translations: { placeholder: 'Search the docs…' } });
      mount.querySelector('input')?.focus();
    };
    js.onerror = () => { mount.textContent = 'Search is built with the site. Run npm run build to try it locally.'; };
    document.head.append(js);
  } else {
    mount.querySelector('input')?.focus();
  }
}

// ---- the docs menu on phones -----------------------------------------------------

function docnav(open: boolean) {
  const nav = $('#docnav');
  if (!nav) return;
  nav.dataset.open = String(open);
  $('[data-docnav-open]')?.setAttribute('aria-expanded', String(open));
  const scrim = $('.docnav-scrim');
  if (scrim) scrim.hidden = !open;
  document.documentElement.classList.toggle('no-scroll', open);
  if (open) nav.querySelector<HTMLElement>('[aria-current="page"], a')?.focus();
  else $<HTMLElement>('[data-docnav-open]')?.focus();
}

// ---- every page --------------------------------------------------------------------

let cleanups: (() => void)[] = [];

document.addEventListener('astro:page-load', () => {
  markNav();
  stackTables();
  tabs();
  contents();
  countUp();
  heroPart();
});

document.addEventListener('astro:before-swap', () => {
  for (const fn of cleanups) fn();
  cleanups = [];
  docnav(false);
});

const BASE = import.meta.env.BASE_URL.replace(/\/?$/, '/');

/** The top bar persists across pages, so its current link is updated here. */
function markNav() {
  const rel = location.pathname.startsWith(BASE) ? location.pathname.slice(BASE.length) : '';
  for (const a of $$<HTMLAnchorElement>('.nav-links a')) {
    const own = new URL(a.href).pathname.slice(BASE.length);
    const inSection = (a.dataset.section ?? '').split(' ').some((s) => s && rel.startsWith(s));
    if (rel === own) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    a.classList.toggle('is-here', rel !== own && inSection);
  }
}

/** Tables: give each cell its column name so narrow screens can stack rows. */
function stackTables() {
  for (const wrap of $$('.table-scroll:not(.stack)')) {
    const heads = $$('thead th', wrap).map((th) => th.textContent?.trim() ?? '');
    for (const tr of $$('tbody tr', wrap)) {
      [...tr.children].forEach((cell, i) => {
        if (cell.tagName !== 'TD') return;
        (cell as HTMLElement).dataset.label = heads[i] || '';
        if (cell.childNodes.length > 1) {  // the stacked layout is a two column grid: label, then content
          const span = document.createElement('span');
          span.append(...cell.childNodes);
          cell.append(span);
        }
      });
    }
    wrap.classList.add('stack');
  }
}

/** Tabs follow the ARIA tabs pattern; the selected one is kept in the query string as ?run=. */
function tabs() {
  for (const list of $$('[role="tablist"]')) {
    const all = $$<HTMLButtonElement>('[role="tab"]', list);
    const select = (tab: HTMLButtonElement, focus = false) => {
      for (const t of all) {
        const on = t === tab;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        const panel = document.getElementById(t.getAttribute('aria-controls')!);
        if (panel) panel.hidden = !on;
      }
      if (focus) tab.focus();
    };
    const remember = (tab: HTMLButtonElement) => {
      const u = new URL(location.href);
      u.searchParams.set('run', tab.id.replace(/^tab-/, ''));
      history.replaceState(history.state, '', u);
    };
    const start = all.find((t) => t.id === `tab-${new URLSearchParams(location.search).get('run')}`);
    if (start) select(start);
    list.addEventListener('click', (e) => {
      const t = (e.target as Element).closest<HTMLButtonElement>('[role="tab"]');
      if (t) { select(t); remember(t); }
    });
    list.addEventListener('keydown', (e) => {
      const i = all.indexOf(document.activeElement as HTMLButtonElement);
      const step = ({ ArrowRight: 1, ArrowLeft: -1, Home: -i, End: all.length - 1 - i } as Record<string, number>)[e.key];
      if (i < 0 || step === undefined) return;
      e.preventDefault();
      const next = all[(i + step + all.length) % all.length];
      select(next, true);
      remember(next);
    });
  }
}

/** The Contents list highlights the section being read. */
function contents() {
  const buttons = $$<HTMLButtonElement>('.contents [data-scroll-to]');
  if (!buttons.length || !('IntersectionObserver' in window)) return;
  const targets = buttons.map((b) => document.getElementById(b.dataset.scrollTo!)).filter(Boolean) as HTMLElement[];
  const mark = (id: string) => buttons.forEach((b) => {
    if (b.dataset.scrollTo === id) b.setAttribute('aria-current', 'location'); else b.removeAttribute('aria-current');
  });
  const seen = new Map<string, boolean>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) seen.set(e.target.id, e.isIntersecting);
    const first = targets.find((t) => seen.get(t.id));
    if (first) mark(first.id);
  }, { rootMargin: '-80px 0px -60% 0px' });
  targets.forEach((t) => io.observe(t));
  mark(targets[0].id);
  cleanups.push(() => io.disconnect());
}

/** Headline numbers count up once when they come into view; instantly under reduced motion. */
function countUp() {
  const els = $$('[data-count]');
  if (!els.length || reduceMotion() || !('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      const el = e.target as HTMLElement;
      const end = Number(el.dataset.count);
      const t0 = performance.now();
      const tick = (t: number) => {
        const k = Math.min((t - t0) / 700, 1);
        el.textContent = String(Math.round(end * (1 - (1 - k) ** 3)));
        if (k < 1) requestAnimationFrame(tick);
      };
      el.textContent = '0';
      requestAnimationFrame(tick);
    }
  }, { threshold: 0.6 });
  els.forEach((el) => io.observe(el));
  cleanups.push(() => io.disconnect());
}

/** The 3D part in the landing page's hero; it is disposed when the page is left. */
function heroPart() {
  const root = $('#viewer');
  if (!root) return;
  import('./hero').then((m) => {
    if (!root.isConnected) return;
    const dispose = m.initHero(root);
    cleanups.push(dispose);
  });
}
