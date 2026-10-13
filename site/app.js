// Small page behaviors: copy buttons, stacked tables on phones, and the
// load readouts for the cast mount. The 3D viewer loads only when its
// section comes near the screen.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// Copy buttons
const sr = document.createElement('p');
sr.className = 'visually-hidden';
sr.setAttribute('aria-live', 'polite');
document.body.append(sr);
document.addEventListener('click', async (e) => {
  const b = e.target.closest('button.copy');
  if (!b) return;
  try {
    await navigator.clipboard.writeText(b.dataset.copy);
  } catch {
    const pre = b.closest('[data-copy-block]')?.querySelector('pre');
    if (pre) getSelection().selectAllChildren(pre);
    sr.textContent = 'Selected. Press Command C or Control C to copy.';
    return;
  }
  b.textContent = 'Copied';
  b.dataset.done = '';
  sr.textContent = 'Copied to the clipboard';
  setTimeout(() => { b.textContent = 'Copy'; delete b.dataset.done; }, 1600);
});

// Tables: give each cell its column name so narrow screens can stack rows.
for (const wrap of $$('.table-scroll')) {
  const heads = $$('thead th', wrap).map((th) => th.textContent.trim());
  for (const tr of $$('tbody tr', wrap)) {
    [...tr.children].forEach((cell, i) => {
      if (cell.tagName !== 'TD') return;
      cell.dataset.label = heads[i] || '';
      // the stacked layout is a two column grid: label, then the cell's content as one item
      if (cell.childNodes.length > 1) {
        const span = document.createElement('span');
        span.append(...cell.childNodes);
        cell.append(span);
      }
    });
  }
  wrap.classList.add('stack');
}

// Tabs: arrow keys move between them, as the ARIA tabs pattern expects.
for (const list of $$('[role="tablist"]')) {
  const tabs = $$('[role="tab"]', list);
  const select = (tab, focus) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    }
    if (focus) tab.focus();
  };
  const fromUrl = new URLSearchParams(location.search).get('run');
  const start = tabs.find((t) => t.id === `tab-${fromUrl}`);
  if (start) select(start);
  const remember = (tab) => {
    const u = new URL(location.href);
    u.searchParams.set('run', tab.id.replace(/^tab-/, ''));
    history.replaceState(null, '', u);
  };
  list.addEventListener('click', (e) => { const t = e.target.closest('[role="tab"]'); if (t) { select(t); remember(t); } });
  list.addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: tabs.length - 1 - i }[e.key];
    if (i < 0 || step === undefined) return;
    e.preventDefault();
    const next = tabs[(i + step + tabs.length) % tabs.length];
    select(next, true);
    remember(next);
  });
}

// The cast mount: the readout follows the load slider even without WebGL.
const viewer = $('#viewer');
if (viewer) {
  const fmt = (x, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const NB = '\u00a0';
  const loadIn = $('#load'), bend = $('#bend');
  const listeners = [], resetters = [];
  const part = {
    loadN: 2000, peak: 123.695, allow: 165, target: 2, defl: 0.2649,
  };
  const BEND = 60;  // displacement is drawn 60 times larger so the bending shows
  const current = () => ({
    scale: +loadIn.value / part.loadN, exag: bend.getAttribute('aria-pressed') === 'true' ? BEND : 0, hatch: true,
  });
  const WORD = { holds: 'Holds', marginal: 'Marginal', fails: 'Fails' };

  // Size the reading for its widest value, at the top of the slider, so dragging never moves anything.
  const live = $('.live');
  const fitReadout = () => live.style.setProperty('--readout-w', `${fmt(part.peak * +loadIn.max / part.loadN, 1).length + 4}ch`);

  function update() {
    const { scale } = current();
    const peak = part.peak * scale;
    const sf = peak > 0 ? part.allow / peak : Infinity;
    const verdict = sf >= part.target ? 'holds' : sf >= 1 ? 'marginal' : 'fails';
    $('#load-out').textContent = `${fmt(+loadIn.value)}${NB}N`;
    $('#r-peak').textContent = fmt(peak, 1);
    const word = $('#verdict-word');
    if (word.dataset.verdict !== verdict) {  // a live region: only touch it when the verdict changes
      word.dataset.verdict = verdict;
      word.textContent = WORD[verdict];
    }
    loadIn.setAttribute('aria-valuetext', `${fmt(+loadIn.value)} newtons: peak ${fmt(peak)} megapascals, ${WORD[verdict].toLowerCase()}`);
    const at = Math.min(peak / part.allow, 1) * 100;
    $('#legend-peak').style.left = peak > part.allow ? 'calc(100% + 6px)' : `${at}%`;
    for (const fn of listeners) fn(current());
  }

  // The load lives in the URL so a reading can be shared, e.g. ?load=3000
  const q = new URLSearchParams(location.search);
  const v = q.get('load');
  if (v !== null && v !== '' && isFinite(+v)) loadIn.value = Math.min(+loadIn.max, Math.max(+loadIn.min, +v));
  loadIn.addEventListener('input', update);
  loadIn.addEventListener('change', () => {
    const u = new URL(location.href);
    u.searchParams.set('load', loadIn.value);
    history.replaceState(null, '', u);
  });
  bend.addEventListener('click', () => {
    bend.setAttribute('aria-pressed', String(bend.getAttribute('aria-pressed') !== 'true'));
    update();
  });

  // Numbers come from the exported result when it is reachable.
  const base = viewer.dataset.part || 'data/cast-motor-mount';
  fetch(`${base}.json?v=${viewer.dataset.v || 1}`).then((r) => r.json()).then((m) => {
    const v = m.variants.distributed;
    Object.assign(part, { loadN: m.load_N, peak: v.governing_MPa, allow: m.allowable_MPa, target: m.target_sf, defl: v.max_disp_mm });
    fitReadout();
    update();
  }).catch(() => {});
  fitReadout();
  update();

  const status = $('.viewer-status', viewer);
  const gl = (() => {
    try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
  })();
  const fallback = (why) => {
    viewer.dataset.state = 'static';
    status.textContent = why;
    bend.hidden = true;
  };
  if (!gl || !('DecompressionStream' in window)) {
    fallback('This browser cannot draw the 3D part, so this is the static render Wobbl wrote.');
  } else {
    const ui = {
      onChange: (fn) => listeners.push(fn),
      onReset: (fn) => resetters.push(fn),
      current,
    };
    const go = () => import('./viewer.js?v=4').then((m) => m.start(viewer, ui)).catch((err) => {
      console.error(err);
      fallback('The 3D part did not load, so this is the static render Wobbl wrote. Reload the page to try again.');
    });
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); go(); }
      }, { rootMargin: '800px 0px' });
      io.observe(viewer);
    } else {
      go();
    }
  }
}

// A link to a heading inside a folded section opens that section.
const openTarget = () => {
  const el = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  const fold = el && el.closest('details');
  if (fold && !fold.open) { fold.open = true; el.scrollIntoView(); }
};
addEventListener('hashchange', openTarget);
openTarget();
