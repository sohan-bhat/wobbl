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

// The cast mount: readouts follow the load slider even without WebGL.
const viewer = $('#viewer');
if (viewer) {
  const fmt = (x, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const NB = '\u00a0';
  const loadIn = $('#load'), exagIn = $('#exag'), hatchIn = $('#hatch');
  const listeners = [], resetters = [];
  const part = {
    loadN: 2000, peak: 123.695, allow: 165, target: 2, defl: 0.2649,
  };
  const current = () => ({ scale: +loadIn.value / part.loadN, exag: +exagIn.value, hatch: hatchIn.checked });

  function update() {
    const { scale, exag } = current();
    const peak = part.peak * scale;
    const sf = peak > 0 ? part.allow / peak : Infinity;
    const verdict = sf >= part.target ? 'holds' : sf >= 1 ? 'marginal' : 'fails';
    $('#load-out').textContent = `${fmt(+loadIn.value)}${NB}N`;
    $('#exag-out').textContent = exag === 0 ? 'true shape' : `${exag}x`;
    $('#r-peak').textContent = fmt(peak, 1);
    $('#r-sf').textContent = !isFinite(sf) ? 'no load' : sf > 99 ? 'above 99' : fmt(sf, 2);
    $('#r-defl').textContent = fmt(part.defl * scale, 3);
    const box = $('.verdict');
    if (box.dataset.verdict !== verdict || !update.done) {
      // the word is a live region: only touch it when the verdict changes
      box.dataset.verdict = verdict;
      $('#verdict-word').textContent = { holds: 'Holds', marginal: 'Marginal', fails: 'Fails' }[verdict];
      update.done = true;
    }
    loadIn.setAttribute('aria-valuetext', `${fmt(+loadIn.value)} newtons, ${box.dataset.verdict}`);
    $('#verdict-why').textContent = {
      holds: !isFinite(sf) ? 'No load, no stress.' : `Meets the safety factor ${part.target} rule for castings.`,
      marginal: `Survives the load, but the safety factor is below the target of ${part.target}.`,
      fails: `The peak passes the ${part.allow}${NB}MPa yield stress.`,
    }[verdict];
    const at = Math.min(peak / part.allow, 1) * 100;
    $('#legend-peak').style.left = peak > part.allow ? 'calc(100% + 6px)' : `${at}%`;
    for (const fn of listeners) fn(current());
  }

  function thresholds() {
    const a = part.loadN * (part.allow / part.target) / part.peak;
    const b = part.loadN * part.allow / part.peak;
    $('#load-hint').textContent = `Meets the target up to ${fmt(Math.round(a))}${NB}N. Yields past ${fmt(Math.round(b))}${NB}N.`;
  }

  // The slider state lives in the URL so a reading can be shared, e.g. ?load=3000
  const q = new URLSearchParams(location.search);
  const clampIn = (el, v) => { if (v !== null && v !== '' && isFinite(+v)) el.value = Math.min(+el.max, Math.max(+el.min, +v)); };
  clampIn(loadIn, q.get('load'));
  clampIn(exagIn, q.get('exag'));
  const saveUrl = () => {
    const u = new URL(location.href);
    u.searchParams.set('load', loadIn.value);
    u.searchParams.set('exag', exagIn.value);
    history.replaceState(null, '', u);
  };
  for (const el of [loadIn, exagIn, hatchIn]) el.addEventListener('input', update);
  for (const el of [loadIn, exagIn]) el.addEventListener('change', saveUrl);
  $('#reset-view').addEventListener('click', () => resetters.forEach((fn) => fn()));

  // Numbers come from the exported result when it is reachable.
  const base = viewer.dataset.part || 'data/cast-motor-mount';
  fetch(`${base}.json?v=${viewer.dataset.v || 1}`).then((r) => r.json()).then((m) => {
    const v = m.variants.distributed;
    Object.assign(part, { loadN: m.load_N, peak: v.governing_MPa, allow: m.allowable_MPa, target: m.target_sf, defl: v.max_disp_mm });
    thresholds();
    update();
  }).catch(() => {});
  thresholds();
  update();

  const status = $('.viewer-status', viewer);
  const gl = (() => {
    try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
  })();
  const fallback = (why) => {
    viewer.dataset.state = 'static';
    status.textContent = why;
    $('#reset-view').hidden = true;
  };
  if (!gl || !('DecompressionStream' in window)) {
    fallback('This browser cannot draw the 3D part, so this is the static render Wobbl wrote. The sliders still update the numbers.');
  } else {
    const ui = {
      onChange: (fn) => listeners.push(fn),
      onReset: (fn) => resetters.push(fn),
      current,
    };
    const go = () => import('./viewer.js?v=3').then((m) => m.start(viewer, ui)).catch((err) => {
      console.error(err);
      fallback('The 3D part did not load, so this is the static render Wobbl wrote. Reload the page to try again; the sliders still update the numbers.');
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
