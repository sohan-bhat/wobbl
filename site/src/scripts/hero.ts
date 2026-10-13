// The landing page's 3D part: the belt pull slider, the live reading and the three.js view.
// The reading follows the slider even without WebGL; the 3D view loads when the part is
// near the screen. initHero returns a function that tears everything down, which runs
// when the visitor leaves the page.
type Listener = (s: { scale: number; exag: number; hatch: boolean }) => void;

export function initHero(viewer: HTMLElement): () => void {
  const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
  const fmt = (x: number, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const NB = ' ';
  const loadIn = $<HTMLInputElement>('#load');
  const bend = $<HTMLButtonElement>('#bend');
  const live = $('.live');
  const listeners: Listener[] = [];
  const resetters: (() => void)[] = [];
  const part = { loadN: 2000, peak: 123.695, allow: 165, target: 2 };
  const BEND = 60; // displacement is drawn 60 times larger so the bending shows
  const WORD: Record<string, string> = { holds: 'Holds', marginal: 'Marginal', fails: 'Fails' };
  const current = () => ({
    scale: +loadIn.value / part.loadN, exag: bend.getAttribute('aria-pressed') === 'true' ? BEND : 0, hatch: true,
  });
  let stop: (() => void) | null = null;
  let gone = false;

  // The reading is sized for its widest value, at the top of the slider, so dragging moves nothing.
  const fit = () => live.style.setProperty('--readout-w', `${fmt(part.peak * +loadIn.max / part.loadN, 1).length + 4}ch`);

  function update() {
    const { scale } = current();
    const peak = part.peak * scale;
    const sf = peak > 0 ? part.allow / peak : Infinity;
    const verdict = sf >= part.target ? 'holds' : sf >= 1 ? 'marginal' : 'fails';
    $('#load-out').textContent = `${fmt(+loadIn.value)}${NB}N`;
    $('#r-peak').textContent = fmt(peak, 1);
    const word = $('#verdict-word');
    if (word.dataset.verdict !== verdict) { // a live region, so it changes only with the verdict
      word.dataset.verdict = verdict;
      word.textContent = WORD[verdict];
    }
    loadIn.setAttribute('aria-valuetext', `${fmt(+loadIn.value)} newtons, peak ${fmt(peak)} megapascals, ${WORD[verdict].toLowerCase()}`);
    $('#legend-peak').style.left = peak > part.allow ? 'calc(100% + 6px)' : `${Math.min(peak / part.allow, 1) * 100}%`;
    for (const fn of listeners) fn(current());
  }

  // The load lives in the query string so a reading can be shared, for example ?load=3000
  const v = new URLSearchParams(location.search).get('load');
  if (v !== null && v !== '' && isFinite(+v)) loadIn.value = String(Math.min(+loadIn.max, Math.max(+loadIn.min, +v)));
  const onInput = () => update();
  const onChange = () => {
    const u = new URL(location.href);
    u.searchParams.set('load', loadIn.value);
    history.replaceState(history.state, '', u);
  };
  const onBend = () => {
    bend.setAttribute('aria-pressed', String(bend.getAttribute('aria-pressed') !== 'true'));
    update();
  };
  loadIn.addEventListener('input', onInput);
  loadIn.addEventListener('change', onChange);
  bend.addEventListener('click', onBend);

  // Numbers come from the exported result when it is reachable.
  fetch(`${viewer.dataset.part}.json?v=${viewer.dataset.v || 1}`).then((r) => r.json()).then((m) => {
    const d = m.variants.distributed;
    Object.assign(part, { loadN: m.load_N, peak: d.governing_MPa, allow: m.allowable_MPa, target: m.target_sf });
    fit();
    update();
  }).catch(() => {});
  fit();
  update();

  const status = viewer.querySelector<HTMLElement>('.viewer-status')!;
  const fallback = (why: string) => {
    viewer.dataset.state = 'static';
    status.textContent = why;
    bend.hidden = true;
  };
  const gl = (() => {
    try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
  })();
  let io: IntersectionObserver | null = null;
  if (!gl || !('DecompressionStream' in window)) {
    fallback('This browser cannot draw the 3D part, so this is the static render Wobbl wrote.');
  } else {
    const ui = { onChange: (fn: Listener) => listeners.push(fn), onReset: (fn: () => void) => resetters.push(fn), current };
    const go = () => import('./viewer.js').then((m) => m.start(viewer, ui)).then((dispose: () => void) => {
      if (gone) dispose(); else stop = dispose;
    }).catch((err) => {
      console.error(err);
      fallback('The 3D part did not load, so this is the static render Wobbl wrote. Reload the page to try again.');
    });
    io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io?.disconnect(); go(); }
    }, { rootMargin: '400px 0px' });
    io.observe(viewer);
  }

  return () => {
    gone = true;
    io?.disconnect();
    loadIn.removeEventListener('input', onInput);
    loadIn.removeEventListener('change', onChange);
    bend.removeEventListener('click', onBend);
    stop?.();
  };
}
