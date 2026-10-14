// The landing page's 3D part: the size and direction of a load on the motor shaft, the live reading,
// the 3D view and its idle animation. The reading comes from the export's peak table over every
// direction, so it follows the controls at no cost, with or without WebGL. Input is folded into one
// update a frame. The 3D view starts once the part is on screen and the browser is idle; until its
// first frame is drawn the static render stays. All state lives here, in memory: the hero never
// reads or writes the URL. initHero returns a function that tears everything down, which runs when
// the visitor leaves the page.
import { createAutoplay, type Autoplay, type Frame, type Mode, type View } from './autoplay';
import { WORD, angleWords, fmt, offCheck, reading, tablePerN, type PartData, type Peaks } from './reading';

export interface HeroState { load: number; az: number; el: number; exag: number }
type Listener = (s: HeroState) => void;
interface Viewer {
  dispose(): void;
  view: { get(): View; set(v: View): void; hot(on: boolean): void; hand(opacity: number): void };
}

export function initHero(viewer: HTMLElement): () => void {
  const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
  const man: PartData = JSON.parse($('#part-data').textContent || '{}');
  const NB = ' ';
  const loadIn = $<HTMLInputElement>('#load');
  const bend = $<HTMLButtonElement>('#bend');
  const note = $('#part-note');
  const word = $('#verdict-word');
  const said = $('#part-said');
  const out = { load: $('#load-out'), peak: $('#r-peak'), legend: $('#legend-peak') };
  const listeners = new Set<Listener>();
  const resetters = new Set<() => void>();
  const BEND = 60; // displacement is drawn 60 times larger so the bending shows
  const state: HeroState = { load: +loadIn.value, az: Math.round(man.pull.default_deg), el: 0, exag: BEND };
  const abort = new AbortController();
  let stop: (() => void) | null = null;
  let auto: Autoplay | null = null;
  let peaks: Peaks | null = null;

  // Write text only when it changes, so an update touches as little of the page as it can.
  const shown = new Map<Element, string>();
  const put = (el: Element, text: string) => { if (shown.get(el) !== text) { shown.set(el, text); el.textContent = text; } };

  function update() {
    state.load = +loadIn.value;
    state.exag = bend.getAttribute('aria-pressed') === 'true' ? BEND : 0;
    const r = reading(man, state.load, tablePerN(man, peaks, state.az, state.el));
    put(out.load, `${fmt(state.load)}${NB}N`);
    put(out.peak, fmt(r.peak, 1));
    if (word.dataset.verdict !== r.verdict) { // a live region, so it changes only with the verdict
      word.dataset.verdict = r.verdict;
      word.textContent = WORD[r.verdict];
    }
    const text = `${fmt(state.load)} newtons, peak ${fmt(r.peak)} megapascals, ${WORD[r.verdict].toLowerCase()}`;
    if (shown.get(loadIn) !== text) { shown.set(loadIn, text); loadIn.setAttribute('aria-valuetext', text); }
    out.legend.style.left = r.peak > man.allowable_MPa ? 'calc(100% + 6px)' : `${(r.peak / man.allowable_MPa) * 100}%`;
    const off = offCheck(man, state.az, state.el);
    note.hidden = !off;
    if (off) {
      const conv = (off.converged_MPa * state.load) / man.load_N;
      put(note, `Near this direction the mesh refined for the default load reads ${Math.abs(Math.round(off.difference_pct))}% ` +
        `${off.difference_pct > 0 ? 'high' : 'low'}. Wobbl’s own study of that direction converges to ${fmt(conv, 1)} MPa at this load.`);
    }
    for (const fn of listeners) fn({ ...state });
  }
  let queued = 0;
  const schedule = () => { if (!queued) queued = requestAnimationFrame(() => { queued = 0; update(); }); };

  // The arrow keys on the part say where the load points once the visitor stops pressing, not at every press.
  let sayTimer = 0;
  function announce() {
    const r = reading(man, state.load, tablePerN(man, peaks, state.az, state.el));
    said.textContent = `Load ${angleWords(state.az, state.el)}. Peak ${fmt(r.peak, 1)} megapascals, ${WORD[r.verdict].toLowerCase()}.`;
  }
  function setDir(az: number, el: number, byKey = false) {
    const a = ((Math.round(az) % 360) + 360) % 360, e = Math.max(-90, Math.min(90, Math.round(el)));
    if (a !== state.az || e !== state.el) {
      state.az = a;
      state.el = e;
      schedule();
    }
    if (byKey) { clearTimeout(sayTimer); sayTimer = window.setTimeout(announce, 700); }
  }

  // The peak table is small and fetched once; until it is in, the reading is the default direction's.
  fetch(viewer.dataset.peaks!, { signal: abort.signal }).then((r) => r.json()).then((p: Peaks) => {
    peaks = p;
    schedule();
  }).catch(() => {});

  // Only a real click, tap, drag or key on the part or the load stops the idle animation: hovering,
  // a wheel or a swipe that scrolls the page does not, and neither does the bending toggle, which only
  // changes how the part is drawn. It plays again after a quiet spell.
  const touched = () => auto?.hold();
  const onBend = () => {
    bend.setAttribute('aria-pressed', String(bend.getAttribute('aria-pressed') !== 'true'));
    schedule();
  };
  const onInput = () => { touched(); schedule(); };
  const passive = { passive: true };
  loadIn.addEventListener('input', onInput, passive);
  loadIn.addEventListener('pointerdown', touched, passive);
  loadIn.addEventListener('keydown', touched, passive);
  bend.addEventListener('click', onBend);
  update();

  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const onReduce = () => {
    if (!reduce.matches) return;
    auto?.hold();
    auto?.destroy();
    auto = null;
  };
  reduce.addEventListener('change', onReduce);
  const saveData = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData === true;
  function startAutoplay(v: Viewer) {
    if (reduce.matches || saveData) return;
    const probe = viewer.querySelector<HTMLElement>('.probe')!;
    auto = createAutoplay({
      root: viewer,
      home: v.view.get(),
      phone: matchMedia('(pointer: coarse)').matches,
      read: (): Frame => ({ az: state.az, el: state.el, load: state.load, view: v.view.get(), hand: 0 }),
      apply(f: Frame) {
        loadIn.value = String(Math.round(f.load / 10) * 10);
        state.az = f.az;
        state.el = f.el;
        v.view.set(f.view);
        v.view.hand(f.hand);
        cancelAnimationFrame(queued);
        queued = 0;
        update(); // already inside an animation frame
      },
      onMode(m: Mode) {
        // while it plays the verdict changes every few seconds, so screen readers are kept out of that
        word.setAttribute('aria-live', m === 'playing' ? 'off' : 'polite');
        // while it plays the arrow looks held, as the visitor's own drag would make it
        v.view.hot(m === 'playing');
        if (m === 'playing') probe.hidden = true;
        else { v.view.hand(0); state.az = Math.round(state.az); state.el = Math.round(state.el); }
      },
    });
    auto.play();
  }

  const status = viewer.querySelector<HTMLElement>('.viewer-status')!;
  const fallback = (why: string) => {
    viewer.dataset.state = 'static';
    status.textContent = why;
    bend.hidden = true;
  };
  // WebGL drawn in software, with no graphics card behind it, would hold the page up on every frame,
  // so it counts as no WebGL: the browser says so when asked to fail on a major performance caveat.
  // Where WebGL works on an OffscreenCanvas, the part is drawn in a worker, off the main thread.
  let inWorker = false;
  const gl = (() => {
    const o = { failIfMajorPerformanceCaveat: true };
    const release = (ctx: unknown) => (ctx as WebGLRenderingContext | null)?.getExtension('WEBGL_lose_context')?.loseContext();
    try {
      if ('transferControlToOffscreen' in HTMLCanvasElement.prototype && typeof OffscreenCanvas !== 'undefined') {
        const ctx = new OffscreenCanvas(1, 1).getContext('webgl2', o);
        if (ctx) { release(ctx); inWorker = true; return true; }
      }
      const c = document.createElement('canvas');
      const ctx = c.getContext('webgl2', o) || c.getContext('webgl', o);
      release(ctx);
      return !!ctx;
    } catch { return false; }
  })();
  let io: IntersectionObserver | null = null;
  let idleId = 0;
  if (!gl || !('DecompressionStream' in window) || !('Worker' in window)) {
    fallback('This browser cannot draw the 3D part quickly here, so this is the static render Wobbl wrote. The numbers still follow the load.');
  } else {
    const ui = {
      data: man,
      signal: abort.signal,
      state: () => ({ ...state }),
      onChange: (fn: Listener) => { listeners.add(fn); return () => { listeners.delete(fn); }; },
      onReset: (fn: () => void) => { resetters.add(fn); return () => { resetters.delete(fn); }; },
      setDir,
      touched,
      peaks: () => peaks,
      worker: inWorker,
    };
    const go = () => import('./viewer.js').then((m) => m.start(viewer, ui)).then((v: Viewer) => {
      if (abort.signal.aborted) { v.dispose(); return; }
      stop = v.dispose;
      startAutoplay(v);
    }).catch((err) => {
      if (abort.signal.aborted) return;
      console.error(err);
      fallback('The 3D part did not load, so this is the static render Wobbl wrote. Reload the page to try again.');
    });
    // only once the part is on screen, and then when the browser has a moment
    const whenIdle = (fn: () => void) => {
      idleId = 'requestIdleCallback' in window ? requestIdleCallback(fn, { timeout: 1500 }) : window.setTimeout(fn, 200);
    };
    io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io?.disconnect(); whenIdle(go); }
    });
    io.observe(viewer);
  }

  return () => {
    abort.abort();
    io?.disconnect();
    if ('cancelIdleCallback' in window) cancelIdleCallback(idleId); else clearTimeout(idleId);
    cancelAnimationFrame(queued);
    clearTimeout(sayTimer);
    auto?.destroy();
    auto = null;
    loadIn.removeEventListener('input', onInput);
    loadIn.removeEventListener('pointerdown', touched);
    loadIn.removeEventListener('keydown', touched);
    bend.removeEventListener('click', onBend);
    reduce.removeEventListener('change', onReduce);
    stop?.();
    listeners.clear();
    resetters.clear();
  };
}
