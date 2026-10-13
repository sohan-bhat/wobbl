// The landing page's 3D part: the belt pull's size and direction, the live reading and the three.js
// view. The reading comes from the peak the export found for every whole
// degree, so it follows the controls at no cost, with or without WebGL. Input is folded into one
// update a frame. The 3D view starts once the part is on screen and the browser is idle; until its
// first frame is drawn the static render stays. All state lives here, in memory: the hero never
// reads or writes the URL. initHero returns a function that tears everything down, which runs
// when the visitor leaves the page.
import { WORD, angleWords, fmt, offCheck, reading, tablePerN, wrap, type PartData } from './reading';

export interface HeroState { load: number; angle: number; exag: number }
type Listener = (s: HeroState) => void;
interface Viewer { dispose(): void }

export function initHero(viewer: HTMLElement): () => void {
  const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
  const man: PartData = JSON.parse($('#part-data').textContent || '{}');
  const NB = '\u00a0';
  const loadIn = $<HTMLInputElement>('#load');
  const bend = $<HTMLButtonElement>('#bend');
  const note = $('#part-note');
  const word = $('#verdict-word');
  const said = $('#part-said');
  const out = {
    load: $('#load-out'), peak: $('#r-peak'), target: $('#r-target'), yields: $('#r-yield'), legend: $('#legend-peak'),
  };
  const listeners = new Set<Listener>();
  const resetters = new Set<() => void>();
  const BEND = 60; // displacement is drawn 60 times larger so the bending shows
  const state: HeroState = { load: +loadIn.value, angle: wrap(man.pull.default_deg), exag: BEND };
  const abort = new AbortController();
  let stop: (() => void) | null = null;

  // Write text only when it changes, so an update touches as little of the page as it can.
  const shown = new Map<Element, string>();
  const put = (el: Element, text: string) => { if (shown.get(el) !== text) { shown.set(el, text); el.textContent = text; } };

  function update() {
    state.load = +loadIn.value;
    state.exag = bend.getAttribute('aria-pressed') === 'true' ? BEND : 0;
    const deg = wrap(state.angle);
    const r = reading(man, state.load, tablePerN(man, state.angle));
    put(out.load, `${fmt(state.load)}${NB}N at ${deg}°`);
    put(out.peak, fmt(r.peak, 1));
    if (word.dataset.verdict !== r.verdict) { // a live region, so it changes only with the verdict
      word.dataset.verdict = r.verdict;
      word.textContent = WORD[r.verdict];
    }
    put(out.target, `${fmt(r.target)}${NB}N`);
    put(out.yields, `${fmt(r.yields)}${NB}N`);
    const text = `${fmt(state.load)} newtons, peak ${fmt(r.peak)} megapascals, ${WORD[r.verdict].toLowerCase()}`;
    if (shown.get(loadIn) !== text) { shown.set(loadIn, text); loadIn.setAttribute('aria-valuetext', text); }
    out.legend.style.left = r.peak > man.allowable_MPa ? 'calc(100% + 6px)' : `${(r.peak / man.allowable_MPa) * 100}%`;
    const off = offCheck(man, deg);
    note.hidden = !off;
    if (off) {
      const conv = (off.converged_MPa * state.load) / man.load_N;
      put(note, `Near this direction the mesh refined for the default pull reads ${Math.abs(Math.round(off.difference_pct))}% ` +
        `${off.difference_pct > 0 ? 'high' : 'low'}. Wobbl’s own study of the ${off.angle_deg}° pull converges to ${fmt(conv, 1)} MPa at this load.`);
    }
    for (const fn of listeners) fn({ ...state });
  }
  let queued = 0;
  const schedule = () => { if (!queued) queued = requestAnimationFrame(() => { queued = 0; update(); }); };

  // The arrow keys on the part say where the pull points once the visitor stops pressing, not at every press.
  let sayTimer = 0;
  function announce() {
    const r = reading(man, state.load, tablePerN(man, state.angle));
    said.textContent = `Belt pull ${angleWords(state.angle)}. Peak ${fmt(r.peak, 1)} megapascals, ${WORD[r.verdict].toLowerCase()}.`;
  }
  function setAngle(deg: number, byKey = false) {
    const d = wrap(deg);
    if (d !== state.angle) {
      state.angle = d;
      schedule();
    }
    if (byKey) { clearTimeout(sayTimer); sayTimer = window.setTimeout(announce, 700); }
  }

  const onBend = () => {
    bend.setAttribute('aria-pressed', String(bend.getAttribute('aria-pressed') !== 'true'));
    schedule();
  };
  const passive = { passive: true };
  loadIn.addEventListener('input', schedule, passive);
  bend.addEventListener('click', onBend);
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
  let idleId = 0;
  if (!gl || !('DecompressionStream' in window) || !('Worker' in window)) {
    fallback('This browser cannot draw the 3D part, so this is the static render Wobbl wrote. The numbers still follow the controls.');
  } else {
    const ui = {
      data: man,
      signal: abort.signal,
      state: () => ({ ...state }),
      onChange: (fn: Listener) => { listeners.add(fn); return () => { listeners.delete(fn); }; },
      onReset: (fn: () => void) => { resetters.add(fn); return () => { resetters.delete(fn); }; },
      setAngle,
    };
    const go = () => import('./viewer.js').then((m) => m.start(viewer, ui)).then((v: Viewer) => {
      if (abort.signal.aborted) { v.dispose(); return; }
      stop = v.dispose;
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
    loadIn.removeEventListener('input', schedule);
    bend.removeEventListener('click', onBend);
    clearTimeout(sayTimer);
    stop?.();
    listeners.clear();
    resetters.clear();
  };
}
