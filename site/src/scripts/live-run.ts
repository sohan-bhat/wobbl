// Strength and heat runs on the example pages, played back from their own results. A strength run's
// load ramps from zero to the run's load and, since the models are linear elastic, every stress and
// displacement grows in step with it, so each frame is exact. A heat run plays its saved temperature
// fields on a sped up clock. A scene run's input ramps from its first value to the value its question
// asked about, once, and then the slider under the view moves it, with a probe under the mouse. The part
// loads once the figure comes near the screen and plays once it is
// on screen; it pauses off screen and in a hidden tab. With reduced motion, without WebGL or before
// the script runs, the figure keeps Wobbl's own heatmap and the final numbers. Nothing here reads or
// writes the URL. initLiveRuns returns a function that tears everything down when the page is left.
import { Orbit, corners, fitTo, orbitControls } from './orbit.js';
import { decoder, inWorker, offscreen, onPage, webgl, type Back } from './play-back';
import { num, sig } from './scene-file.js';

interface Ev { k: number; text: string }
interface Poster { src: string; srcset: string; width: number; height: number; alt: string }
interface PaneData { rel: string; name: string; file: string; run: Record<string, any>; event: Ev | null; final: string; poster: Poster | null; verdict?: string }
interface Data {
  kind: 'strength' | 'heat' | 'scene'; fix: number; unit: string; loads: string[]; steady: boolean; total: string; title: string;
  pace: { tag: string; seconds: number; speed: number; span: number };
  meter: { lo: number; top: number; marks: { at: number; text: string }[] };
  panes: PaneData[];
}

const LEAD = 0.6; // seconds the part sits unloaded, or at the start of the clock, before the run plays
const NB = ' ';

/** 1234.5 as "1,234.5", without Intl, as reading.ts does. */
function fmt(x: number, d = 0): string {
  const s = Math.abs(x).toFixed(d);
  const dot = s.indexOf('.');
  const whole = (dot < 0 ? s : s.slice(0, dot)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (x < 0 && Number(s) !== 0 ? '-' : '') + whole + (dot < 0 ? '' : s.slice(dot));
}
const clock = (s: number) => { const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s - m * 60)).padStart(2, '0')}`; };
const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);
/** y at x on a piecewise linear curve. */
function interp(xs: number[], ys: number[], x: number): number {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}

export function initLiveRuns(): () => void {
  const figs = [...document.querySelectorAll<HTMLElement>('figure[data-lrun]')];
  if (!figs.length) return () => {};
  const live = !matchMedia('(prefers-reduced-motion: reduce)').matches && 'DecompressionStream' in window && 'Worker' in window
    && 'IntersectionObserver' in window && 'ResizeObserver' in window && webgl();
  const dec = decoder();
  const worker = live && offscreen();
  const stops = figs.map((fig) => liveRun(fig, live, worker, dec));
  return () => { for (const s of stops) s(); dec.dispose(); };
}

function liveRun(fig: HTMLElement, live: boolean, worker: boolean, dec: ReturnType<typeof decoder>): () => void {
  const data: Data = JSON.parse(fig.querySelector('[data-lrun-data]')!.textContent || '{}');
  const q = <T extends Element = HTMLElement>(s: string, r: ParentNode = fig) => r.querySelector<T>(s)!;
  const qa = <T extends Element = HTMLElement>(s: string, r: ParentNode = fig) => [...r.querySelectorAll<T>(s)];
  const view = q('.lrun-view'), tag = fig.querySelector<HTMLElement>('.lrun-tag'), replay = q<HTMLButtonElement>('.lrun-replay'), status = q('.lrun-status');
  const paneEls = qa('.lrun-pane'), rowEls = qa('.lrun-row');
  const now = q('[data-now]'), of = q('[data-of]'), said = q('[data-said]');
  const eventEl = fig.querySelector<HTMLElement>('[data-event]');
  const fixBtns = qa<HTMLButtonElement>('[data-fix]');
  const strength = data.kind === 'strength';
  const sceneRun = data.kind === 'scene';
  const slider = fig.querySelector<HTMLInputElement>('.lrun-slider input');
  const alsoEl = fig.querySelector<HTMLElement>('[data-also]');
  const probeEl = fig.querySelector<HTMLElement>('.lrun-probe');
  const toggleBtns = qa<HTMLButtonElement>('[data-toggle]');
  const look = { bend: true, mag: true };
  const shown = data.fix > 0 ? [0, data.fix] : [0];
  const abort = new AbortController();
  const on = (el: EventTarget, type: string, fn: (e: any) => void, o?: AddEventListenerOptions) =>
    el.addEventListener(type, fn, { ...o, signal: abort.signal });

  // Write only what changed, so a frame touches as little of the page as it can.
  const was = new Map<Element, string>();
  const put = (el: Element | null, text: string) => { if (el && was.get(el) !== text) { was.set(el, text); el.textContent = text; } };
  const set = (el: HTMLElement, key: string, v: string | null) => {
    if (el.getAttribute(key) === v) return;
    if (v === null) el.removeAttribute(key); else el.setAttribute(key, v);
  };

  const frac = (v: number) => clamp01((v - data.meter.lo) / (data.meter.top - data.meter.lo));
  const level = (p: PaneData, v: number) => {
    const r = p.run;
    if (sceneRun) {
      const q = r.q;
      if (!q || q.limit === null || !Number.isFinite(v)) return 'info';
      return v > q.limit ? 'fails' : q.target !== null && v > q.target ? 'marginal' : 'holds';
    }
    if (strength) return v >= r.allowable_MPa ? 'fails' : v >= r.allowable_MPa / r.target_sf ? 'marginal' : 'holds';
    return v > r.limit_C ? 'fails' : v >= r.limit_C - 5 ? 'marginal' : 'holds';
  };
  const verdictColor: Record<string, string> = { holds: 'holds', ok: 'holds', marginal: 'marginal', warm: 'marginal', fails: 'fails', hot: 'fails' };

  // ---- a scene: where along its input the view stands --------------------------------------
  // pos is a frame position, 0 to the last frame, with the question's own value at home
  const sc = sceneRun ? data.panes[0].run : null;
  const F = sc ? sc.input.t.length : 1;
  const home = sc ? sc.input.home : 0;
  let pos = home, free = false;
  const frameAt = (p: number) => {
    const a = Math.min(Math.max(Math.floor(p), 0), F - 1), b = Math.min(a + 1, F - 1);
    return { a, b, k: sc.input.between === 'step' ? 0 : Math.min(Math.max(p - a, 0), 1) };
  };
  const along = (vs: (number | null)[], p: number) => {
    const { a, b, k } = frameAt(p), x = vs[a] ?? NaN, y = vs[b] ?? NaN;
    return k ? x + (y - x) * k : x;
  };
  const withUnit = (v: number, unit: string, d = 3) => `${sig(v, d)}${unit ? NB + unit : ''}`;
  const verdictWord: Record<string, string> = { holds: 'HOLDS', marginal: 'MARGINAL', fails: 'FAILS' };

  /** The scene's readings at pos: the input, the meter's reading and its verdict, the other readings and the event. */
  function sceneReadings(end: boolean) {
    const inp = sc.input, p0 = data.panes[0], rq = sc.q;
    const atHome = Math.abs(pos - home) < 1e-6;
    put(now, `${num(along(inp.t, pos))}${inp.unit ? NB + inp.unit : ''}`);
    of.hidden = free || end;
    const el = rowEls[0], v = rq ? along(rq.values, pos) : NaN;
    if (rq) {
      put(q('[data-n]', el), sig(v, rq.digits));
      const fill = q('[data-fill]', el);
      fill.style.setProperty('--f', frac(v).toFixed(4));
    }
    const w = q('[data-word]', el);
    // the run's own verdict at the value it was asked about, else the reading against its limit
    const lv = atHome && (end || free) ? verdictColor[p0.verdict ?? ''] ?? 'marginal' : level(p0, v);
    if (rq) set(q('[data-fill]', el), 'data-level', lv === 'info' ? 'holds' : lv);
    if ((end || free) && (atHome || lv !== 'info')) {
      set(w, 'data-verdict', lv);
      put(w, atHome ? (p0.verdict ?? '').toUpperCase() : verdictWord[lv] ?? '');
      if (w.hasAttribute('data-hide')) { w.removeAttribute('data-hide'); if (end && !free) w.setAttribute('data-land', ''); }
    } else {
      set(w, 'data-hide', '');
      w.removeAttribute('data-land');
    }
    if (alsoEl) put(alsoEl, sc.also.map((x: any, i: number) => `${i ? x.name : x.name[0].toUpperCase() + x.name.slice(1)} ${withUnit(along(x.values, pos), x.unit, x.digits)}`).join(', '));
    if (eventEl && p0.event) eventEl.hidden = pos < p0.event.k - 1e-9;
    if (slider) {
      if (!slider.matches(':active')) slider.value = String(pos);
      slider.setAttribute('aria-valuetext', `${now.textContent}${rq ? `, ${rq.name} ${sig(v, rq.digits)}${rq.unit ? ` ${rq.unit}` : ''}` : ''}`);
    }
  }

  // ---- the readings, at playback progress u (0 to 1) -------------------------------------
  const simSeconds = (u: number) => u * data.pace.span;
  /** The live value of a pane: the peak stress at this share of the load, or the hottest point now. */
  function valueAt(p: PaneData, u: number): number {
    const r = p.run;
    if (strength) return u * r.governing_MPa;
    if (r.transient) return interp(r.times_s, r.hottest_C, simSeconds(u));
    return u >= 1 ? r.t_max_C : r.air_C + interp(r.warmup.t_s, r.warmup.theta, simSeconds(u)) * (r.t_max_C - r.air_C);
  }
  function readings(u: number) {
    const end = u >= 1;
    if (sceneRun) {
      if (!free) pos = home * u;
      sceneReadings(end);
      return;
    }
    if (strength) {
      if (data.loads.length === 1) {
        const [v, unit] = data.loads[0].split(' ');
        const x = Number(v) * u;
        put(now, end ? data.total : `${fmt(x, Number(v) < 100 ? 1 : 0)}${NB}${unit}`);
      } else {
        put(now, end ? data.total : `${fmt(u * 100)}%`);
      }
      of.hidden = end;
    } else {
      put(now, end ? data.total : clock(simSeconds(u)));
      of.hidden = end || data.steady;
    }
    shown.forEach((pi, row) => {
      const p = data.panes[pi], el = rowEls[row], v = valueAt(p, u);
      put(q('[data-n]', el), end ? p.final : fmt(v, 1));
      const fill = q('[data-fill]', el);
      fill.style.setProperty('--f', frac(v).toFixed(4));
      set(fill, 'data-level', end ? verdictColor[p.run.verdict] ?? 'marginal' : level(p, v));
      const w = q('[data-word]', el);
      if (end && w.hasAttribute('data-hide')) {
        w.removeAttribute('data-hide');
        w.setAttribute('data-land', '');
      } else if (!end) {
        set(w, 'data-hide', '');
        w.removeAttribute('data-land');
      }
    });
    if (eventEl) {
      const ev = shown.map((pi) => data.panes[pi].event).find((e) => e);
      const t = strength ? u : simSeconds(u);
      eventEl.hidden = !ev || (!end && t < ev.k);
      if (ev) put(eventEl, ev.text);
    }
  }
  /** The same readings and words as the page was built with, for the final state. */
  function finalState() {
    readings(1);
    for (const el of rowEls) q('[data-word]', el).removeAttribute('data-land');
  }

  /** What a screen reader hears once a run has played. */
  function sayEnd() {
    if (sceneRun) {
      const p0 = data.panes[0], rq = sc.q;
      said.textContent = `At ${now.textContent}${rq ? `, ${rq.name} ${p0.final}${rq.unit ? ` ${rq.unit}` : ''}` : ''}, ${p0.verdict ?? ''}.`;
      return;
    }
    const parts = shown.map((pi) => {
      const p = data.panes[pi], who = shown.length > 1 ? `${p.name}, ` : '';
      return strength ? `${who}peak ${p.final} megapascals, ${p.run.verdict}.` : `${who}hottest point ${p.final} C, ${p.run.verdict}.`;
    });
    said.textContent = parts.join(' ');
  }

  // ---- which fix plays beside the part -----------------------------------------------------
  function showPane(row: number, pi: number) {
    shown[row] = pi;
    const p = data.panes[pi], cell = paneEls[row];
    for (const el of qa('[data-name]', cell).concat(qa('[data-name]', rowEls[row]))) put(el, p.name);
    const img = cell.querySelector<HTMLImageElement>('.lrun-poster');
    if (img && p.poster) {
      img.srcset = p.poster.srcset;
      img.src = p.poster.src;
      img.alt = p.poster.alt;
    }
    set(q('[data-word]', rowEls[row]), 'data-verdict', verdictColor[p.run.verdict] ?? 'marginal');
    put(q('[data-word]', rowEls[row]), p.run.verdict.toUpperCase());
  }

  // ---- the 3D view ------------------------------------------------------------------------------
  // back draws the parts; ready once every shown part has come in
  let back: Back | null = null, ready = false, canvas: HTMLCanvasElement | null = null, orbit: Orbit | null = null;
  let t = 0, playing = false, started = false, done = false, visible = false, last = 0, raf = 0, pending = 0;
  let w = 1, h = 1, px = 1;
  const duration = data.pace.seconds;
  const u = () => clamp01((t - LEAD) / duration);
  const lead = data.panes[0].run;
  const lo = sceneRun ? lead.bounds.lo : lead.mesh.bounds_min, hi = sceneRun ? lead.bounds.hi : lead.mesh.bounds_max;
  const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  const top = strength ? Math.max(...data.panes.map((p) => p.run.governing_MPa)) : 0;

  /** The rectangle of each part in CSS pixels: side by side when the view is wider than tall. */
  const views = () => (shown.length === 1 ? [[0, 0, w, h]]
    : w >= h ? [[0, 0, w / 2, h], [w / 2, 0, w / 2, h]] : [[0, 0, w, h / 2], [0, h / 2, w, h / 2]]);
  // the points the view has to hold: the corners of the part's box, and the arrow's ends at full load
  const box = corners(lo, hi);
  if (strength && lead.arrow) {
    const a = lead.arrow, len = 0.42 * radius * (a.pull ? 1 : -1);
    box.push(a.point, a.point.map((x: number, i: number) => x + len * a.dir[i]));
  }
  /** Back the camera off until every one of those points is inside the view with a margin. */
  function fitCamera() {
    const [, , vw, vh] = views()[0];
    orbit!.aspect = vw / vh;
    fitTo(orbit!, box, radius * 1.2);
  }
  function homeCamera() {
    const v = lead.camera.view;
    orbit!.az = Math.atan2(v[1], v[0]);
    orbit!.el = Math.asin(Math.max(-1, Math.min(1, v[2])));
    fitCamera();
  }

  function draw() {
    pending = 0;
    if (!ready || !back || !orbit) return;
    const k = u();
    const f: any = {
      camera: { position: orbit.eye, target: orbit.target, fov: orbit.fov, aspect: orbit.aspect },
      views: views(), k, radius,
    };
    if (sceneRun) {
      const { a, b, k: kk } = frameAt(pos), T = lead.input.t;
      f.scene = { a, b, k: kk, t: kk ? T[a] + (T[b] - T[a]) * kk : T[a], bend: look.bend, mag: look.mag };
      f.k = 1;
    } else if (!strength) {
      if (lead.transient) {
        const s = simSeconds(k), ts: number[] = lead.times_s;
        let i = 0;
        while (i < ts.length - 2 && ts[i + 1] <= s) i++;
        f.heat = { i0: i, i1: i + 1, mix: clamp01((s - ts[i]) / (ts[i + 1] - ts[i])) };
        f.k = 1;
      } else {
        f.heat = { i0: 0, i1: 0, mix: 0 };
        f.k = k >= 1 ? 1 : interp(lead.warmup.t_s, lead.warmup.theta, simSeconds(k));
      }
    }
    back.frame(f);
    markers(k);
  }
  const requestDraw = () => { if (!pending && ready) pending = requestAnimationFrame(draw); };

  /** The ring on each part's peak (or the hottest point, where it is on the surface), moved with the bending. */
  function markers(k: number) {
    views().forEach(([, , vw, vh], row) => {
      const p = data.panes[shown[row]], r = p.run, mark = q('.lrun-mark', paneEls[row]);
      let at: number[] | null = null;
      if (sceneRun) {
        const pk = r.peaks;
        if (pk) {
          const { a, b, k: kk } = frameAt(pos), A = pk[a], B = pk[b] ?? A;
          if (A && B) {
            const key = look.bend ? 'pe' : 'p0';
            at = A[key].map((x: number, i: number) => x + (B[key][i] - x) * kk);
            const mv = mark.querySelector('[data-mark-v]');
            if (mv) put(mv, withUnit(A.v + (B.v - A.v) * kk, r.q && r.q.field ? r.q.unit : (lead.q?.unit ?? ''), r.q?.digits ?? 3));
          }
        }
      } else if (strength) {
        const f = r.exag * k;
        at = r.governing_xyz.map((x: number, i: number) => x + f * r.governing_disp[i]);
      } else if (r.marker) {
        at = r.hot_xyz;
      }
      if (!at || !orbit) { mark.hidden = true; return; }
      const [x, y, z] = orbit.project(at);
      const sx = ((x + 1) / 2) * vw, sy = ((1 - y) / 2) * vh;
      mark.hidden = !(z < 1 && sx > 6 && sy > 6 && sx < vw - 6 && sy < vh - 6);
      mark.style.translate = `${sx.toFixed(1)}px ${sy.toFixed(1)}px`;
    });
  }

  function tick(ts: number) {
    raf = 0;
    if (!playing) return;
    t += last ? Math.min((ts - last) / 1000, 0.1) : 0;
    last = ts;
    const k = u();
    readings(k);
    if (pending) { cancelAnimationFrame(pending); pending = 0; }
    draw();
    if (k >= 1) { finish(); return; }
    raf = requestAnimationFrame(tick);
  }
  function play() {
    if (playing || done || !ready) return;
    playing = true;
    last = 0;
    raf = requestAnimationFrame(tick);
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(raf);
    raf = 0;
  }
  function finish() {
    pause();
    done = true;
    replay.hidden = false;
    sayEnd();
  }
  function restart() {
    pause();
    t = 0;
    free = false;
    done = false;
    started = true;
    replay.hidden = true;
    said.textContent = '';
    readings(0);
    requestDraw();
    if (visible && !document.hidden) play();
  }
  const canGo = () => visible && !document.hidden;
  function maybeStart() {
    if (!ready || !canGo()) return;
    if (!started) {
      started = true;
      canvas!.style.opacity = '';
      // the poster stays under the canvas while it fades in
      setTimeout(() => { if (ready) view.dataset.state = 'live'; }, 320);
      if (tag) tag.hidden = false;
      status.textContent = '';
      restart();
    } else if (!done) {
      play();
    }
  }

  async function build() {
    status.textContent = strength ? 'Loading the 3D part' : sceneRun ? 'Loading the 3D scene' : 'Loading the run';
    w = view.clientWidth || 1;
    h = view.clientHeight || 1;
    const coarse = matchMedia('(pointer: coarse)').matches || Math.min(screen.width, screen.height) < 600;
    px = Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2);
    canvas = document.createElement('canvas');
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${sceneRun ? `3D view of ${data.title}` : `3D ${strength ? 'stress' : 'temperature'} map of the ${data.title}`}, played from the run's own results. ` +
      'Drag or press the arrow keys to turn it, and double click or press Home to put it back.');
    canvas.style.opacity = '0';
    canvas.style.transition = 'opacity 0.3s ease';
    view.prepend(canvas);
    back = worker ? inWorker(canvas, data.kind, w, h, px) : await onPage(canvas, data.kind, w, h, px, dec);
    // the fix beside the part can change while the files come in, so wait for whichever is chosen
    let ids: number[];
    do {
      ids = [...shown];
      await Promise.all(ids.map((pi, row) => back!.add(row, data.panes[pi].file, data.panes[pi].run, top)));
    } while (ids.some((v, i) => v !== shown[i]));
    if (abort.signal.aborted) return;
    ready = true;
    const target = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
    orbit = new Orbit({ target, az: 0, el: 0, dist: radius * 4, fov: sceneRun ? lead.camera.fov : 30, aspect: 1, near: radius / 50, far: radius * 40,
      minDist: radius * 1.1, maxDist: radius * 8 });
    homeCamera();
    on(canvas, 'dblclick', () => { homeCamera(); requestDraw(); });
    on(canvas, 'keydown', (e: KeyboardEvent) => {
      const turn = ({ ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] } as Record<string, number[]>)[e.key];
      if (e.key === 'Home') homeCamera();
      else if (turn) { orbit!.az += (turn[0] * Math.PI) / 18; orbit!.el += (turn[1] * Math.PI) / 18; orbit!.update(); }
      else return;
      e.preventDefault();
      requestDraw();
    });
    const off = orbitControls(canvas, orbit, { rotateSpeed: 0.8, onChange: requestDraw });
    abort.signal.addEventListener('abort', off, { once: true });
    const ro = new ResizeObserver(() => {
      const nw = view.clientWidth, nh = view.clientHeight;
      if (!nw || !nh || (nw === w && nh === h)) return;
      w = nw;
      h = nh;
      back!.size(w, h, px);
      fitCamera();
      requestDraw();
    });
    ro.observe(view);
    abort.signal.addEventListener('abort', () => ro.disconnect(), { once: true });
    if (sceneRun) sceneControls();
    draw();
    maybeStart();
  }

  // ---- a scene's slider, its toggles and the probe under the mouse ------------------------------
  function sceneControls() {
    const goTo = (p: number) => {
      // taking the slider ends the replay where it is, and the slider moves the scene from then on
      if (playing || !done) { pause(); done = true; replay.hidden = false; }
      free = true;
      pos = Math.min(Math.max(p, 0), F - 1);
      readings(1);
      requestDraw();
    };
    if (slider) {
      slider.disabled = false;
      on(slider, 'input', () => goTo(Number(slider.value)));
    }
    for (const b of toggleBtns) {
      on(b, 'click', () => {
        const k = b.dataset.toggle as 'bend' | 'mag';
        look[k] = !look[k];
        b.setAttribute('aria-pressed', String(look[k]));
        requestDraw();
      });
    }
    if (!probeEl || !canvas) return;
    let hover: PointerEvent | null = null, pr = 0, asked = 0;
    const hide = () => { hover = null; probeEl.hidden = true; };
    const read = async () => {
      pr = 0;
      if (!hover || !orbit || !back) return;
      const e = hover, r = canvas!.getBoundingClientRect();
      const nx = ((e.clientX - r.left) / r.width) * 2 - 1, ny = 1 - ((e.clientY - r.top) / r.height) * 2;
      const { origin, dir } = orbit.ray(nx, ny), id = ++asked;
      const hit = await back.probe(0, origin, dir).catch(() => null);
      if (id !== asked || !hover) return;
      if (!hit) { probeEl.hidden = true; return; }
      put(q('[data-probe-v]', probeEl), withUnit(hit.v, hit.unit, hit.digits + 1));
      put(q('[data-probe-k]', probeEl), hit.name);
      put(q('[data-probe-x]', probeEl), hit.limit > 0 ? `${Math.round((hit.v / hit.limit) * 100)}% of ${hit.limitLabel || 'the limit'}` : '');
      const x = e.clientX - r.left, y = e.clientY - r.top;
      probeEl.classList.toggle('l', x > r.width - 200);
      probeEl.style.translate = `${x.toFixed(1)}px ${y.toFixed(1)}px`;
      probeEl.hidden = false;
    };
    on(canvas, 'pointermove', (e: PointerEvent) => {
      if (e.pointerType !== 'mouse' || e.buttons) { hide(); return; }
      hover = e;
      if (!pr) pr = requestAnimationFrame(read);
    }, { passive: true });
    on(canvas, 'pointerleave', hide);
    on(canvas, 'pointerdown', hide);
    abort.signal.addEventListener('abort', () => cancelAnimationFrame(pr), { once: true });
  }

  // ---- when to load, play and pause ------------------------------------------------------------
  if (fixBtns.length) {
    for (const b of fixBtns) {
      on(b, 'click', async () => {
        const pi = Number(b.dataset.fix);
        if (pi === shown[1]) return;
        for (const o of fixBtns) o.setAttribute('aria-pressed', String(o === b));
        if (!ready) {
          showPane(1, pi);
          finalState();
          return;
        }
        try { await back!.add(1, data.panes[pi].file, data.panes[pi].run, top); } catch { return; }
        if (abort.signal.aborted || b.getAttribute('aria-pressed') !== 'true') return;
        showPane(1, pi);
        restart();
      });
    }
  }
  if (!live) return () => abort.abort();

  on(replay, 'click', () => restart());
  on(document, 'visibilitychange', () => { if (document.hidden) pause(); else maybeStart(); });
  // the part loads once the figure comes near the screen, and then when the browser has a moment
  let idle = 0;
  const near = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    near.disconnect();
    idle = 'requestIdleCallback' in window ? requestIdleCallback(go, { timeout: 1200 }) : window.setTimeout(go, 150);
  }, { rootMargin: '400px 0px' });
  const go = () => build().catch((err) => {
    if (abort.signal.aborted) return;
    console.error(err);
    canvas?.remove();
    back?.dispose();
    back = null;
    ready = false;
    view.dataset.state = 'poster';
    status.textContent = 'The 3D view did not load, so this is the heatmap Wobbl drew.';
    finalState();
  });
  near.observe(fig);
  const seen = new IntersectionObserver(([e]) => {
    visible = e.intersectionRatio >= 0.4;
    if (visible) maybeStart(); else pause();
  }, { threshold: [0, 0.4] });
  seen.observe(view);

  return () => {
    abort.abort();
    near.disconnect();
    seen.disconnect();
    pause();
    cancelAnimationFrame(pending);
    if ('cancelIdleCallback' in window) cancelIdleCallback(idle); else clearTimeout(idle);
    back?.dispose();
    back = null;
    ready = false;
    canvas?.remove();
  };
}
