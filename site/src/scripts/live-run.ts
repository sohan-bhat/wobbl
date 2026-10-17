// Strength and heat runs on the example pages, played back from their own results. A strength run's
// load ramps from zero to the run's load and, since the models are linear elastic, every stress and
// displacement grows in step with it, so each frame is exact. A heat run plays its saved temperature
// fields on a sped up clock. A scene run's input ramps from its first value to the value its question
// asked about, once, and then the slider under the view moves it, with a probe under the mouse. The part
// loads once the figure comes near the screen and plays once it is
// on screen; it pauses off screen and in a hidden tab. With reduced motion, without WebGL or before
// the script runs, the figure keeps Wobbl's own heatmap and the final numbers. Nothing here reads or
// writes the URL. initLiveRuns returns a function that tears everything down when the page is left.
import { Orbit, orbitControls } from './orbit.js';
import { decoder, inWorker, offscreen, onPage, webgl, type Back } from './play-back';
import { num, sig } from './scene-file.js';
import { aimAt, followAt } from './scene-view.js';
import { clamp01, fitView, frameAt, homeView, interp, newOrbit, paneViews, runFrame, runView, sceneFrame as sceneFrameOf } from './run-view.js';

interface Ev { k: number; text: string }
/** A pane's still, the first frame of its 3D view, with the narrow screen's (narrow) and what shows where the view cannot run */
interface Poster { src: string; srcset: string; width: number; height: number; alt: string; narrow?: string; fallback?: { src: string; srcset: string } | null }
interface PaneData {
  rel: string; name: string; file: string; run: Record<string, any>; event: Ev | null; final: string; poster: Poster | null;
  /** a scene pane: the run's own verdict, the frame position it stands at once played and whether that is the value asked about */
  verdict?: string; home?: number; asked?: boolean; at?: string;
}
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

const GAP = 10; // the least room between two labels on one line
const LINE = 15; // a line of labels, as styles/live-run.css sets their height

/** Lay out one row of labels: each centered on its mark (the first starts at its mark and the slider's last ends at
 *  its own), slid back inside the row where it would stick out, and on the first line it fits beside the ones before. */
function fitTicks(box: HTMLElement) {
  const spans = [...box.children] as HTMLElement[];
  if (!spans.length || !box.offsetParent) return;
  for (const s of spans) { s.style.translate = ''; s.style.top = ''; }
  const row = (box.closest('.lrun-row, .lrun-slider') as HTMLElement | null) ?? box;
  const br = box.getBoundingClientRect(), rr = row.getBoundingClientRect();
  const lo = rr.left - br.left, hi = rr.right - br.left;
  const placed = spans.filter((s) => s.offsetWidth > 0).map((s) => {
    const r = s.getBoundingClientRect(), w = r.width;
    const x = Math.min(Math.max(r.left - br.left, lo), hi - w);
    return { s, x, w, dx: x - (r.left - br.left) };
  }).sort((a, b) => a.x - b.x);
  const ends: number[] = [];
  for (const p of placed) {
    let line = ends.findIndex((e) => p.x >= e + GAP);
    if (line < 0) { line = ends.length; ends.push(0); }
    ends[line] = p.x + p.w;
    if (Math.abs(p.dx) > 0.5) p.s.style.translate = `${p.dx.toFixed(1)}px 0`;
    if (line) p.s.style.top = `${line * LINE}px`;
  }
  if (ends.length > 1) box.dataset.lines = String(Math.min(ends.length, 3)); else delete box.dataset.lines;
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
      // a target the scene named makes a reading past it marginal; without one the run's own verdict never
      // calls a reading under its limit anything but holding
      return v > q.limit ? 'fails' : q.named && q.target !== null && v > q.target ? 'marginal' : 'holds';
    }
    if (strength) return v >= r.allowable_MPa ? 'fails' : v >= r.allowable_MPa / r.target_sf ? 'marginal' : 'holds';
    return v > r.limit_C ? 'fails' : v >= r.limit_C - 5 ? 'marginal' : 'holds';
  };
  const verdictColor: Record<string, string> = { holds: 'holds', ok: 'holds', marginal: 'marginal', warm: 'marginal', fails: 'fails', hot: 'fails' };

  // ---- a scene: where along its input each pane stands --------------------------------------
  // pos[row] is a frame position, 0 to the last frame, with the pane's own value at its home
  const homeOf = (pi: number) => data.panes[pi].home ?? data.panes[pi].run.input.home;
  const pos: number[] = shown.map((pi) => (sceneRun ? homeOf(pi) : 0));
  let free = false;
  // the frame position of an input value, read between the saved frames around it
  const posOf = (ts: number[], v: number) => {
    const up = ts[ts.length - 1] >= ts[0];
    for (let i = 1; i < ts.length; i++) {
      if (up ? v <= ts[i] : v >= ts[i]) return i - 1 + (ts[i] === ts[i - 1] ? 0 : Math.min(Math.max((v - ts[i - 1]) / (ts[i] - ts[i - 1]), 0), 1));
    }
    return ts.length - 1;
  };
  const along = (run: any, vs: (number | null)[], p: number) => {
    const { a, b, k } = frameAt(run, p), x = vs[a] ?? NaN, y = vs[b] ?? NaN;
    return k ? x + (y - x) * k : x;
  };
  const withUnit = (v: number, unit: string, d = 3) => `${sig(v, d)}${unit ? NB + unit : ''}`;
  const verdictWord: Record<string, string> = { holds: 'HOLDS', marginal: 'MARGINAL', fails: 'FAILS' };

  /** The scene's readings: the input, each pane's reading and verdict, the other readings and the event. */
  function sceneReadings(end: boolean) {
    const r0 = data.panes[shown[0]].run, inp = r0.input;
    put(now, `${num(along(r0, inp.t, pos[0]))}${inp.unit ? NB + inp.unit : ''}`);
    of.hidden = free || end;
    shown.forEach((pi, row) => {
      const p = data.panes[pi], r = p.run, rq = r.q, el = rowEls[row], v = rq ? along(r, rq.values, pos[row]) : NaN;
      if (rq) {
        put(q('[data-n]', el), sig(v, rq.digits));
        q('[data-fill]', el).style.setProperty('--f', frac(v).toFixed(4));
      }
      const w = q('[data-word]', el);
      // the run's own verdict at the value it was asked about, else the reading against its limit
      const atHome = p.asked !== false && Math.abs(pos[row] - homeOf(pi)) < 1e-6;
      const lv = atHome && (end || free) ? verdictColor[p.verdict ?? ''] ?? 'marginal' : level(p, v);
      if (rq) set(q('[data-fill]', el), 'data-level', lv === 'info' ? 'holds' : lv);
      if ((end || free) && (atHome || lv !== 'info')) {
        set(w, 'data-verdict', lv);
        // in the run's own words: a run that reads ok away from its limit says OK, not HOLDS
        put(w, atHome ? (p.verdict ?? '').toUpperCase() : lv === 'holds' && p.verdict === 'ok' ? 'OK' : verdictWord[lv] ?? '');
        if (w.hasAttribute('data-hide')) { w.removeAttribute('data-hide'); if (end && !free) w.setAttribute('data-land', ''); }
      } else {
        set(w, 'data-hide', '');
        w.removeAttribute('data-land');
      }
    });
    if (alsoEl) put(alsoEl, r0.also.map((x: any, i: number) => `${i ? x.name : x.name[0].toUpperCase() + x.name.slice(1)} ${withUnit(along(r0, x.values, pos[0]), x.unit, x.digits)}`).join(', '));
    if (eventEl) {
      const row = shown.findIndex((pi) => data.panes[pi].event);
      const ev = row >= 0 ? data.panes[shown[row]].event! : null;
      eventEl.hidden = !ev || pos[row] < ev.k - 1e-9;
      if (ev) put(eventEl, ev.text);
    }
    if (slider) {
      if (!slider.matches(':active')) slider.value = String(along(r0, inp.t, pos[0]));
      const rq = r0.q, v = rq ? along(r0, rq.values, pos[0]) : NaN;
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
      if (!free) shown.forEach((pi, row) => { pos[row] = homeOf(pi) * u; });
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
      said.textContent = shown.map((pi) => {
        const p = data.panes[pi], rq = p.run.q, who = shown.length > 1 ? `${p.name}, ` : `At ${now.textContent}, `;
        return `${who}${rq ? `${rq.name} ${p.final}${rq.unit ? ` ${rq.unit}` : ''}, ` : ''}${p.verdict ?? ''}.`;
      }).join(' ');
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
    const at = rowEls[row].querySelector('[data-at]');
    if (at && p.at) put(at, p.at);
    // a scene pane reads in its own unit, which can differ from the one beside it
    if (sceneRun) put(q('[data-u]', rowEls[row]), p.run.q?.unit ? NB + p.run.q.unit : '');
    showPoster(cell, p);
    set(q('[data-word]', rowEls[row]), 'data-verdict', verdictColor[p.run.verdict] ?? 'marginal');
    put(q('[data-word]', rowEls[row]), p.run.verdict.toUpperCase());
  }

  // a pane's still, or once the figure has fallen back, what the run ends on: Wobbl's own heatmap or a scene's card
  let fellBack = false;
  function showPoster(cell: HTMLElement, p: PaneData) {
    const img = cell.querySelector<HTMLImageElement>('.lrun-poster'), source = img?.parentElement?.querySelector('source');
    const pic = p.poster && (fellBack && p.poster.fallback ? { ...p.poster, ...p.poster.fallback, narrow: undefined } : p.poster);
    if (!img || !pic) return;
    if (source) source.srcset = pic.narrow ?? (pic.srcset || pic.src);
    img.srcset = pic.srcset;
    img.src = pic.src;
    img.alt = pic.alt;
  }
  /** Where the 3D view cannot run, the figure shows what each run ends on, as its final numbers do. */
  function fallBack() {
    fellBack = true;
    shown.forEach((pi, row) => showPoster(paneEls[row], data.panes[pi]));
  }

  // ---- the 3D view ------------------------------------------------------------------------------
  // back draws the parts; ready once every shown part has come in
  let back: Back | null = null, ready = false, canvas: HTMLCanvasElement | null = null, orbit: Orbit | null = null;
  let t = 0, playing = false, started = false, done = false, visible = false, last = 0, raf = 0, pending = 0;
  let w = 1, h = 1, px = 1;
  const duration = data.pace.seconds;
  const u = () => clamp01((t - LEAD) / duration);
  const lead = data.panes[0].run;
  // what the view holds and where its camera opens, as the stills drawn at build time open (run-view.js)
  const rv = runView(data.kind, lead), radius: number = rv.radius;
  const top = strength ? Math.max(...data.panes.map((p) => p.run.governing_MPa)) : 0;

  /** The rectangle of each part in CSS pixels: side by side when the view is wider than tall. */
  const views = () => paneViews(shown.length, w, h);
  /** Back the camera off until everything the view holds is inside it with a margin. */
  function fitCamera() {
    const [, , vw, vh] = views()[0];
    fitView(orbit!, rv, vw, vh, pos[0]);
  }
  function homeCamera() {
    const [, , vw, vh] = views()[0];
    homeView(orbit!, rv, vw, vh, pos[0]);
  }
  /** Each pane's camera, the inset beside a view that follows its subject, and the scene state each is drawn at. */
  function sceneFrame(f: any) {
    sceneFrameOf(f, orbit, rv, shown.map((pi) => data.panes[pi].run), pos, views(), look, insets);
  }
  const insets: any[] = [];

  function draw() {
    pending = 0;
    if (!ready || !back || !orbit) return;
    const k = u();
    const f: any = runFrame(rv, orbit, views(), k, look.bend, data.pace.span);
    // each pane's scene at its own place along the input
    if (sceneRun) sceneFrame(f);
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
          const { a, b, k: kk } = frameAt(r, pos[row]), A = pk[a], B = pk[b] ?? A;
          if (A && B) {
            const key = look.bend ? 'pe' : 'p0';
            at = A[key].map((x: number, i: number) => x + (B[key][i] - x) * kk);
            const mv = mark.querySelector('[data-mark-v]');
            if (mv) put(mv, withUnit(A.v + (B.v - A.v) * kk, r.q && r.q.field ? r.q.unit : '', r.q?.digits ?? 3));
          }
        }
      } else if (strength) {
        const f = (look.bend ? r.exag : 1) * k;
        at = r.governing_xyz.map((x: number, i: number) => x + f * r.governing_disp[i]);
      } else if (r.marker) {
        at = r.hot_xyz;
      }
      if (!at || !orbit) { mark.hidden = true; return; }
      if (sceneRun && lead.follow) {
        const r = data.panes[shown[row]].run, { a, b, k: kk } = frameAt(r, pos[row]);
        aimAt(orbit, followAt(r, a, b, kk) ?? orbit.target, vw, vh, insets[row]?.rect);
      }
      const [x, y, z] = orbit.project(at);
      let sx = ((x + 1) / 2) * vw, sy = ((1 - y) / 2) * vh;
      let shows = z < 1 && sx > 6 && sy > 6 && sx < vw - 6 && sy < vh - 6;
      const ins = insets[row];
      const inside = (r: number[], x: number, y: number) => x > r[0] && x < r[0] + r[2] && y > r[1] && y < r[1] + r[3];
      if (ins && shows && inside(ins.rect, sx, sy)) shows = false;
      const lab = mark.querySelector<HTMLElement>('.lrun-mark-v');
      let lx = '', ly = '';
      if (!shows && ins) {
        // a peak out of the close view is marked where it shows in the inset, its value kept inside the inset:
        // slid left over the ring and lifted above it when the inset is too narrow beside the ring
        const [ix, iy, iw, ih] = ins.rect, [u, w2, z2] = ins.orbit.project(at);
        sx = ix + ((u + 1) / 2) * iw;
        sy = iy + ((1 - w2) / 2) * ih;
        shows = z2 < 1 && sx > ix + 4 && sy > iy + 4 && sx < ix + iw - 4 && sy < iy + ih - 4;
        const lw = lab ? lab.offsetWidth : 0, right = ix + iw - 4;
        if (lab && lw && sx + 14 + lw > right) {
          const x = Math.max(right - lw - sx, ix + 4 - sx);
          lx = `${x.toFixed(1)}px`;
          if (x < 12) ly = sy - iy > 44 ? '-38px' : '14px';
        }
      } else if (lab && shows) {
        // the value beside the ring stays inside its pane and clear of the pane's name and the replay button:
        // right of the ring, else left of it, else below or above, else slid in from the edge
        mark.hidden = false;
        const lw = lab.offsetWidth, lh = lab.offsetHeight, pr = paneEls[row].getBoundingClientRect();
        const avoid: number[][] = [];
        for (const o of [...qa('.lrun-name', paneEls[row]), replay]) {
          if (o.hidden) continue;
          const b = o.getBoundingClientRect();
          if (b.width) avoid.push([b.left - pr.left, b.top - pr.top, b.width, b.height]);
        }
        const fits = (x: number, y: number) => x >= 4 && y >= 4 && x + lw <= vw - 4 && y + lh <= vh - 4
          && !avoid.some((b) => x < b[0] + b[2] && x + lw > b[0] && y < b[1] + b[3] && y + lh > b[1]);
        const tries = [[14, -11], [-14 - lw, -11], [-lw / 2, 14], [-lw / 2, -14 - lh]];
        const pick = tries.find(([dx, dy]) => fits(sx + dx, sy + dy));
        const [dx, dy] = pick ?? [Math.min(Math.max(14, 4 - sx), vw - 4 - lw - sx), Math.min(Math.max(-11, 4 - sy), vh - 4 - lh - sy)];
        if (lw && dx !== 14) lx = `${dx.toFixed(1)}px`;
        if (lw && dy !== -11) ly = `${dy.toFixed(1)}px`;
      }
      if (lab) { lab.style.left = lx; lab.style.top = ly; }
      mark.hidden = !shows;
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
    orbit = newOrbit(rv);
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
    for (const b of toggleBtns) {
      on(b, 'click', () => {
        const k = b.dataset.toggle as 'bend' | 'mag';
        look[k] = !look[k];
        b.setAttribute('aria-pressed', String(look[k]));
        requestDraw();
      });
    }
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
      shown.forEach((pi, row) => { pos[row] = Math.min(Math.max(p, 0), data.panes[pi].run.input.t.length - 1); });
      readings(1);
      requestDraw();
    };
    if (slider) {
      slider.disabled = false;
      on(slider, 'input', () => goTo(posOf(data.panes[shown[0]].run.input.t, Number(slider.value))));
    }
    if (!probeEl || !canvas) return;
    let hover: PointerEvent | null = null, pr = 0, asked = 0;
    const hide = () => { hover = null; probeEl.hidden = true; };
    const read = async () => {
      pr = 0;
      if (!hover || !orbit || !back) return;
      const e = hover, r = canvas!.getBoundingClientRect();
      const ir = insets[0]?.rect;
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (ir && px > ir[0] && px < ir[0] + ir[2] && py > ir[1] && py < ir[1] + ir[3]) { probeEl.hidden = true; return; }
      const nx = ((e.clientX - r.left) / r.width) * 2 - 1, ny = 1 - ((e.clientY - r.top) / r.height) * 2;
      const { origin, dir } = orbit.ray(nx, ny), id = ++asked;
      const hit = await back.probe(0, origin, dir).catch(() => null);
      if (id !== asked || !hover) return;
      if (!hit) { probeEl.hidden = true; return; }
      put(q('[data-probe-v]', probeEl), withUnit(hit.v, hit.unit, hit.digits + 1));
      put(q('[data-probe-k]', probeEl), hit.name);
      put(q('[data-probe-x]', probeEl), hit.limit > 0 ? `${Math.round((hit.v / hit.limit) * 100)}% of ${hit.limitLabel || 'the limit'}` : '');
      probeEl.hidden = false;
      placeProbe(e.clientX - r.left, e.clientY - r.top, r);
    };
    // The probe sits beside the pointer where it fits inside the view and clear of the peak's ring and value, the
    // part names, the replay button and the inset: right of the pointer, else left, below or above it, slid in
    // from the edges where none fits whole.
    const placeProbe = (x: number, y: number, r: DOMRect) => {
      const pw = probeEl.offsetWidth, ph = probeEl.offsetHeight, W = r.width, H = r.height, M = 6;
      const boxes: number[][] = [];
      for (const o of qa('.lrun-mark:not([hidden]) .target, .lrun-mark:not([hidden]) .lrun-mark-v, .lrun-name, .lrun-replay:not([hidden])', view)) {
        const b = o.getBoundingClientRect();
        if (b.width && b.height) boxes.push([b.left - r.left, b.top - r.top, b.width, b.height]);
      }
      if (insets[0]?.rect) boxes.push(insets[0].rect);
      const over = (ax: number, ay: number, b: number[]) =>
        Math.max(0, Math.min(ax + pw, b[0] + b[2]) - Math.max(ax, b[0])) * Math.max(0, Math.min(ay + ph, b[1] + b[3]) - Math.max(ay, b[1]));
      const tries = [[x + 14, y - ph / 2], [x - 14 - pw, y - ph / 2], [x - pw / 2, y + 18], [x - pw / 2, y - 18 - ph]];
      let best = tries[0], cost = Infinity;
      for (const [tx, ty] of tries) {
        const ax = Math.min(Math.max(tx, M), Math.max(M, W - M - pw)), ay = Math.min(Math.max(ty, M), Math.max(M, H - M - ph));
        // covering the point under the pointer hides what is read; being slid in from an edge costs a little
        const covers = x > ax - 6 && x < ax + pw + 6 && y > ay - 6 && y < ay + ph + 6;
        const c = boxes.reduce((a, b) => a + over(ax, ay, b), 0) + (covers ? pw * ph : 0) + Math.abs(ax - tx) + Math.abs(ay - ty);
        if (c < cost) { cost = c; best = [ax, ay]; }
      }
      probeEl.style.translate = `${best[0].toFixed(1)}px ${best[1].toFixed(1)}px`;
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
          if (sceneRun) pos[1] = homeOf(pi);
          finalState();
          return;
        }
        try { await back!.add(1, data.panes[pi].file, data.panes[pi].run, top); } catch { return; }
        if (abort.signal.aborted || b.getAttribute('aria-pressed') !== 'true') return;
        showPane(1, pi);
        if (sceneRun) pos[1] = homeOf(pi);
        restart();
      });
    }
  }
  // the labels under the meters and the slider: each kept inside its row and, where two would touch, the later one
  // dropped to a line of its own, measured again whenever the figure changes width
  const tickBoxes = qa('.lrun-ticks');
  const fitAll = () => { for (const b of tickBoxes) fitTicks(b); };
  if (tickBoxes.length) {
    const tro = new ResizeObserver(fitAll);
    tro.observe(fig);
    abort.signal.addEventListener('abort', () => tro.disconnect(), { once: true });
    document.fonts?.ready.then(() => { if (!abort.signal.aborted) fitAll(); });
  }

  if (!live) {
    fallBack();
    return () => abort.abort();
  }

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
    fallBack();
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
