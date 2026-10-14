// Plays a motion run back from the frames FlightSim.astro wrote. The player keeps the clock at real
// time or the speedup the stage shows, ticks the readouts, flags each event as it happens and shows
// the answer at the end before the run starts over. What it hands each frame is a SimState in
// meters, so the drawing layer can be swapped: svgStage draws the 2D stage, and a 3D viewer can take
// its place by implementing Stage. A stage plays only while it is on screen and the tab is visible.
// Under reduced motion nothing runs, and the page keeps the last moment it was drawn with.
import { clock, meters, pace } from './flight-format';

interface Frame { t: number; x: number; y: number; v: number; p: number }

/** One moment of the run, as the drawing layer gets it. */
export interface SimState {
  t: number;              // seconds since release or ignition
  done: boolean;          // the run has reached the ground
  x_m: number;            // distance from the start along the ground, in meters
  alt_m: number;          // height above the ground, in meters
  heading: number;        // direction of travel in degrees from straight up, positive downrange
  phase: string;          // rail, boost, coast, drogue, main, fall and so on, from trajectory.csv
  ghost_alt_m?: number;   // for a drop, the same body falling without air
}

/** A drawing layer: draws a moment, and says where on the stage (as fractions) a flag should point. */
export interface Stage {
  draw(s: SimState): void;
  anchor(s: SimState, ghost: boolean): [number, number];
}

const HOLD = 4500;     // ms the answer stays up before the run starts over
const FLAG = 1700;     // ms a flag stays up

/** The run as the page wrote it: frames in stage units with the scale that turns them into meters. */
function readRun(fig: HTMLElement) {
  const d = fig.dataset;
  const frames: Frame[] = (d.frames ?? '').split(';').map((f) => {
    const [t, x, y, v, p] = f.split(',').map(Number);
    return { t, x, y, v, p };
  });
  const scale = Number(d.scale) || 1;            // stage units per meter
  const ground = Number(d.ground);
  const origin = Number(d.origin);               // the start's position along the stage
  const [gT, gX, gTop] = (d.ghost ?? '').split(',').map(Number);
  return {
    frames, scale, ground, origin, w: Number(d.w), h: Number(d.h),
    // phases as key=label pairs: the key steers the drawing, the label is what the readout says
    phases: (d.phases ?? '').split(';').map((p) => p.split('=')),
    events: (d.events ?? '').split(';').filter(Boolean).map((e) => {
      const [t, name, who] = e.split(',');
      return { t: Number(t), name, ghost: who === 'ghost' };
    }),
    ghost: d.ghost ? { t: gT, x: gX, alt: (ground - gTop) / scale } : null,
    speed: Number(d.speed) || 1,
    decimals: Number(d.decimals) || 1,
  };
}
type RunData = ReturnType<typeof readRun>;

/** The 2D stage FlightSim.astro drew: the body over the ground, the path so far, its flame and parachutes. */
export function svgStage(fig: HTMLElement, run: RunData): Stage {
  const { frames, scale, ground, origin, w, h, ghost } = run;
  const obj = fig.querySelector<SVGGElement>('.sim-obj')!;
  const trail = fig.querySelector<SVGPathElement>('.sim-trail')!;
  const rot = fig.querySelector<SVGGElement>('.sim-rot');
  const flame = fig.querySelector<SVGPathElement>('.sim-flame');
  const drogue = fig.querySelector<SVGGElement>('[data-chute="drogue"]');
  const main = fig.querySelector<SVGGElement>('[data-chute="main"]');
  const ghostEl = ghost ? fig.querySelector<SVGPathElement>('.sim-ghost') : null;
  const keys = new Set(run.phases.map(([k]) => k));
  const show = (el: SVGElement | null, on: boolean) => { if (el) el.style.display = on ? '' : 'none'; };
  const X = (x: number) => origin + x * scale;
  const Y = (alt: number) => ground - alt * scale;

  // the trail as one path, cut after each frame, so the path so far is a slice of it
  let full = '';
  const cut = frames.map((f, i) => (full += `${i ? 'L' : 'M'}${f.x} ${f.y}`).length);
  let i = 0;

  return {
    draw(s) {
      if (frames[i].t > s.t) i = 0;
      while (i < frames.length - 2 && frames[i + 1].t <= s.t) i++;
      const x = X(s.x_m).toFixed(1), y = Y(s.alt_m).toFixed(1);
      obj.setAttribute('transform', `translate(${x} ${y})`);
      trail.setAttribute('d', `${full.slice(0, cut[i])}L${x} ${y}`);
      if (rot) {
        const hanging = s.phase === 'drogue' || s.phase === 'main';
        rot.setAttribute('transform', `rotate(${hanging ? 0 : Math.round(s.heading)})`);
        show(flame, (s.phase === 'rail' || s.phase === 'boost') && !s.done);
        show(drogue, hanging && keys.has('drogue'));
        show(main, s.phase === 'main');
        if (s.phase === 'main') drogue?.setAttribute('transform', 'translate(0 -18)'); else drogue?.removeAttribute('transform');
      }
      if (ghostEl && ghost && s.ghost_alt_m !== undefined) {
        ghostEl.setAttribute('transform', `translate(${ghost.x} ${Y(s.ghost_alt_m).toFixed(1)})`);
      }
    },
    anchor(s, onGhost) {
      const x = onGhost && ghost ? ghost.x : X(s.x_m);
      const y = Y(onGhost && s.ghost_alt_m !== undefined ? s.ghost_alt_m : s.alt_m);
      return [x / w, y / h];
    },
  };
}

function player(fig: HTMLElement, makeStage: (fig: HTMLElement, run: RunData) => Stage = svgStage) {
  const run = readRun(fig);
  const { frames, phases, events, ghost, speed, decimals, scale, ground, origin } = run;
  const stage = makeStage(fig, run);
  const tEnd = frames[frames.length - 1].t;
  const flag = fig.querySelector<HTMLElement>('.sim-flag');
  const answer = fig.querySelector<HTMLElement>('.sim-answer');
  const dds = [...fig.querySelectorAll<HTMLElement>('.sim-read dd')];
  const toggle = fig.querySelector<HTMLButtonElement>('.sim-toggle');

  let t = 0, i = 0, next = 0, last = 0, raf = 0, holdUntil = 0, flagUntil = 0;
  let playing = false, started = false, held = false;

  /** The run at time t, stepped straight between the frames around it. */
  function state(): { s: SimState; v: number; label: string } {
    while (i < frames.length - 2 && frames[i + 1].t <= t) i++;
    const a = frames[i], b = frames[i + 1] ?? a;
    const k = b.t > a.t ? Math.min(Math.max((t - a.t) / (b.t - a.t), 0), 1) : 1;
    const [key, label] = phases[k >= 1 ? b.p : a.p] ?? ['', ''];
    const heading = (Math.atan2(b.x - a.x, a.y - b.y) * 180) / Math.PI;
    const s: SimState = {
      t, done: t >= tEnd,
      x_m: (a.x + (b.x - a.x) * k - origin) / scale,
      alt_m: Math.max(ground - (a.y + (b.y - a.y) * k), 0) / scale,
      heading: Number.isFinite(heading) ? heading : 0, phase: key,
      // without air the drop follows h = h0 (1 - (t / t_land)^2) until it lands
      ghost_alt_m: ghost ? (t >= ghost.t ? 0 : ghost.alt * (1 - (t / ghost.t) ** 2)) : undefined,
    };
    return { s, v: a.v + (b.v - a.v) * k, label };
  }

  function draw() {
    const { s, v, label } = state();
    stage.draw(s);
    if (dds[0]) dds[0].textContent = clock(t, decimals);
    if (dds[1]) dds[1].textContent = meters(s.alt_m);
    if (dds[2]) dds[2].textContent = pace(v);
    if (dds[3]) dds[3].textContent = s.done ? 'Landed' : label;
    // a flag for each event the run has just passed, beside the body and kept on the stage
    while (next < events.length && events[next].t <= t) {
      if (flag) {
        const [fx, fy] = stage.anchor(s, events[next].ghost);
        flag.textContent = events[next].name;
        flag.hidden = false;
        flag.style.left = `${fx * 100}%`;
        flag.style.top = `${Math.min(Math.max(fy * 100, 16), 84)}%`;
        flag.toggleAttribute('data-left', fx > 0.6);
        flagUntil = performance.now() + FLAG;
      }
      next++;
    }
  }

  function reset() {
    t = 0; i = 0; next = 0;
    answer?.classList.add('is-waiting');
    if (flag) flag.hidden = true;
    draw();
  }

  function tick(now: number) {
    raf = 0;
    if (!playing) return;
    const dt = last ? Math.min((now - last) / 1000, 0.25) : 0;
    last = now;
    if (holdUntil) {
      if (now >= holdUntil) { holdUntil = 0; reset(); }
    } else {
      t = Math.min(t + dt * speed, tEnd);
      draw();
      if (t >= tEnd) { answer?.classList.remove('is-waiting'); holdUntil = now + HOLD; }
    }
    if (flag && !flag.hidden && now > flagUntil) flag.hidden = true;
    raf = requestAnimationFrame(tick);
  }

  const self = {
    visible: false,
    arm() { if (!started) { started = true; reset(); } },
    play() {
      if (playing || held) return;
      self.arm();
      playing = true; last = 0;
      raf ||= requestAnimationFrame(tick);
    },
    pause() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
  };
  // the stage's own button holds it still until it is pressed again
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener('click', () => {
      held = !held;
      toggle.textContent = held ? 'Play' : 'Pause';
      if (held) self.pause(); else if (self.visible && !document.hidden) self.play();
    });
  }
  return self;
}

/** Start every stage on the page, each playing while it is on screen; returns a function that stops them. */
export function initSims(figs: HTMLElement[]): () => void {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) return () => {};
  const players = new Map(figs.map((f) => [f as Element, player(f)]));
  const seen = new WeakSet<Element>();
  const onScreen = new Set<Element>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const p = players.get(e.target)!;
      // a stage below the screen as the page opens goes back to its start out of sight
      if (!seen.has(e.target)) { seen.add(e.target); if (!e.isIntersecting) p.arm(); }
      if (e.intersectionRatio >= 0.3) { onScreen.add(e.target); p.visible = true; if (!document.hidden) p.play(); }
      else if (!e.isIntersecting) { onScreen.delete(e.target); p.visible = false; p.pause(); }
    }
  }, { threshold: [0, 0.3] });
  figs.forEach((f) => io.observe(f));
  const vis = () => players.forEach((p, el) => (document.hidden || !onScreen.has(el) ? p.pause() : p.play()));
  document.addEventListener('visibilitychange', vis);
  return () => {
    io.disconnect();
    document.removeEventListener('visibilitychange', vis);
    players.forEach((p) => p.pause());
  };
}
