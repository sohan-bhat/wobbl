// Plays a motion run back from the frames FlightSim.astro wrote: the body moves along its path at
// real time or the speedup the stage shows, the readouts tick with the data, a flag names each
// event as it happens, and the answer shows at the end before the run starts over. A stage plays
// only while it is on screen and the tab is visible. Under reduced motion nothing runs, and the
// page keeps the last moment it was drawn with.
import { clock, meters, pace } from './flight-format';

interface Frame { t: number; x: number; y: number; v: number; p: number }

const HOLD = 4500;     // ms the answer stays up before the run starts over
const FLAG = 1700;     // ms a flag stays up

function player(fig: HTMLElement) {
  const d = fig.dataset;
  const frames: Frame[] = (d.frames ?? '').split(';').map((f) => {
    const [t, x, y, v, p] = f.split(',').map(Number);
    return { t, x, y, v, p };
  });
  // phases as key=label pairs: the key steers the drawing, the label is what the readout says
  const phases = (d.phases ?? '').split(';').map((p) => p.split('='));
  const events = (d.events ?? '').split(';').filter(Boolean).map((e) => {
    const [t, name, who] = e.split(',');
    return { t: Number(t), name, ghost: who === 'ghost' };
  });
  // the same drop without air: when it lands, where it falls and from how high on the stage
  const [gT, gX, gTop] = (d.ghost ?? '').split(',').map(Number);
  const ghost = d.ghost ? fig.querySelector<SVGPathElement>('.sim-ghost') : null;
  const speed = Number(d.speed) || 1;
  const decimals = Number(d.decimals) || 1;
  const scale = Number(d.scale) || 1;            // stage units per meter
  const ground = Number(d.ground);
  const [W, H] = [Number(d.w), Number(d.h)];
  const tEnd = frames[frames.length - 1].t;
  const keys = new Set(phases.map(([k]) => k));

  const obj = fig.querySelector<SVGGElement>('.sim-obj')!;
  const trail = fig.querySelector<SVGPathElement>('.sim-trail')!;
  const rot = fig.querySelector<SVGGElement>('.sim-rot');
  const flame = fig.querySelector<SVGPathElement>('.sim-flame');
  const drogue = fig.querySelector<SVGGElement>('[data-chute="drogue"]');
  const main = fig.querySelector<SVGGElement>('[data-chute="main"]');
  const flag = fig.querySelector<HTMLElement>('.sim-flag');
  const answer = fig.querySelector<HTMLElement>('.sim-answer');
  const dds = [...fig.querySelectorAll<HTMLElement>('.sim-read dd')];
  const toggle = fig.querySelector<HTMLButtonElement>('.sim-toggle');

  // the trail as one path, cut after each frame, so the path so far is a slice of it
  let full = '';
  const cut = frames.map((f, i) => (full += `${i ? 'L' : 'M'}${f.x} ${f.y}`).length);

  let t = 0, i = 0, next = 0, last = 0, raf = 0, holdUntil = 0, flagUntil = 0;
  let playing = false, started = false, held = false;
  const show = (el: SVGElement | null, on: boolean) => { if (el) el.style.display = on ? '' : 'none'; };

  function draw() {
    while (i < frames.length - 2 && frames[i + 1].t <= t) i++;
    const a = frames[i], b = frames[i + 1] ?? a;
    const k = b.t > a.t ? Math.min(Math.max((t - a.t) / (b.t - a.t), 0), 1) : 1;
    const x = a.x + (b.x - a.x) * k, y = a.y + (b.y - a.y) * k, v = a.v + (b.v - a.v) * k;
    const [key, label] = phases[k >= 1 ? b.p : a.p] ?? ['', ''];
    const done = t >= tEnd;
    obj.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
    trail.setAttribute('d', `${full.slice(0, cut[i])}L${x.toFixed(1)} ${y.toFixed(1)}`);
    if (rot) {
      const hanging = key === 'drogue' || key === 'main';
      const ang = hanging ? 0 : (Math.atan2(b.x - a.x, a.y - b.y) * 180) / Math.PI;
      rot.setAttribute('transform', `rotate(${Number.isFinite(ang) ? Math.round(ang) : 0})`);
      show(flame, (key === 'rail' || key === 'boost') && !done);
      show(drogue, hanging && keys.has('drogue'));
      show(main, key === 'main');
      if (key === 'main') drogue?.setAttribute('transform', 'translate(0 -18)'); else drogue?.removeAttribute('transform');
    }
    const gy = t >= gT ? ground : gTop + (ground - gTop) * (t / gT) ** 2;
    ghost?.setAttribute('transform', `translate(${gX} ${gy.toFixed(1)})`);
    if (dds[0]) dds[0].textContent = clock(t, decimals);
    if (dds[1]) dds[1].textContent = meters(Math.max(ground - y, 0) / scale);
    if (dds[2]) dds[2].textContent = pace(v);
    if (dds[3]) dds[3].textContent = done ? 'Landed' : label;
    // a flag for each event the run has just passed, beside the body and kept on the stage
    while (next < events.length && events[next].t <= t) {
      if (flag) {
        const e = events[next];
        const [fx, fy] = e.ghost ? [gX, gy] : [x, y];
        flag.textContent = e.name;
        flag.hidden = false;
        flag.style.left = `${(fx / W) * 100}%`;
        flag.style.top = `${Math.min(Math.max((fy / H) * 100, 16), 84)}%`;
        flag.toggleAttribute('data-left', fx / W > 0.6);
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
