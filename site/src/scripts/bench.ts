// Plays a circuit run's bench (CircuitBench.astro) from power on to steady state. The drawing is a
// still of the final state that Wobbl wrote from the run, and the parts that change carry data-*
// attributes (see src/wobbl/circuits/bench.py): readings climb through data-series, current moves
// along the wires at data-flow units a second, the switch node toggles with data-pwm, the scope draws
// its trace with data-sweep, and the verdict, marked data-after, shows at steady state. A bench plays
// only while it is on screen and the tab is visible. Under reduced motion nothing runs, and the page
// keeps the final state it was drawn with.

const HOLD = 7000;  // ms at steady state before a bench with controls powers on again
const ONCE = 3000;  // ms a bench without controls keeps running at steady state before it rests
const DOT = 12;     // the spacing of the moving dots, as Wobbl dashes them

const nums = (s: string | undefined) => (s ?? '').trim().split(/\s+/).map(Number);

/** The value at p, from 0 to 1, along samples spread evenly over the ramp. */
function at(vals: number[], p: number): number {
  if (vals.length < 2) return vals[0] ?? 0;
  const x = Math.min(Math.max(p, 0), 1) * (vals.length - 1);
  const i = Math.min(Math.floor(x), vals.length - 2);
  return vals[i] + (vals[i + 1] - vals[i]) * (x - i);
}

function mix(a: string, b: string, k: number): string {
  const c = (h: string, i: number) => parseInt(h.slice(1 + 2 * i, 3 + 2 * i), 16);
  return `rgb(${[0, 1, 2].map((i) => Math.round(c(a, i) + (c(b, i) - c(a, i)) * k)).join(',')})`;
}

/** Whether a "duty period [inverted]" switch is on at t seconds. */
function on(spec: string | undefined, t: number): boolean {
  if (!spec) return true;
  const [duty, period] = nums(spec);
  const high = (t % period) / period < duty;
  return spec.includes('inverted') ? !high : high;
}

function player(fig: HTMLElement) {
  const svg = fig.querySelector<SVGSVGElement>('svg[data-live]');
  if (!svg) return null;
  const ramp = Number(svg.dataset.live) || 2;
  const loop = fig.dataset.bench === 'loop';
  const all = <T extends SVGElement>(sel: string) => [...svg.querySelectorAll<T>(sel)];
  const series = all('[data-series]').map((el) => {
    const [dec, unit] = (el.dataset.fmt ?? '').split(' ');
    return { el, vals: nums(el.dataset.series), attr: el.dataset.attr, dec: Number(dec) || 0, unit: unit ?? '' };
  });
  const flows = all('[data-flow]').map((el) => ({
    el, speed: Number(el.dataset.flow), k: el.dataset.k ? nums(el.dataset.k) : null, pwm: el.dataset.pwm, off: 0,
  }));
  const gates = all('[data-pwm]:not([data-flow])').map((el) => ({ el, spec: el.dataset.pwm, lit: el.getAttribute('opacity') ?? '1' }));
  const waves = all('[data-wave]').map((el) => ({ el, vals: nums(el.dataset.wave), period: Number(el.dataset.period) || 1 }));
  const sweeps = all('[data-sweep]').map((el) => ({ el, period: el.dataset.sweep === 'ramp' ? 0 : Number(el.dataset.sweep) || ramp }));
  const heats = all('[data-heat]').map((el) => ({ el, colors: (el.dataset.heat ?? '').split(' ') }));
  const after = all('[data-after]');
  const drifts = all('[data-drift]').map((el) => ({ el, period: Number(el.dataset.drift) || 1.6 }));
  const view = fig.querySelector<HTMLElement>('.bench-view');
  const toggle = fig.querySelector<HTMLButtonElement>('.bench-toggle');

  let t = 0, last = 0, raf = 0, holdUntil = 0;
  let playing = false, started = false, held = false, rested = false;

  function draw(dt: number) {
    const p = Math.min(t / ramp, 1);
    for (const s of series) {
      const v = at(s.vals, p);
      if (s.attr) s.el.setAttribute(s.attr, v.toFixed(3));
      else s.el.textContent = v.toFixed(s.dec) + (s.unit ? ` ${s.unit}` : '');
    }
    for (const h of heats) h.el.style.fill = mix(h.colors[0], h.colors[1], p);
    for (const el of after) el.style.visibility = t >= ramp ? '' : 'hidden';
    for (const s of sweeps) s.el.style.strokeDashoffset = String(1 - (s.period ? (t % s.period) / s.period : p));
    for (const g of gates) g.el.style.opacity = on(g.spec, t) ? g.lit : '0';
    for (const w of waves) w.el.style.opacity = at(w.vals, (t % w.period) / w.period).toFixed(3);
    for (const f of flows) {
      if (on(f.pwm, t)) f.off = (f.off - f.speed * (f.k ? at(f.k, p) : 1) * dt) % DOT;
      f.el.style.strokeDashoffset = f.off.toFixed(2);
    }
    for (const d of drifts) {
      const k = (t % d.period) / d.period;
      d.el.style.transform = `translateY(${(-10 * k).toFixed(1)}px)`;
      d.el.style.opacity = (0.9 * Math.sin(Math.PI * k)).toFixed(2);
    }
  }

  function reset() {
    t = 0;
    flows.forEach((f) => (f.off = 0));
    sweeps.forEach(({ el }) => (el.style.strokeDasharray = '1 1'));
    draw(0);
  }

  function tick(now: number) {
    raf = 0;
    if (!playing) return;
    const dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
    last = now;
    if (view?.hidden) {  // the schematic is showing: hold the bench where it is
      raf = requestAnimationFrame(tick);
      return;
    }
    if (holdUntil && now >= holdUntil) {
      holdUntil = 0;
      if (!loop) { rested = true; self.pause(); return; }
      reset();
    } else {
      t += dt;
      draw(dt);
      if (t >= ramp && !holdUntil) holdUntil = now + (loop ? HOLD : ONCE);
    }
    raf = requestAnimationFrame(tick);
  }

  const self = {
    visible: false,
    arm() { if (!started) { started = true; reset(); } },
    play() {
      if (playing || held || rested) return;
      self.arm();
      playing = true; last = 0;
      raf ||= requestAnimationFrame(tick);
    },
    pause() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
  };
  // the bench's own button holds it still until it is pressed again
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

/** Start every bench on the page, each playing while it is on screen; returns a function that stops them. */
export function initBenches(figs: HTMLElement[]): () => void {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) return () => {};
  const players = new Map<Element, NonNullable<ReturnType<typeof player>>>();
  for (const f of figs) {
    const p = player(f);
    if (p) players.set(f, p);
  }
  const seen = new WeakSet<Element>();
  const onScreen = new Set<Element>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const p = players.get(e.target)!;
      // a bench out of sight as the page opens goes back to power on, where nobody sees it jump
      if (!seen.has(e.target)) { seen.add(e.target); if (!e.isIntersecting) p.arm(); }
      if (e.intersectionRatio >= 0.3) { onScreen.add(e.target); p.visible = true; if (!document.hidden) p.play(); }
      else if (!e.isIntersecting) { onScreen.delete(e.target); p.visible = false; p.pause(); }
    }
  }, { threshold: [0, 0.3] });
  players.forEach((_, el) => io.observe(el));
  const vis = () => players.forEach((p, el) => (document.hidden || !onScreen.has(el) ? p.pause() : p.play()));
  document.addEventListener('visibilitychange', vis);
  return () => {
    io.disconnect();
    document.removeEventListener('visibilitychange', vis);
    players.forEach((p) => p.pause());
  };
}
