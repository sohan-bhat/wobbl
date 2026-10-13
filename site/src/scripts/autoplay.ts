// The hero part's idle animation: a slow loop that swings the view, carries the load on the shaft
// round a path over the sphere of directions and eases its size up and down, so the verdict moves
// through holds, marginal and fails. It stops the moment the visitor clicks, taps, drags or presses a
// key, and comes back after a quiet spell, blending in from wherever they left things. It only ever
// changes the view and the load, never the URL.

/** The camera around its target: azimuth and elevation in radians, distance in mm. */
export interface View { az: number; el: number; dist: number }
/** az and el give the load's direction in degrees; hand is how much of the hint pointer shows, 0 to 1 */
export interface Frame { az: number; el: number; load: number; view: View; hand: number }
export type Mode = 'playing' | 'held';

const LOOP = 16000; // ms for one loop
const BLEND = 1000; // ms to ease from the visitor's state into the loop
const QUIET = 7000; // ms without input before it plays again

const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
/** From a toward b by s, the short way round a circle of the given period. */
const turn = (a: number, b: number, s: number, period: number) => {
  const d = ((((b - a) % period) + period * 1.5) % period) - period / 2;
  return a + d * s;
};

export interface Autoplay {
  play(): void;
  hold(): void;
  destroy(): void;
  readonly mode: Mode;
}

/**
 * read() gives the state now, apply(frame) shows a frame, home is the starting view and start the
 * default direction. Phones get at most 30 frames a second. onMode reports playing, and held by the
 * visitor, playing again after a quiet spell.
 */
export function createAutoplay(o: {
  root: HTMLElement; read(): Frame; apply(f: Frame): void; home: View; start: { az: number; el: number }; phone: boolean;
  onMode(m: Mode): void;
}): Autoplay {
  let mode: Mode = 'held';
  let t = 0, last = 0, lastShown = 0, raf = 0, quiet = 0;
  let from: Frame | null = null, blendAt = 0;
  let onScreen = true, shown = !document.hidden, leaving = false;

  // The loop: the load goes once round the shaft while it tips out along the shaft and back in twice,
  // a smooth path over the sphere of directions, its size swells once between 800 and 3,000 N, and
  // the view swings ten degrees either side of where it started, little enough that the arrow keeps
  // facing the visitor. On the very first loop a pointer rides on the arrow's head, as if it were
  // dragging it round, and fades out; once the visitor has stepped in, it does not come back.
  let first = true, played = 0;
  const at = (ms: number): Frame => {
    const p = (ms / LOOP) * Math.PI * 2;
    return {
      az: (((o.start.az + (360 * ms) / LOOP) % 360) + 360) % 360,
      el: o.start.el + 30 * Math.sin(2 * p),
      load: 1900 + 1100 * Math.sin(p),
      view: { az: o.home.az + 0.17 * Math.sin(p), el: o.home.el + 0.045 * Math.sin(2 * p), dist: o.home.dist },
      hand: first ? Math.min(1, played / 400, (LOOP - played) / 700) * 0.9 : 0,
    };
  };

  const running = () => mode === 'playing' && onScreen && shown && !leaving;
  function tick(now: number) {
    raf = 0;
    if (!running()) return;
    raf = requestAnimationFrame(tick);
    const dt = Math.min(now - (last || now), 50); // a hidden tab or a stall never skips ahead
    t = (t + dt) % LOOP;
    played += dt;
    if (played >= LOOP) first = false;
    last = now;
    if (o.phone && now - lastShown < 30) return;
    lastShown = now;
    let f = at(t);
    if (from) {
      const s = smooth((now - blendAt) / BLEND);
      f = {
        az: turn(from.az, f.az, s, 360),
        el: from.el + (f.el - from.el) * s,
        load: from.load + (f.load - from.load) * s,
        view: {
          az: turn(from.view.az, f.view.az, s, Math.PI * 2),
          el: from.view.el + (f.view.el - from.view.el) * s,
          dist: from.view.dist + (f.view.dist - from.view.dist) * s,
        },
        hand: f.hand * s,
      };
      if (s >= 1) from = null;
    }
    o.apply(f);
  }
  const wake = () => {
    if (running() && !raf) { last = 0; raf = requestAnimationFrame(tick); }
  };
  const sleep = () => { cancelAnimationFrame(raf); raf = 0; };
  function set(m: Mode) {
    clearTimeout(quiet);
    if (m === 'held') quiet = window.setTimeout(() => set('playing'), QUIET);
    if (m === mode) return;
    if (m === 'playing') { from = o.read(); blendAt = performance.now(); } else if (played) first = false;
    mode = m;
    if (m === 'playing') wake(); else sleep();
    o.onMode(m);
  }

  // Off screen, in a hidden tab or while the page changes, it waits where it is.
  const io = new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; if (onScreen) wake(); else sleep(); });
  io.observe(o.root);
  const onVisible = () => { shown = !document.hidden; if (shown) wake(); else sleep(); };
  const onLeave = () => { leaving = true; sleep(); };
  document.addEventListener('visibilitychange', onVisible);
  document.addEventListener('astro:before-preparation', onLeave);

  return {
    play: () => set('playing'),
    hold: () => set('held'),
    destroy() {
      sleep();
      clearTimeout(quiet);
      io.disconnect();
      document.removeEventListener('visibilitychange', onVisible);
      document.removeEventListener('astro:before-preparation', onLeave);
    },
    get mode() { return mode; },
  };
}
