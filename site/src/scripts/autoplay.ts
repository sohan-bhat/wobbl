// The hero part's idle animation: the view turns all the way round the part while the load on the
// shaft wanders over the sphere of directions and swells and shrinks, so the verdict moves through
// holds, marginal and fails. Nothing is scripted. Each time it plays it starts from wherever things
// are, the visitor's own view and load included, and takes a fresh random path from there. It stops
// the moment the visitor clicks, taps, drags or presses a key, and comes back after a quiet spell.
// It only ever changes the view and the load, never the URL.

/** The camera around its target: azimuth and elevation in radians, distance in mm. */
export interface View { az: number; el: number; dist: number }
/** az and el give the load's direction in degrees; hand is how much of the hint pointer shows, 0 to 1 */
export interface Frame { az: number; el: number; load: number; view: View; hand: number }
export type Mode = 'playing' | 'held';

const TURN = 12000; // ms for the view to go once round the part
const EASE = 1000; // ms to get up to speed from a standstill
const QUIET = 5250; // ms without input before it plays again
const HAND = 12000; // ms the hint pointer rides on the arrow the first time it plays

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** A smooth random drift between -amp and amp: three slow waves at random speeds and phases. */
function drift(amp: number, period: number) {
  const waves = [[0.55, 1], [0.3, 0.6], [0.15, 0.37]].map(([a, k]) => ({
    a: a * amp, w: (Math.PI * 2) / (period * k * rand(0.75, 1.25)), ph: rand(0, Math.PI * 2),
  }));
  return (ms: number) => waves.reduce((s, v) => s + v.a * Math.sin(v.w * ms + v.ph), 0);
}

export interface Autoplay {
  play(): void;
  hold(): void;
  destroy(): void;
  readonly mode: Mode;
}

/**
 * read() gives the state now, apply(frame) shows a frame and home is the starting view. Phones get at
 * most 30 frames a second. onMode reports playing, and held by the visitor, playing again after a
 * quiet spell.
 */
export function createAutoplay(o: {
  root: HTMLElement; read(): Frame; apply(f: Frame): void; home: View; phone: boolean;
  onMode(m: Mode): void;
}): Autoplay {
  let mode: Mode = 'held';
  let t = 0, last = 0, lastShown = 0, raf = 0, quiet = 0;
  let onScreen = true, shown = !document.hidden, leaving = false;

  // A path is made fresh each time it plays, starting from the frame it is given. The view turns
  // round the part at a steady pace, one way or the other, its height drifting a little about the
  // home view's. The load goes round the shaft at its own pace, one way or the other, tipping out
  // along the shaft and back as it goes, and its size drifts between 800 and 3,000 N. Each of these
  // slides from where it started into its drift over a few seconds, and time itself eases in, so
  // nothing jumps. The first time it plays a pointer rides on the arrow's head, as if it were
  // dragging it round, and fades out; once the visitor has stepped in, it does not come back.
  let first = true, played = 0;
  let at: (ms: number) => Frame = () => o.read();
  function newPath(f: Frame) {
    const spin = Math.random() < 0.5 ? -1 : 1, round = Math.random() < 0.5 ? -1 : 1;
    const pace = (360 / TURN) * rand(0.7, 1.2); // degrees a ms the load goes round the shaft
    const camEl = drift(0.15, 9000), dirAz = drift(40, 7000), dirEl = drift(40, 6000), size = drift(1300, 9000);
    at = (ms) => {
      const fade = 1 - Math.exp(-ms / 2500); // how far each one has slid from its start into its drift
      const toward = (from: number, to: number) => from + (to - from) * fade;
      return {
        az: ((((f.az + round * pace * ms + dirAz(ms) * fade) % 360) + 360) % 360),
        el: toward(f.el, clamp(dirEl(ms), -60, 60)),
        load: toward(f.load, clamp(1900 + size(ms), 800, 3000)),
        view: {
          az: (f.view.az + (spin * Math.PI * 2 * ms) / TURN) % (Math.PI * 2),
          el: toward(f.view.el, clamp(o.home.el + camEl(ms), 0.05, 1.2)),
          dist: f.view.dist,
        },
        hand: first ? Math.min(1, played / 400, (HAND - played) / 700) * 0.9 : 0,
      };
    };
  }
  // time along the path, easing in from a standstill over the first second
  const along = (ms: number) => (ms < EASE ? (ms * ms) / (2 * EASE) : ms - EASE / 2);

  const running = () => mode === 'playing' && onScreen && shown && !leaving;
  function tick(now: number) {
    raf = 0;
    if (!running()) return;
    raf = requestAnimationFrame(tick);
    const dt = Math.min(now - (last || now), 50); // a hidden tab or a stall never skips ahead
    t += dt;
    played += dt;
    if (played >= HAND) first = false;
    last = now;
    if (o.phone && now - lastShown < 30) return;
    lastShown = now;
    o.apply(at(along(t)));
  }
  const wake = () => {
    if (running() && !raf) { last = 0; raf = requestAnimationFrame(tick); }
  };
  const sleep = () => { cancelAnimationFrame(raf); raf = 0; };
  function set(m: Mode) {
    clearTimeout(quiet);
    if (m === 'held') quiet = window.setTimeout(() => set('playing'), QUIET);
    if (m === mode) return;
    if (m === 'playing') { t = 0; newPath(o.read()); } else if (played) first = false;
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
