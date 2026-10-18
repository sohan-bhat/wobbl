// A scene's motion (src/wobbl/viz/track.py): the physical event at the value its question asked about, played on
// its own clock. The scene's slider sweeps an input such as a drop height; the motion is what happens at the asked
// value, frame by frame in physical time, with the playback second each frame is shown at, so the fall of a part
// plays in real time and the millisecond of its impact slowed thousands of times. Objects carry tracks: a rigid
// pose, a field and an elastic shape stored as a few shapes per vertex and a weight per frame, a size and a glow.
// Nothing here touches three.js or the page: the drawing worker (play.js, play-scene.js), the page (live-run.ts,
// run-view.js) and the build (scripts/scenes.mjs) all read the motion through these functions.

const lerp = (a, b, k) => a + (b - a) * k;

/** The motion frames either side of physical time tau and the share of the way to the next. */
export function motionFrame(mo, tau) {
  const t = mo.t, M = t.length;
  if (!(tau > t[0])) return { a: 0, b: 0, k: 0 };
  if (tau >= t[M - 1]) return { a: M - 1, b: M - 1, k: 0 };
  let lo = 0, hi = M - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= tau) lo = mid; else hi = mid; }
  return { a: lo, b: hi, k: (tau - t[lo]) / (t[hi] - t[lo]) };
}

/** The physical time shown at playback second p of the motion. */
export function timeAt(mo, p) {
  const t = mo.t, s = mo.play, M = t.length;
  if (!(p > 0)) return t[0];
  if (p >= s[M - 1]) return t[M - 1];
  let lo = 0, hi = M - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s[mid] <= p) lo = mid; else hi = mid; }
  return lerp(t[lo], t[hi], (p - s[lo]) / (s[hi] - s[lo]));
}

/** How many times slower than real time the motion plays at playback second p (below 1 where it is sped up). */
export function slowAt(mo, p) {
  const t = mo.t, s = mo.play, M = t.length;
  let i = 0;
  while (i < M - 2 && s[i + 1] <= p) i++;
  return (s[i + 1] - s[i]) / (t[i + 1] - t[i]);
}

/** The motion's length in playback seconds. */
export const motionSeconds = (mo) => mo.play[mo.play.length - 1];

/** Each tracked array of a scene object, decoded for the drawing (arr is scene-file.js's reader), and the poses of
 * a cycle it repeats on the steady clock. */
export function decodeTrack(o, arr, d) {
  const cy = o.steady && o.steady.cycle;
  if (cy) d.cycle = { pos: arr(cy.pos), quat: arr(cy.quat), n: cy.pos.shape[0] };
  // a sweep that kept the event at several values has a track for each of its motions (motionOf)
  if (o.tracks) d.tracks = o.tracks.map((tr) => (tr ? oneTrack(tr, arr) : null));
  if (o.track) d.track = oneTrack(o.track, arr);
}

/** The motion a scene plays at variant mv: the asked one, or one of the motions a sweep kept at each value. */
export function motionOf(h, mv) {
  const m = h.motion;
  if (!m || !m.variants) return m;
  return m.variants[Number.isInteger(mv) && m.variants[mv] ? mv : m.asked ?? 0];
}

/** An object's track over motion variant mv (motionOf), decoded. */
export function trackOf(d, h, mv) {
  if (!d.tracks || !h.motion || !h.motion.variants) return d.track || null;
  return d.tracks[Number.isInteger(mv) && mv < d.tracks.length ? mv : h.motion.asked ?? 0] || null;
}

function oneTrack(tr, arr) {
  const out = {};
  if (tr.pos) { out.pos = arr(tr.pos); out.quat = arr(tr.quat); }
  if (tr.size) out.size = arr(tr.size);
  if (tr.glow) out.glow = tr.glow;
  if (tr.field) out.field = { coef: arr(tr.field.coef), basis: arr(tr.field.basis), r: tr.field.rank };
  if (tr.deform) out.deform = { coef: arr(tr.deform.coef), basis: arr(tr.deform.basis), r: tr.deform.rank, ex: tr.deform.exaggerate || 1 };
  if (tr.drawn) out.drawn = arr(tr.drawn);
  if (tr.value) out.value = tr.value.values.map((v) => (v === null ? NaN : v));
  if (tr.when) out.when = tr.when;
  return out;
}

const at = (A, w, mf, i) => (mf.k ? lerp(A[mf.a * w + i], A[mf.b * w + i], mf.k) : A[mf.a * w + i]);

/** A tracked pose at motion frame mf, as scene-file.js poseAt gives one, or null. */
export function trackPose(tr, mf) {
  if (!tr || !tr.pos) return null;
  const p = [0, 1, 2].map((i) => at(tr.pos, 3, mf, i));
  let q = [0, 1, 2, 3].map((i) => at(tr.quat, 4, mf, i));
  const l = Math.hypot(...q) || 1;
  q = q.map((v) => v / l);
  return { p, q };
}

/** A low rank field or shape at motion frame mf into out: n values of c components each. */
export function rankSum(L, mf, n, c, out) {
  const r = L.r, w = new Float64Array(r);
  for (let j = 0; j < r; j++) w[j] = at(L.coef, r, mf, j);
  const B = L.basis, m = n * c;
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < r; j++) s += w[j] * B[j * m + i];
    out[i] = s;
  }
  return out;
}

/** A tracked value per frame (a glow), or null. */
export const trackSeries = (s, mf) => (s ? lerp(s[mf.a], s[mf.b], mf.k) : null);

/** A tracked body's size at motion frame mf, as many numbers as its size has. */
export function trackSize(tr, mf, n) {
  if (!tr || !tr.size) return null;
  return Array.from({ length: n }, (_, i) => at(tr.size, n, mf, i));
}

/** The per axis scale that takes a primitive drawn at size s0 to size s, in its own frame (play-scene.js primitive). */
export function sizeScale(shape, s0, s) {
  const k = (i) => (s0[i] ? s[i] / s0[i] : 1);
  switch (shape) {
    case 'sphere': return [k(0), k(0), k(0)];
    case 'box': case 'ellipsoid': return [k(0), k(1), k(2)];
    case 'plane': return [k(0), k(1), 1];
    case 'torus': return [k(0), k(0), k(1)];
    default: return [k(0), k(0), k(1)];
  }
}

/** A round number, 1, 2 or 5 times a power of ten, nearest x on a log scale. */
export function nice(x) {
  if (!(x > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  return [1, 2, 5, 10].map((m) => m * p).reduce((a, b) => (Math.abs(Math.log(b / x)) < Math.abs(Math.log(a / x)) ? b : a));
}

/**
 * How many times slower than real time a fluid's tracers move, one round factor for the whole view: the fastest
 * fluid (vmax, scene units per second) crosses about a third of the scene (radius R) in a second of playback.
 */
export const flowSlow = (vmax, R) => nice(vmax / (0.3 * R));

/** The words for a factor of real time: shown 200 times slower, in real time, shown 5 times faster. */
export function rateText(slow) {
  // two figures past a hundred, as the clock's own rate read between its frames wavers in the last digit
  const r = slow >= 1 ? slow : 1 / slow, k = r >= 100 ? +r.toPrecision(2) : Math.round(r);
  if (k === 1) return 'in real time';
  // a molecule's clock, trillions of times slower, in words
  const big = [[1e15, 'quadrillion'], [1e12, 'trillion'], [1e9, 'billion'], [1e6, 'million']].find(([b]) => r >= b);
  if (big) return `shown ${+(r / big[0]).toPrecision(3)} ${big[1]} times ${slow >= 1 ? 'slower' : 'faster'}`;
  return `shown ${k.toLocaleString('en-US')} times ${slow >= 1 ? 'slower' : 'faster'}`;
}

/** A readout's value at motion frame mf, or NaN when it has none along the motion. */
export function motionValue(r, mf) {
  const v = r.motion;
  if (!v) return NaN;
  return mf.k ? lerp(v[mf.a], v[mf.b], mf.k) : v[mf.a];
}

// ---- the steady clock (src/wobbl/viz/track.py Steady) ----
// What keeps moving at whatever value the slider stands at: a rotor turning at the asked speed, a part swinging
// through its shape at the frequency that drives it, a linkage turning with its motor. Each object has a rate per
// input frame, in turns or cycles per second of real time, and a phase the view turns on by that rate at the
// slider's value over the clock's slowed time, so moving the slider changes the motion at once and nothing jumps.

/** How many times slower than real time the steady clock runs at input frame a, b, k (its header's slow). */
export function steadySlow(st, a, b, k) {
  const s = st && st.slow;
  if (!s || !s.length) return 1;
  return s.length === 1 ? s[0] : lerp(s[a], s[b] ?? s[a], k);
}

/** An object's rate at input frame a, b, k, in cycles per second of real time. */
export const steadyRate = (sd, a, b, k) => (sd.rate.length === 1 ? sd.rate[0] : lerp(sd.rate[a], sd.rate[b] ?? sd.rate[a], k));

/** A cycle's pose at phase ph (in cycles, any real number), between its evenly spread poses, the last joining the first. */
export function cyclePose(cy, ph) {
  const n = cy.n, x = (((ph % 1) + 1) % 1) * n, a = Math.floor(x) % n, b = (a + 1) % n, k = x - Math.floor(x);
  const P = cy.pos, Q = cy.quat;
  const p = [0, 1, 2].map((i) => lerp(P[a * 3 + i], P[b * 3 + i], k));
  // the shorter way between the two quaternions, made unit again
  const dot = Q[a * 4] * Q[b * 4] + Q[a * 4 + 1] * Q[b * 4 + 1] + Q[a * 4 + 2] * Q[b * 4 + 2] + Q[a * 4 + 3] * Q[b * 4 + 3];
  const sg = dot < 0 ? -1 : 1;
  let q = [0, 1, 2, 3].map((i) => lerp(Q[a * 4 + i], sg * Q[b * 4 + i], k));
  const l = Math.hypot(...q) || 1;
  q = q.map((v) => v / l);
  return { p, q };
}
