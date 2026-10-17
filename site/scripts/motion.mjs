// What an example page needs to play a scene's motion (src/scripts/play-motion.js), worked out once at build time
// from the scene file: the motion's clock, the meter's reading and the peak marker at every motion frame, the box
// the moving things fill at each (for the camera, which keeps them and where they come to rest in view), the
// point a follow view keeps in the middle, and how much slower than real time a fluid's tracers move.
import { flowSlow, rankSum } from '../src/scripts/play-motion.js';
import { poseMatrix, skipped } from '../src/scripts/scene-file.js';

const r6 = (x) => (Number.isFinite(x) ? +x.toPrecision(6) : null);
const apply = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3], m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7],
  m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11]];
const radiusOf = (b) => Math.hypot(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) / 2;

/** A tracked object's arrays, decoded as play-motion.js decodeTrack does for the drawing. */
function trackOf(o, arr) {
  const tr = o.track;
  return {
    pos: tr.pos && arr(tr.pos), quat: tr.quat && arr(tr.quat),
    field: tr.field && { coef: arr(tr.field.coef), basis: arr(tr.field.basis), r: tr.field.rank },
    deform: tr.deform && { coef: arr(tr.deform.coef), basis: arr(tr.deform.basis), r: tr.deform.rank, ex: tr.deform.exaggerate || 1 },
  };
}

/**
 * The motion's part of a scene's page data, or null when it has none. q is the meter's reading (scripts/scenes.mjs),
 * also the readings said under it, tPos the frame position of an input value, bounds the scene's box.
 */
export function motionPlay(h, arr, { q, also, tPos, bounds, follow }) {
  const mo = h.motion;
  const flowing = h.objects.filter((o) => o.kind === 'lines' && o.speed && !skipped(o));
  const R = Math.max(radiusOf(bounds), 1e-9);
  let slow = null;
  if (flowing.length) {
    let vmax = 0;
    for (const o of flowing) { const s = arr(o.speed); for (let i = 0; i < s.length; i++) if (s[i] > vmax) vmax = s[i]; }
    slow = flowSlow(vmax, R);
  }
  if (!mo) return slow ? { flow: { slow } } : null;
  const M = mo.t.length, design = h.timeline.design;
  const out = {
    t: mo.t, play: mo.play, label: mo.label || 'time', unit: mo.unit || 's', note: mo.note || '',
    at: r6(design !== undefined ? tPos(design) : h.timeline.frames - 1),
  };
  if (slow) out.flow = { slow };
  // every tracked solid the run is about, at each motion frame: its points where they are and its field's peak
  const solids = h.objects.map((o, oi) => ({ o, oi })).filter(({ o }) => o.track && !skipped(o) && (o.kind === 'mesh' || o.kind === 'body'));
  const lo = Array.from({ length: M }, () => [Infinity, Infinity, Infinity]), hi = Array.from({ length: M }, () => [-Infinity, -Infinity, -Infinity]);
  const peaks = new Array(M).fill(null), fpeak = new Array(M).fill(NaN), centers = new Array(M).fill(null);
  const grp = h.magnify && h.magnify.groups && h.magnify.groups[0] ? h.magnify.groups[0].objects : [];
  for (const { o, oi } of solids) {
    const tr = trackOf(o, arr);
    const P = o.kind === 'mesh' ? arr(o.pos) : null;
    const n = P ? P.length / 3 : 0;
    const u = tr.deform ? new Float32Array(n * 3) : null, v = tr.field ? new Float32Array(n) : null;
    // the corners of a body, or a thinned set of a mesh's points for its box
    const local = P ? Array.from({ length: Math.min(n, 400) }, (_, j) => Math.floor((j * n) / Math.min(n, 400))) : null;
    const half = o.kind === 'body' ? (o.shape === 'box' ? o.size : [o.size[0], o.size[0], o.size[1] ?? o.size[0]]) : null;
    for (let f = 0; f < M; f++) {
      const mf = { a: f, b: f, k: 0 };
      const pose = tr.pos ? { p: [0, 1, 2].map((i) => tr.pos[f * 3 + i]), q: [0, 1, 2, 3].map((i) => tr.quat[f * 4 + i]) } : null;
      const m = pose ? poseMatrix(pose.q.map((x) => x / (Math.hypot(...pose.q) || 1)), pose.p) : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
      if (u) rankSum(tr.deform, mf, n, 3, u);
      if (v) rankSum(tr.field, mf, n, 1, v);
      const grow = (p) => { for (let k = 0; k < 3; k++) { lo[f][k] = Math.min(lo[f][k], p[k]); hi[f][k] = Math.max(hi[f][k], p[k]); } };
      let c = null;
      if (P) {
        for (const j of local) grow(apply(m, [P[j * 3] + (u ? u[j * 3] : 0), P[j * 3 + 1] + (u ? u[j * 3 + 1] : 0), P[j * 3 + 2] + (u ? u[j * 3 + 2] : 0)]));
      } else {
        for (let cc = 0; cc < 8; cc++) grow(apply(m, [cc & 1 ? half[0] : -half[0], cc & 2 ? half[1] : -half[1], cc & 4 ? half[2] : -half[2]]));
      }
      if (grp.includes(oi) && pose) c = pose.p;
      if (c && !centers[f]) centers[f] = c.map(r6);
      // the field's peak on what moves, with its place drawn at its true and its exaggerated shape
      if (v && o.track.field.f === 0) {
        let best = -Infinity, bi = -1;
        for (let i = 0; i < n; i++) if (Number.isFinite(v[i]) && v[i] > best) { best = v[i]; bi = i; }
        if (bi >= 0 && !(best <= fpeak[f])) {
          fpeak[f] = best;
          const b = [P[bi * 3], P[bi * 3 + 1], P[bi * 3 + 2]], ex = tr.deform ? tr.deform.ex : 1;
          const du = u ? [u[bi * 3], u[bi * 3 + 1], u[bi * 3 + 2]] : [0, 0, 0];
          peaks[f] = { v: r6(best), p0: apply(m, b.map((x, k) => x + du[k])).map(r6), pe: apply(m, b.map((x, k) => x + du[k] * ex)).map(r6) };
        }
      }
    }
  }
  if (solids.length) out.box = lo.map((l, f) => [l.map(r6), hi[f].map(r6)]);
  if (follow && centers.every(Boolean)) out.follow = centers;
  if (peaks.some(Boolean)) out.peaks = peaks;
  // the meter's reading along the motion: the field's peak on what moves, or the readout's own motion values
  if (q) {
    const r = h.readouts.find((x) => x.name === q.name && x.motion);
    if (r) out.q = r.motion.map(r6);
    else if (q.field && peaks.some(Boolean)) out.q = fpeak.map(r6);
  }
  out.also = (also || []).map((a) => { const r = h.readouts.find((x) => x.name === a.name); return r && r.motion ? r.motion.map(r6) : null; });
  out.readouts = h.readouts.filter((r) => r.motion).map((r) => ({ name: r.name, unit: r.unit || '', digits: r.digits ?? 3, values: r.motion.map(r6) }));
  return out;
}
