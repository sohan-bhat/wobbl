// What an example page needs to play a scene's motion (src/scripts/play-motion.js), worked out once at build time
// from the scene file: the motion's clock, the meter's reading and the peak marker at every motion frame, the box
// the moving things fill at each (for the camera, which keeps them and where they come to rest in view), the
// point a follow view keeps in the middle, and how much slower than real time a fluid's tracers move.
import { flowSlow, motionOf, rankSum } from '../src/scripts/play-motion.js';
import { poseMatrix, shownAt, skipped } from '../src/scripts/scene-file.js';

const r6 = (x) => (Number.isFinite(x) ? +x.toPrecision(6) : null);
const apply = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3], m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7],
  m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11]];
const radiusOf = (b) => Math.hypot(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) / 2;

/** A track's arrays, decoded as play-motion.js decodeTrack does for the drawing. */
function trackOf(tr, arr) {
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
  const design = h.timeline.design;
  const flowing = h.objects.filter((o) => o.kind === 'lines' && o.speed && !skipped(o));
  const R = Math.max(radiusOf(bounds), 1e-9);
  let slow = null;
  if (flowing.length) {
    let vmax = 0;
    for (const o of flowing) { const s = arr(o.speed); for (let i = 0; i < s.length; i++) if (s[i] > vmax) vmax = s[i]; }
    slow = flowSlow(vmax, R);
  }
  // a line's flow (a glow along a wire carrying current) moves on the view's own clock too, with nothing slowed
  const glows = h.objects.some((o) => o.kind === 'lines' && o.flow && !o.speed && !skipped(o));
  // what keeps moving at the slider's value on the scene's steady clock (play-motion.js), and how much slower than
  // real time that clock runs at each input frame
  const steady = h.steady && h.objects.some((o) => o.steady && !skipped(o)) ? { slow: h.steady.slow, note: h.steady.note || '' } : null;
  // the rate the part colored by the main field swings at, so the peak's marker swings with it on the page
  const sw = steady && h.objects.find((o) => o.steady && o.steady.swing && o.field && o.field.f === 0 && !skipped(o));
  if (sw) steady.swing = sw.steady.rate;
  if (!h.motion) {
    const out = slow ? { flow: { slow } } : glows ? { flow: { glows: true } } : null;
    return steady ? { ...out, steady } : out;
  }
  const out = variantPlay(h, arr, h.motion.asked, { q, also, tPos, follow });
  if (slow) out.flow = { slow };
  else if (glows) out.flow = { glows: true };
  if (steady) out.steady = steady;
  // a sweep that kept the event at several values: each value's own motion, worked out the same way, for the page
  // to fetch when it plays one (scripts/scenes.mjs writes them beside the scene file)
  const vs = h.motion.variants;
  if (vs && vs.length > 1) {
    out.variants = vs.map((v, i) => ({ value: v.at ?? design, at: r6(tPos(v.at ?? design)), asked: i === (h.motion.asked ?? 0) }));
    out.variantData = vs.map((_, i) => (i === (h.motion.asked ?? 0) ? null : variantPlay(h, arr, i, { q, also, tPos, follow })));
  }
  return out;
}

/** The page data of motion variant vi (play-motion.js motionOf): its clock, its readings and what the camera keeps. */
function variantPlay(h, arr, vi, { q, also, tPos, follow }) {
  const mo = motionOf(h, vi), M = mo.t.length, design = h.timeline.design;
  const trk = (o) => (o.tracks && h.motion.variants ? o.tracks[vi] : o.track);
  const rm = (r) => (r.motions && h.motion.variants ? r.motions[vi] : r.motion);
  const out = {
    t: mo.t, play: mo.play, label: mo.label || 'time', unit: mo.unit || 's', note: mo.note || '',
    at: r6(mo.at !== undefined ? tPos(mo.at) : design !== undefined ? tPos(design) : h.timeline.frames - 1),
  };
  // every tracked solid the run is about, at each motion frame: its points where they are and its field's peak
  const solids = h.objects.map((o, oi) => ({ o, oi })).filter(({ o }) => trk(o) && !skipped(o) && (o.kind === 'mesh' || o.kind === 'body'));
  const lo = Array.from({ length: M }, () => [Infinity, Infinity, Infinity]), hi = Array.from({ length: M }, () => [-Infinity, -Infinity, -Infinity]);
  const peaks = new Array(M).fill(null), fpeak = new Array(M).fill(NaN);
  // what a follow view keeps in frame at each motion frame: the followed group and anything with a window of its own
  // over the motion while it shows (a parachute over the rocket's nose)
  const gp = new Array(M).fill(null);
  const flo = Array.from({ length: M }, () => [Infinity, Infinity, Infinity]), fhi = Array.from({ length: M }, () => [-Infinity, -Infinity, -Infinity]);
  const grp = h.magnify && h.magnify.groups && h.magnify.groups[0] ? h.magnify.groups[0].objects : [];
  for (const { o, oi } of solids) {
    const tr = trackOf(trk(o), arr);
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
      if (pose && grp[0] === oi) gp[f] = pose.p;
      const kept = follow && (grp.includes(oi) || trk(o).when) && shownAt(trk(o).when, mo.t[f]);
      const grow = (p) => {
        for (let k = 0; k < 3; k++) {
          lo[f][k] = Math.min(lo[f][k], p[k]); hi[f][k] = Math.max(hi[f][k], p[k]);
          if (kept) { flo[f][k] = Math.min(flo[f][k], p[k]); fhi[f][k] = Math.max(fhi[f][k], p[k]); }
        }
      };
      if (P) {
        for (const j of local) grow(apply(m, [P[j * 3] + (u ? u[j * 3] : 0), P[j * 3 + 1] + (u ? u[j * 3 + 1] : 0), P[j * 3 + 2] + (u ? u[j * 3 + 2] : 0)]));
      } else {
        for (let cc = 0; cc < 8; cc++) grow(apply(m, [cc & 1 ? half[0] : -half[0], cc & 2 ? half[1] : -half[1], cc & 4 ? half[2] : -half[2]]));
      }
      // the field's peak on what moves, with its place drawn at its true and its exaggerated shape
      if (v && trk(o).field.f === 0) {
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
  // a field's peak under the low rank sum's own error is that error, not a reading: an unloaded part reads 0
  const errs = solids.map(({ o }) => trk(o).field && trk(o).field.f === 0 ? trk(o).field.err ?? 0 : 0);
  const floor = Math.max(0, ...errs) * Math.max(0, ...fpeak.filter(Number.isFinite));
  for (let f = 0; f < M; f++) {
    if (fpeak[f] <= floor) { fpeak[f] = 0; peaks[f] = null; }
  }
  // whether the motion moves or recolors what the main field colors, so the input's peak no longer marks it
  if (solids.some(({ o }) => o.field && o.field.f === 0 && (trk(o).pos || trk(o).value || trk(o).deform))) out.fieldMoves = true;
  if (solids.length) out.box =lo.map((l, f) => [l.map(r6), hi[f].map(r6)]);
  if (follow && flo.every((l) => Number.isFinite(l[0]))) {
    // the middle of what the follow view keeps and how much larger than the followed group alone it is, eased over
    // half a second of playback either side, so the camera backs off as a parachute opens instead of jumping: the
    // middle's offset from the subject is averaged and the size is the largest it is about to be
    const W = 0.5, play = mo.play;
    const off = flo.map((l, f) => l.map((x, k) => (x + fhi[f][k]) / 2 - (gp[f] ? gp[f][k] : 0)));
    // (with a little room past what it keeps, as the follow view sits its subject a little above the middle)
    const K = flo.map((l, f) => { const g = Math.hypot(...l.map((x, k) => fhi[f][k] - x)) / 2 / (follow.r || 1); return g > 1.05 ? g * 1.15 : 1; });
    out.follow = [];
    out.followK = [];
    for (let f = 0; f < M; f++) {
      let ws = 0, kk = K[f];
      const o = [0, 0, 0];
      for (let g = 0; g < M; g++) {
        const d = Math.abs(play[g] - play[f]);
        if (d > W) continue;
        const w = 1 - d / W;
        ws += w;
        for (let k = 0; k < 3; k++) o[k] += w * off[g][k];
        kk = Math.max(kk, 1 + (K[g] - 1) * w);
      }
      out.follow.push(o.map((x, k) => r6(x / ws + (gp[f] ? gp[f][k] : 0))));
      out.followK.push(r6(kk));
    }
  }
  if (peaks.some(Boolean)) out.peaks = peaks;
  // the meter's reading along the motion: the field's peak on what moves, or the readout's own motion values
  if (q) {
    const r = h.readouts.find((x) => (x.name === q.name || x.name === q.name.replace(/^peak /, '')) && rm(x));
    if (r) out.q = rm(r).map(r6);
    else if (q.field && peaks.some(Boolean)) out.q = fpeak.map(r6);
    else if (q.field) {
      // a path colored by the field that traces itself over the motion: the field's peak on what is drawn so far
      const paths = h.objects.filter((o) => o.kind === 'lines' && trk(o) && trk(o).drawn && o.field && o.field.f === 0 && !skipped(o));
      if (paths.length) {
        out.q = mo.t.map((t) => {
          let best = -Infinity;
          for (const o of paths) {
            const dr = arr(trk(o).drawn), v = arr(o.field.values);
            for (let i = 0; i < dr.length; i++) if (dr[i] <= t + 1e-9 && v[i] > best) best = v[i];
          }
          return Number.isFinite(best) ? r6(best) : null;
        });
      }
    }
  }
  out.also = (also || []).map((a) => { const r = h.readouts.find((x) => x.name === a.name); return r && rm(r) ? rm(r).map(r6) : null; });
  // what only the motion reads, said in place of the readings along the input while it plays
  if (mo.readouts) out.live = mo.readouts.map((r) => ({ name: r.name, unit: r.unit || '', digits: r.digits ?? 3, values: r.values.map(r6) }));
  out.readouts = h.readouts.filter((r) => rm(r)).map((r) => ({ name: r.name, unit: r.unit || '', digits: r.digits ?? 3, values: rm(r).map(r6) }));
  return out;
}
