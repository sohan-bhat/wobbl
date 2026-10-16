// A run's scene.wobbl, read the way Wobbl's own viewer reads it (src/wobbl/viz/viewer/core.js), for the
// example pages' playback. It runs in the drawing worker, which lays the scene out for the GPU, and at build
// time in scripts/scenes.mjs, which works out once what the page shows under the view: where the peak of
// the scene's main field sits at every saved frame, the box the camera holds and the readings. Nothing here
// touches the page.

const T = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, f32: Float32Array };

/** The header and an array reader for an unzipped scene file. */
export function readScene(raw) {
  if (String.fromCharCode(raw[0], raw[1], raw[2], raw[3]) !== 'WBSC') throw new Error('this is not a Wobbl scene');
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = dv.getUint32(8, true);
  const h = JSON.parse(new TextDecoder().decode(raw.subarray(12, 12 + n)));
  let at = 12 + n;
  at += (8 - (at % 8)) % 8;
  const blob = raw.slice(at);
  const arr = (d) => {
    if (!d) return null;
    if (d.b === undefined) return d;
    const b = h.buffers[d.b], q = new T[b.t](blob.buffer.slice(b.o, b.o + b.n));
    if (!d.lo) return q;
    const M = b.t === 'u8' ? 255 : 65535, top = d.nan ? M - 1 : M, c = d.c, o = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) {
      const k = i % c;
      o[i] = d.nan && q[i] === M ? NaN : d.lo[k] + (q[i] / top) * (d.hi[k] - d.lo[k]);
    }
    return o;
  };
  return { h, arr };
}

/** A number to the significant digits a reading asks for, as the viewer prints it: 3,439 or 5.72. */
export function sig(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(Math.max(d - 1, 1));
  const dec = a === 0 ? 0 : Math.max(0, d - 1 - Math.floor(Math.log10(a)));
  return v.toLocaleString('en-US', { minimumFractionDigits: Math.min(dec, 6), maximumFractionDigits: Math.min(dec, 6) });
}

/** A value along an input, to four significant figures without trailing zeros: 10 mph, 6.25 mph, 2,000 N. */
export function num(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '';
  return Number(v.toPrecision(4)).toLocaleString('en-US', { maximumFractionDigits: 6 });
}

// ---- colors, as the viewer's maps
const hex = (s) => { const v = parseInt(s.slice(1), 16); return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; };
const STOPS = {
  viridis: '440154 482878 3e4989 31688e 26828e 1f9e89 35b779 6ece58 b5de2b fde725',
  inferno: '000004 1b0c41 4a0c6b 781c6d a52c60 cf4446 ed6925 fb9b06 f7d13d fcffa4',
  coolwarm: '3b4cc0 6889ee 9abbff c9d8ef edd1c2 f7a889 e26952 b40426',
  blues: '0b1a33 0f3b6e 1663a6 2f8fd1 6bb8e8 b6e0f7',
  gray: '303338 e6e6e6',
  heat: '9a9488 c9a05a e8742a f2401f',
  // low in blue, high in red, as Wobbl's own plots draw a level or a pressure (matplotlib's RdBu_r)
  RdBu: '053061 2166ac 4393c3 92c5de d1e5f0 f7f7f7 fddbc7 f4a582 d6604d b2182b 67001f',
};
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function turbo(x) {
  const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)];
}
const LUTS = {};
/** 256 colors of a named map as bytes, r g b after each other. */
export function lut(name) {
  if (LUTS[name]) return LUTS[name];
  const o = new Uint8Array(256 * 3), s = (STOPS[name] || '').split(' ').filter(Boolean).map((x) => hex(`#${x}`));
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let c;
    if (!s.length) c = turbo(x);
    else {
      const f = x * (s.length - 1), a = Math.min(Math.floor(f), s.length - 2), k = f - a;
      c = [0, 1, 2].map((j) => s[a][j] + (s[a + 1][j] - s[a][j]) * k);
    }
    o[i * 3] = Math.round(c[0] * 255); o[i * 3 + 1] = Math.round(c[1] * 255); o[i * 3 + 2] = Math.round(c[2] * 255);
  }
  return (LUTS[name] = o);
}
export const colorHex = hex;

// ---- playback position: frame a, the next frame b and the share k of the way between them
const lerp = (a, b, k) => a + (b - a) * k;
export const at2 = (arr, n, a, b, k, i) => (k ? lerp(arr[a * n + i], arr[b * n + i], k) : arr[a * n + i]);
export const series = (s, a, b, k) => (s.length === 1 ? s[0] : lerp(s[a], s[b], k));
export const shownAt = (w, t) => !w || ((w[0] === null || t >= w[0] - 1e-9) && (w[1] === null || t <= w[1] + 1e-9));

/** A rigid pose as a 3x4 row major matrix from a quaternion (w, x, y, z) and a position. */
export function poseMatrix(q, p) {
  const [w, x, y, z] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), p[0],
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), p[1],
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), p[2]];
}
const apply = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3], m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7],
  m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11]];

/** The pose of an object at frame a, b, k, or null when it has none. */
export function poseAt(pp, pq, a, b, k) {
  if (!pp || !pq) return null;
  const p = pp.length > 3 ? [0, 1, 2].map((i) => at2(pp, 3, a, b, k, i)) : [pp[0], pp[1], pp[2]];
  let q = pq.length > 4 ? [0, 1, 2, 3].map((i) => at2(pq, 4, a, b, k, i)) : [pq[0], pq[1], pq[2], pq[3]];
  const l = Math.hypot(...q) || 1;
  q = q.map((v) => v / l);
  return { p, q };
}

/** The magnification an object is drawn at, and the point it grows about (null: its own pose). */
export function magnifyOf(h, index) {
  const m = h.magnify;
  if (!m) return null;
  for (const g of m.groups || []) if (g.objects.includes(index)) return { factor: g.factor ?? m.factor, anchor: g.anchor ?? null };
  return null;
}

/** Objects the site leaves out: guides drawn only to carry the viewer's text labels. */
export const skipped = (o) => !!o.guide;

/**
 * The world position of point local (in the object's own frame) of object o at a frame, drawn with the
 * bending exaggerated or not, and magnified or not.
 */
function placeAt(h, oi, o, pose, local, mag) {
  if (!pose) return local;
  const m = mag && magnifyOf(h, oi);
  if (!m) return apply(poseMatrix(pose.q, pose.p), local);
  const s = m.factor, c = m.anchor ? m.anchor.map((x, i) => x + s * (pose.p[i] - x)) : pose.p;
  return apply(poseMatrix(pose.q, c), local.map((x) => x * s));
}

/**
 * Where the main color field (field 0) peaks at each saved frame, on what is drawn then: its value and the
 * point, without and with the bending drawn larger. Null when the scene has no field.
 */
export function fieldPeaks(h, arr) {
  if (!h.fields.length) return null;
  const F = h.timeline.frames, T0 = h.timeline.t, out = [];
  const objs = h.objects.map((o) => ({
    o, P: o.pos ? arr(o.pos) : null, fv: o.field ? arr(o.field.values || o.field.frames) : null,
    du: o.deform ? arr(o.deform.values || o.deform.frames) : null, pp: o.pose ? arr(o.pose.pos) : o.kind === 'body' ? arr(o.pos) : null,
    pq: o.pose ? arr(o.pose.quat) : o.kind === 'body' ? arr(o.quat) : null, rv: o.reveal ? arr(o.reveal.values) : null,
  }));
  for (let f = 0; f < F; f++) {
    const t = T0[f];
    let best = null;
    objs.forEach((ob, oi) => {
      const o = ob.o;
      if (!o.field || o.field.f !== 0 || skipped(o) || !shownAt(o.when, t)) return;
      const pose = poseAt(ob.pp, ob.pq, f, f, 0);
      const consider = (v, at) => { if (Number.isFinite(v) && (!best || v > best.v)) best = { v, ...at() }; };
      if (o.kind === 'mesh') {
        const n = ob.P.length / 3, th = o.reveal ? o.reveal.at[f] : Infinity, sc = o.field.scale;
        for (let i = 0; i < n; i++) {
          if (ob.rv && ob.rv[i] > th) continue;
          const v = sc ? ob.fv[i] * sc[f] : o.field.frames ? ob.fv[f * n + i] : ob.fv[i];
          consider(v, () => {
            const u = [0, 1, 2].map((k) => {
              if (!ob.du) return 0;
              const ds = o.deform.scale;
              return ds ? ob.du[i * 3 + k] * ds[f] : ob.du[f * n * 3 + i * 3 + k];
            });
            const ex = o.deform ? o.deform.exaggerate || 1 : 1;
            const base = [ob.P[i * 3], ob.P[i * 3 + 1], ob.P[i * 3 + 2]];
            return {
              p0: placeAt(h, oi, o, pose, base.map((x, k) => x + u[k]), false),
              pe: placeAt(h, oi, o, pose, base.map((x, k) => x + u[k] * ex), true),
            };
          });
        }
      } else if (o.kind === 'body') {
        const v = series(o.field.values, f, f, 0);
        consider(v, () => ({ p0: pose.p, pe: placeAt(h, oi, o, pose, [0, 0, 0], true) }));
      } else if (o.kind === 'lines' && ob.fv) {
        for (let i = 0; i < ob.fv.length; i++) consider(ob.fv[i], () => { const p = [ob.P[i * 3], ob.P[i * 3 + 1], ob.P[i * 3 + 2]]; return { p0: p, pe: p }; });
      } else if (o.kind === 'points' && ob.fv) {
        const sh = o.pos.shape, n = sh[sh.length - 2], many = ob.P.length > n * 3;
        for (let i = 0; i < n; i++) {
          const v = o.field.frames ? ob.fv[f * n + i] : ob.fv[i];
          consider(v, () => { const p = [0, 1, 2].map((k) => (many ? ob.P[f * n * 3 + i * 3 + k] : ob.P[i * 3 + k])); return { p0: p, pe: p }; });
        }
      }
    });
    out.push(best ? { v: +best.v.toPrecision(6), p0: best.p0.map((x) => +x.toPrecision(6)), pe: best.pe.map((x) => +x.toPrecision(6)) } : null);
  }
  return out;
}

/**
 * The box the camera holds: everything the run is about over every frame, magnified, with the props (a wall,
 * a bench, the ground) left to run out of the view as a table runs out of a photograph. A scene of props
 * only is framed whole.
 */
export function sceneBounds(h, arr) {
  const b = boundsOf(h, arr, (o) => !o.prop);
  return b || boundsOf(h, arr, () => true) || { lo: [-1, -1, -1], hi: [1, 1, 1] };
}

/** The box of the objects keep(o, index) passes, magnified or not, over every frame or at frame only. */
function boundsOf(h, arr, keep, mag = true, frame = -1) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const grow = (p, r = 0) => {
    for (let k = 0; k < 3; k++) if (Number.isFinite(p[k])) { lo[k] = Math.min(lo[k], p[k] - r); hi[k] = Math.max(hi[k], p[k] + r); }
  };
  const F = h.timeline.frames;
  h.objects.forEach((o, oi) => {
    if (o.kind === 'ground' || skipped(o) || !keep(o, oi)) return;
    if (frame >= 0 && !shownAt(o.when, h.timeline.t[frame])) return;
    if (o.kind === 'mesh' || o.kind === 'body') {
      // the object's own box, then that box's corners at every pose it takes
      let blo, bhi;
      if (o.kind === 'mesh') {
        const P = arr(o.pos);
        if (!o.pose) { for (let i = 0; i < P.length; i += 3) grow([P[i], P[i + 1], P[i + 2]]); return; }
        blo = [Infinity, Infinity, Infinity]; bhi = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < P.length; i++) { const k = i % 3; blo[k] = Math.min(blo[k], P[i]); bhi[k] = Math.max(bhi[k], P[i]); }
      } else {
        const z = o.size, r = z[0], hl = z[1] ?? r;
        const half = { box: z, sphere: [r, r, r], ellipsoid: z, plane: [z[0], z[1], 0], torus: [z[0] + z[1], z[0] + z[1], z[1]],
          capsule: [r, r, hl + r] }[o.shape] ?? [r, r, hl];
        blo = half.map((x) => -x); bhi = half;
      }
      const box = [0, 1, 2, 3, 4, 5, 6, 7].map((c) => [c & 1 ? bhi[0] : blo[0], c & 2 ? bhi[1] : blo[1], c & 4 ? bhi[2] : blo[2]]);
      const pp = arr(o.pose ? o.pose.pos : o.pos), pq = arr(o.pose ? o.pose.quat : o.quat);
      const many = pp.length > 3 || pq.length > 4;
      for (let f = 0; f < (many ? F : 1); f++) {
        const at = frame >= 0 && many ? frame : f;
        const pose = poseAt(pp, pq, at, at, 0);
        for (const c of box) grow(placeAt(h, oi, o, pose, c, mag));
      }
    } else if (o.kind === 'lines' || o.kind === 'points' || o.kind === 'atoms') {
      const P = arr(o.pos);
      for (let i = 0; i < P.length; i += 3) grow([P[i], P[i + 1], P[i + 2]]);
    } else if (o.kind === 'arrows') {
      const O = arr(o.origin), V = arr(o.vec), sc = o.scale ? Math.max(...o.scale) : 1, n = Math.min(O.length, V.length);
      for (let i = 0; i < n; i += 3) { grow([O[i], O[i + 1], O[i + 2]]); grow([O[i] + V[i] * sc, O[i + 1] + V[i + 1] * sc, O[i + 2] + V[i + 2] * sc]); }
    }
  });
  if (!Number.isFinite(lo[0])) return null;
  for (let k = 0; k < 3; k++) if (!Number.isFinite(lo[k])) { lo[k] = -1; hi[k] = 1; }
  return { lo, hi };
}

/**
 * Small moving things a scene did not magnify itself, such as a rock falling 300 m, found and grown so they
 * show at the scene's size: every posed mesh or body the run is about whose own size is under half a percent
 * of the scene's, drawn about its own pose so it reaches about eight percent. The page says so beside a
 * toggle that puts them back at their true size. Returns the scene's magnify entry, or null.
 */
export function autoMagnify(h, arr, bounds) {
  if (h.magnify) return h.magnify;
  const { lo, hi } = bounds, R = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  const small = [];
  let size = 0, name = '';
  h.objects.forEach((o, i) => {
    if (o.prop || skipped(o) || !(o.pose || (o.kind === 'body' && o.pos && o.pos.b !== undefined))) return;
    let ext = 0;
    if (o.kind === 'mesh') { const P = arr(o.pos); for (let j = 0; j < P.length; j += 3) ext = Math.max(ext, Math.hypot(P[j], P[j + 1], P[j + 2])); }
    else if (o.kind === 'body') ext = Math.max(...o.size);
    else return;
    if (ext > 0 && ext < R * 0.005) { small.push(i); if (ext > size) { size = ext; name = o.name || ''; } }
  });
  if (!small.length) return null;
  const raw = (R * 0.08) / size, p = 10 ** Math.floor(Math.log10(raw));
  const factor = [1, 2, 5, 10].map((x) => x * p).reduce((a, b) => (Math.abs(b - raw) < Math.abs(a - raw) ? b : a));
  const label = name ? name[0].toUpperCase() + name.slice(1) : 'Drawn';
  return { factor, label, groups: [{ objects: small, anchor: null }] };
}

const SOLID = new Set(['mesh', 'body', 'points', 'atoms']);
const radiusOf = (b) => (b ? Math.hypot(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) / 2 : 0);

/**
 * What the camera frames, beyond the box sceneBounds gives. subject is the box of the solid things the run is
 * about (no props, no lines or arrows), at their true size, when it is well inside that box: a wiper's linkage
 * inside the arcs its blade sweeps, framed up close with the glass and the arcs as context. follow is set when
 * the scene had to magnify small moving things (a rocket on its 3 km flight, a rock falling 300 m): their
 * center at every saved frame and their radius, at their true size, so the main view can follow them up close
 * as NASA's Eyes follows a craft, with the whole scene in an inset. dir is the side the scene looks at them
 * from, from a camera preset that follows one of them when it has one.
 */
export function sceneFraming(h, arr, bounds = sceneBounds(h, arr), frame = 0) {
  const all = radiusOf(bounds);
  // what holds the subject and is not much bigger than it (a crucible, a mold box) is framed with it, where a
  // bench or the ground under it runs out of the view
  const mid = [0, 1, 2].map((k) => (bounds.lo[k] + bounds.hi[k]) / 2);
  let hold = null;
  h.objects.forEach((o, oi) => {
    if (!o.prop || skipped(o) || !(o.kind === 'mesh' || o.kind === 'body')) return;
    const b = boundsOf(h, arr, (x, i) => i === oi, true);
    if (!b || radiusOf(b) > 1.5 * all || ![0, 1, 2].every((k) => b.lo[k] <= mid[k] && b.hi[k] >= mid[k])) return;
    hold = hold || { lo: [...bounds.lo], hi: [...bounds.hi] };
    for (let k = 0; k < 3; k++) { hold.lo[k] = Math.min(hold.lo[k], b.lo[k]); hold.hi[k] = Math.max(hold.hi[k], b.hi[k]); }
  });
  let subject = boundsOf(h, arr, (o) => !o.prop && SOLID.has(o.kind), false);
  if (subject && radiusOf(subject) > 0.6 * all) subject = null;
  if (subject) hold = null;
  // the corners of each solid the subject is made of, where they stand at the frame the page opens on, so
  // the camera fits the subject itself and not the box round it
  let points = null;
  if (subject) {
    points = [];
    h.objects.forEach((o, oi) => {
      if (o.prop || !SOLID.has(o.kind) || skipped(o)) return;
      const b = boundsOf(h, arr, (x, i) => i === oi, false, frame);
      if (!b) return;
      for (let c = 0; c < 8; c++) points.push([c & 1 ? b.hi[0] : b.lo[0], c & 2 ? b.hi[1] : b.lo[1], c & 4 ? b.hi[2] : b.lo[2]].map((x) => +x.toPrecision(6)));
    });
    if (!points.length) points = null;
  }
  let follow = null;
  const g = h.magnify && h.magnify.groups && h.magnify.groups[0];
  if (g && g.objects.length) {
    const F = h.timeline.frames, c = [];
    // the solid members only: a see through copy left at the start (a rock's release point) is context
    const solid = g.objects.filter((oi) => (h.objects[oi].opacity ?? 1) >= 1);
    const members = solid.length ? solid : g.objects;
    let r = 0, long = 1;
    for (let f = 0; f < F; f++) {
      const b = boundsOf(h, arr, (o, oi) => members.includes(oi), false, f);
      if (!b) { c.push(c[c.length - 1] || [0, 0, 0]); continue; }
      c.push([0, 1, 2].map((k) => +((b.lo[k] + b.hi[k]) / 2).toPrecision(7)));
      r = Math.max(r, radiusOf(b));
      const d = [0, 1, 2].map((k) => b.hi[k] - b.lo[k]).sort((x, y) => y - x);
      long = Math.max(long, d[0] / Math.max(d[1], 1e-12));
    }
    const preset = ((h.camera && h.camera.presets) || []).find((p) => p.follow !== undefined && g.objects.includes(p.follow) && p.dir);
    // a long thin subject (a rocket) fills the view along its length; a compact one (a rock) is shown with room
    // round it for the ground and the path it came down
    const fill = long > 3 ? 1.15 : 2.4;
    if (r > 0) follow = { c, r: +r.toPrecision(6), fill, dir: preset ? preset.dir.map((x) => +x.toPrecision(6)) : null };
  }
  return { subject, points, follow, hold };
}
