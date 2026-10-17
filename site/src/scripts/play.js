// A run's playback file, decoded and laid out for the GPU. It runs in play-worker.js, off the page's
// main thread. The files come from scripts/sitegen/export_playback.py, in the coding the landing page's
// part uses (scripts/export_site_part.py), so the byte planes, the residuals and the smooth shading are
// part.js's own.
import { creases, integers, planes, shade } from './part.js';
import { readScene, skipped } from './scene-file.js';
import { decodeTrack } from './play-motion.js';

/** Nodes and triangles of the surface, and a reader for the integer fields stored on its nodes. */
function surface(run, bytes) {
  const { triangles: T, corners: nc, nodes: n, linear, bounds_min: lo, bounds_max: hi } = run.mesh;
  const L = run.layout;
  const codes = planes(bytes, L.index.offset, T * 3);
  const ct = new Int32Array(T * 3);
  const pa = new Int32Array(nc).fill(-1), pb = new Int32Array(nc).fill(-1);
  let hw = 0;
  for (let t = 0, j = 0; t < T; t++) {
    for (let k = 0; k < 3; k++, j++) {
      let v;
      if (codes[j] === 0) {
        v = hw++;
        if (k > 0) pa[v] = ct[t * 3];
        if (k > 1) pb[v] = ct[t * 3 + 1];
      } else {
        v = hw - codes[j];
      }
      ct[j] = v;
    }
  }
  if (hw !== nc) throw new Error('playback data: corner count does not match');
  let tri = ct, ends = new Int32Array(0);
  if (!linear) {
    // midside nodes, numbered as the walk first crosses each edge, and each curved triangle split in four
    const mid = new Int32Array(T * 3), seen = new Map();
    ends = new Int32Array((n - nc) * 2);
    let m = nc;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < 3; k++) {
        const a = ct[t * 3 + k], b = ct[t * 3 + ((k + 1) % 3)];
        const lo2 = a < b ? a : b, hi2 = a < b ? b : a, key = lo2 * nc + hi2;
        let id = seen.get(key);
        if (id === undefined) {
          id = m++;
          seen.set(key, id);
          ends[(id - nc) * 2] = lo2;
          ends[(id - nc) * 2 + 1] = hi2;
        }
        mid[t * 3 + k] = id;
      }
    }
    if (m !== n) throw new Error('playback data: node count does not match');
    tri = new Uint32Array(T * 12);
    for (let t = 0; t < T; t++) {
      const a = ct[t * 3], b = ct[t * 3 + 1], c = ct[t * 3 + 2];
      const ab = mid[t * 3], bc = mid[t * 3 + 1], ca = mid[t * 3 + 2];
      tri.set([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], t * 12);
    }
  }
  const field = (key) => {
    const ch = L[key].channels;
    return integers(planes(bytes, L[key].offset, ch * n), ch, n, nc, pa, pb, ends);
  };
  const qp = field('position');
  const pos = new Float32Array(n * 3);
  for (let k = 0; k < 3; k++) {
    const s = (hi[k] - lo[k]) / 65535;
    for (let i = 0; i < n; i++) pos[i * 3 + k] = lo[k] + qp[k * n + i] * s;
  }
  return { n, pos, tri: tri instanceof Uint32Array ? tri : Uint32Array.from(tri), field };
}

/**
 * Everything the GPU needs for one run, one entry per drawn vertex.
 *   position  float x3   mm
 *   normal    int8 x3
 * A strength run adds, at the run's own load:
 *   value     float      the run's stress measure in MPa
 *   disp      float x3   displacement in mm
 *   excl      uint8      255 in a zone Wobbl excludes from the answer
 * A heat run adds frames: the temperature above the air at every saved time, frame after frame, and,
 * when it was cut open, the cut faces with their own frames (see section).
 */
export function decodePlay(run, buf) {
  const bytes = new Uint8Array(buf);
  const { n, pos, tri, field } = surface(run, bytes);
  const { index, vNode, normal } = shade(pos, tri, n);
  const nv = vNode.length;
  const position = new Float32Array(nv * 3), nrm = new Int8Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const i = vNode[v];
    for (let k = 0; k < 3; k++) {
      position[v * 3 + k] = pos[i * 3 + k];
      nrm[v * 3 + k] = Math.round(normal[v * 3 + k] * 127);
    }
  }
  const ed = creases(tri, index, n);
  const arrays = { position, normal: nrm, index: nv < 65536 ? Uint16Array.from(index) : index, edges: nv < 65536 ? Uint16Array.from(ed) : Uint32Array.from(ed) };
  const L = run.layout;
  if (run.kind === 'strength') {
    const qv = field('value'), qu = field('disp');
    const excl = new Uint8Array(buf, L.excluded.offset, n);
    const vs = L.value.step, us = L.disp.step;
    arrays.value = new Float32Array(nv);
    arrays.disp = new Float32Array(nv * 3);
    arrays.excl = new Uint8Array(nv);
    const whole = hatchZones(index, vNode, excl, position);
    for (let v = 0; v < nv; v++) {
      const i = vNode[v];
      arrays.value[v] = qv[i] * vs;
      for (let k = 0; k < 3; k++) arrays.disp[v * 3 + k] = qu[k * n + i] * us;
      arrays.excl[v] = excl[i] && whole[v] ? 255 : 0;
    }
  } else {
    // each saved frame is stored as its change from the one before
    const q = field('frames'), F = L.frames.channels, step = L.frames.step;
    const frames = new Float32Array(F * nv);
    const sum = new Int32Array(n);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < n; i++) sum[i] += q[f * n + i];
      for (let v = 0; v < nv; v++) frames[f * nv + v] = sum[vNode[v]] * step;
    }
    arrays.frames = frames;
    if (run.cut) Object.assign(arrays, section(run.cut, L, bytes, buf, F, step));
    // the air the part warms in still air, estimated (play-flow.js heatAir): x, y, z, speed, rise per point
    if (run.air && L.air) arrays.air = new Float32Array(buf.slice(L.air.offset, L.air.offset + run.air.points * 20));
  }
  return { arrays, vertices: nv };
}

/**
 * Where a strength run's excluded zones are hatched: 1 on each drawn vertex of a zone worth hatching. A zone is a
 * patch of triangles wholly in the excluded set and on one smooth face (crease edges split it), and it is worth
 * hatching when it is a patch, not a sliver: its area against its outline squared is at least what a strip a
 * tenth as wide as it is long has. A thin band where the zone round a fixed face runs a little way up a side face
 * would otherwise show as a comb of short dashes along the edge.
 */
function hatchZones(index, vNode, excl, pos) {
  const nt = index.length / 3, inZone = new Uint8Array(nt), parent = new Int32Array(nt);
  for (let t = 0; t < nt; t++) {
    parent[t] = t;
    inZone[t] = excl[vNode[index[t * 3]]] && excl[vNode[index[t * 3 + 1]]] && excl[vNode[index[t * 3 + 2]]] ? 1 : 0;
  }
  const find = (x) => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  const edges = new Map(), nv = vNode.length;
  for (let t = 0; t < nt; t++) {
    if (!inZone[t]) continue;
    for (let e = 0; e < 3; e++) {
      const a = index[t * 3 + e], b = index[t * 3 + ((e + 1) % 3)], key = a < b ? a * nv + b : b * nv + a;
      const o = edges.get(key);
      if (o === undefined) edges.set(key, t);
      else { parent[find(o)] = find(t); edges.set(key, -1); }
    }
  }
  const len = (a, b) => Math.hypot(pos[a * 3] - pos[b * 3], pos[a * 3 + 1] - pos[b * 3 + 1], pos[a * 3 + 2] - pos[b * 3 + 2]);
  const area = new Map(), edge = new Map();
  for (let t = 0; t < nt; t++) {
    if (!inZone[t]) continue;
    const a = index[t * 3], b = index[t * 3 + 1], c = index[t * 3 + 2];
    const ux = pos[b * 3] - pos[a * 3], uy = pos[b * 3 + 1] - pos[a * 3 + 1], uz = pos[b * 3 + 2] - pos[a * 3 + 2];
    const vx = pos[c * 3] - pos[a * 3], vy = pos[c * 3 + 1] - pos[a * 3 + 1], vz = pos[c * 3 + 2] - pos[a * 3 + 2];
    const r = find(t);
    area.set(r, (area.get(r) || 0) + 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx));
  }
  for (const [key, t] of edges) {
    if (t < 0) continue;
    const a = Math.floor(key / nv), b = key % nv, r = find(t);
    edge.set(r, (edge.get(r) || 0) + len(a, b));
  }
  const keep = new Set();
  for (const [r, A] of area) { const P = edge.get(r) || 0; if (P > 0 && A / (P * P) >= 0.1 / 4.84) keep.add(r); }
  const whole = new Uint8Array(nv);
  for (let t = 0; t < nt; t++) {
    if (!inZone[t] || !keep.has(find(t))) continue;
    whole[index[t * 3]] = 1; whole[index[t * 3 + 1]] = 1; whole[index[t * 3 + 2]] = 1;
  }
  return whole;
}

/**
 * The cut faces of a heat run whose hottest point is inside the part: a grid on each face, two triangles
 * a cell where all four corners are in the part, a normal facing the side that was cut away, and the
 * temperature above the air at every grid point for every saved frame.
 */
function section(cut, L, bytes, buf, F, step) {
  const np = cut.points;
  const inside = new Uint8Array(buf, L.cut_inside.offset, np);
  const d = planes(bytes, L.cut_frames.offset, F * np);
  const cutFrames = new Float32Array(F * np);
  for (let i = 0; i < np; i++) {
    let sum = 0;
    for (let f = 0; f < F; f++) { sum += d[f * np + i]; cutFrames[f * np + i] = sum * step; }
  }
  const cutPosition = new Float32Array(np * 3), cutNormal = new Int8Array(np * 3), tris = [];
  let base = 0;
  for (const face of cut.faces) {
    const [ua, u0, u1, nu] = face.u, [va, v0, v1, nv] = face.v;
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const k = base + i * nv + j;
        cutPosition[k * 3 + face.axis] = face.at;
        cutPosition[k * 3 + ua] = u0 + ((u1 - u0) * i) / (nu - 1);
        cutPosition[k * 3 + va] = v0 + ((v1 - v0) * j) / (nv - 1);
        cutNormal[k * 3 + face.axis] = 127 * cut.sign[face.axis];
        if (i < nu - 1 && j < nv - 1) {
          const a = k, b = k + nv, c = k + nv + 1, e = k + 1;
          if (inside[a] && inside[b] && inside[c] && inside[e]) tris.push(a, b, c, a, c, e);
        }
      }
    }
    base += nu * nv;
  }
  return { cutPosition, cutNormal, cutIndex: np < 65536 ? Uint16Array.from(tris) : Uint32Array.from(tris), cutFrames };
}

// ---- scenes ----------------------------------------------------------------------------------------
// A scene.wobbl laid out for the same drawing: every mesh welded where its vertices coincide and shaded
// smooth across its faces with sharp CAD edges kept sharp (part.js's shade), every line made a round tube,
// and each object's per frame data kept as it was stored, for the stage to bring to any slider position.

/** Vertices at the same place merged, so a mesh stored with its corners split still shades smooth. */
function weld(P, tri) {
  const n = P.length / 3;
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n * 3; i++) { const k = i % 3; if (P[i] < lo[k]) lo[k] = P[i]; if (P[i] > hi[k]) hi[k] = P[i]; }
  const tol = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1e-6, 1e-12);
  const seen = new Map(), id = new Int32Array(n), rep = [];
  for (let i = 0; i < n; i++) {
    const key = `${Math.round(P[i * 3] / tol)},${Math.round(P[i * 3 + 1] / tol)},${Math.round(P[i * 3 + 2] / tol)}`;
    let w = seen.get(key);
    if (w === undefined) { w = rep.length; seen.set(key, w); rep.push(i); }
    id[i] = w;
  }
  const wp = new Float32Array(rep.length * 3);
  rep.forEach((i, w) => { wp[w * 3] = P[i * 3]; wp[w * 3 + 1] = P[i * 3 + 1]; wp[w * 3 + 2] = P[i * 3 + 2]; });
  const wt = new Uint32Array(tri.length);
  for (let j = 0; j < tri.length; j++) wt[j] = id[tri[j]];
  // triangles that collapsed in the weld are dropped
  const keep = [];
  for (let t = 0; t < wt.length; t += 3) if (wt[t] !== wt[t + 1] && wt[t + 1] !== wt[t + 2] && wt[t] !== wt[t + 2]) keep.push(wt[t], wt[t + 1], wt[t + 2]);
  return { pos: wp, tri: Uint32Array.from(keep), rep: Int32Array.from(rep), id, n: rep.length };
}

/**
 * A light Taubin smoothing of a mesh Wobbl thinned to fit its scene (thousands of vertices kept by clustering),
 * which leaves a zigzag along sharp edges: each pass pulls every vertex toward its neighbours and then pushes
 * it back a little further, so the zigzag evens out while the part keeps its size.
 */
function taubin(pos, tri, n, passes = 10) {
  const start = new Uint32Array(n + 1);
  for (let t = 0; t < tri.length; t += 3) for (let e = 0; e < 3; e++) { start[tri[t + e] + 1] += 2; }
  for (let i = 0; i < n; i++) start[i + 1] += start[i];
  const fill = start.slice(0, n), nb = new Uint32Array(start[n]);
  for (let t = 0; t < tri.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tri[t + e], b = tri[t + ((e + 1) % 3)], c = tri[t + ((e + 2) % 3)];
      nb[fill[a]++] = b; nb[fill[a]++] = c;
    }
  }
  const tmp = new Float32Array(pos.length);
  const step = (f) => {
    for (let i = 0; i < n; i++) {
      const s0 = start[i], s1 = start[i + 1];
      if (s1 === s0) { tmp.set(pos.subarray(i * 3, i * 3 + 3), i * 3); continue; }
      let x = 0, y = 0, z = 0;
      for (let j = s0; j < s1; j++) { const k = nb[j] * 3; x += pos[k]; y += pos[k + 1]; z += pos[k + 2]; }
      const c = s1 - s0;
      tmp[i * 3] = pos[i * 3] + f * (x / c - pos[i * 3]);
      tmp[i * 3 + 1] = pos[i * 3 + 1] + f * (y / c - pos[i * 3 + 1]);
      tmp[i * 3 + 2] = pos[i * 3 + 2] + f * (z / c - pos[i * 3 + 2]);
    }
    pos.set(tmp);
  };
  for (let p = 0; p < passes; p++) { step(0.5); step(-0.53); }
}

/**
 * Bilateral mesh denoising (Fleishman, Drori and Cohen-Or 2003) of a thinned mesh: each vertex moves along its
 * normal by a weighted mean of its neighbours' heights above its tangent plane, with neighbours far along the
 * normal weighted down, so the ripples clustering leaves on flat faces and rims flatten out while real edges,
 * where the neighbours sit far off the plane, stay sharp.
 */
function denoise(pos, tri, n, passes = 4) {
  const start = new Uint32Array(n + 1);
  for (let t = 0; t < tri.length; t += 3) for (let e = 0; e < 3; e++) start[tri[t + e] + 1] += 2;
  for (let i = 0; i < n; i++) start[i + 1] += start[i];
  const fill = start.slice(0, n), nb = new Uint32Array(start[n]);
  for (let t = 0; t < tri.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tri[t + e];
      nb[fill[a]++] = tri[t + ((e + 1) % 3)];
      nb[fill[a]++] = tri[t + ((e + 2) % 3)];
    }
  }
  const vn = new Float32Array(n * 3), next = new Float32Array(pos.length);
  for (let p = 0; p < passes; p++) {
    vn.fill(0);
    for (let t = 0; t < tri.length; t += 3) {
      const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const wx = pos[c] - pos[a], wy = pos[c + 1] - pos[a + 1], wz = pos[c + 2] - pos[a + 2];
      const x = uy * wz - uz * wy, y = uz * wx - ux * wz, z = ux * wy - uy * wx;
      for (const v of [a, b, c]) { vn[v] += x; vn[v + 1] += y; vn[v + 2] += z; }
    }
    next.set(pos);
    for (let i = 0; i < n; i++) {
      const s0 = start[i], s1 = start[i + 1];
      if (s1 === s0) continue;
      const o = i * 3, l = Math.hypot(vn[o], vn[o + 1], vn[o + 2]) || 1;
      const nx = vn[o] / l, ny = vn[o + 1] / l, nz = vn[o + 2] / l;
      // the spatial reach from the farthest neighbour, the height spread from the neighbours' heights
      let far = 0, mean = 0, sq = 0;
      for (let j = s0; j < s1; j++) {
        const q = nb[j] * 3, dx = pos[q] - pos[o], dy = pos[q + 1] - pos[o + 1], dz = pos[q + 2] - pos[o + 2];
        far = Math.max(far, dx * dx + dy * dy + dz * dz);
        const h = dx * nx + dy * ny + dz * nz;
        mean += h; sq += h * h;
      }
      const c = s1 - s0, sc2 = 2 * Math.max(far, 1e-30), ss = sq / c - (mean / c) ** 2, ss2 = 2 * Math.max(ss, far * 1e-4);
      let sum = 0, w = 0;
      for (let j = s0; j < s1; j++) {
        const q = nb[j] * 3, dx = pos[q] - pos[o], dy = pos[q + 1] - pos[o + 1], dz = pos[q + 2] - pos[o + 2];
        const h = dx * nx + dy * ny + dz * nz, k = Math.exp(-(dx * dx + dy * dy + dz * dz) / sc2 - (h * h) / ss2);
        sum += k * h; w += k;
      }
      const m = w > 0 ? sum / w : 0;
      next[o] = pos[o] + nx * m; next[o + 1] = pos[o + 1] + ny * m; next[o + 2] = pos[o + 2] + nz * m;
    }
    pos.set(next);
  }
}

/** A polyline as a round tube: positions, normals, triangles and, for each vertex, the point it rings and where that is. */
function tubes(P, runs, radius, seg = 10) {
  const pos = [], nrm = [], idx = [], pt = [], run = [], axis = [];
  let p0 = 0;
  runs.forEach((r, ri) => {
    const pts = [];
    for (let i = 0; i < r; i++) {
      const j = (p0 + i) * 3, q = [P[j], P[j + 1], P[j + 2]];
      if (!Number.isFinite(q[0])) continue;
      const last = pts[pts.length - 1];
      if (last && Math.hypot(q[0] - last.q[0], q[1] - last.q[1], q[2] - last.q[2]) < radius * 0.05) continue;
      pts.push({ q, i: p0 + i });
    }
    p0 += r;
    if (pts.length < 2) return;
    // a frame carried along the line so the tube never twists
    const tan = (i) => {
      const a = pts[Math.max(i - 1, 0)].q, b = pts[Math.min(i + 1, pts.length - 1)].q;
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], l = Math.hypot(...d) || 1;
      return d.map((x) => x / l);
    };
    let t = tan(0), u = Math.abs(t[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const unit = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
    let nx = unit(cross(t, u));
    const base = pos.length / 3;
    pts.forEach((p, i) => {
      t = tan(i);
      let ny = unit(cross(t, nx));
      nx = unit(cross(ny, t));
      for (let s = 0; s < seg; s++) {
        const a = (s / seg) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
        const d = [0, 1, 2].map((k) => nx[k] * c + ny[k] * sn);
        pos.push(p.q[0] + d[0] * radius, p.q[1] + d[1] * radius, p.q[2] + d[2] * radius);
        nrm.push(Math.round(d[0] * 127), Math.round(d[1] * 127), Math.round(d[2] * 127));
        pt.push(p.i);
        run.push(ri);
        axis.push(p.q[0], p.q[1], p.q[2]);
      }
      if (i) {
        const r0 = base + (i - 1) * seg, r1 = base + i * seg;
        for (let s = 0; s < seg; s++) {
          const s1 = (s + 1) % seg;
          idx.push(r0 + s, r1 + s, r1 + s1, r0 + s, r1 + s1, r0 + s1);
        }
      }
    });
  });
  return { position: new Float32Array(pos), normal: new Int8Array(nrm), index: new Uint32Array(idx), point: new Uint32Array(pt), run: new Uint16Array(run), axis: new Float32Array(axis) };
}

/**
 * Everything the stage needs for a scene: its header and, for each object it draws, the arrays as stored
 * plus a smooth shaded surface for meshes and a tube for lines. run is the page's summary (scripts/scenes.mjs).
 */
export function decodeScene(run, buf) {
  const { h, arr } = readScene(new Uint8Array(buf));
  // the magnification the build worked out, when the scene gave none of its own
  if (run.magnify) h.magnify = run.magnify;
  const lo = run.bounds.lo, hi = run.bounds.hi;
  const R = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 1e-9);
  let vertices = 0;
  const objs = h.objects.map((o) => {
    if (skipped(o)) return null;
    const d = {};
    if (o.kind === 'mesh') {
      const P = arr(o.pos), w = weld(P, arr(o.tri));
      // a part thinned to fit the file is smoothed a little along the zigzag its thinning left
      const thinned = w.n > 4000 && !o.flat;
      if (thinned) {
        taubin(w.pos, w.tri, w.n);
        denoise(w.pos, w.tri, w.n);
        for (let i = 0; i < P.length / 3; i++) { const k = w.id[i] * 3; P[i * 3] = w.pos[k]; P[i * 3 + 1] = w.pos[k + 1]; P[i * 3 + 2] = w.pos[k + 2]; }
      }
      const { index, vNode, normal } = shade(w.pos, w.tri, w.n);
      const nv = vNode.length, node = new Uint32Array(nv), nrm = new Int8Array(nv * 3);
      for (let v = 0; v < nv; v++) {
        node[v] = w.rep[vNode[v]];
        for (let k = 0; k < 3; k++) nrm[v * 3 + k] = Math.round(normal[v * 3 + k] * 127);
      }
      // closed when nearly every edge is shared by two triangles: a shell that can be drawn cut open
      const edges = new Map();
      for (let t = 0; t < w.tri.length; t += 3) {
        for (let e = 0; e < 3; e++) {
          const a = w.tri[t + e], b = w.tri[t + ((e + 1) % 3)], key = a < b ? a * w.n + b : b * w.n + a;
          edges.set(key, (edges.get(key) || 0) + 1);
        }
      }
      let two = 0;
      for (const c of edges.values()) if (c === 2) two++;
      d.closed = edges.size > 0 && two / edges.size > 0.95;
      // a thinned mesh is left without outlines, since its thinning breaks its edges into dashes
      const ed = o.flat || thinned ? [] : creases(w.tri, index, w.n);
      Object.assign(d, { P, node, normal: nrm, index: nv < 65536 ? Uint16Array.from(index) : index, edges: nv < 65536 ? Uint16Array.from(ed) : Uint32Array.from(ed) });
      if (o.field) d.fv = arr(o.field.values || o.field.frames);
      if (o.deform) d.du = arr(o.deform.values || o.deform.frames);
      if (o.reveal) d.rv = arr(o.reveal.values);
      if (o.pose) { d.pp = arr(o.pose.pos); d.pq = arr(o.pose.quat); }
      vertices += nv;
    } else if (o.kind === 'body') {
      d.pp = arr(o.pos); d.pq = arr(o.quat);
    } else if (o.kind === 'lines') {
      d.P = arr(o.pos);
      if (o.field) d.fv = arr(o.field.values);
      if (o.drawn) d.td = arr(o.drawn);
      Object.assign(d, tubes(d.P, o.runs, Math.max(o.width || 1, 0.6) * R * 0.0016));
      vertices += d.position.length / 3;
    } else if (o.kind === 'points' || o.kind === 'atoms') {
      d.P = arr(o.pos);
      if (o.field) d.fv = arr(o.field.values || o.field.frames);
      if (o.bonds) d.bonds = arr(o.bonds);
    } else if (o.kind === 'arrows') {
      d.O = arr(o.origin); d.V = arr(o.vec);
    }
    // what the object does over the scene's motion, and a fluid's own speed along its lines (play-motion.js)
    decodeTrack(o, arr, d);
    if (o.speed) d.sp = arr(o.speed);
    return d;
  });
  delete h.buffers;
  return { arrays: { header: h, objs }, vertices };
}

/** Every typed array in what decodeScene returns, to hand over without copying. */
export function transfers(arrays) {
  const out = [];
  for (const d of arrays.objs || []) if (d) for (const v of Object.values(d)) if (ArrayBuffer.isView(v)) out.push(v.buffer);
  return [...new Set(out)];
}

/** Fetch a run's playback file (or its scene), gunzip it when it is gzipped, and decode it as its run says. */
export async function loadRun(file, run) {
  const res = await fetch(file);
  if (!res.ok) throw new Error(`playback data: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return run.kind === 'scene' ? decodeScene(run, buf) : decodePlay(run, buf);
}
