// A run's playback file, decoded and laid out for the GPU. It runs in play-worker.js, off the page's
// main thread. The files come from scripts/sitegen/export_playback.py, in the coding the landing page's
// part uses (scripts/export_site_part.py), so the byte planes, the residuals and the smooth shading are
// part.js's own.
import { integers, planes, shade } from './part.js';

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
  const arrays = { position, normal: nrm, index: nv < 65536 ? Uint16Array.from(index) : index };
  const L = run.layout;
  if (run.kind === 'strength') {
    const qv = field('value'), qu = field('disp');
    const excl = new Uint8Array(buf, L.excluded.offset, n);
    const vs = L.value.step, us = L.disp.step;
    arrays.value = new Float32Array(nv);
    arrays.disp = new Float32Array(nv * 3);
    arrays.excl = new Uint8Array(nv);
    for (let v = 0; v < nv; v++) {
      const i = vNode[v];
      arrays.value[v] = qv[i] * vs;
      for (let k = 0; k < 3; k++) arrays.disp[v * 3 + k] = qu[k * n + i] * us;
      arrays.excl[v] = excl[i] ? 255 : 0;
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
  }
  return { arrays, vertices: nv };
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
