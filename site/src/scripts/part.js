// The hero part's mesh, decoded and laid out for the GPU. It runs in part-worker.js, off the
// page's main thread. The file comes from scripts/export_site_part.py, whose docstring describes it.

/** Zigzag int32 values stored as four byte planes. */
export function planes(bytes, at, count) {
  const out = new Int32Array(count);
  const b1 = at + count, b2 = at + 2 * count, b3 = at + 3 * count;
  for (let i = 0; i < count; i++) {
    const z = (bytes[at + i] | (bytes[b1 + i] << 8) | (bytes[b2 + i] << 16) | (bytes[b3 + i] << 24)) >>> 0;
    out[i] = (z >>> 1) ^ -(z & 1);
  }
  return out;
}

/** Residuals back to integers: corners from the corners known before them, midside nodes from their edge. */
export function integers(r, channels, n, nc, pa, pb, ends) {
  const q = new Int32Array(r.length);
  for (let c = 0; c < channels; c++) {
    const o = c * n;
    for (let i = 0; i < nc; i++) {
      const a = pa[i], b = pb[i];
      const p = a < 0 ? (i ? q[o + i - 1] : 0) : b < 0 ? q[o + a] : Math.floor((q[o + a] + q[o + b]) / 2);
      q[o + i] = r[o + i] + p;
    }
    for (let i = nc; i < n; i++) {
      const e = (i - nc) * 2;
      q[o + i] = r[o + i] + Math.floor((q[o + ends[e]] + q[o + ends[e + 1]]) / 2);
    }
  }
  return q;
}

/**
 * Nodes, triangles and the three unit cases from the gunzipped file: node positions (n x 3), the
 * linear triangles VTK would draw (four per curved triangle), stress per newton (n x 18: case 1 xx yy
 * zz xy yz zx, then cases 2 and 3), displacement per newton (n x 9) and the excluded zone flags.
 */
export function decodePart(man, buf) {
  const bytes = new Uint8Array(buf);
  const { triangles: T, corners: nc, nodes: n, bounds_min: lo, bounds_max: hi } = man.mesh;
  const L = man.layout;

  // corners by distance back from the next new one, midside nodes numbered as edges are first crossed
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
  if (hw !== nc) throw new Error('part data: corner count does not match');
  const mid = new Int32Array(T * 3), ends = new Int32Array((n - nc) * 2), seen = new Map();
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
  if (m !== n) throw new Error('part data: node count does not match');

  const field = (key, channels) => integers(planes(bytes, L[key].offset, channels * n), channels, n, nc, pa, pb, ends);
  const qp = field('position', 3), qs = field('stress', 18), qu = field('disp', 9);

  const pos = new Float32Array(n * 3);
  for (let k = 0; k < 3; k++) {
    const s = (hi[k] - lo[k]) / 65535;
    for (let i = 0; i < n; i++) pos[i * 3 + k] = lo[k] + qp[k * n + i] * s;
  }
  const S = new Float64Array(n * 18), U = new Float64Array(n * 9);
  const ss = L.stress.step, ds = L.disp.step;
  for (let c = 0; c < 18; c++) for (let i = 0; i < n; i++) S[i * 18 + c] = qs[c * n + i] * ss;
  for (let c = 0; c < 9; c++) for (let i = 0; i < n; i++) U[i * 9 + c] = qu[c * n + i] * ds;
  // nodes near a peak carry a finer step
  const nh = L.hot.count;
  const gaps = planes(bytes, L.hot.offset, nh), fine = planes(bytes, L.hot_stress.offset, nh * 18);
  const every = Math.round(ss / L.hot_stress.step), fs = L.hot_stress.step;
  for (let j = 0, i = 0; j < nh; j++) {
    i += gaps[j];
    for (let c = 0; c < 18; c++) S[i * 18 + c] = (qs[c * n + i] * every + fine[c * nh + j]) * fs;
  }
  const excl = new Uint8Array(buf, L.excluded.offset, n).slice();

  const tri = new Uint32Array(T * 12);
  for (let t = 0; t < T; t++) {
    const a = ct[t * 3], b = ct[t * 3 + 1], c = ct[t * 3 + 2];
    const ab = mid[t * 3], bc = mid[t * 3 + 1], ca = mid[t * 3 + 2];
    tri.set([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], t * 12);
  }
  return { n, pos, tri, S, U, excl };
}

/**
 * One vertex per smooth group of the triangles around each node, so flat faces shade flat across
 * sharp CAD edges and fillets shade smooth. Two triangles at a node share a group when they share
 * an edge there and their normals are within 35 degrees, as VTK splits them.
 */
export function shade(pos, tri, n) {
  const nt = tri.length / 3, COS = Math.cos((35 * Math.PI) / 180);
  const fn = new Float32Array(nt * 3), un = new Float32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    const a = tri[t * 3] * 3, b = tri[t * 3 + 1] * 3, c = tri[t * 3 + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx, l = Math.hypot(x, y, z) || 1;
    fn[t * 3] = x; fn[t * 3 + 1] = y; fn[t * 3 + 2] = z;
    un[t * 3] = x / l; un[t * 3 + 1] = y / l; un[t * 3 + 2] = z / l;
  }
  // triangle corners around each node
  const start = new Uint32Array(n + 1);
  for (let j = 0; j < tri.length; j++) start[tri[j] + 1]++;
  for (let i = 0; i < n; i++) start[i + 1] += start[i];
  const fill = start.slice(0, n), slots = new Uint32Array(tri.length);
  for (let j = 0; j < tri.length; j++) slots[fill[tri[j]]++] = j;

  const index = new Uint32Array(tri.length);
  let vNode = new Uint32Array(n + (n >> 2)), vNrm = new Float32Array(vNode.length * 3), nv = 0;
  const parent = new Int32Array(64), ids = new Int32Array(64);
  const find = (x) => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  for (let v = 0; v < n; v++) {
    const s0 = start[v], d = Math.min(start[v + 1] - s0, 64);
    for (let p = 0; p < d; p++) { parent[p] = p; ids[p] = -1; }
    for (let p = 0; p < d; p++) {
      const jp = slots[s0 + p], tp = (jp / 3) | 0, kp = jp % 3;
      const p1 = tri[tp * 3 + ((kp + 1) % 3)], p2 = tri[tp * 3 + ((kp + 2) % 3)];
      for (let q = p + 1; q < d; q++) {
        const jq = slots[s0 + q], tq = (jq / 3) | 0, kq = jq % 3;
        const q1 = tri[tq * 3 + ((kq + 1) % 3)], q2 = tri[tq * 3 + ((kq + 2) % 3)];
        if (p1 !== q1 && p1 !== q2 && p2 !== q1 && p2 !== q2) continue;
        if (un[tp * 3] * un[tq * 3] + un[tp * 3 + 1] * un[tq * 3 + 1] + un[tp * 3 + 2] * un[tq * 3 + 2] < COS) continue;
        parent[find(p)] = find(q);
      }
    }
    for (let p = 0; p < d; p++) {
      const r = find(p), jp = slots[s0 + p], tp = (jp / 3) | 0;
      if (ids[r] < 0) {
        if (nv === vNode.length) {
          const grow = new Uint32Array(nv * 2); grow.set(vNode); vNode = grow;
          const g2 = new Float32Array(nv * 6); g2.set(vNrm); vNrm = g2;
        }
        ids[r] = nv;
        vNode[nv++] = v;
      }
      const id = ids[r];
      vNrm[id * 3] += fn[tp * 3]; vNrm[id * 3 + 1] += fn[tp * 3 + 1]; vNrm[id * 3 + 2] += fn[tp * 3 + 2];
      index[jp] = id;
    }
  }
  const normal = vNrm.slice(0, nv * 3);
  for (let i = 0; i < normal.length; i += 3) {
    const l = Math.hypot(normal[i], normal[i + 1], normal[i + 2]) || 1;
    normal[i] /= l; normal[i + 1] /= l; normal[i + 2] /= l;
  }
  return { index, vNode: vNode.slice(0, nv), normal };
}

/**
 * The edges a CAD viewer outlines, as pairs of drawn vertices: every edge where shade split the smooth
 * groups (the faces meet at more than its crease angle) and every border edge of an open surface. Drawn as
 * thin lines over the faces, as three.js's LDraw viewer and most CAD viewers draw them, so flat faces and
 * small parts read crisply. tri holds node numbers and index the drawn vertex of each corner (shade's).
 */
export function creases(tri, index, n) {
  const seen = new Map(), out = [];
  for (let t = 0; t < tri.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const j0 = t + k, j1 = t + ((k + 1) % 3), a = tri[j0], b = tri[j1];
      const key = a < b ? a * n + b : b * n + a;
      const first = seen.get(key);
      if (first === undefined) { seen.set(key, j0 * 4 + (a < b ? 0 : 1)); continue; }
      if (first < 0) continue;
      // the corners of the first triangle on this edge, ordered as the nodes are
      const f0 = (first >> 2), flip = first & 1, ft = f0 - (f0 % 3), f1 = ft + ((f0 - ft + 1) % 3);
      const ia = flip ? index[f1] : index[f0], ib = flip ? index[f0] : index[f1];
      const ja = a < b ? index[j0] : index[j1], jb = a < b ? index[j1] : index[j0];
      if (ia !== ja || ib !== jb) out.push(ia, ib);
      seen.set(key, -1);
    }
  }
  for (const v of seen.values()) {
    if (v < 0) continue;
    const f0 = v >> 2, ft = f0 - (f0 % 3), f1 = ft + ((f0 - ft + 1) % 3);
    out.push(index[f0], index[f1]);
  }
  return out;
}

/**
 * Everything the GPU needs, one entry per drawn vertex, as small integers the GPU reads back as
 * -1 to 1 (normalized). The shader multiplies each channel by its largest value, in range, and
 * combines the three unit cases for the load's size and direction, so moving the load only changes
 * uniforms.
 *   position                   float x3
 *   normal                     int8 x3
 *   stressA, stressB, stressC  int16 x4  xx yy zz xy of cases 1, 2 and 3, MPa per newton
 *   shearAB                    int16 x4  yz zx of case 1, then of case 2
 *   shearC                     int16 x4  yz zx of case 3, then 1 in an excluded zone
 *   dispA, dispB, dispC        int16 x3  displacement of each case in mm per newton
 */
export function buildMesh(man, buf) {
  const { n, pos, tri, S, U, excl } = decodePart(man, buf);
  const { index, vNode, normal } = shade(pos, tri, n);
  const nv = vNode.length;
  // which of a node's 18 stress or 9 displacement values goes in each channel
  const pick = {
    stressA: [0, 1, 2, 3], stressB: [6, 7, 8, 9], stressC: [12, 13, 14, 15], shearAB: [4, 5, 10, 11], shearC: [16, 17],
  };
  const pickU = { dispA: [0, 1, 2], dispB: [3, 4, 5], dispC: [6, 7, 8] };
  const range = {};
  const quantize = (key, from, src, stride, size) => {
    const ch = from[key], top = ch.map(() => 1e-30);
    for (let i = 0; i < n; i++) ch.forEach((c, k) => { top[k] = Math.max(top[k], Math.abs(src[i * stride + c])); });
    const out = new Int16Array(nv * size);
    for (let v = 0; v < nv; v++) {
      const o = vNode[v] * stride;
      for (let k = 0; k < ch.length; k++) out[v * size + k] = Math.round((src[o + ch[k]] / top[k]) * 32767);
    }
    range[key] = top;
    return out;
  };
  const arrays = {};
  for (const key of ['stressA', 'stressB', 'stressC', 'shearAB', 'shearC']) arrays[key] = quantize(key, pick, S, 18, 4);
  for (const key of ['dispA', 'dispB', 'dispC']) arrays[key] = quantize(key, pickU, U, 9, 3);
  const position = new Float32Array(nv * 3), nrm = new Int8Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const i = vNode[v];
    position[v * 3] = pos[i * 3]; position[v * 3 + 1] = pos[i * 3 + 1]; position[v * 3 + 2] = pos[i * 3 + 2];
    nrm[v * 3] = Math.round(normal[v * 3] * 127); nrm[v * 3 + 1] = Math.round(normal[v * 3 + 1] * 127);
    nrm[v * 3 + 2] = Math.round(normal[v * 3 + 2] * 127);
    arrays.shearC[v * 4 + 2] = excl[i] ? 32767 : 0;
  }
  const idx = nv < 65536 ? Uint16Array.from(index) : index;
  return { arrays: { position, normal: nrm, ...arrays, index: idx }, range };
}
