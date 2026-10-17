// Lays a scene run out for its Cycles still (site/blender/still.py): every object where the example page shows it
// once its replay has run, in world space, with the colors the page gives it, and the camera the page opens on.
// It reads the scene the way the page does (src/scripts/play.js and scene-file.js) and takes the camera from
// src/scripts/scene-view.js, so the still keeps the page's framing. Writes <out>.json and <out>.bin.
//
//   node site/blender/still-scene.mjs <scene.wobbl> <play json of the run> <out base> [w h]
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { decodeScene } from '../src/scripts/play.js';
import { at2, colorHex, lut, magnifyOf, poseAt, series, shownAt } from '../src/scripts/scene-file.js';
import { aimAt, cameraOf, followAt, insetView, mainOrbit, tubeScale } from '../src/scripts/scene-view.js';

const OVER = [242, 242, 238], NONE = [85, 88, 94];
const bytes = (c) => colorHex(c || '#9aa0a6').map((x) => Math.round(x * 255));
// a see through surface is drawn solid in the color it would show over the page's background, as the page does
const BG = [14, 15, 17];
const behind = (rgb, a) => rgb.map((c, i) => Math.round(BG[i] + (c - BG[i]) * Math.min(1, Math.max(a, 0.3) * 1.25)));

function quatMat(q, p, s = 1) {
  const [w, x, y, z] = q;
  // column major 4x4, as Blender's Matrix takes rows we write rows
  return [
    [s * (1 - 2 * (y * y + z * z)), s * 2 * (x * y - w * z), s * 2 * (x * z + w * y), p[0]],
    [s * 2 * (x * y + w * z), s * (1 - 2 * (x * x + z * z)), s * 2 * (y * z - w * x), p[1]],
    [s * 2 * (x * z - w * y), s * 2 * (y * z + w * x), s * (1 - 2 * (x * x + y * y)), p[2]],
    [0, 0, 0, 1]];
}

export function layout(sceneBytes, run, w = 640, h = 480) {
  const raw = sceneBytes[0] === 0x1f && sceneBytes[1] === 0x8b ? gunzipSync(sceneBytes) : sceneBytes;
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const { arrays } = decodeScene(run, ab);
  const H = arrays.header;
  const fields = H.fields.map((f) => ({ ...f, lut: lut(f.colormap || 'turbo'), lo: f.range[0], hi: f.range[1] }));
  const colorOf = (f, v) => {
    if (Number.isNaN(v)) return NONE;
    if (f.limit !== undefined && v > f.limit) return OVER;
    const i = Math.min(255, Math.max(0, Math.round(((v - f.lo) / (f.hi - f.lo || 1)) * 255))) * 3;
    return [f.lut[i], f.lut[i + 1], f.lut[i + 2]];
  };
  const { lo, hi } = run.bounds;
  const R = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 1e-9);
  const T = run.input.t, pIn = run.input.home, a = Math.floor(pIn), b = Math.min(a + 1, T.length - 1);
  const k = run.input.between === 'step' ? 0 : pIn - a;
  const t = k ? T[a] + (T[b] - T[a]) * k : T[a];
  const orbit = mainOrbit(run, w, h, a, b, k, { search: true });
  const ins = insetView(run, w, h, orbit.az, orbit.el);
  let tube = 1;
  if (run.follow) {
    aimAt(orbit, followAt(run, a, b, k), w, h, ins && ins.rect);
    tube = tubeScale(run, orbit, h);
  }
  const sb = run.subject, near = sb && sb.lo.map((x, i) => [x - 0.25 * (sb.hi[i] - x), sb.hi[i] + 0.25 * (sb.hi[i] - x)]);
  const far = (P) => {
    for (let i = 0; i < P.length; i += 3) if (near.every(([l, u], j) => P[i + j] >= l && P[i + j] <= u)) return false;
    return true;
  };

  const chunks = [];
  let at = 0;
  const put = (typed) => {
    const buf = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
    const o = { o: at, n: typed.length, t: typed instanceof Float32Array ? 'f32' : typed instanceof Uint32Array ? 'u32' : 'u8' };
    chunks.push(buf);
    at += buf.length;
    const pad = (4 - (at % 4)) % 4;
    if (pad) { chunks.push(Buffer.alloc(pad)); at += pad; }
    return o;
  };

  /** Every object at the frame, magnified or not and with lines thickened by tubeScale or not. */
  function objects(mag, tubeK) {
    const out = [];
    H.objects.forEach((o, oi) => {
      const d = arrays.objs[oi];
      if (!d || !shownAt(o.when, t)) return;
      if (near && o.kind === 'lines' && far(d.P)) return;
      const a1 = o.opacity ?? 1;
      const base = { oi, name: o.name || '', kind: o.kind, prop: !!o.prop, opacity: a1, color: a1 < 1 ? behind(bytes(o.color), a1) : bytes(o.color), field: !!o.field };
      if (o.glow) base.glow = series(o.glow, a, b, k);
      const m = mag && magnifyOf(H, oi);
      const placed = (pose) => {
        if (!pose) return null;
        const s = m ? m.factor : 1;
        const p = m && m.anchor ? m.anchor.map((x, i) => x + s * (pose.p[i] - x)) : pose.p;
        return quatMat(pose.q, p, s);
      };
      if (o.kind === 'body') {
        const M = placed(poseAt(d.pp, d.pq, a, b, k));
        const e = { ...base, shape: o.shape, size: o.size, matrix: M };
        if (o.field) e.color = colorOf(fields[o.field.f], series(o.field.values, a, b, k));
        out.push(e);
      } else if (o.kind === 'mesh') {
        const n = d.P.length / 3, cur = new Float32Array(d.P);
        if (d.du) {
          const ex = o.deform.exaggerate || 1, sc = o.deform.scale;
          if (sc) { const s = series(sc, a, b, k) * ex; for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + d.du[i] * s; }
          else for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + at2(d.du, n * 3, a, b, k, i) * ex;
        }
        // colors per node: the field through its map, or the object's own color
        const col = new Uint8Array(n * 3);
        if (o.field && d.fv) {
          const f = fields[o.field.f], sc = o.field.scale;
          for (let i = 0; i < n; i++) {
            const v = sc ? d.fv[i] * series(sc, a, b, k) : o.field.frames ? at2(d.fv, n, a, b, k, i) : d.fv[i];
            col.set(colorOf(f, v), i * 3);
          }
        } else for (let i = 0; i < n; i++) col.set(base.color, i * 3);
        // triangles over the welded nodes, dropping those past a reveal cut
        const cutAt = o.reveal ? series(o.reveal.at, a, b, k) : Infinity;
        const tri = [];
        for (let i = 0; i < d.index.length; i += 3) {
          const A = d.node[d.index[i]], B = d.node[d.index[i + 1]], C = d.node[d.index[i + 2]];
          if (d.rv && (d.rv[A] > cutAt || d.rv[B] > cutAt || d.rv[C] > cutAt)) continue;
          tri.push(A, B, C);
        }
        const M = o.pose ? placed(poseAt(d.pp, d.pq, a, b, k)) : null;
        out.push({ ...base, flat: !!o.flat, closed: !!d.closed, matrix: M, P: put(cur), C: put(col), T: put(Uint32Array.from(tri)) });
      } else if (o.kind === 'lines') {
        const nv = d.position.length / 3, pos = new Float32Array(nv * 3), col = new Uint8Array(nv * 3);
        for (let v = 0; v < nv; v++) {
          for (let j = 0; j < 3; j++) pos[v * 3 + j] = d.axis[v * 3 + j] + (d.position[v * 3 + j] - d.axis[v * 3 + j]) * tubeK;
          col.set(o.field && d.fv ? colorOf(fields[o.field.f], d.fv[d.point[v]]) : base.color, v * 3);
        }
        const keep = (v) => !((d.td && d.td[d.point[v]] > t) || (o.at && o.at[d.run[v]] > t));
        const tri = [];
        for (let i = 0; i < d.index.length; i += 3) if (keep(d.index[i]) && keep(d.index[i + 1]) && keep(d.index[i + 2])) tri.push(d.index[i], d.index[i + 1], d.index[i + 2]);
        if (tri.length) out.push({ ...base, P: put(pos), C: put(col), T: put(Uint32Array.from(tri)) });
      } else if (o.kind === 'points' || o.kind === 'atoms') {
        const atoms = o.kind === 'atoms', sh = o.pos.shape, n = atoms ? o.el.length : sh[sh.length - 2], many = d.P.length > n * 3;
        const pts = new Float32Array(n * 4), col = new Uint8Array(n * 3);
        const RAD = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, S: 1.05, P: 1.07, Cl: 1.02, Na: 1.66, K: 2.03, Li: 1.28, Fe: 1.32, Al: 1.21, Si: 1.11, Cu: 1.32, Ti: 1.6 };
        const CPK = { H: 'f2f2f2', C: '555a60', N: '3b6fe0', O: 'e8423b', F: '7fd17f', P: 'ff8c1a', S: 'f2d230', Cl: '33d14a', Na: 'a35cf2', K: '8f40d4', Li: 'cc80ff', Mg: '8aff00', Al: 'bfa6a6', Si: 'd9b380', Fe: 'e06633', Cu: 'c88033', Zn: '7d80b0', Ti: 'bfc2c7', Ni: '50d050', Ca: '3dff00', B: 'ffb5b5', Ar: '80d1e3', He: 'd9ffff' };
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < 3; j++) pts[i * 4 + j] = many ? at2(d.P, n * 3, a, b, k, i * 3 + j) : d.P[i * 3 + j];
          const e = atoms ? o.el[i] : '';
          pts[i * 4 + 3] = Number.isNaN(pts[i * 4]) ? 0 : e ? (RAD[e] || 1.2) * o.radius : o.size || R * 0.01;
          let c = e ? bytes(`#${CPK[e] || 'c0c4ca'}`) : bytes(o.color || '#ffd400');
          if (o.field && d.fv) c = colorOf(fields[o.field.f], o.field.frames ? at2(d.fv, n, a, b, k, i) : d.fv[i]);
          col.set(c, i * 3);
        }
        const e = { ...base, X: put(pts), C: put(col) };
        if (d.bonds && d.bonds.length) e.bonds = put(Uint32Array.from(d.bonds)), e.bondR = 0.11 * ((o.radius || 0.35) / 0.35);
        out.push(e);
      } else if (o.kind === 'arrows') {
        const n = o.vec.shape[o.vec.shape.length - 2], mV = d.V.length > n * 3, mO = d.O.length > n * 3, s = (o.scale ? series(o.scale, a, b, k) : 1) * tubeK;
        const list = [];
        for (let i = 0; i < n; i++) {
          const og = [0, 1, 2].map((j) => (mO ? at2(d.O, n * 3, a, b, k, i * 3 + j) : d.O[i * 3 + j]));
          const v = [0, 1, 2].map((j) => (mV ? at2(d.V, n * 3, a, b, k, i * 3 + j) : d.V[i * 3 + j]) * s);
          const len = Math.hypot(...v);
          if (!(len > R * 1e-4 * tubeK)) continue;
          const r = (o.radius || Math.max(len * 0.035 / tubeK, R * 0.006)) * tubeK;
          list.push({ o: og, v, r });
        }
        if (list.length) out.push({ ...base, arrows: list, color: bytes(o.color || '#ffd400') });
      } else if (o.kind === 'ground') {
        out.push({ ...base, center: o.center, size: o.size, z: o.z, grid: o.grid || null });
      }
    });
    return out;
  }

  const cam = cameraOf(orbit);
  const main = { camera: { eye: cam.position, target: cam.target, fov: cam.fov, aspect: w / h }, objects: objects(!run.follow, tube) };
  let inset = null;
  if (ins) {
    const c = cameraOf(ins.orbit);
    inset = { rect: ins.rect, camera: { eye: c.position, target: c.target, fov: c.fov, aspect: ins.rect[2] / ins.rect[3] }, objects: objects(true, 1) };
  }
  return {
    json: { version: 1, w, h, radius: R, subject: run.subject || run.hold || run.bounds, bounds: run.bounds, follow: !!run.follow, up: H.up || 'z', main, inset },
    bin: Buffer.concat(chunks),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [scene, play, out, w, h] = process.argv.slice(2);
  const run = JSON.parse(readFileSync(play, 'utf-8'));
  const { json, bin } = layout(readFileSync(scene), run, Number(w || 640), Number(h || 480));
  writeFileSync(`${out}.json`, JSON.stringify(json));
  writeFileSync(`${out}.bin`, bin);
}
