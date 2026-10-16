// A run's scene drawn on the strength and heat playback's stage (play-draw.js): the same light, the same
// background and the same load arrows, with each of the scene's objects brought to the slider's position.
// Meshes come smooth shaded from play.js and are colored by their field, bodies are true primitives, lines
// are round tubes and props are solid and matte. The viewer's text labels and its readout list are left
// to the page, which shows only the peak marker and the probe.
import {
  BackSide, BoxGeometry, BufferAttribute, BufferGeometry, CapsuleGeometry, Color, CylinderGeometry, DoubleSide,
  Group, InstancedBufferAttribute, InstancedMesh, LineBasicMaterial, LineSegments, Matrix4, Mesh,
  PlaneGeometry, Quaternion, Raycaster, Scene, ShaderMaterial, SphereGeometry, TorusGeometry, Vector3,
} from 'three';
import { LIT, arrow } from './play-draw.js';
import { at2, colorHex, lut, magnifyOf, poseAt, series, shownAt } from './scene-file.js';

const VERT = /* glsl */ `
attribute vec3 tint;
attribute float cutv;
varying vec3 vC;
varying vec3 vN;
varying vec3 vV;
varying float vR;
void main() {
  vec3 c = tint;
  vec4 p = vec4(position, 1.0);
  vec3 n = normal;
#ifdef USE_INSTANCING
  p = instanceMatrix * p;
  n = mat3(instanceMatrix) * n;
#endif
#ifdef USE_INSTANCING_COLOR
  c = instanceColor;
#endif
  vC = c;
  vR = cutv;
  vec4 mv = modelViewMatrix * p;
  vN = normalize(normalMatrix * n);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

// built when first used, since play-draw.js, which gives the light, imports this module too
const frag = () => /* glsl */ `
uniform float uCut;
uniform float uSpec;
uniform float uGlow;
uniform vec3 uGlowC;
uniform float uAlpha;
varying vec3 vC;
varying vec3 vN;
varying vec3 vV;
varying float vR;
${LIT}
void main() {
  // a print rising layer by layer, or a line drawn up to the moment shown
  if (vR > uCut) discard;
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 col = mix(lit(vC, n, v, uSpec), uGlowC, uGlow);
  gl_FragColor = vec4(col, uAlpha);
}`;

const OVER = [242, 242, 238], NONE = [85, 88, 94];
const CPK = { H: 'f2f2f2', C: '555a60', N: '3b6fe0', O: 'e8423b', F: '7fd17f', P: 'ff8c1a', S: 'f2d230', Cl: '33d14a', Na: 'a35cf2', K: '8f40d4', Li: 'cc80ff', Mg: '8aff00', Al: 'bfa6a6', Si: 'd9b380', Fe: 'e06633', Cu: 'c88033', Zn: '7d80b0', Ti: 'bfc2c7', Ni: '50d050', Ca: '3dff00', B: 'ffb5b5', Ar: '80d1e3', He: 'd9ffff' };
const RAD = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, S: 1.05, P: 1.07, Cl: 1.02, Na: 1.66, K: 2.03, Li: 1.28, Fe: 1.32, Al: 1.21, Si: 1.11, Cu: 1.32, Ti: 1.6 };
const bytes = (c) => colorHex(c).map((x) => Math.round(x * 255));

// Nothing is drawn see through. Where a scene asked for a see through surface (glass, a mold box, a beaker,
// a shell round a field), it is drawn solid in the color it would have shown over the page's background,
// and when it is a closed shell, with its near side cut away so what it holds still shows. Props are drawn
// darker and matte, so the part the run is about stays the brightest thing in the view.
const BG = [14, 15, 17];
const behind = (rgb, a) => rgb.map((c, i) => Math.round(BG[i] + (c - BG[i]) * Math.min(1, Math.max(a, 0.3) * 1.25)));
const PROP = 0.62;

function material(o, { prop = false, shell = false } = {}) {
  const cut = shell && (o.opacity ?? 1) < 1;
  return new ShaderMaterial({
    vertexShader: VERT, fragmentShader: frag(),
    uniforms: {
      uCut: { value: 1e30 }, uSpec: { value: prop ? 0.25 : 1 }, uGlow: { value: 0 }, uGlowC: { value: new Color(1, 0.85, 0.2) },
      uAlpha: { value: 1 },
    },
    side: cut ? BackSide : DoubleSide,
    // a cut open shell gives way to whatever it touches, such as the plate a part sits on
    polygonOffset: cut, polygonOffsetFactor: cut ? 2 : 0, polygonOffsetUnits: cut ? 2 : 0,
  });
}

/** A primitive as Wobbl's scenes size it (MuJoCo's sizes), standing along z. */
function primitive(shape, s) {
  let g;
  switch (shape) {
    case 'sphere': g = new SphereGeometry(s[0], 48, 32); break;
    case 'ellipsoid': g = new SphereGeometry(1, 48, 32); g.scale(s[0], s[1], s[2]); break;
    case 'box': g = new BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2); break;
    case 'cylinder': g = new CylinderGeometry(s[0], s[0], s[1] * 2, 48); g.rotateX(Math.PI / 2); break;
    case 'hex': g = new CylinderGeometry(s[0], s[0], s[1] * 2, 6); g.rotateX(Math.PI / 2); g = g.toNonIndexed(); g.computeVertexNormals(); break;
    case 'capsule': g = new CapsuleGeometry(s[0], s[1] * 2, 10, 32); g.rotateX(Math.PI / 2); break;
    case 'cone': g = new CylinderGeometry(0, s[0], s[1] * 2, 48); g.rotateX(Math.PI / 2); break;
    case 'torus': g = new TorusGeometry(s[0], s[1], 24, 72); break;
    case 'plane': g = new PlaneGeometry(s[0] * 2, s[1] * 2); break;
    default: g = new BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
  }
  return g;
}

function tinted(geo, rgb) {
  const n = geo.getAttribute('position').count, t = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) t.set(rgb, i * 3);
  geo.setAttribute('tint', new BufferAttribute(t, 3, true));
  if (!geo.getAttribute('cutv')) geo.setAttribute('cutv', new BufferAttribute(new Float32Array(n).fill(-1e30), 1));
  return geo;
}

const tmpM = new Matrix4(), tmpQ = new Quaternion(), tmpP = new Vector3(), tmpS = new Vector3(), Y = new Vector3(0, 1, 0);

/**
 * The scene's objects for one view. set(f) brings them to f = { a, b, k, t, bend, mag }: frame a, the next
 * frame b, the share k between them, the time or input value t, the bending drawn larger or not and the
 * small things magnified or not.
 */
export function sceneParts({ arrays, run }) {
  const h = arrays.header, scene = new Scene();
  const lo = run.bounds.lo, hi = run.bounds.hi;
  const R = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 1e-9);
  const fields = h.fields.map((f) => ({ ...f, lut: lut(f.colormap || 'turbo'), lo: f.range[0], hi: f.range[1] }));
  const colorOf = (f, v, out, o) => {
    const c = Number.isNaN(v) ? NONE : f.limit !== undefined && v > f.limit ? OVER : null;
    if (c) { out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; return; }
    const i = Math.min(255, Math.max(0, Math.round(((v - f.lo) / (f.hi - f.lo || 1)) * 255))) * 3;
    out[o] = f.lut[i]; out[o + 1] = f.lut[i + 1]; out[o + 2] = f.lut[i + 2];
  };
  const items = [];
  h.objects.forEach((o, oi) => {
    const d = arrays.objs[oi];
    if (!d) return;
    let rgb = bytes(o.color || '#9aa0a6');
    if (o.prop && o.kind !== 'ground') rgb = rgb.map((c) => Math.round(c * PROP));
    const it = { o, d, oi, mag: magnifyOf(h, oi), f: o.field ? fields[o.field.f] : null, base: (o.opacity ?? 1) < 1 ? behind(rgb, o.opacity) : rgb };
    if (o.kind === 'mesh') {
      const geo = new BufferGeometry(), nv = d.node.length;
      geo.setIndex(new BufferAttribute(d.index, 1));
      const pos = new Float32Array(nv * 3);
      for (let v = 0; v < nv; v++) for (let k = 0; k < 3; k++) pos[v * 3 + k] = d.P[d.node[v] * 3 + k];
      geo.setAttribute('position', new BufferAttribute(pos, 3));
      geo.setAttribute('normal', new BufferAttribute(d.normal, 3, true));
      geo.setAttribute('tint', new BufferAttribute(new Uint8Array(nv * 3), 3, true));
      const cut = new Float32Array(nv).fill(-1e30);
      if (d.rv) for (let v = 0; v < nv; v++) cut[v] = d.rv[d.node[v]];
      geo.setAttribute('cutv', new BufferAttribute(cut, 1));
      it.n = d.P.length / 3;
      it.cur = d.P.slice();
      it.val = new Float32Array(it.n).fill(NaN);
      it.mesh = new Mesh(geo, material(o, { prop: !!o.prop, shell: d.closed }));
      it.colored = false;
    } else if (o.kind === 'body') {
      it.mesh = new Mesh(tinted(primitive(o.shape, o.size), it.base), material(o, { prop: !!o.prop, shell: o.shape !== 'plane' }));
      if (o.glow) it.mesh.material.uniforms.uGlowC.value.setRGB(...colorHex(o.color).map((v) => Math.min(1, v * 0.5 + 0.55)));
    } else if (o.kind === 'lines') {
      const geo = new BufferGeometry(), nv = d.position.length / 3;
      geo.setIndex(new BufferAttribute(d.index, 1));
      geo.setAttribute('position', new BufferAttribute(d.position, 3));
      geo.setAttribute('normal', new BufferAttribute(d.normal, 3, true));
      const tint = new Uint8Array(nv * 3), tmp = [0, 0, 0];
      for (let v = 0; v < nv; v++) {
        if (it.f && d.fv) { colorOf(it.f, d.fv[d.point[v]], tmp, 0); tint.set(tmp, v * 3); } else tint.set(it.base, v * 3);
      }
      geo.setAttribute('tint', new BufferAttribute(tint, 3, true));
      // a trajectory drawn up to the moment shown, or a crack front that appears at its time
      const cut = new Float32Array(nv).fill(-1e30);
      for (let v = 0; v < nv; v++) {
        if (d.td) cut[v] = d.td[d.point[v]];
        if (o.at) cut[v] = Math.max(cut[v], o.at[d.run[v]]);
      }
      geo.setAttribute('cutv', new BufferAttribute(cut, 1));
      it.mesh = new Mesh(geo, material({ ...o, opacity: 1 }, { prop: !!o.prop }));
      it.cuts = !!(d.td || o.at);
    } else if (o.kind === 'points' || o.kind === 'atoms') {
      const atoms = o.kind === 'atoms', sh = o.pos.shape;
      it.n = atoms ? o.el.length : sh[sh.length - 2];
      it.rad = new Float32Array(it.n);
      const cols = new Float32Array(it.n * 3);
      for (let i = 0; i < it.n; i++) {
        const e = atoms ? o.el[i] : '', c = e ? colorHex(`#${CPK[e] || 'c0c4ca'}`) : colorHex(o.color || '#ffd400');
        cols.set(c, i * 3);
        it.rad[i] = e ? (RAD[e] || 1.2) * o.radius : o.size || R * 0.01;
      }
      const geo = tinted(new SphereGeometry(1, 24, 16), [255, 255, 255]);
      it.mesh = new InstancedMesh(geo, material(o), it.n);
      it.mesh.instanceColor = new InstancedBufferAttribute(cols, 3);
      it.mesh.frustumCulled = false;
      if (d.bonds && d.bonds.length) {
        const bg = tinted(new CylinderGeometry(1, 1, 1, 14), [150, 154, 160]);
        it.bonds = new InstancedMesh(bg, material(o), d.bonds.length / 2);
        it.bonds.frustumCulled = false;
        scene.add(it.bonds);
      }
    } else if (o.kind === 'arrows') {
      const sh = o.vec.shape;
      it.n = sh[sh.length - 2];
      it.mesh = new Group();
      it.arrows = [];
      for (let i = 0; i < it.n; i++) { const g = arrow(o.color || '#ffd400'); it.arrows.push(g); it.mesh.add(g); }
    } else if (o.kind === 'ground') {
      const g = new Group(), s = o.size;
      const plane = new Mesh(tinted(new PlaneGeometry(s, s), it.base), material(o, { prop: true }));
      plane.position.set(o.center[0], o.center[1], o.z - R * 2e-4);
      g.add(plane);
      if (o.grid) {
        const pts = [], c = o.center, z = o.z + R * 1e-4, half = s / 2;
        for (let x = -half; x <= half + 1e-9; x += o.grid) {
          pts.push(c[0] + x, c[1] - half, z, c[0] + x, c[1] + half, z, c[0] - half, c[1] + x, z, c[0] + half, c[1] + x, z);
        }
        const lg = new BufferGeometry();
        lg.setAttribute('position', new BufferAttribute(new Float32Array(pts), 3));
        g.add(new LineSegments(lg, new LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.07, depthWrite: false })));
      }
      it.mesh = g;
    }
    if (!it.mesh) return;
    it.mesh.frustumCulled = false;
    scene.add(it.mesh);
    items.push(it);
  });

  /** The object's matrix for a pose, grown about its anchor when it is magnified. */
  function place(obj, it, pose, mag) {
    const m = mag && it.mag;
    const s = m ? m.factor : 1;
    const p = m && m.anchor ? m.anchor.map((x, i) => x + s * (pose.p[i] - x)) : pose.p;
    tmpQ.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
    obj.matrixAutoUpdate = false;
    obj.matrix.compose(tmpP.set(p[0], p[1], p[2]), tmpQ, tmpS.set(s, s, s));
    obj.matrixWorldNeedsUpdate = true;
  }

  function meshFrame(it, a, b, k, bend) {
    const { o, d } = it, n = it.n, nv = d.node.length, geo = it.mesh.geometry;
    if (d.du) {
      const ex = bend ? o.deform.exaggerate || 1 : 1, sc = o.deform.scale, cur = it.cur;
      if (sc) { const s = series(sc, a, b, k) * ex; for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + d.du[i] * s; }
      else for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + at2(d.du, n * 3, a, b, k, i) * ex;
      const pa = geo.getAttribute('position');
      for (let v = 0; v < nv; v++) { const j = d.node[v] * 3; pa.array[v * 3] = cur[j]; pa.array[v * 3 + 1] = cur[j + 1]; pa.array[v * 3 + 2] = cur[j + 2]; }
      pa.needsUpdate = true;
      geo.boundingSphere = null;
    }
    if (d.fv) {
      const sc = o.field.scale;
      if (sc) { const s = series(sc, a, b, k); for (let i = 0; i < n; i++) it.val[i] = d.fv[i] * s; }
      else if (o.field.frames) for (let i = 0; i < n; i++) it.val[i] = at2(d.fv, n, a, b, k, i);
      else it.val.set(d.fv.subarray(0, n));
    }
    if (d.fv || !it.colored) {
      const t = geo.getAttribute('tint');
      for (let v = 0; v < nv; v++) {
        if (it.f && d.fv) colorOf(it.f, it.val[d.node[v]], t.array, v * 3);
        else { t.array[v * 3] = it.base[0]; t.array[v * 3 + 1] = it.base[1]; t.array[v * 3 + 2] = it.base[2]; }
      }
      t.needsUpdate = true;
      it.colored = true;
    }
    it.mesh.material.uniforms.uCut.value = o.reveal ? series(o.reveal.at, a, b, k) : 1e30;
  }

  function pointsFrame(it, a, b, k) {
    const { o, d } = it, n = it.n, many = d.P.length > n * 3, tmp = [0, 0, 0];
    const at = (i, c) => (many ? at2(d.P, n * 3, a, b, k, i * 3 + c) : d.P[i * 3 + c]);
    for (let i = 0; i < n; i++) {
      const x = at(i, 0), r = Number.isNaN(x) ? 0 : it.rad[i];
      tmpM.makeScale(r, r, r).setPosition(x, at(i, 1), at(i, 2));
      it.mesh.setMatrixAt(i, tmpM);
      if (it.f && d.fv) {
        colorOf(it.f, o.field.frames ? at2(d.fv, n, a, b, k, i) : d.fv[i], tmp, 0);
        it.mesh.instanceColor.setXYZ(i, tmp[0] / 255, tmp[1] / 255, tmp[2] / 255);
      }
    }
    it.mesh.instanceMatrix.needsUpdate = true;
    if (it.mesh.instanceColor) it.mesh.instanceColor.needsUpdate = true;
    if (it.bonds) {
      const r = 0.11 * ((o.radius || 0.35) / 0.35), A = new Vector3(), B = new Vector3(), D = new Vector3();
      for (let j = 0; j < d.bonds.length; j += 2) {
        const p = d.bonds[j], q = d.bonds[j + 1];
        A.set(at(p, 0), at(p, 1), at(p, 2));
        B.set(at(q, 0), at(q, 1), at(q, 2));
        D.subVectors(B, A);
        const l = D.length() || 1e-9;
        tmpQ.setFromUnitVectors(Y, D.divideScalar(l));
        tmpM.compose(A.add(B).multiplyScalar(0.5), tmpQ, tmpS.set(r, l, r));
        it.bonds.setMatrixAt(j / 2, tmpM);
      }
      it.bonds.instanceMatrix.needsUpdate = true;
    }
  }

  function arrowsFrame(it, a, b, k) {
    const { o, d } = it, n = it.n, mV = d.V.length > n * 3, mO = d.O.length > n * 3, s = o.scale ? series(o.scale, a, b, k) : 1;
    const dir = new Vector3();
    for (let i = 0; i < n; i++) {
      const og = [0, 1, 2].map((j) => (mO ? at2(d.O, n * 3, a, b, k, i * 3 + j) : d.O[i * 3 + j]));
      const v = [0, 1, 2].map((j) => (mV ? at2(d.V, n * 3, a, b, k, i * 3 + j) : d.V[i * 3 + j]) * s);
      const len = Math.hypot(...v), g = it.arrows[i];
      g.visible = len > R * 1e-4;
      if (!g.visible) continue;
      const r = o.radius || Math.max(len * 0.035, R * 0.006);
      const head = Math.min(len * 0.45, r * 7), body = Math.max(len - head, len * 0.01), hs = head / 8;
      dir.set(v[0] / len, v[1] / len, v[2] / len);
      g.quaternion.setFromUnitVectors(Y, dir);
      g.position.set(og[0], og[1], og[2]);
      const { shaft, head: cone } = g.userData;
      shaft.scale.set(r, body, r);
      shaft.position.set(0, body / 2, 0);
      // the head is as long as seven shaft radii and 2.6 of them wide, as the viewer draws it (the cone is 3 by 8)
      cone.scale.set((2.6 * r) / 3, hs, (2.6 * r) / 3);
      cone.position.set(0, body + head / 2, 0);
    }
  }

  let key = '';
  function set(f) {
    const { a, b, k, t, bend = true, mag = true } = f;
    const next = `${a},${b},${k.toFixed(5)},${t},${bend},${mag}`;
    if (next === key) return;
    key = next;
    for (const it of items) {
      const o = it.o;
      it.mesh.visible = shownAt(o.when, t);
      if (it.bonds) it.bonds.visible = it.mesh.visible;
      if (!it.mesh.visible) continue;
      const uni = it.mesh.material?.uniforms;
      if (uni && o.glow) uni.uGlow.value = series(o.glow, a, b, k) * 0.85;
      if (o.kind === 'mesh') {
        meshFrame(it, a, b, k, bend);
        const pose = poseAt(it.d.pp, it.d.pq, a, b, k);
        if (pose) place(it.mesh, it, pose, mag);
      } else if (o.kind === 'body') {
        place(it.mesh, it, poseAt(it.d.pp, it.d.pq, a, b, k), mag);
        if (it.f) {
          const c = [0, 0, 0];
          colorOf(it.f, series(o.field.values, a, b, k), c, 0);
          const ta = it.mesh.geometry.getAttribute('tint');
          if (ta.array[0] !== c[0] || ta.array[1] !== c[1] || ta.array[2] !== c[2]) {
            for (let i = 0; i < ta.count; i++) ta.array.set(c, i * 3);
            ta.needsUpdate = true;
          }
        }
      } else if (o.kind === 'lines') {
        if (it.cuts) uni.uCut.value = t;
      } else if (o.kind === 'points' || o.kind === 'atoms') pointsFrame(it, a, b, k);
      else if (o.kind === 'arrows') arrowsFrame(it, a, b, k);
    }
  }

  const ray = new Raycaster();
  /** The field under a ray from origin along dir, on the nearest colored surface it meets. */
  function probe(origin, dir) {
    ray.set(new Vector3(...origin), new Vector3(...dir));
    scene.updateMatrixWorld();
    const targets = items.filter((it) => it.f && it.mesh.visible && (it.o.kind === 'mesh' || it.o.kind === 'body') && !it.o.prop).map((it) => it.mesh);
    const hit = ray.intersectObjects(targets, false).find((x) => {
      const it = items.find((i) => i.mesh === x.object);
      return !(it.o.kind === 'mesh' && it.d.rv && x.face && it.d.rv[it.d.node[x.face.a]] > it.mesh.material.uniforms.uCut.value);
    });
    if (!hit) return null;
    const it = items.find((i) => i.mesh === hit.object), f = it.f;
    let v;
    if (it.o.kind === 'mesh') {
      const pa = it.mesh.geometry.getAttribute('position'), loc = it.mesh.worldToLocal(hit.point.clone());
      let best = Infinity;
      for (const c of [hit.face.a, hit.face.b, hit.face.c]) {
        const dd = (pa.getX(c) - loc.x) ** 2 + (pa.getY(c) - loc.y) ** 2 + (pa.getZ(c) - loc.z) ** 2;
        if (dd < best) { best = dd; v = it.val[it.d.node[c]]; }
      }
    } else {
      const kk = key.split(',').map(Number);
      v = series(it.o.field.values, kk[0], kk[1], kk[2]);
    }
    if (!Number.isFinite(v)) return null;
    return { v, name: f.name, unit: f.unit || '', digits: f.digits ?? 3, limit: f.limit ?? null, limitLabel: f.limit_label || '', point: hit.point.toArray() };
  }

  function dispose() {
    scene.traverse((x) => { x.geometry?.dispose?.(); x.material?.dispose?.(); });
  }
  return { scene, set, probe, dispose };
}
