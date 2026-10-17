// Tracers that ride a fluid's lines (streamlines from a solved flow, or a convection plume Wobbl estimated) as fast
// as the fluid goes at each point of them, slowed by one factor for the whole view so a flow of metres a second
// shows on a part a few centimetres across. Each tracer takes the color of the fluid's value where it is, such as
// its temperature, on the same map as the part, lifted out of the black the way the part's darkest colors are. The
// tracers are one THREE.Points object moved on the CPU each frame, a few hundred to a little over a thousand of
// them, so a phone keeps up. Where they start is fixed, so a still drawn at build time shows them where the live
// view first does.
import { AdditiveBlending, BufferAttribute, BufferGeometry, NormalBlending, Points, ShaderMaterial } from 'three';
import { flowSlow } from './play-motion.js';

const lerp = (a, b, k) => a + (b - a) * k;

/** A small fixed sequence of numbers in [0, 1), so tracers start in the same places every time. */
function seeded(n) {
  let x = 0x9e3779b9;
  return Array.from({ length: n }, () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 4294967296; });
}

const VERT = /* glsl */ `
attribute vec3 tcolor;
attribute float talpha;
uniform float uSize;
uniform float uViewH;
varying vec3 vC;
varying float vA;
void main() {
  vC = tcolor;
  vA = talpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // a tracer as wide as uSize in the scene, never under two device pixels across
  gl_PointSize = max(2.0, uSize * projectionMatrix[1][1] * 0.5 * uViewH / max(-mv.z, 1e-6));
}`;

const FRAG = /* glsl */ `
varying vec3 vC;
varying float vA;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r = dot(q, q);
  if (r > 1.0) discard;
  // a soft round bead with a brighter core
  float a = vA * (1.0 - smoothstep(0.35, 1.0, r));
  gl_FragColor = vec4(mix(vC, vec3(1.0), 0.25 * (1.0 - smoothstep(0.0, 0.3, r))), a);
}`;

/** Bytes of an sRGB color lifted out of the dark, as play-draw.js fieldColor lifts a map's darkest colors. */
function lifted(c, out, o) {
  const lin = c.map((x) => { const v = x / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  const y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  const k = Math.min(Math.max(0.06 / Math.max(y, 1e-4), 1), 4.5);
  for (let i = 0; i < 3; i++) {
    const v = Math.min(lin[i] * k, 1);
    out[o + i] = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
  }
}

/**
 * Tracers along the lines of a scene object o (decoded arrays d: P the points, sp the fluid's speed at each in
 * scene units per second, fv its value at each), colored by field (play-scene.js's field with its map) or the
 * line's color base. R is the scene's radius. Returns { object, advance(dt), at(a, b, k), slow }.
 */
export function flowTracers({ o, d, field = null, R, base = [143, 184, 222], count = 0, size = 0, slow: given = 0 }) {
  const lines = [];
  let p0 = 0, total = 0, vmax = 0;
  for (const r of o.runs) {
    const xs = [], ss = [], vs = [], fs = [];
    let len = 0;
    for (let i = 0; i < r; i++) {
      const j = (p0 + i) * 3, x = d.P[j], y = d.P[j + 1], z = d.P[j + 2];
      if (!Number.isFinite(x)) continue;
      if (xs.length) { const q = xs.length - 3; len += Math.hypot(x - xs[q], y - xs[q + 1], z - xs[q + 2]); }
      xs.push(x, y, z); ss.push(len); vs.push(Math.max(d.sp[p0 + i] || 0, 0)); fs.push(d.fv ? d.fv[p0 + i] : NaN);
    }
    p0 += r;
    const v = vs.reduce((m, x) => Math.max(m, x), 0);
    if (xs.length < 6 || len <= 0 || v <= 0) continue;
    vmax = Math.max(vmax, v);
    total += len;
    lines.push({ xs, ss, vs, fs, len });
  }
  const slow = given || flowSlow(vmax, R);
  // about one tracer every 3% of the scene's size along the lines, a few hundred to 1,400 in all
  const n = lines.length ? Math.round(Math.min(Math.max(count || total / (R * 0.03), 200), 1400)) : 0;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), alpha = new Float32Array(n).fill(0.9);
  const which = new Uint16Array(n), s = new Float32Array(n), seg = new Uint32Array(n);
  const rnd = seeded(n * 2);
  // tracers share the lines by length, each at a fixed share of the way along
  let acc = 0, li = 0;
  for (let i = 0; i < n; i++) {
    const target = ((i + rnd[i]) / n) * total;
    while (li < lines.length - 1 && acc + lines[li].len < target) acc += lines[li++].len;
    which[i] = li;
    s[i] = Math.min(Math.max(target - acc, 0), lines[li].len);
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('tcolor', new BufferAttribute(col, 3));
  geo.setAttribute('talpha', new BufferAttribute(alpha, 1));
  const colored = !!(field && d.fv);
  const mat = new ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
    blending: colored ? NormalBlending : AdditiveBlending,
    uniforms: { uSize: { value: size || R * 0.012 }, uViewH: { value: 600 } },
  });
  const object = new Points(geo, mat);
  object.frustumCulled = false;
  object.renderOrder = 3;
  const sc = o.field && o.field.scale, off = o.field && o.field.base !== undefined ? o.field.base : 0;
  let mult = 1, add = 0;
  const tmp = [0, 0, 0], c8 = [0, 0, 0];
  function place() {
    for (let i = 0; i < n; i++) {
      const L = lines[which[i]], ss = L.ss;
      let j = seg[i];
      // the segment that holds the tracer, searched on from where it was
      if (ss[j] > s[i] || j >= ss.length - 1) j = 0;
      while (j < ss.length - 2 && ss[j + 1] < s[i]) j++;
      seg[i] = j;
      const k = (s[i] - ss[j]) / ((ss[j + 1] - ss[j]) || 1);
      for (let c = 0; c < 3; c++) pos[i * 3 + c] = lerp(L.xs[j * 3 + c], L.xs[j * 3 + 3 + c], k);
      // fade in at the start of a line and out at its end, so a tracer never pops
      const u = s[i] / L.len;
      alpha[i] = 0.9 * Math.min(1, u / 0.06, (1 - u) / 0.06);
      if (colored) {
        const v = add + mult * lerp(L.fs[j], L.fs[j + 1], k), f = field;
        if (!Number.isFinite(v)) { tmp[0] = 0.33; tmp[1] = 0.35; tmp[2] = 0.37; }
        else {
          const q = Math.min(255, Math.max(0, Math.round(((v - f.lo) / (f.hi - f.lo || 1)) * 255))) * 3;
          c8[0] = f.lut[q]; c8[1] = f.lut[q + 1]; c8[2] = f.lut[q + 2];
          lifted(c8, tmp, 0);
        }
        col[i * 3] = tmp[0]; col[i * 3 + 1] = tmp[1]; col[i * 3 + 2] = tmp[2];
      } else {
        col[i * 3] = base[0] / 255; col[i * 3 + 1] = base[1] / 255; col[i * 3 + 2] = base[2] / 255;
      }
    }
    geo.getAttribute('position').needsUpdate = true;
    geo.getAttribute('tcolor').needsUpdate = true;
    geo.getAttribute('talpha').needsUpdate = true;
  }
  place();
  return {
    object, slow, count: n,
    /** Move every tracer on by dt seconds of playback: the fluid's own speed there, slowed by slow. */
    advance(dt, rate = 1) {
      if (!n || !(dt > 0)) return;
      const step = Math.min(dt, 0.1) * rate / slow;
      for (let i = 0; i < n; i++) {
        const L = lines[which[i]], j = seg[i], k = (s[i] - L.ss[j]) / ((L.ss[j + 1] - L.ss[j]) || 1);
        s[i] += Math.max(lerp(L.vs[j], L.vs[Math.min(j + 1, L.vs.length - 1)], k), vmax * 0.02) * step;
        if (s[i] >= L.len) { s[i] -= L.len; seg[i] = 0; }
      }
      place();
    },
    /** The input frame the view stands at, for a fluid whose values scale with it (base + scale x value). */
    at(a, b, k) {
      const m = sc ? lerp(sc[a], sc[b] ?? sc[a], k) : 1, ad = sc ? off : 0;
      if (m === mult && ad === add) return;
      mult = m; add = ad;
      place();
    },
    /** Values drawn as add + mult x the line's own, as a plume whose warming grows while the part warms up. */
    level(m, ad = 0) {
      if (m === mult && ad === add) return;
      mult = m; add = ad;
      place();
    },
    /** The view's height in device pixels, for the tracers' size on screen. */
    view(h) { mat.uniforms.uViewH.value = h; },
    dispose() { geo.dispose(); mat.dispose(); },
  };
}

/**
 * Move a stage part's tracers on to the frame's flow clock f.ft (seconds of the view's own time, which the page
 * runs while the view is on screen), and size them for a view hpx device pixels tall. A part's tracers are its
 * scene's (play-scene.js) or its own (a heat run's air, part.flows), with part.flowRate(f) for a speed that
 * grows as the part warms. True when any tracer moved.
 */
export function stepFlows(part, f, hpx) {
  const flows = part.sp ? part.sp.flows : part.flows;
  if (!flows || !flows.length) return false;
  const ft = Number.isFinite(f.ft) ? f.ft : 0, dt = part.ft === undefined ? 0 : ft - part.ft;
  part.ft = ft;
  const rate = part.flowRate ? part.flowRate(f) : 1;
  for (const fl of flows) {
    fl.view(hpx);
    if (dt > 0) fl.advance(dt, rate);
  }
  return dt > 0;
}

/**
 * A heat run's air (scripts/sitegen/export_playback.py still_air): the lines of its estimated plume as tracers,
 * colored cool to warm by the air's rise above the room, with the plume's speed and warming
 * following the warm up: speed as its share of the steady rise to the speed_exp, warming to the rise_exp.
 * Returns the tracers and part.flowRate's function, or null.
 */
export function heatAir(run, arrays, R, lutOf) {
  const a = run.air, A = arrays.air;
  if (!a || !A) return null;
  const n = a.points, P = new Float32Array(n * 3), sp = new Float32Array(n), fv = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    P[i * 3] = A[i * 5]; P[i * 3 + 1] = A[i * 5 + 1]; P[i * 3 + 2] = A[i * 5 + 2];
    sp[i] = A[i * 5 + 3]; fv[i] = A[i * 5 + 4];
  }
  // cool to warm over the air's own rise, as the run's scene colors it
  const field = { lut: lutOf('coolwarm'), lo: 0, hi: a.top };
  const fl = flowTracers({ o: { runs: a.runs }, d: { P, sp, fv }, field, R, slow: a.slow, size: R * 0.014 });
  const rate = (f) => {
    const th = Math.min(Math.max(f.k ?? 1, 0), 1);
    fl.level(th ** a.rise_exp);
    return th ** a.speed_exp;
  };
  return { fl, rate };
}
