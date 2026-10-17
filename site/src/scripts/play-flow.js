// Tracers that ride a fluid's lines (streamlines from a solved flow, or a convection plume Wobbl estimated) as fast
// as the fluid goes at each point of them, slowed by one factor for the whole view so a flow of metres a second
// shows on a part a few centimetres across. Each tracer is a short streak, a bright head with a tail of beads behind
// it along its line, as long as the way the fluid there goes in a fifth of a second of playback and fading out, so
// fast fluid draws long streaks and slow fluid short ones and the whole reads as a fluid moving. Every bead takes
// the color of the fluid's value where it is, such as its temperature, on the same map as the part, lifted out of
// the black the way the part's darkest colors are, so a streak cools along its length as the air does. The beads
// are one THREE.Points object moved on the CPU each frame, up to about eleven thousand, which a phone keeps up
// with. Where the tracers start is fixed, so a still drawn at build time shows them where the live view first does.
// A line's flow (scene-file.js: a glow moving along each line at a speed per line and frame, as current along a
// wire, faster for more of it) rides the same way, at the speed the scene gives each line and backward where it is
// negative, with no slowing since the scene's speeds are already in seconds of playback.
import { AdditiveBlending, BufferAttribute, BufferGeometry, NormalBlending, Points, ShaderMaterial } from 'three';
import { flowSlow } from './play-motion.js';

const lerp = (a, b, k) => a + (b - a) * k;
const TAIL = 7;           // beads behind each tracer's head
const TRAIL = 0.3;        // seconds of playback a streak spans

/** A small fixed sequence of numbers in [0, 1), so tracers start in the same places every time. */
function seeded(n) {
  let x = 0x9e3779b9;
  return Array.from({ length: n }, () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 4294967296; });
}

const VERT = /* glsl */ `
attribute vec3 tcolor;
attribute float talpha;
attribute float tsize;
uniform float uSize;
uniform float uViewH;
varying vec3 vC;
varying float vA;
void main() {
  vC = tcolor;
  vA = talpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // a bead as wide as uSize in the scene, a tail's beads narrower, never under two device pixels across
  gl_PointSize = max(2.0, tsize * uSize * projectionMatrix[1][1] * 0.5 * uViewH / max(-mv.z, 1e-6));
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

/** A map's 256 colors as floats, lifted out of the dark as play-draw.js fieldColor lifts a map's darkest colors. */
function liftedLut(lut) {
  const out = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const lin = [0, 1, 2].map((c) => { const v = lut[i * 3 + c] / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    const y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    const k = Math.min(Math.max(0.06 / Math.max(y, 1e-4), 1), 4.5);
    for (let c = 0; c < 3; c++) {
      const v = Math.min(lin[c] * k, 1);
      out[i * 3 + c] = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
    }
  }
  return out;
}

/**
 * Tracers along the lines of a scene object o (decoded arrays d: P the points, sp the fluid's speed at each in
 * scene units per second, fv its value at each), colored by field (play-scene.js's field with its map) or the
 * line's color base. R is the scene's radius. through draws them over the solid the fluid runs inside (a cold
 * plate's channel), as a section would show it. glows is a line's flow ({ speed: [per frame, per line], line_speed,
 * spacing }), whose speeds take the place of the fluid's. Returns { object, advance(dt, rate), at(a, b, k), slow }.
 */
export function flowTracers({ o, d, field = null, R, base = [143, 184, 222], count = 0, size = 0, slow: given = 0, through = false, glows = null }) {
  const lines = [];
  let p0 = 0, total = 0, vmax = 0, ri = -1;
  for (const r of o.runs) {
    ri++;
    const xs = [], ss = [], vs = [], fs = [];
    let len = 0;
    for (let i = 0; i < r; i++) {
      const j = (p0 + i) * 3, x = d.P[j], y = d.P[j + 1], z = d.P[j + 2];
      if (!Number.isFinite(x)) continue;
      if (xs.length) { const q = xs.length - 3; len += Math.hypot(x - xs[q], y - xs[q + 1], z - xs[q + 2]); }
      xs.push(x, y, z); ss.push(len); vs.push(glows ? 1 : Math.max(d.sp[p0 + i] || 0, 0)); fs.push(d.fv ? d.fv[p0 + i] : NaN);
    }
    p0 += r;
    const v = vs.reduce((m, x) => Math.max(m, x), 0);
    if (xs.length < 6 || len <= 0 || v <= 0) continue;
    vmax = Math.max(vmax, v);
    total += len;
    lines.push({ xs, ss, vs, fs, len, ri });
  }
  const slow = glows ? 1 : given || flowSlow(vmax, R);
  // each line's signed speed at the input frame shown, for a line's flow (1 for a fluid, whose speed is per point)
  const lm = new Float32Array(lines.length).fill(1);
  const top = glows ? Math.max(1e-9, ...(glows.speed || [glows.line_speed || [1]]).flat().map((v) => Math.abs(v || 0))) : 1;
  function glowAt(a, b, k) {
    if (!glows) return false;
    const sp = glows.speed, ls = glows.line_speed;
    let moved = false;
    lines.forEach((L, i) => {
      let v;
      if (ls) v = ls[L.ri];
      else if (sp.length === 1 && !Array.isArray(sp[0])) v = sp[0];
      else if (sp.length === 1) v = sp[0][L.ri];
      else { const A = sp[a] || sp[0], B = sp[b] || A; v = lerp(A[L.ri] ?? A[0], B[L.ri] ?? B[0], k); }
      v = Number.isFinite(v) ? v : 0;
      if (v !== lm[i]) { lm[i] = v; moved = true; }
    });
    return moved;
  }
  glowAt(0, 0, 0);
  // about one tracer every 3% of the scene's size along the lines (a line's flow: one every spacing it gives), a
  // few hundred to 1,400 in all, each a head and its tail of beads
  const every = glows ? Math.max(glows.spacing || R * 0.03, R * 0.004) : R * 0.03;
  const n = lines.length ? Math.round(Math.min(Math.max(count || total / every, glows ? 1 : 200), 1400)) : 0;
  const K = TAIL + 1, nb = n * K;
  const pos = new Float32Array(nb * 3), col = new Float32Array(nb * 3), alpha = new Float32Array(nb), bsize = new Float32Array(nb);
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
  // a tail's beads thin and fade toward its end
  const fadeOf = new Float32Array(K), sizeOf = new Float32Array(K);
  for (let j = 0; j < K; j++) { const u = j / K; fadeOf[j] = (1 - u) ** 1.4; sizeOf[j] = 1 - 0.45 * u; }
  for (let i = 0; i < nb; i++) bsize[i] = sizeOf[i % K];
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('tcolor', new BufferAttribute(col, 3));
  geo.setAttribute('talpha', new BufferAttribute(alpha, 1));
  geo.setAttribute('tsize', new BufferAttribute(bsize, 1));
  const colored = !!(field && d.fv);
  const bead = size || R * 0.012;
  const mat = new ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, depthTest: !through,
    blending: colored ? NormalBlending : AdditiveBlending,
    uniforms: { uSize: { value: bead }, uViewH: { value: 600 } },
  });
  const object = new Points(geo, mat);
  object.frustumCulled = false;
  object.renderOrder = 3;
  const sc = o.field && o.field.scale, off = o.field && o.field.base !== undefined ? o.field.base : 0;
  const lut = colored ? liftedLut(field.lut) : null;
  let mult = 1, add = 0, rate = 1;
  const gray = [0.33, 0.35, 0.37];
  /** The segment of line L that holds arc length x, searched from segment j. */
  const segOf = (ss, x, j) => {
    if (j >= ss.length - 1) j = ss.length - 2;
    while (j > 0 && ss[j] > x) j--;
    while (j < ss.length - 2 && ss[j + 1] < x) j++;
    return j;
  };
  function place() {
    for (let i = 0; i < n; i++) {
      const L = lines[which[i]], ss = L.ss;
      const j = segOf(ss, s[i], seg[i]);
      seg[i] = j;
      const k = (s[i] - ss[j]) / ((ss[j + 1] - ss[j]) || 1);
      // the streak's length: as far as the fluid at the head goes in TRAIL seconds of playback, at its pace now,
      // behind it the way it moves
      const m = lm[which[i]], dir = m < 0 ? -1 : 1;
      const v = Math.max(lerp(L.vs[j], L.vs[j + 1], k), vmax * 0.02) * rate * Math.abs(m);
      const step = (dir * v * TRAIL) / slow / TAIL;
      // a tail shorter than a bead is drawn as the head alone, so a still fluid shows beads, not a smear; a line's
      // flow that carries nothing at this frame shows none
      const tailA = Math.min(1, Math.abs(step) * TAIL / (bead * 1.5));
      const on = glows ? Math.min(1, Math.abs(m) / (0.04 * top)) : 1;
      let jj = j;
      for (let b = 0; b < K; b++) {
        const o3 = (i * K + b) * 3, x = s[i] - b * step;
        if ((b && (x < 0 || x > L.len || tailA <= 0.02)) || on <= 0) { alpha[i * K + b] = 0; continue; }
        jj = segOf(ss, x, jj);
        const kk = (x - ss[jj]) / ((ss[jj + 1] - ss[jj]) || 1);
        for (let c = 0; c < 3; c++) pos[o3 + c] = lerp(L.xs[jj * 3 + c], L.xs[jj * 3 + 3 + c], kk);
        // fade in at the start of a line and out at its end, so a tracer never pops
        const u = x / L.len;
        alpha[i * K + b] = 0.92 * on * fadeOf[b] * (b ? tailA : 1) * (glows ? 1 : Math.min(1, u / 0.06, (1 - u) / 0.06));
        if (colored) {
          const val = add + mult * lerp(L.fs[jj], L.fs[jj + 1], kk), f = field;
          if (!Number.isFinite(val)) { col[o3] = gray[0]; col[o3 + 1] = gray[1]; col[o3 + 2] = gray[2]; }
          else {
            const q = Math.min(255, Math.max(0, Math.round(((val - f.lo) / (f.hi - f.lo || 1)) * 255))) * 3;
            col[o3] = lut[q]; col[o3 + 1] = lut[q + 1]; col[o3 + 2] = lut[q + 2];
          }
        } else {
          col[o3] = base[0] / 255; col[o3 + 1] = base[1] / 255; col[o3 + 2] = base[2] / 255;
        }
      }
    }
    geo.getAttribute('position').needsUpdate = true;
    geo.getAttribute('tcolor').needsUpdate = true;
    geo.getAttribute('talpha').needsUpdate = true;
  }
  place();
  return {
    object, slow, count: n,
    /** Move every tracer on by dt seconds of playback: the fluid's own speed there, slowed by slow, at a share
     *  rate of its full pace (a plume speeding up as its part warms). */
    advance(dt, r = 1) {
      const moved = n && dt > 0;
      if (moved) {
        const step = Math.min(dt, 0.1) * r / slow;
        for (let i = 0; i < n; i++) {
          const L = lines[which[i]], j = seg[i], k = (s[i] - L.ss[j]) / ((L.ss[j + 1] - L.ss[j]) || 1);
          s[i] += Math.max(lerp(L.vs[j], L.vs[Math.min(j + 1, L.vs.length - 1)], k), vmax * 0.02) * step * lm[which[i]];
          if (s[i] >= L.len) { s[i] -= L.len; seg[i] = 0; } else if (s[i] < 0) { s[i] += L.len; seg[i] = L.ss.length - 2; }
        }
      }
      if (moved || r !== rate) { rate = r; place(); }
    },
    /** The input frame the view stands at, for a fluid whose values scale with it (base + scale x value). */
    at(a, b, k) {
      const m = sc ? lerp(sc[a], sc[b] ?? sc[a], k) : 1, ad = sc ? off : 0;
      const g = glowAt(a, b, k);
      if (m === mult && ad === add && !g) return;
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
    fl.advance(dt, rate);
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
