// Drawing a run's playback with three.js, the way the landing page's part is drawn (part-draw.js): the
// same lighting, the same turbo colors for stress and hatching over zones Wobbl excludes from the
// answer, with inferno for temperature as Wobbl's own heatmaps use. One canvas holds one or two parts,
// each in its own part of the canvas, so a part and its fix load up side by side. Every frame only
// changes uniforms, except a heat run's frames, which are copied in as the clock passes them.
import {
  BufferAttribute, BufferGeometry, ConeGeometry, CylinderGeometry, DoubleSide, Group, Mesh, MeshBasicMaterial,
  PerspectiveCamera, Scene, ShaderMaterial, SphereGeometry, Vector3, WebGLRenderer,
} from 'three';
import { sceneParts } from './play-scene.js';

const TURBO = /* glsl */ `
vec3 ramp(float x) {
  const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kr2 = vec2(-152.94239396, 59.28637943);
  const vec2 kg2 = vec2(4.27729857, 2.82956604);
  const vec2 kb2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2));
}`;

// Inferno as a polynomial (Matt Zucker, CC0), as matplotlib draws it to within a few percent
const INFERNO = /* glsl */ `
vec3 ramp(float t) {
  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  t = clamp(t, 0.0, 1.0);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}`;

const STRESS_VERT = /* glsl */ `
attribute float value;
attribute vec3 disp;
attribute float excl;
uniform float uK;
uniform float uExag;
uniform float uTop;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
${TURBO}
void main() {
  // linear elastic: at a fraction k of the run's load, every stress and displacement is k times the run's
  float x = uK * value / uTop;
  vC = x > 1.002 ? vec3(0.949, 0.949, 0.933) : ramp(x);
  vE = excl;
  vec3 p = position + (uExag * uK) * disp;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const HEAT_VERT = /* glsl */ `
attribute float t0;
attribute float t1;
uniform float uMix;
uniform float uK;
uniform vec2 uScale;
uniform float uAir;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
varying vec3 vW;
varying float vT;
${INFERNO}
void main() {
  // the temperature above the air, between two saved frames (or the steady field scaled for the warm up)
  float rise = uK * mix(t0, t1, uMix);
  vC = ramp((rise - uScale.x) / uScale.y);
  vE = 0.0;
  vW = position;
  vT = uAir + rise;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

// The light every playback surface is drawn in: a key light and a fill from the viewer's side, and a small
// highlight, so a strength run, a heat run and a scene all read the same. spec scales the highlight, which a
// prop keeps low so it reads matte.
export const LIT = /* glsl */ `
vec3 lit(vec3 c, vec3 n, vec3 v, float spec) {
  vec3 l1 = normalize(vec3(0.35, 0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.7, -0.1, 0.45));
  float d = 0.42 + 0.6 * max(dot(n, l1), 0.0) + 0.22 * max(dot(n, l2), 0.0);
  float s = spec * 0.16 * pow(max(dot(n, normalize(l1 + v)), 0.0), 28.0);
  return c * d + vec3(s);
}`;

const FRAG = /* glsl */ `
uniform float uPx;
${LIT}
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
#ifdef HEAT
uniform float uIso;
uniform vec3 uCutP;
uniform vec3 uCutS;
varying vec3 vW;
varying float vT;
#endif
void main() {
#ifdef HEAT
  // the corner cut away to show the inside: nothing of the outer surface is drawn there
  vec3 r = (vW - uCutP) * uCutS;
  if (r.x > 0.0 && r.y > 0.0 && r.z > 0.0) discard;
#endif
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 col = lit(vC, n, v, 1.0);
  if (vE > 0.5) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72);
  }
#ifdef HEAT
  // isotherms, a thin dark line at every multiple of uIso degrees, as wide on screen wherever they fall
  if (uIso > 0.0) {
    float f = vT / uIso, w = fwidth(f), d = abs(fract(f + 0.5) - 0.5);
    col = mix(col, vec3(0.055, 0.059, 0.067), 0.55 * (1.0 - smoothstep(0.6 * w, 1.6 * w, d)));
  }
#endif
  gl_FragColor = vec4(col, 1.0);
}`;

const YELLOW = 0xffd400;

const heatMaterial = (uniforms) => new ShaderMaterial({
  vertexShader: HEAT_VERT, fragmentShader: FRAG, side: DoubleSide, uniforms, defines: { HEAT: '' }, extensions: { derivatives: true },
});

/** The load as an arrow: a solid one, and a faint copy that shows through the part when it is behind it. */
export function arrow(color = YELLOW) {
  const g = new Group();
  const solid = new MeshBasicMaterial({ color });
  const ghost = new MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
  const parts = {};
  for (const [k, geo] of Object.entries({ shaft: new CylinderGeometry(1, 1, 1, 12), head: new ConeGeometry(3, 8, 20) })) {
    const m = new Mesh(geo, solid);
    const x = new Mesh(geo, ghost);
    x.renderOrder = 2;
    m.add(x);
    parts[k] = m;
    g.add(m);
  }
  g.userData = parts;
  return g;
}

/**
 * One canvas, one or more parts. Each part is { run, arrays, top, row } where top is the stress at the
 * top of the color scale and row the view it is drawn in. frame({ camera, views, k, heat }) draws every
 * part into its view rectangle ([x, y, w, h] in CSS pixels from the top left).
 */
export function createStage({ canvas, w, h, px, kind }) {
  const renderer = new WebGLRenderer({ canvas, antialias: px < 2, alpha: true, powerPreference: 'default' });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(px);
  renderer.setSize(w, h, false);
  renderer.setScissorTest(true);
  const camera = new PerspectiveCamera(30, 1, 1, 10000);
  camera.up.set(0, 0, 1);
  let cssW = w, cssH = h;
  const parts = [];

  function add({ run, arrays, top, row = 0 }) {
    if (kind === 'scene') {
      // a scene brings its own objects to the slider's position; the light, the camera and the views are these
      const sp = sceneParts({ arrays, run });
      const part = { run, arrays, scene: sp.scene, sp, row, uniforms: { uK: { value: 0 }, uPx: { value: px } }, load: null };
      parts.push(part);
      return part;
    }
    const scene = new Scene();
    const geo = new BufferGeometry();
    geo.setIndex(new BufferAttribute(arrays.index, 1));
    geo.setAttribute('position', new BufferAttribute(arrays.position, 3));
    geo.setAttribute('normal', new BufferAttribute(arrays.normal, 3, true));
    const uniforms = { uPx: { value: px }, uK: { value: 0 } };
    let material;
    const nv = arrays.position.length / 3;
    if (kind === 'strength') {
      geo.setAttribute('value', new BufferAttribute(arrays.value, 1));
      geo.setAttribute('disp', new BufferAttribute(arrays.disp, 3));
      geo.setAttribute('excl', new BufferAttribute(arrays.excl, 1, true));
      Object.assign(uniforms, { uExag: { value: run.exag }, uTop: { value: top } });
      material = new ShaderMaterial({ vertexShader: STRESS_VERT, fragmentShader: FRAG, side: DoubleSide, uniforms });
    } else {
      geo.setAttribute('t0', new BufferAttribute(new Float32Array(arrays.frames.subarray(0, nv)), 1));
      geo.setAttribute('t1', new BufferAttribute(new Float32Array(arrays.frames.subarray(0, nv)), 1));
      const [lo, hi] = run.scale_C;
      Object.assign(uniforms, {
        uMix: { value: 0 }, uScale: { value: [lo - run.air_C, hi - lo] }, uAir: { value: run.air_C }, uIso: { value: run.iso_C ?? 0 },
        uCutP: { value: new Vector3(...(run.cut ? run.cut.point : [0, 0, 0])) },
        uCutS: { value: new Vector3(...(run.cut ? run.cut.sign : [0, 0, 0])) },
      });
      material = heatMaterial(uniforms);
    }
    const mesh = new Mesh(geo, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    // the cut faces of a part cut open at its hottest point: the same colors and isotherms, never cut themselves
    let cut = null;
    if (kind === 'heat' && arrays.cutIndex) {
      const g = new BufferGeometry();
      g.setIndex(new BufferAttribute(arrays.cutIndex, 1));
      g.setAttribute('position', new BufferAttribute(arrays.cutPosition, 3));
      g.setAttribute('normal', new BufferAttribute(arrays.cutNormal, 3, true));
      const np = arrays.cutPosition.length / 3;
      g.setAttribute('t0', new BufferAttribute(new Float32Array(arrays.cutFrames.subarray(0, np)), 1));
      g.setAttribute('t1', new BufferAttribute(new Float32Array(arrays.cutFrames.subarray(0, np)), 1));
      const m = new Mesh(g, heatMaterial({ ...uniforms, uCutS: { value: new Vector3(0, 0, 0) } }));
      m.frustumCulled = false;
      scene.add(m);
      cut = { geo: g, np };
    }
    let load = null;
    if (kind === 'strength' && run.arrow) {
      load = arrow();
      scene.add(load);
    }
    const part = { run, arrays, scene, geo, uniforms, load, nv, row, cut, pair: [-1, -1] };
    parts.push(part);
    return part;
  }

  function remove(part) {
    const i = parts.indexOf(part);
    if (i >= 0) parts.splice(i, 1);
    part.scene.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
  }

  const Y = new Vector3(0, 1, 0), dir = new Vector3(), at = new Vector3();
  /** Point a part's arrow for the load fraction k: a pull starts at its point, a push ends on the face. */
  function aim(part, k, radius, bend) {
    const a = part.run.arrow, g = part.load;
    const len = 0.42 * radius * Math.max(k, 0.02);
    g.visible = k > 0.005;
    dir.fromArray(a.dir);
    const f = (bend === false ? 1 : part.run.exag) * k;
    at.set(a.point[0] + f * a.disp[0], a.point[1] + f * a.disp[1], a.point[2] + f * a.disp[2]);
    g.quaternion.setFromUnitVectors(Y, dir);
    const s = radius / 60; // arrow thickness in proportion to the part
    // the head keeps its shape as the arrow grows: as wide as it is long, in the cone's own proportions
    const head = Math.min(8 * s, len * 0.45), body = Math.max(len - head, 0.01), hs = head / 8;
    const { shaft, head: cone } = g.userData;
    shaft.scale.set(Math.min(s, hs), body, Math.min(s, hs));
    cone.scale.set(hs, hs, hs);
    if (a.pull) {
      g.position.copy(at);
      shaft.position.set(0, body / 2, 0);
      cone.position.set(0, body + head / 2, 0);
    } else {
      g.position.copy(at).addScaledVector(dir, -len);
      shaft.position.set(0, body / 2, 0);
      cone.position.set(0, body + head / 2, 0);
    }
  }

  /** The heat frames either side of a moment, on the surface and the cut faces: copied in only when they change. */
  function frames(part, i0, i1) {
    if (part.pair[0] === i0 && part.pair[1] === i1) return;
    const pairs = [[part.geo, part.arrays.frames, part.nv]];
    if (part.cut) pairs.push([part.cut.geo, part.arrays.cutFrames, part.cut.np]);
    for (const [geo, all, n] of pairs) {
      const a = geo.getAttribute('t0'), b = geo.getAttribute('t1');
      a.array.set(all.subarray(i0 * n, (i0 + 1) * n));
      b.array.set(all.subarray(i1 * n, (i1 + 1) * n));
      a.needsUpdate = true;
      b.needsUpdate = true;
    }
    part.pair = [i0, i1];
  }

  return {
    add,
    remove,
    /**
     * f.camera: { position, target, fov, aspect }; f.views: one [x, y, w, h] per part; f.k: the load
     * fraction or the warm up's share of the steady rise; f.heat: { i0, i1, mix } for a run with frames;
     * f.radius: the part's radius in mm, for the arrow.
     */
    frame(f) {
      const c = f.camera;
      camera.position.fromArray(c.position);
      camera.lookAt(c.target[0], c.target[1], c.target[2]);
      camera.fov = c.fov;
      camera.aspect = c.aspect;
      // a scene can be metres or kilometres across, so the clipping planes come with the camera
      if (c.near) { camera.near = c.near; camera.far = c.far; }
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
      renderer.setScissor(0, 0, cssW, cssH);
      renderer.setViewport(0, 0, cssW, cssH);
      renderer.clear();
      parts.forEach((part) => {
        const [x, y, vw, vh] = f.views[part.row];
        part.uniforms.uK.value = f.k;
        // the bending toggle: drawn larger than it is, or at its true size
        if (part.uniforms.uExag) part.uniforms.uExag.value = f.bend === false ? 1 : part.run.exag;
        if (part.sp) part.sp.set(f.scenes ? f.scenes[part.row] : f.scene);
        if (part.load) aim(part, f.k, f.radius, f.bend);
        if (f.heat) {
          frames(part, f.heat.i0, f.heat.i1);
          part.uniforms.uMix.value = f.heat.mix;
        }
        // three.js counts the viewport from the bottom left
        renderer.setViewport(x, cssH - y - vh, vw, vh);
        renderer.setScissor(x, cssH - y - vh, vw, vh);
        renderer.render(part.scene, camera);
      });
    },
    size(nw, nh, npx) {
      cssW = nw;
      cssH = nh;
      renderer.setPixelRatio(npx);
      renderer.setSize(nw, nh, false);
      for (const p of parts) p.uniforms.uPx.value = npx;
    },
    /** What the main field reads under a ray (origin and dir in the scene's units) in view row, or null. */
    probe(row, origin, dir) {
      const part = parts.find((p) => p.row === row);
      return part && part.sp ? part.sp.probe(origin, dir) : null;
    },
    compile() {
      for (const p of parts) renderer.compile(p.scene, camera);
    },
    dispose() {
      for (const p of [...parts]) remove(p);
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
