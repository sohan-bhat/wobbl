// Drawing a run's playback with three.js, the way the landing page's part is drawn (part-draw.js): the
// same lighting, the same turbo colors for stress and hatching over zones Wobbl excludes from the
// answer, with inferno for temperature as Wobbl's own heatmaps use. One canvas holds one or two parts,
// each in its own part of the canvas, so a part and its fix load up side by side. Every frame only
// changes uniforms, except a heat run's frames, which are copied in as the clock passes them.
import {
  BufferAttribute, BufferGeometry, ConeGeometry, CylinderGeometry, DoubleSide, Group, Mesh, MeshBasicMaterial,
  MeshPhysicalMaterial, LineSegments, NoToneMapping, VSMShadowMap, ShadowMaterial, PMREMGenerator, PerspectiveCamera, PlaneGeometry, Scene, ShaderMaterial,
  SRGBColorSpace, Vector3, WebGLRenderer,
} from 'three';
import { sceneParts } from './play-scene.js';
import { instance } from './play-models.js';
import {
  FINISHES, LOOK, SRGB, aimKey, decodeStudioBytes, depthMaterial, finishOf, finishShader, keyLight, loadStudio, studioEnvironment,
  studioShader, turnStudio,
} from './play-look.js';

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

// The light every playback surface is drawn in, so a strength run, a heat run and a scene all read the same.
// It is worked out in linear light and tone mapped with Khronos PBR Neutral before it goes out as sRGB, the
// way model-viewer draws a product, so a color map keeps its own hues on every face. The lights move with the
// camera as ParaView's light kit does: a soft room light, brighter from above (three.js's RoomEnvironment,
// which model-viewer uses), a key light up and to the right, a fill from the left, a little headlight, and a
// rim on the silhouette that keeps dark parts apart from the dark page. spec scales the highlight and the rim,
// which a prop keeps low so it reads matte.
export const LIT = /* glsl */ `
vec3 toLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 toSRGB(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 neutral(vec3 c) {
  const float start = 0.76, desat = 0.15;
  float x = min(c.r, min(c.g, c.b));
  c -= x < 0.08 ? x - 6.25 * x * x : 0.04;
  float peak = max(c.r, max(c.g, c.b));
  if (peak < start) return c;
  const float d = 1.0 - start;
  float top = 1.0 - d * d / (peak + d - start);
  c *= top / peak;
  float g = 1.0 - 1.0 / (desat * (peak - top) + 1.0);
  return mix(c, vec3(top), g);
}
vec3 lit(vec3 c, vec3 n, vec3 v, float spec) {
  vec3 a = toLinear(c);
  // a very dark color (the bottom of turbo, a black fin) is lifted until it reads as a lit solid with visible
  // faces, keeping its hue: dark colors hide shading (Moreland). Props keep their darkness, as context.
  float y = dot(a, vec3(0.2126, 0.7152, 0.0722));
  if (spec > 0.5) a *= clamp(0.06 / max(y, 1e-4), 1.0, 4.5);
  vec3 key = normalize(vec3(0.42, 0.62, 0.66));
  vec3 fill = normalize(vec3(-0.78, 0.08, 0.42));
  float nv = max(dot(n, v), 0.0);
  float room = mix(0.30, 0.62, 0.5 + 0.5 * n.y);
  float d = room + 0.92 * max(dot(n, key), 0.0) + 0.24 * max(dot(n, fill), 0.0) + 0.10 * nv;
  float fres = pow(1.0 - nv, 4.0);
  vec3 rim = (0.06 + 0.14 * spec) * fres * (0.55 + 0.45 * max(n.y, 0.0)) * (0.35 + 0.65 * sqrt(a));
  float s = spec * (0.12 * pow(max(dot(n, normalize(key + v)), 0.0), 48.0) + 0.03 * fres);
  return toSRGB(neutral(a * d + rim + vec3(s)));
}
// an outline darker than its face, or lighter where the face is too dark for a darker line to show
vec3 edgeTone(vec3 col) {
  float y = dot(col, vec3(0.2126, 0.7152, 0.0722));
  return mix(col + vec3(0.11), col * 0.4, smoothstep(0.1, 0.28, y));
}`;

const FRAG = /* glsl */ `
uniform float uPx;
#ifdef EDGE
uniform float uEdgeA;
#endif
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
  // the hatching is drawn only on a face turned toward the viewer: the hidden face behind an edge (a fixed
  // base under a plate's side) ties with the near face along the edge itself and would show there as teeth
  float facing = dot(n, v);
  n = faceforward(n, -v, n);
  vec3 col = lit(vC, n, v, 1.0);
#ifdef EDGE
  col = edgeTone(col);
#else
  // and it fades out on a face seen nearly edge on, where its stripes would only crowd into dashes
  if (vE > 0.5 && facing > 0.25) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72 * smoothstep(0.25, 0.45, facing));
  }
#endif
#ifdef HEAT
  // isotherms, a thin dark line at every multiple of uIso degrees, as wide on screen wherever they fall
  if (uIso > 0.0) {
    float f = vT / uIso, w = fwidth(f), d = abs(fract(f + 0.5) - 0.5);
    col = mix(col, vec3(0.055, 0.059, 0.067), 0.55 * (1.0 - smoothstep(0.6 * w, 1.6 * w, d)));
  }
#endif
#ifdef EDGE
  gl_FragColor = vec4(col, uEdgeA);
#else
  gl_FragColor = vec4(col, 1.0);
#endif
}`;

const YELLOW = 0xffd400;

// ---- the part's surface, a physically based material in the studio's light (play-look.js) --------------
// The field's colors are the part's paint: the base color three.js's MeshPhysicalMaterial lights, so the map
// reads on every face, while its metalness, roughness, clear coat and surface texture come from the material
// the run names. The darkest colors of a map are lifted until they read as a lit solid, as before.
const FIELD_COLOR = /* glsl */ `
${SRGB}
vec3 fieldColor(vec3 c) {
  vec3 a = toLinear(c);
  float y = dot(a, vec3(0.2126, 0.7152, 0.0722));
  return a * clamp(0.06 / max(y, 1e-4), 1.0, 4.5);
}`;

const INK = 'vec3(0.0042, 0.0047, 0.0056)';
export const FIELD_SPEC = { value: 0.45 };
export const EDGE_ALPHA = { value: 0.4 };

function stressShader(sh) {
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', `#include <common>
attribute float value;
attribute vec3 disp;
attribute float excl;
uniform float uK;
uniform float uExag;
uniform float uTop;
varying vec3 vC;
varying float vE;
${TURBO}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
  transformed += (uExag * uK) * disp;
  {
    // linear elastic: at a fraction k of the run's load, every stress and displacement is k times the run's
    float x = uK * value / uTop;
    vC = x > 1.002 ? vec3(0.949, 0.949, 0.933) : ramp(x);
    vE = excl;
  }`);
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', `#include <common>
uniform float uPx;
varying vec3 vC;
varying float vE;
${FIELD_COLOR}`)
    .replace('#include <color_fragment>', '#include <color_fragment>\n  diffuseColor.rgb = fieldColor(vC);')
    .replace('#include <opaque_fragment>', `
  // the hatching over excluded zones, only on a face turned toward the viewer and faded out as it turns edge on
  if (vE > 0.5) {
    float facing = dot(normalize(vNormal), normalize(vViewPosition));
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (facing > 0.25 && k < 2.5 * uPx) outgoingLight = mix(outgoingLight, ${INK}, 0.72 * smoothstep(0.25, 0.45, facing));
  }
#include <opaque_fragment>`);
}

function heatShader(sh) {
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', `#include <common>
attribute float t0;
attribute float t1;
uniform float uMix;
uniform float uK;
uniform vec2 uScale;
uniform float uAir;
varying vec3 vC;
varying vec3 vW;
varying float vT;
${INFERNO}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
  {
    // the temperature above the air, between two saved frames (or the steady field scaled for the warm up)
    float rise = uK * mix(t0, t1, uMix);
    vC = ramp((rise - uScale.x) / uScale.y);
    vW = position;
    vT = uAir + rise;
  }`);
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', `#include <common>
uniform float uIso;
uniform vec3 uCutP;
uniform vec3 uCutS;
varying vec3 vC;
varying vec3 vW;
varying float vT;
${FIELD_COLOR}`)
    .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
  // the corner cut away to show the inside: nothing of the outer surface is drawn there
  vec3 cutR = (vW - uCutP) * uCutS;
  if (cutR.x > 0.0 && cutR.y > 0.0 && cutR.z > 0.0) discard;`)
    .replace('#include <color_fragment>', '#include <color_fragment>\n  diffuseColor.rgb = fieldColor(vC);')
    .replace('#include <opaque_fragment>', `
  // isotherms, a thin dark line at every multiple of uIso degrees, as wide on screen wherever they fall
  if (uIso > 0.0) {
    float f = vT / uIso, w = fwidth(f), d = abs(fract(f + 0.5) - 0.5);
    outgoingLight = mix(outgoingLight, ${INK}, 0.55 * (1.0 - smoothstep(0.6 * w, 1.6 * w, d)));
  }
#include <opaque_fragment>`);
}

/** A strength or heat part's faces: its finish (play-look.js FINISHES), the field's uniforms shared with its outlines. */
function partMaterial(kind, uniforms, finish, mm = 1) {
  const f = FINISHES[finish] || FINISHES.painted;
  const m = new MeshPhysicalMaterial({
    color: 0xffffff, metalness: f.metalness, roughness: f.roughness, clearcoat: f.clearcoat, clearcoatRoughness: 0.16,
    // a softer reflection than bare paint would give, so a white softbox never washes a hot spot out
    specularIntensity: FIELD_SPEC.value,
    side: DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    studioShader(sh);
    finishShader(sh, finish, mm);
    (kind === 'strength' ? stressShader : heatShader)(sh);
  };
  m.customProgramCacheKey = () => `wobbl-part-${kind}-${finish}`;
  m.extensions = { derivatives: true };
  return m;
}

/** The outlines of a strength or heat part: the same field, lit the old simple way and darkened. */
function partEdgeMaterial(kind, uniforms) {
  uniforms.uEdgeA = { value: EDGE_ALPHA.value };
  return new ShaderMaterial({
    vertexShader: kind === 'strength' ? STRESS_VERT : HEAT_VERT, fragmentShader: FRAG, side: DoubleSide, uniforms,
    defines: kind === 'strength' ? { EDGE: '' } : { EDGE: '', HEAT: '' }, extensions: { derivatives: true },
    // faint, as a real part's edges only catch a little light: enough to keep a fin apart from the next
    transparent: true, depthWrite: false,
  });
}

/**
 * A part's crease and border edges as thin dark lines over its faces (part.js's creases): the same vertices
 * as its faces and a material that colors them by the same field, so they bend and color with it. The faces
 * are pushed back a little in depth so the lines win.
 */
export function edgeLines(geo, edges, material, faces) {
  if (!edges || !edges.length) return null;
  const g = new BufferGeometry();
  for (const [k, a] of Object.entries(geo.attributes)) g.setAttribute(k, a);
  g.setIndex(new BufferAttribute(edges, 1));
  material.polygonOffset = false;
  material.side = DoubleSide;
  if (faces) {
    faces.polygonOffset = true;
    faces.polygonOffsetFactor = Math.max(faces.polygonOffsetFactor, 1);
    faces.polygonOffsetUnits = Math.max(faces.polygonOffsetUnits, 1);
  }
  const lines = new LineSegments(g, material);
  lines.frustumCulled = false;
  return lines;
}

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
  // linear light out to sRGB; the tone mapping is PBR Neutral inside each lit material (play-look.js)
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = VSMShadowMap;
  // the key light's shadow is drawn only when the part moves or the light does, not as the camera orbits
  renderer.shadowMap.autoUpdate = false;
  let env = null, studioAz = null;
  const lookAt = new Vector3();
  renderer.setPixelRatio(px);
  renderer.setSize(w, h, false);
  renderer.setScissorTest(true);
  renderer.autoClear = false;
  const camera = new PerspectiveCamera(30, 1, 1, 10000);
  camera.up.set(0, 0, 1);
  let cssW = w, cssH = h;
  const parts = [];
  // behind each view a soft pool of light, as a studio backdrop or model-viewer's floor glow gives, so dark
  // parts keep their outline and the view has depth, fading to the page's own black at the edges
  const backdrop = new Scene();
  const glow = new Mesh(new PlaneGeometry(2, 2), new ShaderMaterial({
    depthTest: false, depthWrite: false,
    uniforms: { uAspect: { value: 1 } },
    vertexShader: 'varying vec2 vP; void main() { vP = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: /* glsl */ `
      uniform float uAspect;
      varying vec2 vP;
      void main() {
        vec2 q = vec2(vP.x * max(uAspect, 1.0), vP.y * max(1.0 / uAspect, 1.0));
        float r = length(q * vec2(0.82, 1.0) - vec2(0.0, 0.1));
        float k = 1.0 - smoothstep(0.05, 1.25, r);
        vec3 c = mix(vec3(0.055, 0.063, 0.067), vec3(0.106, 0.118, 0.133), k * k);
        // a little noise against banding in so dark a gradient
        float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        gl_FragColor = vec4(c + (n - 0.5) / 255.0, 1.0);
      }`,
  }));
  glow.frustumCulled = false;
  backdrop.add(glow);

  function add({ run, arrays, top, row = 0 }) {
    if (kind === 'scene') {
      // a scene brings its own objects to the slider's position; the light, the camera and the views are these
      const sp = sceneParts({ arrays, run });
      sp.scene.environment = env && env.texture;
      const key = keyLight(px > 1.5 ? 2048 : 1024);
      sp.scene.add(key, key.target);
      const part = { run, arrays, scene: sp.scene, sp, row, key, dirty: true, uniforms: { uK: { value: 0 }, uPx: { value: px } }, load: null };
      parts.push(part);
      return part;
    }
    const scene = new Scene();
    scene.environment = env && env.texture;
    // the part's finish from the material its run names (lib/playback.ts adds it from the run's study)
    const finish = finishOf(run.material);
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
      material = partMaterial('strength', uniforms, finish);
    } else {
      geo.setAttribute('t0', new BufferAttribute(new Float32Array(arrays.frames.subarray(0, nv)), 1));
      geo.setAttribute('t1', new BufferAttribute(new Float32Array(arrays.frames.subarray(0, nv)), 1));
      const [lo, hi] = run.scale_C;
      Object.assign(uniforms, {
        uMix: { value: 0 }, uScale: { value: [lo - run.air_C, hi - lo] }, uAir: { value: run.air_C }, uIso: { value: run.iso_C ?? 0 },
        uCutP: { value: new Vector3(...(run.cut ? run.cut.point : [0, 0, 0])) },
        uCutS: { value: new Vector3(...(run.cut ? run.cut.sign : [0, 0, 0])) },
      });
      material = partMaterial('heat', uniforms, finish);
    }
    const mesh = new Mesh(geo, material);
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.customDepthMaterial = kind === 'strength'
      ? depthMaterial(uniforms, { pars: 'attribute vec3 disp;\nuniform float uK;\nuniform float uExag;', main: 'transformed += (uExag * uK) * disp;' }, null, 'stress')
      : depthMaterial(uniforms, { pars: 'varying vec3 vW;', main: 'vW = position;' },
        { pars: 'uniform vec3 uCutP;\nuniform vec3 uCutS;\nvarying vec3 vW;', main: 'vec3 cutR = (vW - uCutP) * uCutS;\nif (cutR.x > 0.0 && cutR.y > 0.0 && cutR.z > 0.0) discard;' }, 'heat');
    scene.add(mesh);
    const key = keyLight(px > 1.5 ? 2048 : 1024);
    scene.add(key, key.target);
    // a part held at its base stands on the studio's floor: only its shadow is drawn there, as model-viewer
    // draws a product's, so the page's own background still shows round it
    let floor = null;
    if (/z\s*=\s*min/.test(run.fixed || '')) {
      const lo = run.mesh.bounds_min, hi = run.mesh.bounds_max;
      const R = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
      floor = new Mesh(new PlaneGeometry(R * 8, R * 8), new ShadowMaterial({ opacity: 0.42, depthWrite: false }));
      floor.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - R * 1e-3);
      floor.receiveShadow = true;
      floor.renderOrder = -1;
      scene.add(floor);
    }
    const lines = edgeLines(geo, arrays.edges, partEdgeMaterial(kind, uniforms), material);
    if (lines) scene.add(lines);
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
      const m = new Mesh(g, partMaterial('heat', { ...uniforms, uCutS: { value: new Vector3(0, 0, 0) } }, finish));
      m.frustumCulled = false;
      scene.add(m);
      cut = { geo: g, np };
    }
    let load = null;
    if (kind === 'strength' && run.arrow) {
      load = arrow();
      scene.add(load);
    }
    const part = { run, arrays, scene, geo, uniforms, load, nv, row, cut, key, dirty: true, pair: [-1, -1] };
    parts.push(part);
    return part;
  }

  function remove(part) {
    const i = parts.indexOf(part);
    if (i >= 0) parts.splice(i, 1);
    part.scene.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
  }

  /** Put the camera where a frame says: { position, target, fov, aspect, near, far }, with the studio turned with it. */
  function look(c) {
    lookAt.fromArray(c.target);
    camera.position.fromArray(c.position);
    camera.lookAt(c.target[0], c.target[1], c.target[2]);
    camera.fov = c.fov;
    camera.aspect = c.aspect;
    // a scene can be metres or kilometres across, so the clipping planes come with the camera
    if (c.near) { camera.near = c.near; camera.far = c.far; }
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
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

  /**
   * Light the stage with the studio photograph at url (or its bytes), prefiltered for this renderer, or with
   * three.js's RoomEnvironment when there is none or it cannot be had, so a still drawn offline is lit too.
   */
  async function light(source) {
    let rt = null;
    try {
      if (source) rt = studioEnvironment(renderer, await (typeof source === 'string' ? loadStudio(source) : decodeStudioBytes(source)));
    } catch (err) {
      rt = null;
    }
    if (!rt) {
      const { RoomEnvironment } = await import('three/examples/jsm/environments/RoomEnvironment.js');
      const pm = new PMREMGenerator(renderer);
      rt = pm.fromScene(new RoomEnvironment(renderer), 0.04);
      pm.dispose();
    }
    env?.dispose();
    env = rt;
    for (const p of parts) p.scene.environment = env.texture;
  }

  /**
   * Put a part's models in (play-models.js loadModels): a scene's in place of its shapes, a strength or heat
   * run's round the part in its own millimetre frame.
   */
  function dress(part, loaded) {
    if (!loaded || !parts.includes(part)) return;
    part.dirty = true;
    if (part.sp) { part.sp.dress(loaded); return; }
    for (const e of loaded.spec.add || []) {
      const g = loaded.models.get(e.file);
      if (g) part.scene.add(instance(g, e).group);
    }
  }

  return {
    add,
    remove,
    light,
    dress,
    /**
     * f.camera: { position, target, fov, aspect }; f.views: one [x, y, w, h] per part; f.k: the load
     * fraction or the warm up's share of the steady rise; f.heat: { i0, i1, mix } for a run with frames;
     * f.radius: the part's radius in mm, for the arrow.
     */
    frame(f) {
      // the studio is set from the view the stage opens on and stays put as the camera orbits
      if (studioAz === null) {
        studioAz = Math.atan2(f.camera.position[1] - f.camera.target[1], f.camera.position[0] - f.camera.target[0]);
        turnStudio(studioAz);
      }
      look(f.camera);
      renderer.setScissor(0, 0, cssW, cssH);
      renderer.setViewport(0, 0, cssW, cssH);
      renderer.clear();
      parts.forEach((part) => {
        const [x, y, vw, vh] = f.views[part.row];
        part.uniforms.uK.value = f.k;
        // the bending toggle: drawn larger than it is, or at its true size
        if (part.uniforms.uExag) part.uniforms.uExag.value = f.bend === false ? 1 : part.run.exag;
        if (part.sp) {
          part.sp.set(f.scenes ? f.scenes[part.row] : f.scene);
          part.sp.shade(renderer);
        }
        if (part.load) aim(part, f.k, f.radius, f.bend);
        if (f.heat) {
          frames(part, f.heat.i0, f.heat.i1);
          part.uniforms.uMix.value = f.heat.mix;
        }
        // a view that follows its subject has a camera of its own for each pane
        if (f.cameras) look(f.cameras[part.row]);
        // three.js counts the viewport from the bottom left
        renderer.setViewport(x, cssH - y - vh, vw, vh);
        renderer.setScissor(x, cssH - y - vh, vw, vh);
        glow.material.uniforms.uAspect.value = vw / vh;
        renderer.render(backdrop, camera);
        renderer.clearDepth();
        // the shadow box covers what the camera sees round its target; drawn again only when something moved
        const r = camera.position.distanceTo(lookAt) * Math.tan((camera.fov * Math.PI) / 360) * 1.25;
        const box = aimKey(part.key, studioAz, lookAt, r);
        const state = part.sp ? JSON.stringify(f.scenes ? f.scenes[part.row] : f.scene) : `${f.k},${f.bend},${f.heat ? `${f.heat.i0},${f.heat.i1},${f.heat.mix}` : ''}`;
        if (part.dirty || box !== part.box || state !== part.state) renderer.shadowMap.needsUpdate = true;
        part.dirty = false;
        part.box = box;
        part.state = state;
        renderer.render(part.scene, camera);
        const ins = f.insets && f.insets[part.row];
        if (ins && part.sp) {
          // the whole scene in a corner of the view, on a solid ground with a hairline round it
          const [ix, iy, iw, ih] = ins.rect, bx = x + ix, by = cssH - (y + iy) - ih;
          renderer.setScissor(bx - 1, by - 1, iw + 2, ih + 2);
          renderer.setClearColor(0x2a2d32, 1);
          renderer.clear();
          renderer.setScissor(bx, by, iw, ih);
          renderer.setClearColor(0x141619, 1);
          renderer.clear();
          renderer.setClearColor(0x000000, 0);
          part.sp.set(ins.scene);
          look(ins.camera);
          renderer.setViewport(bx, by, iw, ih);
          renderer.clearDepth();
          // the inset keeps the main view's shadow, which it is too small to show
          renderer.render(part.scene, camera);
          look(f.cameras ? f.cameras[part.row] : f.camera);
        }
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
      env?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}

// the studio's settings and the finishes, for a page or a tool that tunes them
export { FINISHES, LOOK };
