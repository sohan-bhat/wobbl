// A run's scene drawn on the strength and heat playback's stage (play-draw.js): the same light, the same
// background and the same load arrows, with each of the scene's objects brought to the slider's position.
// Meshes come smooth shaded from play.js and are colored by their field, bodies are true primitives, lines
// are round tubes and props are solid and matte. The viewer's text labels and its readout list are left
// to the page, which shows only the peak marker and the probe.
import {
  AdditiveBlending, BackSide, Box3, BoxGeometry, BufferAttribute, BufferGeometry, CapsuleGeometry, Color, CustomBlending, CylinderGeometry,
  DoubleSide, DataTexture, FrontSide, Group, InstancedBufferAttribute, LinearFilter, InstancedMesh, LineBasicMaterial, LineSegments,
  Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, OneFactor, OneMinusSrcAlphaFactor, PlaneGeometry, Quaternion, Raycaster, Scene,
  ShaderMaterial, SphereGeometry, TorusGeometry, Vector3,
} from 'three';
import { LIT, arrow, edgeLines } from './play-draw.js';
import { FINISHES, SRGB, fieldStudioShader, finishOf, studioShader } from './play-look.js';
import { instance, keepColor, subject } from './play-models.js';
import { contactShadow } from './contact-shadow.js';
import { overlay } from './play-refine.js';
import { at2, colorHex, lut, magnifyOf, poseAt, series, shownAt } from './scene-file.js';
import { motionFrame, rankSum, sizeScale, trackPose, trackSeries, trackSize } from './play-motion.js';
import { flowTracers } from './play-flow.js';

const VERT = /* glsl */ `
attribute vec3 tint;
attribute float cutv;
#ifdef TUBE
// a line drawn as a tube grows or thins about its own axis, so it keeps its width on screen in a close view
attribute vec3 axis;
uniform float uTube;
#endif
#ifdef FIELD
attribute vec2 fval;
varying vec2 vF;
#endif
varying vec3 vC;
varying vec3 vN;
varying vec3 vV;
varying float vR;
void main() {
  vec3 c = tint;
  vec4 p = vec4(position, 1.0);
#ifdef TUBE
  p.xyz = axis + (position - axis) * uTube;
#endif
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
#ifdef FIELD
  vF = fval;
#endif
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
#ifdef FIELD
uniform sampler2D uLut;
uniform float uLimit;
varying vec2 vF;
#endif
${LIT}
void main() {
  // a print rising layer by layer, or a line drawn up to the moment shown
  if (vR > uCut) discard;
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 c = vC;
#ifdef FIELD
  // the field's value is carried across each triangle and only then looked up in the map, as ParaView does
  // by default, so a contour runs straight across the faces instead of along the mesh's vertices
  c = vF.y > 0.5 ? vec3(${NONE.map((x) => (x / 255).toFixed(4)).join(', ')})
    : vF.x > uLimit ? vec3(${OVER.map((x) => (x / 255).toFixed(4)).join(', ')})
    : texture2D(uLut, vec2(clamp(vF.x, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
#endif
  vec3 col = mix(lit(c, n, v, uSpec), uGlowC, uGlow);
#ifdef EDGE
  col = edgeTone(col);
#endif
  gl_FragColor = vec4(col, uAlpha);
}`;

const OVER = [242, 242, 238], NONE = [85, 88, 94];
const CPK = { H: 'f2f2f2', C: '555a60', N: '3b6fe0', O: 'e8423b', F: '7fd17f', P: 'ff8c1a', S: 'f2d230', Cl: '33d14a', Na: 'a35cf2', K: '8f40d4', Li: 'cc80ff', Mg: '8aff00', Al: 'bfa6a6', Si: 'd9b380', Fe: 'e06633', Cu: 'c88033', Zn: '7d80b0', Ti: 'bfc2c7', Ni: '50d050', Ca: '3dff00', B: 'ffb5b5', Ar: '80d1e3', He: 'd9ffff' };
const RAD = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, S: 1.05, P: 1.07, Cl: 1.02, Na: 1.66, K: 2.03, Li: 1.28, Fe: 1.32, Al: 1.21, Si: 1.11, Cu: 1.32, Ti: 1.6 };
const bytes = (c) => colorHex(c).map((x) => Math.round(x * 255));

// Nothing is drawn see through. Where a scene asked for a see through surface (glass, a mold box, a beaker,
// a shell round a field), it is drawn solid in the color it would have shown over the page's background,
// and when it is a closed shell or mostly clear, with its near side cut away so what it holds still shows. Props are drawn
// darker and matte, so the part the run is about stays the brightest thing in the view.
const BG = [14, 15, 17];
const behind = (rgb, a, least = 0.3) => rgb.map((c, i) => Math.round(BG[i] + (c - BG[i]) * Math.min(1, Math.max(a, least) * 1.25)));
const PROP = 0.82;

function fieldUniforms(field, uniforms) {
  const rgba = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) { rgba.set(field.lut.subarray(i * 3, i * 3 + 3), i * 4); rgba[i * 4 + 3] = 255; }
  const tex = new DataTexture(rgba, 256, 1);
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.needsUpdate = true;
  uniforms.uLut = { value: tex };
  uniforms.uLimit = { value: field.limit !== undefined ? (field.limit - field.lo) / (field.hi - field.lo || 1) : 1e30 };
}

/** A face material's depth alone, revealed and pushed back as the faces are. */
function prepass(faces) {
  // pushed back a hair further than the faces, so a face is never refused against its own depth
  const m = new MeshBasicMaterial({
    colorWrite: false, side: DoubleSide, polygonOffset: true,
    polygonOffsetFactor: faces.polygonOffsetFactor + 1.5, polygonOffsetUnits: faces.polygonOffsetUnits + 4,
  });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uCut = faces.uniforms.uCut;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float cutv;\nvarying float vR;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vR = cutv;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uCut;\nvarying float vR;')
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n  if (vR > uCut) discard;');
  };
  m.customProgramCacheKey = () => 'wobbl-scene-prepass';
  return m;
}

/** The flat lit material outlines are drawn in, sharing the face material's uniforms so they cut and color with it. */
function outline(faces, field) {
  return new ShaderMaterial({
    vertexShader: VERT, fragmentShader: frag(), uniforms: faces.uniforms, defines: field ? { FIELD: '', EDGE: '' } : { EDGE: '' },
    side: DoubleSide,
  });
}

// ---- the physically based look (play-look.js) ----
// Every object is a three.js MeshPhysicalMaterial in the studio's light, with what the scene stores on top: its
// color per vertex (tint), a field looked up per pixel, a print revealed layer by layer (cutv), a line's tube
// grown about its axis, a glow. A see through pane is real glass now: its reflections stay whole while what
// is behind it shows through.
function sceneShader(sh, { field, tube, glass, unlit, lift }) {
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', `#include <common>
attribute vec3 tint;
attribute float cutv;
varying vec3 vC;
varying float vR;
${tube ? 'attribute vec3 axis;\nuniform float uTube;' : ''}
${field ? 'attribute vec2 fval;\nvarying vec2 vF;' : ''}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
  vC = tint;
#ifdef USE_INSTANCING_COLOR
  vC = instanceColor;
#endif
  vR = cutv;
  ${tube ? 'transformed = axis + (transformed - axis) * uTube;' : ''}
  ${field ? 'vF = fval;' : ''}`);
  const none = NONE.map((x) => (x / 255).toFixed(4)).join(', '), over = OVER.map((x) => (x / 255).toFixed(4)).join(', ');
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', `#include <common>
uniform float uCut;
uniform float uGlow;
uniform vec3 uGlowC;
uniform float uAlpha;
varying vec3 vC;
varying float vR;
${field ? 'uniform sampler2D uLut;\nuniform float uLimit;\nvarying vec2 vF;' : ''}
${SRGB}`)
    .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
  // a print rising layer by layer, or a line drawn up to the moment shown
  if (vR > uCut) discard;`)
    .replace('#include <color_fragment>', `#include <color_fragment>
  vec3 baseC = vC;
  ${field ? `// the field's value carried across each triangle and only then looked up in the map, as ParaView does
  baseC = vF.y > 0.5 ? vec3(${none}) : vF.x > uLimit ? vec3(${over})
    : texture2D(uLut, vec2(clamp(vF.x, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;` : ''}
  diffuseColor.rgb = toLinear(baseC);
  ${lift ? `{
    // a very dark color of a map lifted until it reads as a lit solid, keeping its hue (Moreland)
    float y = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    diffuseColor.rgb *= clamp(0.06 / max(y, 1e-4), 1.0, 4.5);
  }` : ''}`)
    .replace('#include <opaque_fragment>', `
  ${unlit ? `{
    // a flame or a plume glows: brightest where it is thickest along the view, white hot in the core and fading
    // to nothing at its edges, added to what is behind it
    float through = pow(clamp(abs(dot(normalize(vNormal), normalize(vViewPosition))), 0.0, 1.0), 1.4);
    outgoingLight = mix(diffuseColor.rgb, vec3(1.0, 0.93, 0.8), 0.45 * through) * (0.35 + 1.6 * through);
  }` : ''}
  outgoingLight = mix(outgoingLight, toLinear(uGlowC) * 1.6, uGlow);
  ${glass ? `// glass: what is behind shows through, the reflections stay whole (premultiplied, one minus alpha)
  outgoingLight = totalDiffuse * uAlpha + totalSpecular;
  diffuseColor.a = clamp(uAlpha + dot(totalSpecular, vec3(0.3333)), 0.0, 1.0);` : 'diffuseColor.a = uAlpha;'}
#include <opaque_fragment>`);
}

function material(o, { prop = false, shell = false, field = null, tube = false, finish = 'painted', glass = false, unlit = false } = {}) {
  // a closed shell is cut open whenever it was asked to be see through, an open surface when mostly clear
  const a = o.opacity ?? 1, cut = !glass && a < 1 && (shell || a < 0.5);
  const uniforms = {
    uCut: { value: 1e30 }, uSpec: { value: prop ? 0.25 : 1 }, uGlow: { value: 0 }, uGlowC: { value: new Color(1, 0.85, 0.2) },
    uAlpha: { value: glass ? Math.min(0.32, Math.max(0.08, a * 0.45)) : 1 },
  };
  if (tube) uniforms.uTube = { value: 1 };
  if (field) fieldUniforms(field, uniforms);
  const f = FINISHES[finish] || FINISHES.painted;
  const m = new MeshPhysicalMaterial({
    metalness: f.metalness, roughness: f.roughness, clearcoat: f.clearcoat, clearcoatRoughness: 0.12,
    side: cut ? BackSide : glass ? FrontSide : DoubleSide,
    // a cut open shell gives way to whatever it touches, such as the plate a part sits on
    polygonOffset: cut, polygonOffsetFactor: cut ? 2 : 0, polygonOffsetUnits: cut ? 2 : 0,
  });
  if (glass) {
    Object.assign(m, { transparent: true, depthWrite: false, blending: CustomBlending, blendSrc: OneFactor, blendDst: OneMinusSrcAlphaFactor });
  }
  if (unlit) Object.assign(m, { transparent: true, depthWrite: false, blending: AdditiveBlending, side: FrontSide });
  const opts = { field: !!field, tube, glass, unlit, lift: !prop };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    // glass and a glow show what is behind them or their own light, which the resting view's occlusion would darken
    if (field && !prop) fieldStudioShader(sh);
    else studioShader(sh, !glass && !unlit);
    sceneShader(sh, opts);
  };
  m.customProgramCacheKey = () => `wobbl-scene-${+opts.field}${+tube}${+glass}${+unlit}${+opts.lift}${+!prop}`;
  // the uniforms the stage brings to each frame, as on the shader materials before
  m.uniforms = uniforms;
  // a part the run is about, colored by its field, shows through the models in front of it (play-models.js)
  return field && !prop && !glass ? subject(m) : m;
}

/** What an object is made of, from its name and what it is: a pin is bare steel, a motor's can black paint. */
function finishFor(o, run, pane, title) {
  const n = (o.name || '').toLowerCase();
  if (pane || /glass|windshield|window|pane\b|beaker|jar/.test(n)) return 'glass';
  if (/exhaust|flame|plume|fire|spark|glow/.test(n)) return 'emissive';
  if (!o.prop && o.field) {
    // a part colored by its field keeps the map readable: its finish is the material its run names, or what its
    // name says, never a bare mirror metal
    if (run.material) return finishOf(run.material);
    // the object's name, the example's (steel-hook) and the scene's title (an aluminum strut) say what it is
    const said = `${n} ${run.example || ''} ${title || ''}`.toLowerCase(), named = finishOf(said);
    if (named !== 'painted') return named;
    if (/hook|bolt|shaft|spring|rod\b|strut|beam|wire/.test(said)) return 'steel';
    if (/casing|closure|nozzle|bracket|plate|frame/.test(said)) return 'aluminum_machined';
    return 'painted';
  }
  if (/\bpin\b|shaft|axle|bolt|screw|nut\b|washer|rail|spindle|rod\b|spring|wire|hook|chain|bearing/.test(n)) return 'steel_bare';
  if (/nozzle|closure|casing|plate|flange|bracket|frame|stand|saddle/.test(n)) return o.prop ? 'aluminum_bare' : 'aluminum_machined';
  if (/motor|gearbox|housing|can\b/.test(n)) return 'paint_black';
  if (/rubber|tire|tyre|blade|seal|gasket/.test(n)) return 'rubber';
  if (/grain|propellant|wood|bench|desk|table/.test(n)) return 'matte';
  if (/bed\b|ground|floor|pad\b/.test(n)) return 'floor';
  return o.prop ? 'painted' : 'paint_gloss';
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
  const items = [], flows = [];
  // a view framed close on its subject (scene-view.js) leaves out lines that never come near it, such as the
  // arc a wiper blade's tip sweeps far up the glass, which would float in the close view without their context
  const sb = run.subject, near = sb && sb.lo.map((x, k) => [x - 0.25 * (sb.hi[k] - x), sb.hi[k] + 0.25 * (sb.hi[k] - x)]);
  const far = (P) => {
    for (let i = 0; i < P.length; i += 3) if (near.every(([l, u], k) => P[i + k] >= l && P[i + k] <= u)) return false;
    return true;
  };
  h.objects.forEach((o, oi) => {
    const d = arrays.objs[oi];
    if (!d) return;
    if (near && o.kind === 'lines' && far(d.P)) return;
    let rgb = bytes(o.color || '#9aa0a6');
    // a see through pane (a windshield) is glass
    const z = o.kind === 'body' && o.size, pane = o.prop && z && (o.shape === 'plane' || (o.shape === 'box' && Math.min(...z) < 0.05 * Math.max(...z)));
    const fin = o.kind === 'ground' ? 'floor' : finishFor(o, run, pane && (o.opacity ?? 1) < 1, h.title), glass = fin === 'glass' && o.kind !== 'mesh' || (fin === 'glass' && (o.opacity ?? 1) < 1);
    const metal = (FINISHES[fin] || FINISHES.painted).metalness > 0.5;
    if (o.prop && o.kind !== 'ground' && !metal && !glass) {
      // a prop is a little darker and grayer, so it stays behind the part the run is about
      const y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
      rgb = rgb.map((c) => Math.round((c + (y - c) * 0.15) * PROP));
    }
    const a = o.opacity ?? 1;
    const it = { o, d, oi, fin, glass, mag: magnifyOf(h, oi), f: o.field ? fields[o.field.f] : null, base: a < 1 && !glass ? behind(rgb, a) : rgb };
    const look = { prop: !!o.prop, finish: fin, glass, unlit: fin === 'emissive' };
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
      if (it.f && d.fv) geo.setAttribute('fval', new BufferAttribute(new Float32Array(nv * 2), 2));
      it.n = d.P.length / 3;
      it.cur = d.P.slice();
      it.val = new Float32Array(it.n).fill(NaN);
      const field = it.f && d.fv ? it.f : null;
      it.mesh = new Mesh(geo, material(o, { ...look, shell: d.closed, field }));
      it.colored = false;
      // a shell drawn cut open shows its inside, where outlines would only clutter it
      if (it.mesh.material.side !== BackSide && !glass) {
        const lines = edgeLines(geo, d.edges, outline(it.mesh.material, field), it.mesh.material);
        if (lines) it.mesh.add(lines);
      }
      // a big mesh is drawn into depth first, so its many small triangles are lit only where they are seen
      if (nv > 20000 && it.mesh.material.side === DoubleSide && !glass && fin !== 'emissive') {
        const pre = new Mesh(geo, prepass(it.mesh.material));
        pre.renderOrder = -2;
        pre.frustumCulled = false;
        it.mesh.add(pre);
      }
    } else if (o.kind === 'body') {
      it.mesh = new Mesh(tinted(primitive(o.shape, o.size), it.base), material(o, { ...look, shell: o.shape !== 'plane' }));
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
      if (d.track && d.track.drawn) {
        it.cutI = cut;
        it.cutM = new Float32Array(nv);
        for (let v = 0; v < nv; v++) it.cutM[v] = d.track.drawn[d.point[v]];
      }
      geo.setAttribute('axis', new BufferAttribute(d.axis, 3));
      it.mesh = new Mesh(geo, material({ ...o, opacity: 1 }, { prop: !!o.prop, tube: true, finish: 'paint_gloss' }));
      it.cuts = !!(d.td || o.at);
      // a fluid's lines carry its speed: tracers ride them as fast as it goes, slowed by a factor (play-flow.js)
      if (d.sp) {
        const fl = flowTracers({ o, d, field: it.f, R, base: it.base, through: !!o.inside });
        flows.push(fl);
        scene.add(overlay(fl.object));
      } else if (o.flow) {
        // a glow moving along each line as fast as the scene says, as current along a wire, drawn as tracers lit
        // a little brighter than the line they ride
        const fl = flowTracers({ o, d, R, base: it.base.map((c) => Math.round(c + (255 - c) * 0.6)), glows: o.flow, size: Math.max(R * 0.012, (o.flow.spacing || 0) * 0.18) });
        flows.push(fl);
        scene.add(overlay(fl.object));
      }
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
      it.mesh = new InstancedMesh(geo, material(o, { finish: 'paint_gloss' }), it.n);
      it.mesh.instanceColor = new InstancedBufferAttribute(cols, 3);
      it.mesh.frustumCulled = false;
      if (d.bonds && d.bonds.length) {
        const bg = tinted(new CylinderGeometry(1, 1, 1, 14), [150, 154, 160]);
        it.bonds = new InstancedMesh(bg, material(o, { finish: 'paint_gloss' }), d.bonds.length / 2);
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
      const plane = new Mesh(tinted(new PlaneGeometry(s, s), it.base), material(o, { prop: true, finish: 'floor' }));
      plane.receiveShadow = true;
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
    // a glowing body, or anything glowing over the motion (a flame), glows toward a light tint of its own color
    const gc = ((o.kind === 'body' && o.glow) || d.track?.glow) && it.mesh.material?.uniforms?.uGlowC;
    if (gc) gc.value.setRGB(...colorHex(o.color).map((v) => Math.min(1, v * 0.5 + 0.55)));
    // every solid casts and takes the key light's soft shadow; glass, glows and lines do neither
    if (['mesh', 'body', 'points', 'atoms'].includes(o.kind) && !glass && fin !== 'emissive') {
      it.mesh.castShadow = !it.d?.rv;
      it.mesh.receiveShadow = true;
      if (it.bonds) { it.bonds.castShadow = true; it.bonds.receiveShadow = true; }
    }
    // lines and arrows are left out of the depth the occlusion is worked out from, so they leave no smudge on the
    // faces behind them
    if (o.kind === 'lines' || o.kind === 'arrows') overlay(it.mesh);
    // whatever shows a field's colors, the part itself, a line or a body colored by its value and their outlines,
    // is kept out of the stage's occlusion, so the colors stay as the legend gives them (play-refine.js), and so is
    // anything that glows, which lights itself
    if ((it.f && !o.prop) || o.glow) it.mesh.traverse((x) => { if (x.material && !Array.isArray(x.material)) keepColor(x.material); });
    scene.add(it.mesh);
    items.push(it);
  });

  /** The object's matrix for a pose, grown about its anchor when it is magnified, or about its own pivot point
   * while it follows the motion (moving), as a rocket grown from its nozzle so it stands on its pad. */
  function place(obj, it, pose, mag, sz = null, moving = false) {
    const m = mag && it.mag;
    const s = m ? m.factor : 1;
    tmpQ.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
    let p = m && m.anchor ? m.anchor.map((x, i) => x + s * (pose.p[i] - x)) : pose.p;
    if (m && moving && m.pivot) {
      const v = tmpP.set(m.pivot[0], m.pivot[1], m.pivot[2]).applyQuaternion(tmpQ).multiplyScalar(1 - s);
      p = [pose.p[0] + v.x, pose.p[1] + v.y, pose.p[2] + v.z];
    }
    obj.matrixAutoUpdate = false;
    obj.matrix.compose(tmpP.set(p[0], p[1], p[2]), tmpQ, sz ? tmpS.set(s * sz[0], s * sz[1], s * sz[2]) : tmpS.set(s, s, s));
    obj.matrixWorldNeedsUpdate = true;
  }

  function meshFrame(it, a, b, k, bend, mf = null) {
    const { o, d } = it, n = it.n, nv = d.node.length, geo = it.mesh.geometry;
    // over the scene's motion a tracked shape and field take the place of the input's (play-motion.js)
    const tr = mf && d.track;
    if (tr && tr.deform) {
      const cur = it.cur, u = it.tu || (it.tu = new Float32Array(n * 3)), ex = bend ? tr.deform.ex : 1;
      rankSum(tr.deform, mf, n, 3, u);
      for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + u[i] * ex;
      const pa = geo.getAttribute('position');
      for (let v = 0; v < nv; v++) { const j = d.node[v] * 3; pa.array[v * 3] = cur[j]; pa.array[v * 3 + 1] = cur[j + 1]; pa.array[v * 3 + 2] = cur[j + 2]; }
      pa.needsUpdate = true;
      geo.boundingSphere = null;
    } else if (d.du) {
      const ex = bend ? o.deform.exaggerate || 1 : 1, sc = o.deform.scale, cur = it.cur;
      if (sc) { const s = series(sc, a, b, k) * ex; for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + d.du[i] * s; }
      else for (let i = 0; i < n * 3; i++) cur[i] = d.P[i] + at2(d.du, n * 3, a, b, k, i) * ex;
      const pa = geo.getAttribute('position');
      for (let v = 0; v < nv; v++) { const j = d.node[v] * 3; pa.array[v * 3] = cur[j]; pa.array[v * 3 + 1] = cur[j + 1]; pa.array[v * 3 + 2] = cur[j + 2]; }
      pa.needsUpdate = true;
      geo.boundingSphere = null;
    }
    if (tr && tr.field) rankSum(tr.field, mf, n, 1, it.val);
    else if (d.fv) {
      const sc = o.field.scale;
      if (sc) { const s = series(sc, a, b, k); for (let i = 0; i < n; i++) it.val[i] = d.fv[i] * s; }
      else if (o.field.frames) for (let i = 0; i < n; i++) it.val[i] = at2(d.fv, n, a, b, k, i);
      else it.val.set(d.fv.subarray(0, n));
    }
    if (it.f && d.fv) {
      // the value, and 1 where there is none: a vertex with no value next to ones with values takes their mean,
      // so the gray of no value starts halfway along the edge instead of the map running down to its bottom color
      const fa = geo.getAttribute('fval'), A = fa.array, f = it.f, span = f.hi - f.lo || 1;
      let gaps = false;
      for (let v = 0; v < nv; v++) {
        const x = it.val[d.node[v]], none = Number.isNaN(x);
        A[v * 2] = none ? 0 : (x - f.lo) / span;
        A[v * 2 + 1] = none ? 1 : 0;
        gaps ||= none;
      }
      if (gaps) {
        const ix = d.index, sum = it.nanSum || (it.nanSum = new Float32Array(nv)), cnt = it.nanCnt || (it.nanCnt = new Uint16Array(nv));
        sum.fill(0); cnt.fill(0);
        for (let t = 0; t < ix.length; t += 3) {
          for (let e = 0; e < 3; e++) {
            const p = ix[t + e];
            if (!A[p * 2 + 1]) continue;
            for (const q of [ix[t + (e + 1) % 3], ix[t + (e + 2) % 3]]) if (!A[q * 2 + 1]) { sum[p] += A[q * 2]; cnt[p]++; }
          }
        }
        for (let v = 0; v < nv; v++) if (cnt[v]) A[v * 2] = sum[v] / cnt[v];
      }
      fa.needsUpdate = true;
    } else if (!it.colored) {
      const t = geo.getAttribute('tint');
      for (let v = 0; v < nv; v++) { t.array[v * 3] = it.base[0]; t.array[v * 3 + 1] = it.base[1]; t.array[v * 3 + 2] = it.base[2]; }
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

  // arrows are sized for the whole scene, so a close view shrinks them about their tails as it thins the lines
  function arrowsFrame(it, a, b, k, tube = 1) {
    const { o, d } = it, n = it.n, mV = d.V.length > n * 3, mO = d.O.length > n * 3, s = (o.scale ? series(o.scale, a, b, k) : 1) * tube;
    const dir = new Vector3();
    for (let i = 0; i < n; i++) {
      const og = [0, 1, 2].map((j) => (mO ? at2(d.O, n * 3, a, b, k, i * 3 + j) : d.O[i * 3 + j]));
      const v = [0, 1, 2].map((j) => (mV ? at2(d.V, n * 3, a, b, k, i * 3 + j) : d.V[i * 3 + j]) * s);
      const len = Math.hypot(...v), g = it.arrows[i];
      g.visible = len > R * 1e-4 * tube;
      if (!g.visible) continue;
      const r = (o.radius || Math.max(len * 0.035 / tube, R * 0.006)) * tube;
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
    const { a, b, k, t, bend = true, mag = true, tube = 1 } = f;
    for (const it of items) if (it.o.kind === 'lines') it.mesh.material.uniforms.uTube.value = tube;
    // the motion's physical time when it plays (play-motion.js), else the input's frames alone
    const mo = h.motion, mf = mo && Number.isFinite(f.m) ? motionFrame(mo, f.m) : null;
    for (const fl of flows) fl.at(a, b, k);
    const next = `${a},${b},${k.toFixed(5)},${t},${bend},${mag},${tube},${mf ? f.m : ''}`;
    if (next === key) return;
    key = next;
    shadowDirty = true;
    for (const it of items) {
      const o = it.o;
      // over the motion an object with a window of its own shows only inside it (a parachute once it opens)
      const tw = mf && it.d.track && it.d.track.when;
      it.mesh.visible = !it.hidden && (tw ? shownAt(tw, f.m) : shownAt(o.when, t));
      if (it.bonds) it.bonds.visible = it.mesh.visible;
      if (!it.mesh.visible) continue;
      const uni = it.mesh.material?.uniforms;
      const tr = mf && it.d.track;
      if (uni && tr && tr.glow) uni.uGlow.value = trackSeries(tr.glow, mf) * 0.85;
      else if (uni && o.glow) uni.uGlow.value = series(o.glow, a, b, k) * 0.85;
      if (o.kind === 'mesh') {
        meshFrame(it, a, b, k, bend, mf);
        const pose = (tr && trackPose(tr, mf)) || poseAt(it.d.pp, it.d.pq, a, b, k);
        if (pose) place(it.mesh, it, pose, mag);
      } else if (o.kind === 'body') {
        const sz = tr && trackSize(tr, mf, o.size.length);
        place(it.mesh, it, (tr && trackPose(tr, mf)) || poseAt(it.d.pp, it.d.pq, a, b, k), mag, sz && sizeScale(o.shape, o.size, sz));
        if (it.f) {
          // a body colored by one value: its value at the motion's moment while it plays (a motor's torque), else
          // at the input's frame
          const c = [0, 0, 0];
          it.cv = tr && tr.value ? trackSeries(tr.value, mf) : series(o.field.values, a, b, k);
          colorOf(it.f, it.cv, c, 0);
          it.model?.tint(c);
          const ta = it.mesh.geometry.getAttribute('tint');
          if (ta.array[0] !== c[0] || ta.array[1] !== c[1] || ta.array[2] !== c[2]) {
            for (let i = 0; i < ta.count; i++) ta.array.set(c, i * 3);
            ta.needsUpdate = true;
          }
        }
      } else if (o.kind === 'lines') {
        // a path traced by the motion's time while it plays, by the input's otherwise
        const ca = it.mesh.geometry.getAttribute('cutv');
        if (it.cutM) {
          const want = mf ? it.cutM : it.cutI;
          if (ca.array !== want) { ca.array = want; ca.needsUpdate = true; }
        }
        if (mf && it.cutM) uni.uCut.value = f.m;
        else if (it.cuts) uni.uCut.value = t;
        else if (it.cutM) uni.uCut.value = 1e30;
      } else if (o.kind === 'points' || o.kind === 'atoms') pointsFrame(it, a, b, k);
      else if (o.kind === 'arrows') arrowsFrame(it, a, b, k, tube);
    }
    rideAll();
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
      v = it.cv;
    }
    if (!Number.isFinite(v)) return null;
    return { v, name: f.name, unit: f.unit || '', digits: f.digits ?? 3, limit: f.limit ?? null, limitLabel: f.limit_label || '', point: hit.point.toArray() };
  }

  // ---- a contact shadow on the floor or bench the subject stands on (contact-shadow.js) ----
  // worked out at the first frame drawn, once every prop has its pose: the floor is the ground, or the top of
  // the highest prop just under the subject's lowest point and under its footprint. A view that follows a small
  // subject over a long path gets none, since its floor is far away.
  let shadow = null, shadowDirty = true, shadowTried = false;
  function findShadow() {
    shadowTried = true;
    if (run.follow) return;
    const sb = run.subject || run.bounds, bottom = sb.lo[2], tol = 0.02 * R;
    const fx0 = sb.lo[0], fx1 = sb.hi[0], fy0 = sb.lo[1], fy1 = sb.hi[1];
    scene.updateMatrixWorld(true);
    let floor = -Infinity;
    const box = new Box3();
    for (const it of items) {
      if (it.o.kind === 'ground') { if (it.o.z <= bottom + tol) floor = Math.max(floor, it.o.z); continue; }
      if (!it.o.prop || !(it.o.kind === 'mesh' || it.o.kind === 'body') || !it.mesh.visible) continue;
      box.setFromObject(it.mesh);
      const top = box.max.z;
      const under = box.min.x < fx1 && box.max.x > fx0 && box.min.y < fy1 && box.max.y > fy0;
      if (under && top <= bottom + tol && top >= bottom - 0.25 * (sb.hi[2] - sb.lo[2] + R * 0.1)) floor = Math.max(floor, top);
    }
    if (!Number.isFinite(floor)) return;
    const casters = shadowCasters;
    for (const it of items) {
      if (it.o.prop || !['mesh', 'body', 'points', 'atoms'].includes(it.o.kind)) continue;
      casters.add(it.mesh);
      if (it.bonds) casters.add(it.bonds);
      for (const m of it.model?.meshes || []) casters.add(m);
    }
    if (!casters.size) return;
    const size = Math.max(fx1 - fx0, fy1 - fy0) * 1.5 + R * 0.12;
    const reach = Math.max(sb.hi[2] - floor, R * 0.2) * 0.7;
    shadow = contactShadow({ casters, center: [(fx0 + fx1) / 2, (fy0 + fy1) / 2], size, floor: floor + R * 4e-4, reach });
    scene.add(shadow.plane);
  }
  /** Redraw the contact shadow when the scene has moved since it was last drawn. */
  function shade(renderer) {
    if (!shadowTried) findShadow();
    if (!shadow || !shadowDirty) return;
    shadowDirty = false;
    shadow.render(renderer, scene);
  }

  function dispose() {
    shadow?.dispose();
    scene.traverse((x) => { x.geometry?.dispose?.(); x.material?.dispose?.(); });
  }
  // ---- the Blender models (play-models.js) in place of the plain shapes ----
  const shadowCasters = new Set();
  /** Hide an object's own shape and its outlines, keeping it for the probe and the bounds. */
  function hideShape(it) {
    it.mesh.material.visible = false;
    // its outlines and its depth pass go with it
    for (const c of it.mesh.children) c.visible = false;
  }
  /** Put the models loadModels found in place: each under its object, so it moves as the object does. */
  function dress(loaded) {
    if (!loaded) return;
    const { spec, models } = loaded;
    const pick = (name, index) => items.filter((it) => (index !== undefined ? it.oi === index : it.o.name === name)
      && (it.o.kind === 'mesh' || it.o.kind === 'body'));
    for (const e of spec.replace || []) {
      const g = models.get(e.file);
      if (!g) continue;
      for (const it of pick(e.object, e.index)) {
        if (it.model) continue;
        const m = instance(g, e);
        hideShape(it);
        it.mesh.add(m.group);
        it.model = m;
        m.tint(it.base);
        if ((it.f && !it.o.prop) || it.o.glow) m.keepTint();
        if (shadowCasters.has(it.mesh)) for (const x of m.meshes) shadowCasters.add(x);
      }
    }
    for (const name of spec.hide || []) {
      for (const it of items) {
        if (it.o.name !== name) continue;
        // a ground the models replace (a concrete pad) goes with its grid; anything else keeps its shape for the probe
        if (it.o.kind === 'ground') { it.mesh.visible = false; it.hidden = true; }
        else if (it.o.kind === 'mesh' || it.o.kind === 'body') hideShape(it);
      }
    }
    for (const e of spec.add || []) {
      const g = models.get(e.file);
      if (!g) continue;
      const m = instance(g, e), at = e.parent ? pick(e.parent)[0] : null;
      (at ? at.mesh : scene).add(m.group);
      if (!at) { adds.push(m); ride(m); }
    }
    // a model's faint copy over the subject is drawn only when the scene has a subject colored by a field left
    // in view (a wiper's links are all models: nothing to show through)
    const anySubject = items.some((it) => !it.model && !it.o.prop && it.f && (it.o.kind === 'mesh' || it.o.kind === 'body'));
    for (const it of items) it.model?.ghosts(anySubject);
    for (const m of adds) m.ghosts(anySubject);
    // everything is brought to the frame again, so a tinted model takes its field's color
    key = '';
    shadowDirty = true;
  }

  // a model fixed on a part that bends (a motor bolted to the tip of an arm that swings) rides the part's nearest
  // point: the part's vertex inside the model's box nearest its middle, when the model touches the part at all
  const riders = [];
  const rbox = new Box3(), rc = new Vector3();
  function ride(m) {
    const bent = items.filter((it) => it.o.kind === 'mesh' && !it.o.prop && !it.d.pp && !it.d.track?.pos && (it.d.du || it.d.track?.deform));
    if (!bent.length) return;
    scene.updateMatrixWorld(true);
    rbox.setFromObject(m.group);
    if (rbox.isEmpty()) return;
    rbox.getCenter(rc);
    let best = null;
    for (const it of bent) {
      const P = it.d.P;
      for (let i = 0; i < P.length / 3; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
        const out = Math.hypot(Math.max(rbox.min.x - x, 0, x - rbox.max.x), Math.max(rbox.min.y - y, 0, y - rbox.max.y), Math.max(rbox.min.z - z, 0, z - rbox.max.z));
        const d = out * 1e3 + Math.hypot(x - rc.x, y - rc.y, z - rc.z);
        if (out <= R * 0.01 && (!best || d < best.d)) best = { it, i, d };
      }
    }
    const g = m.group, fixed = !g.matrixAutoUpdate;
    if (best) riders.push({ m, it: best.it, i: best.i, fixed, at: fixed ? new Vector3().setFromMatrixPosition(g.matrix) : g.position.clone() });
  }
  /** Move every riding model by its point's displacement as the part is drawn now. */
  function rideAll() {
    for (const r of riders) {
      const { it, i } = r, P = it.d.P, c = it.cur;
      if (!c) continue;
      const x = r.at.x + c[i * 3] - P[i * 3], y = r.at.y + c[i * 3 + 1] - P[i * 3 + 1], z = r.at.z + c[i * 3 + 2] - P[i * 3 + 2], g = r.m.group;
      if (r.fixed) { g.matrix.elements[12] = x; g.matrix.elements[13] = y; g.matrix.elements[14] = z; g.matrixWorldNeedsUpdate = true; }
      else g.position.set(x, y, z);
    }
  }

  // the models added round the subject, veiled when one stands between the camera and the subject
  const adds = [], sbox = run.subject || run.bounds;
  const middle = new Vector3(...[0, 1, 2].map((k) => (sbox.lo[k] + sbox.hi[k]) / 2));
  function veil(from) {
    for (const m of adds) m.veil(from, middle);
  }

  /** Move every tracer on by dt seconds of the page's clock; true while any are moving. */
  function advance(dt) {
    for (const fl of flows) fl.advance(dt);
    return flows.length > 0;
  }

  return { scene, set, probe, shade, dispose, dress, veil, advance, flows };
}
