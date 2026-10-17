// The stage's view refined while it rests, the way Sketchfab and the three.js TAA example refine a still model.
// Once the same frame has been asked for twice, the stage draws it into a target of its own and works out ambient
// occlusion from the depth it left (scalable ambient obscurance, one texel per pixel, smoothed by depth), which the
// surfaces then take in over a few display frames; then it draws the frame again and again with the camera moved
// by a fraction of a pixel (a Halton sequence) and shows the mean, sixteen frames for clean edges, outlines and
// hatching. Anything that moves starts it again, and while the view moves nothing here runs, so orbiting and
// dragging the slider cost what they did.
//
// The occlusion is not laid over the finished frame: each surface reads it while it is lit (play-look.js SSAO) and
// dims only the light that comes from all round, the studio's diffuse light and reflections, as three.js's own
// aoMap does. The key light and its shadow stay whole, so a resting view keeps the modelling and the cast shadows of
// the moving one, and a face lit by the key light is not greyed. The part the run is about, anything else colored
// by a field and anything that glows are built without it (play-look.js fieldStudioShader, play-models.js
// keepColor), so a stress or temperature color shows exactly as the legend gives it. A still drawn at build time
// (run-poster.js) is the plain frame with the same occlusion, which the mean of the jittered frames differs from
// only along its edges, so the view the page holds settles on its still.
import {
  DepthStencilFormat, DepthTexture, HalfFloatType, Mesh, NearestFilter, NoBlending, OrthographicCamera, PlaneGeometry, RGBAFormat, RGFormat,
  SRGBColorSpace, Scene, ShaderMaterial, UnsignedInt248Type, Vector2, WebGLRenderTarget,
} from 'three';
import { SSAO } from './play-look.js';

/**
 * Lines and arrows are drawn on a layer of their own, left out of the depth the occlusion is worked out from, so a
 * thin flux line or a load arrow never casts a smudge on the face behind it.
 */
export const OVERLAY = 1;
export function overlay(object) {
  object.traverse((o) => o.layers.set(OVERLAY));
  return object;
}

// display frames the occlusion fades in over, then jittered frames in the mean
const RAMP = 5, SAMPLES = 16;
const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
// in pixels of the drawing, a Halton sequence over the middle 0.6 of a pixel: the mean of a box that wide smooths
// a stepped edge and leaves a texture's finest detail nearly as sharp as the moving frame shows it, where the whole
// pixel would soften it. The first is the plain frame, so the mean starts from
// what the view showed.
const SPREAD = 0.6;
const JITTER = Array.from({ length: SAMPLES }, (_, i) => (i ? [(halton(i, 2) - 0.5) * SPREAD, (halton(i, 3) - 0.5) * SPREAD] : [0, 0]));

const QUAD_VERT = 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }';

// Scalable ambient obscurance (McGuire, Mara and Luebke 2012) at every pixel: the view space position of each pixel
// from the depth, a normal from the nearer neighbour on each side, and taps on a spiral turned one of sixteen ways
// in a four by four tile, which the smoothing after it averages out. Writes the occlusion and the view depth, or a
// depth of +1 where nothing was drawn.
const AO_FRAG = /* glsl */ `
uniform sampler2D tDepth;
uniform mat4 uInvProj;
uniform vec4 uRect;
uniform vec2 uTexel;
uniform float uRadius;
uniform float uPxPerUnit;
uniform float uIntensity;
#define TAPS 16
vec3 viewAt(vec2 px) {
  float d = texture2D(tDepth, px * uTexel).x;
  vec4 p = uInvProj * vec4((px - uRect.xy) / uRect.zw * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  return p.xyz / p.w;
}
void main() {
  vec2 px = gl_FragCoord.xy;
  float d = texture2D(tDepth, px * uTexel).x;
  if (d >= 0.999999) { gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0); return; }
  vec3 P = viewAt(px);
  vec3 r = viewAt(px + vec2(1.0, 0.0)) - P, l = P - viewAt(px - vec2(1.0, 0.0));
  vec3 u = viewAt(px + vec2(0.0, 1.0)) - P, b = P - viewAt(px - vec2(0.0, 1.0));
  vec3 N = normalize(cross(abs(r.z) < abs(l.z) ? r : l, abs(u.z) < abs(b.z) ? u : b));
  // the radius on screen at this depth, kept under a quarter of the view
  float rpx = min(uRadius * uPxPerUnit / max(-P.z, 1e-6), uRect.w * 0.25);
  if (rpx < 2.0) { gl_FragColor = vec4(1.0, P.z, 0.0, 1.0); return; }
  vec2 t4 = mod(floor(px), 4.0);
  float spin = 6.2831853 * (mod(t4.x * 5.0 + t4.y * 11.0, 16.0) + 0.5) / 16.0;
  float r2 = uRadius * uRadius, sum = 0.0;
  for (int i = 0; i < TAPS; i++) {
    float t = (float(i) + 0.5) / float(TAPS);
    float a = spin + t * 43.98;
    vec2 q = px + vec2(cos(a), sin(a)) * (t * rpx);
    if (q.x < uRect.x || q.y < uRect.y || q.x > uRect.x + uRect.z || q.y > uRect.y + uRect.w) continue;
    vec3 v = viewAt(floor(q) + 0.5) - P;
    float vv = dot(v, v), vn = dot(v, N);
    float f = max(r2 - vv, 0.0) / r2;
    sum += f * f * f * max((vn - 0.01 * uRadius) / (vv + 0.02 * r2), 0.0);
  }
  float ao = clamp(1.0 - uIntensity * uRadius * sum * (2.0 / float(TAPS)), 0.0, 1.0);
  gl_FragColor = vec4(ao, P.z, 0.0, 1.0);
}`;

// the four by four tile averaged, each neighbour weighted down as its depth parts from this pixel's, so the
// occlusion never crosses an edge; the depth is kept for the surfaces to check it against their own
const BLUR_FRAG = /* glsl */ `
uniform sampler2D tAO;
uniform vec2 uTexel;
void main() {
  vec2 uv = gl_FragCoord.xy * uTexel;
  vec2 c = texture2D(tAO, uv).xy;
  if (c.y > 0.0) { gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0); return; }
  float sum = 0.0, w = 0.0, tol = 0.03 * abs(c.y) + 1e-6;
  for (int i = -2; i < 2; i++) {
    for (int j = -2; j < 2; j++) {
      vec2 s = texture2D(tAO, uv + vec2(float(i), float(j)) * uTexel).xy;
      float k = s.y > 0.0 ? 0.0 : max(0.0, 1.0 - abs(s.y - c.y) / tol);
      sum += s.x * k; w += k;
    }
  }
  gl_FragColor = vec4(w > 0.0 ? sum / w : c.x, c.y, 0.0, 1.0);
}`;

const MEAN_FRAG = /* glsl */ `
uniform sampler2D tNew;
uniform sampler2D tOld;
uniform vec2 uTexel;
uniform float uK;
void main() {
  vec2 uv = gl_FragCoord.xy * uTexel;
  gl_FragColor = mix(texture2D(tOld, uv), texture2D(tNew, uv), uK);
}`;

const COPY_FRAG = /* glsl */ `
uniform sampler2D tMap;
uniform vec2 uTexel;
void main() { gl_FragColor = texture2D(tMap, gl_FragCoord.xy * uTexel); }`;

function pass(frag, uniforms) {
  return new ShaderMaterial({ vertexShader: QUAD_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false, blending: NoBlending });
}

/** A rectangle in CSS pixels as device pixels, rounded the way three.js rounds the canvas's viewport. */
function devicePx(x, y, w, h, p) {
  return [Math.round(x * p), Math.round(y * p), Math.round(w * p), Math.round(h * p)];
}

/**
 * The refinement for a renderer, or null where it cannot draw into float targets (it then never runs). steps is
 * how many display frames a rest takes. For step k of a rest: begin(k) points the drawing at the frame's target
 * and gives the surfaces the occlusion as far as it has faded in; rect() sets a rectangle in the target as
 * renderer.setViewport and setScissor would; jitter(camera, w, h, k) moves the camera by the step's fraction of a
 * pixel and jitter(..., -1) puts it back; at step 0, occlude() works the occlusion out for one pane from what has
 * been drawn there; end() shows the frame, or the mean of the jittered ones. lay(s) has the surfaces drawn next
 * take s of the occlusion (a still's second drawing takes it whole, an inset none), and moving() takes it away
 * from them for a frame drawn straight to the canvas.
 */
export function createRefiner(renderer) {
  const gl = renderer.getContext();
  const full = !!gl.getExtension('EXT_color_buffer_float');
  if (!renderer.capabilities.isWebGL2 || !(full || gl.getExtension('EXT_color_buffer_half_float'))) return null;
  const size = new Vector2();
  let w = 0, h = 0, frame = null, rawAO = null, ao = null, means = null, flip = 0;
  const texel = { value: new Vector2() };
  const quad = new Mesh(new PlaneGeometry(2, 2));
  quad.frustumCulled = false;
  const scene = new Scene().add(quad), cam = new OrthographicCamera();
  const aoU = {
    tDepth: { value: null }, uInvProj: { value: null }, uRect: { value: [0, 0, 1, 1] }, uTexel: texel, uRadius: { value: 1 },
    uPxPerUnit: { value: 1 }, uIntensity: { value: 1.5 },
  };
  const aoM = pass(AO_FRAG, aoU);
  const blurU = { tAO: { value: null }, uTexel: texel };
  const blurM = pass(BLUR_FRAG, blurU);
  const meanU = { tNew: { value: null }, tOld: { value: null }, uTexel: texel, uK: { value: 1 } };
  const meanM = pass(MEAN_FRAG, meanU);
  const copyU = { tMap: { value: null }, uTexel: texel };
  const copyM = pass(COPY_FRAG, copyU);

  const target = (opts = {}) => {
    const t = new WebGLRenderTarget(w, h, { type: HalfFloatType, depthBuffer: false, magFilter: NearestFilter, minFilter: NearestFilter, ...opts });
    t.texture.generateMipmaps = false;
    return t;
  };
  function fit() {
    renderer.getDrawingBufferSize(size);
    if (size.x === w && size.y === h && frame) return;
    disposeTargets();
    w = size.x; h = size.y;
    texel.value.set(1 / w, 1 / h);
    // the frame, drawn as the canvas would be: sRGB values, its depth and stencil kept for the occlusion
    const depth = new DepthTexture(w, h, UnsignedInt248Type);
    depth.format = DepthStencilFormat;
    frame = target({ depthBuffer: true, stencilBuffer: true, depthTexture: depth });
    frame.texture.colorSpace = SRGBColorSpace;
    // three.js writes sRGB, as to the canvas, only into a target it takes for a headset's: the frame stands in for the canvas
    frame.isXRRenderTarget = true;
    frame.scissorTest = true;
    // the occlusion and its depth, in two channels where the graphics chip can draw them
    const two = { format: full ? RGFormat : RGBAFormat };
    rawAO = target(two);
    rawAO.scissorTest = true;
    ao = target(two);
    ao.scissorTest = true;
    means = [target(), target()];
  }
  function draw(material, to) {
    quad.material = material;
    renderer.setRenderTarget(to);
    renderer.render(scene, cam);
  }
  function rect(x, y, vw, vh) {
    frame.viewport.set(...devicePx(x, y, vw, vh, renderer.getPixelRatio()));
    frame.scissor.copy(frame.viewport);
    renderer.setViewport(x, y, vw, vh);
    renderer.setScissor(x, y, vw, vh);
  }
  function show(t, cw, ch) {
    copyU.tMap.value = t.texture;
    renderer.setRenderTarget(null);
    renderer.setViewport(0, 0, cw, ch);
    renderer.setScissor(0, 0, cw, ch);
    draw(copyM, null);
  }
  function lay(strength) {
    SSAO.tSSAO.value = strength > 0 ? ao.texture : null;
    SSAO.uSSAO.value.set(1 / w, 1 / h, strength);
  }
  return {
    steps: RAMP + SAMPLES,
    ramp: RAMP,
    /** Draw step k into the frame's target, the surfaces taking k / RAMP of the occlusion, all of it from RAMP on. */
    begin(k) {
      fit();
      lay(Math.min(1, k / RAMP));
      renderer.setRenderTarget(frame);
    },
    lay,
    moving() { SSAO.tSSAO.value = null; SSAO.uSSAO.value.set(0, 0, 0); },
    rect,
    jitter(camera, vw, vh, k, sign = 1) {
      if (k < RAMP) return;
      // a fraction of a pixel of the drawing, at the pixel ratio it is drawn at
      const p = renderer.getPixelRatio(), [jx, jy] = JITTER[k - RAMP], e = camera.projectionMatrix.elements;
      e[8] += sign * (2 * jx) / (vw * p);
      e[9] += sign * (2 * jy) / (vh * p);
    },
    /** Work the occlusion out for one pane (x, y, vw, vh in CSS pixels from the bottom left) seen by camera, radius in its units. */
    occlude(camera, x, y, vw, vh, radius) {
      const r = devicePx(x, y, vw, vh, renderer.getPixelRatio());
      aoU.tDepth.value = frame.depthTexture;
      aoU.uInvProj.value = camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
      aoU.uRect.value = r;
      aoU.uRadius.value = radius;
      aoU.uPxPerUnit.value = camera.projectionMatrix.elements[5] * 0.5 * r[3];
      for (const t of [rawAO, ao]) { t.viewport.set(...r); t.scissor.set(...r); }
      draw(aoM, rawAO);
      blurU.tAO.value = rawAO.texture;
      draw(blurM, ao);
      renderer.setRenderTarget(frame);
    },
    /** Show step k on the canvas, cw by ch CSS pixels: the frame while the occlusion fades in, then the mean. */
    end(k, cw, ch, still) {
      if (still || k < RAMP) { show(frame, cw, ch); return; }
      const old = means[flip], out = means[1 - flip];
      flip = 1 - flip;
      meanU.tNew.value = frame.texture;
      meanU.tOld.value = old.texture;
      meanU.uK.value = 1 / (k - RAMP + 1);
      draw(meanM, out);
      show(out, cw, ch);
    },
    dispose() {
      disposeTargets();
      for (const m of [aoM, blurM, meanM, copyM]) m.dispose();
      quad.geometry.dispose();
    },
  };
  function disposeTargets() {
    frame?.depthTexture?.dispose();
    for (const t of [frame, rawAO, ao, ...(means || [])]) t?.dispose();
    frame = rawAO = ao = means = null;
  }
}
