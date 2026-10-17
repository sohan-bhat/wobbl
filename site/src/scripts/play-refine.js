// The stage's view refined while it rests, the way Sketchfab and the three.js TAA example refine a still model.
// Once the same frame has been asked for twice, the stage draws it into a target of its own and works out ambient
// occlusion from the depth it left (scalable ambient obscurance, at half resolution and smoothed), which fades in
// over a few display frames; then it draws the frame again and again with the camera moved by a fraction of a pixel
// (a Halton sequence) and shows the mean, sixteen frames for clean edges, outlines and hatching. Anything that
// moves starts it again, and while the view moves nothing here runs, so orbiting and dragging the slider cost what
// they did. The occlusion never touches the part the run is about: its pixels are marked in the stencil buffer's
// first bit (play-models.js subject), anything else colored by a field or glowing in its second (keepColor), and
// the occlusion is laid only where neither is, so a stress or temperature color shows exactly as the legend gives
// it. A still drawn at build time (run-poster.js) is the plain frame with the same occlusion, which the mean of the
// jittered frames differs from only along its edges, so the view the page holds settles on its still.
import {
  CustomBlending, DepthStencilFormat, DepthTexture, EqualStencilFunc, HalfFloatType, KeepStencilOp, LinearFilter, Mesh, NearestFilter,
  NoBlending, OneFactor, OrthographicCamera, PlaneGeometry, SRGBColorSpace, Scene, ShaderMaterial, SrcColorFactor, UnsignedInt248Type,
  Vector2, WebGLRenderTarget, ZeroFactor,
} from 'three';

/**
 * Lines and arrows are drawn on a layer of their own, after the occlusion is worked out, so a thin flux line or a
 * load arrow never casts a smudge on the face behind it.
 */
export const OVERLAY = 1;
export function overlay(object) {
  object.traverse((o) => o.layers.set(OVERLAY));
  return object;
}

// display frames the occlusion fades in over, then jittered frames in the mean
const RAMP = 5, SAMPLES = 16;
const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
// the first is the plain frame, so the mean starts from what the view showed
const JITTER = Array.from({ length: SAMPLES }, (_, i) => (i ? [halton(i, 2) - 0.5, halton(i, 3) - 0.5] : [0, 0]));

const QUAD_VERT = 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }';

// Scalable ambient obscurance (McGuire, Mara and Luebke 2012) at half resolution: the view space position of each
// pixel from the depth, a normal from the nearer neighbour on each side, and taps on a spiral turned one of sixteen
// ways in a four by four tile, which the smoothing after it averages out. Writes the occlusion and the depth.
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
  vec2 hp = floor(gl_FragCoord.xy), px = hp * 2.0 + 0.5;
  float d = texture2D(tDepth, px * uTexel).x;
  if (d >= 0.999999) { gl_FragColor = vec4(1.0, -1e9, 0.0, 1.0); return; }
  vec3 P = viewAt(px);
  vec3 r = viewAt(px + vec2(1.0, 0.0)) - P, l = P - viewAt(px - vec2(1.0, 0.0));
  vec3 u = viewAt(px + vec2(0.0, 1.0)) - P, b = P - viewAt(px - vec2(0.0, 1.0));
  vec3 N = normalize(cross(abs(r.z) < abs(l.z) ? r : l, abs(u.z) < abs(b.z) ? u : b));
  // the radius on screen at this depth, kept under a quarter of the view
  float rpx = min(uRadius * uPxPerUnit / max(-P.z, 1e-6), uRect.w * 0.25);
  if (rpx < 2.0) { gl_FragColor = vec4(1.0, P.z, 0.0, 1.0); return; }
  vec2 t4 = mod(hp, 4.0);
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

// the four by four tile averaged, each neighbour weighted down as its depth parts from this pixel's
const BLUR_FRAG = /* glsl */ `
uniform sampler2D tAO;
uniform vec2 uTexel;
void main() {
  vec2 uv = gl_FragCoord.xy * uTexel;
  vec2 c = texture2D(tAO, uv).xy;
  if (c.y < -1e8) { gl_FragColor = vec4(1.0); return; }
  float sum = 0.0, w = 0.0, tol = 0.03 * abs(c.y) + 1e-6;
  for (int i = -2; i < 2; i++) {
    for (int j = -2; j < 2; j++) {
      vec2 s = texture2D(tAO, uv + vec2(float(i), float(j)) * uTexel).xy;
      float k = s.y < -1e8 ? 0.0 : max(0.0, 1.0 - abs(s.y - c.y) / tol);
      sum += s.x * k; w += k;
    }
  }
  gl_FragColor = vec4(vec3(w > 0.0 ? sum / w : c.x), 1.0);
}`;

// laid over the frame by multiplying, only where the stencil says nothing colored by a field was drawn
const APPLY_FRAG = /* glsl */ `
uniform sampler2D tAO;
uniform vec2 uTexel;
uniform float uStrength;
void main() {
  float ao = texture2D(tAO, gl_FragCoord.xy * uTexel).r;
  // the frame holds sRGB values: the occlusion is a factor on linear light, so it is taken into sRGB first
  gl_FragColor = vec4(vec3(pow(mix(1.0, ao, uStrength), 1.0 / 2.2)), 1.0);
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

function pass(frag, uniforms, extra = {}) {
  return new ShaderMaterial({ vertexShader: QUAD_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false, blending: NoBlending, ...extra });
}

/**
 * The refinement for a renderer, or null where it cannot draw into float targets (it then never runs). steps is
 * how many display frames a rest takes. For step k of a rest: begin() points the drawing at the frame's target and
 * rect() sets a rectangle in it as renderer.setViewport and setScissor would; jitter(camera, w, h, k) moves the
 * camera by the step's fraction of a pixel and jitter(..., -1) puts it back; occlude() lays the occlusion over one
 * pane; end() shows the frame, or the mean of the jittered ones. still draws a still: the plain frame, with its
 * occlusion whole.
 */
export function createRefiner(renderer) {
  const gl = renderer.getContext();
  if (!renderer.capabilities.isWebGL2 || !(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'))) return null;
  const size = new Vector2();
  let w = 0, h = 0, frame = null, half = null, soft = null, means = null, flip = 0;
  const texel = { value: new Vector2() }, halfTexel = { value: new Vector2() };
  const quad = new Mesh(new PlaneGeometry(2, 2));
  quad.frustumCulled = false;
  const scene = new Scene().add(quad), cam = new OrthographicCamera();
  const aoU = {
    tDepth: { value: null }, uInvProj: { value: null }, uRect: { value: [0, 0, 1, 1] }, uTexel: texel, uRadius: { value: 1 },
    uPxPerUnit: { value: 1 }, uIntensity: { value: 2 },
  };
  const aoM = pass(AO_FRAG, aoU);
  const blurU = { tAO: { value: null }, uTexel: halfTexel };
  const blurM = pass(BLUR_FRAG, blurU);
  const applyU = { tAO: { value: null }, uTexel: texel, uStrength: { value: 1 } };
  const applyM = pass(APPLY_FRAG, applyU, {
    blending: CustomBlending, blendSrc: ZeroFactor, blendDst: SrcColorFactor, blendSrcAlpha: ZeroFactor, blendDstAlpha: OneFactor,
    // only where neither the part (the first bit) nor anything else colored by a field (the second) was drawn
    stencilWrite: true, stencilRef: 0, stencilFunc: EqualStencilFunc, stencilFuncMask: 3, stencilWriteMask: 0,
    stencilFail: KeepStencilOp, stencilZFail: KeepStencilOp, stencilZPass: KeepStencilOp,
  });
  const meanU = { tNew: { value: null }, tOld: { value: null }, uTexel: texel, uK: { value: 1 } };
  const meanM = pass(MEAN_FRAG, meanU);
  const copyU = { tMap: { value: null }, uTexel: texel };
  const copyM = pass(COPY_FRAG, copyU);

  const target = (tw, th, opts = {}) => {
    const t = new WebGLRenderTarget(tw, th, { type: HalfFloatType, depthBuffer: false, magFilter: NearestFilter, minFilter: NearestFilter, ...opts });
    t.texture.generateMipmaps = false;
    return t;
  };
  function fit() {
    renderer.getDrawingBufferSize(size);
    if (size.x === w && size.y === h && frame) return;
    disposeTargets();
    w = size.x; h = size.y;
    const hw = Math.ceil(w / 2), hh = Math.ceil(h / 2);
    texel.value.set(1 / w, 1 / h);
    halfTexel.value.set(1 / hw, 1 / hh);
    // the frame, drawn as the canvas would be: sRGB values, its depth and stencil kept for the occlusion
    const depth = new DepthTexture(w, h, UnsignedInt248Type);
    depth.format = DepthStencilFormat;
    frame = target(w, h, { depthBuffer: true, stencilBuffer: true, depthTexture: depth });
    frame.texture.colorSpace = SRGBColorSpace;
    // three.js writes sRGB, as to the canvas, only into a target it takes for a headset's: the frame stands in for the canvas
    frame.isXRRenderTarget = true;
    frame.scissorTest = true;
    half = target(hw, hh);
    half.scissorTest = true;
    soft = target(hw, hh, { magFilter: LinearFilter, minFilter: LinearFilter });
    soft.scissorTest = true;
    means = [target(w, h), target(w, h)];
  }
  function draw(material, to) {
    quad.material = material;
    renderer.setRenderTarget(to);
    renderer.render(scene, cam);
  }
  function rect(x, y, vw, vh) {
    const p = renderer.getPixelRatio();
    frame.viewport.set(x * p, y * p, vw * p, vh * p).floor();
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
  return {
    steps: RAMP + SAMPLES,
    ramp: RAMP,
    begin() {
      fit();
      renderer.setRenderTarget(frame);
    },
    rect,
    jitter(camera, vw, vh, k, sign = 1) {
      if (k < RAMP) return;
      const p = renderer.getPixelRatio(), [jx, jy] = JITTER[k - RAMP], e = camera.projectionMatrix.elements;
      e[8] += sign * (2 * jx) / (vw * p);
      e[9] += sign * (2 * jy) / (vh * p);
    },
    /** Lay the occlusion over one pane (x, y, vw, vh in CSS pixels from the bottom left) seen by camera, radius in its units. */
    occlude(camera, x, y, vw, vh, k, radius, still) {
      const p = renderer.getPixelRatio();
      const r = [Math.floor(x * p), Math.floor(y * p), Math.floor(vw * p), Math.floor(vh * p)];
      const hr = r.map((v, i) => (i < 2 ? Math.floor(v / 2) : Math.ceil(v / 2)));
      // worked out on the plain frame that starts a rest, and kept for the jittered ones
      if (k === 0 || still) {
        aoU.tDepth.value = frame.depthTexture;
        aoU.uInvProj.value = camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        aoU.uRect.value = r;
        aoU.uRadius.value = radius;
        aoU.uPxPerUnit.value = camera.projectionMatrix.elements[5] * 0.5 * r[3];
        for (const t of [half, soft]) { t.viewport.set(...hr); t.scissor.set(...hr); }
        draw(aoM, half);
        blurU.tAO.value = half.texture;
        draw(blurM, soft);
      }
      applyU.tAO.value = soft.texture;
      applyU.uStrength.value = still ? 1 : Math.min(1, (k + 1) / RAMP);
      renderer.setRenderTarget(frame);
      rect(x, y, vw, vh);
      quad.material = applyM;
      renderer.render(scene, cam);
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
      for (const m of [aoM, blurM, applyM, meanM, copyM]) m.dispose();
      quad.geometry.dispose();
    },
  };
  function disposeTargets() {
    frame?.depthTexture?.dispose();
    for (const t of [frame, half, soft, ...(means || [])]) t?.dispose();
    frame = half = soft = means = null;
  }
}
