// Drawing the hero part with three.js, which is bundled from npm. The data holds three unit loads on
// the motor shaft, two at right angles in the pulley plane and one along the shaft, solved on one mesh
// (scripts/solve_site_directions.py, scripts/export_site_part.py). The model is linear elastic, so a
// load of F newtons along the unit direction u is exactly F (u1 case1 + u2 case2 + u3 case3). The
// vertex shader combines the three stress tensors and takes von Mises from the result at every
// vertex, since von Mises does not add up across loads; moving the load only changes uniforms. This runs in draw-worker.js where the browser can draw
// in a worker, so WebGL never holds up the page, and on the page itself where it cannot.
import {
  BufferAttribute, BufferGeometry, ConeGeometry, CylinderGeometry, DoubleSide, Group, Line, LineBasicMaterial,
  Mesh, MeshBasicMaterial, PerspectiveCamera, PlaneGeometry, Scene, ShaderMaterial, SphereGeometry, Vector2,
  Vector3, Vector4, WebGLRenderTarget, WebGLRenderer,
} from 'three';

const VERT = /* glsl */ `
attribute vec4 stressA;
attribute vec4 stressB;
attribute vec4 stressC;
attribute vec4 shearAB;
attribute vec4 shearC;
attribute vec3 dispA;
attribute vec3 dispB;
attribute vec3 dispC;
uniform float uF;
uniform vec3 uU;
uniform float uExag;
uniform float uLimit;
// each channel's largest value: the attributes arrive as -1 to 1
uniform vec4 uRA;
uniform vec4 uRB;
uniform vec4 uRC;
uniform vec4 uRS;
uniform vec2 uRSC;
uniform vec3 uRDA;
uniform vec3 uRDB;
uniform vec3 uRDC;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
#ifdef PICK
varying float vS;
#endif
// Turbo, as a polynomial (Google, Apache 2.0), so the color needs no texture lookup
vec3 turbo(float x) {
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
}
void main() {
  // the stress tensor for this load, then von Mises from it
  vec4 s = uU.x * (stressA * uRA) + uU.y * (stressB * uRB) + uU.z * (stressC * uRC);
  vec4 h = shearAB * uRS;
  vec2 t = uU.x * h.xy + uU.y * h.zw + uU.z * (shearC.xy * uRSC);
  vec3 d = s.xyz - s.yzx;
  float vm = uF * sqrt(0.5 * dot(d, d) + 3.0 * (s.w * s.w + dot(t, t)));
  float x = vm / uLimit;
  vC = x > 1.0 ? vec3(0.949, 0.949, 0.933) : turbo(x);
  vE = shearC.z;
#ifdef PICK
  vS = vm;
#endif
  vec3 p = position + (uExag * uF) * (uU.x * dispA * uRDA + uU.y * dispB * uRDB + uU.z * dispC * uRDC);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
uniform float uHatch;
uniform float uPx;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
void main() {
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 l1 = normalize(vec3(0.35, 0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.7, -0.1, 0.45));
  float d = 0.34 + 0.6 * max(dot(n, l1), 0.0) + 0.2 * max(dot(n, l2), 0.0);
  float s = 0.16 * pow(max(dot(n, normalize(l1 + v)), 0.0), 28.0);
  vec3 col = vC * d + vec3(s);
  if (uHatch > 0.5 && vE > 0.5) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// The probe reads von Mises under the pointer from a one pixel render: 24 bits of MPa / 4096 in
// red, green and blue, and alpha 1 in an excluded zone, a half elsewhere, 0 off the part.
const PICK_FRAG = /* glsl */ `
varying float vS;
varying float vE;
void main() {
  float a = clamp(vS / 4096.0, 0.0, 0.99999) * 255.0;
  float r = floor(a); a = (a - r) * 255.0;
  float g = floor(a); a = (a - g) * 255.0;
  gl_FragColor = vec4(r / 255.0, g / 255.0, floor(a) / 255.0, vE > 0.5 ? 1.0 : 0.5);
}`;

/** The belt pull: drawn solid, plus a faint copy that shows through the part when it is behind it. */
function arrow(color) {
  const g = new Group();
  const solid = new MeshBasicMaterial({ color });
  const ghost = new MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
  const parts = {};
  for (const [k, geo] of Object.entries({
    shaft: new CylinderGeometry(1.1, 1.1, 1, 12),
    head: new ConeGeometry(3.6, 10, 20),
    hub: new SphereGeometry(3.2, 16, 10),
  })) {
    const m = new Mesh(geo, solid);
    const x = new Mesh(geo, ghost);
    x.renderOrder = 2;
    m.add(x);
    parts[k] = m;
    g.add(m);
  }
  g.userData = { ...parts, materials: [solid, ghost] };
  return g;
}

const SIZES = { position: 3, normal: 3, stressA: 4, stressB: 4, stressC: 4, shearAB: 4, shearC: 4, dispA: 3, dispB: 3, dispC: 3 };
const COLD = 0xffd400, HOT = 0xfff07a;

/**
 * Everything that touches WebGL, for a canvas that is either an OffscreenCanvas in a worker
 * (draw-worker.js) or the page's own canvas where workers cannot draw. The page keeps the camera,
 * the input and the labels, and sends a frame whenever something changes.
 *
 *   canvas   where to draw; w, h and px its size in CSS pixels and device pixels per CSS pixel
 *   man      the manifest; mesh a promise of buildMesh()'s result
 *   pause    resolves when it is fine to take the next step (an idle moment on the main thread,
 *            straight away in a worker, where a long step holds up nothing the visitor sees)
 */
export async function createDrawer({ canvas, man, mesh: meshP, w, h, px, pause }) {
  const renderer = new WebGLRenderer({ canvas, antialias: px < 2, alpha: true, powerPreference: 'default' });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(px);
  renderer.setSize(w, h, false);
  await pause();

  const scene = new Scene();
  const camera = new PerspectiveCamera(30, w / h, 1, 5000);
  camera.up.set(0, 0, 1);
  const uniforms = {
    uF: { value: man.load_N },
    uU: { value: new Vector3(1, 0, 0) },
    uExag: { value: 60 },
    uLimit: { value: man.allowable_MPa },
    uHatch: { value: 1 },
    uPx: { value: px },
    uRA: { value: new Vector4() },
    uRB: { value: new Vector4() },
    uRC: { value: new Vector4() },
    uRS: { value: new Vector4() },
    uRSC: { value: new Vector2() },
    uRDA: { value: new Vector3() },
    uRDB: { value: new Vector3() },
    uRDC: { value: new Vector3() },
  };
  const material = new ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, side: DoubleSide, uniforms });
  const pickMaterial = new ShaderMaterial({
    vertexShader: VERT, fragmentShader: PICK_FRAG, side: DoubleSide, uniforms, defines: { PICK: '' },
  });
  const geo = new BufferGeometry();
  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false;
  scene.add(mesh);
  const pickScene = new Scene();
  const pickMesh = new Mesh(geo, pickMaterial);
  pickMesh.frustumCulled = false;
  pickScene.add(pickMesh);
  const pickTarget = new WebGLRenderTarget(1, 1);

  // The base is bolted flat to a frame: show that plane.
  const lo = new Vector3(...man.mesh.bounds_min), hi = new Vector3(...man.mesh.bounds_max);
  const size = new Vector3().subVectors(hi, lo);
  const center = new Vector3().addVectors(lo, hi).multiplyScalar(0.5);
  const floor = new Mesh(new PlaneGeometry(size.x * 1.4, size.y * 1.4), new MeshBasicMaterial({ color: 0x15171a }));
  floor.position.set(center.x, center.y, lo.z - 0.4);
  scene.add(floor);

  // The load acts on the motor shaft at the pulley, off the part, and can point any way.
  const pl = man.pull;
  const pAt = new Vector3(...pl.point);
  const e1 = new Vector3(...pl.e1), e2 = new Vector3(...pl.e2), e3 = new Vector3(...pl.e3);
  const pDir = new Vector3(), Y = new Vector3(0, 1, 0), side = new Vector3(), across = new Vector3();
  const pull = arrow(COLD);
  pull.position.copy(pAt);
  scene.add(pull);
  // While the arrow is under the pointer or held, guides show the sphere its tip moves on, so depth
  // reads: a faint shell, the circle in the pulley plane and the circle through the arrow and the shaft.
  const circle = (a, b) => {
    const pts = [];
    for (let k = 0; k <= 96; k++) {
      const t = (k / 96) * Math.PI * 2;
      pts.push(new Vector3().addScaledVector(a, Math.cos(t)).addScaledVector(b, Math.sin(t)));
    }
    return new BufferGeometry().setFromPoints(pts);
  };
  const lineMat = new LineBasicMaterial({ color: COLD, transparent: true, opacity: 0.55, depthTest: false });
  const ring = new Group();
  const equator = new Line(circle(e1, e2), lineMat);
  const meridian = new Line(circle(new Vector3(1, 0, 0), new Vector3(0, 1, 0)), lineMat);
  meridian.matrixAutoUpdate = false;
  const shell = new Mesh(new SphereGeometry(1, 32, 16),
    new MeshBasicMaterial({ color: COLD, transparent: true, opacity: 0.07, depthWrite: false, depthTest: false }));
  ring.add(shell, equator, meridian);
  ring.position.copy(pAt);
  ring.renderOrder = 3;
  ring.visible = false;
  scene.add(ring);

  // Compile the shaders before the mesh arrives; with KHR_parallel_shader_compile it does not block.
  await pause();
  await Promise.all([renderer.compileAsync(scene, camera), renderer.compileAsync(pickScene, camera)]);
  // first use of the probe's target and shader, on the still empty mesh
  await pause();
  const intoPick = () => {
    renderer.setRenderTarget(pickTarget);
    renderer.render(pickScene, camera);
    renderer.setRenderTarget(null);
  };
  intoPick();
  const { arrays: m, range } = await meshP;
  uniforms.uRA.value.fromArray(range.stressA);
  uniforms.uRB.value.fromArray(range.stressB);
  uniforms.uRC.value.fromArray(range.stressC);
  uniforms.uRS.value.fromArray(range.shearAB);
  uniforms.uRSC.value.fromArray(range.shearC);
  uniforms.uRDA.value.fromArray(range.dispA);
  uniforms.uRDB.value.fromArray(range.dispB);
  uniforms.uRDC.value.fromArray(range.dispC);
  // Send the mesh to the GPU a few attributes at a time, each in its own step, by drawing it into
  // the probe's one pixel target; the first frame on screen then has nothing left to send.
  geo.setIndex(new BufferAttribute(m.index, 1));
  for (const group of [['position', 'normal', 'shearC'], ['stressA', 'stressB', 'stressC'], ['shearAB', 'dispA', 'dispB', 'dispC']]) {
    for (const name of group) geo.setAttribute(name, new BufferAttribute(m[name], SIZES[name], name !== 'position'));
    await pause();
    intoPick();
  }

  // One frame of state from the page: the camera, and the load and how it is drawn.
  function set(f) {
    const c = f.camera;
    camera.position.fromArray(c.position);
    camera.lookAt(c.target[0], c.target[1], c.target[2]);
    camera.fov = c.fov;
    camera.aspect = c.aspect;
    if (c.view) camera.setViewOffset(...c.view); else camera.clearViewOffset();
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    uniforms.uF.value = f.load;
    uniforms.uU.value.fromArray(f.u);
    uniforms.uExag.value = f.exag;
    uniforms.uLimit.value = f.limit;
    pDir.copy(e1).multiplyScalar(f.u[0]).addScaledVector(e2, f.u[1]).addScaledVector(e3, f.u[2]);
    const len = f.len, grow = Math.min(f.grow, 1);
    pull.visible = len > 0.5;
    const { shaft, head, hub, materials: [solid, faint] } = pull.userData;
    pull.quaternion.setFromUnitVectors(Y, pDir);
    const body = Math.max(len - 10 * grow, 0);
    head.scale.setScalar(Math.max(grow, 0.01));
    shaft.scale.set(1, Math.max(body, 0.01), 1);
    shaft.position.set(0, body / 2, 0);
    head.position.set(0, body + 5 * grow, 0);
    hub.position.set(0, 0, 0);
    solid.color.setHex(f.hot ? HOT : COLD);
    faint.color.setHex(f.hot ? HOT : COLD);
    faint.opacity = f.hot ? 0.5 : 0.35;
    ring.visible = f.ring;
    ring.scale.setScalar(Math.max(len, 16));
    // the circle through the arrow and the shaft
    side.copy(e1).multiplyScalar(f.u[0]).addScaledVector(e2, f.u[1]);
    if (side.lengthSq() < 1e-6) side.copy(e1);
    side.normalize();
    across.crossVectors(side, e3);
    meridian.matrix.makeBasis(side, e3, across);
    meridian.matrixWorldNeedsUpdate = true;
  }
  const px1 = new Uint8Array(4);
  return {
    draw(f) { set(f); renderer.render(scene, camera); },
    size(nw, nh, npx) {
      renderer.setPixelRatio(npx);
      renderer.setSize(nw, nh, false);
      uniforms.uPx.value = npx;
    },
    // von Mises under one CSS pixel of the view as last drawn: [red, green, blue, alpha], see PICK_FRAG
    pick(x, y, vw, vh) {
      const view = camera.view && camera.view.enabled ? { ...camera.view } : null;
      camera.setViewOffset(vw, vh, x, y, 1, 1);
      renderer.setRenderTarget(pickTarget);
      renderer.clear();
      renderer.render(pickScene, camera);
      renderer.readRenderTargetPixels(pickTarget, 0, 0, 1, 1, px1);
      renderer.setRenderTarget(null);
      if (view) camera.setViewOffset(view.fullWidth, view.fullHeight, view.offsetX, view.offsetY, view.width, view.height);
      else camera.clearViewOffset();
      return [...px1];
    },
    dispose() {
      scene.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      pickMaterial.dispose();
      pickTarget.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
